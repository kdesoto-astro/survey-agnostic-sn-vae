# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module for the loop primitives."""
from __future__ import annotations

from collections.abc import Sequence
from functools import partial
import inspect
import itertools
import operator
from typing import Any, Callable, TypeVar

import jax
import weakref
from jax._src import config
from jax._src import core
from jax._src import linear_util as lu
from jax._src.core import ConcreteArray, ShapedArray, raise_to_shaped
from jax.tree_util import (tree_flatten, tree_unflatten, treedef_is_leaf,
                           tree_map, tree_flatten_with_path, keystr)
from jax._src.api_util import shaped_abstractify
from jax._src.tree_util import equality_errors
from jax._src import ad_checkpoint
from jax._src import ad_util
from jax._src import api
from jax._src import dispatch
from jax._src import dtypes
from jax._src import effects
from jax._src import source_info_util
from jax._src import util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import xla
from jax._src.lax import lax
from jax._src.lax import slicing
from jax._src.lax import windowed_reductions
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src import state
from jax._src.state import discharge as state_discharge
from jax._src.numpy.ufuncs import logaddexp
from jax._src.traceback_util import api_boundary
from jax._src.typing import Array
from jax._src.util import (partition_list, safe_map, safe_zip, split_list,
                           unzip2, weakref_lru_cache, merge_lists)
import numpy as np

from jax._src.lax.control_flow.common import (
    _abstractify, _avals_short, _check_tree_and_avals, _initial_style_jaxpr,
    _initial_style_jaxpr_attrs, _make_closed_jaxpr_attrs, _prune_zeros,
    _typecheck_param)

_map = safe_map
zip = safe_zip

T = TypeVar('T')
BooleanNumeric = Any  # A bool, or a Boolean array.

Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')

def scan(f: Callable[[Carry, X], tuple[Carry, Y]],
         init: Carry,
         xs: X,
         length: int | None = None,
         reverse: bool = False,
         unroll: int | bool = 1,
         _split_transpose: bool = False) -> tuple[Carry, Y]:
  """Scan a function over leading array axes while carrying along state.
  The `Haskell-like type signature`_ in brief is
  """
  if not callable(f):
    raise TypeError("lax.scan: f argument should be a callable.")
  xs_flat, xs_tree = tree_flatten(xs)

  try:
    lengths = [x.shape[0] for x in xs_flat]
  except AttributeError as err:
    msg = "scan got value with no leading axis to scan over: {}."
    raise ValueError(
      msg.format(', '.join(str(x) for x in xs_flat
                           if not hasattr(x, 'shape')))) from err

  if length is not None:
    length = int(length)
    if not all(length == l for l in lengths):
      msg = ("scan got `length` argument of {} which disagrees with "
             "leading axis sizes {}.")
      raise ValueError(msg.format(length, [x.shape[0] for x in xs_flat]))
  else:
    unique_lengths = set(lengths)
    if len(unique_lengths) > 1:
      msg = "scan got values with different leading axis sizes: {}."
      raise ValueError(msg.format(', '.join(str(x.shape[0]) for x in xs_flat)))
    elif len(unique_lengths) == 0:
      msg = "scan got no values to scan over and `length` not provided."
      raise ValueError(msg)
    else:
      length, = unique_lengths

  if config.disable_jit.value:
    if length == 0:
      raise ValueError("zero-length scan is not supported in disable_jit() mode because the output type is unknown.")
    carry = init
    ys = []
    maybe_reversed = reversed if reverse else lambda x: x
    for i in maybe_reversed(range(length)):
      xs_slice = [_index_array(i, core.get_aval(x), x) for x in xs_flat]
      carry, y = f(carry, tree_unflatten(xs_tree, xs_slice))
      ys.append(y)
    stack = lambda *ys: jax.numpy.stack(ys)
    stacked_y = tree_map(stack, *maybe_reversed(ys))
    return carry, stacked_y

  xs_avals = [core.raise_to_shaped(core.get_aval(x)) for x in xs_flat]
  x_avals = [core.mapped_aval(length, 0, aval) for aval in xs_avals]

  def _create_jaxpr(init):
    init_flat, init_tree = tree_flatten(init)
    in_flat, in_tree = tree_flatten((init, xs))

    carry_avals = tuple(_map(_abstractify, init_flat))
    jaxpr, consts, out_tree, attrs_tracked = _initial_style_jaxpr_attrs(
        f, in_tree, (*carry_avals, *x_avals), "scan")
    out_tree_children = out_tree.children()
    if len(out_tree_children) != 2:
      msg = "scan body output must be a pair, got {}."
      raise TypeError(msg.format(tree_unflatten(out_tree, jaxpr.out_avals)))
    carry_avals_out = jaxpr.out_avals[:out_tree_children[0].num_leaves]
    return (init_flat, carry_avals, carry_avals_out, init_tree, in_flat, jaxpr,
            consts, out_tree, out_tree_children, attrs_tracked)

  # The carry input and output avals must match exactly. However, we want to account for
  # the case when init contains weakly-typed values (e.g. Python scalars), with avals that
  # may not match the output despite being compatible by virtue of their weak type.
  # To do this, we compute the jaxpr in two passes: first with the raw inputs, and if
  # necessary, a second time with modified init values.
  init_flat, carry_avals, carry_avals_out, init_tree, *rest = _create_jaxpr(init)
  new_init_flat, changed = _promote_weak_typed_inputs(init_flat, carry_avals, carry_avals_out)
  if changed:
    init = tree_unflatten(init_tree, new_init_flat)
    init_flat, carry_avals, carry_avals_out, init_tree, *rest = _create_jaxpr(init)
  in_flat, jaxpr, consts, out_tree, out_tree_children, attrs_tracked = rest
  num_carry = len(init_flat)

  _check_scan_carry_type(f, init, out_tree_children[0], carry_avals_out)
  disallowed_effects = effects.control_flow_allowed_effects.filter_not_in(jaxpr.effects)
  if disallowed_effects:
    raise NotImplementedError(
        f'Effects not supported in `scan`: {disallowed_effects}')

  if isinstance(unroll, bool):
    unroll = max(length, 1) if unroll else 1
  if attrs_tracked:
    in_state = _get_states(attrs_tracked)
    in_carry, in_ext = split_list(in_flat, [num_carry])
    in_flat = [*in_state, *in_carry, *in_ext]
    num_carry += len(attrs_tracked)
  out = scan_p.bind(*consts, *in_flat,
                    reverse=reverse, length=length, jaxpr=jaxpr,
                    num_consts=len(consts), num_carry=num_carry,
                    linear=(False,) * (len(consts) + len(in_flat)),
                    unroll=unroll,
                    _split_transpose=_split_transpose)
  if attrs_tracked:
    out_state, out = split_list(out, [len(attrs_tracked)])
    _set_states(attrs_tracked, out_state)
  return tree_unflatten(out_tree, out)