from typing import Optional, Callable, Tuple
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.tree_util as jtu
import numpy as np
import optax
import wandb

from survey_agnostic_sn_vae.autoencoder.contrastive_loss import contrastive_loss

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


@jax.jit
def sample_with_dynamic_indices(array, indices):
    """For some reason works around concrete index error."""
    return array[:,:,indices]
    
class TimeDistributedDense(eqx.Module):
    """Mimics PyTorch's TimeDistributed Layer for a Dense Layer."""
    dense: eqx.Module
    
    def __init__(self, features_in, features_out, key):
        self.dense = eqx.nn.Linear(
            features_in, features_out, key=key
        )
    
    def __call__(self, x):
        # Assuming x has shape (batch_size, time_steps, features_in)
        return jax.vmap(self.dense, in_axes=0, out_axes=0)(x)  # Apply across the time axis


@jax.jit
def gru_scan_no_cache_miss(carry, scan_in):
    """Avoids the cache misses that I was seeing
    from the built-in equinox GRU.call function."""
    (
        weight_ih,
        weight_hh,
        bias,
        bias_n,
        hidden
    ) = carry
    igate_0, igate_1, igate_2 = jnp.split(weight_ih @ scan_in + bias, 3)
    hgate_0, hgate_1, hgate_2 = jnp.split(weight_hh @ hidden, 3)
    reset = jnn.sigmoid(igate_0 + hgate_0)
    inp = jnn.sigmoid(igate_1 + hgate_1)
    new = jnn.tanh(igate_2 + reset * (hgate_2 + bias_n))

    return (
        weight_ih,
        weight_hh,
        bias,
        bias_n,
        new + inp * (hidden - new),
     ), new + inp * (hidden - new)

class SymmetricEncoder(eqx.Module):
    """The SymmetricEncoder Equinox module.
    Uses GRUs to convert input time series to latent
    mean and variance. Enforces symmetry among filter inputs:
    effectively identical to summing the filter vals within each
    input and feeding into a smaller Encoder.
    """
    vmapped_layers: list
    layers_after_gru: list
    gru: eqx.Module
    prelu: eqx.nn.PReLU
    init_state: jnp.ndarray

    def __init__(
            self,
            hidden_dim: int,
            key=jax.random.key(42),
        ):
        key1, key2, key3, key4 = jax.random.split(key, num=4)

        self.vmapped_layers = [
            TimeDistributedDense(5, hidden_dim, key=key1), # t, m_combined, merr_combined, wv_combined, width_combined
            eqx.nn.PReLU(),
            TimeDistributedDense(hidden_dim, hidden_dim, key=key2), # t, m_combined, merr_combined, wv_combined, width_combined
            #eqx.nn.PReLU(),
            #TimeDistributedDense(hidden_dim, hidden_dim, key=key3), # t, m_combined, merr_combined, wv_combined, width_combined
        ]
        self.prelu = eqx.nn.PReLU()
        self.gru = eqx.nn.GRUCell(hidden_dim, hidden_dim, key=key4)
        self.layers_after_gru = [eqx.nn.PReLU(),]
        self.init_state = jnp.zeros(self.gru.hidden_size)

    def __call__(self, x, filt_mask, t_end_idx):
                
        for layer in self.vmapped_layers:
            x = jax.vmap(layer)(x)

        combined_layer = jnp.mean(x*filt_mask[:,jnp.newaxis,jnp.newaxis], axis=0)
        x = self.prelu(combined_layer)
        carry = (
            self.gru.weight_ih,
            self.gru.weight_hh,
            self.gru.bias,
            self.gru.bias_n,
            self.init_state
        )
        out, hiddens = jax.lax.scan(gru_scan_no_cache_miss, carry, x)
        x = hiddens[t_end_idx]
        for layer in self.layers_after_gru:
            x = layer(x)
        return x


class Decoder(eqx.Module):
    """The Decoder Equinox module.
    Samples from the latent space distribution and
    passes sample through time-distributed MLP to return magnitudes."""
    layers: list

    def __init__(
            self,
            out_dim: int,
            hidden_dim: int,
            key=jax.random.key(42),
        ):
        key1, key2, key3 = jax.random.split(key, num=3)
        self.layers = [
            TimeDistributedDense(out_dim, hidden_dim, key=key1),
            eqx.nn.PReLU(),
            TimeDistributedDense(hidden_dim, hidden_dim, key=key2),
            eqx.nn.PReLU(),
            TimeDistributedDense(hidden_dim, 1, key=key3),
            eqx.nn.PReLU(),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

@jax.jit
def sample_with_dynamic_indices(array, indices):
    """For some reason works around concrete index error."""
    return array[:,:,indices]

@jax.jit
def generate_decoder_input(seq):
    return jnp.reshape(seq[:,:,[0,4,5]], (seq.shape[0]*seq.shape[1], 3))

class VAE(eqx.Module):
    """Full VAE, including encoder, decoder + sampler."""
    encoder: eqx.Module
    decoder: eqx.Module
    latent_mean: eqx.nn.Linear
    latent_logvar: eqx.nn.Linear
    sample_noise: jax.Array
    lc_nums: jax.Array

    def __init__(
            self,
            hidden_dim: int,
            out_dim: int,
            key=jax.random.key(42),
        ):
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, num=6)
        self.encoder = SymmetricEncoder(hidden_dim, key1)
        self.decoder = Decoder(
            out_dim+3, hidden_dim, key2
        )

        self.latent_mean = eqx.nn.Linear(hidden_dim, out_dim, key=key3)
        self.latent_logvar = eqx.nn.Linear(hidden_dim, out_dim, key=key4)
        self.sample_noise = jax.random.normal(key=key5, shape=(10_000, 1_000, out_dim))
        self.lc_nums = jax.random.bernoulli(key=key6, shape=(10_000, 1_000, 21))

    def __call__(self, encoder_input, encoder_mask, vmapped_idx, epoch_count):

        filt_mask = self.lc_nums[epoch_count, vmapped_idx] * encoder_mask[:-1]
        x = self.encoder(encoder_input[:,:,[0,1,2,4,5]], filt_mask, encoder_mask[-1])
        
        sub_dec = generate_decoder_input(encoder_input)
        mu = self.latent_mean(x)
        logvar = self.latent_logvar(x)
        z = mu + self.sample_noise[epoch_count, vmapped_idx] * jnp.exp(logvar / 2.)
        x = z[jnp.newaxis, :]
        x = jnp.repeat(x, sub_dec.shape[0], axis=0)
        x = jnp.concatenate((x, sub_dec), axis=1)
        x = self.decoder(x)
        x = jnp.reshape(x, (-1, encoder_input.shape[1]))
        return x, mu, logvar, z
    
@partial(jax.jit, static_argnums=(5,6,7,8))
@eqx.filter_value_and_grad(has_aux=True)
def compute_loss(
    model, epoch, encoder_input, encoder_mask, matches,
    include_reconstructive,include_kl,
    contrastive_distance,contrastive_temp,
):
    """Calculate the overall loss of model on x, given
    true recreations y. Matches indicate arrays from
    same event."""
    pred_y, mu, logvar, z = jax.vmap(model, in_axes=(0,0,0,None))(
        encoder_input, encoder_mask, jnp.arange(len(encoder_input)), epoch
    )
    loss = 0.0
    kl = jnp.nan
    cl = jnp.nan
    rl = jnp.nan
    if include_reconstructive:
        rl = reconstruction_loss(encoder_input, pred_y)
        loss += rl
    if include_kl:
        kl = kl_loss(mu, logvar)
        loss += kl
    if contrastive_distance:
        cl = contrastive_loss(
            z, mu, logvar, matches,
            distance=contrastive_distance,
            temp=contrastive_temp
        )
        loss += cl
    return loss, (rl, kl, cl)

@partial(jax.jit, static_argnums=(6,7,8,9))
@eqx.filter_value_and_grad(has_aux=True)
def compute_loss_frozen(
    diff_model, static_model,
    epoch, encoder_input, encoder_mask, matches,
    include_reconstructive,include_kl,
    contrastive_distance,
    contrastive_temp
):
    """Compute loss for partially frozen model."""
    model = eqx.combine(diff_model, static_model)
    pred_y, mu, logvar, z = jax.vmap(model, in_axes=(0,0,0,None))(
        encoder_input, encoder_mask, jnp.arange(len(encoder_input)), epoch
    )
    loss = 0
    kl = jnp.nan
    cl = jnp.nan
    rl = jnp.nan
    if include_reconstructive:
        rl = reconstruction_loss(encoder_input, pred_y)
        loss += rl
    if include_kl:
        kl = kl_loss(mu, logvar)
        loss += kl
    if contrastive_distance:
        cl = contrastive_loss(
            z, mu, logvar, matches,
            distance=contrastive_distance,
            temp=contrastive_temp
        )
        loss += cl
    return loss, (rl, kl, cl)

@jax.jit
def reconstruction_loss(y, pred_y):
    """
    Calculate the reconstruction loss of the model.
    """
    numerator = (y[:,:,:,1] - pred_y) ** 2 * (1 - y[:,:,:,3]) / y[:,:,:,2] ** 2
    # prevent outliers from greatly affecting gradients
    #num_clipped = jnp.clip(numerator, min=0.0, max=10.) + 0.1*numerator
    return jax.numpy.sum(numerator) / jax.numpy.sum(1 - y[:,:,:,3])

@jax.jit
def kl_loss(mu, logvar):
    """
    Calculate the reconstruction loss of the model.
    """
    return - 0.5 * jax.numpy.mean(1 + logvar - mu ** 2 - jax.numpy.exp(logvar))

@jax.jit
def update_slices(carry, start_index):
    updated_data1, updated_data2, encoder_data, shuffled_idx1, shuffled_idx2 = carry
    new_slice_data1 = sample_with_dynamic_indices(encoder_data, start_index+shuffled_idx1)
    new_slice_data2 = sample_with_dynamic_indices(encoder_data, start_index+shuffled_idx2)

    updated_data1 = jax.lax.dynamic_update_slice(updated_data1, new_slice_data1, (0, 0, start_index))
    updated_data2 = jax.lax.dynamic_update_slice(updated_data2, new_slice_data2, (0, 0, start_index))

    return (updated_data1, updated_data2, encoder_data, shuffled_idx1, shuffled_idx2), None

@jax.jit
def dataloader(encoder_data, encoder_mask, ids, shuffle_key):
    """Shuffle and load encoder and decoder arrays.
    Within this function we generate (1) pairs of samples with subsets
    of LCs from the same events to enforce our contrastive loss, 
    (2) shuffle bands randomly, and (3) generate decoder array from
    encoder array."""
    
    dataset_size = len(encoder_data)
    encoder_pairs = jnp.vstack((encoder_data, encoder_data))
    encoder_mask_pairs = jnp.vstack((encoder_mask, encoder_mask))
    match_data = jnp.tile(ids, 2)

    if shuffle_key is None:
        return encoder_pairs, encoder_mask_pairs, match_data, None, None
    
    new_key, key = jax.random.split(shuffle_key)
    perm = jax.random.permutation(key=key, x=ids)
    perm_doubled = jnp.repeat(perm, 2)
    even_idxs = jnp.arange(1, dataset_size, 2)
    perm_doubled = perm_doubled.at[even_idxs].set(perm_doubled[even_idxs-1] + dataset_size) # makes sure pairs are summoned together
    return encoder_pairs, encoder_mask_pairs, match_data, perm_doubled, new_key


@partial(jax.jit, static_argnums=2)
def make_step(carry, perm_idxs, statics_hashable, statics_nonhashable):
    (
        model_params,
        opt_state,
        encoder_data,
        encoder_mask,
        matches,
        epoch
    ) = carry

    (
        static_model,
        filter_spec,
        transfer_learning,
        include_reconstructive,
        include_kl,
        distance,
        temp,
        optim
    ) = statics_hashable

    (
        val_encoder_data,
        val_encoder_mask,
        val_matches,
    ) = statics_nonhashable

    #flat_model, flat_opt_state = A
    #model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
    #opt_state = jax.tree_util.tree_unflatten(treedef_opt_state, flat_opt_state)

    encoder_batch = encoder_data[perm_idxs]
    mask_batch = encoder_mask[perm_idxs]
    matches_batch = matches[perm_idxs]
    model = eqx.combine(model_params, static_model)

    """
    if transfer_learning:
        (loss, (rl, kl, cl)), grads = compute_loss_frozen(
            diff_model, static_model,
            epoch, encoder_batch, decoder_batch, matches_batch,
            include_reconstructive,include_kl,distance,temp
        )
    """
        
    (loss, (rl, kl, cl)), grads = compute_loss(
        model, epoch, encoder_batch, mask_batch, matches_batch,
        include_reconstructive,include_kl,distance,temp
    )
    (val_loss, (val_rl, val_kl, val_cl)), _ = compute_loss(
        model, epoch, val_encoder_data, val_encoder_mask, val_matches,
        include_reconstructive,include_kl,distance,temp
    )
    updates, update_opt_state = optim.update(grads, opt_state)
    update_model = eqx.apply_updates(model, updates)
    update_model_params, _ = eqx.partition(update_model, eqx.is_array)
    #flat_update_model = jax.tree_util.tree_leaves(update_model)
    #flat_update_opt_state = jax.tree_util.tree_leaves(update_opt_state)

    #return (flat_update_model, flat_update_opt_state), jnp.array([loss, rl, kl, cl, val_loss, val_rl, val_kl, val_cl])
    new_carry = (
        update_model_params,
        update_opt_state,
        encoder_data,
        encoder_mask,
        matches,
        epoch
    )
    return new_carry, jnp.array([loss, rl, kl, cl, val_loss, val_rl, val_kl, val_cl])

@partial(jax.jit, static_argnums=2)
def epoch_iterator(carry, epoch, statics_hashable, statics_nonhashable):

    # carry: (model, opt_state, key)

    encoder_inputs, encoder_masks, ids = statics_nonhashable[:3]
    batch_size = statics_hashable[0]

    encoder_data, encoder_mask, matches, perm, new_key = dataloader(
        encoder_inputs, encoder_masks, ids, carry[-1]
    )
    
    perm_new_len = (len(encoder_data) // batch_size) * batch_size
    perm_reshaped = jnp.reshape(perm[:perm_new_len], shape=(-1,batch_size))

    carry_init = (
        *carry[:-1],
        encoder_data, # generated in loop
        encoder_mask,
        matches,
        epoch,
    )
    # statics[-1] is make_step_partials
    carry_final, losses = jax.lax.scan(statics_hashable[-1], carry_init, perm_reshaped)
    A = (*carry_final[:-4], new_key)
    return A, jnp.mean(losses, axis=0)

def fit_model(
        model,
        encoder_inputs_ragged,
        ids,
        val_encoder_inputs_ragged,
        val_ids,
        learning_rate=1e-3,
        num_epochs=1000,
        batch_size=32,
        transfer_learning=False,
        wandb_log=False,
        include_reconstructive: bool=True,
        include_kl: bool=True,
        contrastive_params: Optional[str]=None
    ):
    """Main loop of the VAE.
    If transfer_learning is True, only unfreeze mu, logvar, and first decoder layers."""

    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda tree: (
            tree.latent_mean.weight,
            tree.latent_mean.bias,
            tree.latent_logvar.weight,
            tree.latent_logvar.bias,
            tree.decoder.layers[0].dense.weight,
            tree.decoder.layers[0].dense.bias,
        ),
        filter_spec,
        replace=(True, True, True, True, True, True),
    )

    key = jax.random.key(42)
    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    #flat_model, treedef_model = jax.tree_util.tree_flatten(model)
    #flat_opt_state, treedef_opt_state = jax.tree_util.tree_flatten(opt_state)

    # convert ids to numerics
    _, ids = np.unique(ids, return_inverse=True)
    _, val_ids = np.unique(val_ids, return_inverse=True)
    
    # make train and val masks
    max_num_filts = max([x.shape[0] for x in encoder_inputs_ragged])
    max_num_val_filts = max([x.shape[0] for x in val_encoder_inputs_ragged])
    max_num_filts = max(max_num_filts, max_num_val_filts)
    max_num_val_t = max([x.shape[1] for x in val_encoder_inputs_ragged])
    max_num_t = max([x.shape[1] for x in encoder_inputs_ragged])
    max_num_t = max(max_num_t, max_num_val_t)
    
    print('mask shape', (max_num_filts, max_num_t))
    
    encoder_masks = jnp.zeros((len(encoder_inputs_ragged), max_num_filts+1), dtype=int)
    encoder_inputs = jnp.zeros((len(encoder_inputs_ragged), max_num_filts, max_num_t, 6))
    for i, x in enumerate(encoder_inputs_ragged):
        encoder_masks = encoder_masks.at[i, :x.shape[0]].set(1)
        encoder_masks = encoder_masks.at[i, -1].set(x.shape[1])
        encoder_inputs = encoder_inputs.at[i, :x.shape[0], :x.shape[1]].set(x)
        encoder_inputs = encoder_inputs.at[i, x.shape[0]:, :, 3].set(1)
        encoder_inputs = encoder_inputs.at[i, :, x.shape[1]:, 3].set(1)
            
    val_encoder_masks = jnp.zeros(
        (len(val_encoder_inputs_ragged), max_num_filts+1), dtype=int
    )
    val_encoder_inputs = jnp.zeros(
        (len(val_encoder_inputs_ragged), max_num_filts, max_num_t, 6)
    )
    for i, x in enumerate(val_encoder_inputs_ragged):
        val_encoder_masks = val_encoder_masks.at[i, :x.shape[0]].set(1)
        val_encoder_masks = val_encoder_masks.at[i, -1].set(x.shape[1])
        val_encoder_inputs = val_encoder_inputs.at[i, :x.shape[0], :x.shape[1]].set(x)
        val_encoder_inputs = val_encoder_inputs.at[i, x.shape[0]:, :, 3].set(1)
        val_encoder_inputs = val_encoder_inputs.at[i, :, x.shape[1]:, 3].set(1)

    val_encoder_data, val_encoder_mask, val_matches, _, _ = dataloader(
        val_encoder_inputs, val_encoder_masks, val_ids, None
    )

    if contrastive_params != 'None':
        assert "_" in contrastive_params
        temp = float(contrastive_params.split("_")[1])
        distance = contrastive_params.split("_")[0]
    else:
        distance, temp = None, None
    
    log_losses = []
    log_val_losses = []

    model_params, static_model = eqx.partition(model, eqx.is_array)

    statics_hashable = (
        batch_size,
        static_model,
        filter_spec,
        transfer_learning,
        include_reconstructive,
        include_kl,
        distance,
        temp,
        optim
    )
    statics_nonhashable = (
        encoder_inputs,
        encoder_masks,
        ids,
        val_encoder_data,
        val_encoder_mask,
        val_matches,
    )

    make_step_partial = partial(
        make_step,
        statics_hashable=statics_hashable[1:],
        statics_nonhashable=statics_nonhashable[3:]
    )

    A = (
        model_params,
        opt_state,
        key,
    )

    # Use partial to fix statics
    epoch_iterator_partial = partial(
        epoch_iterator, statics_hashable=(*statics_hashable, make_step_partial),
        statics_nonhashable=statics_nonhashable
    )

    for e in range(num_epochs):
        #(flat_model, flat_opt_state, key), (l, rl, kl, cl, vl, vrl, vkl, vcl) = epoch_iterator((flat_model, flat_opt_state, key), e)
        print(e)
        A, out = epoch_iterator_partial(A, e)

        loss, rl, kl, cl, vl, vrl, vkl, vcl = jnp.log10(out).block_until_ready()

        if wandb_log:
            wandb.log({
                "log_loss": loss,
                "val_log_loss": vl,
                "reconstruction_log_loss": rl,
                "kl_log_loss": kl,
                "contrastive_log_loss": cl,
                "val_reconstruction_log_loss": vrl,
                "val_kl_log_loss": vkl,
                "val_contrastive_log_loss": vcl,
            })
        
        else:
            print(f"Epoch {e}: Training Log Loss={round(loss, 3)}, Validation Log Loss={round(vl, 3)}")

        log_losses.append(loss)
        log_val_losses.append(vl)

    #model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
    print('finished')
    model = eqx.combine(model_params, static_model)
    return model, log_losses, log_val_losses

@jax.jit
def evaluate(
    model,
    encoder_input,
    decoder_input,
):
    """Evaluate the model on a dataset."""
    if decoder_input is None:
        decoder_input = generate_decoder_input(encoder_input)
    
    inference_model = eqx.nn.inference_mode(model)
    pred_y, mu, logvar, z = jax.vmap(
        inference_model, in_axes=(0,0,0,None)
    )(encoder_input, decoder_input, jnp.arange(len(encoder_input)), 0)
    return {
        "pred_y": pred_y,
        "mu": mu,
        "logvar": logvar,
        "z": z
    }


    