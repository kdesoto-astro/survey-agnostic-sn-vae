"""Module to generate the MOSFIT light curves for VAE training."""

import mosfit
import numpy as np
from typing import List
from objects import *

def generate_LCs_from_model(
    model_name: str,
    survey_list: List[Survey],
    num=1000
) -> List[LightCurve]:
    """Generate clean light curves from a MOSFIT
    model.

    Parameters
    ----------
    model_name : str
        The name of the built-in MOSFIT model.
    num : int, optional
        The number of light curves to generate from
        that model. Defaults to 1000.

    Returns
    ----------
    LightCurveSet
        The set of generated light curves.
    """
    model = mosfit.model.Model(model=model_name)
    # Generate a random input vector of free parameters.
    x = np.random.rand(my_model.get_num_free_parameters())

    # Produce model output.
    outputs = my_model.run(x)
    print(
        'Keys in output: `{}`'.format(
            ', '.join(list(outputs.keys()))
        )
    )