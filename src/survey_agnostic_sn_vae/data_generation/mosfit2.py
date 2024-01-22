"""Module to generate the MOSFIT light curves for VAE training."""

import mosfit
import os
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
    orig_path = os.getcwd()
    
    mosfit_path = os.path.dirname(
        os.path.realpath(mosfit.__file__)
    )
    
    print("Switching to MOSFIT path: %s" % mosfit_path)
    os.chdir(mosfit_path)
    
    model = mosfit.model.Model(model=model_name)

    lcs = []
    
    for s in survey_list:
        s_name = s.name
        s_bands = s.bands
        s_times = s.generate_sample_times(20)
        
        fitter = mosfit.fitter.Fitter()
        
        data = fitter.generate_dummy_data(
            s.name,
            time_list=s_times,
            band_list=s.bands,
            band_instruments=[s.name,],
        )
        model.load_data(
            data,
            band_list=s.bands,
            band_instruments=[s.name,]
        )
        # Generate a random input vector of free parameters.
        x = np.random.rand(model.get_num_free_parameters())

        # Produce model output.
        outputs = model.run(x)
        print(
            'Keys in output: `{}`'.format(
                ', '.join(list(outputs.keys()))
            )
        )
    
    print("Switching back to original working directory")
    os.chdir(orig_path)