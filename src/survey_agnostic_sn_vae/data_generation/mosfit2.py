"""Module to generate the MOSFIT light curves for VAE training."""

import mosfit
import os
import numpy as np
from typing import List, Dict, Any
from survey_agnostic_sn_vae.data_generation.objects import *
from survey_agnostic_sn_vae.data_generation.utils import *

def generate_LCs_from_model(
    model_type: str,
    survey_list: List[Survey],
    num=1000,
    output_path=None
) -> List[Transient]:
    """Generate clean light curves from a MOSFIT
    model, and save to folder.

    Parameters
    ----------
    model_name : str
        The name of the built-in MOSFIT model.
    num : int, optional
        The number of light curves to generate from
        that model. Defaults to 1000.

    Returns
    ----------
    List[Transients]
        The set of transients generated with associated LCs.
    """
    orig_path = os.getcwd()
    if output_path is None:
        output_path = orig_path

    print(output_path)
    mosfit_path = os.path.dirname(
        os.path.realpath(mosfit.__file__)
    )

    print("Switching to MOSFIT path: %s" % mosfit_path)
    os.chdir(mosfit_path)

    with suppress_stdout():
        fitter = mosfit.fitter.Fitter()

        tmp_dir = os.path.join(output_path, "products")
        # generate initial LCs/model params
        fitter.fit_events(
            models=[model_type,],
            max_time=1000.0,
            iterations=0,
            write=True,
            output_path=output_path,
            num_walkers=num,
        )
        file_loc = os.path.join(
            tmp_dir, f"{model_type}.json"
        )
        data = open_walkers_file(file_loc)
        transients = generate_transients_from_samples(data)
        
        for t in transients:
            for i, s in enumerate(survey_list):
                fitter._event_name = i
                t.generate_lightcurve(
                    s, output_path,
                    fitter=fitter
                )
            t.save(
                os.path.join(
                    output_path, "transients"
                )
            )
            for f in glob.glob(os.path.join(tmp_dir, "*")):
                os.remove(f)
                
    print("Switching back to original working directory")
    os.chdir(orig_path)

    return transients, fitter


def generate_transients_from_samples(
    data : Dict[str, Any]
) -> List[Transient]:
    """Generate list of Transient objects from
    simulated clean LC samples.
    """
    realizations = data['models'][0]['realizations']
    param_dicts = [rn['parameters'] for rn in realizations]
    param_names = [
        p for p in list(param_dicts[0].keys()) \
        if 'value' in param_dicts[0][p]
    ]
    model_type = data['name']

    transients = []
    for p in param_dicts:
        model_params = {
            param: p[param]['value'] for param in param_names
        }

        transients.append(
            Transient(
                model_type,
                model_params,
            )
        )

    return transients






