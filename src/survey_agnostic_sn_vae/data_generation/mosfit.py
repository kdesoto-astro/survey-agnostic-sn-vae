"""Module to generate the MOSFIT light curves for VAE training."""

import mosfit
import numpy as np
import objects

def generate_LCs_from_model(
    model_name: str,
    num=1000
) -> list[LightCurve]:
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


def extract_survey_observables(
    lightcurve: LightCurve,
    survey_list: SurveyList,
) -> FilteredLightCurveSet:
    """Extract light curves as observed through
    each survey in survey_list.

    Parameters
    ----------
    lightcurve : LightCurve
        modeled MOSFIT light curve object
    survey_list : SurveyList
        The Surveys to return observables for.

    Returns
    -------
    FilteredLightCurves
        Collection of observed light curves for each survey.
    """
    pass
