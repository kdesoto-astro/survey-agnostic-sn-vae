import numpy as np
import mosfit
import os
from typing import Dict
import json

from .utils import *

DEFAULT_FITTER = mosfit.fitter.Fitter()


def generate_random_id():
    """Just make it unlikely to repeat."""
    num = np.random.randint(0, 999_999_999)
    return str(num).zfill(9)


class Survey:
    def __init__(self, name, bands, cadence):
        self.name = name
        self.bands = bands
        self.cadence = cadence
        
    def generate_sample_times(self, num_points):
        initial_time = -5 + np.random.random_sample() * 10
        return [initial_time + self.cadence * i for i in range(num_points)]

    
class Transient:
    """Container for Transient object.
    Contains MOSFIT model used, model parameters,
    and all LightCurve objects generated from those
    parameters.
    """
    def __init__(
        self, model_type, model_params,
        obj_id=generate_random_id(),
        **kwargs
    ):
        self.obj_id = obj_id
        self.model_type = model_type
        self.model_params = model_params
        self.fixed_param_list = []
        
        for p in self.model_params:
            self.fixed_param_list.append(p)
            self.fixed_param_list.append(
                self.model_params[p]
            )
            
        self.lightcurves = []
            

    def generate_lightcurve(
        self, survey, output_path, num_times=30,
        fitter=DEFAULT_FITTER
    ):
        """Generate LightCurve object for a given
        Survey, from the model_params specified.
        
        Parameters
        ----------
        survey : Survey
            The survey to use to generate LightCurve
        """
        s_name = survey.name
        s_bands = survey.bands
        s_times = survey.generate_sample_times(num_times)
        
        fitter.fit_events(
            models=[self.model_type,],
            time_list=s_times,
            band_list=s_bands,
            band_instruments=[s_name,],
            max_time=200.0,
            iterations=0,
            write=True,
            output_path=output_path,
            num_walkers=1,
            user_fixed_parameters=self.fixed_param_list,
            suffix=self.obj_id
        )
        
        file_loc = os.path.join(
            output_path,
            f"products/{self.model_type}_{self.obj_id}.json"
        )
        data = open_walkers_file(file_loc)
        phot_arrs = extract_photometry(data)
            
        self.lightcurves.append(
            LightCurve.from_arrays(
                *phot_arrs, survey,
                obj_id=fitter._event_name,
                transient_id=self.obj_id
            )
        )
        
        
class LightCurve:
    def __init__(
        self, timepoints, flux,
        flux_err, survey,
        obj_id=generate_random_id(),
        transient_id=None,
        **kwargs
    ):
        self.obj_id = obj_id
        self.transient_id = transient_id
        
        self.bands = flux.keys()
        self.timepoints = timepoints
        if flux.keys() != flux_err.keys():
            raise ValueError("Make sure flux and flux err have the same bands")
        self.flux = flux
        self.flux_err = flux_err
        self.survey = survey
        
    @classmethod
    def from_arrays(
        cls,
        times,
        fluxes,
        flux_errors,
        bands,
        survey,
        **kwargs
    ):
        """Generate LightCurve object dictionaries from
        equal-dimensioned arrays."""
        if not (
            len(times) == len(fluxes) == len(flux_errors) == len(bands)
        ):
            raise ValueError("Input arrays must all be the same length.")
                    
        times = np.asarray(times)
        fluxes = np.asarray(fluxes)
        flux_errors = np.asarray(flux_errors)
        bands = np.asarray(bands)
        
        sort_idxs = np.argsort(times)
        
        times_sorted = times[sort_idxs]
        fluxes_sorted = fluxes[sort_idxs]
        flux_errors_sorted = flux_errors[sort_idxs]
        bands_sorted = bands[sort_idxs]
        
        # TODO: this currently assumes every band is sampled for each
        # timestamp - add method for band interpolation
        
        f_dict = {}
        f_err_dict = {}
        
        t_unique = np.unique(times_sorted)
        for b in np.unique(bands):
            band_idxs = bands_sorted == b
            fluxes_b = fluxes_sorted[band_idxs]
            flux_errors_b = flux_errors_sorted[band_idxs]
            assert len(fluxes_b) == len(flux_errors_b)
            assert len(fluxes_b) == len(t_unique)
            
            f_dict[b] = fluxes_b
            f_err_dict[b] = flux_errors_b
        
        return cls(t_unique, f_dict, f_err_dict, survey, **kwargs)
            
            
    def find_max_flux(self, bands = None):
        if bands == None:
            bands = self.bands
        max_times = {}
        max_fluxes = {}
        for band in bands:
            max_index = np.argmax(self.flux[band])
            max_times[band] = self.timepoints[max_index]
            max_fluxes[band] = self.flux[band][max_index]
        return max_times, max_fluxes
    
    def get_encoding_format(self):
        pass
            
        
        
        
        
        
        
        
        