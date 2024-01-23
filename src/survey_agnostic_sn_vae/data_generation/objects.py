import numpy as np
import mosfit
import os, glob
from typing import Dict
import json
import pickle

from survey_agnostic_sn_vae.data_generation.utils import *

DEFAULT_FITTER = mosfit.fitter.Fitter()


class Survey:
    def __init__(self, name, bands, cadence):
        self.name = name
        self.bands = bands
        self.cadence = cadence
        self.band_wavelengths = {}
        
        orig_path = os.getcwd()
    
        mosfit_path = os.path.dirname(
        os.path.realpath(mosfit.__file__)
    )

        print("Switching to MOSFIT path: %s" % mosfit_path)
        os.chdir(mosfit_path)
        # generate initial LCs/model params
        DEFAULT_FITTER.fit_events(
            models=['slsn'],
            max_time=1000.0,
            iterations=0,
            write=False,
            time_list=[1,2],
            band_list=self.bands,
            band_instruments=self.name,
        )
        model = DEFAULT_FITTER._model
        
        for task in model._call_stack:
            cur_task = model._call_stack[task]
            mod_name = cur_task.get('class', task)
            if mod_name == 'photometry':
                photometry = model._modules[task]
                num_bands = len(photometry._unique_bands)
                average_wavelengths = photometry._average_wavelengths
                mask = average_wavelengths != 0
                for i in range(num_bands):
                    if mask[i]:
                        self.band_wavelengths[photometry._unique_bands[i]['name']] = photometry._average_wavelengths[i]
                        
        print("Switching back to original working directory")
        os.chdir(orig_path)
        
    def generate_sample_times(self, num_points):
        initial_time = -5 + np.random.random_sample() * 10
        return [initial_time + self.cadence * i for i in range(num_points)]

    def print_band_wavelengths(self):
        for k in self.band_wavelengths.keys():
            print(k, self.band_wavelengths[k])
    
class Transient:
    """Container for Transient object.
    Contains MOSFIT model used, model parameters,
    and all LightCurve objects generated from those
    parameters.
    """
    def __init__(
        self, model_type, model_params,
        obj_id=None,
        **kwargs
    ):
        if obj_id is None:
            obj_id = str(hash(self))
            
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
        
        tmp_dir = os.path.join(output_path, "products")
        
        file_loc = os.path.join(
            tmp_dir,
            f"{self.model_type}_{self.obj_id}.json"
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
        
    def save(self, output_dir):
        """Save Transient object to pickle file.
        """
        save_name = os.path.join(
            output_dir,
            f"{self.model_type}_{self.obj_id}.pickle"
        )
        
        with open(save_name, 'wb') as handle:
            pickle.dump(
                self, handle,
                protocol=pickle.HIGHEST_PROTOCOL
            )
            
    @classmethod
    def load(cls, filepath):
        """Load Transient object from pickle file.
        """
        with open(filepath, 'rb') as handle:
            obj = pickle.load(handle)
            return obj
        
        
class LightCurve:
    def __init__(
        self, timepoints, mag,
        mag_err, survey,
        obj_id=None,
        transient_id=None,
        **kwargs
    ):
        if obj_id is None:
            obj_id = str(hash(self))
        self.obj_id = obj_id
        self.transient_id = transient_id
        
        self.bands = np.asarray(list(mag.keys()))
        self.timepoints = timepoints
        if mag.keys() != mag_err.keys():
            raise ValueError(
                "Make sure mag and mag err have the same bands"
            )
        self.mag = mag
        self.mag_err = mag_err
        self.survey = survey
        
    def get_arrays(self):
        t_arr = np.tile(self.timepoints, len(self.bands))
        m_arr = np.ravel([self.mag[b] for b in self.bands])
        m_err_arr = np.ravel([
            self.mag_err[b] for b in self.bands
        ])
        b_arr = np.ravel([
            np.tile(b, len(self.timepoints)) for b in self.bands
        ])
                
        return t_arr, m_arr, m_err_arr, b_arr
    
    @classmethod
    def from_arrays(
        cls,
        times,
        mags,
        mag_errors,
        bands,
        survey,
        **kwargs
    ):
        """Generate LightCurve object dictionaries from
        equal-dimensioned arrays."""
        if not (
            len(times) == len(mags) == len(mag_errors) == len(bands)
        ):
            raise ValueError("Input arrays must all be the same length.")
                    
        times = np.asarray(times)
        mags = np.asarray(mags)
        mag_errors = np.asarray(mag_errors)
        bands = np.asarray(bands)
        
        sort_idxs = np.argsort(times)
        
        times_sorted = times[sort_idxs]
        mags_sorted = mags[sort_idxs]
        mag_errors_sorted = mag_errors[sort_idxs]
        bands_sorted = bands[sort_idxs]
        
        
        # TODO: this currently assumes every band is sampled for each
        # timestamp - add method for band interpolation
        
        m_dict = {}
        m_err_dict = {}
        
        t_unique = np.unique(times_sorted)
        for b in np.unique(bands):
            band_idxs = bands_sorted == b
            mags_b = mags_sorted[band_idxs]
            mag_errors_b = mag_errors_sorted[band_idxs]
            assert len(mags_b) == len(mag_errors_b)
            assert len(mags_b) == len(t_unique)
            
            m_dict[b] = mags_b
            m_err_dict[b] = mag_errors_b
        
        return cls(
            t_unique, f_dict,
            f_err_dict, survey,
            **kwargs
        )
            
            
    def find_peak_mag(self, bands = None):
        if bands == None:
            bands = self.bands
        max_times = {}
        peak_mags = {}
        for band in bands:
            max_index = np.argmin(self.mag[band])
            max_times[band] = self.timepoints[max_index]
            peak_mags[band] = self.mag[band][max_index]
        return max_times, peak_mags
    
    
    def get_encoding_format(self):
        pass
            
        
        
        
        
        
        
        