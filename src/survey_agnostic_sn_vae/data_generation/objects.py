import numpy as np
import mosfit
import os, glob
from typing import Dict
import json
import yaml
import pickle
import pathlib
from astropy.cosmology import Planck15 as cosmo
from astropy import units as un

from survey_agnostic_sn_vae.data_generation.utils import *

LIMITING_MAGS = {'LSST': 26.9, 'ZTF': 20.8, 'PanSTARRS': 23.3}
AVG_UNCERTAINTIES = {'LSST': 0.1, 'ZTF': 0.2, 'PanSTARRS': 0.12}
    
CONSTRAINT_FOLDER = os.path.join(
    pathlib.Path(__file__).parent.resolve(),
    "model_constraints"
)

DEFAULT_FITTER = mosfit.fitter.Fitter()

class ModelConstraints:
    def __init__(self, model_type):
        """Create dictionary of constraints based on
        model type.
        """
        default_dist = (cosmo.luminosity_distance(0.1) / un.Mpc).value
        #shared by all models
        default_constraints = {
            'redshift': 0.02, # constant for now
            'avhost': 0.0,
            'ebv': 0.0,
            'lumdist': default_dist,
            'texplosion': np.random.random()*50.0
        }
        constraint_fn = os.path.join(CONSTRAINT_FOLDER, f"{model_type}.yaml")
        
        with open(constraint_fn, "r", encoding="utf-8") as fh:
            self.model_constraints = yaml.safe_load(fh)
        
        for c in default_constraints:
            self.model_constraints[c] = default_constraints[c]
            
        print(self.model_constraints)
        
    def to_list(self):
        """Convert model constraints to list to feed into
        MOSFIT's fitter.
        """
        model_list = []
        
        for c in self.model_constraints:
            model_list.append(c)
            model_list.append(self.model_constraints[c])
            
        print(model_list)
        return model_list
        
        
class Survey:
    def __init__(self, name, bands, cadence):
        self.name = name
        self.limiting_magnitude = None
        self.avg_uncertainty = 0.1
        if self.name in LIMITING_MAGS.keys():
            self.limiting_magnitude = LIMITING_MAGS[self.name]
            self.avg_uncertainty = AVG_UNCERTAINTIES[self.name]
            
        self.bands = bands
        self.cadence = cadence
        self.band_wavelengths = {}
        
        orig_path = os.getcwd()
    
        mosfit_path = os.path.dirname(
        os.path.realpath(mosfit.__file__)
    )

        print("Switching to MOSFIT path: %s" % mosfit_path)
        os.chdir(mosfit_path)
        
        with suppress_stdout():
            # generate initial LCs/model params
            DEFAULT_FITTER.fit_events(
                models=['slsn'],
                max_time=500.0,
                iterations=0,
                limiting_magnitude = self.limiting_magnitude,
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
        
    def generate_sample_times(self, final_time, max_points=50):
        """Approximate sampling cadence for survey.
        """
        times = {}
        if np.isscalar(self.cadence):
            ref_band = self.bands[0]
            day_offset = np.round(
                np.random.normal(scale=1.0, size=max_points)
            ).astype(int)
            day_offset[day_offset < -1*self.cadence + 1] = 0.0
            small_offsets = np.random.normal(scale=0.1, size=max_points)
            small_offsets = np.clip(small_offsets, a_min=0.25, a_max=0.25)
            times[ref_band] = self.cadence + day_offset + small_offsets
            times[ref_band] = np.cumsum(times[ref_band])
            
            for b in self.bands:
                if b == ref_band:
                    continue
                small_offsets = np.random.normal(scale=0.1, size=max_points)
                small_offsets = np.clip(small_offsets, a_min=0.25, a_max=0.25)

                times[b] = times[ref_band] + small_offsets
                print(times[b])

                times[b] = times[b][times[b] < final_time]
                

        else:
            for b in self.bands:
                day_offset = np.round(
                    np.random.normal(scale=1.0, size=max_points)
                ).astype(int)
                day_offset[day_offset < -1*self.cadence[b] + 1] = 0.0
                small_offsets = np.random.normal(scale=0.1, size=max_points)
                times[b] = self.cadence[b] + day_offset + small_offsets
                times[b] = np.cumsum(times[b])
                print(times[b])
                times[b] = times[b][times[b] < final_time]
                
        t_list = []
        for b in times:
            times[b] -= times[b][0]
            t_list.extend(list(times[b]))

        return times, sorted(t_list)

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
        self, survey, output_path, max_time=500,
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
        s_times, t_list = survey.generate_sample_times(max_time)
        
        fitter.fit_events(
            models=[self.model_type,],
            time_list=t_list,
            band_list=s_bands,
            band_instruments=[s_name,],
            max_time=500.0,
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
        phot_arrs = extract_photometry(data, s_times)

        lc = LightCurve.from_arrays(
            *phot_arrs, survey,
            obj_id=fitter._event_name,
            transient_id=self.obj_id
        )
        lc.add_noise()
        lc.apply_limiting_mag()
        
        self.lightcurves.append(lc)
        
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
        self, times, mag,
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
        print(self.bands)
        if mag.keys() != mag_err.keys():
            raise ValueError(
                "Make sure mag and mag err have the same bands"
            )
        self.times = times
        self.mag = mag
        self.mag_err = mag_err
        self.survey = survey
        
        # cast correct types
        for b in self.bands:
            self.times[b] = self.times[b].astype(float)
            self.mag[b] = self.mag[b].astype(float)
            self.mag_err[b] = self.mag_err[b].astype(float)
        
    def get_arrays(self):
        t_arr, m_arr, m_err_arr, b_arr = [], [], [], []
        for b in self.bands:
            t_arr.extend(self.times[b])
            m_arr.extend(self.mag[b])
            m_err_arr.extend(self.mag_err[b])
            b_arr.extend([b,] * len(self.times[b]))
                
        return (
            np.asarray(t_arr),
            np.asarray(m_arr),
            np.asarray(m_err_arr),
            np.asarray(b_arr)
        )
    
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
        
        t_dict = {}
        m_dict = {}
        m_err_dict = {}
        
        for b in np.unique(bands):
            band_idxs = bands_sorted == b
            times_b = times_sorted[band_idxs]
            mags_b = mags_sorted[band_idxs]
            mag_errors_b = mag_errors_sorted[band_idxs]
            assert len(mags_b) == len(mag_errors_b)
            assert len(mags_b) == len(times_b)
            
            t_dict[b] = times_b
            m_dict[b] = mags_b
            m_err_dict[b] = mag_errors_b
        
        return cls(
            t_dict, m_dict,
            m_err_dict, survey,
            **kwargs
        )
            
            
    def find_peak_mag(self, bands = None, composite = False):
        if bands == None:
            bands = self.bands

        max_times = {}
        peak_mags = {}
        b_inc = []
        for band in bands:
            if len(self.times[band]) == 0:
                continue
            max_index = np.argmin(self.mag[band])
            max_times[band] = self.times[band][max_index]
            peak_mags[band] = self.mag[band][max_index]
            b_inc.append(band)
        
        if composite:
            peak_mag_arr = [peak_mags[b] for b in b_inc]
            max_band = b_inc[np.argmin(peak_mag_arr)]

            return max_times[max_band], peak_mags[max_band]
        
        return max_times, peak_mags
    
    
    def apply_limiting_mag(self):
        """Remove points according to limiting magnitude.
        """
        for b in self.bands:
            keep_idx = (self.mag[b] < self.survey.limiting_magnitude)
            self.times[b] = self.times[b][keep_idx]
            self.mag[b] = self.mag[b][keep_idx]
            self.mag_err[b] = self.mag_err[b][keep_idx]
        
    def add_noise(self):
        """Turn from clean LC to noisy.
        """
        # first, fix uncertainties
        for b in self.bands:
            self.mag_err[b] = np.clip(
                np.random.normal(
                    loc=self.survey.avg_uncertainty,
                    scale=self.survey.avg_uncertainty / 5,
                    size=len(self.mag_err[b])
                ),
                a_min=self.survey.avg_uncertainty / 2,
                a_max=np.inf
            )
            # then, add noise to mags based on mag_err
            self.mag[b] += np.random.normal(
                scale=self.mag_err[b]
            )
            
            
        
        
        
        
        
        
        