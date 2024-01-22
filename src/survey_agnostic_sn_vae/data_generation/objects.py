import numpy as np

class Survey:
    def __init__(self, name, bands, cadence):
        self.name = name
        self.bands = bands
        self.cadence = cadence
        
    def generate_sample_times(self, num_points):
        initial_time = -5 + np.random.random_sample() * 10
        return [initial_time + self.cadence * i for i in range(num_points)]
        

        
class LightCurve:
    def __init__(self, timepoints, flux, flux_err, survey):
        self.bands = flux.keys()
        self.timepoints = timepoints
        if flux.keys() != flux_err.keys():
            raise ValueError("Make sure flux and flux err have the same bands")
        self.flux = flux
        self.flux_err = flux_err
        self.survey = survey
        
    @class_method
    def from_arrays(cls, times, fluxes, flux_errors, bands, survey):
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
        
        return cls(t_unique, f_dict, f_err_dict, survey)
            
            
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
            
        
        
        
        
        
        
        
        