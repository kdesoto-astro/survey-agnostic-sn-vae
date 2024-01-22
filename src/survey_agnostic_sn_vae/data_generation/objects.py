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
        if flux.keys() !== flux_err.keys():
            raise ValueError("Make sure flux and flux err have the same bands")
        self.flux = flux
        self.flux_err = flux_err
        self.survey = survey
        
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
            
        
        
        
        
        
        
        
        