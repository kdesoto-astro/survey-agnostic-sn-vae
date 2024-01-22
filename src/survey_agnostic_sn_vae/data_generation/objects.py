

class Survey:
    def __init__(self, name, bands, cadence):
        self.name = name
        self.bands = bands
        self.cadence = cadence
        
    def generate_sample_times(self, num_points):
        initial_time = -5 + np.random.random_sample() * 10
        return [initial_time + self.cadence * i for i in range(num_points)]
        
        
        
        
        

        
class LightCurve:
    def __init__(self, bands, timepoints, flux, flux_err, survey):
        self.bands = bands
        self.timepoints = timepoints
        
        if self.flux.shape != (len(self.bands), len(self.timepoints)):
            raise ValueError('Flux array shape should be (bands, timepoints)')
            
        if self.flux_err.shape != self.flux.shape:
            raise ValueError('Flux and flux err should be the same shape')
            
        self.flux = flux
        self.flux_err = flux_err
        self.survey = survey
        
    def find_max_flux(self, band = None):
        if band == None:
            best_indexes = np.argmax(flux, dims=1)
            return best_indexes
        
        
        
        
        
        
        
        