import json
import numpy as np
from typing import Dict, List, Any
import os

def open_walkers_file(
    walkers_file: str
) -> Dict:
    """Open walkers JSON file into dictionary.
    
    Parameters
    ----------
    walkers_file : str
        The file where walkers (+ photometry) info is stored.
    """
    with open(walkers_file, 'r', encoding = 'utf-8') as f:
        data = json.loads(f.read())
        while 'name' not in data: # recursive
            data = data[list(data.keys())[0]]
    return data

def extract_photometry(
    data: Dict
) -> np.array:
    """Extract photometry from a MOSFIT walkers file.
    Assumes ONE LIGHT CURVE.
    
    Parameters
    ----------
    data : dict
        Stored walkers (inc. photometry) info
    
    Returns
    ----------
    numpy array of shape (4, num_times)
        The time, mag, mag_error, and filter arrays.
    """
    
    phot = data['photometry']
    used_bands = []
    
    band_attr = ['band', 'instrument', 'telescope', 'system', 'bandset']
    band_list = list(set([
        tuple(
            x.get(y, '') for y in band_attr
        ) for x in phot if 'band' in x and 'magnitude' in x
    ]))
    
    for full_band in band_list:
        (band, inst, tele, syst, bset) = full_band
        
        realization = []
        
        for ph in phot:
            if not tuple(ph.get(y, '') for y in band_attr) == full_band:
                continue
            
            upper_mag = float(
                ph.get(
                    'e_upper_magnitude',
                    ph.get('e_magnitude', 0.0)
                )
            )
            lower_mag = float(
                ph.get(
                    'e_lower_magnitude',
                    ph.get('e_magnitude', 0.0)
                )
            )
            m_err = max(upper_mag, lower_mag)
            realization.append([
                float(ph['time']),
                float(ph['magnitude']),
                m_err,
                band
            ])
            
    return np.asarray(realization).T
    
    
            
            
            