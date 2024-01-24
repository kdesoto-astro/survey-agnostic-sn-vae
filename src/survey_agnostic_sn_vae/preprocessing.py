import os, glob
from survey_agnostic_sn_vae.data_generation.utils import *
from survey_agnostic_sn_vae.data_generation.objects import Transient

from superraenn.preprocess import save_lcs
from superraenn.lc import LightCurve

DEFAULT_ZPT = 26.3

def generate_superraenn_lc_file(
    transient_dir,
    save_dir
):
    """Generate lcs.npz file for use by SuperRAENN.
    """
    all_transient_files = glob.glob(
        os.path.join(transient_dir, "*.pickle")
    )
    all_transients = [
        Transient.load(t) for t in all_transient_files
    ]
    
    sr_lcs = []

    # Update the LC objects with info from the metatable
    for transient in all_transients:
        params = transient.model_params
        if params['redshift'] <= 0.0:
            continue
            
        for lc in transient.lightcurves:
        
            t, m, merr, b = lc.get_arrays()
            
            f, ferr = convert_mag_to_flux(
                m, merr, DEFAULT_ZPT
            )
            
            sr_lc = LightCurve(
                name=lc.obj_id,
                times=t,
                fluxes=f,
                flux_errs=ferr,
                filters=b,
            )
            
            sr_lc.add_LC_info(
                zpt=DEFAULT_ZPT,
                mwebv=0.0,
                redshift=params['redshift'],
                lim_mag=lc.survey.limiting_magnitude,
                obj_type=transient.model_type
            )
            
            sr_lc.wavelengths = lc.survey.band_wavelengths
            sr_lc.get_abs_mags()
            sr_lc.sort_lc()
            pmjd = sr_lc.find_peak(
                lc.find_peak_mag(composite=True)[0]
            )
            sr_lc.shift_lc(pmjd)
            sr_lc.correct_time_dilation()
            filt_dict = {f: i for i, f in enumerate(lc.bands)}
            sr_lc.filter_names_to_numbers(filt_dict)
            sr_lc.cut_lc()
            sr_lc.make_dense_LC(len(lc.bands))
            
            # pad dense LC to six bands
            if len(lc.bands) <= 3:
                tile_factor = int(6 / len(lc.bands))
                sr_lc.dense_lc = np.repeat(
                    sr_lc.dense_lc, tile_factor,
                    axis=1
                )
                print(sr_lc.dense_lc.shape)
                
            elif len(lc.bands) < 6:
                sr_lc.dense_lc = np.repeat(
                    sr_lc.dense_lc, 2,
                    axis=1
                )[:,:6,:]
                print(sr_lc.dense_lc.shape)
            
            sr_lcs.append(sr_lc)
            
    save_lcs(sr_lcs, save_dir)