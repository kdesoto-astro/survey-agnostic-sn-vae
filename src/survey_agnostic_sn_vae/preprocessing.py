import os, glob
import numpy as np
import datetime

from survey_agnostic_sn_vae.data_generation.utils import *
from survey_agnostic_sn_vae.data_generation.objects import Transient

from superraenn.preprocess import save_lcs
from superraenn.lc import LightCurve

DEFAULT_ZPT = 26.3
now = datetime.datetime.now()
date = str(now.strftime("%Y-%m-%d"))

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

    transient_id = 0

    # Update the LC objects with info from the metatable
    for i, transient in enumerate(all_transients):
        print(i)
        params = transient.model_params
        if params['redshift'] <= 0.0:
            continue
            
        for lc in transient.lightcurves:
            t, m, merr, b = lc.get_arrays()
            
            if len(t) < 5:
                continue
                
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
            
            sr_lc.group = transient_id

            sr_lc.get_abs_mags()
            sr_lc.sort_lc()
            pmjd = sr_lc.find_peak(
                lc.find_peak_mag(composite=True)[0]
            )
            sr_lc.shift_lc(pmjd)
            sr_lc.correct_time_dilation()
            filt_dict = {f: i for i, f in enumerate(lc.bands)}

            sr_lc.wavelengths = np.zeros(len(lc.bands))
            for f in filt_dict:
                sr_lc.wavelengths[filt_dict[f]] = lc.survey.band_wavelengths[f]
                
            sr_lc.filter_names_to_numbers(filt_dict)
            sr_lc.cut_lc()
            try:
                sr_lc.make_dense_LC(len(lc.bands))
            except:
                print("SKIPPED")
                continue
            
            # pad dense LC to six bands
            if len(lc.bands) <= 3:
                tile_factor = int(6 / len(lc.bands))
                sr_lc.dense_lc = np.repeat(
                    sr_lc.dense_lc, tile_factor,
                    axis=1
                )
                sr_lc.wavelengths = np.repeat(
                    sr_lc.wavelengths, tile_factor,
                )
                
            elif len(lc.bands) < 6:
                sr_lc.dense_lc = np.repeat(
                    sr_lc.dense_lc, 2,
                    axis=1
                )[:,:6,:]
                sr_lc.wavelengths = np.repeat(
                    sr_lc.wavelengths, 2
                )[:6]
                
            sr_lcs.append(sr_lc)
        transient_id += 1
            
    save_lcs(sr_lcs, save_dir)
    
    
def prep_input(input_lc_file, new_t_max=200.0, filler_err=3.0,
               save=False, load=False, outdir=None, prep_file=None):
    """
    Prep input file for fitting

    Parameters
    ----------
    input_lc_file : str
        True flux values
    new_t_max : float
        Predicted flux values
    filler_err : float
        Predicted flux values
    save : bool
        Predicted flux values
    load : bool
        Predicted flux values
    outdir : str
        Predicted flux values
    prep_file : str
        Predicted flux values

    Returns
    -------
    sequence : numpy.ndarray
        Array LC flux times, values and errors
    outseq : numpy.ndarray
        An array of LC flux values and limiting magnitudes
    ids : numpy.ndarray
        Array of SN names
    sequence_len : float
        Maximum length of LC values
    nfilts : int
        Number of filters in LC files
    """
    lightcurves = np.load(input_lc_file, allow_pickle=True)['lcs']
    lengths = []
    ids = []
    for lightcurve in lightcurves:
        lengths.append(len(lightcurve.times))
        ids.append(lightcurve.name)

    sequence_len = 183 #np.max(lengths)
    nfilts = np.shape(lightcurves[0].dense_lc)[1]
    nfiltsp1 = nfilts+1
    nfiltsp2 = 2*nfilts+1
    n_lcs = len(lightcurves)
    # convert from LC format to list of arrays
    # sequence = np.zeros((n_lcs, sequence_len, nfilts*2+1))
    sequence = np.zeros((n_lcs, sequence_len, nfilts*3 + 2))

    lms = []
    for i, lightcurve in enumerate(lightcurves):
        sequence[i, 0:lengths[i], 0] = lightcurve.times[:sequence_len]
        sequence[i, 0:lengths[i], 1:nfiltsp1] = lightcurve.dense_lc[:sequence_len, :, 0] # fluxes
        sequence[i, 0:lengths[i], nfiltsp1:nfiltsp2] = lightcurve.dense_lc[:sequence_len, :, 1] # flux errors
        sequence[i, lengths[i]:, 0] = np.max(lightcurve.times)+new_t_max
        sequence[i, lengths[i]:, 1:nfiltsp1] = lightcurve.abs_lim_mag
        sequence[i, lengths[i]:, nfiltsp1:nfiltsp2] = filler_err
        sequence[i, :, nfiltsp2:-1] = lightcurve.wavelengths
        sequence[i, :, -1] = lightcurve.group
        
        lms.append(lightcurve.abs_lim_mag)

    # Flip because who needs negative magnitudes
    sequence[:, :, 1:nfiltsp1] = -1.0 * sequence[:, :, 1:nfiltsp1]
    
    if load:
        prep_data = np.load(prep_file)
        bandmin = prep_data['bandmin']
        bandmax = prep_data['bandmax']
        wavemin = prep_data['wavemin']
        wavemax = prep_data['wavemax']
    else:
        bandmin = np.min(sequence[:, :, 1:nfiltsp1])
        bandmax = np.max(sequence[:, :, 1:nfiltsp1])
        wavemin = np.min(sequence[:, :, nfiltsp2:-1])
        wavemax = np.max(sequence[:, :, nfiltsp2:-1])

    # Normalize flux values, flux errors, and wavelengths to be between 0 and 1
    #bandmin = np.min(sequence[:, :, 1:nfiltsp1], axis=1)
    #bandmax = np.max(sequence[:, :, 1:nfiltsp1], axis=1)
    sequence[:, :, 1:nfiltsp1] = (sequence[:, :, 1:nfiltsp1] - bandmin) \
        / (bandmax - bandmin)
    sequence[:, :, nfiltsp1:nfiltsp2] = (sequence[:, :, nfiltsp1:nfiltsp2]) \
        / (bandmax - bandmin)
    sequence[:, :, nfiltsp2:-1] = (sequence[:, :, nfiltsp2:-1] - wavemin) \
        / (wavemax - wavemin)

    new_lms = np.reshape(np.repeat(lms, sequence_len), (len(lms), -1))

    outseq = np.reshape(sequence[:, :, 0], (len(sequence), sequence_len, 1)) * 1.0
    outseq = np.dstack((outseq, new_lms))
    # tile for each wv
    outseq_tiled = np.repeat(outseq, 6, axis=1)
    outseq_wvs = np.reshape(sequence[:, :, nfiltsp2:-1], (len(sequence), nfilts*sequence_len, 1)) * 1.0
    outseq_tiled = np.dstack((outseq_tiled, outseq_wvs))
        
    if save:
        model_prep_file = os.path.join(outdir,'prep_'+date+'.npz')
        np.savez(model_prep_file, wavemin=wavemin, wavemax=wavemax, bandmin=bandmin, bandmax=bandmax)
        model_prep_file = os.path.join(outdir,'prep.npz')
        np.savez(model_prep_file, wavemin=wavemin, wavemax=wavemax, bandmin=bandmin, bandmax=bandmax)
    return sequence, outseq_tiled, ids, sequence_len, nfilts
