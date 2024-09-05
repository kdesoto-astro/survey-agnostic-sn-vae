"""Converts set of Transient objects to arrays for ML training."""
import os
import glob
import datetime

import numpy as np
import h5py
from snapi import Transient, Photometry

now = datetime.datetime.now()
DATE = str(now.strftime("%Y-%m-%d"))

def prep_input(
        transient_dir,
        static_length=32,
        filter_instrument=None,
        save=False, outdir=None,
        save_fn=None,
        load=False, prep_file=None
    ):
    """
    Prep input file for fitting

    Parameters
    ----------
    transient_dir : str
        Where all transient files are stored
    static_length : int
        Length of LCs when padded
    filter_instrument: str
        if not None, only include LCs from this instrument
    save : bool
        Predicted flux values
    load : bool
        Predicted flux values
    outdir : str
        Predicted flux values
    prep_file : str
        Predicted flux values

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
    all_transient_fns = glob.glob(os.path.join(transient_dir, '*.hdf5'))
    num = len(all_transient_fns)
    nfilts = 6 # everything will be tiled to 6 filters
    nfiltsp1 = nfilts + 1
    nfiltsp2 = 2 * nfilts + 1
    nfiltsp3 = 3 * nfilts + 1
    nfiltsp4 = 4 * nfilts + 1

    dense_arrs = np.zeros((num, static_length, nfilts*5+1))
    meta_dict = {
        'ids': [],
        'classes': []
    }
    for i, transient_fn in enumerate(all_transient_fns):
        if i % 50 == 0:
            print(f"Pre-processed {i} out of {num} transients...")
        try:
            transient = Transient.load(transient_fn)
            photometry = transient.photometry
            if filter_instrument is not None:
                photometry.filter_by_instrument(filter_instrument)
            if len(photometry) < 2: # we do need at least 2 bands
                dense_arrs[i, :, :] = np.nan
                continue
            if len(photometry) > 6: # randomly select 6
                rand_idx = np.random.choice(len(photometry), 6, replace=False)
                photometry = Photometry(np.array(list(photometry.light_curves))[rand_idx])
            photometry.tile(6)
            photometry.phase()
            dense_arr = photometry.dense_array(error_mask=10)
            if dense_arr.shape[0] < static_length:
                dense_arrs[i,:dense_arr.shape[0]] = dense_arr
                dense_arrs[i,dense_arr.shape[0]:,0] = 1000.
                dense_arrs[i,dense_arr.shape[0]:,1:nfiltsp1] = -6 # much fainter than any transient
                dense_arrs[i,dense_arr.shape[0]:,nfiltsp1:nfiltsp2] = 10 # huge uncertainty
                dense_arrs[i,dense_arr.shape[0]:,nfiltsp2:nfiltsp3] = 1 # mask out later
                dense_arrs[i,dense_arr.shape[0]:,nfiltsp3:] = dense_arrs[i,0,nfiltsp3:] # constants 
            else:
                dense_arrs[i] = dense_arr[:static_length]
            meta_dict['ids'].append(transient.id)
            meta_dict['classes'].append(transient.spec_class)
        except Exception:
            dense_arrs[i, :, :] = np.nan
            continue
        
    for k in meta_dict:
        meta_dict[k] = np.asarray(meta_dict[k])

    # filter out nan rows
    dense_arrs = dense_arrs[~np.all(np.isnan(dense_arrs), axis=(1,2))]
    print(f"New number of events: {len(dense_arrs)}")

    # Flip because who needs negative magnitudes
    dense_arrs[:, :, 1:nfiltsp1] = -1.0 * dense_arrs[:, :, 1:nfiltsp1]

    if load and (prep_file is not None):
        prep_data = np.load(prep_file)
        bandmin = prep_data['bandmin']
        bandmax = prep_data['bandmax']
        wavemin = prep_data['wavemin']
        wavemax = prep_data['wavemax']
    else:
        timesteps_consider = dense_arrs[:, :, 0] < 1000.
        bandmin, bandmax = np.nanpercentile(dense_arrs[timesteps_consider, 1:nfiltsp1], q=[2., 98.])
        #bandmax = np.nanmax(dense_arrs[timesteps_consider, 1:nfiltsp1])
        wavemin, wavemax = np.nanpercentile(dense_arrs[timesteps_consider, nfiltsp3:nfiltsp4], q=[2., 98.])
        #wavemax = np.nanmax(dense_arrs[timesteps_consider, nfiltsp3:nfiltsp4])


    # rescale times by factor of 100
    dense_arrs[:,:,0] /= 100.
    
    # Normalize flux values, flux errors, and wavelengths to be between 0 and 1
    dense_arrs[:, :, 1:nfiltsp1] = (dense_arrs[:, :, 1:nfiltsp1] - bandmin) \
        / (bandmax - bandmin)
    dense_arrs[:, :, nfiltsp1:nfiltsp2] = (dense_arrs[:, :, nfiltsp1:nfiltsp2]) \
        / (bandmax - bandmin)
    dense_arrs[:, :, nfiltsp3:nfiltsp4] = (dense_arrs[:, :, nfiltsp3:nfiltsp4] - wavemin) \
        / (wavemax - wavemin)
    dense_arrs[:, :, nfiltsp4:] = (dense_arrs[:, :, nfiltsp4:]) \
        / (wavemax - wavemin)
    
    dense_arrs[:, :, nfiltsp1:nfiltsp2] = np.clip(dense_arrs[:,:,nfiltsp1:nfiltsp2], a_min=0.01, a_max=None)
    print(np.min(dense_arrs[:, :, nfiltsp1:nfiltsp2]), np.max(dense_arrs[:, :, nfiltsp1:nfiltsp2]))

    if (save_fn is None) and (outdir is not None):
        save_fn = os.path.join(outdir,'prep_'+DATE+'.h5')
    if save and (save_fn is not None):
        h5f = h5py.File(save_fn, 'w')
        ds = h5f.create_dataset('encoder_input', data=np.array(dense_arrs, dtype=np.float32))
        ds.attrs['wavemin'] = wavemin
        ds.attrs['wavemax'] = wavemax
        ds.attrs['bandmin'] = bandmin
        ds.attrs['bandmax'] = bandmax

        for k, dict_val in meta_dict.items():
            data_encoded = np.array([str(s).encode('utf-8') for s in dict_val])
            h5f.create_dataset(k, data=data_encoded)

        h5f.close()

    return (
        np.array(dense_arrs, dtype=np.float32),
        meta_dict
    )
