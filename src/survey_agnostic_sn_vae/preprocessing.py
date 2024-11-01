"""Converts set of Transient objects to arrays for ML training."""
import os
import glob
import datetime

import numpy as np
import h5py
from snapi import Transient

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
    all_transient_fns = glob.glob(os.path.join(transient_dir, '*.h5'))
    num = len(all_transient_fns)

    dense_arrs = []
    ids = []
    meta_dict = {
        'class': []
    }
    for i, transient_fn in enumerate(all_transient_fns):
        if i % 50 == 0:
            print(f"Pre-processed {i} out of {num} transients...")
        #try:
        transient = Transient.load(transient_fn)
        photometry = transient.photometry
        if len(photometry) < 2: # we do need at least 2 bands
            continue
        if filter_instrument is not None:
            photometry = photometry.filter_by_instrument(filter_instrument)
        if len(photometry) == 0: # TODO: why difference with abvoe statement?
            continue
        skip=False
        for lc in photometry.light_curves:
            if len(lc) > 64:
                skip=True # just skip it for now
                break
        if skip:
            continue
        dense_arr = photometry.dense_array(error_mask=10)
        dense_arr[:,:,1] *= -1 # get rid of negative mags
        dense_arr[:,:,0] /= 100. # get closer to unity
        dense_arrs.append(dense_arr)
        ids.append(transient_fn.split("/")[-1].split(".")[0])
        meta_dict['class'].append(str(transient.spec_class))
        #except Exception:
        #    continue

    # filter out nan rows
    print(f"New number of events: {len(dense_arrs)}")

    if load and (prep_file is not None):
        with h5py.File(prep_file, 'r') as prep_data:
            bandmin = prep_data['encoder_input'].attrs['bandmin']
            bandmax = prep_data['encoder_input'].attrs['bandmax']
            wavemin = prep_data['encoder_input'].attrs['wavemin']
            wavemax = prep_data['encoder_input'].attrs['wavemax']
    else:
        all_f = np.concatenate([dense_arr[:,:,1].ravel() for dense_arr in dense_arrs], axis=None)
        all_wv = np.concatenate([dense_arr[:,:,4].ravel() for dense_arr in dense_arrs], axis=None)
        bandmin, bandmax = np.nanpercentile(all_f, q=[2., 98.])
        wavemin, wavemax = np.nanpercentile(all_wv, q=[2., 98.])

    if (save_fn is None) and (outdir is not None):
        save_fn = os.path.join(outdir,'prep_'+DATE+'.h5')
        
    os.remove(save_fn)
    h5f = h5py.File(save_fn, 'w')
    h5f.attrs['wavemin'] = wavemin
    h5f.attrs['wavemax'] = wavemax
    h5f.attrs['bandmin'] = bandmin
    h5f.attrs['bandmax'] = bandmax
    
    # Normalize flux values, flux errors, and wavelengths to be between 0 and 1
    for i, dense_arr in enumerate(dense_arrs):
        dense_arr[:, :, 1] = (dense_arr[:, :, 1] - bandmin) / (bandmax - bandmin)
        dense_arr[:, :, 2] /= (bandmax - bandmin)
        dense_arr[:, :, 2] = np.clip(dense_arr[:,:,2], a_min=0.01, a_max=None)
        dense_arr[:, :, 4] = (dense_arr[:, :, 4] - wavemin) / (wavemax - wavemin)
        dense_arr[:, :, 5] /= (wavemax - wavemin)
        
        ds = h5f.create_dataset(ids[i], data=dense_arr)
        for k, dict_val in meta_dict.items():
            ds.attrs[k] = dict_val[i]

    h5f.close()

    return (
        dense_arrs,
        ids,
        meta_dict
    )
