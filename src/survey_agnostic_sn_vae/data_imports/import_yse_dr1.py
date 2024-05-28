import os, glob
import pandas as pd
from survey_agnostic_sn_vae.preprocessing import save_lcs
from survey_agnostic_sn_vae.data_imports.import_helper import create_lc

import numpy as np
#import matplotlib.pyplot as plt

DEFAULT_ZPT = 27.5
# X for ZTF-*g*; Y for ZTF-*r*
# no y-band for Panstarrs

LIM_MAGS = {
    'g': 22.0,
    'r': 21.8,
    'i': 21.5,
    'z': 20.9,
    'X': 20.8,
    'Y': 20.6
}

# TODO: change limmags in raenn to be per-datapoint

BAND_WIDTHS = {
    'g':1148.66,
    'r':1397.73,
    'i':1292.39,
    'z':1038.82,
    'X':1317.15,
    'Y':1488.99
}
# lambda-cen
BAND_WAVELENGTHS = {
    'g': 4936.01,
    'r': 6206.17,
    'i': 7553.48,
    'z': 8704.75,
    'X': 4805.92,
    'Y': 6435.87,
}


def find_meta_start_of_data(fn):
    """Helper function for YSE fn import.
    """
    obj_name = fn.split("/")[-1].split(".")[0]
    meta = {'NAME': obj_name}
    meta_keys = [
        'MWEBV',
        'REDSHIFT_FINAL',
        'SPEC_CLASS',
        'SPEC_CLASS_BROAD'
    ]
    with open(fn, 'r') as f:
        for i, row in enumerate(f):
            for k in meta_keys:
                if k in row:
                    meta[k] = row.split(":")[1].split("+")[0].strip()
            if row.strip() == '':
                return i + 5, meta

def import_single_yse_lc(fn):
    """Imports single YSE SNANA file.
    """
    header, meta = find_meta_start_of_data(fn)
    meta['LIM_MAGS'] = LIM_MAGS
    meta['BAND_WIDTHS'] = BAND_WIDTHS
    meta['BAND_WAVELENGTHS'] = BAND_WAVELENGTHS
    meta['ZPT'] = DEFAULT_ZPT
    
    df = pd.read_csv(
        fn, header=header,
        on_bad_lines='skip',
        sep='\s+'
    )

    df = df.drop(columns=['VARLIST:', 'FIELD', 'FLAG'])
    joint_lc = df.copy()
    
    # filter into ZTF and YSE LCs
    ztf_lc = df[(df.FLT == 'X') | (df.FLT == 'Y')]
    yse_lc = df[~df.index.isin(ztf_lc.index)]
    
    if len(ztf_lc.index) == 0:
        lcs = {'YSE': yse_lc,}
    elif len(yse_lc.index) == 0:
        lcs = {'ZTF': ztf_lc,}
    else:
        lcs = {'ZTF': ztf_lc, 'YSE': yse_lc, 'joint': joint_lc}
        
    sr_lcs = {}

    # first calculate peak of joint light curve
    joint_peak_time = df['MJD'][df['FLUXCAL'].idxmax()]

    meta['PEAK_TIME'] = joint_peak_time
    
    for lc_survey in lcs:
        meta_survey = meta.copy()
        meta_survey['SURVEY'] = lc_survey
        
        lc = lcs[lc_survey]
        lc = lc.dropna(axis=0, how='any')

        t = lc.MJD.to_numpy()
        f = lc.FLUXCAL.to_numpy()
        ferr = lc.FLUXCALERR.to_numpy()
        b = lc.FLT.to_numpy()

        sr_lc = create_lc(
            t, f, ferr, b, meta_survey
        )
        if sr_lc is not None:
            sr_lcs[lc_survey] = sr_lc
            
    return sr_lcs


def generate_raenn_file(test_dir, save_dir):
    """Generate lcs.npz file for raenn training.
    """
    lcs = []
    for i, fn in enumerate(glob.glob(
        os.path.join(test_dir, "*.dat")
        )):
        if i % 100 == 0:
            print(i)
        lc_dict = import_single_yse_lc(fn)
        for k in lc_dict:
            lcs.append(lc_dict[k])
    print(len(lcs))

    save_lcs(lcs, save_dir)


if __name__ == "__main__":
    test_dir = '../../../data/yse_dr1_zenodo_snr_geq_4/'
    #test_dir = '../data/yse_dr1_zenodo_snr_geq_4/'

    save_dir = '../../../data/superraenn/yse/'
    #save_dir = '../data/superraenn/yse/'
    generate_raenn_file(test_dir, save_dir)
