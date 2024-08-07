import os
import glob

import numpy as np
from astropy.timeseries import TimeSeries
from astropy.time import Time
import pandas as pd

from snapi import Filter, LightCurve, Photometry, Transient

from survey_agnostic_sn_vae.preprocessing import save_lcs
from survey_agnostic_sn_vae.data_imports.import_helper import create_lc
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
    with open(fn, 'r', encoding='utf-8') as f:
        for i, row in enumerate(f):
            for k in meta_keys:
                if k+":" in row:
                    meta[k] = row.split(":")[1].split("+")[0].strip()
            if row.strip() == '':
                return i + 5, meta

def import_single_yse_lc(fn):
    """Imports single YSE SNANA file.
    """
    header, meta = find_meta_start_of_data(fn)

    lcs = []

    df = pd.read_csv(fn, skiprows=header, delim_whitespace=True)
    df = df.drop(columns=['VARLIST:', 'FIELD', 'FLAG'])
    t = Time(df['MJD'], format='mjd').datetime
    df.set_index(pd.DatetimeIndex(t, name='time'), inplace=True)
    df['fluxes'] = df['FLUXCAL']
    df['flux_errs'] = df['FLUXCALERR']
    df['zpts'] = DEFAULT_ZPT
    df = df[['fluxes', 'flux_errs', 'zpts', 'FLT']]

    # convert to astropy Timeseries
    for b in np.unique(df['FLT']):
        if b == 'X':
            b_true = 'g'
        elif b == 'Y':
            b_true = 'r'
        else:
            b_true = b
        df_b = df[df['FLT'] == b]
        df_b = df_b.drop(columns=['FLT'])
        ts = TimeSeries.from_pandas(df_b)
        filt = Filter(
            band=b_true,
            instrument = 'ZTF' if b in ['X', 'Y'] else 'YSE',
            center=BAND_WAVELENGTHS[b],
            width=BAND_WIDTHS[b],
        )
        lc = LightCurve(
            ts, filt=filt
        )
        lcs.append(lc)
    return lcs, meta


def generate_yse_transients(test_dir, save_dir):
    """
    Generate SNAPI Transient objects from YSE data.
    If ZTF data is already present, merge the two.
    """
    for i, fn in enumerate(
        glob.glob(
            os.path.join(test_dir, "*.dat")
        )
    ):
        if i % 100 == 0:
            print(i)
        lcs, meta = import_single_yse_lc(fn)

        # check if file already exists (aka from ZTF)
        prefix = os.path.basename(fn).split(".")[0]
        if os.path.exists(os.path.join(save_dir, prefix+".hdf5")):
            transient = Transient.load(os.path.join(save_dir, prefix+".hdf5"))
            photometry = transient.photometry
            if photometry is not None:
                for lc in lcs:
                    photometry.add_lightcurve(lc)
            else:
                photometry = Photometry(lcs)
            merged_transient = Transient(
                iid=prefix,
                ra=transient.ra,
                dec=transient.dec,
                redshift=transient.redshift,
                spec_class=transient.spec_class,
                internal_names=transient.internal_names,
                photometry=photometry
            )
            merged_transient.save(os.path.join(save_dir, prefix+".hdf5"))
        else:
            transient = Transient(
                iid=prefix,
                redshift=meta['REDSHIFT_FINAL'],
                spec_class=meta['SPEC_CLASS'], # TODO: broad or not? canonicalize?
                photometry=Photometry(lcs) # TODO: add MWEBV
            )
            transient.save(os.path.join(save_dir, prefix+".hdf5"))


if __name__ == "__main__":
    TEST_DIR = '../../../data/yse_dr1_zenodo_snr_geq_4/'
    SAVE_DIR = '../../../data/superraenn/yse/'
    generate_yse_transients(TEST_DIR, SAVE_DIR)
