import os
import glob

import numpy as np
from astropy.timeseries import TimeSeries
from astropy.time import Time
import astropy.units as u
import pandas as pd

from snapi import Filter, LightCurve, Photometry, Transient

#import matplotlib.pyplot as plt

DEFAULT_ZPT = 27.5
# X for ZTF-*g*; Y for ZTF-*r*
# no y-band for Panstarrs

BAND_WIDTHS = {
    'g':1148.66,
    'r':1397.73,
    'i':1292.39,
    'z':1038.82,
    'X':1317.15,
    'Y':1553.43,
}
# lambda-eff
BAND_WAVELENGTHS = {
    'g': 4810.16,
    'r': 6155.47,
    'i': 7503.03,
    'z': 8668.36,
    'X': 4746.48,
    'Y': 6366.38,
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
                return i + 6, meta

def import_single_yse_lc(fn):
    """Imports single YSE SNANA file.
    """
    header, meta = find_meta_start_of_data(fn)

    lcs = []

    df = pd.read_csv(fn, skiprows=header, delim_whitespace=True, skipfooter=1)
    df = df.drop(columns=['VARLIST:', 'FIELD', 'FLAG'])
    t = Time(df['MJD'], format='mjd').datetime
    df.set_index(pd.DatetimeIndex(t, name='time'), inplace=True)
    df['flux'] = df['FLUXCAL']
    df['flux_unc'] = df['FLUXCALERR']
    df['zpt'] = DEFAULT_ZPT
    df = df[['flux', 'flux_unc', 'zpt', 'FLT']]

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
            center=BAND_WAVELENGTHS[b] * u.AA,
            width=BAND_WIDTHS[b] * u.AA,
        )
        lc = LightCurve(
            ts, filt=filt
        )
        lc.calibrate_fluxes(23.90)
        # absolute magnitudes
        abs_lc = lc.absolute(float(meta['REDSHIFT_FINAL']))
        # correct for extinction
        lcs.append(abs_lc)
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
            if transient.coordinates is not None: # import_ztf.py generated file
                photometry = transient.photometry.filter_by_instrument("ZTF")
                new_lcs = set()
                for lc in lcs:
                    new_lc = lc.correct_extinction(coordinates=transient.coordinates) # SO IT ALIGNS
                    photometry.add_lightcurve(new_lc)
                merged_transient = Transient(
                    iid=prefix,
                    ra=transient.coordinates.ra,
                    dec=transient.coordinates.dec,
                    redshift=transient.redshift,
                    spec_class=transient.spec_class,
                    internal_names=transient.internal_names,
                    photometry=photometry
                )
                merged_transient.mwebv = float(meta['MWEBV'])
                merged_transient.save(os.path.join(save_dir, prefix+".hdf5"))
                continue

        new_lcs = set()
        for lc in lcs:
            new_lcs.add(
                lc.correct_extinction(mwebv=float(meta['MWEBV']))
            )
        transient = Transient(
            iid=prefix,
            redshift=meta['REDSHIFT_FINAL'],
            spec_class=meta['SPEC_CLASS'], # TODO: broad or not? canonicalize?
            photometry=Photometry(lcs),
        )
        transient.mwebv = float(meta['MWEBV'])
        transient.save(os.path.join(save_dir, prefix+".hdf5"))


if __name__ == "__main__":
    TEST_DIR = '../../../data/yse_dr1_zenodo_snr_geq_4/'
    SAVE_DIR = '../../../data/superraenn/yse/'
    generate_yse_transients(TEST_DIR, SAVE_DIR)
