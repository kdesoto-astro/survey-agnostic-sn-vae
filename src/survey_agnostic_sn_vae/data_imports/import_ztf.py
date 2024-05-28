import pandas as pd
import numpy as np
import os

from astropy.coordinates import SkyCoord
from dustmaps.config import config as dustmaps_config
from dustmaps.sfd import SFDQuery
from superphot_plus.sfd import dust_filepath

from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.lightcurve import Lightcurve

from survey_agnostic_sn_vae.preprocessing import save_lcs
from survey_agnostic_sn_vae.data_generation.utils import convert_mag_to_flux
from survey_agnostic_sn_vae.data_imports.import_helper import create_lc

DEFAULT_ZPT = 26.3

LIM_MAGS = {
    'g': 20.8,
    'r': 20.6
}

BAND_WIDTHS = {
    'g':1317.15,
    'r':1488.99
}
# lambda-cen
BAND_WAVELENGTHS = {
    'g': 4805.92,
    'r': 6435.87,
}

def get_band_extinctions(ra, dec):
    """Get g- and r-band extinctions in magnitudes for a single
    supernova lightcurve based on right ascension (RA) and declination
    (DEC).

    Parameters
    ----------
    ra : float
        The right ascension of the object of interest, in degrees.
    dec : float
        The declination of the object of interest, in degrees.
    wvs : list or np.ndarray
        Array of wavelengths, in angstroms.


    Returns
    -------
    ext_dict : Dict
        A dictionary mapping bands to extinction magnitudes for the given coordinates.
    """
    dustmaps_config["data_dir"] = dust_filepath
    sfd = SFDQuery()

    # First look up the amount of mw dust at this location
    coords = SkyCoord(ra, dec, frame="icrs", unit="deg")
      # from https://dustmaps.readthedocs.io/en/latest/examples.html
    return sfd(coords)


def prep_lcs_superraenn(
    dataset_csv,
    probs_csv,
    data_dir,
    save_dir,
):
    """Run equivalent of superraenn-prep on processed light curves."""
    print("STARTS")
    os.makedirs(save_dir, exist_ok=True)

    full_df = pd.read_csv(dataset_csv)
    all_names = full_df.NAME.to_numpy()
    iau_names = full_df.IAU.to_numpy()
    labels = full_df.CLASS.to_numpy()
    redshifts = full_df.Z.to_numpy()

    final_names = pd.read_csv(probs_csv).Name.to_numpy()

    my_lcs = []

    for i, name in enumerate(all_names):
        if i % 500 == 0:
            print(i)

        if name not in final_names:
            continue

        l_canon = SnClass.canonicalize(labels[i])
        """
        lc = Lightcurve.from_file(
            os.path.join(
                data_dir,
                name + ".npz"
            )
        )
        """
        filename = os.path.join(
            data_dir, name + ".csv"
        )
        single_df = pd.read_csv(filename)
        sub_df = single_df[["mjd", "ra", "dec", "fid", "magpsf", "sigmapsf"]]
        pruned_df = sub_df.dropna(subset=["mjd", "fid", "magpsf", "sigmapsf"])
        pruned_df2 = pruned_df.drop(
            pruned_df[pruned_df['fid'] > 2].index
        ) # remove i band
        sorted_df = pruned_df2.sort_values(by=['mjd'])
        sorted_df['bandpass'] = np.where(sorted_df.fid.to_numpy() == 1, 'g', 'r')
        sorted_df = sorted_df.drop(columns=['fid',])

        ra = np.nanmean(sorted_df.ra.to_numpy())
        dec = np.nanmean(sorted_df.dec.to_numpy())

        if np.isnan(ra) or np.isnan(dec):
            return None
        
        mwebv = get_band_extinctions(ra, dec)
        
        m = sorted_df.magpsf.to_numpy()
        merr = sorted_df.sigmapsf.to_numpy()
        b = sorted_df.bandpass.to_numpy()
        t = sorted_df.mjd.to_numpy()
        
        f, ferr = convert_mag_to_flux(m, merr, 26.3)
        
        snr_mask = (f/ferr >= 4.0)
        t = t[snr_mask]
        f = f[snr_mask]
        ferr = ferr[snr_mask]
        b = b[snr_mask]
        
        meta = {
            'MWEBV': mwebv,
            'ZPT': 26.3,
            'REDSHIFT_FINAL': redshifts[i],
            'LIM_MAGS': LIM_MAGS,
            'SURVEY': 'ZTF',
            'SPEC_CLASS': l_canon,
            'PEAK_TIME': t[np.argmax(f)],
            'BAND_WAVELENGTHS': BAND_WAVELENGTHS,
            'BAND_WIDTHS': BAND_WIDTHS
        }
        
        try:
            if np.isnan(iau_names[i]):
                meta['NAME'] = name
                print("NO IAU")

            else:
                meta['NAME'] = iau_names[i]
        except:
            meta['NAME'] = iau_names[i]
            
        sr_lc = create_lc(
            t, f, ferr, b, meta
        )
        if sr_lc is not None:
            my_lcs.append(sr_lc)

    print(len(my_lcs))
    save_lcs(my_lcs, save_dir)
