import pandas as pd
import numpy as np
import os

from astropy.coordinates import SkyCoord
from dustmaps.config import config as dustmaps_config
from dustmaps.sfd import SFDQuery
from superphot_plus.sfd import dust_filepath

from superphot_plus.supernova_class import SupernovaClass as SnClass
from snapi.query_agents import TNSQueryAgent, ALeRCEQueryAgent, ANTARESQueryAgent
from snapi.transient import Transient

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


def generate_ztf_transients(
    save_dir,
):
    """Generate ZTF transients for the VAE."""
    print("STARTS")
    os.makedirs(save_dir, exist_ok=True)

    tns_agent = TNSQueryAgent()
    alerce_agent = ALeRCEQueryAgent()
    antares_agent = ANTARESQueryAgent()

    all_names = tns_agent.retrieve_all_names() # only spectroscopically classified

    for i, n in enumerate(all_names):
        if i % 1000 == 0:
            print(f"Processed {i} of {len(all_names)} objects.")
        # if file exists already, just skip
        #if os.path.exists(os.path.join(save_dir, n+".hdf5")):
        #    print(f"SKIPPED {n}: File already exists")
        #    continue
        transient = Transient(iid=n)
        qr_tns, success = tns_agent.query_transient(transient, local=True) # we dont want spectra
        if not success:
            print("SKIPPED {n}: TNS query failed")
            continue
        for result in qr_tns:
            transient.ingest_query_info(result.to_dict())
        if transient.spec_class not in ["SN II", "SN IIL", "SN IIP"]:
            continue
        qr_alerce, success = alerce_agent.query_transient(transient)
        if not success:
            print(f"SKIPPED {n}: ALeRCE query failed")
            continue
        for result in qr_alerce:
            transient.ingest_query_info(result.to_dict())

        qr_antares, success = antares_agent.query_transient(transient)
        if not success:
            print(f"SKIPPED {n}: ANTARES query failed")
            continue
        for result in qr_antares:
            transient.ingest_query_info(result.to_dict())

        # quality cuts
        photometry = transient.photometry.filter_by_instrument("ZTF")
        try:
            r_lc = [lc for lc in photometry.light_curves if str(lc.filter) == "ZTF_r"][0]
            g_lc = [lc for lc in photometry.light_curves if str(lc.filter) == "ZTF_g"][0]
        except IndexError:
            print(f"SKIPPED {n}: Missing a band")
            continue
        
        good_quality = True
        for lc in [r_lc, g_lc]:
            high_snr_detections = lc.detections[lc.detections['mag_unc'] <= (5 / 8. / np.log(10))] # SNR >= 4

            # number of high-SNR detections cut
            if len(high_snr_detections) < 5:
                print(f"SKIPPED {n}: Not enough high-SNR detections")
                good_quality = False
                break

            # variability cut
            if (
                np.max(high_snr_detections['mag']) - np.min(high_snr_detections['mag'])
             ) < 3 * np.mean(high_snr_detections['mag_unc']):
                print(f"SKIPPED {n}: Amplitude too small")
                good_quality = False
                break

            # second variability cut
            if np.std(high_snr_detections['mag']) < np.mean(high_snr_detections['mag_unc']):
                print(f"SKIPPED {n}: Variability too small")
                good_quality = False
                break

        if not good_quality:
            continue

        transient.spec_class = SnClass.canonicalize(transient.spec_class)
        transient.save(os.path.join(save_dir, n+".hdf5"))
        
        # TODO: convert band names to ints
        # TODO: mwebv = get_band_extinctions(ra, dec)
        # TODO: add lim_mags to either Filter or LightCurve/Photometry