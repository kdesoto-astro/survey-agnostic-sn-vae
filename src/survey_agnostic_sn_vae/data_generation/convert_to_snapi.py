# Convert synthetic transients to SNAPI format
import os

import astropy.units as u
from astropy.time import Time
from snapi import Transient as SNAPITransient
from snapi import LightCurve, Filter, Photometry

from survey_agnostic_sn_vae.data_generation.objects import Transient

def convert_transient_to_snapi(transient_fn, save_dir):
    """Convert transient file to SNAPI format.
    (Apparently they're already almost the same)
    """
    orig_transient = Transient.load(transient_fn)
    snapi_transient = SNAPITransient(iid=orig_transient.obj_id)

    snapi_lcs = set()
    for lc in orig_transient.lightcurves:
        for b in lc.bands:
            if len(lc.times[b]) < 5:
                continue
            filt = Filter(
                instrument=lc.survey.name,
                band=str(b),
                center=lc.survey.band_wavelengths[b] * u.AA,
                width=lc.survey.band_widths[b] * u.AA
            )
            snapi_lc = LightCurve(
                times=Time(lc.times[b], format='mjd', scale='utc'),
                mags=lc.mag[b],
                mag_errs=lc.mag_err[b],
                filt=filt
            )
            snapi_abs_lc = snapi_lc.absolute(0.02) # WARNING: HARD CODED
            snapi_lcs.add(snapi_abs_lc)

    if len(snapi_lcs) < 2:
        return

    phot = Photometry(snapi_lcs)
    phot.phase()
    snapi_transient.photometry = phot
    snapi_transient.redshift = 0.02
    snapi_transient.mwebv = 0.0
    snapi_transient.save(
        os.path.join(save_dir, snapi_transient.id + ".hdf5")
    )
    return
