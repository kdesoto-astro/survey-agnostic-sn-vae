import numpy as np
import pandas as pd

from survey_agnostic_sn_vae.preprocessing import LightCurve, save_lcs


BAND_COLORS = {
    'g': 'green',
    'r': 'orange',
    'i': 'pink',
    'z': 'purple',
    'X': 'blue',
    'Y': 'red',
}
    
def create_lc(
    t, f, ferr, b, meta
):
    mask_nans = np.isnan(t) | np.isnan(f)
    mask_nans = mask_nans | np.isnan(ferr)
    mask_nans = mask_nans | ~np.isin(b, list(BAND_COLORS.keys()))
    
    t = t[~mask_nans]
    f = f[~mask_nans]
    ferr = ferr[~mask_nans]
    b = b[~mask_nans]
    
    keep_mask = np.ones(len(t)).astype(bool)
    
    for b_uq in np.unique(b):
        b_mask = b == b_uq
        # variability cut
        if np.mean(ferr[b_mask]) >= np.std(f[b_mask]):
            keep_mask[b_mask] = False

        if 3 * np.mean(ferr[b_mask]) >= (np.max(f[b_mask]) - np.min(f[b_mask])):
            keep_mask[b_mask] = False
        
    t = t[keep_mask]
    f = f[keep_mask]
    ferr = ferr[keep_mask]
    b = b[keep_mask]
    
        
    if len(t) < 5:
        return None
    if len(np.unique(b)) < 2:
        return None

    sr_lc = LightCurve(
        name=meta['NAME'],
        survey=meta['SURVEY'],
        times=t,
        fluxes=f,
        flux_errs=ferr,
        filters=b,
    )
    sr_lc.add_LC_info(
        zpt=meta['ZPT'],
        mwebv=float(meta['MWEBV']),
        redshift=float(meta['REDSHIFT_FINAL']),
        lim_mag_dict=meta['LIM_MAGS'],
        obj_type=meta['SPEC_CLASS']
    )

    sr_lc.group = hash(meta['NAME'])
    sr_lc.get_abs_mags(replace_nondetections=False)
    sr_lc.sort_lc()
    pmjd = meta['PEAK_TIME']
    sr_lc.shift_lc(pmjd)
    sr_lc.correct_time_dilation()

    filt_list = np.unique(sr_lc.filters)

    if len(filt_list) < 2:
        return None

    sr_lc.wavelengths = np.zeros(len(filt_list))
    sr_lc.filt_widths = np.zeros(len(filt_list))

    for j, f in enumerate(filt_list):
        sr_lc.wavelengths[j] = meta['BAND_WAVELENGTHS'][f]
        sr_lc.filt_widths[j] = meta['BAND_WIDTHS'][f]

    sr_lc.filter_names_to_numbers(filt_list)
    wvs = np.array([meta['BAND_WAVELENGTHS'][f] for f in filt_list])
    sr_lc.correct_extinction(wvs)
    sr_lc.cut_lc()
    try:
        sr_lc.make_dense_LC(len(filt_list))
    except:
        return None

    if len(sr_lc.dense_times) < 5:
        return None
    
    sr_lc.tile()
    return sr_lc