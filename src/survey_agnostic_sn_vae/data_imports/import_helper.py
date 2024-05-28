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

def plot_lc(ax, lc, **kwargs):
    """Plot single LC.
    """
    for i, b in enumerate(lc.filt_list):
        ax.errorbar(
            lc.times[lc.filters == i], lc.abs_mags[lc.filters == i],
            yerr=lc.abs_mags_err[lc.filters == i], fmt='o', 
            color=BAND_COLORS[b],
            **kwargs
        )
    
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
    
    if meta['NAME'] == '2021rh':
        print(meta['SURVEY'], np.mean(ferr[b == 'X']), np.std(f[b == 'X']))
        
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

    """
    # check for variability in dense LCs
    dense_fluxes = sr_lc.dense_lc[:,:,0]
    dense_fluxerrs = sr_lc.dense_lc[:,:,1]
    variability_mask = (np.std(dense_fluxes, axis=0) > np.mean(dense_fluxerrs, axis=0)) # true means keep

    if sr_lc.name in ['2021rh', '2020advk']:
        print(sr_lc.name, sr_lc.survey, variability_mask)
        
    # just throw out entire LC if none of the bands are good
    if not np.any(variability_mask):
        return None

    # otherwise, keep the good bands
    sr_lc.dense_lc = sr_lc.dense_lc[:,variability_mask,:]
    sr_lc.ordered_abs_lim_mags = sr_lc.ordered_abs_lim_mags[variability_mask]
    sr_lc.wavelengths = sr_lc.wavelengths[variability_mask]
    sr_lc.filt_widths = sr_lc.filt_widths[variability_mask]
    sr_lc.masked_vals = sr_lc.masked_vals[:,variability_mask]
    """
    
    sr_lc.tile()
    return sr_lc