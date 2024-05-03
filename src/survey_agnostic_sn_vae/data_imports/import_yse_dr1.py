import pandas as pd
from survey_agnostic_sn_vae.preprocessing import LightCurve
import numpy as np

DEFAULT_ZPT = 27.5
# X for ZTF-*g*; Y for ZTF-*r*
# no y-band for Panstarrs
"""
LIM_MAGS = {
    'g': 22.0,
    'r': 21.8,
    'i': 21.5,
    'z': 20.9,
    'X': 20.8,
    'Y': 20.6
}
"""
# TODO: change limmags in raenn to be per-datapoint

LIM_MAGS = {
    'ZTF': 20.8,
    'YSE': 22.0
} # use most sensitive band

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
    print(meta)
    df = pd.read_csv(
        fn, header=header,
        on_bad_lines='skip',
        sep='\s+'
    )
    
    df = df.drop(columns=['VARLIST:', 'FIELD', 'FLAG'])
    
    # filter into ZTF and YSE LCs
    ztf_lc = df[(df.FLT == 'X') | (df.FLT == 'Y')]
    yse_lc = df[~df.index.isin(ztf_lc.index)]

    lcs = {'ZTF': ztf_lc, 'YSE': yse_lc}
    sr_lcs = {}
    
    for lc_survey in lcs:
        print(lc_survey)
        lc = lcs[lc_survey]
        lc = lc.dropna(axis=0, how='any')
        
        t = lc.MJD.to_numpy()
        f = lc.FLUXCAL.to_numpy()
        ferr = lc.FLUXCALERR.to_numpy()
        b = lc.FLT.to_numpy()
        
        sr_lc = LightCurve(
            name=meta['NAME'],
            times=t,
            fluxes=f,
            flux_errs=ferr,
            filters=b,
        )
        sr_lc.add_LC_info(
            zpt=DEFAULT_ZPT,
            mwebv=float(meta['MWEBV']),
            redshift=float(meta['REDSHIFT_FINAL']),
            lim_mag=LIM_MAGS[lc_survey],
            obj_type=meta['SPEC_CLASS']
        )
        sr_lc.group = hash(meta['NAME'])
        sr_lc.get_abs_mags()
        sr_lc.sort_lc()
        pmjd = sr_lc.find_peak(
            t[np.argmax(f)]
        )
        sr_lc.shift_lc(pmjd)
        sr_lc.correct_time_dilation()
        
        filt_dict = {f: i for i, f in enumerate(np.unique(b))}

        sr_lc.wavelengths = np.zeros(len(np.unique(b)))
        sr_lc.filt_widths = np.zeros(len(np.unique(b)))

        for f in filt_dict:
            sr_lc.wavelengths[filt_dict[f]] = BAND_WAVELENGTHS[f]
            sr_lc.filt_widths[filt_dict[f]] = BAND_WIDTHS[f]

        sr_lc.filter_names_to_numbers(filt_dict)
        sr_lc.cut_lc()
        try:
            sr_lc.make_dense_LC(len(np.unique(b)))
        except:
            print("SKIPPED")
            continue

        # pad dense LC to six bands
        if len(np.unique(b)) <= 3:
            tile_factor = int(6 / len(np.unique(b)))
            sr_lc.dense_lc = np.repeat(
                sr_lc.dense_lc, tile_factor,
                axis=1
            )
            sr_lc.wavelengths = np.repeat(
                sr_lc.wavelengths, tile_factor,
            )
            sr_lc.filt_widths = np.repeat(
                sr_lc.filt_widths, tile_factor,
            )

        elif len(np.unique(b)) < 6:
            sr_lc.dense_lc = np.repeat(
                sr_lc.dense_lc, 2,
                axis=1
            )[:,:6,:]
            sr_lc.wavelengths = np.repeat(
                sr_lc.wavelengths, 2
            )[:6]
            sr_lc.filt_widths = np.repeat(
                sr_lc.filt_widths, 2
            )[:6]

        sr_lcs[lc_survey] = sr_lc
        
    return sr_lcs
    
    
    
if __name__ == "__main__":
    test_fn = '../../../data/yse_dr1_zenodo_snr_geq_4/2019pmd.snana.dat'
    import_single_yse_lc(test_fn)