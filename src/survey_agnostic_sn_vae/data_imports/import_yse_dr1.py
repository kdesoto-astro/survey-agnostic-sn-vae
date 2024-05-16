import os, glob
import pandas as pd
from survey_agnostic_sn_vae.preprocessing import LightCurve, save_lcs
import numpy as np

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
"""
LIM_MAGS = {
    'ZTF': 20.8,
    'YSE': 22.0
} # use most sensitive band
"""
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
    df = pd.read_csv(
        fn, header=header,
        on_bad_lines='skip',
        sep='\s+'
    )
    
    df = df.drop(columns=['VARLIST:', 'FIELD', 'FLAG'])
    
    # filter into ZTF and YSE LCs
    ztf_lc = df[(df.FLT == 'X') | (df.FLT == 'Y')]
    yse_lc = df[~df.index.isin(ztf_lc.index)]
    # add joint lf (just from df)

    lcs = {'ZTF': ztf_lc, 'YSE': yse_lc} # add joint LC
    sr_lcs = {}
    
    # first calculates peak of joint light curve
    for lc_survey in lcs:
        lc = lcs[lc_survey]
        lc = lc.dropna(axis=0, how='any')
        
        t = lc.MJD.to_numpy()
        f = lc.FLUXCAL.to_numpy()
        ferr = lc.FLUXCALERR.to_numpy()
        b = lc.FLT.to_numpy()
        
        if len(t) < 5:
            print("SKIPPED T")
            continue
            
        # variability cut
        if np.mean(ferr) >= np.std(f):
            print("SKIPPED A")
            continue
        if 3 * np.mean(ferr) >= (np.max(f) - np.min(f)):
            print("SKIPPED B")
            continue
        
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
            lim_mag_dict=LIM_MAGS,
            obj_type=meta['SPEC_CLASS']
        )

        sr_lc.group = hash(meta['NAME'])
        sr_lc.get_abs_mags()
        sr_lc.sort_lc()
        pmjd = sr_lc.find_peak() # replace with joint peak
        sr_lc.shift_lc(pmjd)
        sr_lc.correct_time_dilation()
        
        filt_list = np.unique(sr_lc.filters)
        
        if len(filt_list) < 2:
            print("SKIPPED C")
            continue
        
        sr_lc.wavelengths = np.zeros(len(filt_list))
        sr_lc.filt_widths = np.zeros(len(filt_list))

        for j, f in enumerate(filt_list):
            sr_lc.wavelengths[j] = BAND_WAVELENGTHS[f]
            sr_lc.filt_widths[j] = BAND_WIDTHS[f]

        sr_lc.filter_names_to_numbers(filt_list)
        sr_lc.cut_lc()
        try:
            sr_lc.make_dense_LC(len(filt_list))
        except:
            continue
            
        if len(sr_lc.dense_times) < 5:
            print("SKIPPED D")
            continue

        sr_lc.tile()
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
    
    save_lcs(lcs, save_dir)
    
    
if __name__ == "__main__":
    test_dir = '../../../data/yse_dr1_zenodo_snr_geq_4/'
    save_dir = '../../../data/superraenn/yse/'
    generate_raenn_file(test_dir, save_dir)