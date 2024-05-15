import pandas as pd
import numpy as np

from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.lightcurve import Lightcurve

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
        l_oneword = l_canon.replace(" ", "")
            
        lc = Lightcurve.from_file(
            os.path.join(
                data_dir,
                name + ".npz"
            )
        )

        sr_lc = LightCurve(
            name,
            lc.times[lc.bands != 'i'],
            lc.fluxes[lc.bands != 'i'],
            lc.flux_errors[lc.bands != 'i'],
            lc.bands[lc.bands != 'i']
        )
        
        sr_lc.add_LC_info(
            zpt=26.3,
            redshift=redshifts[i],
            lim_mag_dict={'g': 20.6, 'r': 20.8},
            obj_type=l_canon
        )

        sr_lc.group = hash(name)
        sr_lc.get_abs_mags()
        sr_lc.sort_lc()
        pmjd = sr_lc.find_peak()
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
        my_lcs.append(sr_lc)
        
    save_lcs(my_lcs, save_dir)