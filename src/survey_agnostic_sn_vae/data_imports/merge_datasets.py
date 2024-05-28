import numpy as np
import matplotlib.pyplot as plt
import os

from survey_agnostic_sn_vae.preprocessing import LightCurve
from survey_agnostic_sn_vae.data_imports.import_yse_dr1 import save_lcs
from survey_agnostic_sn_vae.data_imports.import_helper import plot_lc
    
def check_overlap(lc1, lc2):
    """Check whether two files reference the same
    object + survey
    """
    return (lc1.name == lc2.name) & (
        lc1.survey == lc2.survey
    )

def plot_overlap(lc_fn1, lc_fn2, save_dir):
    """Plot overlapping LCs to check self-consistency.
    """
    lcs1 = np.load(lc_fn1, allow_pickle=True)['lcs']
    lcs2 = np.load(lc_fn2, allow_pickle=True)['lcs']
    
    for lc1 in lcs1: # O(n^2) runtime :( 
        for lc2 in lcs2:
            if not check_overlap(lc1, lc2):
                continue
            
            # align the 2 LCs
            lc2.times -= lc2.times[np.argmin(lc2.abs_mags)]
            lc1.times -= lc1.times[np.argmin(lc1.abs_mags)]
            fig, ax = plt.subplots()
            plot_lc(ax, lc1, alpha=0.5)
            plot_lc(ax, lc2, alpha=0.2)
            ax.invert_yaxis()
            ax.set_title(f'{lc1.name}: {lc1.mwebv} vs. {lc2.mwebv}')
            ax.set_xlabel("MJD")
            ax.set_ylabel("Absolute Magnitude")
            plt.savefig(
                os.path.join(
                    save_dir, f'{lc1.name}.pdf'
                )
            )
            plt.close()
            
            
def merge_two_lcs(lc_fn1, lc_fn2, save_fn, batchsize=128):
    """Merge two LC sets, defaulting to the LC in
    lc_fn1 when there is overlap.
    
    The batchsize is to ensure there's matching pairs in each
    batch (and not just a sequence of ZTF-only LCs).
    """
    lcs1 = np.load(lc_fn1, allow_pickle=True)['lcs'].tolist()
    lcs2 = np.load(lc_fn2, allow_pickle=True)['lcs'].tolist()
    
    lcs = lcs1.copy()
    
    lcs_other = []
    for lc2 in lcs2:
        repeat = False
        for lc1 in lcs1:
            if check_overlap(lc1, lc2):
                repeat = True
                break
        if not repeat:
            lcs_other.append(lc2)
    
    lcs_combined = []
    curr_idx1 = 0
    curr_idx2 = 0
    N = len(lcs)
    
    b = round(len(lcs) * batchsize / (len(lcs) + len(lcs_other)))
    b2 = round(len(lcs_other) * batchsize / (len(lcs) + len(lcs_other)))
    while True:
        if curr_idx1+b >= N:
            lcs_combined.extend(lcs[curr_idx1:])
            lcs_combined.extend(lcs_other[curr_idx2:])
            break
        else:
            lcs_combined.extend(lcs[curr_idx1:curr_idx1+b])
            lcs_combined.extend(lcs_other[curr_idx2:curr_idx2+b2])
        curr_idx1 += b
        curr_idx2 += b2
    
    print(len(lcs1), len(lcs2), len(lcs_combined))

    np.savez(save_fn, lcs=lcs_combined)
   
        
            