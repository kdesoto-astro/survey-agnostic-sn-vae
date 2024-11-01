import os
import numpy as np
from snapi import Transient, Photometry

def single_phot_augment(fn):
    save_dir = "snapi_transients_augmented"
    n_max = 64
    rng = np.random.default_rng()

    transient = Transient.load(fn)
    
    aug_set0 = []
    aug_set1 = []
    aug_set2 = []
    aug_set3 = []

    rand_mint = rng.uniform(low=np.min(transient.photometry.times), high=np.max(transient.photometry.times))
    rand_maxt = rng.uniform(low=rand_mint, high=np.max(transient.photometry.times))

    # add phase wiggles of sigma = 3 days (to account for cadence)
    phase_wiggles = rng.normal(loc=0., scale=3., size=3)

    for lc in transient.photometry.light_curves:
        # first, resample 3 times
        resampled_mags = lc.resample(mags=True, num=3)
        
        aug0 = lc.copy()
        if len(aug0) > n_max:
            aug0.subsample(n_max)
        aug_set0.append(aug0)

        # augment 1: just resample at existing times
        aug = lc.copy()
        aug.mags = resampled_mags[0]
        if len(aug) > n_max:
            aug.subsample(n_max)
        aug_set1.append(aug)

        # augment 2: resample, then subsample LC
        aug1 = lc.copy()
        aug1.mags = resampled_mags[1]
        if len(aug1.times) > 5:
            subsample_num = rng.integers(low=5, high=min(len(aug1.times), n_max))
            aug1.subsample(subsample_num)
        aug_set2.append(aug1)

        # augment 3: resample, truncate, then subsample
        aug2 = lc.copy()
        aug2.mags = resampled_mags[2]
        aug2.truncate(max_t=rand_maxt, min_t=rand_mint)
        if len(aug2.times) < 5:
            continue
        if len(aug2.times) > 5:
            subsample_num2 = rng.integers(low=5, high=min(len(aug2.times), n_max))
            aug2.subsample(subsample_num2)
        aug_set3.append(aug2)

    for i, aug_lcs in enumerate([aug_set0, aug_set1, aug_set2, aug_set3]):
        if len(aug_lcs) < 2:
            continue
        augmented_photometry = Photometry(aug_lcs)
        augmented_photometry.phase() # re-phase
        if i > 0:
            augmented_photometry.phase(phase_wiggles[i-1]) # phase wiggle
        transient.photometry = augmented_photometry
        transient.save(
            os.path.join(save_dir, f"{transient.id}_{i}.h5")
        )
