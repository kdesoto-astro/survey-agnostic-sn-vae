import pytest

import os
import numpy as np
import pandas as pd
import torch

from survey_agnostic_sn_vae.raenn import VAE
from survey_agnostic_sn_vae.preprocessing import (
    calc_outseq,
    LightCurve
)

TEST_DIR = os.path.dirname(__file__)

DATA_DIR_NAME = "data"

# pylint: disable=missing-function-docstring, redefined-outer-name

@pytest.fixture
def test_data_dir():
    return os.path.join(TEST_DIR, DATA_DIR_NAME)

@pytest.fixture
def input_shape():
    nfilts = 6
    return (7, 4*nfilts+2)

@pytest.fixture
def test_vae(input_shape):
    model = VAE(
        (1, input_shape[0], input_shape[1]-1), # first dimension has no impact
        hidden_dim=8,
        latent_dim=2,
        device='cpu'
    )
    model.eval()
    return model


@pytest.fixture
def test_sequence(input_shape):
    """Test sequence for outseq formatting.
    """
    nfilts = 6
    seq = np.random.normal(size=(5,*input_shape)) # 5 samples, 7 timesteps, 6 bands
    # set same wavelengths/groupid for each timestamp
    seq[:,:,2*nfilts+1:] = seq[:,0:1,2*nfilts+1:]
    return seq.astype(np.float32)

@pytest.fixture
def test_inputs(test_sequence):
    input1 = torch.from_numpy(test_sequence[:,:,:-1])
    test_lms = np.random.normal(size=(test_sequence.shape[0], 6))
    input2 = torch.from_numpy(
        calc_outseq(test_sequence, test_lms)
    )
    input3 = test_sequence[:,:,-1]
    return input1, input2, input3

@pytest.fixture
def test_twoband_lightcurve():
    """Test 2-band Lightcurve object.
    """
    test_t = np.random.uniform(size=10)
    test_f = np.random.uniform(size=10) * 1000 + 10000
    test_ferr = np.random.uniform(size=10) * 100 + 100
    test_b = ["g",] * 5
    test_b.extend(["r",] * 5)
    test_b = np.array(test_b)
    
    lc = LightCurve(
        name='test',
        times=test_t,
        fluxes=test_f,
        flux_errs=test_ferr,
        filters=test_b,
        zpt=26.3,
        redshift=0.1,
        lim_mag_dict={'g': 20.6, 'r': 20.8}
    )
    lc.get_abs_mags()
    lc.filter_names_to_numbers(['g', 'r'])

    lc.make_dense_LC(2)
    lc.wavelengths = [1000., 2300.]
    lc.filt_widths = [200., 300.]
    return lc
        