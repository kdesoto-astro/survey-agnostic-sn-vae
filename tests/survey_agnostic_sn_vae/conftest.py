import pytest

import os
import numpy as np
import pandas as pd
import torch

from survey_agnostic_sn_vae.raenn import VAE
from survey_agnostic_sn_vae.preprocessing import calc_outseq

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