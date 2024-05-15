import numpy as np

from survey_agnostic_sn_vae.preprocessing import calc_outseq
from survey_agnostic_sn_vae.raenn import VAE


def test_two_band_tiling(test_twoband_lightcurve) -> None:
    """Ensures when there are two bands, that:
    (1) the bands are repeated 3 times in succession
    (2) the associated sequence and outseq rows are perfectly replicated
    (3) the decodings for said repeated vectors are identical
    """
    # first, tile lc and check tiling
    test_twoband_lightcurve.tile()
    
    assert test_twoband_lightcurve.dense_lc.shape[1] == 6
    
    assert np.all(
        np.diff(test_twoband_lightcurve.dense_lc[:,:3], axis=1) == 0.0
    )
    
    assert np.all(
        np.diff(test_twoband_lightcurve.dense_lc[:,3:], axis=1) == 0.0
    )
    
    nfilts = 6
    nfiltsp1 = nfilts+1
    nfiltsp2 = 2*nfilts+1
    nfiltsp3 = 3*nfilts+1
    
    N = len(test_twoband_lightcurve.dense_times)
    
    sequence = np.zeros((1, N, 4*nfilts+2))
    # next, convert into format for VAE evaluation
    sequence[0, 0:N, 0] = test_twoband_lightcurve.dense_times / 1000. # to keep values small
    sequence[0, 0:N, 1:nfiltsp1] = test_twoband_lightcurve.dense_lc[:, :, 0] # fluxes
    sequence[0, 0:N, nfiltsp1:nfiltsp2] = test_twoband_lightcurve.dense_lc[:, :, 1] # flux errors
    sequence[0, N:, 0] = (np.max(test_twoband_lightcurve.times)+200.) / 1000.
    sequence[0, N:, 1:nfiltsp1] = test_twoband_lightcurve.ordered_abs_lim_mags[np.newaxis,:]
    sequence[0, N:, nfiltsp1:nfiltsp2] = 1.0
    sequence[0, :, nfiltsp2:nfiltsp3] = test_twoband_lightcurve.wavelengths
    sequence[0, :, nfiltsp3:-1] = test_twoband_lightcurve.filt_widths
    
    assert np.all(np.diff(sequence[0,:,1:4], axis=1) == 0.0)
    assert np.all(np.diff(sequence[0,:,nfiltsp1:nfiltsp1+3], axis=1) == 0.0)
    assert np.all(np.diff(sequence[0,:,nfiltsp2:nfiltsp2+3], axis=1) == 0.0)
    assert np.all(np.diff(sequence[0,:,nfiltsp3:nfiltsp3+3], axis=1) == 0.0)

    assert np.all(np.diff(sequence[0,:,4:7], axis=1) == 0.0)
    assert np.all(np.diff(sequence[0,:,nfiltsp1+3:nfiltsp1+6], axis=1) == 0.0)
    assert np.all(np.diff(sequence[0,:,nfiltsp2+3:nfiltsp2+6], axis=1) == 0.0)
    assert np.all(np.diff(sequence[0,:,nfiltsp3+3:nfiltsp3+6], axis=1) == 0.0)
    
    # check outseq self-consistency
    lms = np.array([test_twoband_lightcurve.ordered_abs_lim_mags,])
    outseq_tiled = calc_outseq(sequence, lms)
    
    assert np.all(np.diff(outseq_tiled[:,0:3,:], axis=1) == 0.0)
    assert np.all(np.diff(outseq_tiled[:,3:6,:], axis=1) == 0.0)
    assert np.all(np.diff(outseq_tiled[:,6:9,:], axis=1) == 0.0)
    assert np.all(np.diff(outseq_tiled[:,9:12,:], axis=1) == 0.0)
    
    # finally, check if vae decodings are identical
    
    model = VAE(
        sequence[:,:,:-1].shape, # first dimension has no impact
        hidden_dim=8,
        latent_dim=2,
        device='cpu'
    )
    model.eval()
    return model

    x_hat, z, mean, log_var = model.forward(sequence[:,:,:-1], outseq_tiled)
    
    assert np.all(np.diff(x_hat[:,:,:3], axis=2) == 0.0)
    assert np.all(np.diff(x_hat[:,:,3:], axis=2) == 0.0)


    
    
    
    
    
    
    