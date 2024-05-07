from survey_agnostic_sn_vae.preprocessing import *


def test_outseq(test_sequence, input_shape) -> None:
    """Test that outsequence is being formatted as
    expected.
    """
    test_lms = np.random.normal(size=test_sequence.shape[0])
    outseq_tiled = calc_outseq(test_sequence, test_lms)
    
    assert outseq_tiled.shape == (5, *input_shape)
    
    # we expect band wavelengths to be repeated every 6th value (a,b,c,d,e,f,a,b,...)
    assert np.all(outseq_tiled[:,0,2] == outseq_tiled[:,6,2])
    assert np.all(outseq_tiled[:,7,2] == outseq_tiled[:,13,2])
    assert np.all(outseq_tiled[:,0,2] != outseq_tiled[:,1,2])
    assert np.all(outseq_tiled[:,0,3] == outseq_tiled[:,6,3])
    assert np.all(outseq_tiled[:,7,3] == outseq_tiled[:,13,3])
    assert np.all(outseq_tiled[:,0,3] != outseq_tiled[:,1,3])
    
    # we expect time steps to be same in groups of 6
    assert np.all(np.diff(outseq_tiled[:,:6,0]) == 0.0)
    assert np.all(outseq_tiled[:,5,0] != outseq_tiled[:,6,0])