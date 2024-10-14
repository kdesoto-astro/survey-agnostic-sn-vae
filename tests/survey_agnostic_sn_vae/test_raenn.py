from survey_agnostic_sn_vae.archival.raenn import *

def test_contrastive_loss() -> None:
    """Test contrastive loss is working as expected.
    """
    test_samples = torch.from_numpy(
        np.random.normal(size=(10,5))
    ) # 10 events, 5 dimensions
    test_ids = np.random.normal(size=5)
    test_ids = torch.from_numpy(np.repeat(test_ids, 2))
    cl = contrastive_loss(test_samples, test_ids)
    assert cl > 0.0
    
    # test that perfect contrast --> smallest loss
    test_samples[1::2] = test_samples[::2]
    cl2 = contrastive_loss(test_samples, test_ids)
    assert cl2 < cl
    
    
def test_merge(test_vae, test_inputs) -> None:
    """Test that merging of decoder inputs
    is happening as expected.
    """
    input1, input2, _ = test_inputs
    z, _ = test_vae.encode(input1)
    merged = test_vae.merge(z, input2).detach().numpy()
    
    assert merged.shape == (5,42,6)
    
    # check encodings same for every timestamp
    assert np.all(np.diff(merged[:,:,:2], axis=1) == 0.0)

    # we expect band wavelengths/lms to be repeated every 6th value (a,b,c,d,e,f,a,b,...)
    assert np.all(merged[:,0,3:6] == merged[:,6,3:6])
    assert np.all(merged[:,7,3:6] == merged[:,13,3:6])
    assert np.all(merged[:,0,3:6] != merged[:,1,3:6])
    
    # we expect time steps to be same in groups of 6
    assert np.all(np.diff(merged[:,:6,2]) == 0.0)
    assert np.all(merged[:,5,2] != merged[:,6,2])
    
    
def test_decoding(test_vae, test_inputs) -> None:
    """Test TimeDistributed layers are working as intended.
    """
    input1, input2, _ = test_inputs
    z, _ = test_vae.encode(input1)
    merged = test_vae.merge(z, input2)
    
    decodings = test_vae.decode(merged.float())
    assert np.all(np.diff(decodings.detach(), axis=0) != 0.0)
    
    # because decodings are TimeDistributed, we expect
    # decodings of rows of repeated columns to be identical
    merged[:, :, :] = merged[:, 0:1, :]
    assert np.all(merged[:,0].detach().numpy() == merged[:,1].detach().numpy())
    decodings_same = test_vae.decode(merged.float())
    assert np.all(np.diff(decodings_same.detach(), axis=1) == 0.0)
    

def test_reshape_decoding(test_vae, test_inputs) -> None:
    """Test that reshaping of decoder outputs
    for loss function is being done correctly.
    """
    input1, input2, _ = test_inputs
    z, _ = test_vae.encode(input1)
    # repeat time steps, NOT BAND WAVELENGTHS
    input2[:,:,0] = input2[:,0:1,0]
    merged = test_vae.merge(z, input2)
    decodings = test_vae.decode(merged.float())
    d_reshaped = test_vae.reshape_decoding(decodings).detach().numpy()
    
    assert d_reshaped.shape == (5, 7, 6)
    
    # check that the axis-2 arrays are identical to each other
    assert np.all(np.diff(d_reshaped, axis=1) == 0.0)
    
    # now check axis-1 reshape
    merged[:,:,:] = merged[:,0:1,:]
    decodings = test_vae.decode(merged.float())
    d_reshaped = test_vae.reshape_decoding(decodings).detach().numpy()
    
    assert np.all(np.diff(d_reshaped, axis=2) == 0.0)
    assert np.all(np.diff(d_reshaped, axis=1) == 0.0)
    assert np.all(np.diff(d_reshaped, axis=0) != 0.0)
