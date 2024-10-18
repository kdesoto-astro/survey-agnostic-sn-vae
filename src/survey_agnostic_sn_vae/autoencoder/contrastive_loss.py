import jax.numpy as jnp
from jax import lax
from equinox import filter_jit

@filter_jit
def contrastive_loss(
    z, mu, logvar, matches,
    distance='mahalonobis',
    temp=1.0,
):
    """Calculate contrastive loss term.
    matches indicates matching events."""
    eps = 1e-6 # for numerical stability

    N = z.shape[0]
    # z is the latent variables
    S_i = jnp.repeat(z[jnp.newaxis,:,:-1], N, axis=0) # exclude last latent space dimension
    S_j = jnp.transpose(S_i, axes=[1, 0, 2])
    
    Mu_i = jnp.repeat(mu[jnp.newaxis,:,:-1], N, axis=0)
    Mu_j = jnp.transpose(Mu_i, axes=[1, 0, 2])
    
    Sig_i = jnp.repeat(logvar[jnp.newaxis,:,:-1], N, axis=0)
    Sig_j = jnp.transpose(Sig_i, axes=[1,0,2])
    stddev1 = jnp.exp(0.5*Sig_i)
    stddev2 = jnp.exp(0.5*Sig_j)
    
    # make "adjacency matrix" type thing for object IDs
    objid_mat = jnp.repeat(matches[jnp.newaxis,:], N, axis=0)
    objid_bool_mat = objid_mat == jnp.transpose(objid_mat, axes=[1,0])

    # Distance for object IDs is 1 if they're the same and 0 otherwise
    objid_dist = objid_bool_mat.astype(jnp.float32)
    objid_dist = jnp.fill_diagonal(objid_dist, 0.0, inplace=False) # unset diagonal
    
    # check rows where there's NO matches
    #no_match_idxs = jnp.all(objid_dist == 0.0, axis=0)
    
    # inverse identity matrix
    denom_arr = 1. - jnp.eye(N)
    
    # SOFT NEAREST NEIGHBORS LOSS
    if distance == 'cosine':
        cos_sim = jnp.sum(S_i * S_j, axis=-1) / (jnp.linalg.norm(S_i + eps, axis=-1) * jnp.linalg.norm(S_j + eps, axis=-1))
        dists = 1 - cos_sim
    
    elif distance == 'cosinemeans':
        cos_sim = jnp.sum(Mu_i * Mu_j, axis=-1) / (jnp.linalg.norm(Mu_i + eps, axis=-1) * jnp.linalg.norm(Mu_j + eps, axis=-1))
        dists = 1 - cos_sim
        
    elif distance == 'euclidean':
        dists = jnp.linalg.norm(S_i - S_j, axis=-1)
        
    elif distance == 'euclideanmeans':
        dists = jnp.linalg.norm(Mu_i - Mu_j, axis=-1)
        
    elif distance == 'kl':
        kl = jnp.log(stddev2/stddev1) + (stddev1**2 + (Mu_i - Mu_j).pow(2)) / (2*stddev2**2) - 0.5
        dists = jnp.median(kl, axis=-1)
        
    elif distance == 'mahalanobis':
        dists = jnp.linalg.norm((S_i - Mu_j) / (stddev2 + eps), axis=-1)
        
    elif distance == 'wasserstein':
        w_squared = jnp.linalg.norm(Mu_i - Mu_j, axis=-1)**2 + jnp.sum(stddev1**2 + stddev2**2 - 2*stddev1*stddev2, axis=-1)
        dists = w_squared

    else:
        raise ValueError(f"distance metric {distance} not implemented!")
    
    dists = lax.clamp(0., dists, 10.)
    exp_sims = jnp.exp(-dists / temp)
    num = jnp.nanmean(objid_dist * exp_sims, axis=1)
    denom = jnp.nanmean(denom_arr * exp_sims, axis=1)
    ratio = jnp.log(num / denom)
    
    l = -1. * jnp.nanmean(ratio)

    return l