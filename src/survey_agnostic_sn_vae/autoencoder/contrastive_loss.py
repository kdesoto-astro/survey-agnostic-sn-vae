import jax.numpy as jnp
from jax import lax
from equinox import filter_jit

@filter_jit
def contrastive_loss(
    z, mu, logvar, matches,
    distance='mahalanobis',
    temp=1.0,
):
    """Calculate contrastive loss term.
    matches indicates matching events."""
    eps = 1e-6 # for numerical stability

    N = z.shape[0]
    # z is the latent variables
    S_i = jnp.unsqueeze(z, 0).repeat((N,1,1))
    S_j = jnp.transpose(S_i, axes=[1, 0, 2])
    
    Mu_i = jnp.unsqueeze(mu, 0).repeat((N,1,1))
    Mu_j = jnp.transpose(Mu_i, axes=[1, 0, 2])
    
    Sig_i = jnp.unsqueeze(logvar, 0).repeat((N,1,1))
    Sig_j = jnp.transpose(Sig_i, axes=[1,0,2])
    stddev1 = jnp.exp(0.5*Sig_i)
    stddev2 = jnp.exp(0.5*Sig_j)
    
    # make "adjacency matrix" type thing for object IDs
    objid_mat = jnp.unsqueeze(matches, 0).repeat((N,1))
    objid_bool_mat = jnp.eq(objid_mat, jnp.transpose(objid_mat, axes=[1,0,2]))

    # Distance for object IDs is 0 if they're the same and 1 otherwise
    objid_dist = objid_bool_mat.type(jnp.float32)
    objid_dist[range(N), range(N)] = 0.0 # unset diagonal
    
    # check rows where there's NO matches
    no_match_idxs = jnp.all(objid_dist == 0.0, axis=0)
    
    # inverse identity matrix
    denom_arr = 1. - jnp.eye(N).to(z.device)
    
    # SOFT NEAREST NEIGHBORS LOSS
    if distance == 'cosine':
        cos_sim = jnp.dot(S_i, S_j) / (jnp.linalg.norm(S_i + eps) * jnp.linalg.norm(S_j + eps))
        dists = 1 - cos_sim
    
    elif distance == 'cosine_means':
        cos_sim = jnp.dot(Mu_i, Mu_j) / (jnp.linalg.norm(Mu_i + eps) * jnp.linalg.norm(Mu_j + eps))
        dists = 1 - cos_sim
        
    elif distance == 'euclidean':
        dists = jnp.linalg.norm(S_i - S_j, axis=-1)
        
    elif distance == 'euclidean_means':
        dists = jnp.linalg.norm(Mu_i - Mu_j, axis=-1)
        
    elif distance == 'kl':
        kl = jnp.log(stddev2/stddev1) + (stddev1**2 + (Mu_i - Mu_j).pow(2)) / (2*stddev2**2) - 0.5
        dists = jnp.median(kl, axis=-1)
        
    elif distance == 'mahalonobis':
        dists = jnp.linalg.norm((S_i - Mu_j)/ stddev2, axis=-1)
        
    elif distance == 'wasserstein':
        w_squared = jnp.linalg.norm(Mu_i - Mu_j, axis=-1)**2 + jnp.sum(stddev1**2 + stddev2**2 - 2*stddev1*stddev2, axis=-1)
        dists = w_squared

    else:
        raise ValueError(f"distance metric {distance} not implemented!")
    
    dists = lax.clamp(eps, dists, 50.)
    exp_sims = jnp.exp(-dists / temp)
    num = jnp.sum(
        (objid_dist * exp_sims)[~no_match_idxs][:,~no_match_idxs],
        axis=1
    )
    denom = jnp.sum(
        (denom_arr * exp_sims)[~no_match_idxs][:,~no_match_idxs],
        axis=1
    )
    ratio = jnp.log(num / denom)
    
    l = -1 * jnp.mean(ratio)

    return l