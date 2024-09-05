import functools as ft
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import wandb

from survey_agnostic_sn_vae.autoencoder.contrastive_loss import contrastive_loss
jax.config.update('jax_platform_name', 'cpu')

@jax.jit
def sample_with_dynamic_indices(array, indices):
    """For some reason works around concrete index error."""
    return array[:,:,indices]
    
class TimeDistributedDense(eqx.Module):
    """Mimics PyTorch's TimeDistributed Layer for a Dense Layer."""
    dense: eqx.Module
    
    def __init__(self, features_in, features_out, key):
        self.dense = eqx.nn.Linear(
            features_in, features_out, key=key
        )
    
    def __call__(self, x):
        # Assuming x has shape (batch_size, time_steps, features_in)
        return jax.vmap(self.dense, in_axes=0, out_axes=0)(x)  # Apply across the time axis

class SymmetricEncoder(eqx.Module):
    """The SymmetricEncoder Equinox module.
    Uses GRUs to convert input time series to latent
    mean and variance. Enforces symmetry among filter inputs:
    effectively identical to summing the filter vals within each
    input and feeding into a smaller Encoder.
    """
    vmapped_layers: list
    layers_after_gru: list
    gru: eqx.Module
    prelu: eqx.nn.PReLU

    def __init__(
            self,
            #input_dim: int,
            hidden_dim: int,
            key=jax.random.key(42),
        ):
        key1, key2, key3, key4 = jax.random.split(key, num=4)

        self.vmapped_layers = [
            TimeDistributedDense(5, hidden_dim, key=key1), # t, m_combined, merr_combined, wv_combined, width_combined
            eqx.nn.PReLU(),
            TimeDistributedDense(hidden_dim, hidden_dim, key=key2), # t, m_combined, merr_combined, wv_combined, width_combined
            #eqx.nn.PReLU(),
            #TimeDistributedDense(hidden_dim, hidden_dim, key=key3), # t, m_combined, merr_combined, wv_combined, width_combined
        ]
        self.prelu = eqx.nn.PReLU()
        self.gru = eqx.nn.GRUCell(hidden_dim, hidden_dim, key=key4)
        self.layers_after_gru = [eqx.nn.PReLU(),]

    def __call__(self, x):
        x = jnp.delete(x, jnp.arange(13, 19), axis=-1, assume_unique_indices=True) # we don't want the mask to be passed through encoder

        # make symmetric via shared sublayer
        sublayer1 = x[:,[0,1,7,13,19]]
        sublayer2 = x[:,[0,2,8,14,20]]
        sublayer3 = x[:,[0,3,9,15,21]]
        sublayer4 = x[:,[0,4,10,16,22]]
        sublayer5 = x[:,[0,5,11,17,23]]
        sublayer6 = x[:,[0,6,12,18,24]]

        for layer in self.vmapped_layers:
            sublayer1 = layer(sublayer1)
            sublayer2 = layer(sublayer2)
            sublayer3 = layer(sublayer3)
            sublayer4 = layer(sublayer4)
            sublayer5 = layer(sublayer5)
            sublayer6 = layer(sublayer6)

        combined_layer = sublayer1 + sublayer2 + sublayer3 + sublayer4 + sublayer5 + sublayer6
        x = self.prelu(combined_layer)

        def scan_fn(carry, inp):
            return self.gru(inp, carry), None
        
        init_state = jnp.zeros(self.gru.hidden_size)
        x, _ = jax.lax.scan(scan_fn, init_state, x)
        for layer in self.layers_after_gru:
            x = layer(x)
        return x


class Decoder(eqx.Module):
    """The Decoder Equinox module.
    Samples from the latent space distribution and
    passes sample through time-distributed MLP to return magnitudes."""
    layers: list

    def __init__(
            self,
            out_dim: int,
            hidden_dim: int,
            key=jax.random.key(42),
        ):
        key1, key2, key3 = jax.random.split(key, num=3)
        self.layers = [
            TimeDistributedDense(out_dim, hidden_dim, key=key1),
            eqx.nn.PReLU(),
            TimeDistributedDense(hidden_dim, hidden_dim, key=key2),
            eqx.nn.PReLU(),
            TimeDistributedDense(hidden_dim, 1, key=key3),
            eqx.nn.PReLU(),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class VAE(eqx.Module):
    """Full VAE, including encoder, decoder + sampler."""
    encoder: eqx.Module
    decoder: eqx.Module
    latent_mean: eqx.nn.Linear
    latent_logvar: eqx.nn.Linear
    sample_noise: jax.Array

    def __init__(
            self,
            hidden_dim: int,
            out_dim: int,
            key=jax.random.key(42),
        ):
        key1, key2, key3, key4, key5 = jax.random.split(key, num=5)
        self.encoder = SymmetricEncoder(hidden_dim, key1)
        self.decoder = Decoder(
            out_dim+3, hidden_dim, key2
        )

        self.latent_mean = eqx.nn.Linear(hidden_dim, out_dim, key=key3)
        self.latent_logvar = eqx.nn.Linear(hidden_dim, out_dim, key=key4)
        self.sample_noise = jax.random.normal(key=key5, shape=(10_000, 1_000, out_dim)) 

        #self.reparam_key = key5

    def __call__(self, encoder_input, decoder_input, vmapped_idx, epoch_count):
        x = self.encoder(encoder_input)
        mu = self.latent_mean(x)
        logvar = self.latent_logvar(x)
        z = mu + self.sample_noise[epoch_count, vmapped_idx] * jnp.exp(logvar / 2.)
        # repeat num_times * 6
        x = z[jnp.newaxis, :]
        x = jnp.repeat(x, decoder_input.shape[0], axis=0)
        x = jnp.concatenate((x, decoder_input), axis=1)
        x = self.decoder(x)
        x = jnp.reshape(x, (-1, 6)) # TODO: un-hardcode
        return x, mu, logvar, z
    
@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def compute_loss(
    model, epoch, encoder_input, decoder_input, matches,
    include_reconstructive: bool=True,
    include_kl: bool=True,
    contrastive_distance: Optional[str]=None,
    contrastive_temp: Optional[float]=None,
):
    """Calculate the overall loss of model on x, given
    true recreations y. Matches indicate arrays from
    same event."""
    pred_y, mu, logvar, z = jax.vmap(model, in_axes=(0,0,0,None))(
        encoder_input, decoder_input,
        jnp.arange(len(encoder_input)),
        epoch
    )
    loss = 0
    kl = jnp.nan
    cl = jnp.nan
    rl = jnp.nan
    if include_reconstructive:
        rl = reconstruction_loss(encoder_input, pred_y)
        loss += rl
    if include_kl:
        kl = kl_loss(mu, logvar)
        loss += kl
    if contrastive_distance:
        cl = contrastive_loss(
            z, mu, logvar, matches,
            distance=contrastive_distance,
            temp=contrastive_temp
        )
        loss += cl
    return loss, (rl, kl, cl)

@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def compute_loss_frozen(
    diff_model, static_model,
    epoch, encoder_input, decoder_input, matches,
    include_reconstructive: bool=True,
    include_kl: bool=True,
    contrastive_distance: Optional[str]=None,
    contrastive_temp: Optional[float]=None,
):
    """Compute loss for partially frozen model."""
    model = eqx.combine(diff_model, static_model)
    pred_y, mu, logvar, z = jax.vmap(model, in_axes=(0,0,0,None))(
        encoder_input, decoder_input,
        jnp.arange(len(encoder_input)),
        epoch
    )
    loss = 0
    kl = jnp.nan
    cl = jnp.nan
    rl = jnp.nan
    if include_reconstructive:
        rl = reconstruction_loss(encoder_input, pred_y)
        loss += rl
    if include_kl:
        kl = kl_loss(mu, logvar)
        loss += kl
    if contrastive_distance:
        cl = contrastive_loss(
            z, mu, logvar, matches,
            distance=contrastive_distance,
            temp=contrastive_temp
        )
        loss += cl
    return loss, (rl, kl, cl)

@eqx.filter_jit
def reconstruction_loss(y, pred_y):
    """
    Calculate the reconstruction loss of the model.
    """
    numerator = (y[:,:,1:7] - pred_y) ** 2 * (1 - y[:,:,13:19]) / y[:,:,7:13] ** 2
    # prevent outliers from greatly affecting gradients
    #num_clipped = jnp.clip(numerator, min=0.0, max=10.) + 0.1*numerator
    return jax.numpy.sum(numerator) / jax.numpy.sum(1 - y[:,:,13:19])

@eqx.filter_jit
def kl_loss(mu, logvar):
    """
    Calculate the reconstruction loss of the model.
    """
    return - 0.5 * jax.numpy.mean(1 + logvar - mu ** 2 - jax.numpy.exp(logvar))

def generate_decoder_input(sequence):
    """Calculate decoder input given a sequence."""
    nfilts = 6
    nfiltsp3 = 3 * nfilts + 1
    nfiltsp4 = 4 * nfilts + 1
    sequence_len = sequence.shape[1]
    outseq = jnp.reshape(sequence[:, :, 0], (len(sequence), sequence_len, 1)) * 1.0

    # tile for each wv
    outseq_tiled = jnp.repeat(outseq, nfilts, axis=1)
    outseq_wvs = jnp.reshape(sequence[:, :, nfiltsp3:nfiltsp4], (len(sequence), nfilts*sequence_len, 1)) * 1.0
    outseq_filter_widths = jnp.reshape(sequence[:, :, nfiltsp4:], (len(sequence), nfilts*sequence_len, 1)) * 1.0
    outseq_tiled = jnp.dstack((outseq_tiled, outseq_wvs, outseq_filter_widths))

    return outseq_tiled

@eqx.filter_jit
def dataloader(encoder_data, ids, shuffle_key:Optional[jax.random.key]=None):
    """Shuffle and load encoder and decoder arrays.
    Within this function we generate (1) pairs of samples with subsets
    of LCs from the same events to enforce our contrastive loss, 
    (2) shuffle bands randomly, and (3) generate decoder array from
    encoder array."""

    if shuffle_key is None:
        shuffled_idx1 = jnp.array([0,2,4,0,2,4])
        shuffled_idx2 = jnp.array([1,3,5,1,3,5])
    
    else:
        dataset_size = encoder_data.shape[0]
        indices = jnp.arange(dataset_size)
        new_key, key = jax.random.split(shuffle_key)
        perm = jax.random.permutation(key=key, x=indices)
        shuffled_vals = jax.random.choice(key, 6, (6,), replace=False) # can be repeats
        shuffled_idx1 = shuffled_vals[jnp.repeat(shuffled_vals[:3], 2)]
        shuffled_idx2 = shuffled_vals[jnp.repeat(shuffled_vals[3:], 2)]

    @jax.jit
    def update_slices(carry, start_index):
        updated_data1, updated_data2 = carry
        new_slice_data1 = sample_with_dynamic_indices(encoder_data, start_index+shuffled_idx1)
        new_slice_data2 = sample_with_dynamic_indices(encoder_data, start_index+shuffled_idx2)

        updated_data1 = jax.lax.dynamic_update_slice(updated_data1, new_slice_data1, (0, 0, start_index))
        updated_data2 = jax.lax.dynamic_update_slice(updated_data2, new_slice_data2, (0, 0, start_index))

        return (updated_data1, updated_data2), None

    batch_split1 = jnp.array(encoder_data)
    batch_split2 = jnp.array(encoder_data)

    carry = (batch_split1, batch_split2,)
    carry, _ = jax.lax.scan(update_slices, carry, 6 * jnp.arange(5) + 1 )
    (batch_split1, batch_split2) = carry
    encoder_pairs = jnp.vstack((batch_split1, batch_split2))
    decoder_data = generate_decoder_input(encoder_pairs)
    match_data = jnp.tile(ids, 2)

    if shuffle_key is None:
        return encoder_pairs, decoder_data, match_data, None, None
    
    perm_doubled = jnp.repeat(perm, 2)
    even_idxs = jnp.arange(1,dataset_size, 2)
    perm_doubled = perm_doubled.at[even_idxs].set(perm_doubled[even_idxs-1] + dataset_size) # makes sure pairs are summoned together
    return encoder_pairs, decoder_data, match_data, perm_doubled, new_key

def fit_model(
        model, encoder_inputs,
        ids,
        val_encoder_inputs,
        val_ids,
        learning_rate=1e-3,
        num_epochs=1000,
        batch_size=32,
        transfer_learning=False,
        wandb_log=False,
        include_reconstructive: bool=True,
        include_kl: bool=True,
        contrastive_params: Optional[str]=None
    ):
    """Main loop of the VAE.
    If transfer_learning is True, only unfreeze mu, logvar, and first decoder layers."""

    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda tree: (
            tree.latent_mean.weight,
            tree.latent_mean.bias,
            tree.latent_logvar.weight,
            tree.latent_logvar.bias,
            tree.decoder.layers[0].dense.weight,
            tree.decoder.layers[0].dense.bias,
        ),
        filter_spec,
        replace=(True, True, True, True, True, True),
    )

    key = jax.random.key(42)
    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    #flat_model, treedef_model = jax.tree_util.tree_flatten(model)
    #flat_opt_state, treedef_opt_state = jax.tree_util.tree_flatten(opt_state)

    # convert ids to numerics
    _, ids = np.unique(ids, return_inverse=True)
    _, val_ids = np.unique(val_ids, return_inverse=True)

    val_encoder_data, val_decoder_data, val_matches, _, _ = dataloader(val_encoder_inputs, val_ids, shuffle_key=None)

    if contrastive_params is not None:
        assert "_" in contrastive_params
        temp = float(contrastive_params.split("_")[1])
        distance = contrastive_params.split("_")[0]
    else:
        distance, temp = None, None
    
    #@eqx.filter_jit
    def epoch_iterator(A, epoch):
        encoder_data, decoder_data, matches, perm, new_key = dataloader(encoder_inputs, ids, shuffle_key=A[-1])

        @eqx.filter_jit
        def make_step(A, perm_idxs):
            model, opt_state = A
            #flat_model, flat_opt_state = A
            #model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
            #opt_state = jax.tree_util.tree_unflatten(treedef_opt_state, flat_opt_state)

            encoder_batch = encoder_data[perm_idxs]
            decoder_batch = decoder_data[perm_idxs]
            matches_batch = matches[perm_idxs]

            if transfer_learning:
                diff_model, static_model = eqx.partition(model, filter_spec)
                (loss, (rl, kl, cl)), grads = compute_loss_frozen(
                    diff_model, static_model,
                    epoch, encoder_batch, decoder_batch, matches_batch,
                    include_reconstructive=include_reconstructive,
                    include_kl=include_kl, contrastive_distance=distance,
                    contrastive_temp=temp
                )
                
            else:
                (loss, (rl, kl, cl)), grads = compute_loss(
                    model, epoch, encoder_batch, decoder_batch, matches_batch,
                    include_reconstructive=include_reconstructive,
                    include_kl=include_kl, contrastive_distance=distance,
                    contrastive_temp=temp,
                )
            (val_loss, (val_rl, val_kl, val_cl)), _ = compute_loss(
                model, epoch, val_encoder_data, val_decoder_data, val_matches,
                include_reconstructive=include_reconstructive,
                include_kl=include_kl, contrastive_distance=distance,
                contrastive_temp=temp,
            )
            updates, update_opt_state = optim.update(grads, opt_state)
            update_model = eqx.apply_updates(model, updates)
            #flat_update_model = jax.tree_util.tree_leaves(update_model)
            #flat_update_opt_state = jax.tree_util.tree_leaves(update_opt_state)

            #return (flat_update_model, flat_update_opt_state), jnp.array([loss, rl, kl, cl, val_loss, val_rl, val_kl, val_cl])
            return (update_model, update_opt_state), jnp.array([loss, rl, kl, cl, val_loss, val_rl, val_kl, val_cl])
        
        perm_new_len = (len(encoder_data) // batch_size) * batch_size
        perm_reshaped = jnp.reshape(perm[:perm_new_len], shape=(-1,batch_size))
        
        if transfer_learning:
            aggregate = []
            for p in perm_reshaped:
                (A1, A2), out = make_step(A[:2], p)
                aggregate.append(out)
                A = (A1, A2, new_key)
            aggregate = jnp.array(aggregate)
        else:
            (A1, A2), aggregate = jax.lax.scan(make_step, A[:2], perm_reshaped)
            A = (A1, A2, new_key)
        return A, jnp.mean(aggregate, axis=0)
    
    log_losses = []
    log_val_losses = []
    for e in range(num_epochs):
        #(flat_model, flat_opt_state, key), (l, rl, kl, cl, vl, vrl, vkl, vcl) = epoch_iterator((flat_model, flat_opt_state, key), e)
        (model, opt_state, key), (l, rl, kl, cl, vl, vrl, vkl, vcl) = epoch_iterator((model, opt_state, key), e)
        loss = jnp.log10(l).item()
        val_loss = jnp.log10(vl).item()
        rl_loss = jnp.log10(rl).item()
        kl_loss_term = jnp.log10(kl).item()
        cl_loss = jnp.log10(cl).item()
        rl_val_loss = jnp.log10(vrl).item()
        kl_val_loss = jnp.log10(vkl).item()
        cl_val_loss = jnp.log10(vcl).item()

        if wandb_log:
            wandb.log({
                "log_loss": loss,
                "val_log_loss": val_loss,
                "reconstruction_log_loss": rl_loss,
                "kl_log_loss": kl_loss_term,
                "contrastive_log_loss": cl_loss,
                "val_reconstruction_log_loss": rl_val_loss,
                "val_kl_log_loss": kl_val_loss,
                "val_contrastive_log_loss": cl_val_loss,
            })
        
        else:
            print(f"Epoch {e}: Training Log Loss={round(loss, 3)}, Validation Log Loss={round(val_loss, 3)}")

        log_losses.append(loss)
        log_val_losses.append(val_loss)

    #model = jax.tree_util.tree_unflatten(treedef_model, flat_model)

    return model, log_losses, log_val_losses

@eqx.filter_jit
def evaluate(
    model,
    encoder_input,
    decoder_input=None,
):
    """Evaluate the model on a dataset."""
    if decoder_input is None:
        decoder_input = generate_decoder_input(encoder_input)
    
    inference_model = eqx.nn.inference_mode(model)
    pred_y, mu, logvar, z = jax.vmap(
        inference_model, in_axes=(0,0,0,None)
    )(encoder_input, decoder_input, jnp.arange(len(encoder_input)), 0)
    return {
        "pred_y": pred_y,
        "mu": mu,
        "logvar": logvar,
        "z": z
    }


    