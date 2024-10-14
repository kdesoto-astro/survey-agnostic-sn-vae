import matplotlib.pyplot as plt
import numpy as np
from snapi import LightCurve, Photometry, Formatter, Filter
from astropy.time import Time
import astropy.units as u
import jax.numpy as jnp
import jax.random as random
import h5py


from survey_agnostic_sn_vae.autoencoder.raenn_equinox import (
    evaluate, generate_decoder_input,
    reconstruction_loss,
    dataloader
)
from survey_agnostic_sn_vae.autoencoder.contrastive_loss import contrastive_loss

def plot_decodings(
    encoder_inputs,
    ids,
    model,
    names=None,
    count=None,
):
    """Given a set of encoder_inputs and ids, and a number of list of IDs,
    plot decodings against original LCs. Return plots as a generator."""

    if names is None: # default to count
        if count is None:
           count = len(ids)

        # grab first "count" inputs
        encoder_sub = encoder_inputs[:count]
        id_sub = ids[:count]

    else:
        idx_sub = np.isin(ids, names)
        encoder_sub = encoder_inputs[idx_sub]
        id_sub = ids[idx_sub]


    encoder_dense = np.repeat(encoder_sub, 50, axis=1)
    for i in range(encoder_dense.shape[0]):
        times_wo_limits = encoder_sub[i,np.any(encoder_sub[i,:,13:19] == 0.0, axis=-1),0]
        dense_times = np.linspace(np.min(times_wo_limits), np.max(times_wo_limits), num=encoder_dense.shape[1])
        encoder_dense[i,:,0] = dense_times[np.newaxis, :]
    decoder_dense = generate_decoder_input(encoder_dense)
    
    result_dict = evaluate(model, encoder_sub, decoder_input=decoder_dense)
    y_pred = result_dict['pred_y']

    result_dict_orig = evaluate(model, encoder_sub)
    y_pred_orig = result_dict_orig['pred_y']
    
    nfilts = 6
    for i, iid in enumerate(id_sub):
        # plot the transient itself
        times = encoder_sub[i,:,0] * 100.
        dense_mags = encoder_sub[i,:,1:1+nfilts]
        dense_mag_errs = encoder_sub[i,:,1+nfilts:1+2*nfilts]
        interpolated_mask = encoder_sub[i,:,1+2*nfilts:1+3*nfilts]
        #TODO: add wavelength for filter labeling

        rec_lcs = set()
        for j in range(nfilts):
            recreation = LightCurve(
                times=Time(times[times < 1000.], format='mjd'),
                mags=-1*dense_mags[times < 1000.,j],
                mag_errs=dense_mag_errs[times < 1000.,j],
                upper_limits=interpolated_mask[times < 1000.,j].astype(bool),
                filt = Filter(
                    instrument="test",
                    band=str(j),
                    center=np.nan * u.AA
                )
            )
            rec_lcs.add(recreation)

        formatter = Formatter()

        fig, ax = plt.subplots()
        photometry = Photometry(rec_lcs)
        photometry.plot(ax=ax, formatter=formatter)

        formatter.reset_colors()
        formatter.reset_markers()

        # now overplot decodings
        for j in range(nfilts):
            plt.plot(
                encoder_dense[i,:,0] * 100.,
                -1*y_pred[i, :, j],
                label=j,
                color=formatter.edge_color
            )
            formatter.rotate_colors()

        recon_loss = reconstruction_loss(encoder_sub[i:i+1], y_pred_orig[i:i+1])

        ax.set_title(f"{iid.decode('utf-8')}: recon_loss={round(recon_loss, 2):0.2f}")
        formatter.make_plot_pretty(ax)
        formatter.add_legend(ax)

        yield fig, ax

def add_ellipse(ax, center, semimajor, semiminor, n_points=20, formatter=None, no_edge=False):
    if formatter is None:
        formatter = Formatter()
    theta = np.linspace(0, 2*np.pi, n_points)
    x = center[0] + semimajor * np.cos(theta)
    y = center[1] + semiminor * np.sin(theta)
    area = np.pi * semimajor * semiminor
    (legend_obj,) = ax.fill(x, y, color=formatter.face_color, alpha=min(0.01 / area, 1))
    if not no_edge:
        ax.plot(x, y, color=formatter.face_color, alpha=min(0.02 / area, 1))
    return ax, legend_obj

def plot_latent_space_helper(
        ax, means, logvars, matches=None, samples=None,
        classes=None, formatter=None,
    ):
    """plot latent space given the means and log-variances."""

    if formatter is None:
        formatter = Formatter()

    radii = np.sqrt(np.exp(logvars))
    center = means[:,0:2] # len (Nsamples, 2)
    rad = radii[:,0:2]
    key = random.key(42)

    if classes is not None:
        for c in np.unique(classes):
            key, subkey= random.split(key)
            repeat_factor = int(round(2*len(classes) / len(classes[classes == c])))
            center_rep = jnp.repeat(center[classes == c], repeat_factor, axis=0)
            rad_rep = jnp.repeat(rad[classes == c], repeat_factor, axis=0)
            random_samples = random.normal(subkey, shape=center_rep.shape)
            sampled_pts = center_rep + random_samples*rad_rep
            ax.scatter(
                sampled_pts[:0,0], sampled_pts[:0,1], s=formatter.marker_size,
                marker=formatter.marker_style,
                alpha=1,
                color=formatter.edge_color,
                label=c
            )
            ax.scatter(
                sampled_pts[:,0], sampled_pts[:,1], s=formatter.marker_size,
                marker=formatter.marker_style,
                alpha=formatter.nondetect_alpha,
                color=formatter.edge_color,
            )
            formatter.rotate_colors()
    
    formatter.rotate_colors()
    if matches is not None:
        for uid in jnp.unique(matches):
            matched_z1 = m0[matches == uid]
            matched_z2 = m1[matches == uid]
            N = len(matched_z1)
            for i in range(N):
                for j in range(i+1, N):
                    dist = float(jnp.sqrt((
                        matched_z1[i] - matched_z1[j]
                    )**2 + (
                        matched_z2[i] - matched_z2[j]
                    )**2))
                    if np.isnan(dist) or dist > 1:
                        dist = 1.0
                    ax.plot(
                        [matched_z1[i], matched_z1[j]],
                        [matched_z2[i], matched_z2[j]],
                        color=formatter.edge_color, linewidth=1,
                        alpha=dist
                    )
        cl = contrastive_loss(
            samples, means, logvars, matches,
            distance='mahalonobis',
            temp=1.0,
        )
        ax.set_title(f"Contrastive Loss: {cl}")
    """
    ax.scatter(
        m0, m1,
        s=10,
        color=formatter.edge_color, alpha=1,
        marker=formatter.marker_style
    )
    """
    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    formatter.make_plot_pretty(ax)
    return ax

def plot_latent_space(
    encoder_inputs,
    ids,
    model,
    classes=None,
    use_matches=True,
):
    """Plot latent space from encoder inputs and ids."""
    # convert ids to numerics
    _, ids = np.unique(ids, return_inverse=True)
    encoder_data, _, matches, _, _ = dataloader(encoder_inputs, ids, shuffle_key=None)

    if not use_matches:
        matches = None

    result_dict = evaluate(model, encoder_data)
    means = result_dict['mu']
    logvars = result_dict['logvar']
    samples = result_dict['z']
    fig, ax = plt.subplots()
    ax = plot_latent_space_helper(
        ax, means, logvars, samples=samples,
        matches=matches, classes=classes
    )
    return fig, ax


def plot_lc_all_wvs(
    ax, 
    encoder_input, model, wvs,
    prep_fn,
    formatter1=None,
    formatter2=None,
):
    encoder_input = np.array([encoder_input,])
    if formatter1 is None:
        formatter1 = Formatter()
    if formatter2 is None:
        formatter2 = Formatter()

    encoder_dense = np.repeat(encoder_input, 500, axis=1)
    min_t = min(np.min(encoder_dense[0,:,0]) - 0.03, -0.1)
    max_t = max(np.max(encoder_dense[0,encoder_dense[0,:,0] < 10.,0]) + 0.03, 0.4)
    dense_times = np.linspace(min_t, max_t, num=encoder_dense.shape[1])
    encoder_dense[0,:,0] = dense_times[np.newaxis, :]
    decoder_dense = generate_decoder_input(encoder_dense)
    
    filler_width = 1300.
    with h5py.File(prep_fn, 'r') as prep_data:
        wavemin = prep_data['encoder_input'].attrs['wavemin']
        wavemax = prep_data['encoder_input'].attrs['wavemax']
        bandmin = prep_data['encoder_input'].attrs['bandmin']
        bandmax = prep_data['encoder_input'].attrs['bandmax']

    for wv in wvs:
        # replace wv in decoder_dense
        decoder_dense = decoder_dense.at[:,:,1].set( (wv - wavemin) / (wavemax - wavemin) )
        decoder_dense = decoder_dense.at[:,:,2].set( filler_width / (wavemax - wavemin) )
        result_dict = evaluate(model, encoder_input, decoder_input=decoder_dense)
        y_pred = result_dict['pred_y']

        ax.plot(
            encoder_dense[0,:,0] * 100.,
            -1*(y_pred[0, :, 0] * (bandmax - bandmin) + bandmin),
            color=formatter1.edge_color
        )
        formatter1.rotate_colors()
    
    times = encoder_input[0,:,0] * 100.
    mags = encoder_input[0,:,1:7]
    mag_errs = encoder_input[0,:,7:13]
    wv_cen = encoder_input[0,:,19:25] * (wavemax - wavemin) + wavemin
    interpolated_mask = encoder_input[0,:,13:19].astype(bool)

    rec_lcs = set()
    for j in range(6):
        recreation = LightCurve(
            times=Time(times[~interpolated_mask[:,j] & (times < 1000.)], format='mjd'),
            mags=-1*(mags[~interpolated_mask[:,j] & (times < 1000.),j] * (bandmax - bandmin) + bandmin),
            mag_errs=mag_errs[~interpolated_mask[:,j] & (times < 1000.),j] * (bandmax - bandmin),
            filt = Filter(
                instrument=str(round(wv_cen[0,j])),
                band="AA",
                center=np.nan * u.AA
            )
        )
        rec_lcs.add(recreation)

    photometry = Photometry(rec_lcs)
    photometry.plot(ax=ax, formatter=formatter2)
    return ax

    



    




