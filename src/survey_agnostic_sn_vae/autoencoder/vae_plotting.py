import matplotlib.pyplot as plt
import numpy as np
from snapi import LightCurve, Photometry, Formatter, Filter
from astropy.time import Time
import astropy.units as u
import jax.numpy as jnp

from survey_agnostic_sn_vae.autoencoder.raenn_equinox import (
    evaluate, generate_decoder_input,
    reconstruction_loss
)

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

def add_ellipse(ax, center, semimajor, semiminor, n_points=100, formatter=None):
    if formatter is None:
        formatter = Formatter()
    theta = np.linspace(0, 2*np.pi, n_points)
    x = center[0] + semimajor * np.cos(theta)
    y = center[1] + semiminor * np.sin(theta)
    ax.fill(x, y, color=formatter.face_color, alpha=0.05)
    ax.plot(x, y, color=formatter.face_color, alpha=0.1)
    return ax

def plot_latent_space_helper(
        means, logvars, matches=None
    ):
    """plot latent space given the means and log-variances."""
    fig, ax = plt.subplots()
    formatter = Formatter()

    radii = np.sqrt(np.exp(logvars))
    m0 = means[:,0]
    m1 = means[:,1]
    r0 = radii[:,0]
    r1 = radii[:,1]
    
    for i, _ in enumerate(m0):
        center = (m0[i], m1[i])
        ax = add_ellipse(ax, center, r0[i], r1[i], formatter=formatter)
        
    if matches:
        for uid in jnp.unique(matches):
            matched_z1 = m0[matches == uid]
            matched_z2 = m1[matches == uid]
            N = len(matched_z1)
            for i in range(N):
                for j in range(i+1, N):
                    dist = jnp.sqrt((
                        matched_z1[i] - matched_z1[j]
                    )**2 + (
                        matched_z2[i] - matched_z2[j]
                    )**2)
                    dist[jnp.isnan(dist)] = 1
                    ax.plot(
                        matched_z1[[i,j]],
                        matched_z2[[i,j]],
                        color=formatter.edge_color, linewidth=1,
                        alpha=0.2
                    )
    ax.scatter(
        m0, m1,
        s=10,
        color=formatter.edge_color, alpha=1,
        marker=formatter.marker_style
    )
    #ax.set_xlim((-1, 1))
    #ax.set_ylim((-1, 1))
    formatter.make_plot_pretty(ax)
    return fig, ax

def plot_latent_space(
    encoder_inputs,
    model,
    matches=None,
):
    """Plot latent space from encoder inputs and ids."""
    result_dict = evaluate(model, encoder_inputs[:100])
    means = result_dict['mu']
    logvars = result_dict['logvar']
    fig, ax = plot_latent_space_helper(means, logvars, matches=matches)
    return fig, ax





