import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

SURVEY_SYMBOLS = {
    'LSST': '_',
    'ZTF': '^',
    'PanSTARRS': '*',
    '2MASS': '+',
    'Swift': 'o'
}
BAND_COLORS = {
    'u': 'blue',
    'g': 'green',
    'r': 'orange',
    'i': 'red',
    'z': 'pink',
    'y': 'purple',
    'w': 'gray',
    'B': 'cyan',
    'U': 'magenta',
    'UVM2': 'yellow',
    'UVW1': 'chartreuse',
    'UVW2':  'gold',
    'V': 'teal',
    'H': 'brown',
    'J': 'maroon',
    'Ks': 'navy'
}

def plot_lightcurves(transient, save_path, surveys=None):
    """Overlay all light curves listed in `surveys'
    for one transient in a single plot.
    """
    print(len(transient.lightcurves))
    if surveys is None:
        lcs = transient.lightcurves
    else:
        lcs = [
            l for l in transient.lightcurves if l.survey.name in surveys
        ]
        
    for lc in lcs:
        for b_uniq in lc.bands:
            print(len(lc.times[b_uniq]), len(lc.mag[b_uniq]))

            plt.errorbar(
                lc.times[b_uniq],
                lc.mag[b_uniq],
                yerr=lc.mag_err[b_uniq],
                fmt=SURVEY_SYMBOLS[lc.survey.name],
                label=f'{lc.survey.name}-{b_uniq}',
                ms=4,
                alpha=0.5,
                color=BAND_COLORS[b_uniq]
            )
        """
        plt.axhline(
            y=lc.survey.limiting_magnitude, color='black',
            linestyle='dotted', linewidth=1
        )
        """
        
    plt.xlabel('MJD')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    
def add_ellipse(ax, center, semimajor, semiminor, n_points=100):
    theta = np.linspace(0, 2*np.pi, n_points)
    x = center[0] + semimajor * np.cos(theta)
    y = center[1] + semiminor * np.sin(theta)
    ax.fill(x, y, color='gray', alpha=0.05)
    ax.plot(x, y, color='gray', alpha=0.1)
    return ax
    
def plot_latent_space(
    means, logvars, samples,
    ids, save_fn, show_contrastive=True
):
    """Plot latent space. If show_contrastive,
    shows lines connecting light curves from the
    same underlying object.
    """
    fig, ax = plt.subplots()
    radii = np.sqrt(np.exp(logvars))
    m0 = means[:,0]
    m1 = means[:,1]
    r0 = radii[:,0]
    r1 = radii[:,1]
    s0 = samples[:,0]
    s1 = samples[:,1]
    
    for i, _ in enumerate(m0):
        center = (m0[i], m1[i])
        ax = add_ellipse(ax, center, r0[i], r1[i])
        
    if show_contrastive:
        for uid in torch.unique(ids):
            matched_z1 = m0[ids == uid]
            matched_z2 = m1[ids == uid]
            N = len(matched_z1)
            for i in range(N):
                for j in range(i+1, N):
                    dist = torch.sqrt((
                        matched_z1[i] - matched_z1[j]
                    )**2 + (
                        matched_z2[i] - matched_z2[j]
                    )**2)
                    dist[torch.isnan(dist)] = 1
                    plt.plot(
                        matched_z1[[i,j]],
                        matched_z2[[i,j]],
                        color='red', linewidth=1,
                        alpha=0.2
                    )
    
    plt.scatter(s0, s1, s=1, color='k')
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.savefig(save_fn)
    plt.close()
    

            