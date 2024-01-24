import matplotlib.pyplot as plt
import numpy as np

SURVEY_SYMBOLS = {
    'LSST': '_',
    'ZTF': '^',
    'PanSTARRS': '*'
}
BAND_COLORS = {
    'u': 'blue',
    'g': 'green',
    'r': 'orange',
    'i': 'red',
    'z': 'pink',
    'y': 'purple'
}

def plot_lightcurves(transient, save_path, surveys=None):
    """Overlay all light curves listed in `surveys'
    for one transient in a single plot.
    """
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
        plt.axhline(
            y=lc.survey.limiting_magnitude, color='black',
            linestyle='dotted', linewidth=1
        )
        
    plt.xlabel('MJD')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

            