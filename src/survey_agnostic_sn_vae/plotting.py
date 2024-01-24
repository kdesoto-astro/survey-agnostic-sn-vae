import matplotlib.pyplot as plt
import numpy as np

def plot_lightcurve(transient, save_dir, surveys=None):
    """Overlay all light curves listed in `surveys'
    for one transient in a single plot.
    """
    