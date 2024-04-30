import os, glob
import numpy as np
import datetime

from survey_agnostic_sn_vae.data_generation.utils import *
from survey_agnostic_sn_vae.data_generation.objects import Transient

DEFAULT_ZPT = 26.3
now = datetime.datetime.now()
date = str(now.strftime("%Y-%m-%d"))

from astropy.cosmology import Planck13 as cosmo
import numpy as np
import scipy
import george
import extinction
import logging

class LightCurve(object):
    """Light Curve class
    """
    def __init__(self, name, times, fluxes, flux_errs, filters,
                 zpt=0, mwebv=0, redshift=None, lim_mag=None,
                 obj_type=None):

        self.name = name
        self.times = times
        self.fluxes = fluxes
        self.flux_errs = flux_errs
        self.filters = filters
        self.zpt = zpt
        self.mwebv = mwebv
        self.redshift = redshift
        self.lim_mag = lim_mag
        self.obj_type = obj_type

        self.abs_mags = None
        self.abs_mags_err = None
        self.abs_lim_mag = None

    def sort_lc(self):
        gind = np.argsort(self.times)
        self.times = self.times[gind]
        self.fluxes = self.fluxes[gind]
        self.flux_errs = self.flux_errs[gind]
        self.filters = self.filters[gind]
        if self.abs_mags is not None:
            self.abs_mags = self.abs_mags[gind]
            self.abs_mags_err = self.abs_mags_err[gind]

    def find_peak(self, tpeak_guess):
        gind = np.where((np.abs(self.times-tpeak_guess) < 100.0) &
                        (self.fluxes/self.flux_errs > 3.0))
        if len(gind[0]) == 0:
            gind = np.where((np.abs(self.times - tpeak_guess) < 100.0))
        if self.abs_mags is not None:
            tpeak = self.times[gind][np.argmin(self.abs_mags[gind])]
        return tpeak

    def cut_lc(self, limit_before=70, limit_after=200):
        gind = np.where((self.times > -limit_before) &
                        (self.times < limit_after))
        self.times = self.times[gind]
        self.fluxes = self.fluxes[gind]
        self.flux_errs = self.flux_errs[gind]
        self.filters = self.filters[gind]
        if self.abs_mags is not None:
            self.abs_mags = self.abs_mags[gind]
            self.abs_mags_err = self.abs_mags_err[gind]

    def shift_lc(self, t0=0):
        self.times = self.times - t0

    def correct_time_dilation(self):
        self.times = self.times / (1.+self.redshift)

    def correct_extinction(self, wvs):
        alams = extinction.fm07(wvs, self.mwebv)
        for i, alam in enumerate(alams):
            gind = np.where(self.filters == str(i))
            self.abs_mags[gind] = self.abs_mags[gind] - alam

    def add_LC_info(self, zpt=27.5, mwebv=0.0, redshift=0.0,
                    lim_mag=25.0, obj_type='-'):
        self.zpt = zpt
        self.mwebv = mwebv
        self.redshift = redshift
        self.lim_mag = lim_mag
        self.obj_type = obj_type

    def get_abs_mags(self, replace_nondetections=True, mag_err_fill=1.0):
        """
        Convert flux into absolute magnitude

        Parameters
        ----------
        replace_nondetections : bool
            Replace nondetections with limiting mag.

        Returns
        -------
        self.abs_mags : list
            Absolute magnitudes

        Examples
        --------
        """
        k_correction = 2.5 * np.log10(1.+self.redshift)
        dist = cosmo.luminosity_distance([self.redshift]).value[0]  # returns dist in Mpc

        self.abs_mags = -2.5 * np.log10(self.fluxes) + self.zpt - 5. * \
            np.log10(dist*1e6/10.0) + k_correction
        # Sketchy way to calculate error - update later
        self.abs_mags_plus_err = -2.5 * np.log10(self.fluxes + self.flux_errs) + self.zpt - 5. * \
            np.log10(dist*1e6/10.0) + k_correction
        self.abs_mags_err = np.abs(self.abs_mags_plus_err - self.abs_mags)

        if replace_nondetections:
            abs_lim_mag = self.lim_mag - 5.0 * np.log10(dist * 1e6 / 10.0) + \
                            k_correction
            gind = np.where((np.isnan(self.abs_mags)) |
                            np.isinf(self.abs_mags) |
                            np.isnan(self.abs_mags_err) |
                            np.isinf(self.abs_mags_err) |
                            (self.abs_mags > self.lim_mag))

            self.abs_mags[gind] = abs_lim_mag
            self.abs_mags_err[gind] = mag_err_fill
        self.abs_lim_mag = abs_lim_mag

        return self.abs_mags, self.abs_mags_err

    def filter_names_to_numbers(self, filt_dict):
        for i, filt in enumerate(self.filters):
            self.filters[i] = filt_dict[filt]

    def make_dense_LC(self, nfilts):
        gp_mags = self.abs_mags - self.abs_lim_mag
        dense_fluxes = np.zeros((len(self.times), nfilts))
        dense_errs = np.zeros((len(self.times), nfilts))
        stacked_data = np.vstack([self.times, self.filters]).T
        x_pred = np.zeros((len(self.times)*nfilts, 2))
        kernel = np.var(gp_mags) * george.kernels.ExpSquaredKernel([100, 1], ndim=2)
        gp = george.GP(kernel)
        gp.compute(stacked_data, self.abs_mags_err)

        def neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(gp_mags)

        def grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(gp_mags)

        result = scipy.optimize.minimize(neg_ln_like,
                                         gp.get_parameter_vector(),
                                         jac=grad_neg_ln_like)
        gp.set_parameter_vector(result.x)
        for jj, time in enumerate(self.times):
            x_pred[jj*nfilts:jj*nfilts+nfilts, 0] = [time]*nfilts
            x_pred[jj*nfilts:jj*nfilts+nfilts, 1] = np.arange(nfilts)
        pred, pred_var = gp.predict(gp_mags, x_pred, return_var=True)

        for jj in np.arange(nfilts):
            gind = np.where(x_pred[:, 1] == jj)[0]
            dense_fluxes[:, int(jj)] = pred[gind] + self.abs_lim_mag
            dense_errs[:, int(jj)] = np.sqrt(pred_var[gind])
        self.dense_lc = np.dstack((dense_fluxes, dense_errs))
        gp.recompute()
        self.gp = gp
        self.gp_mags = gp_mags
        return gp, gp_mags
        # Need except statementgp.set_parameter_vector([1, 100, 1])


def save_lcs(lc_list, output_dir):
    """
    Save light curves as a lightcurve object

    Parameters
    ----------
    lc_list : list
        list of light curve files
    output_dir : Output directory of light curve file

    Todo:
    ----------
    - Add option for LC file name
    """
    now = datetime.datetime.now()
    date = str(now.strftime("%Y-%m-%d"))
    file_name = 'lcs_' + date + '.npz'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if output_dir[-1] != '/':
        output_dir += '/'

    output_file = output_dir + file_name
    np.savez(output_file, lcs=lc_list)
    # Also easy save to latest
    np.savez(output_dir+'lcs.npz', lcs=lc_list)

    logging.info(f'Saved to {output_file}')


def generate_superraenn_lc_file(
    transient_dir,
    save_dir
):
    """Generate lcs.npz file for use by SuperRAENN.
    """
    all_transient_files = glob.glob(
        os.path.join(transient_dir, "*.pickle")
    )
    all_transients = [
        Transient.load(t) for t in all_transient_files
    ]
    
    sr_lcs = []

    transient_id = 0

    # Update the LC objects with info from the metatable
    for i, transient in enumerate(all_transients):
        if i % 50 == 0:
            print(i)
        params = transient.model_params
        if params['redshift'] <= 0.0:
            continue
            
        for lc in transient.lightcurves:
            if lc.survey.name != 'LSST':
                continue
            t, m, merr, b = lc.get_arrays()
            
            if len(t) < 5:
                continue
                
            f, ferr = convert_mag_to_flux(
                m, merr, DEFAULT_ZPT
            )
            
            sr_lc = LightCurve(
                name=lc.obj_id,
                times=t,
                fluxes=f,
                flux_errs=ferr,
                filters=b,
            )
            
            sr_lc.add_LC_info(
                zpt=DEFAULT_ZPT,
                mwebv=0.0,
                redshift=params['redshift'],
                lim_mag=lc.survey.limiting_magnitude,
                obj_type=transient.model_type
            )
            
            sr_lc.group = transient_id

            sr_lc.get_abs_mags()
            sr_lc.sort_lc()
            pmjd = sr_lc.find_peak(
                lc.find_peak_mag(composite=True)[0]
            )
            sr_lc.shift_lc(pmjd)
            sr_lc.correct_time_dilation()
            filt_dict = {f: i for i, f in enumerate(lc.bands)}

            sr_lc.wavelengths = np.zeros(len(lc.bands))
            for f in filt_dict:
                sr_lc.wavelengths[filt_dict[f]] = lc.survey.band_wavelengths[f]
                
            sr_lc.filter_names_to_numbers(filt_dict)
            sr_lc.cut_lc()
            try:
                sr_lc.make_dense_LC(len(lc.bands))
            except:
                print("SKIPPED")
                continue
            
            # pad dense LC to six bands
            if len(lc.bands) <= 3:
                tile_factor = int(6 / len(lc.bands))
                sr_lc.dense_lc = np.repeat(
                    sr_lc.dense_lc, tile_factor,
                    axis=1
                )
                sr_lc.wavelengths = np.repeat(
                    sr_lc.wavelengths, tile_factor,
                )
                
            elif len(lc.bands) < 6:
                sr_lc.dense_lc = np.repeat(
                    sr_lc.dense_lc, 2,
                    axis=1
                )[:,:6,:]
                sr_lc.wavelengths = np.repeat(
                    sr_lc.wavelengths, 2
                )[:6]
                
            sr_lcs.append(sr_lc)
        transient_id += 1
            
    save_lcs(sr_lcs, save_dir)
    
    
def prep_input(input_lc_file, new_t_max=200.0, filler_err=3.0,
               save=False, load=False, outdir=None, prep_file=None):
    """
    Prep input file for fitting

    Parameters
    ----------
    input_lc_file : str
        True flux values
    new_t_max : float
        Predicted flux values
    filler_err : float
        Predicted flux values
    save : bool
        Predicted flux values
    load : bool
        Predicted flux values
    outdir : str
        Predicted flux values
    prep_file : str
        Predicted flux values

    Returns
    -------
    sequence : numpy.ndarray
        Array LC flux times, values and errors
    outseq : numpy.ndarray
        An array of LC flux values and limiting magnitudes
    ids : numpy.ndarray
        Array of SN names
    sequence_len : float
        Maximum length of LC values
    nfilts : int
        Number of filters in LC files
    """
    lightcurves = np.load(input_lc_file, allow_pickle=True)['lcs']
    lengths = []
    ids = []
    for lightcurve in lightcurves:
        lengths.append(len(lightcurve.times))
        ids.append(lightcurve.name)

    sequence_len = 183 #np.max(lengths)
    nfilts = np.shape(lightcurves[0].dense_lc)[1]
    nfiltsp1 = nfilts+1
    nfiltsp2 = 2*nfilts+1
    n_lcs = len(lightcurves)
    # convert from LC format to list of arrays
    # sequence = np.zeros((n_lcs, sequence_len, nfilts*2+1))
    sequence = np.zeros((n_lcs, sequence_len, nfilts*3 + 2))

    lms = []
    for i, lightcurve in enumerate(lightcurves):
        sequence[i, 0:lengths[i], 0] = lightcurve.times[:sequence_len] / 1000. # to keep values small
        sequence[i, 0:lengths[i], 1:nfiltsp1] = lightcurve.dense_lc[:sequence_len, :, 0] # fluxes
        sequence[i, 0:lengths[i], nfiltsp1:nfiltsp2] = lightcurve.dense_lc[:sequence_len, :, 1] # flux errors
        sequence[i, lengths[i]:, 0] = (np.max(lightcurve.times)+new_t_max) / 1000.
        sequence[i, lengths[i]:, 1:nfiltsp1] = lightcurve.abs_lim_mag
        sequence[i, lengths[i]:, nfiltsp1:nfiltsp2] = filler_err
        sequence[i, :, nfiltsp2:-1] = lightcurve.wavelengths
        sequence[i, :, -1] = lightcurve.group
        
        lms.append(lightcurve.abs_lim_mag)

    # Flip because who needs negative magnitudes
    sequence[:, :, 1:nfiltsp1] = -1.0 * sequence[:, :, 1:nfiltsp1]
    
    if load:
        prep_data = np.load(prep_file)
        bandmin = prep_data['bandmin']
        bandmax = prep_data['bandmax']
        wavemin = prep_data['wavemin']
        wavemax = prep_data['wavemax']
    else:
        bandmin = np.min(sequence[:, :, 1:nfiltsp1])
        bandmax = np.max(sequence[:, :, 1:nfiltsp1])
        wavemin = np.min(sequence[:, :, nfiltsp2:-1])
        wavemax = np.max(sequence[:, :, nfiltsp2:-1])

    # Normalize flux values, flux errors, and wavelengths to be between 0 and 1
    sequence[:, :, 1:nfiltsp1] = (sequence[:, :, 1:nfiltsp1] - bandmin) \
        / (bandmax - bandmin)
    sequence[:, :, nfiltsp1:nfiltsp2] = (sequence[:, :, nfiltsp1:nfiltsp2]) \
        / (bandmax - bandmin)
    sequence[:, :, nfiltsp2:-1] = (sequence[:, :, nfiltsp2:-1] - wavemin) \
        / (wavemax - wavemin)
    sequence[:, :, -1] = sequence[:, :, -1] / np.max(sequence[:, :, -1])

    new_lms = np.reshape(np.repeat(lms, sequence_len), (len(lms), -1))
    new_lms = (-1 * new_lms - bandmin) / (bandmax - bandmin) 

    outseq = np.reshape(sequence[:, :, 0], (len(sequence), sequence_len, 1)) * 1.0
    outseq = np.dstack((outseq, new_lms))
    # tile for each wv
    outseq_tiled = np.repeat(outseq, 6, axis=1)
    outseq_wvs = np.reshape(sequence[:, :, nfiltsp2:-1], (len(sequence), nfilts*sequence_len, 1)) * 1.0
    outseq_tiled = np.dstack((outseq_tiled, outseq_wvs))
        
    if save:
        model_prep_file = os.path.join(outdir,'prep_'+date+'.npz')
        np.savez(model_prep_file, wavemin=wavemin, wavemax=wavemax, bandmin=bandmin, bandmax=bandmax)
        model_prep_file = os.path.join(outdir,'prep.npz')
        np.savez(model_prep_file, wavemin=wavemin, wavemax=wavemax, bandmin=bandmin, bandmax=bandmax)

    return np.array(sequence, dtype=np.float32), np.array(outseq_tiled, dtype=np.float32), ids, sequence_len, nfilts
