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

def mag_to_abs_mag(m, z):
    k_correction = 2.5 * np.log10(1.+z)
    dist = cosmo.luminosity_distance([z]).value[0]  # returns dist in Mpc
    abs_m = m - 5.0 * np.log10(dist * 1e6 / 10.0) + k_correction
    return abs_m

def flux_to_abs_mag(f, f_err, zp, z):
    k_correction = 2.5 * np.log10(1.+z)
    dist = cosmo.luminosity_distance([z]).value[0]  # returns dist in Mpc
    m_err = 2.5 * f_err / (f * np.log(10))
    m = -2.5 * np.log10(f) + zp - 5. * np.log10(dist*1e6/10.0) + k_correction
    return m, m_err

class LightCurve(object):
    """Light Curve class
    """
    def __init__(self, name, times, fluxes, flux_errs, filters,
                 zpt=0, mwebv=0, redshift=None, lim_mag_dict=None,
                 obj_type=None):

        self.name = name
        self.times = times
        self.fluxes = fluxes
        self.flux_errs = flux_errs
        self.filters = filters
        self.zpt = zpt
        self.mwebv = mwebv
        self.redshift = redshift
        self.lim_mag_dict = lim_mag_dict
        self.obj_type = obj_type

        self.abs_mags = None
        self.abs_mags_err = None
        self.abs_lim_mags = None
        self.ordered_abs_lim_mags = None


    def sort_lc(self):
        gind = np.argsort(self.times)
        self.times = self.times[gind]
        self.fluxes = self.fluxes[gind]
        self.flux_errs = self.flux_errs[gind]
        self.filters = self.filters[gind]
        if self.abs_mags is not None:
            self.abs_mags = self.abs_mags[gind]
            self.abs_mags_err = self.abs_mags_err[gind]
            self.abs_lim_mags = self.abs_lim_mags[gind]

    def find_peak(self):
        tpeaks = []
        for f in np.unique(self.filters):
            tpeaks.append(
                self.times[self.filters==f][
                    np.argmin(self.abs_mags[self.filters==f])
                ]
            )
        return np.median(tpeaks)

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
            self.abs_lim_mags = self.abs_lim_mags[gind]

    def shift_lc(self, t0=0):
        self.times = self.times - t0
     
    def truncate(self, maxt):
        gind = self.times <= maxt
        
        self.times = self.times[gind]
        self.fluxes = self.fluxes[gind]
        self.flux_errs = self.flux_errs[gind]
        self.filters = self.filters[gind]
        if self.abs_mags is not None:
            self.abs_mags = self.abs_mags[gind]
            self.abs_mags_err = self.abs_mags_err[gind]
            self.abs_lim_mags = self.abs_lim_mags[gind]
        
    def subsample(self, n_points):
        if n_points >= len(self.times):
            return None
        gind = np.random.choice(
            np.arange(len(self.times)),
            size=n_points,
            replace=False
        )
        self.times = self.times[gind]
        self.fluxes = self.fluxes[gind]
        self.flux_errs = self.flux_errs[gind]
        self.filters = self.filters[gind]
        if self.abs_mags is not None:
            self.abs_mags = self.abs_mags[gind]
            self.abs_mags_err = self.abs_mags_err[gind]
            self.abs_lim_mags = self.abs_lim_mags[gind]
        

    def correct_time_dilation(self):
        self.times = self.times / (1.+self.redshift)

    def correct_extinction(self, wvs):
        alams = extinction.fm07(wvs, self.mwebv)
        for i, alam in enumerate(alams):
            gind = np.where(self.filters == str(i))
            self.abs_mags[gind] = self.abs_mags[gind] - alam

    def add_LC_info(
        self, lim_mag_dict, zpt=27.5,
        mwebv=0.0, redshift=0.0, obj_type='-'
    ):
        self.lim_mag_dict = lim_mag_dict
        self.zpt = zpt
        self.mwebv = mwebv
        self.redshift = redshift
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
        self.abs_mags, self.abs_mags_err = flux_to_abs_mag(
            self.fluxes, self.flux_errs, self.zpt, self.redshift
        )
        self.abs_lim_mags = np.array([
            mag_to_abs_mag(self.lim_mag_dict[f], self.redshift) for f in self.filters
        ])

        if replace_nondetections:
            
            gind = np.where((np.isnan(self.abs_mags)) |
                            np.isinf(self.abs_mags) |
                            np.isnan(self.abs_mags_err) |
                            np.isinf(self.abs_mags_err) |
                            (self.abs_mags > self.abs_lim_mags))

            self.abs_mags[gind] = self.abs_lim_mags[gind]
            self.abs_mags_err[gind] = mag_err_fill

        return self.abs_mags, self.abs_mags_err

    def filter_names_to_numbers(self, filt_list):
        tmp = np.zeros(len(self.filters))
        self.ordered_abs_lim_mags = np.zeros(len(filt_list))
        
        for i, filt in enumerate(filt_list):
            tmp[self.filters == filt] = i
            self.ordered_abs_lim_mags[i] = mag_to_abs_mag(self.lim_mag_dict[filt], self.redshift)
        self.filters = tmp.astype(int)

    def make_dense_LC(self, nfilts):
        gp_mags = self.abs_mags - self.abs_lim_mags
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
        
        for jj in np.arange(nfilts):
            x_pred[jj::nfilts, 0] = self.times
            x_pred[jj::nfilts, 1] = jj
        
        pred, pred_var = gp.predict(gp_mags, x_pred, return_var=True)

        for jj in np.arange(nfilts):
            gind = np.where(x_pred[:, 1] == jj)[0]
            dense_fluxes[:, jj] = pred[gind] + self.ordered_abs_lim_mags[jj]
            dense_errs[:, jj] = np.sqrt(pred_var[gind])
        
        masked_vals = np.zeros(dense_fluxes.shape)
        
        # fill in true values
        for jj in np.arange(nfilts):
            t_filter = self.filters == jj
            dense_fluxes[t_filter, jj] = self.abs_mags[t_filter]
            dense_errs[t_filter, jj] = self.abs_mags_err[t_filter]
            masked_vals[~t_filter, int(jj)] = 1
            # fix broken errors - usually if no points in that band
            mask = dense_errs[:, int(jj)] <= 0.0
            dense_errs[:, int(jj)][mask] = 1.0
            
        # remove redundant time stamps
        eps = 4e-2 # times within one hour considered identical
        t_diffs = np.diff(self.times)
        repeat_idxs = (np.abs(t_diffs) < eps)
        repeat_idxs = np.insert(repeat_idxs, 0, False)
        times = self.times[~repeat_idxs]
        keep_idxs = np.argwhere(~repeat_idxs)
        
        dense_f_condensed = np.zeros((len(times), nfilts))
        dense_err_condensed = np.zeros((len(times), nfilts))
        masked_vals_condensed = np.zeros((len(times), nfilts))
        
        for i in range(len(times)):
            try:
                repeat_idx_subset = np.arange(keep_idxs[i], keep_idxs[i+1])
            except:
                repeat_idx_subset = np.arange(keep_idxs[i], len(self.times))
            # check if observed fluxes in the interval. If so, only average within those.
            # otherwise, average all GP-interpolated values

            for jj in np.arange(nfilts):
                masked_idxs = repeat_idx_subset[masked_vals[repeat_idx_subset, int(jj)] == 0]
                if len(masked_idxs) == 0:
                    masked_idxs = repeat_idx_subset
                    masked_vals_condensed[i, int(jj)] = 1
                else:
                    masked_vals_condensed[i, int(jj)] = 0
                    
                weights = 1. / dense_errs[masked_idxs, int(jj)]**2
                dense_f_condensed[i, int(jj)] = np.average(
                    dense_fluxes[masked_idxs, int(jj)],
                    weights = weights
                )
                new_var = np.var(dense_fluxes[masked_idxs, int(jj)])
                new_var += 1. / np.sum(weights)
                dense_err_condensed[i, int(jj)] = np.sqrt(new_var)
                if dense_err_condensed[i, int(jj)] <= 0.0:
                    dense_err_condensed[i, int(jj)] = 1.0

        self.dense_times = times
        self.dense_lc = np.dstack((dense_f_condensed, dense_err_condensed))
        
        self.gp = gp
        self.gp_mags = gp_mags
        self.masked_vals = masked_vals_condensed
        return gp, gp_mags
    
    def tile(self):
        N = self.dense_lc.shape[1]
        
        # pad dense LC to six bands
        if N <= 3:
            tile_factor = int(6 / N)
            self.dense_lc = np.repeat(
                self.dense_lc, tile_factor,
                axis=1
            )
            self.masked_vals = np.repeat(
                self.masked_vals, tile_factor,
                axis=1
            )
            self.wavelengths = np.repeat(
                self.wavelengths, tile_factor,
            )
            self.filt_widths = np.repeat(
                self.filt_widths, tile_factor,
            )
            self.ordered_abs_lim_mags = np.repeat(
                self.ordered_abs_lim_mags, tile_factor,
            )

        elif N < 6:
            self.dense_lc = np.repeat(
                self.dense_lc, 2,
                axis=1
            )[:,:6,:]
            self.masked_vals = np.repeat(
                self.masked_vals, 2,
                axis=1
            )[:,:6]
            self.wavelengths = np.repeat(
                self.wavelengths, 2
            )[:6]
            self.filt_widths = np.repeat(
                self.filt_widths, 2
            )[:6]
            self.ordered_abs_lim_mags = np.repeat(
                self.ordered_abs_lim_mags, 2
            )[:6]

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
            
        for j, lc in enumerate(transient.lightcurves):
            if lc.survey.name in ['Swift', '2MASS']:
                continue
            t, m, merr, b = lc.get_arrays()
            
            if len(t) < 5:
                #print("SKIPPED")
                continue
                
            f, ferr = convert_mag_to_flux(
                m, merr, DEFAULT_ZPT
            )
            
            sr_lc = LightCurve(
                name=int(0.5*(i+j)*(i+j+1)+j), # cantor pairing function (fancy math)
                times=t,
                fluxes=f,
                flux_errs=ferr,
                filters=b,
            )
            
            sr_lc.add_LC_info(
                zpt=DEFAULT_ZPT,
                mwebv=0.0,
                redshift=params['redshift'],
                lim_mag_dict=lc.survey.lim_mag_dict,
                obj_type=transient.model_type
            )
            
            sr_lc.group = transient_id
            
            npoints = np.random.choice(
                np.arange(5, 50)
            )
            
            sr_lc.sort_lc()
            sr_lc.get_abs_mags()
            pmjd = sr_lc.find_peak()
            sr_lc.shift_lc(pmjd)
            sr_lc.truncate(np.random.uniform(low=0, high=100))
            sr_lc.subsample(npoints)
            
            # re-sort/center
            sr_lc.sort_lc()
            pmjd = sr_lc.find_peak()
            sr_lc.shift_lc(pmjd)
            
            sr_lc.correct_time_dilation()

            filt_list = np.unique(sr_lc.filters)
            sr_lc.wavelengths = np.zeros(len(filt_list))
            sr_lc.filt_widths = np.zeros(len(filt_list))
            
            sr_lc.filter_names_to_numbers(filt_list)
            for j, f in enumerate(filt_list):
                sr_lc.wavelengths[j] = lc.survey.band_wavelengths[f]
                sr_lc.filt_widths[j] = lc.survey.band_widths[f]

            sr_lc.cut_lc()
            
            if len(np.unique(sr_lc.times)) < 5:
                continue
            if len(np.unique(sr_lc.filters)) < 2:
                continue
            try:
                sr_lc.make_dense_LC(len(filt_list))
                
                if len(np.unique(sr_lc.dense_times)) < 5:
                    continue
            except:
                print("SKIPPED")
                continue
                
            sr_lc.tile()
            sr_lcs.append(sr_lc)
            
        transient_id += 1
            
    save_lcs(sr_lcs, save_dir)
    
    
def calc_outseq(sequence, lms, bandmin=None, bandmax=None):
    """Calculates input for decoder from sequence and limiting magnitudes.
    """
    sequence_len = sequence.shape[1]
    nfilts = int((sequence.shape[2] - 2) / 4 )
    nfiltsp2 = 2*nfilts+1
    nfiltsp3 = 3*nfilts+1
    
    new_lms = np.array(lms)[:,np.newaxis,:] * np.ones((len(sequence), sequence_len, 6))
    new_lms = new_lms.reshape((len(sequence),nfilts*sequence_len, 1))
    
    if (bandmin is not None) and (bandmax is not None):
        new_lms = (-1 * new_lms - bandmin) / (bandmax - bandmin) 
    
    outseq = np.reshape(sequence[:, :, 0], (len(sequence), sequence_len, 1)) * 1.0
    # tile for each wv
    outseq_tiled = np.repeat(outseq, 6, axis=1)
    outseq_wvs = np.reshape(sequence[:, :, nfiltsp2:nfiltsp3], (len(sequence), nfilts*sequence_len, 1)) * 1.0
    outseq_filter_wvs = np.reshape(sequence[:, :, nfiltsp3:-1], (len(sequence), nfilts*sequence_len, 1)) * 1.0
    outseq_tiled = np.dstack((outseq_tiled, new_lms, outseq_wvs, outseq_filter_wvs))
    return outseq_tiled
    
    
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
        lengths.append(len(lightcurve.dense_lc))
        ids.append(lightcurve.name)

    ids = np.array(ids)
    
    sequence_len = 32 #min(200, np.max(lengths))
    print(sequence_len)
    lengths = np.clip(lengths, a_min=0, a_max=sequence_len).astype(int)
    
    nfilts = np.shape(lightcurves[0].dense_lc)[1]
    nfiltsp1 = nfilts+1
    nfiltsp2 = 2*nfilts+1
    nfiltsp3 = 3*nfilts+1

    n_lcs = len(lightcurves)
    # convert from LC format to list of arrays
    # sequence = np.zeros((n_lcs, sequence_len, nfilts*2+1))
    sequence = np.zeros((n_lcs, sequence_len, nfilts*4 + 2))
    loss_mask = np.zeros((n_lcs, sequence_len, nfilts))

    lms = []
    for i, lightcurve in enumerate(lightcurves):
        sequence[i, 0:lengths[i], 0] = lightcurve.dense_times[:lengths[i]] / 1000. # to keep values small
        sequence[i, 0:lengths[i], 1:nfiltsp1] = lightcurve.dense_lc[:lengths[i], :, 0] # fluxes
        sequence[i, 0:lengths[i], nfiltsp1:nfiltsp2] = lightcurve.dense_lc[:lengths[i], :, 1] # flux errors
        sequence[i, lengths[i]:, 0] = (np.max(lightcurve.times)+new_t_max) / 1000.
        sequence[i, lengths[i]:, 1:nfiltsp1] = lightcurve.ordered_abs_lim_mags[np.newaxis,:]
        sequence[i, lengths[i]:, nfiltsp1:nfiltsp2] = filler_err
        sequence[i, :, nfiltsp2:nfiltsp3] = lightcurve.wavelengths
        sequence[i, :, nfiltsp3:-1] = lightcurve.filt_widths
        
        loss_mask[i, 0:lengths[i]] = lightcurve.masked_vals[:sequence_len]
        loss_mask[i, lengths[i]:] = 1

        # shuffle filters
        shuffled_idx = np.random.permutation(nfilts)[np.newaxis, :]
        sequence[i, :, 1:nfiltsp1] = np.take_along_axis(
            sequence[i, :, 1:nfiltsp1], shuffled_idx, axis=1
        )
        sequence[i, :, nfiltsp1:nfiltsp2] = np.take_along_axis(
            sequence[i, :, nfiltsp1:nfiltsp2], shuffled_idx, axis=1
        )
        sequence[i, :, nfiltsp2:nfiltsp3] = np.take_along_axis(
            sequence[i, :, nfiltsp2:nfiltsp3], shuffled_idx, axis=1
        )
        sequence[i, :, nfiltsp3:-1] = np.take_along_axis(
            sequence[i, :, nfiltsp3:-1], shuffled_idx, axis=1
        )
        loss_mask[i, :, :] = np.take_along_axis(
            loss_mask[i, :, :], shuffled_idx, axis=1
        )

        amp = np.max(sequence[i, :, 1:nfiltsp1]) - np.min(sequence[i, :, 1:nfiltsp1])
        mask = sequence[i, :, nfiltsp1:nfiltsp2] < 0.01 * amp
        sequence[i, :, nfiltsp1:nfiltsp2][mask] = 0.01 * amp
        
        sequence[i, :, -1] = lightcurve.group
        
        lms.append(
            lightcurve.ordered_abs_lim_mags[shuffled_idx[0]]
        )
        
    # Flip because who needs negative magnitudes
    sequence[:, :, 1:nfiltsp1] = -1.0 * sequence[:, :, 1:nfiltsp1]
    
    """
    # do some sigma clipping before bandmin calculation
    max_mags = np.max(sequence[:, :, 1:nfiltsp1], axis=(1,2))
    bright_mag_limit = np.median(max_mags) + 2. * np.std(max_mags)
    dim_mag_limit =  np.median(max_mags) - 2. * np.std(max_mags)
    mask = (max_mags < bright_mag_limit) & (max_mags > dim_mag_limit)
    sequence = sequence[mask]
    lms = np.array(lms)[mask]
    loss_mask = loss_mask[mask]
    ids = ids[mask]
    """
    #print(len(sequence))
    
    if load:
        prep_data = np.load(prep_file)
        bandmin = prep_data['bandmin']
        bandmax = prep_data['bandmax']
        wavemin = prep_data['wavemin']
        wavemax = prep_data['wavemax']
    else:
        bandmin = np.min(sequence[:, :, 1:nfiltsp1])
        bandmax = np.max(sequence[:, :, 1:nfiltsp1])
        wavemin = np.min(sequence[:, :, nfiltsp2:nfiltsp3])
        wavemax = np.max(sequence[:, :, nfiltsp2:nfiltsp3])

    
    # Normalize flux values, flux errors, and wavelengths to be between 0 and 1
    sequence[:, :, 1:nfiltsp1] = (sequence[:, :, 1:nfiltsp1] - bandmin) \
        / (bandmax - bandmin)
    sequence[:, :, nfiltsp1:nfiltsp2] = (sequence[:, :, nfiltsp1:nfiltsp2]) \
        / (bandmax - bandmin)
    sequence[:, :, nfiltsp2:nfiltsp3] = (sequence[:, :, nfiltsp2:nfiltsp3] - wavemin) \
        / (wavemax - wavemin)
    sequence[:, :, nfiltsp3:-1] = (sequence[:, :, nfiltsp3:-1]) \
        / (wavemax - wavemin)
    sequence[:, :, -1] = sequence[:, :, -1] / np.max(sequence[:, :, -1])
    
    mask = sequence[:, :, nfiltsp1:nfiltsp2] < 0.001
    sequence[:, :, nfiltsp1:nfiltsp2][mask] = 0.001
    
    outseq_tiled = calc_outseq(sequence, lms, bandmin, bandmax)

    if save:
        model_prep_file = os.path.join(outdir,'prep_'+date+'.npz')
        np.savez(model_prep_file, wavemin=wavemin, wavemax=wavemax, bandmin=bandmin, bandmax=bandmax)
        model_prep_file = os.path.join(outdir,'prep.npz')
        np.savez(model_prep_file, wavemin=wavemin, wavemax=wavemax, bandmin=bandmin, bandmax=bandmax)

    return (
        np.array(sequence, dtype=np.float32),
        np.array(outseq_tiled, dtype=np.float32),
        np.array(loss_mask, dtype=np.float32),
        ids, sequence_len, nfilts
    )
