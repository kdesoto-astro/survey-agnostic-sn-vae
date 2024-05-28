from astropy.cosmology import Planck13 as cosmo
import numpy as np
import scipy
import george
import extinction


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
