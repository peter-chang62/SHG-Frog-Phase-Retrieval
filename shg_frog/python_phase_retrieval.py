import numpy as np
import scipy.constants as sc
import scipy.integrate as scint
import matplotlib.pyplot as plt
from shg_frog import BBO
import scipy.interpolate as spi
import copy
import scipy.optimize as spo
from shg_frog.utility.fft import fft, ifft, shift
from shg_frog.light import Pulse


def normalize(x):
    """
    normalize a vector

    Args:
        x (ndarray):
            data to be normalized

    Returns:
        ndarray:
            normalized data
    """

    return x / np.max(abs(x))


def denoise(x, gamma):
    """
    denoise x with threshold gamma

    Args:
        x (ndarray):
            data to be denoised
        gamma (float):
            threshold value
    Returns:
        ndarray:
            denoised data

    Notes:
        The condition is abs(x) >= gamma, and returns:
        x.real - gamma * sgn(x.real) + j(x.imag - gamma * sgn(x.imag))
    """
    return np.where(
        abs(x) >= gamma, x.real - gamma * np.sign(x.real), 0
    ) + 1j * np.where(abs(x) >= gamma, x.imag - gamma * np.sign(x.imag), 0)


def load_data(path):
    """
    loads the spectrogram data

    Args:
        path (string):
            path to the FROG data

    Returns:
        wl_nm (1D array):
            wavelength axis in nanometers
        F_THz (1D array):
            frequency axis in THz
        T_fs (1D array):
            time delay axis in femtoseconds
        spectrogram (2D array):
            the spectrogram with time indexing the row, and wavelength indexing
            the column

    Notes:
        this function extracts relevant variables from the spectrogram data:
            1. time axis
            2. wavelength axis
            3. frequency axis
        no alteration to the data is made besides truncation along the time
        axis to center T0
    """
    spectrogram = np.genfromtxt(path)
    T_fs = spectrogram[:, 0][1:]  # time indexes the row
    wl_nm = spectrogram[0][1:]  # wavelength indexes the column
    F_THz = sc.c * 1e-12 / (wl_nm * 1e-9)  # experimental frequency axis from wl_nm
    spectrogram = spectrogram[1:, 1:]

    # center T0
    x = scint.simps(spectrogram, axis=1)
    ind = np.argmax(x)
    ind_keep = min([ind, len(spectrogram) - ind])
    spectrogram = spectrogram[ind - ind_keep : ind + ind_keep]
    T_fs -= T_fs[ind]
    T_fs = T_fs[ind - ind_keep : ind + ind_keep]

    # wavelength -> frequency
    factor = sc.c / (F_THz * 1e12) ** 2
    factor /= factor.max()
    spectrogram *= factor

    return wl_nm, F_THz, T_fs, normalize(spectrogram)


def func(gamma, args):
    """
    function that is optimized to calculate the error at each retrieval
    iteration

    Args:
        gamma (float):
            scaling factor to multiply the experimental spectrogram
        args (tuple):
            a tuple of: spectrogram, experimental spectrogram (to compare to
            spectrogram). Technically it would matter if their order were
            reversed

    Returns:
        float:
            The calculated error given as the root mean squared of the
            difference between spectrogram and experimental spectrogram
    """
    spctgm, spctgm_exp = args
    return np.sqrt(np.mean(abs(normalize(spctgm) - gamma * normalize(spctgm_exp)) ** 2))


class Retrieval:
    def __init__(self):
        self._wl_nm = None
        self._F_THz = None
        self._T_fs = None
        self._spectrogram = None
        self._min_pm_fthz = None
        self._max_sig_fthz = None
        self._max_pm_fthz = None
        self._pulse = None
        self._pulse_data = None
        self._spectrogram_interp = None
        self._ind_ret = None
        self._error = None
        self._AT2D = None

    # --------------------------------- variables to keep track of ------------

    @property
    def wl_nm(self):
        assert isinstance(
            self._wl_nm, np.ndarray
        ), "no spectrogram data has been loaded yet"
        return self._wl_nm

    @property
    def F_THz(self):
        assert isinstance(
            self._F_THz, np.ndarray
        ), "no spectrogram data has been loaded yet"
        return self._F_THz

    @property
    def T_fs(self):
        assert isinstance(
            self._T_fs, np.ndarray
        ), "no spectrogram data has been loaded yet"
        return self._T_fs

    @property
    def spectrogram(self):
        assert isinstance(
            self._spectrogram, np.ndarray
        ), "no spectrogram data has been loaded yet"
        return self._spectrogram

    @property
    def min_sig_fthz(self):
        assert isinstance(
            self._min_sig_fthz, float
        ), "no signal frequency range has been set yet"
        return self._min_sig_fthz

    @property
    def max_sig_fthz(self):
        assert isinstance(
            self._max_sig_fthz, float
        ), "no signal frequency range has been set yet"
        return self._max_sig_fthz

    @property
    def min_pm_fthz(self):
        assert isinstance(
            self._min_pm_fthz, float
        ), "no phase matching bandwidth has been defined yet"
        return self._min_pm_fthz

    @property
    def max_pm_fthz(self):
        assert isinstance(
            self._max_pm_fthz, float
        ), "no phase matching bandwidth has been defined yet"
        return self._max_pm_fthz

    @property
    def ind_ret(self):
        assert isinstance(
            self._ind_ret, np.ndarray
        ), "no phase matching bandwidth has been defined yet"
        return self._ind_ret

    @property
    def pulse(self):
        assert isinstance(self._pulse, Pulse), "no initial guess has been set yet"
        return self._pulse

    @property
    def pulse_data(self):
        assert isinstance(
            self._pulse_data, Pulse
        ), "no spectrum data has been loaded yet"
        return self._pulse_data

    @property
    def error(self):
        assert isinstance(self._error, np.ndarray), "no retrieval has been run yet"
        return self._error

    @property
    def AT2D(self):
        assert isinstance(self._AT2D, np.ndarray), "no retrieval has been run yet"
        return self._AT2D

    @property
    def spectrogram_interp(self):
        assert isinstance(
            self._spectrogram_interp, np.ndarray
        ), "spectrogram has not been interpolated to the simulation grid"
        return self._spectrogram_interp

    # _______________________ functions _______________________________________

    def load_data(self, path):
        """
        load the data

        Args:
            path (string):
                path to data
        """
        self._wl_nm, self._F_THz, self._T_fs, self._spectrogram = load_data(path)

    def set_signal_freq(self, min_sig_fthz, max_sig_fthz):
        """
        this function is used to denoise the spectrogram before retrieval

        Args:
            min_sig_fthz (float):
                minimum signal frequency
            max_sig_fthz (float):
                maximum signal frequency

        Notes:
            sets the minimum and maximum signal frequency and then denoises the
            parts of the spectrogram that is outside this frequency range,
            this is used purely for calling denoise on the spectrogram, and
            does not set the frequency range to be used for retrieval (that is
            instead set by the phase matching bandwidth)
        """

        self._min_sig_fthz, self._max_sig_fthz = float(min_sig_fthz), float(
            max_sig_fthz
        )
        self.denoise_spectrogram()

    def _get_ind_fthz_nosig(self):
        """
        a convenience function used for denoise_spectrogram()

        Returns:
            1D array of integers:
                indices of the experimental wavelength axis that falls outside
                the signal frequency range (the one used to denoise the
                spectrogram)

        Notes:
                This gets an array of indices for the experimental wavelength
                axis that falls outside the signal frequency range (the one
                that is used to denoise the spectrogram). This can only be
                called after min_sig_fthz and max_sig_fthz have been set by
                set_signal_freq
        """

        mask_fthz_sig = np.logical_and(
            self.F_THz >= self.min_sig_fthz, self.F_THz <= self.max_sig_fthz
        )

        ind_nosig = np.ones(len(self.F_THz))
        ind_nosig[mask_fthz_sig] = 0
        ind_nosig = ind_nosig.nonzero()[0]

        return ind_nosig

    def denoise_spectrogram(self):
        """
        denoise the spectrogram using min_sig_fthz and max_sig_fthz
        """
        self.spectrogram[:] = normalize(self.spectrogram)

        ind_nosig = self._get_ind_fthz_nosig()
        self.spectrogram[:, ind_nosig] = denoise(
            self.spectrogram[:, ind_nosig], 1e-3
        ).real

    def correct_for_phase_matching(self, deg=3.576):
        """
        correct for phase-matching

        Args:
            deg (float, optional):
                non-collinear angle incident into the BBO, default is 3.576
                which is a 1/4" separation of the two beams (most updated FROG
                build)

        Notes:
            the spectrogram is divided by the phase-matching curve, and then
            denoised, so this can only be called after calling
            set_signal_freq
        """

        bbo = BBO.BBOSHG()
        R = bbo.R(
            self.wl_nm * 1e-3 * 2,
            50,
            bbo.phase_match_angle_rad(1.55),
            BBO.deg_to_rad(deg),
        )

        # avoid the zero's of R, I just leverage the fact that the spectrum
        # doesn't usually extend past the first zero on the short-wavelength
        # side
        ind_1perc = np.argmin(abs(R[235:] - 0.01)) + 235

        self.spectrogram[:, ind_1perc:] /= R[ind_1perc:]
        self.denoise_spectrogram()

        self._min_pm_fthz = min(self.F_THz)
        self._max_pm_fthz = self.F_THz[ind_1perc]

    def set_initial_guess(
        self,
        wl_min_nm=1000.0,
        wl_max_nm=2000.0,
        center_wavelength_nm=1560,
        time_window_ps=10,
        NPTS=4096,
    ):
        """
        Args:
            wl_min_nm (float, optional):
                minimum wavlength, default is 1 um
            wl_max_nm (float, optional):
                maximum wavelength, default is 2 um
            center_wavelength_nm (float, optional):
                center wavelength in nanometers, default is 1560
            time_window_ps (int, optional):
                time window in picoseconds, default is 10. This sets the size
                of the time grid
            NPTS (int, optional):
                number of points on the time and frequency grid, default is
                2 ** 12 = 4096

        Notes:
            This initializes a pulse with a sech envelope, whose
            time bandwidth is set according to the intensity autocorrelation
            of the spectrogram. Realize that the spectrogram could have been
            slightly altered depending on whether it's been denoised(called by
            either set_signal_freq or correct_for_phase_matching, but this
            should not influence the time bandwidth significantly)
        """

        # integrate experimental spectrogram across wavelength axis
        x = -scint.simpson(self.spectrogram, x=self.F_THz, axis=1)

        spl = spi.UnivariateSpline(self.T_fs, normalize(x) - 0.5, s=0)
        roots = spl.roots()

        T0 = np.diff(roots[[0, -1]])
        self._pulse = Pulse.Sech(
            NPTS,
            sc.c / (wl_max_nm * 1e-9),
            sc.c / (wl_min_nm * 1e-9),
            sc.c / (center_wavelength_nm * 1e-9),
            1.0e-9,
            T0 * 1e-15,
            time_window_ps * 1e-12,
        )
        phase = np.random.uniform(low=0, high=1, size=self.pulse.n) * np.pi / 8
        self._pulse.a_t = self._pulse.a_t * np.exp(1j * phase)

    def load_spectrum_data(self, wl_um, spectrum):
        """
        Args:
            wl_um (1D array):
                wavelength axis in micron
            spectrum (1D array):
                power spectrum

        Notes:
            This can only be called after having already called
            set_initial_guess. It clones the original pulse and sets the
            envelope in the frequency domain to the transform limited pulse
            calculated from the power spectrum

        """

        # when converting dB to linear scale for data taken by the
        # monochromator, sometimes you get negative values at wavelengths
        # where you have no (or very little) power (experimental error)
        assert np.all(spectrum >= 0), "a negative spectrum is not physical"

        pulse_data: Pulse
        pulse_data = copy.deepcopy(self.pulse)
        p_v_callable = spi.interp1d(
            wl_um, spectrum, kind="linear", bounds_error=False, fill_value=0.0
        )
        p_v = p_v_callable(pulse_data.wl_grid * 1e6)

        # psd in wavelength -> psd in frequency
        factor = sc.c / pulse_data.v_grid**2
        factor /= factor.max()  # normalize
        p_v *= factor

        pulse_data.a_v = p_v**0.5  # phase = 0
        self._pulse_data = pulse_data

    def _intrplt_spctrgrm_to_sim_grid(self):
        """
        This interpolates the spectrogram to the simulation grid. This can only
        be called after calling set_initial_guess and
        correct_for_phase_matching because the simulation grid is defined by
        the pulse's frequency grid, and the interpolation range is narrowed
        down to the phase-matching bandwidth
        """

        gridded = spi.interp2d(
            self.F_THz, self.T_fs, self.spectrogram, bounds_error=True
        )
        # the input goes as column coord, row coord, 2D data
        # so time is the row index, and wavelength is the column index
        spectrogram_interp = gridded(
            self.pulse.v_grid[self.ind_ret] * 1e-12 * 2, self.T_fs
        )

        # scale the interpolated spectrogram to match the pulse energy. I do
        # it here instead of to the experimental spectrogram, because the
        # interpolated spectrogram has the same integration frequency axis
        # as the pulse instance
        x = self.pulse.spectrogram_shg(self.T_fs * 1e-15)
        factor = scint.simpson(scint.simpson(x[:, self.ind_ret])) / scint.simpson(
            scint.simpson(spectrogram_interp)
        )
        spectrogram_interp *= factor
        self._spectrogram_interp = spectrogram_interp

    def retrieve(self, start_time, end_time, itermax, iter_set=None, plot_update=True):
        """
        Args:
            start_time (float):
                start time for retrieval in femtoseconds
            end_time (float):
                end time for retrieval in femtoseconds
            itermax (int):
                number of iterations to use
            iter_set (int, optional):
                iteration at which to set the power spectrum to the
                experimentally measured one, default is None which disables
                this functionality
            plot_update (bool, optional):
                whether to update a plot after each iteration
        """

        assert (iter_set is None) or (
            isinstance(self.pulse_data, Pulse) and isinstance(iter_set, int)
        )

        # self._ind_ret = np.logical_and(
        #     self.pulse.v_grid * 1e-12 * 2 >= self.min_pm_fthz,
        #     self.pulse.v_grid * 1e-12 * 2 <= self.max_pm_fthz,
        # ).nonzero()[0]

        # I use self.ind_ret to set the retrieval's frequency bandwidth.
        # Previously I set the retrieval's frequency bandwidth to the
        # phase-matching bandwidth, but now I want to set it to the signal
        # frequency bandwidth.
        self._ind_ret = np.logical_and(
            self.pulse.v_grid * 1e-12 * 2 >= self.min_sig_fthz,
            self.pulse.v_grid * 1e-12 * 2 <= self.max_sig_fthz,
        ).nonzero()[0]

        self._intrplt_spctrgrm_to_sim_grid()

        ind_start = np.argmin(abs(self.T_fs - start_time))
        ind_end = np.argmin(abs(self.T_fs - end_time))
        delay_time = self.T_fs[ind_start:ind_end] * 1e-15  # mks units
        time_order = np.c_[delay_time, np.arange(ind_start, ind_end)]

        j_excl = np.ones(len(self.pulse.v_grid))
        j_excl[self.ind_ret] = 0
        j_excl = j_excl.nonzero()[0]  # everything but ind_ret

        error = np.zeros(itermax)
        rng = np.random.default_rng()

        AT = np.zeros((itermax, len(self.pulse.a_t)), dtype=np.complex128)

        if plot_update:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax3 = ax2.twinx()

        for itr in range(itermax):
            rng.shuffle(time_order, axis=0)
            alpha = abs(0.2 + rng.standard_normal(1) / 20)
            for dt, j in time_order:
                j = int(j)

                AT_shift = shift(
                    self.pulse.a_t,
                    self.pulse.v_grid - self.pulse.v0,
                    dt,
                    fsc=self.pulse.dt,
                )
                psi_j = AT_shift * self.pulse.a_t
                phi_j = fft(psi_j, fsc=self.pulse.dt)

                amp = abs(phi_j)
                amp[self.ind_ret] = np.sqrt(self.spectrogram_interp[j])
                phase = np.angle(phi_j)
                phi_j[:] = amp * np.exp(1j * phase)

                # denoise everything that is not inside the wavelength range of
                # the spectrogram that is being used for retrieval.
                # Intuitively, this is all the frequencies that you don't
                # think the spectrogram gives reliable results for. The
                # threshold is the max of phi_j / 1000. Otherwise, depending
                # on what pulse energy you decided to run with during
                # retrieval, the 1e-3 threshold can do different things.
                # Intuitively, the threshold should be set close to the noise
                # floor, which is determined by the maximum.
                phi_j[j_excl] = denoise(phi_j[j_excl], 1e-3 * abs(phi_j).max())
                # phi_j[:] = denoise(phi_j[:], 1e-3 * abs(phi_j).max())  # or not

                psi_jp = ifft(phi_j, fsc=self.pulse.dt)
                corr1 = AT_shift.conj() * (psi_jp - psi_j) / np.max(abs(AT_shift) ** 2)
                corr2 = (
                    self.pulse.a_t.conj() * (psi_jp - psi_j) / np.max(self.pulse.p_t)
                )
                corr2 = shift(
                    corr2, self.pulse.v_grid - self.pulse.v0, -dt, fsc=self.pulse.dt
                )

                self.pulse.a_t = self.pulse.a_t + alpha * corr1 + alpha * corr2

                # _____________________________________________________________
                # substitution of power spectrum
                if iter_set is not None:
                    if itr >= iter_set:
                        phase = np.angle(self.pulse.a_v)
                        self.pulse.a_v = abs(self.pulse_data.a_v) * np.exp(1j * phase)
                # _____________________________________________________________
                # center T0
                ind = np.argmax(self.pulse.p_t)
                center = self.pulse.n // 2
                self.pulse.a_t = np.roll(self.pulse.a_t, center - ind)
                # _____________________________________________________________

            # _________________________________________________________________
            # preparing for substitution of power spectrum
            if iter_set is not None:
                if itr == iter_set - 1:  # the one before iter_set
                    self.pulse_data.e_p = self.pulse.e_p
            # _________________________________________________________________

            if plot_update:
                [ax.clear() for ax in [ax1, ax2, ax3]]
                ax1.plot(self.pulse.t_grid * 1e12, self.pulse.p_t)
                ax2.plot(self.pulse.v_grid * 1e-12, self.pulse.p_v)
                ax3.plot(
                    self.pulse.v_grid * 1e-12,
                    np.unwrap(np.angle(self.pulse.a_v)),
                    color="C1",
                )
                ax2.set_xlim(self.min_sig_fthz / 2, self.max_sig_fthz / 2)
                fig.suptitle(itr)
                plt.pause(0.1)

            s = self.pulse.spectrogram_shg(self.T_fs * 1e-15)[
                ind_start:ind_end, self.ind_ret
            ]
            # error[itr] = np.sqrt(np.sum(abs(s - self.spectrogram_interp) ** 2)) / np.sqrt(
            #     np.sum(abs(self.spectrogram_interp) ** 2))
            res = spo.minimize(
                func,
                np.array([1]),
                args=[s, self.spectrogram_interp[ind_start:ind_end]],
            )
            error[itr] = res.fun
            AT[itr] = self.pulse.a_t

            print(itr, error[itr])

        self._error = error
        self._AT2D = AT

    def plot_results(self, set_to_best=True):
        if set_to_best:
            self.pulse.a_t = self.AT2D[np.argmin(self.error)]

        fig, ax = plt.subplots(2, 2)
        ax = ax.flatten()

        # plot time domain
        ax[0].plot(self.pulse.t_grid * 1e12, self.pulse.p_t)

        # plot frequency domain
        ax[1].plot(self.pulse.v_grid * 1e-12, self.pulse.p_v)
        ax[1].set_xlim(self.min_sig_fthz / 2, self.max_sig_fthz / 2)

        # plot the phase on same plot as frequency domain
        axp = ax[1].twinx()
        ind_sig = np.logical_and(
            self.pulse.v_grid * 1e-12 * 2 >= self.min_sig_fthz,
            self.pulse.v_grid * 1e-12 * 2 <= self.max_sig_fthz,
        ).nonzero()[0]
        phase = BBO.rad_to_deg(np.unwrap(np.angle(self.pulse.a_v[ind_sig])))
        axp.plot(self.pulse.v_grid[ind_sig] * 1e-12, phase, color="C1")

        # plot the experimental spectrogram
        ax[2].pcolormesh(self.T_fs, self.F_THz / 2, self.spectrogram.T, cmap="CMRmap_r")
        ax[2].set_ylim(self.min_sig_fthz / 2, self.max_sig_fthz / 2)

        # plot the retrieved spectrogram
        s = self.pulse.spectrogram_shg(self.T_fs * 1e-15)
        ind_spctrmtr = np.logical_and(
            self.pulse.v_grid * 1e-12 * 2 >= min(self.F_THz),
            self.pulse.v_grid * 1e-12 * 2 <= max(self.F_THz),
        ).nonzero()[0]
        ax[3].pcolormesh(
            self.T_fs,
            self.pulse.v_grid[ind_spctrmtr] * 1e-12,
            s[:, ind_spctrmtr].T,
            cmap="CMRmap_r",
        )
        ax[3].set_ylim(self.min_sig_fthz / 2, self.max_sig_fthz / 2)

        # plot the experimental power spectrum
        if isinstance(self._pulse_data, Pulse):
            # res = spo.minimize(func, np.array([1]),
            #                    args=[abs(self.pulse.AW) ** 2, abs(self.pulse_data.AW) ** 2])
            # factor = res.x
            # factor = max(self.pulse.p_v) / max(self.pulse_data.p_v)
            factor = self.pulse.e_p / self.pulse_data.e_p
            ax[1].plot(
                self.pulse_data.v_grid * 1e-12,
                self.pulse_data.p_v * factor,
                color="C2",
            )

        return fig, ax
