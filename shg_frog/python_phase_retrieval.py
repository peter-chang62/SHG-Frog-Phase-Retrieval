import numpy as np
import scipy.constants as sc
import scipy.integrate as scint
import matplotlib.pyplot as plt
from shg_frog import BBO
from scipy.fftpack import next_fast_len
import scipy.interpolate as spi
import copy
import scipy.optimize as spo

try:
    import mkl_fft

    use_mkl = True
except ImportError:

    class mkl_fft:
        """
        reproducing the following functions from mkl:
            fft, ifft, rfft_numpy and irfft_numpy
        """

        def fft(x, axis=-1, forward_scale=1.0):
            return np.fft.fft(x, axis=axis) * forward_scale

        def ifft(x, axis=-1, forward_scale=1.0):
            return np.fft.ifft(x, axis=axis) / forward_scale

        def rfft_numpy(x, axis=-1, forward_scale=1.0):
            return np.fft.rfft(x, axis=axis) * forward_scale

        def irfft_numpy(x, axis=-1, forward_scale=1.0):
            return np.fft.irfft(x, axis=axis) / forward_scale


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


def fft(x, axis=None, fsc=1.0):
    """
    perform fft of array x along specified axis

    Args:
        x (ndarray):
            array on which to perform fft
        axis (None, optional):
            None defaults to axis=-1, otherwise specify the axis. By default I
            throw an error if x is not 1D and axis is not specified
        fsc (float, optional):
            The forward transform scale factor. The default is 1.0.

    Returns:
        ndarray:
            fft of x
    """

    assert (len(x.shape) == 1) or isinstance(
        axis, int
    ), "if x is not 1D, you need to provide an axis along which to perform the fft"

    if axis is None:
        # default is axis=-1
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x), forward_scale=fsc))

    else:
        return np.fft.fftshift(
            mkl_fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis, forward_scale=fsc),
            axes=axis,
        )


def ifft(x, axis=None, fsc=1.0):
    """
    perform ifft of array x along specified axis

    Args:
        x (ndarray):
            array on which to perform ifft
        axis (None, optional):
            None defaults to axis=-1, otherwise specify the axis. By default I
            throw an error if x is not 1D and axis is not specified
        fsc (float, optional):
            The forward transform scale factor. The default is 1.0.

    Returns:
        ndarray:
            ifft of x
    """

    assert (len(x.shape) == 1) or isinstance(
        axis, int
    ), "if x is not 1D, you need to provide an axis along which to perform the fft"

    if axis is None:
        # default is axis=-1
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x), forward_scale=fsc))

    else:
        return np.fft.fftshift(
            mkl_fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis, forward_scale=fsc),
            axes=axis,
        )


def rfft(x, axis=None, fsc=1.0):
    """
    perform rfft of array x along specified axis

    Args:
        x (ndarray):
            array on which to perform rfft
        axis (None, optional):
            None defaults to axis=-1, otherwise specify the axis. By default I
            throw an error if x is not 1D and axis is not specified
        fsc (float, optional):
            The forward transform scale factor. The default is 1.0.

    Returns:
        ndarray:
            rfft of x

    Notes:
        rfft requires that you run ifftshift on the input, but the output does
        not require an fftshift, because the output array starts with the zero
        frequency component
    """

    assert (len(x.shape) == 1) or isinstance(
        axis, int
    ), "if x is not 1D, you need to provide an axis along which to perform the fft"

    if axis is None:
        # default is axis=-1
        return mkl_fft.rfft_numpy(np.fft.ifftshift(x), forwrard_scale=fsc)

    else:
        return mkl_fft.rfft_numpy(
            np.fft.ifftshift(x, axes=axis), axis=axis, forwrard_scale=fsc
        )


def irfft(x, axis=None, fsc=1.0):
    """
    perform irfft of array x along specified axis

    Args:
        x (ndarray):
            array on which to perform irfft
        axis (None, optional):
            None defaults to axis=-1, otherwise specify the axis. By default I
            throw an error if x is not 1D and axis is not specified
        fsc (float, optional):
            The forward transform scale factor. The default is 1.0.

    Returns:
        ndarray:
            irfft of x

    Notes:
        irfft does not require an ifftshift on the input since the output of
        rfft already has the zero frequency component at the start. However,
        to retriev the original ordering, you need to call fftshift on the
        output.
    """

    assert (len(x.shape) == 1) or isinstance(
        axis, int
    ), "if x is not 1D, you need to provide an axis along which to perform the fft"

    if axis is None:
        # default is axis=-1
        return np.fft.fftshift(mkl_fft.irfft_numpy(x, forward_scale=fsc))

    else:
        return np.fft.fftshift(
            mkl_fft.irfft_numpy(x, axis=axis, forward_scale=fsc), axes=axis
        )


def shift(x, freq, shift, fsc=1.0, freq_is_angular=False, x_is_real=False):
    """
    shift a 1D or 2D array

    Args:
        x (1D or 2D array):
            data to be shifted
        freq (1D array):
            frequency axis (units to be complementary to shift)
        shift (float or 1D array):
            float if x is a 1D array, otherwise needs to be an array, one shift
            for each row of x
        fsc (float, optional):
            The forward transform scale factor. The default is 1.0.
        freq_is_angular (bool, optional):
            is the freq provided angular frequency or not
        x_is_real (bool, optional):
            use real fft's or complex fft's, generally stick to complex if you
            want to be safe

    Returns:
        ndarray:
            shifted data
    """

    assert (len(x.shape) == 1) or (len(x.shape) == 2), "x can either be 1D or 2D"
    assert isinstance(freq_is_angular, bool)
    assert isinstance(x_is_real, bool)

    # axis is 0 if 1D or else it's 1
    axis = 0 if len(x.shape) == 1 else 1
    # V is angular frequency
    V = freq if freq_is_angular else freq * 2 * np.pi

    if not axis:
        # 1D scenario
        phase = np.exp(1j * V * shift)
    else:
        # 2D scenario
        assert (
            len(shift) == x.shape[0]
        ), "shift must be an array, one shift for each row of x"
        phase = np.exp(1j * V * np.c_[shift])

    if x_is_real:
        # real fft's
        # freq's shape should be the same as rfftfreq
        ft = rfft(x, axis=axis, fsc=fsc)
        ft *= phase
        return irfft(ft, axis=axis, fsc=fsc)
    else:
        # complex fft
        # freq's shape should be the same aas fftfreq
        ft = fft(x, axis=axis, fsc=fsc)
        ft *= phase
        return ifft(ft, axis=axis, fsc=fsc)


def calculate_spectrogram(pulse, T_delay):
    """
    calculate the spectrogram of a pulse over a given time delay axis

    Args:
        pulse (object):
            pulse instance
        T_delay (1D array):
            time delay axis (mks units)

    Returns:
        2D array:
            the calculated spectrogram over pulse.v_grid and T_delay
    """
    assert isinstance(pulse, Pulse), "pulse must be a Pulse instance"
    AT = np.zeros((len(T_delay), len(pulse.a_t)), dtype=np.complex128)
    AT[:] = pulse.a_t
    AT_shift = shift(
        AT,
        pulse.v_grid - pulse.v0,  # identical to fftfreq
        T_delay,
        fsc=pulse.dt,
        freq_is_angular=False,
        x_is_real=False,
    )
    AT2 = AT * AT_shift
    AW2 = fft(AT2, axis=1, fsc=pulse.dt)
    return abs(AW2) ** 2


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


class TFGrid:

    """
    I need v0 to be centered on the frequency grid for the phase retrieval
    algorithm to work
    """

    def __init__(self, n_points, v0, v_min, v_max, time_window):
        assert isinstance(n_points, int)
        assert time_window > 0
        assert 0 < v_min < v0 < v_max

        # ------------- calculate frequency bandwidth -------------------------
        v_span_pos = (v_max - v0) * 2.0
        v_span_neg = (v0 - v_min) * 2.0
        v_span = max([v_span_pos, v_span_neg])

        # calculate points needed to span both time and frequency bandwidth ---
        n_points_min = next_fast_len(int(np.ceil(v_span * time_window)))
        if n_points_min > n_points:
            print(
                f"changing n_points from {n_points} to {n_points_min} to"
                " support both time and frequency bandwidths"
            )
            n_points = n_points_min
        else:
            n_points_faster = next_fast_len(n_points)
            if n_points_faster != n_points:
                print(
                    f"changing n_points from {n_points} to {n_points_faster}"
                    " for faster fft's"
                )
                n_points = n_points_faster

        # ------------- create time and frequency grids -----------------------
        self._dt = time_window / n_points
        self._v_grid = np.fft.fftshift(np.fft.fftfreq(n_points, self._dt))
        self._v_grid += v0

        self._dv = np.diff(self._v_grid)[0]
        self._t_grid = np.fft.fftshift(np.fft.fftfreq(n_points, self._dv))

        self._v0 = v0
        self._n = n_points

    @property
    def n(self):
        return self._n

    @property
    def dt(self):
        return self._dt

    @property
    def t_grid(self):
        return self._t_grid

    @property
    def dv(self):
        return self._dv

    @property
    def v_grid(self):
        return self._v_grid

    @property
    def v0(self):
        return self._v0

    @property
    def wl_grid(self):
        return sc.c / self.v_grid


class Pulse(TFGrid):
    def __init__(self, n_points, v0, v_min, v_max, time_window, a_t):
        super().__init__(n_points, v0, v_min, v_max, time_window)

        self._a_t = a_t

    @property
    def a_t(self):
        """
        time domain electric field

        Returns:
            1D array
        """
        return self._a_t

    @property
    def a_v(self):
        """
        frequency domain electric field is given as the fft of the time domain
        electric field

        Returns:
            1D array
        """
        return fft(self.a_t, fsc=self.dt)

    @a_t.setter
    def a_t(self, a_t):
        """
        set the time domain electric field

        Args:
            a_t (1D array)
        """
        self._a_t = a_t.astype(np.complex128)

    @a_v.setter
    def a_v(self, a_v):
        """
        setting the frequency domain electric field is accomplished by setting
        the time domain electric field

        Args:
            a_v (1D array)
        """
        self.a_t = ifft(a_v, fsc=self.dt)

    @property
    def phi_v(self):
        """
        frequency domain phase

        Returns:
            1D array
        """
        return np.angle(self.a_v)

    @property
    def phi_t(self):
        """
        time domain phase

        Returns:
            1D array
        """
        return np.angle(self.a_t)

    @phi_v.setter
    def phi_v(self, phi_v):
        """
        sets the frequency domain phase

        Args:
            phi_v (1D array)
        """
        assert isinstance(phi_v, np.ndarray)
        assert phi_v.shape == self.a_v.shape
        self.a_v = abs(self.a_v) * np.exp(1j * phi_v)

    @phi_t.setter
    def phi_t(self, phi_t):
        """
        sets the time domain phase

        Args:
            phi_t (1D array)
        """

        assert isinstance(phi_t, np.ndarray)
        assert phi_t.shape == self.a_t.shape
        self.a_t = abs(self.a_t) * np.exp(1j * phi_t)

    @property
    def p_t(self):
        """
        time domain power

        Returns:
            1D array
        """
        return abs(self.a_t) ** 2

    @property
    def p_v(self):
        """
        frequency domain power

        Returns:
            1D array
        """
        return abs(self.a_v) ** 2

    @property
    def e_p(self):
        """
        pulse energy is calculated by integrating the time domain power

        Returns:
            float
        """
        return scint.simpson(self.p_t, dx=self.dt)

    @e_p.setter
    def e_p(self, e_p):
        """
        setting the pulse energy is done by scaling the electric field

        Args:
            e_p (float)
        """
        e_p_old = self.e_p
        factor_p_t = e_p / e_p_old
        self.a_t = self.a_t * factor_p_t**0.5

    @classmethod
    def Sech(cls, n_points, v_min, v_max, v0, e_p, t_fwhm, time_window):
        assert t_fwhm > 0
        assert e_p > 0

        tf = TFGrid(n_points, v0, v_min, v_max, time_window)
        tf: TFGrid

        a_t = 1 / np.cosh(2 * np.arccosh(2**0.5) * tf.t_grid / t_fwhm)

        p = cls(tf.n, v0, v_min, v_max, time_window, a_t)
        p: Pulse

        p.e_p = e_p
        return p

    def chirp_pulse_W(self, *chirp, v0=None):
        """
        chirp a pulse

        Args:
            *chirp (float):
                any number of floats representing gdd, tod, fod ... in seconds
            v0 (None, optional):
                center frequency for the taylor expansion, default is v0 of the
                pulse
        """
        assert [isinstance(i, float) for i in chirp]
        assert len(chirp) > 0

        if v0 is None:
            v0 = self.v0
        else:
            assert np.all([isinstance(v0, float), v0 > 0])

        v_grid = self.v_grid - v0
        w_grid = v_grid * 2 * np.pi

        factorial = np.math.factorial
        phase = 0
        for n, c in enumerate(chirp):
            n += 2  # start from 2
            phase += (c / factorial(n)) * w_grid**n
        self.a_v *= np.exp(1j * phase)

    def import_p_v(self, v_grid, p_v, phi_v=None):
        """
        import experimental spectrum

        Args:
            v_grid (1D array of floats):
                frequency grid
            p_v (1D array of floats):
                power spectrum
            phi_v (1D array of floats, optional):
                phase, default is transform limited, you would set this
                if you have a frog retrieval, for example
        """
        p_v = np.where(p_v > 0, p_v, 1e-20)
        amp_v = p_v**0.5
        amp_v = spi.interp1d(
            v_grid, amp_v, kind="cubic", bounds_error=False, fill_value=1e-20
        )(self.v_grid)

        if phi_v is not None:
            assert isinstance(phi_v, np.ndarray) and phi_v.shape == p_v.shape
            phi_v = spi.interp1d(
                v_grid, phi_v, kind="cubic", bounds_error=False, fill_value=0.0
            )(self.v_grid)
        else:
            phi_v = 0.0

        a_v = amp_v * np.exp(1j * phi_v)

        e_p = self.e_p
        self.a_v = a_v
        self.e_p = e_p

    @classmethod
    def clone_pulse(cls, pulse):
        assert isinstance(pulse, Pulse) or isinstance(pulse, Pulse)
        pulse: Pulse
        n_points = pulse.n
        v_min = pulse.v_grid[0]
        v_max = pulse.v_grid[-1]
        v0 = pulse.v0
        e_p = pulse.e_p
        time_window = np.diff(pulse.t_grid[[0, -1]])
        t_fwhm = 200e-15  # only affects power spectrum in the Sech call

        p = cls.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm, time_window)

        if isinstance(pulse, Pulse):
            p.a_v[:] = pulse.a_v[:]
        else:
            p.import_p_v(pulse.v_grid, pulse.p_v, phi_v=pulse.phi_v)
        return p


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
        ind_10perc = (
            np.argmin(abs(R[300:] - 0.1)) + 300
        )  # the frog spectrogram doesn't usually extend past here

        self.spectrogram[:, ind_10perc:] /= R[ind_10perc:]
        self.denoise_spectrogram()

        self._min_pm_fthz = min(self.F_THz)
        self._max_pm_fthz = self.F_THz[ind_10perc]

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
        x = calculate_spectrogram(self.pulse, self.T_fs * 1e-15)
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

            s = calculate_spectrogram(self.pulse, self.T_fs * 1e-15)[
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
        s = calculate_spectrogram(self.pulse, self.T_fs * 1e-15)
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
            factor = max(self.pulse.p_v) / max(self.pulse_data.p_v)
            ax[1].plot(
                self.pulse_data.v_grid * 1e-12,
                self.pulse_data.p_v * factor,
                color="C2",
            )

        return fig, ax
