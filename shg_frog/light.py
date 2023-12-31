__all__ = ["Pulse"]

import collections
from shg_frog.utility import resample_v, resample_t, TFGrid
from shg_frog.utility.fft import fft, ifft, shift
import numpy as np
import scipy.integrate as scint
import scipy.interpolate as spi

try:
    import pynlo

    pynlo_imported = True
    print("PYNLO IS INSTALLED")
except ImportError:
    pynlo_imported = False
    print("PYNLO IS NOT INSTALLED")

PowerSpectralWidth = collections.namedtuple(
    "PowerSpectralWidth", ["fwhm", "rms", "eqv"]
)

PowerEnvelopeWidth = collections.namedtuple(
    "PowerEnvelopeWidth", ["fwhm", "rms", "eqv"]
)


class ArrayWrapper(np.lib.mixins.NDArrayOperatorsMixin):

    """
    This array wrapper is used for numpy arrays so that in place modification
    will call the setter function, whereas by default this does not occur
    """

    def __init__(self, getitem, setitem):
        self.__getitem = getitem
        self.__setitem = setitem

    # array indexing (main reason for the ArrayWrapper class)
    def __getitem__(self, key):
        return self.__getitem(key)

    def __setitem__(self, key, val):
        return self.__setitem(key, val)

    # for print out in console
    def __repr__(self):
        return repr(self.__getitem__(...))

    # easily accessible __array__()
    def __array__(self, dtype=None):
        array = self.__getitem__(...)
        if dtype is None:
            return array
        else:
            return array.astype(dtype=dtype)

    # __len__ doesn't seem to be caught by __getattr__
    def __len__(self):
        return len(self.__array__())

    # all the other stuff
    def __getattr__(self, name):
        return getattr(self.__array__(), name)


class PropertyForArray(property):
    def __get__(self, instance, owner):
        """
        The default __get__ would be self.fget(instance), which just calls the
        getter function defined in the owner class
        """

        def getitem(key):
            return self.fget(instance, key)

        def setitem(key, val):
            assert self.fset is not None, f"a setter has not been defined"
            self.fset(instance, val, key)  # key is a kwarg in Pulse so it comes last

        return ArrayWrapper(getitem, setitem)


class Pulse(TFGrid):
    def __init__(self, n_points, v0, v_min, v_max, time_window, a_t):
        super().__init__(n_points, v0, v_min, v_max, time_window)

        self._a_t = a_t.astype(np.complex128)

    @PropertyForArray
    def a_t(self, key=...):
        """
        time domain electric field

        Returns:
            1D array
        """
        return self._a_t[key]

    @PropertyForArray
    def a_v(self, key=...):
        """
        frequency domain electric field is given as the fft of the time domain
        electric field

        Returns:
            1D array
        """
        return fft(self.a_t, fsc=self.dt)[key]

    @a_t.setter
    def a_t(self, a_t, key=...):
        """
        set the time domain electric field

        Args:
            a_t (1D array)
        """
        self._a_t[key] = a_t

    @a_v.setter
    def a_v(self, a_v, key=...):
        """
        setting the frequency domain electric field is accomplished by setting
        the time domain electric field

        Args:
            a_v (1D array)
        """
        if key is not ...:
            _a_v = self.a_v.copy()
            _a_v[key] = a_v
            a_v = _a_v
        self.a_t = ifft(a_v, fsc=self.dt)

    @PropertyForArray
    def phi_v(self, key=...):
        """
        frequency domain phase

        Returns:
            1D array
        """
        return np.angle(self.a_v[key])

    @PropertyForArray
    def phi_t(self, key=...):
        """
        time domain phase

        Returns:
            1D array
        """
        return np.angle(self.a_t[key])

    @phi_v.setter
    def phi_v(self, phi_v, key=...):
        """
        sets the frequency domain phase

        Args:
            phi_v (1D array)
        """
        self.a_v[key] = abs(self.a_v[key]) * np.exp(1j * phi_v)

    @phi_t.setter
    def phi_t(self, phi_t, key=...):
        """
        sets the time domain phase

        Args:
            phi_t (1D array)
        """

        self.a_t[key] = abs(self.a_t[key]) * np.exp(1j * phi_t)

    @PropertyForArray
    def p_t(self, key=...):
        """
        time domain power

        Returns:
            1D array
        """
        return abs(self.a_t[key]) ** 2

    @PropertyForArray
    def p_v(self, key=...):
        """
        frequency domain power

        Returns:
            1D array
        """
        return abs(self.a_v[key]) ** 2

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
        p_v = np.where(p_v > 0, p_v, 0.0)
        amp_v = p_v**0.5
        amp_v = spi.interp1d(
            v_grid, amp_v, kind="cubic", bounds_error=False, fill_value=0.0
        )(self.v_grid)

        if phi_v is not None:
            if pynlo_imported:
                assert (
                    isinstance(phi_v, np.ndarray)
                    or isinstance(phi_v, pynlo.utility.misc.ArrayWrapper)
                    or isinstance(phi_v, ArrayWrapper)
                ) and phi_v.shape == p_v.shape
            else:
                assert (
                    isinstance(phi_v, np.ndarray) or isinstance(phi_v, ArrayWrapper)
                ) and phi_v.shape == p_v.shape
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
        if pynlo_imported:
            assert isinstance(pulse, Pulse) or isinstance(pulse, pynlo.light.Pulse)
        else:
            assert isinstance(pulse, Pulse)
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
            # make sure to unwrap the phase before fitting!
            p.import_p_v(pulse.v_grid, pulse.p_v, phi_v=np.unwrap(pulse.phi_v))
        return p

    def t_width(self, m=None):
        """
        Calculate the width of the pulse in the time domain.

        Set `m` to optionally resample the number of points and change the
        time resolution.

        Parameters
        ----------
        m : float, optional
            The multiplicative number of points at which to resample the power
            envelope. The default is to not resample.

        Returns
        -------
        fwhm : float
            The full width at half maximum of the power envelope.
        rms : float
            The full root-mean-square width of the power envelope.
        eqv : float
            The equivalent width of the power envelope.

        """
        # ---- Power
        p_t = self.p_t

        # ---- Resample
        if m is None:
            n = self.n
            t_grid = self.t_grid
            dt = self.dt
        else:
            assert m > 0, "The point multiplier must be greater than 0."
            n = round(m * self.n)
            resampled = resample_t(self.t_grid, p_t, n)
            p_t = resampled.f_t
            t_grid = resampled.t_grid
            dt = resampled.dt

        # ---- FWHM
        p_max = p_t.max()
        t_selector = t_grid[p_t >= 0.5 * p_max]
        t_fwhm = dt + (t_selector.max() - t_selector.min())

        # ---- RMS
        p_norm = np.sum(p_t * dt)
        t_avg = np.sum(t_grid * p_t * dt) / p_norm
        t_var = np.sum((t_grid - t_avg) ** 2 * p_t * dt) / p_norm
        t_rms = 2 * t_var**0.5

        # ---- Equivalent
        t_eqv = 1 / np.sum((p_t / p_norm) ** 2 * dt)

        # ---- Construct PowerEnvelopeWidth
        t_widths = PowerEnvelopeWidth(fwhm=t_fwhm, rms=t_rms, eqv=t_eqv)
        return t_widths

    def v_width(self, m=None):
        """
        Calculate the width of the pulse in the frequency domain.

        Set `m` to optionally resample the number of points and change the
        frequency resolution.

        Parameters
        ----------
        m : float, optional
            The multiplicative number of points at which to resample the power
            spectrum. The default is to not resample.

        Returns
        -------
        fwhm : float
            The full width at half maximum of the power spectrum.
        rms : float
            The full root-mean-square width of the power spectrum.
        eqv : float
            The equivalent width of the power spectrum.

        """
        # ---- Power
        p_v = self.p_v

        # ---- Resample
        if m is None:
            n = self.n
            v_grid = self.v_grid
            dv = self.dv
        else:
            assert m > 0, "The point multiplier must be greater than 0."
            n = round(m * self.n)
            resampled = resample_v(self.v_grid, p_v, n)
            p_v = resampled.f_v
            v_grid = resampled.v_grid
            dv = resampled.dv

        # ---- FWHM
        p_max = p_v.max()
        v_selector = v_grid[p_v >= 0.5 * p_max]
        v_fwhm = dv + (v_selector.max() - v_selector.min())

        # ---- RMS
        p_norm = np.sum(p_v * dv)
        v_avg = np.sum(v_grid * p_v * dv) / p_norm
        v_var = np.sum((v_grid - v_avg) ** 2 * p_v * dv) / p_norm
        v_rms = 2 * v_var**0.5

        # ---- Equivalent
        v_eqv = 1 / np.sum((p_v / p_norm) ** 2 * dv)

        # ---- Construct PowerSpectralWidth
        v_widths = PowerSpectralWidth(fwhm=v_fwhm, rms=v_rms, eqv=v_eqv)
        return v_widths

    @property
    def tg_v(self):
        """
        The spectral group delay, with units of ``s``.

        Returns
        -------
        ndarray of float

        """
        return self.t_grid[self.n // 2] - np.gradient(
            np.unwrap(self.phi_v) / (2 * np.pi), self.v_grid, edge_order=2
        )

    @property
    def vg_t(self):
        """
        The instantaneous frequency of the complex envelope, with units of
        ``Hz``.

        Returns
        -------
        ndarray of float

        """
        return self.v_grid[self.n // 2] + np.gradient(
            np.unwrap(self.phi_t) / (2 * np.pi), self.t_grid, edge_order=2
        )

    def spectrogram_shg(self, T_delay):
        """
        calculate the shg spectrogram over a given time delay axis

        Args:
            T_delay (1D array):
                time delay axis (mks units)

        Returns:
            2D array:
                the calculated spectrogram over pulse.v_grid and T_delay
        """
        AT = np.zeros((len(T_delay), len(self.a_t)), dtype=np.complex128)
        AT[:] = self.a_t
        AT_shift = shift(
            AT,
            self.v_grid - self.v0,  # identical to fftfreq
            T_delay,
            fsc=self.dt,
            freq_is_angular=False,
            x_is_real=False,
        )
        AT2 = AT * AT_shift
        AW2 = fft(AT2, axis=1, fsc=self.dt)
        return abs(AW2) ** 2
