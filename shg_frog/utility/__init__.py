__all__ = ["fft", "TFGrid", "resample_t", "resample_v"]

# __init__ import
from shg_frog.utility import fft

# imports needed for functions below
import numpy as np
from scipy.fftpack import next_fast_len
import scipy.constants as sc
import collections
from shg_frog.utility.fft import mkl_fft

_ResampledV = collections.namedtuple("ResampledV", ["v_grid", "f_v", "dv", "dt"])

_ResampledT = collections.namedtuple("ResampledT", ["t_grid", "f_t", "dt"])


def resample_v(v_grid, f_v, n):
    """
    Resample frequency-domain data to the given number of points.

    The complementary time data is assumed to be of finite support, so the
    resampling is accomplished by adding or removing trailing and leading time
    bins. Discontinuities in the frequency-domain amplitude will manifest as
    ringing when resampled.

    Parameters
    ----------
    v_grid : array_like of float
        The frequency grid of the input data.
    f_v : array_like of complex
        The frequency-domain data to be resampled.
    n : int
        The number of points at which to resample the input data. When the
        input corresponds to a real-valued time domain representation, this
        number is the number of points in the time domain.

    Returns
    -------
    v_grid : ndarray of float
        The resampled frequency grid.
    f_v : ndarray of real or complex
        The resampled frequency-domain data.
    dv : float
        The spacing of the resampled frequency grid.
    dt : float
        The spacing of the resampled time grid.

    Notes
    -----
    If the number of points is odd, there are an equal number of points on
    the positive and negative side of the time grid. If even, there is one
    extra point on the negative side.

    This method checks if the origin is contained in `v_grid` to determine
    whether real or complex transformations should be performed. In both cases
    the resampling is accomplished by removing trailing and leading time bins.

    For analytic representations, the returned frequency grid is defined
    symmetrically about its reference, as in the `TFGrid` class, and for
    real-valued representations the grid is defined starting at the origin.

    """
    assert isinstance(
        n, (int, np.integer)
    ), "The requested number of points must be an integer"
    assert n > 0, "The requested number of points must be greater than 0."
    assert len(v_grid) == len(
        f_v
    ), "The frequency grid and frequency-domain data must be the same length."
    # ---- Inverse Transform
    dv_0 = np.diff(v_grid).mean()
    if v_grid[0] == 0:
        assert np.isreal(
            f_v[0]
        ), "When the input is in the real-valued representation, the amplitude at the origin must be real."

        # Real-Valued Representation
        if np.isreal(f_v[-1]):
            n_0 = 2 * (len(v_grid) - 1)
        else:
            n_0 = 2 * (len(v_grid) - 1) + 1
        dt_0 = 1 / (n_0 * dv_0)
        f_t = np.fft.fftshift(mkl_fft.irfft_numpy(f_v, fsc=dt_0, n=n_0))
    else:
        # Analytic Representation
        n_0 = len(v_grid)
        dt_0 = 1 / (n_0 * dv_0)
        v_ref_0 = v_grid[n_0 // 2]
        f_t = np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(f_v), fsc=dt_0))

    # ---- Resample
    dn_n = n // 2 - n_0 // 2  # leading time bins
    dn_p = (n - 1) // 2 - (n_0 - 1) // 2  # trailing time bins
    if n > n_0:
        f_t = np.pad(f_t, (dn_n, dn_p), mode="constant", constant_values=0)
    elif n < n_0:
        f_t = f_t[-dn_n : n_0 + dn_p]

    # ---- Transform
    dt = 1 / (n_0 * dv_0)
    dv = 1 / (n * dt)
    if v_grid[0] == 0:
        # Real-Valued Representation
        f_v = mkl_fft.rfft_numpy(np.fft.ifftshift(f_t), fsc=dt)
        v_grid = dv * np.arange(len(f_v))
    else:
        # Analytic Representation
        f_v = np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(f_t), fsc=dt))
        v_grid = dv * (np.arange(n) - (n // 2))
        v_grid += v_ref_0

    # ---- Construct ResampledV
    resampled = _ResampledV(v_grid=v_grid, f_v=f_v, dv=dv, dt=1 / (n * dv))
    return resampled


def resample_t(t_grid, f_t, n):
    """
    Resample time-domain data to the given number of points.

    The complementary frequency data is assumed to be band-limited, so the
    resampling is accomplished by adding or removing high frequency bins.
    Discontinuities in the time-domain amplitude will manifest as ringing when
    resampled.

    Parameters
    ----------
    t_grid : array_like of float
        The time grid of the input data.
    f_t : array_like of real or complex
        The time-domain data to be resampled.
    n : int
        The number of points at which to resample the input data.

    Returns
    -------
    t_grid : ndarray of float
        The resampled time grid.
    f_t : ndarray of real or complex
        The resampled time-domain data.
    dt : float
        The spacing of the resampled time grid.

    Notes
    -----
    If real, the resampling is accomplished by adding or removing the largest
    magnitude frequency components (both positive and negative). If complex,
    the input data is assumed to be analytic, so the resampling is accomplished
    by adding or removing the largest positive frequencies. This method checks
    the input data's type, not the magnitude of its imaginary component, to
    determine if it is real or complex.

    The returned time axis is defined symmetrically about the input's
    reference, such as in the `TFGrid` class.

    """
    assert isinstance(
        n, (int, np.integer)
    ), "The requested number of points must be an integer"
    assert n > 0, "The requested number of points must be greater than 0."
    assert len(t_grid) == len(
        f_t
    ), "The time grid and time-domain data must be the same length."
    # ---- Define Time Grid
    n_0 = len(t_grid)
    dt_0 = np.diff(t_grid).mean()
    t_ref_0 = t_grid[n_0 // 2]
    dv = 1 / (n_0 * dt_0)
    dt = 1 / (n * dv)
    t_grid = dt * (np.arange(n) - (n // 2))
    t_grid += t_ref_0

    # ---- Resample
    if np.isrealobj(f_t):
        # Real-Valued Representation
        f_v = mkl_fft.rfft_numpy(np.fft.ifftshift(f_t), fsc=dt_0)
        if (n > n_0) and not (n % 2):
            f_v[-1] /= 2  # renormalize aliased Nyquist component
        f_t = np.fft.fftshift(mkl_fft.irfft_numpy(f_v, fsc=dt, n=n))
    else:
        # Analytic Representation
        f_v = np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(f_t), fsc=dt_0))
        if n > n_0:
            f_v = np.pad(f_v, (0, n - n_0), mode="constant", constant_values=0)
        elif n < n_0:
            f_v = f_v[:n]
        f_t = np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(f_v), fsc=dt))

    # ---- Construct ResampledT
    resampled = _ResampledT(t_grid=t_grid, f_t=f_t, dt=dt)
    return resampled


class TFGrid:

    """
    I need v0 to be centered on the frequency grid for the phase retrieval
    algorithm to work
    """

    def __init__(self, n_points, v0, v_min, v_max, time_window):
        assert isinstance(n_points, int)
        assert time_window > 0
        assert v_min < v0 < v_max

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
