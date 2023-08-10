import numpy as np

try:
    import mkl_fft

    print("USING MKL FOR FFT'S IN PYTHON SHG FROG PHASE RETRIEVAL")
    use_mkl = True
except ImportError:
    print("NOT USING MKL FOR FFT'S IN PYTHON SHG FROG PHASE RETRIEVAL")

    class mkl_fft:
        """
        reproducing the following functions from mkl:
            fft, ifft, rfft_numpy and irfft_numpy
        """

        def fft(x, axis=-1, fsc=1.0):
            return np.fft.fft(x, axis=axis) * fsc

        def ifft(x, axis=-1, fsc=1.0):
            return np.fft.ifft(x, axis=axis) / fsc

        def rfft_numpy(x, axis=-1, fsc=1.0, n=None):
            return np.fft.rfft(x, axis=axis, n=n) * fsc

        def irfft_numpy(x, axis=-1, fsc=1.0, n=None):
            return np.fft.irfft(x, axis=axis, n=n) / fsc


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
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x), fsc=fsc))

    else:
        return np.fft.fftshift(
            mkl_fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis, fsc=fsc),
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
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x), fsc=fsc))

    else:
        return np.fft.fftshift(
            mkl_fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis, fsc=fsc),
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
        return np.fft.fftshift(mkl_fft.irfft_numpy(x, fsc=fsc))

    else:
        return np.fft.fftshift(mkl_fft.irfft_numpy(x, axis=axis, fsc=fsc), axes=axis)


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
