"""
This file is for obtaining phase-matching curves for BBO.

Attributes:
    deg_to_rad (func): converts degree to radian
    rad_to_deg (func): converts radian to degree
    normalize (func): normalizes a vector
    BBOSHG (class): BBO class
"""
import numpy as np

normalize = lambda vec: vec / np.max(abs(vec))
rad_to_deg = lambda rad: rad * 180.0 / np.pi
deg_to_rad = lambda deg: deg * np.pi / 180.0


class BBOSHG:

    """Summary"""

    def __init__(self):
        """Summary"""
        pass

    """The following refractive index curves are taken from:

    G. Tamošauskas, G. Beresnevičius, D. Gadonas, and A.
    Dubietis, “Transmittance and phase matching of BBO crystal in the 3−5 μm
    range and its application for the characterization of mid-infrared laser
    pulses,” Opt. Mater. Express, vol. 8, no. 6, p. 1410, Jun. 2018, doi:
    10.1364/OME.8.001410. """

    def no(self, wl_um):
        """
        ordinary refractive index for k-vector orthogonal to the c-axis

        Args:
            wl_um (1D array): input wavelength in micron

        Returns:
            1D array: extraordinary refractive index, same length as wl_um
        """
        num1 = 0.90291
        denom1 = 0.003926

        num2 = 0.83155
        denom2 = 0.018786

        num3 = 0.76536
        denom3 = 60.01

        term1 = self._term_sellmeier(wl_um, num1, denom1)
        term2 = self._term_sellmeier(wl_um, num2, denom2)
        term3 = self._term_sellmeier(wl_um, num3, denom3)

        return np.sqrt(1 + term1 + term2 + term3)

    def ne(self, wl_um):
        """
        extraordinary refractive index for k-vector along the c-axis, this is
        the same as ne_theta for 0 degrees

        Args:
            wl_um (1D array): input wavelength in micron

        Returns:
            1D array: extraordinary refractive index, same length as wl_um
        """
        num1 = 1.151075
        denom1 = 0.007142

        num2 = 0.21803
        denom2 = 0.02259

        num3 = 0.656
        denom3 = 263.0

        term1 = self._term_sellmeier(wl_um, num1, denom1)
        term2 = self._term_sellmeier(wl_um, num2, denom2)
        term3 = self._term_sellmeier(wl_um, num3, denom3)

        return np.sqrt(1 + term1 + term2 + term3)

    def _term_sellmeier(self, wl_um, num, denom):
        """
        convenience function to reduce clutter in ne and no
        """
        return (num * wl_um**2) / (wl_um**2 - denom)

    def ne_theta(self, wl_um, theta_rad):
        """
        The extra-ordinary refractive index as a function of the angle between
        the c-axis and the k-vector

        Args:
            wl_um (1D array): input wavelength in micron
            theta_rad (float): phase matching angle in radian

        Returns:
            1D array: extraordinary refractive index, same length as wl_um
        """
        term1 = np.sin(theta_rad) ** 2 / self.ne(wl_um) ** 2
        term2 = np.cos(theta_rad) ** 2 / self.no(wl_um) ** 2

        return 1 / np.sqrt(term1 + term2)

    def phase_match_angle_rad(self, wl_um):
        """
        calculates the phase matching angle for a given wavelength

        Args:
            wl_um (float): input wavelength in micron

        Returns:
            float: phase matching angle in radians
        """
        no = self.no(wl_um)
        no2 = self.no(wl_um / 2.0)
        ne2 = self.ne(wl_um / 2.0)

        num = (1 / no**2) - (1 / no2**2)
        denom = (1 / ne2**2) - 1 / (no2**2)
        sin_theta = np.sqrt(num / denom)
        return np.arcsin(sin_theta)

    def dk(self, wl_um, theta_pm_rad, alpha_rad=0.0):
        """
        Calculates the phase mismatch given the phase matching angle and
        non-collinear angle

        Args:
            wl_um (1D array): input wavelength in micron
            theta_pm_rad (float): phase matching angle
            alpha_rad (float, optional): non-collinear angle for a single
            beam (angle between surface normal and one input beam)

        Returns:
            1D array: phase mismatch in inverse micron, same length as wl_um
        """
        no = self.no(wl_um)
        ne2 = self.ne_theta(wl_um / 2.0, theta_pm_rad)

        k1 = no / wl_um
        k2 = ne2 / (wl_um / 2.0)

        return k2 - 2 * k1 * np.cos(alpha_rad)

    """taken from page 14 in Trebino's FROG book:

    R. Trebino, A. Baltuška, M. S. Pshenichnikov, and D. A. Wiersma, “Measuring
    Ultrashort Pulsesin the Single-Cycle Regime:Frequency-Resolved Optical
    Gating,” in Few-Cycle Laser Pulse Generation and Its Applications, vol.
    95, F. X. Kärtner, Ed. Berlin, Heidelberg: Springer Berlin Heidelberg,
    2004, pp. 231–264. doi: 10.1007/978-3-540-39849-3_5.

    In our case I believe we have the omega^3 dependence """

    def R(self, wl_um, length_um, theta_pm_rad, alpha_rad=0.0):
        """
        The conversion efficiency as a function of wavelength, crystal
        thickness, phase-matching angle and non-collinear nagle

        Args:
            wl_um (1D array): input wavelength in micron
            length_um (float): crystal thickness in micron
            theta_pm_rad (float): phase matching angle in radians
            alpha_rad (float, optional): non-collinear angle in radians

        Returns:
            1D array: conversion efficiency, same length as wl_um
        """
        f2 = 1 / (wl_um / 2.0)
        ne2 = self.ne_theta(wl_um / 2.0, theta_pm_rad)
        no = self.no(wl_um)

        dk = self.dk(wl_um, theta_pm_rad, alpha_rad)

        term1 = f2**3 / ne2
        term2 = ne2**2 - 1
        term3 = no**2 - 1
        term4 = np.sinc(dk * length_um / 2.0)

        return normalize(term1 * (term2 * term3**2) ** 2 * term4**2)
