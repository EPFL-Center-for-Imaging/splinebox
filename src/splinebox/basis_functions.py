"""
This module provides spline basis functions.
"""

import math
import warnings

import numba
import numpy as np


class BasisFunction:
    """
    Base class for all basis functions.

    Parameters
    ----------
    multigenerator : boolean
        This indicates if the basis function generates
        multiple outputs. In practice, this is used to indicate
        if a basis function is meant for a Hermite spline or not.
        Basis functions for Hermite splines return two values
        instead of one.
    support : float
        The support of the function, i.e. the size of the area
        being mapped to non-zero values.
    """

    _unimplemented_message = "This function is not implemented."

    def __init__(self, multigenerator, support):
        self.multigenerator = multigenerator
        self.support = support

    def eval(self, t, derivative=0):
        """
        Evaluate the function at position(s) `t`.

        Parameters
        ----------
        t : float or numpy.array
            The points where the function should be evaluated.
        derivative : [0, 1, 2], default = 0
            Whether to evaluate the function (0) or its first (1)
            or second (2) derivative.

        Returns
        -------
        y : numpy.array
            Values of the function or its first or second derivative
            at position(s) `t`.
        """
        if derivative == 0:
            return self._func(t)
        elif derivative == 1:
            return self._derivative_1(t)
        elif derivative == 2:
            return self._derivative_2(t)
        else:
            raise ValueError(f"derivative has to be 0, 1, or 2 not {derivative}")

    def _func(self, t):
        raise NotImplementedError(BasisFunction._unimplemented_message)

    def _derivative_1(self, t):
        raise NotImplementedError(BasisFunction._unimplemented_message)

    def _derivative_2(self, t):
        raise NotImplementedError(BasisFunction._unimplemented_message)

    def filter_symmetric(self, s):
        """
        The filter used to turn knots into control points for an
        open spline.

        Parameters
        ----------
        s : numpy.array
            An array of knots of shape (n, dim) where n is the number
            of knots and dim is the dimensionality of the codomain, i.e.
            the space in which the curve lives. Note that the knots should
            be padded.

        Returns
        -------
        control_points : numpy.array
            The control points for the spline passing through the knots provided.
        """
        raise NotImplementedError(BasisFunction._unimplemented_message)

    def filter_periodic(self, s):
        """
        The filter used to turn knots into control points for a
        closed spline.

        Parameters
        ----------
        s : numpy.array
            An array of knots of shape (n, dim) where n is the number
            of knots and dim is the dimensionality of the codomain, i.e.
            the space in which the curve lives. Note that the knots should
            be padded.

        Returns
        -------
        control_points : numpy.array
            The control points for the spline passing through the knots provided.
        """
        raise NotImplementedError(BasisFunction._unimplemented_message)

    def refinement_mask(self):
        """
        This function is needed for local refinement (see [Badoual2016]_).
        Basis splines with the 'local refinement property' can be expressed as a
        linear combination of themselfs. This is useful when you iteratively want
        to refine your spline with additional knots in a given interval.
        This creates a non-uniform spline, which is not supported by splinebox.
        We keep it here incase we ever decide to support non-uniform splines in the
        future.
        """
        raise NotImplementedError(BasisFunction._unimplemented_message)


class B1(BasisFunction):
    r"""
    Basis function for a linear (:math:`1^{\text{st}}` order) polynomial basis spline.

    For a detailed theoretical description, including the equation and
    a plot of the function, refer to the :ref:`Polynomial basis (B-spline)`
    section in the documentation.

    The constructor does not require any arguments.

    For more information on the methods and attributes available in this class,
    please see the documentation for :class:`splinebox.basis_functions.BasisFunction`.
    """

    def __init__(self):
        super().__init__(False, 2)

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _func(t):  # pragma: no cover
        val = 0
        if abs(t) >= 0 and abs(t) < 1:
            val = 1 - abs(t)
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _derivative_1(t):  # pragma: no cover
        val = 0
        if t > -1 and t < 0:
            val = 1
        elif t > 0 and t < 1:
            val = -1
        elif t == 0 or t == -1 or t == 1:
            # This is the gradient you'll get at exactly 0
            val = np.nan
        return val

    @staticmethod
    def _derivative_2(t):
        raise RuntimeError("B1 isn't twice differentiable.")

    @staticmethod
    def filter_symmetric(s):
        return s

    @staticmethod
    def filter_periodic(s):
        return s

    def refinement_mask(self):
        order = int(self.support)
        mask = np.zeros(order + 1)
        _multinomial(order, 2, np.zeros(2, dtype=int), 0, 2, order, mask)
        return mask


class B2(BasisFunction):
    r"""
    Basis function for a quadratic (:math:`2^{\text{nd}}` order) polynomial basis spline.

    For a detailed theoretical description, including the equation and
    a plot of the function, refer to the :ref:`Polynomial basis (B-spline)`
    section in the documentation.

    The constructor does not require any arguments.

    For more information on the methods and attributes available in this class,
    please see the documentation for :class:`splinebox.basis_functions.BasisFunction`.
    """

    def __init__(self):
        super().__init__(False, 3)

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _func(t):  # pragma: no cover
        val = 0
        if t >= -1.5 and t <= -0.5:
            val = 0.5 * (t**2) + 1.5 * t + 1.125
        elif t > -0.5 and t <= 0.5:
            val = -t * t + 0.75
        elif t > 0.5 and t <= 1.5:
            val = 0.5 * (t**2) - 1.5 * t + 1.125
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _derivative_1(t):  # pragma: no cover
        val = 0
        if t >= -1.5 and t <= -0.5:
            val = t + 1.5
        elif t > -0.5 and t <= 0.5:
            val = -2 * t
        elif t > 0.5 and t <= 1.5:
            val = t - 1.5
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _derivative_2(t):  # pragma: no cover
        val = 0
        if t >= -1.5 and t <= -0.5:
            val = 1
        elif t > -0.5 and t <= 0.5:
            val = -2
        elif t > 0.5 and t <= 1.5:
            val = 1
        return val

    def refinement_mask(self):
        order = int(self.support)
        mask = np.zeros(order + 1)
        _multinomial(order, 2, np.zeros(2, dtype=int), 0, 2, order, mask)
        return mask


class B3(BasisFunction):
    r"""
    Basis function for a cubic (:math:`3^{\text{rd}}` order) polynomial basis spline.

    For a detailed theoretical description, including the equation and
    a plot of the function, refer to the :ref:`Polynomial basis (B-spline)`
    section in the documentation.

    The constructor does not require any arguments.

    For more information on the methods and attributes available in this class,
    please see the documentation for :class:`splinebox.basis_functions.BasisFunction`.
    """

    def __init__(self):
        super().__init__(False, 4)

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _func(t):  # pragma: no cover
        val = 0
        if abs(t) >= 0 and abs(t) < 1:
            val = 2 / 3 - (abs(t) ** 2) + (abs(t) ** 3) / 2
        elif abs(t) >= 1 and abs(t) <= 2:
            val = ((2 - abs(t)) ** 3) / 6
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _derivative_1(t):  # pragma: no cover
        val = 0
        if t >= 0 and t < 1:
            val = -2 * t + 1.5 * t * t
        elif t > -1 and t < 0:
            val = -2 * t - 1.5 * t * t
        elif t >= 1 and t <= 2:
            val = -0.5 * ((2 - t) ** 2)
        elif t >= -2 and t <= -1:
            val = 0.5 * ((2 + t) ** 2)
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _derivative_2(t):  # pragma: no cover
        val = 0
        if t >= 0 and t < 1:
            val = -2 + 3 * t
        elif t > -1 and t < 0:
            val = -2 - 3 * t
        elif t >= 1 and t <= 2:
            val = 2 - t
        elif t >= -2 and t <= -1:
            val = 2 + t
        return val

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def filter_symmetric(s):  # pragma: no cover
        M = len(s)
        pole = -2 + np.sqrt(3)

        ndim = 1 if s.ndim == 1 else s.shape[1]

        cp = np.zeros((M, ndim))
        eps = 1e-8
        k0 = min(((2 * M) - 2, int(np.ceil(np.log(eps) / np.log(np.abs(pole))))))
        for k in np.arange(k0):
            m = k % (2 * M - 2)
            val = s[2 * M - 2 - m] if m >= M else s[m]
            cp[0] += val * (pole**m)
        cp[0] *= 1 / (1 - (pole ** (2 * M - 2)))

        for k in np.arange(1, M):
            cp[k] = s[k] + pole * cp[k - 1]

        cm = np.zeros((M, ndim))
        cm[M - 1] = cp[M - 1] + (pole * cp[M - 2])
        cm[M - 1] *= pole / ((pole**2) - 1)

        for k in np.arange(M - 2, -1, -1):
            cm[k] = pole * (cm[k + 1] - cp[k])

        c = cm * 6

        shape = c.shape
        c = c.flatten()
        c[np.abs(c) < eps] = 0
        c = c.reshape(shape)
        return c

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def filter_periodic(s):  # pragma: no cover
        M = len(s)
        pole = -2 + np.sqrt(3)

        ndim = 1 if s.ndim == 1 else s.shape[1]

        cp = np.zeros((M, ndim))
        for k in range(M):
            cp[0] += s[(M - k) % M] * (pole**k)
        cp[0] *= 1 / (1 - (pole**M))

        for k in range(1, M):
            cp[k] = s[k] + pole * cp[k - 1]

        cm = np.zeros((M, ndim))
        for k in range(M):
            cm[M - 1] += (pole**k) * cp[k]
        cm[M - 1] *= pole / (1 - (pole**M))
        cm[M - 1] += cp[M - 1]
        cm[M - 1] *= -pole

        for k in range(M - 2, -1, -1):
            cm[k] = pole * (cm[k + 1] - cp[k])

        c = cm * 6

        eps = 1e-8
        shape = c.shape
        c = c.flatten()
        c[np.abs(c) < eps] = 0
        c = c.reshape(shape)
        return c

    def refinement_mask(self):
        order = int(self.support)
        mask = np.zeros(order + 1)
        _multinomial(order, 2, np.zeros(2, dtype=int), 0, 2, order, mask)
        return mask


class Exponential(BasisFunction):
    r"""
    Basis function for an exponential spline.

    For a detailed theoretical description, including the equation and
    a plot of the function, refer to the :ref:`Exponential basis`
    section in the documentation.

    The constructor requires `M`, the number of knots in the spline, as an argument.

    For more information on the methods and attributes available in this class,
    please see the documentation for :class:`splinebox.basis_functions.BasisFunction`.
    """

    def __init__(self, M):
        super().__init__(False, 3)
        self.M = M

    def _func(self, t):
        return self.__func(t, self.support / 2, self.M)

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64)], nopython=True, cache=True)
    def __func(t, half_support, M):  # pragma: no cover
        t += half_support

        alpha = np.pi / M
        L = 1 / (4 * np.sin(alpha) ** 2)

        val = 0
        if t >= 0 and t < 1:
            val = 2 * np.sin(alpha * t) ** 2
        elif t >= 1 and t < 2:
            val = np.cos(2 * alpha * (t - 2)) + np.cos(2 * alpha * (t - 1)) - 2 * np.cos(2 * alpha)
        elif t >= 2 and t <= 3:
            val = 2 * np.sin(alpha * (t - 3)) ** 2

        return L * val

    def _derivative_1(self, t):
        return self.__derivative_1(t, self.support / 2, self.M)

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64)], nopython=True, cache=True)
    def __derivative_1(t, half_support, M):  # pragma: no cover
        t += half_support

        alpha = np.pi / M
        L = 1 / (4 * np.sin(alpha) ** 2)

        val = 0
        if t >= 0 and t <= 1:
            val = 4 * alpha * np.sin(alpha * t) * np.cos(alpha * t)
        elif t > 1 and t <= 2:
            val = 2 * alpha * (np.sin(2 * alpha * (2 - t)) + np.sin(2 * alpha * (1 - t)))
        elif t > 2 and t <= 3:
            val = 4 * alpha * np.sin(alpha * (t - 3)) * np.cos(alpha * (t - 3))

        return L * val

    def _derivative_2(self, t):
        return self.__derivative_2(t, self.support / 2, self.M)

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64)], nopython=True, cache=True)
    def __derivative_2(t, half_support, M):  # pragma: no cover
        t += half_support

        alpha = np.pi / M
        L = 1 / (4 * np.sin(alpha) ** 2)

        val = 0
        if t >= 0 and t <= 1:
            val = 4 * alpha**2 * (np.cos(alpha * t) ** 2 - np.sin(alpha * t) ** 2)
        elif t > 1 and t <= 2:
            val = -4 * alpha**2 * (np.cos(2 * alpha * (2 - t)) + np.cos(2 * alpha * (1 - t)))
        elif t > 2 and t <= 3:
            val = 4 * alpha**2 * (np.cos(alpha * (t - 3)) ** 2 - np.sin(alpha * (t - 3)) ** 2)

        return L * val

    def filter_symmetric(self, s):
        self.M = len(s)

        b0 = self._func(0)
        b1 = self._func(1)

        return self._filter_symmetric(s, self.M, b0, b1)

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def _filter_symmetric(s, M, b0, b1):  # pragma: no cover
        ndim = 1 if s.ndim == 1 else s.shape[1]
        pole = (-b0 + np.sqrt(2 * b0 - 1)) / (1 - b0)

        cp = np.zeros((M, ndim))
        eps = 1e-8
        k0 = min(
            (
                (2 * M) - 2,
                int(np.ceil(np.log(eps) / np.log(np.abs(pole)))),
            )
        )
        for k in np.arange(k0):
            m = k % (2 * M - 2)
            val = s[2 * M - 2 - m] if m >= M else s[m]
            cp[0] += val * (pole**m)
        cp[0] *= 1 / (1 - (pole ** (2 * M - 2)))

        for k in range(1, M):
            cp[k] = s[k] + pole * cp[k - 1]

        cm = np.zeros((M, ndim))
        cm[M - 1] = cp[M - 1] + (pole * cp[M - 2])
        cm[M - 1] *= pole / ((pole * pole) - 1)
        for k in range(M - 2, -1, -1):
            cm[k] = pole * (cm[k + 1] - cp[k])

        c = cm / b1

        shape = c.shape
        c = c.flatten()
        c[np.abs(c) < eps] = 0
        c = c.reshape(shape)
        return c

    def filter_periodic(self, s):
        self.M = len(s)
        b0 = self._func(0)
        return self._filter_periodic(s, self.M, b0)

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def _filter_periodic(s, M, b0):  # pragma: no cover
        ndim = 1 if s.ndim == 1 else s.shape[1]

        pole = (-b0 + np.sqrt(2 * b0 - 1)) / (1 - b0)

        cp = np.zeros((M, ndim))
        cp[0] = s[0]
        for k in range(1, M):
            cp[0] += s[k] * (pole ** (M - k))
        cp[0] *= 1 / (1 - (pole**M))

        for k in range(1, M):
            cp[k] = s[k] + (pole * cp[k - 1])

        cm = np.zeros((M, ndim))
        cm[M - 1] = cp[M - 1]
        for k in range(M - 1):
            cm[M - 1] += cp[k] * (pole ** (k + 1))
        cm[M - 1] *= 1 / (1 - (pole**M))
        cm[M - 1] *= (1 - pole) ** 2

        for k in range(M - 2, -1, -1):
            cm[k] = (pole * cm[k + 1]) + (((1 - pole) ** 2) * cp[k])

        c = cm

        eps = 1e-8
        shape = c.shape
        c = c.flatten()
        c[np.abs(c) < eps] = 0
        c = c.reshape(shape)
        return c

    def refinement_mask(self):
        """
        TODO: This fails the tests. They are set to
        xfail for now but this should be checked at some point.
        """
        warnings.warn(
            "The refinement_mask method of Exponential currently fails our test. This has to be investigated. Double check your results when using it.",
            stacklevel=2,
        )
        order = int(self.support)
        mask = np.zeros(order + 1)

        denominator = 2 ** (order - 1)
        mask[0] = 1 / denominator
        mask[1] = (2 * np.cos(np.pi / self.M) + 1) / denominator
        mask[2] = mask[1]
        mask[3] = 1 / denominator

        return mask


class CatmullRom(BasisFunction):
    r"""
    Basis function for a Catmull Rom spline.

    For a detailed theoretical description, including the equation and
    a plot of the function, refer to the :ref:`Catmull Rom basis`
    section in the documentation.

    The constructor does not require any arguments.

    For more information on the methods and attributes available in this class,
    please see the documentation for :class:`splinebox.basis_functions.BasisFunction`.
    """

    def __init__(self):
        super().__init__(False, 4)

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _func(t):  # pragma: no cover
        val = 0
        if np.abs(t) >= 0 and np.abs(t) <= 1:
            val = (3 / 2) * (np.abs(t) ** 3) - (5 / 2) * (np.abs(t) ** 2) + 1
        elif np.abs(t) > 1 and np.abs(t) <= 2:
            val = (-1 / 2) * (np.abs(t) ** 3) + (5 / 2) * (np.abs(t) ** 2) - 4 * np.abs(t) + 2
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _derivative_1(t):  # pragma: no cover
        val = 0
        if t >= 0 and t <= 1:
            val = t * (4.5 * t - 5)
        elif t >= -1 and t < 0:
            val = -t * (4.5 * t + 5)
        elif t > 1 and t <= 2:
            val = -1.5 * t * t + 5 * t - 4
        elif t >= -2 and t < -1:
            val = 1.5 * t * t + 5 * t + 4
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _derivative_2(t):  # pragma: no cover
        val = 0
        if t >= 0 and t <= 1:
            val = 9 * t - 5
        elif t >= -1 and t < 0:
            val = -9 * t - 5
        elif t > 1 and t <= 2:
            val = -3 * t + 5
        elif t >= -2 and t < -1:
            val = 3 * t + 5
        return val

    @staticmethod
    def filter_symmetric(s):
        return s

    @staticmethod
    def filter_periodic(s):
        return s


class CubicHermite(BasisFunction):
    r"""
    Basis function for a cubic Hermite spline.

    For a detailed theoretical description, including the equation and
    a plot of the function, refer to the :ref:`Cubic Hermite basis`
    section in the documentation.

    The constructor does not require any arguments.

    For more information on the methods and attributes available in this class,
    please see the documentation for :class:`splinebox.basis_functions.BasisFunction`.

    **Note**: This is basis function is a :class:`multigenerator <splinebox.basis_functions.BasisFunction>` and
    :func:`eval <splinebox.basis_functions.BasisFunction.eval>` returns two values.
    """

    def __init__(self):
        super().__init__(True, 2)

    def _func(self, t):
        return np.array([self.h31(t), self.h32(t)])

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def h31(t):  # pragma: no cover
        val = 0
        if t >= 0 and t <= 1:
            val = (1 + (2 * t)) * (t - 1) * (t - 1)
        elif t < 0 and t >= -1:
            val = (1 - (2 * t)) * (t + 1) * (t + 1)
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def h32(t):  # pragma: no cover
        val = 0
        if t >= 0 and t <= 1:
            val = t * (t - 1) * (t - 1)
        elif t < 0 and t >= -1:
            val = t * (t + 1) * (t + 1)
        return val

    def _derivative_1(self, t):
        return np.array([self.h31prime(t), self.h32prime(t)])

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def h31prime(t):  # pragma: no cover
        val = 0
        if t >= 0 and t <= 1:
            val = 6 * t * (t - 1)
        elif t < 0 and t >= -1:
            val = -6 * t * (t + 1)
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def h32prime(t):  # pragma: no cover
        val = 0
        if t >= 0 and t <= 1:
            val = 3 * t * t - 4 * t + 1
        elif t < 0 and t >= -1:
            val = 3 * t * t + 4 * t + 1
        return val

    @staticmethod
    def _derivative_2(t):
        raise RuntimeError("CubicHermite isn't twice differentiable.")

    def h31_autocorrelation(self, i, j, M):  # pragma: no cover
        """
        Derived by V. Uhlman during her PhD.
        """
        warnings.warn("This function is untested.", stacklevel=2)
        if self.support > M:
            raise ValueError("Cannot compute h31Autocorrelation for M<" + str(self.support))
            return

        val = 0
        if np.abs(i - j) == 1:
            val = 9 / ((M - 1) * 70)
        elif i == j:
            val = 13 / ((M - 1) * 35) if i == 0 or i == M - 1 else 26 / ((M - 1) * 35)

        return val

    def h31_periodic_autocorrelation(self, n, M):  # pragma: no cover
        """
        Derived by V. Uhlman during her PhD.
        """
        warnings.warn("This function is untested.", stacklevel=2)
        if self.support > M:
            raise ValueError("Cannot compute h31PeriodicAutocorrelation for M<" + str(self.support))
            return

        nmod = np.mod(n, M)
        val = 0
        if nmod == 0:
            val = 26 / (M * 35)
        elif (nmod == 1) or (nmod == M - 1):
            val = 9 / (M * 35) if M == 2 else 9 / (M * 70)

        return val

    def h32_autocorrelation(self, i, j, M):  # pragma: no cover
        """
        Derived by V. Uhlman during her PhD.
        """
        warnings.warn("This function is untested.", stacklevel=2)
        if self.support > M:
            raise ValueError("Cannot compute h32Autocorrelation for M<" + str(self.support))
            return

        val = 0
        if np.abs(i - j) == 1:
            val = -1 / ((M - 1) * 140)
        elif i == j:
            val = 1 / ((M - 1) * 105) if i == 0 or i == M - 1 else 2 / ((M - 1) * 105)

        return val

    def h32_periodic_autocorrelation(self, n, M):  # pragma: no cover
        """
        Derived by V. Uhlman during her PhD.
        """
        warnings.warn("This function is untested.", stacklevel=2)
        if self.support > M:
            raise ValueError("Cannot compute h32PeriodicAutocorrelation for M<" + str(self.support))
            return

        nmod = np.mod(n, M)
        val = 0
        if nmod == 0:
            val = 2 / (M * 105)
        elif (nmod == 1) or (nmod == M - 1):
            val = -1 / (M * 70) if M == 2 else -1 / (M * 140)

        return val

    def h3_crosscorrelation(self, i, j, M):  # pragma: no cover
        """
        Derived by V. Uhlman during her PhD.
        """
        warnings.warn("This function is untested.", stacklevel=2)
        if self.support > M:
            raise ValueError("Cannot compute h3Crosscorrelation for M<" + str(self.support))
            return

        val = 0
        if i - j == 1:
            val = 13 / ((M - 1) * 420)
        elif i - j == -1:
            val = -13 / ((M - 1) * 420)
        elif i == j:
            if i == 0:
                val = 11 / ((M - 1) * 210)
            elif i == M - 1:
                val = -11 / ((M - 1) * 210)

        return val

    def h3_periodic_crosscorrelation(self, n, M):  # pragma: no cover
        """
        Derived by V. Uhlman during her PhD.
        """
        warnings.warn("This function is untested.", stacklevel=2)
        if self.support > M:
            raise ValueError("Cannot compute h3PeriodicCrosscorrelation for M<" + str(self.support))
            return

        nmod = np.mod(n, M)
        val = 0
        if nmod == 1:
            if M != 2:
                val = 13 / (M * 420)
        elif nmod == M - 1:
            val = -13 / (M * 420)

        return val

    @staticmethod
    def filter_symmetric(s):
        return s

    @staticmethod
    def filter_periodic(s):
        return s


class ExponentialHermite(BasisFunction):
    r"""
    Basis function for an exponential Hermite spline.

    For a detailed theoretical description, including the equation and
    a plot of the function, refer to the :ref:`Exponential Hermite basis`
    section in the documentation.

    The constructor requires `M`, the number of knots in the spline, as an argument.

    For more information on the methods and attributes available in this class,
    please see the documentation for :class:`splinebox.basis_functions.BasisFunction`.

    **Note**: This is basis function is a :class:`multigenerator <splinebox.basis_functions.BasisFunction>` and
    :func:`eval <splinebox.basis_functions.BasisFunction.eval>` returns two values.
    """

    def __init__(self, M):
        super().__init__(True, 2)
        self.M = M

    def _func(self, t):
        return np.array([self._he31(t, self.M), self._he32(t, self.M)])

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64, numba.float64)], nopython=True, cache=True)
    def _he31(t, M):  # pragma: no cover
        def _g1(t, M):
            val = 0
            if t >= 0 and t <= 1:
                alpha = np.pi / M
                denom = (alpha * np.cos(alpha)) - np.sin(alpha)
                num = (
                    (0.5 * (2 * alpha * np.cos(alpha) - np.sin(alpha)))
                    - (alpha * np.cos(alpha) * t)
                    - (0.5 * np.sin(alpha - (2 * alpha * t)))
                )
                val = num / denom
            return val

        val = _g1(t, M) if t >= 0 else _g1(-t, M)
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64, numba.float64)], nopython=True, cache=True)
    def _he32(t, M):  # pragma: no cover
        def _g2(t, M):
            val = 0
            if t >= 0 and t <= 1:
                alpha = np.pi / M
                denom = ((alpha * np.cos(alpha)) - np.sin(alpha)) * 8 * alpha * np.sin(alpha)
                num = (
                    -((2 * alpha * np.cos(2 * alpha)) - np.sin(2 * alpha))
                    - (4 * alpha * np.sin(alpha) * np.sin(alpha) * t)
                    - (2 * np.sin(alpha) * np.cos(2 * alpha * (t - 0.5)))
                    + (2 * alpha * np.cos(2 * alpha * (t - 1)))
                )
                val = num / denom
            return val

        val = _g2(t, M) if t >= 0 else -1 * _g2(-t, M)
        return val

    def _derivative_1(self, t):
        return np.array([self._he31prime(t, self.M), self._he32prime(t, self.M)])

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64, numba.float64)], nopython=True, cache=True)
    def _he31prime(t, M):  # pragma: no cover
        def _g1prime(t, M):
            val = 0
            if t >= 0 and t <= 1:
                alpha = np.pi / M
                denom = (alpha * np.cos(alpha)) - np.sin(alpha)
                num = -(alpha * np.cos(alpha)) + (alpha * np.cos(alpha - (2 * alpha * t)))
                val = num / denom
            return val

        val = _g1prime(t, M) if t >= 0 else -1 * _g1prime(-t, M)
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64, numba.float64)], nopython=True, cache=True)
    def _he32prime(t, M):  # pragma: no cover
        def _g2prime(t, M):
            val = 0
            if t >= 0 and t <= 1:
                alpha = np.pi / M
                denom = ((alpha * np.cos(alpha)) - np.sin(alpha)) * 8 * alpha * np.sin(alpha)
                num = (
                    -(4 * alpha * np.sin(alpha) * np.sin(alpha))
                    + (4 * alpha * np.sin(alpha) * np.sin(2 * alpha * (t - 0.5)))
                    - (4 * alpha**2 * np.sin(2 * alpha * (t - 1)))
                )
                val = num / denom
            return val

        val = _g2prime(t, M) if t >= 0 else _g2prime(-t, M)
        return val

    @staticmethod
    def _derivative_2(x):
        raise RuntimeError("ExponentialHermite isn't twice differentiable.")

    @staticmethod
    def filter_symmetric(s):
        return s

    @staticmethod
    def filter_periodic(s):
        return s


def _multinomial(
    maxValue,
    numberOfCoefficiens,
    kArray,
    iteration,
    dilationFactor,
    order,
    mask,
):
    """
    Recursive function to compute the _multinomial coefficient of (x0+x1+...+xm-1)^N
    This function finds every {k0,...,km-1} such that k0+...+km-1=N
    (cf multinomial theorem on Wikipedia for a detailed explanation)
    """
    if numberOfCoefficiens == 1:
        kArray[iteration] = maxValue

        denominator = 1
        degree = 0
        for k in range(dilationFactor):
            denominator *= math.factorial(kArray[k])
            degree += k * kArray[k]

        coef = math.factorial(order) / denominator
        mask[int(degree)] += coef / (dilationFactor ** (order - 1))

    else:
        for k in range(maxValue + 1):
            kArray[iteration] = k
            _multinomial(
                maxValue - k,
                numberOfCoefficiens - 1,
                kArray,
                iteration + 1,
                dilationFactor,
                order,
                mask,
            )

    return
