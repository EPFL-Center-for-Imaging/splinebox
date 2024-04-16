"""
This module provides spline basis functions.
"""

import math

import numba
import numpy as np


class BasisFunction:
    _unimplemented_message = "This function is not implemented."

    def __init__(self, multigenerator, support):
        """
        Base class for all basis functions.

        Parameters
        ----------
        multigenerator : ???
        support : float or int?
            The support of the function, i.e. the size of the area
            being mapped to non-zero values.
        """
        self.multigenerator = multigenerator
        self.support = support

    def eval(self, x, derivative=0):
        """
        Evaluate the function at position(s) `x`.

        Parameters
        ----------
        x : numpy.array
            The points where the function should be evaluated.
        derivative : [0, 1, 2], default = 0
            Whether to evaluate the function (0) or its first (1)
            or second (2) derivative.

        Returns
        -------
        y : numpy.array
            Values of the function or its first or second derivative
            at position(s) `x`.
        """
        if derivative == 0:
            return self._func(x)
        elif derivative == 1:
            return self._derivative_1(x)
        elif derivative == 2:
            return self._derivative_2(x)
        else:
            raise ValueError(f"derivative has to be 0, 1, or 2 not {derivative}")

    def _func(self, x):
        raise NotImplementedError(BasisFunction._unimplemented_message)

    def _derivative_1(self, x):
        raise NotImplementedError(BasisFunction._unimplemented_message)

    def _derivative_2(self, x):
        raise NotImplementedError(BasisFunction._unimplemented_message)

    def filter_symmetric(self, s):
        """
        ???

        Parameters
        ----------
        s : ?
            ?
        """
        raise NotImplementedError(BasisFunction._unimplemented_message)

    def filter_periodic(self, s):
        """
        ???

        Parameters
        ----------
        s : ?
            ?
        """
        raise NotImplementedError(BasisFunction._unimplemented_message)

    def refinement_mask(self):
        """
        ???
        """
        raise NotImplementedError(BasisFunction._unimplemented_message)


class B1(BasisFunction):
    r"""
    Basis spline of degree 1.

    .. math::
        f(x) = \begin{cases}1 - \lvert x \rvert & \text{for } -1 \leq x \leq 1 \\ 0 & \text{otherwise}\end{cases}
    """

    def __init__(self):
        super().__init__(False, 2)

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _func(x):
        val = 0
        if abs(x) >= 0 and abs(x) < 1:
            val = 1 - abs(x)
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _derivative_1(x):
        val = 0
        if x > -1 and x < 0:
            val = 1
        elif x > 0 and x < 1:
            val = -1
        elif x == 0 or x == -1 or x == 1:
            # This is the gradient you'll get at exactly 0
            val = np.nan
        return val

    @staticmethod
    def _derivative_2(x):
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
        _multinomial(order, 2, np.zeros(2), 0, 2, order, mask)
        return mask


class B2(BasisFunction):
    r"""
    Basis spline of degree 2.

    .. math::
        f(x) = \begin{cases}\frac{x^2}{2} + \frac{3}{2} x + \frac{9}{8} & \text{for } -\frac{3}{2} \leq x \leq -\frac{1}{2} \\ -x^2 + \frac{3}{4} & \text{for } -\frac{1}{2} < x \leq \frac{1}{2} \\ \frac{1}{2} x^2 - \frac{3}{2} x + \frac{9}{8} & \text{for } \frac{1}{2} < x \leq \frac{3}{2} \\ 0 & \text{otherwise}\end{cases}
    """

    def __init__(self):
        super().__init__(False, 3)

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _func(x):
        val = 0
        if x >= -1.5 and x <= -0.5:
            val = 0.5 * (x**2) + 1.5 * x + 1.125
        elif x > -0.5 and x <= 0.5:
            val = -x * x + 0.75
        elif x > 0.5 and x <= 1.5:
            val = 0.5 * (x**2) - 1.5 * x + 1.125
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _derivative_1(x):
        val = 0
        if x >= -1.5 and x <= -0.5:
            val = x + 1.5
        elif x > -0.5 and x <= 0.5:
            val = -2 * x
        elif x > 0.5 and x <= 1.5:
            val = x - 1.5
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _derivative_2(x):
        val = 0
        if x >= -1.5 and x <= -0.5:
            val = 1
        elif x > -0.5 and x <= 0.5:
            val = -2
        elif x > 0.5 and x <= 1.5:
            val = 1
        return val

    def refinement_mask(self):
        order = int(self.support)
        mask = np.zeros(order + 1)
        _multinomial(order, 2, np.zeros(2), 0, 2, order, mask)
        return mask


class B3(BasisFunction):
    r"""
    Basis spline of degree 3.

    .. math::
        f(x) = \begin{cases}
        \frac{2}{3} - \lvert x \rvert^2 + \frac{\lvert x \rvert^3}{2} & \text{for } 0 \leq \lvert x \rvert < 1 \\
        \frac{1}{6}(2 - \lvert x \rvert)^3 & \text{for } 1 \leq \lvert x \rvert \leq 2 \\
        0 & \text{otherwise}\end{cases}
    """

    def __init__(self):
        super().__init__(False, 4)

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _func(x):
        val = 0
        if abs(x) >= 0 and abs(x) < 1:
            val = 2 / 3 - (abs(x) ** 2) + (abs(x) ** 3) / 2
        elif abs(x) >= 1 and abs(x) <= 2:
            val = ((2 - abs(x)) ** 3) / 6
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _derivative_1(x):
        val = 0
        if x >= 0 and x < 1:
            val = -2 * x + 1.5 * x * x
        elif x > -1 and x < 0:
            val = -2 * x - 1.5 * x * x
        elif x >= 1 and x <= 2:
            val = -0.5 * ((2 - x) ** 2)
        elif x >= -2 and x <= -1:
            val = 0.5 * ((2 + x) ** 2)
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _derivative_2(x):
        val = 0
        if x >= 0 and x < 1:
            val = -2 + 3 * x
        elif x > -1 and x < 0:
            val = -2 - 3 * x
        elif x >= 1 and x <= 2:
            val = 2 - x
        elif x >= -2 and x <= -1:
            val = 2 + x
        return val

    @staticmethod
    def filter_symmetric(s):
        M = len(s)
        pole = -2 + np.sqrt(3)

        ndim = 1 if s.ndim == 1 else s.shape[1]

        cp = np.zeros((M, ndim))
        eps = 1e-8
        k0 = np.min(((2 * M) - 2, int(np.ceil(np.log(eps) / np.log(np.abs(pole))))))
        for k in range(k0):
            k = k % (2 * M - 2)
            val = s[2 * M - 2 - k] if k >= M else s[k]
            cp[0] += val * (pole**k)
        cp[0] *= 1 / (1 - (pole ** (2 * M - 2)))

        for k in range(1, M):
            cp[k] = s[k] + pole * cp[k - 1]

        cm = np.zeros((M, ndim))
        cm[M - 1] = cp[M - 1] + (pole * cp[M - 2])
        cm[M - 1] *= pole / ((pole**2) - 1)
        for k in range(M - 2, -1, -1):
            cm[k] = pole * (cm[k + 1] - cp[k])

        c = cm * 6

        c[np.where(abs(c) < eps)] = 0
        return np.squeeze(c)

    @staticmethod
    def filter_periodic(s):
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
        c[np.where(abs(c) < eps)] = 0
        return np.squeeze(c)

    def refinement_mask(self):
        order = int(self.support)
        mask = np.zeros(order + 1)
        _multinomial(order, 2, np.zeros(2), 0, 2, order, mask)
        return mask


class Exponential(BasisFunction):
    r"""
    ???

    .. math::
        f(x) = \begin{cases}
        \frac{L}{\alpha^2} 2 \sin(\frac{\alpha}{2} x) & \text{for } 0 \leq x < 1 \\
        \frac{L}{\alpha^2} (\cos(\alpha (x - 2)) + \cos(\alpha (x - 1)) - 2 \cos(\alpha)) & \text{for } 1 \leq x < 2 \\
        \frac{L}{\alpha^2} 2 \sin(\frac{\alpha}{2} (x- 3))^2 & \text{for } 2 \leq x \leq 3 \\
        0 & \text{otherwise}\end{cases}

    .. math::
        \text{where } L=(\frac{\sin(\pi / M)}{\pi / M})^{-2}
    """

    def __init__(self, M, alpha):
        super().__init__(False, 3)
        self.M = M
        self.alpha = alpha

    def _func(self, x):
        return self.__func(x, self.support / 2, self.M, self.alpha)

    @staticmethod
    @numba.vectorize(
        [numba.float64(numba.float64, numba.float64, numba.float64, numba.float64)], nopython=True, cache=True
    )
    def __func(x, half_support, M, alpha):
        x += half_support
        L = (np.sin(np.pi / M) / (np.pi / M)) ** (-2)

        val = 0
        if x >= 0 and x < 1:
            val = 2 * np.sin(alpha * 0.5 * x) * np.sin(alpha * 0.5 * x)
        elif x >= 1 and x < 2:
            val = np.cos(alpha * (x - 2)) + np.cos(alpha * (x - 1)) - 2 * np.cos(alpha)
        elif x >= 2 and x <= 3:
            val = 2 * np.sin(alpha * 0.5 * (x - 3)) * np.sin(alpha * 0.5 * (x - 3))

        return (L * val) / (alpha * alpha)

    def _derivative_1(self, x):
        return self.__derivative_1(x, self.support / 2, self.M, self.alpha)

    @staticmethod
    @numba.vectorize(
        [numba.float64(numba.float64, numba.float64, numba.float64, numba.float64)], nopython=True, cache=True
    )
    def __derivative_1(x, half_support, M, alpha):
        x += half_support
        L = (np.sin(np.pi / M) / (np.pi / M)) ** (-2)

        val = 0
        if x >= 0 and x <= 1:
            val = alpha * np.sin(alpha * x)
        elif x > 1 and x <= 2:
            val = alpha * (np.sin(alpha * (1 - x)) + np.sin(alpha * (2 - x)))
        elif x > 2 and x <= 3:
            val = alpha * np.sin(alpha * (x - 3))

        return (L * val) / (alpha * alpha)

    def _derivative_2(self, x):
        return self.__derivative_2(x, self.support / 2, self.M, self.alpha)

    @staticmethod
    @numba.vectorize(
        [numba.float64(numba.float64, numba.float64, numba.float64, numba.float64)], nopython=True, cache=True
    )
    def __derivative_2(x, half_support, M, alpha):
        x += half_support
        L = (np.sin(np.pi / M) / (np.pi / M)) ** (-2)

        val = 0
        if x >= 0 and x <= 1:
            val = alpha * alpha * np.cos(alpha * x)
        elif x > 1 and x <= 2:
            val = alpha * alpha * (-np.cos(alpha * (1 - x)) - np.cos(alpha * (2 - x)))
        elif x > 2 and x <= 3:
            val = alpha * alpha * np.cos(alpha * (x - 3))

        return (L * val) / (alpha * alpha)

    def filter_symmetric(self, s):
        self.M = len(s)

        ndim = 1 if s.ndim == 1 else s.shape[1]

        b0 = self._func(0)
        b1 = self._func(1)
        pole = (-b0 + np.sqrt(2 * b0 - 1)) / (1 - b0)

        cp = np.zeros((self.M, ndim))
        eps = 1e-8
        k0 = np.min(
            (
                (2 * self.M) - 2,
                int(np.ceil(np.log(eps) / np.log(np.abs(pole)))),
            )
        )
        for k in range(k0):
            k = k % (2 * self.M - 2)
            val = s[2 * self.M - 2 - k] if k >= self.M else s[k]
            cp[0] += val * (pole**k)
        cp[0] *= 1 / (1 - (pole ** (2 * self.M - 2)))

        for k in range(1, self.M):
            cp[k] = s[k] + pole * cp[k - 1]

        cm = np.zeros((self.M, ndim))
        cm[self.M - 1] = cp[self.M - 1] + (pole * cp[self.M - 2])
        cm[self.M - 1] *= pole / ((pole * pole) - 1)
        for k in range(self.M - 2, -1, -1):
            cm[k] = pole * (cm[k + 1] - cp[k])

        c = cm / b1

        c[np.where(np.abs(c) < eps)] = 0
        return np.squeeze(c)

    def filter_periodic(self, s):
        self.M = len(s)

        ndim = 1 if s.ndim == 1 else s.shape[1]

        b0 = self._func(0)
        pole = (-b0 + np.sqrt(2 * b0 - 1)) / (1 - b0)

        cp = np.zeros((self.M, ndim))
        cp[0] = s[0]
        for k in range(self.M):
            cp[0] += s[k] * (pole ** (self.M - k))
        cp[0] *= 1 / (1 - (pole**self.M))

        for k in range(1, self.M):
            cp[k] = s[k] + (pole * cp[k - 1])

        cm = np.zeros((self.M, ndim))
        cm[self.M - 1] = cp[self.M - 1]
        for k in range(self.M - 1):
            cm[self.M - 1] += cp[k] * (pole ** (k + 1))
        cm[self.M - 1] *= 1 / (1 - (pole**self.M))
        cm[self.M - 1] *= (1 - pole) ** 2

        for k in range(self.M - 2, -1, -1):
            cm[k] = (pole * cm[k + 1]) + (((1 - pole) ** 2) * cp[k])

        c = cm

        eps = 1e-8
        c[np.where(np.abs(c) < eps)] = 0
        return np.squeeze(c)

    def refinement_mask(self):
        order = int(self.support)
        mask = np.zeros(order + 1)

        denominator = 2 ** (order - 1)
        mask[0] = 1 / denominator
        mask[1] = (2 * np.cos(self.alpha) + 1) / denominator
        mask[2] = mask[1]
        mask[3] = 1 / denominator

        return mask


class CatmullRom(BasisFunction):
    """
    ???
    """

    def __init__(self):
        super().__init__(False, 4)

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _func(x):
        val = 0
        if np.abs(x) >= 0 and np.abs(x) <= 1:
            val = (3 / 2) * (np.abs(x) ** 3) - (5 / 2) * (np.abs(x) ** 2) + 1
        elif np.abs(x) > 1 and np.abs(x) <= 2:
            val = (-1 / 2) * (np.abs(x) ** 3) + (5 / 2) * (np.abs(x) ** 2) - 4 * np.abs(x) + 2
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _derivative_1(x):
        val = 0
        if x >= 0 and x <= 1:
            val = x * (4.5 * x - 5)
        elif x >= -1 and x < 0:
            val = -x * (4.5 * x + 5)
        elif x > 1 and x <= 2:
            val = -1.5 * x * x + 5 * x - 4
        elif x >= -2 and x < -1:
            val = 1.5 * x * x + 5 * x + 4
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def _derivative_2(x):
        val = 0
        if x >= 0 and x <= 1:
            val = 9 * x - 5
        elif x >= -1 and x < 0:
            val = -9 * x - 5
        elif x > 1 and x <= 2:
            val = -3 * x + 5
        elif x >= -2 and x < -1:
            val = 3 * x + 5
        return val

    @staticmethod
    def filter_symmetric(s):
        return s

    @staticmethod
    def filter_periodic(s):
        return s


class CubicHermite(BasisFunction):
    """
    ???
    """

    def __init__(self):
        super().__init__(True, 2)

    def _func(self, x):
        return np.array([self.h31(x), self.h32(x)])

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def h31(x):
        val = 0
        if x >= 0 and x <= 1:
            val = (1 + (2 * x)) * (x - 1) * (x - 1)
        elif x < 0 and x >= -1:
            val = (1 - (2 * x)) * (x + 1) * (x + 1)
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def h32(x):
        val = 0
        if x >= 0 and x <= 1:
            val = x * (x - 1) * (x - 1)
        elif x < 0 and x >= -1:
            val = x * (x + 1) * (x + 1)
        return val

    def _derivative_1(self, x):
        return np.array([self.h31prime(x), self.h32prime(x)])

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def h31prime(x):
        val = 0
        if x >= 0 and x <= 1:
            val = 6 * x * (x - 1)
        elif x < 0 and x >= -1:
            val = -6 * x * (x + 1)
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)], nopython=True, cache=True)
    def h32prime(x):
        val = 0
        if x >= 0 and x <= 1:
            val = 3 * x * x - 4 * x + 1
        elif x < 0 and x >= -1:
            val = 3 * x * x + 4 * x + 1
        return val

    @staticmethod
    def _derivative_2(x):
        raise RuntimeError("CubicHermite isn't twice differentiable.")

    def h31Autocorrelation(self, i, j, M):
        if self.support > M:
            raise ValueError("Cannot compute h31Autocorrelation for M<" + str(self.support))
            return

        val = 0
        if np.abs(i - j) == 1:
            val = 9 / ((M - 1) * 70)
        elif i == j:
            val = 13 / ((M - 1) * 35) if i == 0 or i == M - 1 else 26 / ((M - 1) * 35)

        return val

    def h31PeriodicAutocorrelation(self, n, M):
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

    def h32Autocorrelation(self, i, j, M):
        if self.support > M:
            raise ValueError("Cannot compute h32Autocorrelation for M<" + str(self.support))
            return

        val = 0
        if np.abs(i - j) == 1:
            val = -1 / ((M - 1) * 140)
        elif i == j:
            val = 1 / ((M - 1) * 105) if i == 0 or i == M - 1 else 2 / ((M - 1) * 105)

        return val

    def h32PeriodicAutocorrelation(self, n, M):
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

    def h3Crosscorrelation(self, i, j, M):
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

    def h3PeriodicCrosscorrelation(self, n, M):
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
    """
    ???
    """

    def __init__(self, alpha):
        super().__init__(True, 2)
        self.alpha = alpha

    def _func(self, x):
        return np.array([self._he31(x, self.alpha), self._he32(x, self.alpha)])

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64, numba.float64)], nopython=True, cache=True)
    def _he31(x, alpha):
        def _g1(x, alpha):
            val = 0
            if x >= 0 and x <= 1:
                denom = (0.5 * alpha * np.cos(0.5 * alpha)) - np.sin(0.5 * alpha)
                num = (
                    (0.5 * ((alpha * np.cos(0.5 * alpha)) - np.sin(0.5 * alpha)))
                    - (0.5 * alpha * np.cos(0.5 * alpha) * x)
                    - (0.5 * np.sin(0.5 * alpha - (alpha * x)))
                )
                val = num / denom
            return val

        val = _g1(x, alpha) if x >= 0 else _g1(-x, alpha)
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64, numba.float64)], nopython=True, cache=True)
    def _he32(x, alpha):
        def _g2(x, alpha):
            val = 0
            if x >= 0 and x <= 1:
                denom = ((0.5 * alpha * np.cos(0.5 * alpha)) - np.sin(0.5 * alpha)) * (4 * alpha) * np.sin(0.5 * alpha)
                num = (
                    -((alpha * np.cos(alpha)) - np.sin(alpha))
                    - (2 * alpha * np.sin(0.5 * alpha) * np.sin(0.5 * alpha) * x)
                    - (2 * np.sin(0.5 * alpha) * np.cos(alpha * (x - 0.5)))
                    + (alpha * np.cos(alpha * (x - 1)))
                )
                val = num / denom
            return val

        val = _g2(x, alpha) if x >= 0 else -1 * _g2(-x, alpha)
        return val

    def _derivative_1(self, x):
        return np.array([self._he31prime(x, self.alpha), self._he32prime(x, self.alpha)])

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64, numba.float64)], nopython=True, cache=True)
    def _he31prime(x, alpha):
        def _g1prime(x, alpha):
            val = 0
            if x >= 0 and x <= 1:
                denom = (0.5 * alpha * np.cos(0.5 * alpha)) - np.sin(0.5 * alpha)
                num = -(0.5 * alpha * np.cos(0.5 * alpha)) + (0.5 * alpha * np.cos(0.5 * alpha - (alpha * x)))
                val = num / denom
            return val

        val = _g1prime(x, alpha) if x >= 0 else -1 * _g1prime(-x, alpha)
        return val

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64, numba.float64)], nopython=True, cache=True)
    def _he32prime(x, alpha):
        def _g2prime(x, alpha):
            val = 0
            if x >= 0 and x <= 1:
                denom = ((0.5 * alpha * np.cos(0.5 * alpha)) - np.sin(0.5 * alpha)) * (4 * alpha) * np.sin(0.5 * alpha)
                num = (
                    -(2 * alpha * np.sin(0.5 * alpha) * np.sin(0.5 * alpha))
                    + (2 * alpha * np.sin(0.5 * alpha) * np.sin(alpha * (x - 0.5)))
                    - (alpha * alpha * np.sin(alpha * (x - 1)))
                )
                val = num / denom
            return val

        val = _g2prime(x, alpha) if x >= 0 else _g2prime(-x, alpha)
        return val

    @staticmethod
    def _derivative_2(x):
        raise RuntimeError("ExponentialHermite isn't twice differentiable.")
        return

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
