import math

import numpy as np

# TODO: local refinement filters
# TODO: quadratic prefilters


class SplineGenerator:
    unimplementedMessage = "This function is not implemented."

    def __init__(self, multigenerator, support):
        self.multigenerator = multigenerator
        self.support = support

    def value(self, x):
        # This needs to be overloaded
        raise NotImplementedError(SplineGenerator.unimplementedMessage)
        return

    def firstDerivativeValue(self, x):
        # This needs to be overloaded
        raise NotImplementedError(SplineGenerator.unimplementedMessage)
        return

    def secondDerivativeValue(self, x):
        # This needs to be overloaded
        raise NotImplementedError(SplineGenerator.unimplementedMessage)
        return

    def filterSymmetric(self, s):
        # This needs to be overloaded
        raise NotImplementedError(SplineGenerator.unimplementedMessage)
        return

    def filterPeriodic(self, s):
        # This needs to be overloaded
        raise NotImplementedError(SplineGenerator.unimplementedMessage)
        return

    def refinementMask(self):
        # This needs to be overloaded
        raise NotImplementedError(SplineGenerator.unimplementedMessage)
        return


class B1(SplineGenerator):
    def __init__(self):
        SplineGenerator.__init__(self, False, 2.0)

    def value(self, x):
        val = 0.0
        if abs(x) >= 0 and abs(x) < 1:
            val = 1.0 - abs(x)
        return val

    def firstDerivativeValue(self, x):
        val = 0.0
        if x >= -1.0 and x < 0:
            val = -1.0
        elif x >= 0 and x <= 1:
            val = 1.0
        return val

    def secondDerivativeValue(self, x):
        raise RuntimeError("B1 isn't twice differentiable.")
        return

    def filterSymmetric(self, s):
        return s

    def filterPeriodic(self, s):
        return s

    def refinementMask(self):
        order = int(self.support)
        mask = np.zeros(order + 1)
        multinomial(order, 2, np.zeros(2), 0, 2, order, mask)
        return mask


class B2(SplineGenerator):
    def __init__(self):
        SplineGenerator.__init__(self, False, 3.0)

    def value(self, x):
        val = 0.0
        if x >= -1.5 and x <= -0.5:
            val = 0.5 * (x**2) + 1.5 * x + 1.125
        elif x > -0.5 and x <= 0.5:
            val = -x * x + 0.75
        elif x > 0.5 and x <= 1.5:
            val = 0.5 * (x**2) - 1.5 * x + 1.125
        return val

    def firstDerivativeValue(self, x):
        val = 0.0
        if x >= -1.5 and x <= -0.5:
            val = x + 1.5
        elif x > -0.5 and x <= 0.5:
            val = -2.0 * x
        elif x > 0.5 and x <= 1.5:
            val = x - 1.5
        return val

    def secondDerivativeValue(self, x):
        val = 0.0
        if x >= -1.5 and x <= -0.5:
            val = 1.0
        elif x > -0.5 and x <= 0.5:
            val = -2.0
        elif x > 0.5 and x <= 1.5:
            val = 1.0
        return val

    def refinementMask(self):
        order = int(self.support)
        mask = np.zeros(order + 1)
        multinomial(order, 2, np.zeros(2), 0, 2, order, mask)
        return mask


class B3(SplineGenerator):
    def __init__(self):
        SplineGenerator.__init__(self, False, 4.0)

    def value(self, x):
        val = 0.0
        if abs(x) >= 0 and abs(x) < 1:
            val = 2.0 / 3.0 - (abs(x) ** 2) + (abs(x) ** 3) / 2.0
        elif abs(x) >= 1 and abs(x) <= 2:
            val = ((2.0 - abs(x)) ** 3) / 6.0
        return val

    def firstDerivativeValue(self, x):
        val = 0.0
        if x >= 0 and x < 1:
            val = -2.0 * x + 1.5 * x * x
        elif x > -1 and x < 0:
            val = -2.0 * x - 1.5 * x * x
        elif x >= 1 and x <= 2:
            val = -0.5 * ((2.0 - x) ** 2)
        elif x >= -2 and x <= -1:
            val = 0.5 * ((2.0 + x) ** 2)
        return val

    def secondDerivativeValue(self, x):
        val = 0.0
        if x >= 0 and x < 1:
            val = -2.0 + 3.0 * x
        elif x > -1 and x < 0:
            val = -2.0 - 3.0 * x
        elif x >= 1 and x <= 2:
            val = 2.0 - x
        elif x >= -2 and x <= -1:
            val = 2.0 + x
        return val

    def filterSymmetric(self, s):
        M = len(s)
        pole = -2.0 + np.sqrt(3.0)

        cp = np.zeros(M)
        eps = 1e-8
        k0 = np.min(
            ((2 * M) - 2, int(np.ceil(np.log(eps) / np.log(np.abs(pole)))))
        )
        for k in range(k0):
            k = k % (2 * M - 2)
            val = s[2 * M - 2 - k] if k >= M else s[k]
            cp[0] += val * (pole**k)
        cp[0] *= 1.0 / (1.0 - (pole ** (2 * M - 2)))

        for k in range(1, M):
            cp[k] = s[k] + pole * cp[k - 1]

        cm = np.zeros(M)
        cm[M - 1] = cp[M - 1] + (pole * cp[M - 2])
        cm[M - 1] *= pole / ((pole**2) - 1)
        for k in range(M - 2, -1, -1):
            cm[k] = pole * (cm[k + 1] - cp[k])

        c = cm * 6.0

        c[np.where(abs(c) < eps)] = 0.0
        return c

    def filterPeriodic(self, s):
        M = len(s)
        pole = -2.0 + np.sqrt(3.0)

        cp = np.zeros(M)
        for k in range(M):
            cp[0] += s[(M - k) % M] * (pole**k)
        cp[0] *= 1.0 / (1.0 - (pole**M))

        for k in range(1, M):
            cp[k] = s[k] + pole * cp[k - 1]

        cm = np.zeros(M)
        for k in range(M):
            cm[M - 1] += (pole**k) * cp[k]
        cm[M - 1] *= pole / (1.0 - (pole**M))
        cm[M - 1] += cp[M - 1]
        cm[M - 1] *= -pole

        for k in range(M - 2, -1, -1):
            cm[k] = pole * (cm[k + 1] - cp[k])

        c = cm * 6.0

        eps = 1e-8
        c[np.where(abs(c) < eps)] = 0.0
        return c

    def refinementMask(self):
        order = int(self.support)
        mask = np.zeros(order + 1)
        multinomial(order, 2, np.zeros(2), 0, 2, order, mask)
        return mask


class EM(SplineGenerator):
    def __init__(self, M, alpha):
        SplineGenerator.__init__(self, False, 3.0)
        self.M = M
        self.alpha = alpha

    def value(self, x):
        x += self.support / 2.0
        L = (np.sin(np.pi / self.M) / (np.pi / self.M)) ** (-2)

        val = 0.0
        if x >= 0 and x < 1:
            val = (
                2.0
                * np.sin(self.alpha * 0.5 * x)
                * np.sin(self.alpha * 0.5 * x)
            )
        elif x >= 1 and x < 2:
            val = (
                np.cos(self.alpha * (x - 2))
                + np.cos(self.alpha * (x - 1))
                - 2.0 * np.cos(self.alpha)
            )
        elif x >= 2 and x <= 3:
            val = (
                2.0
                * np.sin(self.alpha * 0.5 * (x - 3))
                * np.sin(self.alpha * 0.5 * (x - 3))
            )

        return (L * val) / (self.alpha * self.alpha)

    def firstDerivativeValue(self, x):
        x += self.support / 2.0
        L = (np.sin(np.pi / self.M) / (np.pi / self.M)) ** (-2)

        val = 0.0
        if x >= 0 and x <= 1:
            val = self.alpha * np.sin(self.alpha * x)
        elif x > 1 and x <= 2:
            val = self.alpha * (
                np.sin(self.alpha * (1 - x)) + np.sin(self.alpha * (2 - x))
            )
        elif x > 2 and x <= 3:
            val = self.alpha * np.sin(self.alpha * (x - 3))

        return (L * val) / (self.alpha * self.alpha)

    def secondDerivativeValue(self, x):
        x += self.support() / 2.0
        L = (np.sin(np.pi / self.M) / (np.pi / self.M)) ** (-2)

        val = 0.0
        if x >= 0 and x <= 1:
            val = self.alpha * self.alpha * np.cos(self.alpha * x)
        elif x > 1 and x <= 2:
            val = (
                self.alpha
                * self.alpha
                * (
                    -np.cos(self.alpha * (1 - x))
                    - np.cos(self.alpha * (2 - x))
                )
            )
        elif x > 2 and x <= 3:
            val = self.alpha * self.alpha * self.cos(self.alpha * (x - 3))

        return (L * val) / (self.alpha * self.alpha)

    def filterSymmetric(self, s):
        self.M = len(s)
        b0 = self.value(0)
        b1 = self.value(1)
        pole = (-b0 + np.sqrt(2.0 * b0 - 1.0)) / (1.0 - b0)

        cp = np.zeros(self.M)
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
        cp[0] *= 1.0 / (1.0 - (pole ** (2 * self.M - 2)))

        for k in range(1, self.M):
            cp[k] = s[k] + pole * cp[k - 1]

        cm = np.zeros(self.M)
        cm[self.M - 1] = cp[self.M - 1] + (pole * cp[self.M - 2])
        cm[self.M - 1] *= pole / ((pole * pole) - 1)
        for k in range(self.M - 2, -1, -1):
            cm[k] = pole * (cm[k + 1] - cp[k])

        c = cm / b1

        c[np.where(np.abs(c) < eps)] = 0.0
        return c

    def filterPeriodic(self, s):
        self.M = len(s)
        b0 = self.value(0)
        pole = (-b0 + np.sqrt(2.0 * b0 - 1.0)) / (1.0 - b0)

        cp = np.zeros(self.M)
        cp[0] = s[0]
        for k in range(self.M):
            cp[0] += s[k] * (pole ** (self.M - k))
        cp[0] *= 1.0 / (1.0 - (pole**self.M))

        for k in range(1, self.M):
            cp[k] = s[k] + (pole * cp[k - 1])

        cm = np.zeros(self.M)
        cm[self.M - 1] = cp[self.M - 1]
        for k in range(self.M - 1):
            cm[self.M - 1] += cp[k] * (pole ** (k + 1))
        cm[self.M - 1] *= 1.0 / (1.0 - (pole**self.M))
        cm[self.M - 1] *= (1 - pole) ** 2

        for k in range(self.M - 2, -1, -1):
            cm[k] = (pole * cm[k + 1]) + (((1 - pole) ** 2) * cp[k])

        c = cm

        eps = 1e-8
        c[np.where(np.abs(c) < eps)] = 0.0
        return c

    def refinementMask(self):
        order = int(self.support)
        mask = np.zeros(order + 1)

        denominator = 2.0 ** (order - 1.0)
        mask[0] = 1.0 / denominator
        mask[1] = (2.0 * np.cos(self.alpha) + 1.0) / denominator
        mask[2] = mask[1]
        mask[3] = 1.0 / denominator

        return mask


class Keys(SplineGenerator):
    def __init__(self):
        SplineGenerator.__init__(self, False, 4.0)

    def value(self, x):
        val = 0.0
        if np.abs(x) >= 0 and np.abs(x) <= 1:
            val = (
                (3.0 / 2.0) * (np.abs(x) ** 3)
                - (5.0 / 2.0) * (np.abs(x) ** 2)
                + 1
            )
        elif np.abs(x) > 1 and np.abs(x) <= 2:
            val = (
                (-1.0 / 2.0) * (np.abs(x) ** 3)
                + (5.0 / 2.0) * (np.abs(x) ** 2)
                - 4.0 * np.abs(x)
                + 2.0
            )
        return val

    def firstDerivativeValue(self, x):
        val = 0.0
        if x >= 0 and x <= 1:
            val = x * (4.5 * x - 5.0)
        elif x >= -1 and x < 0:
            val = -x * (4.5 * x + 5.0)
        elif x > 1 and x <= 2:
            val = -1.5 * x * x + 5.0 * x - 4.0
        elif x >= -2 and x < -1:
            val = 1.5 * x * x + 5.0 * x + 4.0
        return val

    def secondDerivativeValue(self, x):
        val = 0.0
        if x >= 0 and x <= 1:
            val = 9.0 * x - 5.0
        elif x >= -1 and x < 0:
            val = -9.0 * x - 5.0
        elif x > 1 and x <= 2:
            val = -3.0 * x + 5.0
        elif x >= -2 and x < -1:
            val = 3.0 * x + 5.0
        return val

    def filterSymmetric(self, s):
        return s

    def filterPeriodic(self, s):
        return s


class H3(SplineGenerator):
    def __init__(self):
        SplineGenerator.__init__(self, True, 2.0)

    def value(self, x):
        return np.array([self.h31(x), self.h32(x)])

    def h31(self, x):
        val = 0.0
        if x >= 0 and x <= 1:
            val = (1.0 + (2.0 * x)) * (x - 1) * (x - 1)
        elif x < 0 and x >= -1:
            val = (1.0 - (2.0 * x)) * (x + 1) * (x + 1)
        return val

    def h32(self, x):
        val = 0.0
        if x >= 0 and x <= 1:
            val = x * (x - 1) * (x - 1)
        elif x < 0 and x >= -1:
            val = x * (x + 1) * (x + 1)
        return val

    def firstDerivativeValue(self, x):
        return np.array([self.h31prime(x), self.h32prime(x)])

    def h31prime(self, x):
        val = 0.0
        if x >= 0 and x <= 1:
            val = 6.0 * x * (x - 1.0)
        elif x < 0 and x >= -1:
            val = -6.0 * x * (x + 1.0)
        return val

    def h32prime(self, x):
        val = 0.0
        if x >= 0 and x <= 1:
            val = 3.0 * x * x - 4.0 * x + 1
        elif x < 0 and x >= -1:
            val = 3.0 * x * x + 4.0 * x + 1
        return val

    def secondDerivativeValue(self, x):
        raise RuntimeError("H3 isn't twice differentiable.")
        return

    def h31Autocorrelation(self, i, j, M):
        if self.support > M:
            raise ValueError(
                "Cannot compute h31Autocorrelation for M<" + str(self.support)
            )
            return

        val = 0.0
        if np.abs(i - j) == 1:
            val = 9.0 / ((M - 1.0) * 70.0)
        elif i == j:
            if (i == 0) or (i == M - 1):
                val = 13.0 / ((M - 1.0) * 35.0)
            else:
                val = 26.0 / ((M - 1.0) * 35.0)

        return val

    def h31PeriodicAutocorrelation(self, n, M):
        if self.support > M:
            raise ValueError(
                "Cannot compute h31PeriodicAutocorrelation for M<"
                + str(self.support)
            )
            return

        nmod = np.mod(n, M)
        val = 0.0
        if nmod == 0:
            val = 26.0 / (M * 35.0)
        elif (nmod == 1) or (nmod == M - 1):
            val = 9.0 / (M * 35.0) if M == 2 else 9.0 / (M * 70.0)

        return val

    def h32Autocorrelation(self, i, j, M):
        if self.support > M:
            raise ValueError(
                "Cannot compute h32Autocorrelation for M<" + str(self.support)
            )
            return

        val = 0.0
        if np.abs(i - j) == 1:
            val = -1.0 / ((M - 1.0) * 140.0)
        elif i == j:
            if (i == 0) or (i == M - 1):
                val = 1.0 / ((M - 1.0) * 105.0)
            else:
                val = 2.0 / ((M - 1.0) * 105.0)

        return val

    def h32PeriodicAutocorrelation(self, n, M):
        if self.support > M:
            raise ValueError(
                "Cannot compute h32PeriodicAutocorrelation for M<"
                + str(self.support)
            )
            return

        nmod = np.mod(n, M)
        val = 0.0
        if nmod == 0:
            val = 2.0 / (M * 105.0)
        elif (nmod == 1) or (nmod == M - 1):
            val = -1.0 / (M * 70.0) if M == 2 else -1.0 / (M * 140.0)

        return val

    def h3Crosscorrelation(self, i, j, M):
        if self.support > M:
            raise ValueError(
                "Cannot compute h3Crosscorrelation for M<" + str(self.support)
            )
            return

        val = 0.0
        if i - j == 1:
            val = 13.0 / ((M - 1.0) * 420.0)
        elif i - j == -1:
            val = -13.0 / ((M - 1.0) * 420.0)
        elif i == j:
            if i == 0:
                val = 11.0 / ((M - 1.0) * 210.0)
            elif i == M - 1:
                val = -11.0 / ((M - 1.0) * 210.0)

        return val

    def h3PeriodicCrosscorrelation(self, n, M):
        if self.support > M:
            raise ValueError(
                "Cannot compute h3PeriodicCrosscorrelation for M<"
                + str(self.support)
            )
            return

        nmod = np.mod(n, M)
        val = 0.0
        if nmod == 1:
            if M != 2:
                val = 13.0 / (M * 420.0)
        elif nmod == M - 1:
            val = -13.0 / (M * 420.0)

        return val


class HE3(SplineGenerator):
    def __init__(self, alpha):
        SplineGenerator.__init__(self, True, 2.0)
        self.alpha = alpha

    def value(self, x):
        return np.array([self.he31(x), self.he32(x)])

    def he31(self, x):
        val = 0.0
        val = self.g1(x) if x >= 0 else self.g1(-x)
        return val

    def g1(self, x):
        val = 0.0
        if x >= 0 and x <= 1:
            denom = (0.5 * self.alpha * np.cos(0.5 * self.alpha)) - np.sin(
                0.5 * self.alpha
            )
            num = (
                (
                    0.5
                    * (
                        (self.alpha * np.cos(0.5 * self.alpha))
                        - np.sin(0.5 * self.alpha)
                    )
                )
                - (0.5 * self.alpha * np.cos(0.5 * self.alpha) * x)
                - (0.5 * np.sin(0.5 * self.alpha - (self.alpha * x)))
            )
            val = num / denom
        return val

    def he32(self, x):
        val = 0.0
        val = self.g2(x) if x >= 0 else -1.0 * self.g2(-x)
        return val

    def g2(self, x):
        val = 0.0
        if x >= 0 and x <= 1:
            denom = (
                (
                    (0.5 * self.alpha * np.cos(0.5 * self.alpha))
                    - np.sin(0.5 * self.alpha)
                )
                * (4.0 * self.alpha)
                * np.sin(0.5 * self.alpha)
            )
            num = (
                -((self.alpha * np.cos(self.alpha)) - np.sin(self.alpha))
                - (
                    2.0
                    * self.alpha
                    * np.sin(0.5 * self.alpha)
                    * np.sin(0.5 * self.alpha)
                    * x
                )
                - (
                    2.0
                    * np.sin(0.5 * self.alpha)
                    * np.cos(self.alpha * (x - 0.5))
                )
                + (self.alpha * np.cos(self.alpha * (x - 1)))
            )
            val = num / denom
        return val

    def firstDerivativeValue(self, x):
        return np.array([self.he31prime(x), self.he32prime(x)])

    def he31prime(self, x):
        val = 0.0
        val = self.g1pPrime(x) if x >= 0 else -1.0 * self.g1prime(-x)
        return val

    def g1prime(self, x):
        val = 0.0
        if x >= 0 and x <= 1:
            denom = (0.5 * self.alpha * np.cos(0.5 * self.alpha)) - np.sin(
                0.5 * self.alpha
            )
            num = -(0.5 * self.alpha * np.cos(0.5 * self.alpha)) + (
                0.5 * self.alpha * np.cos(0.5 * self.alpha - (self.alpha * x))
            )
            val = num / denom
        return val

    def he32prime(self, x):
        val = 0.0
        val = self.g2prime(x) if x >= 0 else self.g2prime(-x)
        return val

    def g2prime(self, x):
        val = 0.0
        if x >= 0 and x <= 1:
            denom = (
                (
                    (0.5 * self.alpha * np.cos(0.5 * self.alpha))
                    - np.sin(0.5 * self.alpha)
                )
                * (4.0 * self.alpha)
                * np.sin(0.5 * self.alpha)
            )
            num = (
                -(
                    2.0
                    * self.alpha
                    * np.sin(0.5 * self.alpha)
                    * np.sin(0.5 * self.alpha)
                )
                + (
                    2.0
                    * self.alpha
                    * np.sin(0.5 * self.alpha)
                    * np.sin(self.alpha * (x - 0.5))
                )
                - (self.alpha * self.alpha * np.sin(self.alpha * (x - 1)))
            )
            val = num / denom
        return val

    def secondDerivativeValue(self, x):
        raise RuntimeError("HE3 isn't twice differentiable.")
        return


# Recursive function to compute the multinomial coefficient of (x0+x1+...+xm-1)^N
# This function finds every {k0,...,km-1} such that k0+...+km-1=N
# (cf multinomial theorem on Wikipedia for a detailed explanation)
def multinomial(
    maxValue,
    numberOfCoefficiens,
    kArray,
    iteration,
    dilationFactor,
    order,
    mask,
):
    if numberOfCoefficiens == 1:
        kArray[iteration] = maxValue

        denominator = 1.0
        degree = 0
        for k in range(dilationFactor):
            denominator *= math.factorial(kArray[k])
            degree += k * kArray[k]

        coef = math.factorial(order) / denominator
        mask[int(degree)] += coef / (dilationFactor ** (order - 1))

    else:
        for k in range(maxValue + 1):
            kArray[iteration] = k
            multinomial(
                maxValue - k,
                numberOfCoefficiens - 1,
                kArray,
                iteration + 1,
                dilationFactor,
                order,
                mask,
            )

    return
