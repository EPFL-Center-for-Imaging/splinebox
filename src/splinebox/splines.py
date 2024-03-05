import copy
import itertools
import multiprocessing
import warnings

import numpy as np
import scipy.integrate
from skimage import measure


class Spline:
    """
    Base class for the construction of a spline.

    Parameters
    ----------
    M : int
        Number of control points.
    basis_function : :class:`splinebox.basis_functions.BasisFunction`
        The basis function used to construct the spline.
    closed : boolean
        Whether or not the spline is closed, i.e. the two ends connect.
    """

    _wrong_dimension_msg = "It looks like coefs is a 2D array with second dimension different than two. I don't know how to handle this yet."
    _wrong_array_size_msg = "It looks like coefs is neither a 1 nor a 2D array. I don't know how to handle this yet."
    _no_coefs_msg = "This model doesn't have any coefficients."
    _unimplemented_msg = "This function is not implemented."

    def __init__(self, M, basis_function, closed):
        if basis_function.support <= M:
            self.M = M
        else:
            raise RuntimeError(
                "M must be greater or equal than the spline generator support size."
            )
            return

        self.basis_function = basis_function
        self.halfSupport = self.basis_function.support / 2.0
        self.closed = closed
        self.coefs = None

    def sample(self, samplingRate, cpuCount=1):
        if self.coefs is None:
            raise RuntimeError(self._no_coefs_msg)
            return

        if len(self.coefs.shape) == 1 or (
            len(self.coefs.shape) == 2 and self.coefs.shape[1] == 2
        ):
            if self.closed:
                N = samplingRate * self.M
            else:
                N = (samplingRate * (self.M - 1)) + 1

            if cpuCount == 1:
                curve = [
                    self.parameterToWorld(float(i) / float(samplingRate))
                    for i in range(N)
                ]
            else:
                cpuCount = np.min((cpuCount, multiprocessing.cpu_count()))
                with multiprocessing.Pool(cpuCount) as pool:
                    iterable = [
                        float(i) / float(samplingRate) for i in range(N)
                    ]
                    res = pool.map(self.parameterToWorld, iterable)

                curve = np.stack(res)
                if len(self.coefs.shape) == 1:
                    curve = curve[~np.all(curve == 0)]
                else:
                    curve = curve[~np.all(curve == 0, axis=1)]

        else:
            raise RuntimeError(self._wrong_array_size_msg)
            return

        return np.stack(curve)

    def draw(self, dimensions, cpuCount=1):
        if self.coefs is None:
            raise RuntimeError(self._no_coefs_msg)
            return

        if len(self.coefs.shape) == 2 and self.coefs.shape[1] == 2:
            if self.closed:
                warnings.warn(
                    "draw() will take ages, go get yourself a coffee.",
                    stacklevel=1,
                )

                if len(dimensions) != 2:
                    raise RuntimeError("dimensions must be a triplet.")
                    return

                xvals = range(dimensions[1])
                yvals = range(dimensions[0])
                pointsList = list(itertools.product(xvals, yvals))

                cpuCount = np.min((cpuCount, multiprocessing.cpu_count()))
                if cpuCount == 1:
                    vals = [self.isInside(p) for p in pointsList]
                    vals = np.asarray(vals)
                else:
                    pool = multiprocessing.Pool(cpuCount)
                    res = pool.map(self.isInside, pointsList)
                    vals = np.stack(res)

                area = np.zeros(dimensions, dtype=np.int8)
                for i in range(len(pointsList)):
                    p = pointsList[i]
                    area[p[1], p[0]] = int(255 * vals[i])

                return area

            else:
                raise RuntimeError(
                    "draw() can only be used with closed curves."
                )
                return
        else:
            raise RuntimeError("draw() can only be used with 2D curves.")
            return

    def windingNumber(self, t):
        r = self.parameterToWorld(t)
        dr = self.parameterToWorld(t, dt=True)

        r2 = np.linalg.norm(r) ** 2
        val = (1.0 / r2) * (r[0] * dr[1] - r[1] * dr[0])
        return val

    def isInside(self, point):
        if self.closed:
            originalCoefs = copy.deepcopy(self.coefs)
            self.coefs = originalCoefs - point

            res = scipy.integrate.quad(self.windingNumber, 0, self.M)

            self.coefs = originalCoefs

            val = res[0]
            if np.abs(val - 2.0 * np.pi) < 1e-6:
                return 1
            elif np.abs(val - np.pi) < 1e-6:
                return 0.5
            else:
                return 0
        else:
            raise RuntimeError(
                "isInside() can only be used with closed curves."
            )
            return

    def getCoefsFromKnots(self, knots):
        knots = np.array(knots)
        if len(knots.shape) == 1:
            if self.closed:
                self.coefs = self.basis_function.filterPeriodic(knots)
            else:
                self.coefs = self.basis_function.filterSymmetric(knots)
        elif len(knots.shape) == 2:
            if knots.shape[1] == 2:
                if self.closed:
                    coefsX = self.basis_function.filterPeriodic(knots[:, 0])
                    coefsY = self.basis_function.filterPeriodic(knots[:, 1])
                else:
                    coefsX = self.basis_function.filterSymmetric(knots[:, 0])
                    coefsY = self.basis_function.filterSymmetric(knots[:, 1])
                self.coefs = np.hstack(
                    (
                        np.array([coefsX]).transpose(),
                        np.array([coefsY]).transpose(),
                    )
                )
            else:
                raise RuntimeError(self._wrong_dimension_msg)
                return
        else:
            raise RuntimeError(self._wrong_array_size_msg)
            return

        return

    def getCoefsFromDenseContour(self, contourPoints):
        N = len(contourPoints)
        phi = np.zeros((N, self.M))
        if len(contourPoints.shape) == 1:
            r = np.zeros(N)
        elif len(contourPoints.shape) == 2 and (contourPoints.shape[1] == 2):
            r = np.zeros((N, 2))

        if self.closed:
            samplingRate = int(N / self.M)
            extraPoints = N % self.M
        else:
            samplingRate = int(N / (self.M - 1))
            extraPoints = N % (self.M - 1)

        for i in range(N):
            r[i] = contourPoints[i]

            if i == 0:
                t = 0
            elif t < extraPoints:
                t += 1.0 / (samplingRate + 1.0)
            else:
                t += 1.0 / samplingRate

            for k in range(self.M):
                tval = self.wrapIndex(t, k) if self.closed else t - k
                if tval > -self.halfSupport and tval < self.halfSupport:
                    basisFactor = self.basis_function.value(tval)
                else:
                    basisFactor = 0.0

                phi[i, k] += basisFactor

        if len(contourPoints.shape) == 1:
            c = np.linalg.lstsq(phi, r, rcond=None)

            self.coefs = np.zeros([self.M])
            for k in range(self.M):
                self.coefs[k] = c[0][k]
        elif len(contourPoints.shape) == 2 and (contourPoints.shape[1] == 2):
            cX = np.linalg.lstsq(phi, r[:, 0], rcond=None)
            cY = np.linalg.lstsq(phi, r[:, 1], rcond=None)

            self.coefs = np.zeros([self.M, 2])
            for k in range(self.M):
                self.coefs[k] = np.array([cX[0][k], cY[0][k]])

        return

    def getCoefsFromBinaryMask(self, binaryMask):
        contours = measure.find_contours(binaryMask, 0)

        if len(contours) > 1:
            raise RuntimeWarning(
                "Multiple objects were found on the binary mask. Only the first one will be processed."
            )

        self.getCoefsFromDenseContour(contours[0])
        return

    def arcLength(self, t0, tf=None):
        if t0 == tf:
            return 0.0

        if tf is None:
            tf = self.M if self.closed else self.M - 1

        if t0 > tf:
            temp = tf
            tf = t0
            t0 = temp

        integral = scipy.integrate.quad(
            lambda t: np.linalg.norm(self.parameterToWorld(t, dt=True)),
            t0,
            tf,
            epsabs=1e-6,
            epsrel=1e-6,
            maxp1=50,
            limit=100,
        )

        return integral[0]

    def lengthToParameterRecursion(
        self, s, currentValue, lowerBound, upperBound, precisionDecimals=4
    ):
        midPoint = lowerBound + (upperBound - lowerBound) / 2
        midPointLength = currentValue + self.arcLength(lowerBound, midPoint)

        if (
            np.round(currentValue, precisionDecimals)
            == np.round(midPointLength, precisionDecimals)
        ) or (
            np.round(s, precisionDecimals)
            == np.round(midPointLength, precisionDecimals)
        ):
            return np.round(midPoint, precisionDecimals)

        elif np.round(s, precisionDecimals) < np.round(
            midPointLength, precisionDecimals
        ):
            return self.lengthToParameterRecursion(
                s, currentValue, lowerBound, midPoint, precisionDecimals
            )
        else:
            return self.lengthToParameterRecursion(
                s, midPointLength, midPoint, upperBound, precisionDecimals
            )

    def lengthToParameter(self, s):
        if self.closed:
            return self.lengthToParameterRecursion(s, 0, 0, self.M)
        else:
            return self.lengthToParameterRecursion(s, 0, 0, self.M - 1)

    def sampleArcLength(self, numSamples, cpuCount=1):
        if self.coefs is None:
            raise RuntimeError(self._no_coefs_msg)
            return

        if len(self.coefs.shape) == 1 or (
            len(self.coefs.shape) == 2 and self.coefs.shape[1] == 2
        ):
            N = numSamples if self.closed else numSamples - 1
            L = self.arcLength(0)

            if cpuCount == 1:
                ts = [0]
                for n in range(1, N):
                    s = n * L / N
                    t = self.lengthToParameter(s)
                    ts.append(t)
                if self.closed:
                    ts.append(self.M)
                else:
                    ts.append(self.M - 1)

            else:
                cpuCount = np.min((cpuCount, multiprocessing.cpu_count()))
                with multiprocessing.Pool(cpuCount) as pool:
                    iterable = [n * L / N for n in range(1, N)]
                    res = pool.map(self.lengthToParameter, iterable)

                ts = np.stack(res)

            curve = np.array([self.parameterToWorld(t) for t in ts])
            if len(self.coefs.shape) == 1:
                curve = curve[~np.all(curve == 0)]
            else:
                curve = curve[~np.all(curve == 0, axis=1)]

        else:
            raise RuntimeError(self._wrong_array_size_msg)
            return

        return np.stack(curve)

    def parameterToWorld(self, t, dt=False):
        if self.coefs is None:
            raise RuntimeError(Spline._no_coefs_msg)
            return

        value = 0.0
        for k in range(self.M):
            tval = self.wrapIndex(t, k) if self.closed else t - k
            if tval > -self.halfSupport and tval < self.halfSupport:
                if dt:
                    splineValue = self.basis_function.firstDerivativeValue(
                        tval
                    )
                else:
                    splineValue = self.basis_function.value(tval)
                value += self.coefs[k] * splineValue
        return value

    def wrapIndex(self, t, k):
        wrappedT = t - k
        if k < t - self.halfSupport:
            if (
                k + self.M >= t - self.halfSupport
                and k + self.M <= t + self.halfSupport
            ):
                wrappedT = t - (k + self.M)
        elif k > t + self.halfSupport and (
            k - self.M >= t - self.halfSupport
            and k - self.M <= t + self.halfSupport
        ):
            wrappedT = t - (k - self.M)
        return wrappedT

    def centroid(self):
        centroid = np.zeros(2)

        for k in range(self.M):
            centroid += self.coefs[k]

        return centroid / self.M

    def translate(self, translationVector):
        for k in range(self.M):
            self.coefs[k] += translationVector

    def scale(self, scalingFactor):
        centroid = self.centroid()

        for k in range(self.M):
            vectorToCentroid = self.coefs[k] - centroid
            self.coefs[k] = centroid + scalingFactor * vectorToCentroid

    def rotate(self, rotationMatrix):
        for k in range(self.M):
            self.coefs[k] = np.matmul(rotationMatrix, self.coefs[k])


class HermiteSpline(Spline):
    """
    Class for the construction of a Hermite spline.

    Parameters
    ----------
    M : int
        Number of control points.
    basis_function : :class:`splinebox.basis_functions.BasisFunction`
        The basis function used to construct the spline.
    closed : boolean
        Whether or not the spline is closed, i.e. the two ends connect.
    """

    coefTangentMismatchMessage = (
        "It looks like coefs and tangents have different shapes."
    )

    def __init__(self, M, basis_function, closed):
        if not basis_function.multigenerator:
            raise RuntimeError(
                "It looks like you are trying to use a single generator to build a multigenerator spline model."
            )
            return

        Spline.__init__(self, M, basis_function, closed)
        self.tangents = None

    def getCoefsFromKnots(self, knots, tangentAtKnots):
        knots = np.array(knots)
        tangentAtKnots = np.array(tangentAtKnots)

        if knots.shape != tangentAtKnots.shape:
            raise RuntimeError(self.coefTangentMismatchMessage)
            return

        if len(knots.shape) == 1:
            self.coefs = knots
            self.tangents = tangentAtKnots
        elif len(knots.shape) == 2:
            if knots.shape[1] == 2:
                self.coefs = knots
                self.tangents = tangentAtKnots
            else:
                raise RuntimeError(Spline._wrong_dimension_msg)
                return
        else:
            raise RuntimeError(Spline._wrong_array_size_msg)
            return

        return

    def getCoefsFromDenseContour(self, contourPoints, tangentAtPoints):
        # TODO
        raise NotImplementedError(Spline._unimplemented_msg)
        return

    def getCoefsFromBinaryMask(self, binaryMask):
        # TODO
        raise NotImplementedError(Spline._unimplemented_msg)
        return

    def parameterToWorld(self, t, dt=False):
        if self.coefs is None:
            raise RuntimeError(self._no_coefs_msg)
            return

        value = 0.0
        for k in range(self.M):
            tval = self.wrapIndex(t, k) if self.closed else t - k
            if tval > -self.halfSupport and tval < self.halfSupport:
                if dt:
                    splineValue = self.basis_function.firstDerivativeValue(
                        tval
                    )
                else:
                    splineValue = self.basis_function.value(tval)
                value += (
                    self.coefs[k] * splineValue[0]
                    + self.tangents[k] * splineValue[1]
                )
        return value

    def scale(self, scalingFactor):
        Spline.scale(self, scalingFactor)

        for k in range(self.M):
            self.tangents[k] *= scalingFactor

    def rotate(self, rotationMatrix):
        Spline.rotate(self, rotationMatrix)

        for k in range(self.M):
            self.tangents[k] = np.matmul(rotationMatrix, self.tangents[k])
