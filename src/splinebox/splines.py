import copy
import itertools
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

    def __init__(self, M, basis_function, closed, coefs=None):
        if basis_function.support <= M:
            self.M = M
        else:
            raise RuntimeError(
                "M must be greater or equal than the spline generator support size."
            )

        self.basis_function = basis_function
        self.halfSupport = self.basis_function.support / 2.0
        self.closed = closed
        self.coefs = coefs

    def copy(self):
        return copy.deepcopy(self)

    def sample(self, samplingRate):
        """
        Should be renamed to eval and take a vector of positions x
        as an argument instead of samplingRate.
        """
        if self.coefs is None:
            raise RuntimeError(self._no_coefs_msg)

        if len(self.coefs.shape) == 1 or (
            len(self.coefs.shape) == 2 and self.coefs.shape[1] == 2
        ):
            if self.closed:
                N = samplingRate * self.M
            else:
                N = (samplingRate * (self.M - 1)) + 1

            curve = [
                self.parameterToWorld(float(i) / float(samplingRate))
                for i in range(N)
            ]
        else:
            raise RuntimeError(self._wrong_array_size_msg)

        return np.stack(curve)

    def draw(self, dimensions):
        """
        Computes a whether a point is inside or outside a closed
        spline on a regular grid of points.

        I would ask the user to provide a grid of points directly instead
        of asking for the dimensions. Like that the user can choose how densly
        they want to sample the grid.
        """
        if self.coefs is None:
            raise RuntimeError(self._no_coefs_msg)

        if len(self.coefs.shape) == 2 and self.coefs.shape[1] == 2:
            if self.closed:
                warnings.warn(
                    "draw() will take ages, go get yourself a coffee.",
                    stacklevel=1,
                )

                if len(dimensions) != 2:
                    raise RuntimeError("dimensions must be a triplet.")

                xvals = range(dimensions[1])
                yvals = range(dimensions[0])
                pointsList = list(itertools.product(xvals, yvals))

                vals = [self.isInside(p) for p in pointsList]
                vals = np.asarray(vals)

                area = np.zeros(dimensions, dtype=np.int8)
                for i in range(len(pointsList)):
                    p = pointsList[i]
                    area[p[1], p[0]] = int(255 * vals[i])

                return area

            else:
                raise RuntimeError(
                    "draw() can only be used with closed curves."
                )
        else:
            raise RuntimeError("draw() can only be used with 2D curves.")

    def windingNumber(self, t):
        """
        ???

        Number that can be integrated along the entire spline
        to determine where it wraps around the origin or not.
        """
        r = self.parameterToWorld(t)
        dr = self.parameterToWorld(t, dt=True)

        r2 = np.linalg.norm(r) ** 2
        val = (1.0 / r2) * (r[0] * dr[1] - r[1] * dr[0])
        return val

    def isInside(self, point):
        """
        Determines if a point is inside the closed spline
        or not. Is it fair game to change the coeff of the
        object or should it be cloned first?
        The :meth:`splinebox.splines.Spline.translate` method should
        be used instead of subtracting the point.

        Parameters
        ----------
        point : numpy.array
        """
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

    def getKnotsFromCoefs(self):
        if len(self.coefs.shape) == 1:
            knots = np.zeros(self.M)
            if self.closed:
                for k in range(self.M):
                    knots[k] = self.parameterToWorld(k, dt=False)
            else:
                for k in range(-self.halfSupport, self.M + self.halfSupport):
                    knots[k] = self.parameterToWorld(k, dt=False)
        elif len(self.coefs.shape) == 2 and (self.coefs.shape[1] == 2):
            knots = np.zeros((self.M, 2))
            if self.closed:
                for k in range(self.M):
                    knots[k] = self.parameterToWorld(k, dt=False)
            else:
                for k in range(-self.halfSupport, self.M + self.halfSupport):
                    knots[k] = self.parameterToWorld(k, dt=False)
        return knots

    def getCoefsFromKnots(self, knots):
        """
        ???

        Fits the spline to go through the knots.
        get is a bad name since nothing is returned.
        fit would be better.
        """
        knots = np.array(knots)
        if len(knots.shape) == 1:
            if self.closed:
                self.coefs = self.basis_function.filterPeriodic(knots)
            else:
                for _i in range(int(self.halfSupport)):
                    knots = np.append(knots, knots[-1])
                for _i in range(int(self.halfSupport)):
                    knots = np.append(knots[0], knots)
                self.coefs = self.basis_function.filterSymmetric(knots)
        elif len(knots.shape) == 2:
            if knots.shape[1] == 2:
                if self.closed:
                    coefsX = self.basis_function.filterPeriodic(knots[:, 0])
                    coefsY = self.basis_function.filterPeriodic(knots[:, 1])
                else:
                    for _i in range(int(self.halfSupport)):
                        knots = np.vstack((knots, knots[-1]))
                    for _i in range(int(self.halfSupport)):
                        knots = np.vstack((knots[0], knots))
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
        else:
            raise RuntimeError(self._wrong_array_size_msg)

    def getCoefsFromDenseContour(
        self, contourPoints, arcLengthParameterization=False
    ):
        """
        ???

        Fits the spline to match a contour.
        get is a bad name since nothing is returned.
        fit would be better.
        Presumably the this is different from getCoefsFromKnots
        because the spline does not have to go through the points.
        """
        N = len(contourPoints)

        if self.closed:
            phi = np.zeros((N, self.M))
        else:
            phi = np.zeros((N, self.M + int(self.basis_function.support)))

        if len(contourPoints.shape) == 1:
            r = np.zeros(N)
        elif len(contourPoints.shape) == 2 and (contourPoints.shape[1] == 2):
            r = np.zeros((N, 2))

        if arcLengthParameterization:
            dist = [
                np.linalg.norm(contourPoints[i] - contourPoints[i - 1])
                for i in range(1, len(contourPoints))
            ]
            if self.closed:
                dist.append(
                    np.linalg.norm(contourPoints[0] - contourPoints[-1])
                )
            arclengths = np.hstack(([0], np.cumsum(dist, 0)))
        else:
            if self.closed:
                samplingRate = int(N / self.M)
                extraPoints = N % self.M
            else:
                samplingRate = int(N / (self.M - 1))
                extraPoints = N % (self.M - 1)

        for i in range(N):
            r[i] = contourPoints[i]

            if arcLengthParameterization:
                if self.closed:
                    t = arclengths[i] * self.M / arclengths[-1]
                else:
                    t = arclengths[i] * (self.M - 1) / arclengths[-1]
            else:
                if i / samplingRate < extraPoints:
                    t = i / (samplingRate + 1.0)
                else:
                    t = i / samplingRate

            if self.closed:
                for k in range(self.M):
                    tval = self.wrapIndex(t, k)
                    if tval > -self.halfSupport and tval < self.halfSupport:
                        basisFactor = self.basis_function.value(tval)
                    else:
                        basisFactor = 0.0
                    phi[i, k] += basisFactor
            else:
                for k in range(self.M + int(self.basis_function.support)):
                    tval = t - (k - self.halfSupport)
                    if tval > -self.halfSupport and tval < self.halfSupport:
                        basisFactor = self.basis_function.value(tval)
                    else:
                        basisFactor = 0.0
                    phi[i, k] += basisFactor

        if len(contourPoints.shape) == 1:
            c = np.linalg.lstsq(phi, r, rcond=None)

            if self.closed:
                self.coefs = np.zeros([self.M])
                for k in range(self.M):
                    self.coefs[k] = c[0][k]
            else:
                self.coefs = np.zeros(
                    [self.M + int(self.basis_function.support)]
                )
                for k in range(self.M + int(self.basis_function.support)):
                    self.coefs[k] = c[0][k]

        elif len(contourPoints.shape) == 2 and contourPoints.shape[1] == 2:
            cX = np.linalg.lstsq(phi, r[:, 0], rcond=None)
            cY = np.linalg.lstsq(phi, r[:, 1], rcond=None)

            if self.closed:
                self.coefs = np.zeros([self.M, 2])
                for k in range(self.M):
                    self.coefs[k] = np.array([cX[0][k], cY[0][k]])
            else:
                self.coefs = np.zeros(
                    [self.M + int(self.basis_function.support), 2]
                )
                for k in range(self.M + int(self.basis_function.support)):
                    self.coefs[k] = np.array([cX[0][k], cY[0][k]])

    def getCoefsFromBinaryMask(self, binaryMask):
        """
        Same as getCoefsFromDenseContour, except the input
        is a binary mask.
        """
        binaryMask_padded = np.zeros(
            (binaryMask.shape[0] + 2, binaryMask.shape[1] + 2)
        )
        binaryMask_padded[1:-1, 1:-1] = binaryMask
        contours = measure.find_contours(binaryMask_padded, 0)

        if len(contours) > 1:
            raise RuntimeWarning(
                "Multiple objects were found on the binary mask. Only the first one will be processed."
            )

        c = contours[0] - 1
        self.getCoefsFromDenseContour(c)

    def arcLength(self, t0, tf=None):
        """
        Integrate along the arc length.

        Can probably be made faster if the basis functions
        accept arrays directly.

        Parameters
        ----------
        t0 : float
            Start point in parameter space.
        tf : float (optional)
            End point in parameter space.
        """
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
        """
        Convert the given arc length s on the curve to a value in parameters space.
        This is done recursively, i.e. check if the point is before or after halfway
        and repeat.

        Some intelegent default can probably be set so the user only has to provide s.
        Or this function should be made private entirely,
        because there is :meth:`splinebox.splines.Spline.lengthToParameter`

        Parameters
        ----------
        s : float
            Arc length on the spline.
        currentValue : float
            The arc length to the lower bound.
        lowerBound : float
            Lower limit in parameter space.
        upperBound : float
            Upper limit in parameters space.
        precisionDecimals : int
            Precision to which the length is matched.
        """
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
        """
        Convert the arc length `s` to the coresponding value in parameter space.

        Parameters
        ----------
        s : float
            Length on curve.
        """
        if self.closed:
            return self.lengthToParameterRecursion(s, 0, 0, self.M)
        else:
            return self.lengthToParameterRecursion(s, 0, 0, self.M - 1)

    def sampleArcLength(self, numSamples, dt=False):
        """
        Evaluate the spline equidistantly spaced along its trajectory.
        Perhaps it makes sense to ask the user to provide an array of distances
        instead of the numSamples.
        """
        if self.coefs is None:
            raise RuntimeError(self.noCoefsMessage)

        if len(self.coefs.shape) == 1 or (
            len(self.coefs.shape) == 2 and self.coefs.shape[1] == 2
        ):
            N = numSamples if self.closed else numSamples - 1
            L = self.arcLength(0)

            ts = [0]
            for n in range(1, N):
                s = n * L / N
                t = self.lengthToParameter(s)
                ts.append(t)
            if self.closed:
                ts.append(self.M)
            else:
                ts.append(self.M - 1)

            curve = np.array([self.parameterToWorld(t, dt=dt) for t in ts])

            if len(self.coefs.shape) == 1:
                curve = curve[~np.all(curve == 0)]
            else:
                curve = curve[~np.all(curve == 0, axis=1)]

        else:
            raise RuntimeError(self.wrongArraySizeMessage)

        return np.stack(curve)

    def parameterToWorld(self, t, dt=False):
        """
        Calculate the spline value at parameter t.
        Maybe there should be two functions for the
        value and the derivative instead of a dt parameter.

        Rename to `eval`?
        """
        if self.coefs is None:
            raise RuntimeError(self._no_coefs_msg)

        value = 0.0
        if self.closed:
            for k in range(self.M):
                tval = self.wrapIndex(t, k)
                if tval > -self.halfSupport and tval < self.halfSupport:
                    if dt:
                        splineValue = self.basis_function.firstDerivativeValue(
                            tval
                        )
                    else:
                        splineValue = self.basis_function.value(tval)
                    value += self.coefs[k] * splineValue
        else:
            for k in range(self.M + int(self.basis_function.support)):
                tval = t - (k - self.halfSupport)
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
        """
        ???

        Should this be private?
        """
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
        """
        Does this correspond to the geometric centroid?
        Probably not, since the coefficient values are used.
        Why is centroid always 2D? Should there be a check for
        dimensionality here?
        """
        centroid = np.zeros(2)

        for k in range(self.M):
            centroid += self.coefs[k]

        return centroid / self.M

    def translate(self, translationVector):
        """
        Translates the spline by a vector.
        Is vector the right name here or can it also be a scalar?
        """
        for k in range(self.M):
            self.coefs[k] += translationVector

    def scale(self, scalingFactor):
        """
        Enlarge or shrink the spline.
        This should probably use :meth:`splinebox.splines.Spline.translate`
        `scalingFactor` can be renamed to `factor`.
        """
        centroid = self.centroid()

        for k in range(self.M):
            vectorToCentroid = self.coefs[k] - centroid
            self.coefs[k] = centroid + scalingFactor * vectorToCentroid

    def rotate(self, rotationMatrix):
        """
        Rotate the spline.
        Should the dimensionality be checked here?
        """
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

    _coef_tangent_mismatch_msg = (
        "It looks like coefs and tangents have different shapes."
    )

    def __init__(self, M, basis_function, closed, coefs=None, tangents=None):
        if not basis_function.multigenerator:
            raise RuntimeError(
                "It looks like you are trying to use a single generator to build a multigenerator spline model."
            )

        super().__init__(M, basis_function, closed, coefs=coefs)
        self.tangents = tangents

    def copy(self):
        return copy.deepcopy(self)

    def getCoefsFromKnots(self, knots, tangentAtKnots):
        knots = np.array(knots)
        tangentAtKnots = np.array(tangentAtKnots)

        if knots.shape != tangentAtKnots.shape:
            raise RuntimeError(self.coefTangentMismatchMessage)

        if len(knots.shape) == 1:
            self.coefs = knots
            self.tangents = tangentAtKnots
        elif len(knots.shape) == 2:
            if knots.shape[1] == 2:
                self.coefs = knots
                self.tangents = tangentAtKnots
            else:
                raise RuntimeError(self._wrong_dimension_msg)
        else:
            raise RuntimeError(self._wrong_array_size_msg)

    def getCoefsFromDenseContour(
        self, contourPoints, arcLengthParameterization=False
    ):
        N = len(contourPoints)
        phi = np.zeros((N, 2 * self.M))

        if len(contourPoints.shape) == 1:
            r = np.zeros(N)
        elif len(contourPoints.shape) == 2 and (contourPoints.shape[1] == 2):
            r = np.zeros((N, 2))

        if arcLengthParameterization:
            dist = [
                np.linalg.norm(contourPoints[i] - contourPoints[i - 1])
                for i in range(1, len(contourPoints))
            ]
            if self.closed:
                dist.append(
                    np.linalg.norm(contourPoints[0] - contourPoints[-1])
                )
            arclengths = np.hstack(([0], np.cumsum(dist, 0)))
        else:
            if self.closed:
                samplingRate = int(N / self.M)
                extraPoints = N % self.M
            else:
                samplingRate = int(N / (self.M - 1))
                extraPoints = N % (self.M - 1)

        for i in range(N):
            r[i] = contourPoints[i]

            if arcLengthParameterization:
                if self.closed:
                    t = arclengths[i] * self.M / arclengths[-1]
                else:
                    t = arclengths[i] * (self.M - 1) / arclengths[-1]
            else:
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
                    basisFactor = [0.0, 0.0]

                phi[i, k] += basisFactor[0]
                phi[i, k + self.M] += basisFactor[1]

        if len(contourPoints.shape) == 1:
            c = np.linalg.lstsq(phi, r, rcond=None)

            self.coefs = np.zeros([self.M])
            self.tangents = np.zeros([self.M])
            for k in range(self.M):
                self.coefs[k] = c[0][k]
                self.tangents[k] = c[0][k + self.M]

        elif len(contourPoints.shape) == 2 and (contourPoints.shape[1] == 2):
            cX = np.linalg.lstsq(phi, r[:, 0], rcond=None)
            cY = np.linalg.lstsq(phi, r[:, 1], rcond=None)

            self.coefs = np.zeros([self.M, 2])
            self.tangents = np.zeros([self.M, 2])
            for k in range(self.M):
                self.coefs[k] = np.array([cX[0][k], cY[0][k]])
                self.tangents[k] = np.array(
                    [cX[0][k + self.M], cY[0][k + self.M]]
                )

    def getCoefsFromBinaryMask(self, binaryMask):
        # TODO: This was deleted by Virginie in her cleaned up version
        raise NotImplementedError(Spline._unimplemented_msg)

    def parameterToWorld(self, t, dt=False):
        if self.coefs is None:
            raise RuntimeError(self._no_coefs_msg)

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
