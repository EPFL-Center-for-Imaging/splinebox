import collections
import copy
import math

import numba
import numpy as np
import scipy.integrate


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

    _wrong_dimension_msg = "It looks like coeffs is a 2D array with second dimension different than two. I don't know how to handle this yet."
    _wrong_array_size_msg = "It looks like coeffs is neither a 1 nor a 2D array. I don't know how to handle this yet."
    _no_coeffs_msg = "This model doesn't have any coefficients."
    _unimplemented_msg = "This function is not implemented."

    def __init__(self, M, basis_function, closed=False, coeffs=None):
        if basis_function.support <= M:
            self.M = M
        else:
            raise RuntimeError("M must be greater or equal than the spline generator support size.")

        self.basis_function = basis_function
        self.halfSupport = self.basis_function.support / 2
        # Number of additional knots used for padding the ends
        # of an open spline
        self.pad = math.ceil(self.halfSupport)
        self.closed = closed
        self.coeffs = coeffs

    @property
    def coeffs(self):
        return self._coeffs

    @coeffs.setter
    def coeffs(self, values):
        if values is not None:
            n = len(values)
            if self.closed and n != self.M:
                raise ValueError(
                    f"The number of coefficients must match the number of knots for a closed spline. You provided {n} coefficients for a spline with M={self.M} knots."
                )
            padded_M = self.M + 2 * self.pad
            if not self.closed and n != padded_M:
                raise ValueError(
                    f"Non-closed splines are padded at the ends with additional knots, i.e. the effective number of knots is M + 2 * ceil(support/2) of the basis function. You provided {n} coefficients for a spline with M={self.M} and a basis function with support={self.basis_function.support}, expected {padded_M}."
                )
        self._coeffs = values

    @property
    def basis_function(self):
        return self._basis_function

    @basis_function.setter
    def basis_function(self, value):
        if value.multigenerator:
            raise ValueError(
                "You are trying to construct a Hermite spline using the ordinary `Spline` class. Use the `HermiteSpline` class instead."
            )
        self._basis_function = value

    def copy(self):
        return copy.deepcopy(self)

    def draw(self, x, y):
        """
        Computes a whether a point is inside or outside a closed
        spline on a regular grid of points.

        I would ask the user to provide a grid of points directly instead
        of asking for the dimensions. Like that the user can choose how densly
        they want to sample the grid.
        """
        if self.coeffs is None:
            raise RuntimeError(self._no_coeffs_msg)

        if self.coeffs.ndim != 2 or self.coeffs.shape[1] != 2:
            raise RuntimeError("draw() can only be used with 2D curves")

        if not self.closed:
            raise RuntimeError("draw() can only be used with closed curves.")

        xx, yy = np.meshgrid(x, y)
        result = self.is_inside(xx.flatten(), yy.flatten())
        return result.reshape(xx.shape)

    def dtheta(self, t):
        r"""
        Helper function for calculating the winding number.

        `dtheta` is the derivative of the polar coordinate :math:`\theta(t)`

        .. math::
            \theta(t) = arctan \left( \frac{y(t)}{x(t)} \right)

        Differentiation yields:

        .. math::
            \frac{d \theta}{dt} = \frac{1}{r^2} \left( x\frac{dy}{dt} - y\frac{dx}{dt} \right) \text{, where } r^2 = x^2 + y^2
        """
        r = self.eval(t)
        dr = self.eval(t, derivative=1)

        r2 = np.linalg.norm(r, axis=1) ** 2
        if np.any(np.isnan(dr)) or np.isclose(r2, 0):
            # Happens the the spline is not differentiable in this location
            # or when the point is at the origin
            # The solution to return 0 is a bit of a hack but works in practice with
            # the numerical integration
            return np.squeeze(np.zeros(r.shape[0]))
        val = (1.0 / r2) * (r[:, 0] * dr[:, 1] - r[:, 1] * dr[:, 0])
        return np.squeeze(val)

    def is_inside(self, x, y):
        r"""
        Determines if a point with coordinates `x`, `y` is inside the spline.
        Only works for closed 2D curves.

        To determine whether a point is inside or outside the spline, the winding number
        is used:

        .. math::
            wind(\gamma, 0) = \frac{1}{2\pi} \oint_\gamma d\theta = \frac{1}{2\pi} \oint_\gamma \left( \frac{x}{r^2}dy - \frac{y}{r^2}dx \right).

        For a description of :math:`d\theta` check :meth:`splinebox.splines_curves.Spline.dtheta`.

        Parameters
        ----------
        x : numpy.ndarray or float
            x coordinate(s) of point(s)
        y : numpy.ndarray or float
            y coordinate(s) of point(s)

        Returns
        -------
        val : float
            1 if the point is inside, 0.5 if its on the curve and 0 if it is outside the curve.
        """
        if not self.closed:
            raise RuntimeError("isInside() can only be used with closed curves.")
        if self.coeffs.ndim != 2 or self.coeffs.shape[1] != 2:
            raise RuntimeError("isInside() can only be used with 2D curves.")

        if isinstance(x, float):
            x = np.array([x])
        if isinstance(y, float):
            y = np.array([y])
        if not np.allclose(x.shape, y.shape):
            raise ValueError("x and y need to have the same shape")

        results = np.zeros(x.shape)
        for coord, point in enumerate(np.stack([x, y], axis=-1)):
            spline_copy = self.copy()
            spline_copy.translate(-point)
            winding_number = scipy.integrate.quad(spline_copy.dtheta, 0, spline_copy.M)[0]
            for ref in np.array([[1, 1], [1, 2], [2, 2], [2, 1]]):
                if np.allclose(point, ref):
                    print(ref, winding_number)
            if np.abs(np.abs(winding_number) - 2 * np.pi) < 1e-6:
                results[coord] = 1
            elif np.abs(winding_number) > 1e-6:
                results[coord] = 0.5
            else:
                results[coord] = 0
        return np.squeeze(results)

    def getKnotsFromCoefs(self):
        if len(self.coeffs.shape) == 1:
            if self.closed:
                knots = np.zeros(self.M)
                for k in range(self.M):
                    knots[k] = self.eval(k)
            else:
                knots = np.zeros(self.M + 2 * self.pad)
                for kn, k in enumerate(range(-self.pad, self.M + self.pad)):
                    knots[kn] = self.eval(k)
        elif len(self.coeffs.shape) == 2 and (self.coeffs.shape[1] == 2):
            if self.closed:
                knots = np.zeros((self.M, 2))
                for k in range(self.M):
                    knots[k] = self.eval(k)
            else:
                knots = np.zeros((self.M + 2 * self.pad, 2))
                for kn, k in enumerate(range(-self.pad, self.M + self.pad)):
                    knots[kn] = self.eval(k)
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
                self.coeffs = self.basis_function.filter_periodic(knots)
            else:
                for _i in range(int(self.pad)):
                    # knots = np.append(knots, knots[-1-_i] + (knots[-1-_i]-knots[-(_i+1)*2]) )
                    knots = np.append(knots, knots[-1])
                for _i in range(int(self.pad)):
                    # knots = np.append(knots[0+_i] + (knots[0+_i]-knots[2*(_i+1)-1]), knots)
                    knots = np.append(knots[0], knots)
                self.coeffs = self.basis_function.filter_symmetric(knots)
        elif len(knots.shape) == 2:
            if knots.shape[1] == 2:
                if self.closed:
                    coeffsX = self.basis_function.filter_periodic(knots[:, 0])
                    coeffsY = self.basis_function.filter_periodic(knots[:, 1])
                else:
                    for _i in range(int(self.pad)):
                        knots = np.vstack((knots, knots[-1]))
                    for _i in range(int(self.pad)):
                        knots = np.vstack((knots[0], knots))
                    coeffsX = self.basis_function.filter_symmetric(knots[:, 0])
                    coeffsY = self.basis_function.filter_symmetric(knots[:, 1])
                self.coeffs = np.hstack(
                    (
                        np.array([coeffsX]).transpose(),
                        np.array([coeffsY]).transpose(),
                    )
                )
            else:
                raise RuntimeError(self._wrong_dimension_msg)
        else:
            raise RuntimeError(self._wrong_array_size_msg)

    def getCoefsFromDenseContour(self, contourPoints, arcLengthParameterization=False):
        """
        ???

        Fits the spline to match a contour.
        get is a bad name since nothing is returned.
        fit would be better.
        Presumably the this is different from getCoefsFromKnots
        because the spline does not have to go through the points.
        """
        N = len(contourPoints)

        phi = np.zeros((N, self.M)) if self.closed else np.zeros((N, self.M + 2 * self.pad))

        if len(contourPoints.shape) == 1:
            r = np.zeros(N)
        elif len(contourPoints.shape) == 2 and (contourPoints.shape[1] == 2):
            r = np.zeros((N, 2))

        if arcLengthParameterization:
            dist = [np.linalg.norm(contourPoints[i] - contourPoints[i - 1]) for i in range(1, len(contourPoints))]
            if self.closed:
                dist.append(np.linalg.norm(contourPoints[0] - contourPoints[-1]))
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
                t = i / (samplingRate + 1.0) if i / samplingRate < extraPoints else i / samplingRate

            if self.closed:
                for k in range(self.M):
                    tval = self.wrapIndex(t, k)
                    if tval > -self.halfSupport and tval < self.halfSupport:
                        basisFactor = self.basis_function.eval(tval)
                    else:
                        basisFactor = 0.0
                    phi[i, k] += basisFactor
            else:
                for k in range(self.M + 2 * self.pad):
                    tval = t - (k - self.halfSupport)
                    if tval > -self.halfSupport and tval < self.halfSupport:
                        basisFactor = self.basis_function.eval(tval)
                    else:
                        basisFactor = 0.0
                    phi[i, k] += basisFactor

        if len(contourPoints.shape) == 1:
            c = np.linalg.lstsq(phi, r, rcond=None)

            if self.closed:
                self.coeffs = np.zeros([self.M])
                for k in range(self.M):
                    self.coeffs[k] = c[0][k]
            else:
                self.coeffs = np.zeros([self.M + 2 * self.pad])
                for k in range(self.M + 2 * self.pad):
                    self.coeffs[k] = c[0][k]

        elif len(contourPoints.shape) == 2 and contourPoints.shape[1] == 2:
            cX = np.linalg.lstsq(phi, r[:, 0], rcond=None)
            cY = np.linalg.lstsq(phi, r[:, 1], rcond=None)

            if self.closed:
                self.coeffs = np.zeros([self.M, 2])
                for k in range(self.M):
                    self.coeffs[k] = np.array([cX[0][k], cY[0][k]])
            else:
                self.coeffs = np.zeros([self.M + 2 * self.pad, 2])
                for k in range(self.M + 2 * self.pad):
                    self.coeffs[k] = np.array([cX[0][k], cY[0][k]])

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
            lambda t: np.linalg.norm(self.eval(t, dt=True)),
            t0,
            tf,
            epsabs=1e-6,
            epsrel=1e-6,
            maxp1=50,
            limit=100,
        )

        return integral[0]

    def lengthToParameterRecursion(self, s, currentValue, lowerBound, upperBound, precisionDecimals=4):
        """
        Convert the given arc length s on the curve to a value in parameters space.
        This is done recursively, i.e. check if the point is before or after halfway
        and repeat.

        Some intelegent default can probably be set so the user only has to provide s.
        Or this function should be made private entirely,
        because there is :meth:`splinebox.spline_curves.Spline.lengthToParameter`

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

        if (np.round(currentValue, precisionDecimals) == np.round(midPointLength, precisionDecimals)) or (
            np.round(s, precisionDecimals) == np.round(midPointLength, precisionDecimals)
        ):
            return np.round(midPoint, precisionDecimals)

        elif np.round(s, precisionDecimals) < np.round(midPointLength, precisionDecimals):
            return self.lengthToParameterRecursion(s, currentValue, lowerBound, midPoint, precisionDecimals)
        else:
            return self.lengthToParameterRecursion(s, midPointLength, midPoint, upperBound, precisionDecimals)

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
        if self.coeffs is None:
            raise RuntimeError(self.noCoefsMessage)

        if len(self.coeffs.shape) == 1 or (len(self.coeffs.shape) == 2 and self.coeffs.shape[1] == 2):
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

            curve = np.array([self.eval(t, dt=dt) for t in ts])

            curve = curve[~np.all(curve == 0)] if len(self.coeffs.shape) == 1 else curve[~np.all(curve == 0, axis=1)]

        else:
            raise RuntimeError(self.wrongArraySizeMessage)

        return np.stack(curve)

    def eval(self, t, derivative=0):
        """
        Evalute the spline or one of its derivatives at
        parameter value(s) `t`.

        Parameters
        ----------
        t : numpy.array, float
            A 1D numpy array or a single float value.
        derivative : int
            Can be 0, 1, 2 for the spline, and its
            first and second derivative respectively.
        """
        if self.coeffs is None:
            raise RuntimeError(self._no_coeffs_msg)
        # Get values at which the basis functions have to be evaluated
        tval = self._get_tval(t)
        basis_function_values = self.basis_function.eval(tval, derivative=derivative)
        value = np.matmul(basis_function_values, self.coeffs)
        return value

    def _get_tval(self, t):
        """
        This is a helper method for `eval`. It is its own method
        to allow :class:`splinebox.spline_curves.HermiteSpline` to
        overwrite the `eval` method using `_get_tval`.
        """
        if not isinstance(t, collections.abc.Iterable):
            t = np.array([t])
        if self.closed:
            # all knot indices
            k = np.arange(self.M)
            # output array for the helper function _wrap_index
            tval = np.full((len(t), len(k)), np.nan)
            # compute the positions at which the basis functions have to be evaluated
            # and save them in tval
            self._wrap_index(t, k, self.halfSupport, self.M, tval)
        else:
            # take into account the padding with additional basis functions
            # for non-closed splines
            k = np.arange(self.M + 2 * self.pad)
            k = k[np.newaxis, :]
            t = t[:, np.newaxis]
            # positions at which the basis functions have to be evaluated
            tval = t - (k - self.halfSupport)
        return tval

    @staticmethod
    @numba.guvectorize(
        [(numba.float64[:], numba.float64[:], numba.float64, numba.int64, numba.float64[:, :])], "(n),(m),(),()->(n,m)"
    )
    def _wrap_index(ts, ks, halfSupport, M, wrapped_tval):
        """
        Fill the wrapped_tval array whenever a value t
        is affected by the basis function at knot k, taking
        into account that basis functions at the begging/end
        affect positions on the opposite end for closed splines.
        """
        outside_tvalue = halfSupport + 1
        for i, k in enumerate(ks):
            for j, t in enumerate(ts):
                if t >= M + k - halfSupport:
                    # t is close enough to the end to be affected by
                    # the k-th basis function from the beginning
                    wrapped_tval[j, i] = t - M - k
                elif t <= halfSupport - (M - k):
                    # t is close enough to the beginning to be affected
                    # by the k-th basis function, which is close to the end
                    wrapped_tval[j, i] = t + M - k
                elif t > k + halfSupport or t < k - halfSupport:
                    # t is outside the support of the k-th basis function
                    # this can be any value outside the support
                    wrapped_tval[j, i] = outside_tvalue
                else:
                    wrapped_tval[j, i] = t - k

    def centroid(self):
        """
        Does this correspond to the geometric centroid?
        Probably not, since the coefficient values are used.
        Why is centroid always 2D? Should there be a check for
        dimensionality here?
        """
        centroid = np.zeros(2)

        for k in range(self.M):
            centroid += self.coeffs[k]

        return centroid / self.M

    def translate(self, vector):
        """
        Translates the spline by a `vector`.

        Parameters
        ----------
        vector : numpy.ndarray
            Displacement vector added to the coefficients.
        """
        self.coeffs = self.coeffs + vector

    def scale(self, scalingFactor):
        """
        Enlarge or shrink the spline.
        This should probably use :meth:`splinebox.spline_curves.Spline.translate`
        `scalingFactor` can be renamed to `factor`.
        """
        centroid = self.centroid()

        for k in range(self.M):
            vectorToCentroid = self.coeffs[k] - centroid
            self.coeffs[k] = centroid + scalingFactor * vectorToCentroid

    def rotate(self, rotation_matrix):
        """
        Rotate the spline with the provided rotation matrix.

        Parameters
        ----------
        rotation_matrix : numpy.ndarray
            The rotation matrix applied to the spline.
        """
        if self.coeffs is None:
            raise RuntimeError(self._no_coeffs_msg)
        if self.coeffs.ndim == 1:
            raise RuntimeError("1D splines can not be rotated.")
        for k in range(len(self.coeffs)):
            self.coeffs[k] = np.matmul(rotation_matrix, self.coeffs[k])


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

    _coef_tangent_mismatch_msg = "It looks like coeffs and tangents have different shapes."
    _no_tangents_msg = "This spline doesn't have any tangents."

    def __init__(self, M, basis_function, closed=False, coeffs=None, tangents=None):
        super().__init__(M, basis_function, closed, coeffs=coeffs)
        self.tangents = tangents

    @property
    def tangents(self):
        return self._tangents

    @tangents.setter
    def tangents(self, values):
        if values is not None:
            n = len(values)
            if self.closed and n != self.M:
                raise ValueError(
                    f"The number of tangents must match the number of knots for a closed spline. You provided {n} tangents for a spline with M={self.M} knots."
                )
            support = self.basis_function.support
            padded_M = self.M + 2 * self.pad
            if not self.closed and n != padded_M:
                raise ValueError(
                    f"Non-closed splines are padded at the ends with additional knots, i.e. the effective number of knots is M + support of the basis function. You provided {n} tangents for a spline with M={self.M} and a basis function with support={support}, expected {padded_M}."
                )
        self._tangents = values

    @property
    def basis_function(self):
        return self._basis_function

    @basis_function.setter
    def basis_function(self, value):
        if not value.multigenerator:
            raise ValueError(
                "It looks like you are trying to use a single generator to build a multigenerator spline model."
            )
        self._basis_function = value

    def getCoefsFromKnots(self, knots, tangentAtKnots):
        knots = np.array(knots)
        tangentAtKnots = np.array(tangentAtKnots)

        if knots.shape != tangentAtKnots.shape:
            raise RuntimeError(self.coefTangentMismatchMessage)

        if len(knots.shape) == 1:
            self.coeffs = knots
            self.tangents = tangentAtKnots
        elif len(knots.shape) == 2:
            if knots.shape[1] == 2:
                self.coeffs = knots
                self.tangents = tangentAtKnots
            else:
                raise RuntimeError(self._wrong_dimension_msg)
        else:
            raise RuntimeError(self._wrong_array_size_msg)

    def getCoefsFromDenseContour(self, contourPoints, arcLengthParameterization=False):
        N = len(contourPoints)
        phi = np.zeros((N, 2 * self.M))

        if len(contourPoints.shape) == 1:
            r = np.zeros(N)
        elif len(contourPoints.shape) == 2 and (contourPoints.shape[1] == 2):
            r = np.zeros((N, 2))

        if arcLengthParameterization:
            dist = [np.linalg.norm(contourPoints[i] - contourPoints[i - 1]) for i in range(1, len(contourPoints))]
            if self.closed:
                dist.append(np.linalg.norm(contourPoints[0] - contourPoints[-1]))
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
                    basisFactor = self.basis_function.eval(tval)
                else:
                    basisFactor = [0.0, 0.0]

                phi[i, k] += basisFactor[0]
                phi[i, k + self.M] += basisFactor[1]

        if len(contourPoints.shape) == 1:
            c = np.linalg.lstsq(phi, r, rcond=None)

            self.coeffs = np.zeros([self.M])
            self.tangents = np.zeros([self.M])
            for k in range(self.M):
                self.coeffs[k] = c[0][k]
                self.tangents[k] = c[0][k + self.M]

        elif len(contourPoints.shape) == 2 and (contourPoints.shape[1] == 2):
            cX = np.linalg.lstsq(phi, r[:, 0], rcond=None)
            cY = np.linalg.lstsq(phi, r[:, 1], rcond=None)

            self.coeffs = np.zeros([self.M, 2])
            self.tangents = np.zeros([self.M, 2])
            for k in range(self.M):
                self.coeffs[k] = np.array([cX[0][k], cY[0][k]])
                self.tangents[k] = np.array([cX[0][k + self.M], cY[0][k + self.M]])

    def eval(self, t, derivative=0):
        """
        Evalute the spline or one of its derivatives at
        parameter value(s) `t`.

        Parameters
        ----------
        t : numpy.array, float
            A 1D numpy array or a single float value.
        derivative : int
            Can be 0, 1, 2 for the spline, and its
            first and second derivative respectively.
        """
        if self.coeffs is None:
            raise RuntimeError(self._no_coeffs_msg)
        if self.tangents is None:
            raise RuntimeError(self._no_tangents_msg)

        tval = self._get_tval(t)
        basis_function_values = self.basis_function.eval(tval, derivative=derivative)
        value = np.matmul(basis_function_values, self.coeffs) + np.matmul(basis_function_values, self.tangents)
        return value

    def scale(self, scalingFactor):
        Spline.scale(self, scalingFactor)

        for k in range(self.M):
            self.tangents[k] *= scalingFactor

    def rotate(self, rotation_matrix):
        Spline.rotate(self, rotation_matrix)

        if self.tangents is None:
            raise RuntimeError(self._no_tangents_msg)

        for k in range(len(self.tangents)):
            self.tangents[k] = np.matmul(rotation_matrix, self.tangents[k])
