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
        self.pad = math.ceil(self.halfSupport) - 1
        self.closed = closed
        self.coeffs = coeffs

    def _check_coeffs(self):
        """
        Most methods require coefficients to be set before they
        can be used. This helper function checks if the coefficients have been
        set.
        """
        if self.coeffs is None:
            raise RuntimeError(self._no_coeffs_msg)

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
                    f"Non-closed splines are padded at the ends with additional knots, i.e. the effective number of knots is M + 2 * (ceil(support/2) - 1) of the basis function. You provided {n} coefficients for a spline with M={self.M} and a basis function with support={self.basis_function.support}, expected {padded_M}."
                )
        self._coeffs = values

    @property
    def knots(self):
        t = np.arange(self.M) if self.closed else np.arange(-self.pad, self.M + self.pad)
        return self.eval(t)

    @knots.setter
    def knots(self, values):
        knots = np.array(values)
        if self.closed:
            self.coeffs = self.basis_function.filter_periodic(knots)
        else:
            # Add constant padding for the ends of the spline
            if knots.ndim == 1:
                knots = knots[:, np.newaxis]
            knots = np.pad(knots, ((self.pad, self.pad), (0, 0)), mode="edge")
            knots = np.squeeze(knots)

            self.coeffs = self.basis_function.filter_symmetric(knots)

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
        self._check_coeffs()

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
        self._check_coeffs()
        r = self.eval(t)
        dr = self.eval(t, derivative=1)
        if r.ndim == 1:
            r = r[np.newaxis, :]
            dr = dr[np.newaxis, :]
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
        self._check_coeffs()
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

    def fit(self, points, arc_length_parameterization=False):
        """
        Fit the provided points with the spline using
        least squares.

        Parameters
        ----------
        points : numpy.ndarray
            The data points that should be fit.
        arc_length_parameterization : bool
            Whether or not to space the knots based on the distance
            between the provided points. This is usefull when the
            points are not equally spaced. Default is `False`.
        """
        if len(points) < self.M:
            raise RuntimeError(
                "You provided fewer data points than you spline has knots. For the fit to have a unique solution you need to provide at least as many data points as your spline has knots. Consider adding more data or reducing the number of knots M."
            )
        if arc_length_parameterization:
            raise NotImplementedError
        else:
            t = np.linspace(0, self.M, len(points) + 1)[:-1] if self.closed else np.linspace(0, self.M - 1, len(points))
        tval = self._get_tval(t)
        basis_function_values = self.basis_function.eval(tval, derivative=0)
        self.coeffs = np.linalg.lstsq(basis_function_values, points, rcond=None)[0]

    def arc_length(self, stop=None, start=0):
        """
        Compute the arc length of the spline between
        the two parameter value specified. if

        Parameters
        ----------
        stop : float (optional)
            Stop point in parameter space.
        start : float (optional)
            Start point in parameter space.
        """
        self._check_coeffs()
        if stop is None:
            stop = self.M if self.closed else self.M - 1

        if start == stop:
            return 0

        if start > stop:
            start, stop = stop, start

        integral = scipy.integrate.quad(
            lambda t: np.linalg.norm(self.eval(t, derivative=1)),
            start,
            stop,
            epsabs=1e-6,
            epsrel=1e-6,
            maxp1=50,
            limit=100,
        )

        return integral[0]

    def _length_to_parameter_recursion(self, s, current_value, lower_bound, upper_bound, atol=1e-4):
        """
        Convert the given arc length s on the curve to a value in parameters space.
        This is done recursively, i.e. check if the point is before or after halfway
        and repeat. It uses binary search.

        Parameters
        ----------
        s : float
            Arc length on the spline.
        current_value : float
            The arc length to the lower bound.
        lower_bound : float
            Lower limit in parameter space.
        upper_bound : float
            Upper limit in parameters space.
        atol : float
            Absolute precision to which the length is matched.
        """
        self._check_coeffs()
        midpoint = lower_bound + (upper_bound - lower_bound) / 2
        midpoint_length = current_value + self.arc_length(lower_bound, midpoint)

        if np.isclose(s, midpoint_length, atol=atol, rtol=0):
            return midpoint
        elif s < midpoint_length:
            return self._length_to_parameter_recursion(s, current_value, lower_bound, midpoint, atol)
        else:
            return self._length_to_parameter_recursion(s, midpoint_length, midpoint, upper_bound, atol)

    def arc_length_to_parameter(self, s, atol=1e-4):
        """
        Convert the arc length `s` to the coresponding value in parameter space.

        Parameters
        ----------
        s : float or np.array
            Length on curve.
        atol : float
            The ablsolute error tolerance.
        """
        self._check_coeffs()
        if not isinstance(s, np.ndarray):
            s = np.array([s])
        sort_indices = np.argsort(s)
        results = np.zeros_like(s)

        current_value = 0
        lower_bound = 0
        upper_bound = self.M if self.closed else self.M - 1

        for i in sort_indices:
            results[i] = self._length_to_parameter_recursion(s[i], current_value, lower_bound, upper_bound, atol=atol)
            lower_bound = results[i]
            current_value = self.arc_length(0, lower_bound)

        return np.squeeze(results)

    def sampleArcLength(self, numSamples, dt=False):
        """
        Evaluate the spline equidistantly spaced along its trajectory.
        Perhaps it makes sense to ask the user to provide an array of distances
        instead of the numSamples.
        """
        self._check_coeffs()

        if len(self.coeffs.shape) == 1 or (len(self.coeffs.shape) == 2 and self.coeffs.shape[1] == 2):
            N = numSamples if self.closed else numSamples - 1
            L = self.arc_length()

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
        self._check_coeffs()
        # Get values at which the basis functions have to be evaluated
        tval = self._get_tval(t)
        basis_function_values = self.basis_function.eval(tval, derivative=derivative)
        value = np.matmul(basis_function_values, self.coeffs)
        return np.squeeze(value)

    def _get_tval(self, t):
        """
        This is a helper method for `eval`. It is its own method
        to allow :class:`splinebox.spline_curves.HermiteSpline` to
        overwrite the `eval` method using `_get_tval`.
        It is also used in :meth:`splinebox.spline_curves.Spline.fit`
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
            k = np.arange(self.M + 2 * self.pad) - self.pad
            k = k[np.newaxis, :]
            t = t[:, np.newaxis]
            # positions at which the basis functions have to be evaluated
            tval = t - k
        return tval

    @staticmethod
    @numba.guvectorize(
        [(numba.float64[:], numba.float64[:], numba.float64, numba.int64, numba.float64[:, :])], "(n),(m),(),()->(n,m)"
    )
    def _wrap_index(ts, ks, halfSupport, M, wrapped_tval):  # pragma: no cover
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

    def _coeffs_centroid(self):
        """
        Helper method for :meth:`splinebox.spline_curves.Spline.scale`
        and :meth:`splinebox.spline_curves.Spline.rotate`.
        Computes the centroid of the coefficients.
        """
        self._check_coeffs()
        return np.mean(self.coeffs, axis=0)

    def translate(self, vector):
        """
        Translates the spline by a `vector`.

        Parameters
        ----------
        vector : numpy.ndarray
            Displacement vector added to the coefficients.
        """
        self._check_coeffs()
        self.coeffs = self.coeffs + vector

    def scale(self, scaling_factor):
        """
        Enlarge or shrink the spline.
        This should probably use :meth:`splinebox.spline_curves.Spline.translate`
        `scalingFactor` can be renamed to `factor`.
        """
        self._check_coeffs()
        centroid = self._coeffs_centroid()
        self.translate(-centroid)
        self.coeffs *= scaling_factor
        self.translate(centroid)

    def rotate(self, rotation_matrix, centred=True):
        """
        Rotate the spline with the provided rotation matrix.

        Parameters
        ----------
        rotation_matrix : numpy.ndarray
            The rotation matrix applied to the spline.
        """
        self._check_coeffs()
        if self.coeffs.ndim == 1:
            raise RuntimeError("1D splines can not be rotated.")

        if centred:
            centroid = self._coeffs_centroid()
            self.translate(-centroid)

        for k in range(len(self.coeffs)):
            self.coeffs[k] = np.matmul(rotation_matrix, self.coeffs[k])

        if centred:
            self.translate(centroid)


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

    def _check_coeffs_and_tangents(self):
        """
        Most methods require coefficients to be set before they
        can be used. This helper function checks if the coefficients have been
        set.
        """
        self._check_coeffs()
        if self.tangents is None:
            raise RuntimeError(self._no_tangents_msg)

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
                    f"Non-closed splines are padded at the ends with additional knots, i.e. the effective number of knots is M + 2 * (ceil(support / 2) - 1) of the basis function. You provided {n} tangents for a spline with M={self.M} and a basis function with support={support}, expected {padded_M}."
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

    def fit(self, points, arc_length_parameterization=False):
        if len(points) < self.M:
            raise RuntimeError(
                "You provided fewer data points than you spline has knots. For the fit to have a unique solution you need to provide at least as many data points as your spline has knots. Consider adding more data or reducing the number of knots M."
            )
        if arc_length_parameterization:
            raise NotImplementedError
        else:
            t = np.linspace(0, self.M, len(points) + 1)[:-1] if self.closed else np.linspace(0, self.M - 1, len(points))
        tval = self._get_tval(t)
        basis_function_values = self.basis_function.eval(tval, derivative=0)
        basis_function_values = np.concatenate([basis_function_values[0], basis_function_values[1]], axis=1)
        half = self.M if self.closed else self.M + 2 * self.pad
        solution = np.linalg.lstsq(basis_function_values, points, rcond=None)[0]
        self.coeffs = solution[:half]
        self.tangents = solution[half:]

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
        self._check_coeffs_and_tangents()

        tval = self._get_tval(t)
        basis_function_values = self.basis_function.eval(tval, derivative=derivative)
        value = np.matmul(basis_function_values[0], self.coeffs) + np.matmul(basis_function_values[1], self.tangents)
        return value

    def scale(self, scaling_factor):
        self._check_coeffs_and_tangents()
        Spline.scale(self, scaling_factor)
        self.tangents *= scaling_factor

    def rotate(self, rotation_matrix, centred=True):
        self._check_coeffs_and_tangents()

        Spline.rotate(self, rotation_matrix, centred=centred)

        for k in range(len(self.tangents)):
            self.tangents[k] = np.matmul(rotation_matrix, self.tangents[k])
