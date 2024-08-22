import collections
import copy
import math

import numba
import numpy as np
import scipy.integrate


def padding_function(knots, pad_length):
    """
    This is the default padding function of splinebox.
    It applies constant padding to the ends of the knots.

    Parameters
    ----------
    knots : numpy array
        The knots to be padded.
    pad_length : int
        The amount of padding at each end.

    Returns
    -------
    padded_knots : numpy array
        Array of padded knots.
    """
    # Add constant padding to the ends
    if knots.ndim == 1:
        knots = knots[:, np.newaxis]
    return np.pad(knots, ((pad_length, pad_length), (0, 0)), mode="edge")


class Spline:
    """
    Base class for the construction of a spline.

    Parameters
    ----------
    M : int
        Number of knots.
    basis_function : :class:`splinebox.basis_functions.BasisFunction`
        The basis function used to construct the spline.
    closed : boolean
        Whether or not the spline is closed, i.e. the two ends are connected.
    control_points : np.array
        The control points of the spline. Optional, can be provided later.
    padding_function : callable
        A function that accepts an array of knots as the first argument and
        the padding size as the second argument. It should return a padded array.
        If `None`, a padded array has to be supplied when setting the `knots`.
        The default is constant padding with the edge values (see :func:`splinebox.spline_curves.padding_function`).
    """

    _wrong_dimension_msg = "It looks like control_points is a 2D array with second dimension different than two. I don't know how to handle this yet."
    _wrong_array_size_msg = (
        "It looks like control_points is neither a 1 nor a 2D array. I don't know how to handle this yet."
    )
    _no_control_points_msg = "This model doesn't have any coefficients."
    _unimplemented_msg = "This function is not implemented."

    def __init__(self, M, basis_function, closed=False, control_points=None, padding_function=padding_function):
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
        self.control_points = control_points
        self.padding_function = padding_function

    def _check_control_points(self):
        """
        Most methods require coefficients to be set before they
        can be used. This helper function checks if the coefficients have been
        set.
        """
        if self.control_points is None:
            raise RuntimeError(self._no_control_points_msg)

    @property
    def control_points(self):
        """
        The control points of this spline, i.e. the c[k]
        in equation :ref:`(1) <theory:eq:1>`.
        """
        return self._control_points

    @control_points.setter
    def control_points(self, values):
        if values is not None:
            n = len(values)
            if self.closed and n != self.M:
                raise ValueError(
                    f"The number of control points must match M for a closed spline. You provided {n} control points for a spline with M={self.M}."
                )
            padded_M = self.M + 2 * self.pad
            if not self.closed and n != padded_M:
                raise ValueError(
                    f"Non-closed splines are padded at the ends, i.e. the effective number of control points is M + 2 * (ceil(support/2) - 1). You provided {n} control points for a spline with M={self.M} and a basis function with support={self.basis_function.support}, expected {padded_M}."
                )
        self._control_points = values

    @property
    def knots(self):
        """
        The knots of this spline, i.e. the values of the spline
        at :math:`t=0,1,...,M`.
        """
        if self.padding_function is None and not self.closed:
            t = np.arange(-self.pad, self.M + self.pad)
        else:
            t = np.arange(self.M)
        return self.eval(t)

    @knots.setter
    def knots(self, values):
        knots = np.array(values)
        n = len(knots)
        if self.closed:
            if n != self.M:
                raise ValueError(
                    f"You provided {n} knots for a closed spline with M={self.M}. Expected {self.M} knots."
                )
            self.control_points = self.basis_function.filter_periodic(knots)
        else:
            padded_M = self.M + 2 * self.pad
            if self.padding_function is None:
                if n != padded_M:
                    raise ValueError(
                        f"If you do not provide a padding function, i.e. `padding_function=None`, you have to pad the knots at the end to have length M + 2 * (ceil(support/2) - 1) before setting them. You provided {n} knots for a spline with M={self.M} and a basis function with support={self.basis_function.support}, expected {padded_M}."
                    )
            else:
                if n != self.M:
                    raise ValueError(f"You provided {n} knots for a spline with M={self.M}. Expected {self.M} knots.")
                knots = self.padding_function(knots, self.pad)
                if len(knots) != padded_M:
                    raise ValueError(
                        f"Incorrect padding. Expected padded to have length {padded_M} instead of {len(knots)}."
                    )

            knots = np.squeeze(knots)

            self.control_points = self.basis_function.filter_symmetric(knots)

    @property
    def basis_function(self):
        """
        The basis function of the spline. Should be an object
        of a specific implementation of the abstract base class
        :class:`splinebox.basis_functions.BasisFunction`.
        """
        return self._basis_function

    @basis_function.setter
    def basis_function(self, value):
        if value.multigenerator:
            raise ValueError(
                "You are trying to construct a Hermite spline using the ordinary `Spline` class. Use the `HermiteSpline` class instead."
            )
        self._basis_function = value

    def copy(self):
        """
        Returns a deep copy of this spline.
        """
        return copy.deepcopy(self)

    def draw(self, x, y):
        """
        Computes whether a point is inside or outside a closed
        spline on a regular grid of points.

        Parameters
        ----------
        x : numpy array
            A 1D array containing the x values of the grid of points.
        y : numpy array
            A 1D array containing the y values of the grid of points.
        """
        self._check_control_points()

        if self.control_points.ndim != 2 or self.control_points.shape[1] != 2:
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
        self._check_control_points()
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

        For a description of :math:`d\theta` check :meth:`splinebox.spline_curves.Spline.dtheta`.

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
        self._check_control_points()
        if self.control_points.ndim != 2 or self.control_points.shape[1] != 2:
            raise RuntimeError("isInside() can only be used with 2D curves.")

        if isinstance(x, (float, int)):
            x = np.array([x])
        if isinstance(y, (float, int)):
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
        results = np.squeeze(results)
        if results.ndim == 0:
            return float(results)
        return results

    def fit(self, points, arc_length_parameterization=False):
        """
        Fit the provided points with the spline using
        least squares.
        For details refer to :ref:`Data approximation`.

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
        self.control_points = np.linalg.lstsq(basis_function_values, points, rcond=None)[0]

    def arc_length(self, stop=None, start=0, epsabs=1e-6, epsrel=1e-6):
        """
        Compute the arc length of the spline between
        the two parameter values specified. If no value for start is give,
        start from the begining of the spline.
        If no value for stop is give, go until the end of the spline.
        When arrays with multiple values are given for start and/or stop,
        an array with all of the arc lengths is returned.

        Parameters
        ----------
        stop : np.array / float (optional)
            Stop point(s) in parameter space.
        start : np.array / float (optional)
            Start point(s) in parameter space.
        epsabs : float (optional)
            Absolute error tolerance. Default is 1e-6.
        epsrel : float (optional)
            Relative error tolerance. Default is 1e-6.
        """
        self._check_control_points()

        if isinstance(stop, collections.abc.Iterable):
            results = np.zeros(len(stop))
            if isinstance(start, collections.abc.Iterable):
                if len(stop) != len(start):
                    raise ValueError(
                        "If you provide array like objects for start and stop, they need to have the same length."
                    )
                for i in range(len(stop)):
                    results[i] = self.arc_length(stop[i], start[i], epsabs=epsabs, epsrel=epsrel)
            else:
                sort_indices = np.argsort(stop)
                sorted_stop = stop[sort_indices]
                for i in range(len(stop)):
                    if i == 0:
                        results[i] = self.arc_length(sorted_stop[i], start, epsabs=epsabs, epsrel=epsrel)
                    else:
                        results[i] = results[i - 1] + self.arc_length(
                            sorted_stop[i], sorted_stop[i - 1], epsabs=epsabs, epsrel=epsrel
                        )
                # Undo the sorting
                results = results[np.argsort(sort_indices)]
            return results

        if stop is None:
            stop = self.M if self.closed else self.M - 1

        if start == stop:
            return 0

        if start > stop:
            start, stop = stop, start

        integral = scipy.integrate.quad(
            lambda t: np.linalg.norm(np.nan_to_num(self.eval(t, derivative=1))),
            start,
            stop,
            epsabs=epsabs,
            epsrel=epsrel,
            maxp1=50,
            limit=100,
        )

        return integral[0]

    def _length_to_parameter_recursion(
        self, s, current_value, lower_bound, upper_bound, intermediate_results=None, atol=1e-4
    ):
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
        intermediate_results : list
            A list where all computed length parameter pairs are stored.
            This can be used to initialize subsequent conversions more efficiently.
        atol : float
            Absolute precision to which the length is matched.
        """
        self._check_control_points()
        midpoint = lower_bound + (upper_bound - lower_bound) / 2
        midpoint_length = current_value + self.arc_length(lower_bound, midpoint, epsabs=atol)
        if intermediate_results is not None:
            intermediate_results.append((midpoint, midpoint_length))

        if np.isclose(s, midpoint_length, atol=atol, rtol=0):
            return midpoint
        elif s < midpoint_length:
            return self._length_to_parameter_recursion(
                s, current_value, lower_bound, midpoint, intermediate_results, atol
            )
        else:
            return self._length_to_parameter_recursion(
                s, midpoint_length, midpoint, upper_bound, intermediate_results, atol
            )

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
        self._check_control_points()
        if not isinstance(s, np.ndarray):
            s = np.array([s])
        sort_indices = np.argsort(s)
        results = np.zeros_like(s)

        current_value = 0
        lower_bound = 0
        upper_bound = self.M if self.closed else self.M - 1
        intermediate_results = []

        def upper_bound_key_func(x):
            diff = x[1] - s[i]
            if diff >= 0:
                return diff
            else:
                return np.inf

        for i in sort_indices:
            if len(intermediate_results) > 0:
                upper_bound, upper_bound_len = min(intermediate_results, key=upper_bound_key_func)
                if s[i] > upper_bound_len:
                    upper_bound = self.M if self.closed else self.M - 1

            results[i] = self._length_to_parameter_recursion(
                s[i], current_value, lower_bound, upper_bound, intermediate_results, atol=atol
            )

            lower_bound = results[i]
            _, current_value = min(intermediate_results, key=lambda x: abs(x[0] - lower_bound))

        return np.squeeze(results)

    def curvilinear_reparametrization_energy(self, epsabs=1e-6, epsrel=1e-6):
        """
        This energy can be used to enforce equal spacing of the knots.

        Implements equation 25 from [Jacob2004]_.
        In order to make the energy scale invariant,
        we added a factor of (arc length)^-4 to the integral.

        Parameters
        ----------
        epsabs : float
            The absolute accuracy for the integration.
            Default is 1e-6.
            For details see scipy.integrate.quad_.
        epsrel : float
            The relative accuracy for the integration.
            Default is 1e-6.
            For details see scipy.integrate.quad_.

        Returns
        -------
        energy : float
            The curvilinear reparametrization energy of the spline.

        .. _scipy.integrate.quad: https://docs.scipy.org/doc/scipy-1.14.0/reference/generated/scipy.integrate.quad.html
        """
        arc_length = self.arc_length()
        c = (arc_length / self.M) ** 2
        upper_limit = self.M if self.closed else self.M - 1
        integral = scipy.integrate.quad(
            lambda t: (np.linalg.norm(np.nan_to_num(self.eval(t, derivative=1))) ** 2 - c) ** 2,
            0,
            upper_limit,
            epsabs=epsabs,
            epsrel=epsrel,
            maxp1=50,
            limit=100,
        )
        return integral[0] / arc_length**4

    def curvature(self, t):
        """
        Compute the curcature of the spline at position(s) t.
        For splines in 1 and 2 dimensions, the signed curvature
        is returned. Otherwise the unsigned curvature is returned.

        Parameters
        ----------
        t : float or numpy array
            The paramter value(s) at which the curvature
            should be calculated.

        Returns
        -------
        k : float or numpy array
            The curvature value.
        """
        first_deriv = self.eval(t, derivative=1)
        second_deriv = self.eval(t, derivative=2)
        if first_deriv.ndim == 1:
            # This assumes a uniform spline, i.e. the derivative of t is constant.
            first_deriv = np.stack([first_deriv, np.ones(len(t))], axis=-1)
            second_deriv = np.stack([second_deriv, np.zeros(len(t))], axis=-1)
        norm_first_deriv = np.linalg.norm(first_deriv, axis=1)
        norm_second_deriv = np.linalg.norm(second_deriv, axis=1)
        if first_deriv.shape[1] == 2:
            # We use a different formular for the 2D case to get the signed curvature instead of the
            # the unsigned curvature. This is useful when plotting curvature combs.
            nominator = first_deriv[:, 1] * second_deriv[:, 0] - first_deriv[:, 0] * second_deriv[:, 1]
        else:
            nominator = np.sqrt(
                norm_first_deriv**2 * norm_second_deriv**2 - np.sum(first_deriv * second_deriv, axis=1) ** 2
            )
        k = nominator / norm_first_deriv**3
        return np.squeeze(k)

    def normal(self, t):
        """
        Returns the normal vector for 1D and 2D splines.
        The normal vector points to the right of the spline
        when facing in the direction of increasing t.

        Parameters
        ----------
        t : float or numpy array
            The parameter value(s) for which the normal
            vector(s) are computed.

        Returns
        -------
        normals : numpy array
            The normal vectors.
        """
        self._check_control_points()
        if self.control_points.ndim != 2:
            raise NotImplementedError(
                "The normal vector is only implemented for curves in 2D. Your spline's codomain is 1 dimensional."
            )
        if self.control_points.shape[1] != 2:
            raise RuntimeError(
                f"The normal vector is only defined for curves in 2D. Your spline's codomain is {self.control_points.shape[1]} dimensional."
            )
        first_deriv = self.eval(t, derivative=1)
        normals = (np.array([[0, -1], [1, 0]]) @ first_deriv.T).T
        normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
        return normals

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
        self._check_control_points()
        # Get values at which the basis functions have to be evaluated
        tval = self._get_tval(t)
        basis_function_values = self.basis_function.eval(tval, derivative=derivative)
        value = np.matmul(basis_function_values, self.control_points)
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

    def _control_points_centroid(self):
        """
        Helper method for :meth:`splinebox.spline_curves.Spline.scale`
        and :meth:`splinebox.spline_curves.Spline.rotate`.
        Computes the centroid of the coefficients.
        """
        self._check_control_points()
        return np.mean(self.control_points, axis=0)

    def translate(self, vector):
        """
        Translates the spline by a `vector`.

        Parameters
        ----------
        vector : numpy.ndarray
            Displacement vector added to the coefficients.
        """
        self._check_control_points()
        self.control_points = self.control_points + vector

    def scale(self, scaling_factor):
        """
        Enlarge or shrink the spline by the provided factor.
        """
        self._check_control_points()
        centroid = self._control_points_centroid()
        self.translate(-centroid)
        self.control_points *= scaling_factor
        self.translate(centroid)

    def rotate(self, rotation_matrix, centred=True):
        """
        Rotate the spline with the provided rotation matrix.

        Parameters
        ----------
        rotation_matrix : numpy.ndarray
            The rotation matrix applied to the spline.
        """
        self._check_control_points()
        if self.control_points.ndim == 1:
            raise RuntimeError("1D splines can not be rotated.")

        if centred:
            centroid = self._control_points_centroid()
            self.translate(-centroid)

        for k in range(len(self.control_points)):
            self.control_points[k] = np.matmul(rotation_matrix, self.control_points[k])

        if centred:
            self.translate(centroid)


class HermiteSpline(Spline):
    """
    Class for the construction of a Hermite spline.
    It inherits from :class:`splinebox.spline_curves.Spline`.
    Here, we only document the additional methods and attributes.
    For information on the inherited methods and attributes refere to the
    documentation of :class:`splinebox.spline_curves.Spline`.

    Parameters
    ----------
    M : int
        Number of control points.
    basis_function : :class:`splinebox.basis_functions.BasisFunction`
        The basis function used to construct the spline. The :class:`multigenerator <splinebox.basis_functions.BasisFunction>`
        attribute has to be true and the :func:`eval <splinebox.basis_functions.BasisFunction.eval>` method has to return two values
        instead of one.
    closed : boolean
        Whether or not the spline is closed, i.e. the two ends are connected.
    """

    _coef_tangent_mismatch_msg = "It looks like control_points and tangents have different shapes."
    _no_tangents_msg = "This spline doesn't have any tangents."

    def __init__(
        self, M, basis_function, closed=False, control_points=None, tangents=None, padding_function=padding_function
    ):
        super().__init__(M, basis_function, closed, control_points=control_points, padding_function=padding_function)
        self.tangents = tangents

    def _check_control_points_and_tangents(self):
        """
        Most methods require coefficients to be set before they
        can be used. This helper function checks if the coefficients have been
        set.
        """
        self._check_control_points()
        if self.tangents is None:
            raise RuntimeError(self._no_tangents_msg)

    @property
    def tangents(self):
        """
        The tangents of this spline.
        """
        return self._tangents

    @tangents.setter
    def tangents(self, values):
        if values is not None:
            n = len(values)
            if self.closed and n != self.M:
                raise ValueError(
                    f"The number of tangents must match M for a closed spline. You provided {n} tangents for a spline with M={self.M}."
                )
            padded_M = self.M + 2 * self.pad
            if not self.closed and n != padded_M:
                raise ValueError(
                    f"Non-closed splines are padded at the ends, i.e. the effective number of tangents is M + 2 * (ceil(support/2) - 1). You provided {n} tangents for a spline with M={self.M} and a basis function with support={self.basis_function.support}, expected {padded_M}."
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
        self.control_points = solution[:half]
        self.tangents = solution[half:]

    def eval(self, t, derivative=0):
        self._check_control_points_and_tangents()

        tval = self._get_tval(t)
        basis_function_values = self.basis_function.eval(tval, derivative=derivative)
        value = np.matmul(basis_function_values[0], self.control_points) + np.matmul(
            basis_function_values[1], self.tangents
        )
        return np.squeeze(value)

    def scale(self, scaling_factor):
        self._check_control_points_and_tangents()
        Spline.scale(self, scaling_factor)
        self.tangents *= scaling_factor

    def rotate(self, rotation_matrix, centred=True):
        self._check_control_points_and_tangents()

        Spline.rotate(self, rotation_matrix, centred=centred)

        for k in range(len(self.tangents)):
            self.tangents[k] = np.matmul(rotation_matrix, self.tangents[k])
