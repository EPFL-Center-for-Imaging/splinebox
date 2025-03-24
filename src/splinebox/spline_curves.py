import collections
import copy
import json
import math
import warnings

import numba
import numpy as np
import scipy.integrate

import splinebox.basis_functions


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
    _no_control_points_msg = "This spline doesn't have any coefficients."
    _unimplemented_msg = "This function is not implemented."

    def __init__(self, M, basis_function, closed=False, control_points=None, padding_function=padding_function):
        if basis_function.support <= M:
            self.M = M
        else:
            raise RuntimeError("M must be greater or equal than the spline generator support size.")

        self.basis_function = basis_function
        self._half_support = self.basis_function.support / 2
        # Number of additional knots used for padding the ends
        # of an open spline
        self._pad = math.ceil(self._half_support) - 1
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

    def __str__(self):
        closed_str = "closed" if self.closed else "open"
        if self.control_points is None:
            return f"uninitialized {closed_str} {self.basis_function} spline with {self.M} knots"
        else:
            nD = 1 if self.control_points.ndim == 1 else self.control_points.shape[1]
            return f"{closed_str} {nD}D {self.basis_function} spline with {self.M} knots"

    def __repr__(self):
        return f"splinebox.spline_curves.Spline(M={repr(self.M)}, basis_function={repr(self.basis_function)}, closed={repr(self.closed)}, control_points=np.{repr(self.control_points)})"

    def __eq__(self, other):
        return (
            self.M == other.M
            and self.basis_function == other.basis_function
            and self.closed == other.closed
            and np.all(self.control_points == other.control_points)
        )

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
        return self(t)

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

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, M):
        if hasattr(self, "control_points") and self.control_points is not None and M != self.M:
            # The has attribute is necessary because M is assigned before control_points in the constructor
            raise RuntimeError(
                "M cannot be changed after the control points were set. Create a new spline or set the control_points to None first."
            )
        self._M = M

    @property
    def closed(self):
        return self._closed

    @closed.setter
    def closed(self, closed):
        if hasattr(self, "control_points") and self.control_points is not None and closed != self.closed:
            # The has attribute is necessary because closed is assigned before control_points in the constructor
            raise RuntimeError(
                "closed cannot be changed after the control points were set. Create a new spline or set the control_points to None first."
            )
        self._closed = closed

    @property
    def half_support(self):
        return self._half_support

    @half_support.setter
    def half_support(self, _):
        raise RuntimeError("The half support is determined by the basis function and cannot be set by the user.")

    @property
    def pad(self):
        return self._pad

    @pad.setter
    def pad(self, _):
        raise RuntimeError(
            "The amount of necessary padding is automatically calculated based on the support of the basis function and cannot be changed."
        )

    def copy(self):
        """
        Returns a deep copy of this spline.
        """
        return copy.deepcopy(self)

    def _to_dict(self, version):
        """
        Helper function that creates a dictionary
        representing the spline that can be saved as a json.
        This is implemented separately from :meth:`splinebox.spline_curves.Spline.to_json`
        to allow the :class:`splinebox.spline_curves.HermiteSpline` to inherit this
        conversion only adding the addiontion tangents.

        Paramters
        ---------
        version : int
            The version of the convertion for future compatibility.
        """
        dictionary_representation = {
            "version": version,
            "M": self.M,
            "basis_function": str(self.basis_function),
            "closed": self.closed,
            "control_points": self.control_points.tolist(),
        }
        return dictionary_representation

    def to_json(self, path, version=1):
        """
        Saves the spline as a json file.

        Parameters
        ----------
        path : str or pathlib.Path
            The path where the json file should be saved.
        version : int
            The version of the json file. Default is latest version.
        """
        with open(path, "w") as f:
            json.dump(self._to_dict(version), f, indent=2)

    @classmethod
    def from_json(cls, path):
        """
        Constructs a spline from a json file that was saved using
        :meth:`splinebox.spline_curves.Spline.to_json`.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the json file.
        """
        with open(path) as f:
            data = json.load(f)
        data = _prepared_dict_for_constructor(data)
        return cls(**data)

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
        r = self(t)
        dr = self(t, derivative=1)
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
        For details refer to :ref:`theory/data_approximation:Data approximation`.

        Parameters
        ----------
        points : numpy.ndarray
            The data points that should be fit.
        arc_length_parameterization : bool
            Whether or not to space the knots based on the distance
            between the provided points. This is usefull when the
            points are not equally spaced. Default is `False`.
        """
        if len(points) < self.M + 2 * self.pad:
            raise RuntimeError(
                f"You provided too few points. For the fit to have a unique solution you need to provide at least as many points as your spline has control points (including padding). This spline has {self.M}+{2 * self.pad}(padding) control points but you provided {len(points)} points. Consider providing more points or reducing the number of knots M. If the number of points is equal to M, consider using `spline.knots = points` instead of fitting."
            )

        if arc_length_parameterization:
            raise NotImplementedError
        else:
            t = np.linspace(0, self.M, len(points) + 1)[:-1] if self.closed else np.linspace(0, self.M - 1, len(points))
        tval = self._get_tval(t)
        basis_function_values = self.basis_function(tval, derivative=0)
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
            lambda t: np.linalg.norm(np.nan_to_num(self(t, derivative=1))),
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
            lambda t: (np.linalg.norm(np.nan_to_num(self(t, derivative=1))) ** 2 - c) ** 2,
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
        Compute the curvature of the spline at position(s) t.
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
        first_deriv = self(t, derivative=1)
        second_deriv = self(t, derivative=2)
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

    def normal(self, t, frame="bishop", initial_vector=None):
        """
        Returns the normal vector for 1D and 2D splines.
        The normal vector points to the right of the spline
        when facing in the direction of increasing t.

        Parameters
        ----------
        t : float or numpy array
            The parameter value(s) for which the normal
            vector(s) are computed.
        frame : "bishop" | "frenet"
            The type of frame used for 3D curves.
        initial_vector : numpy array
            Fixes the initial orientation of the normals at
            position t[0].

        Returns
        -------
        normals : numpy array
            The normal vectors.
        """
        self._check_control_points()
        if self.control_points.ndim != 2:
            raise NotImplementedError(
                "The normal vector is only implemented for curves in 2D and 3D. Your spline's codomain is 1 dimensional."
            )
        if self.control_points.shape[1] == 2:
            first_deriv = self(t, derivative=1)
            normals = (np.array([[0, -1], [1, 0]]) @ first_deriv.T).T
            normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
        elif self.control_points.shape[1] == 3:
            frame = self.moving_frame(t, method=frame, initial_vector=initial_vector)
            normals = frame[:, 1:]
        else:
            raise RuntimeError(
                f"The normal vector is only defined for curves in 2D and 3D. Your spline's codomain is {self.control_points.shape[1]} dimensional."
            )
        return normals

    def moving_frame(self, t, method="frenet", initial_vector=None):
        """
        Compute a moving frame (local orthonormal coordinate system) along the spline.

        This method computes either the Frenet-Serret frame or the Bishop frame [#bishop]_ for
        the spline. A moving frame [#movingframe]_ consists of three orthonormal basis vectors at
        each point on the curve. The Frenet-Serret frame is derived from the curve's
        derivatives but may twist around the curve. The Bishop frame eliminates
        this twist, providing a zero-torsion alternative.

        Parameters
        ----------
        t : np.array
            A 1D array of parameter values at which to evaluate the frame. These
            correspond to positions along the spline.
        method : str, optional
            The type of moving frame to compute. Options are:

            - "frenet": The classical Frenet-Serret frame, based on tangent, normal, and binormal vectors.
            - "bishop": A twist-free frame that requires an initial orientation.

            Default is "frenet".
        initial_vector : np.array or None, optional
            For the Bishop frame, an initial vector that is orthogonal to the tangent
            vector at `t[0]`. This vector determines the initial orientation of the
            basis, which is propagated along the curve without twisting. If None,
            the method computes a suitable initial vector automatically. This
            parameter is ignored when :code:`method="frenet"`.

        Returns
        -------
        frame : np.array
            A 3D numpy array with shape `(len(t), 3, 3)`. The dimensions are:

            - The first axis corresponds to the parameter values in `t`.
            - The second axis contains the three basis vectors at each `t`:
              [tangent, normal, binormal] for "frenet" or equivalent vectors for "bishop".
            - The third axis contains the components of each basis vector in 3D space.

        Raises
        ------
        RuntimeError
            If the spline is not defined in 3D or if the Frenet frame cannot be
            computed due to inflection points, straight segments, or undefined
            tangent/normal vectors.
        ValueError
            If the initial vector for the Bishop frame is not orthogonal to the
            tangent at `t[0]`, or if an invalid `method` is specified.

        Notes
        -----
        - The Frenet frame is not defined at points where the curve has zero curvature,
          such as straight segments or inflection points. In these cases, the Bishop
          frame is recommended.
        - For closed curves, check for discontinuities of the Bishop frame.

        References
        ----------
        .. [#movingframe] `Moving frame <https://en.wikipedia.org/wiki/Moving_frame>`_ on Wikipedia.
        .. [#bishop] Bishop, R. L. (1975). "There is More than One Way to Frame a Curve."
               American Mathematical Monthly, 82(3), 246-251.
        """
        self._check_control_points()
        if self.control_points.ndim != 2 or self.control_points.shape[1] != 3:
            raise RuntimeError("A frame can only be computed for splines in 3D.")
        first_derivative = self(t, derivative=1)

        frame = np.zeros((len(t), 3, 3))
        frame[:, 0] = first_derivative / np.linalg.norm(first_derivative, axis=-1)[:, np.newaxis]

        if method == "frenet":
            second_derivative = self(t, derivative=2)
            frame[:, 2] = np.cross(first_derivative, second_derivative)
            norm_binormal = np.linalg.norm(frame[:, 2], axis=-1)[:, np.newaxis]
            if np.any(np.isclose(norm_binormal, 0)):
                if np.isclose(norm_binormal[0], 0) or np.isclose(norm_binormal[-1], 0):
                    raise RuntimeError(
                        "The Frenet frame cannot be computed at one or both ends of the spline. This is often due to edge padding of the knots. Try to skip t=0 and t=M-1 or change the padding."
                    )
                raise RuntimeError(
                    "The Frenet frame is not defined for splines with inflection points or straight segments, try the Bishop frame instead."
                )
            frame[:, 2] /= norm_binormal
            frame[:, 1] = np.cross(frame[:, 2], frame[:, 0])
        elif method == "bishop":
            if initial_vector is None:
                tangent = frame[0, 0]
                # Try to do the same as for the Frenet frame
                initial_vector = np.cross(np.cross(tangent, self(t[0], derivative=2)), tangent)
                if np.isclose(np.linalg.norm(initial_vector), 0) or np.any(np.isnan(initial_vector)):
                    initial_vector = np.zeros(3)
                    max_axis = np.argmax(np.abs(tangent))
                    other_axis = (max_axis + 1) % 3
                    initial_vector[max_axis] = tangent[other_axis]
                    initial_vector[other_axis] = -tangent[max_axis]
            initial_vector /= np.linalg.norm(initial_vector)
            if not np.isclose(np.dot(frame[0, 0], initial_vector), 0):
                raise ValueError("The initial vector has to be orthogonal to the tangent at t[0].")
            frame[0, 1] = initial_vector
            frame[0, 2] = np.cross(frame[0, 0], initial_vector)
            for i in range(1, len(t)):
                n = np.cross(frame[i - 1, 0], frame[i, 0])
                norm_n = np.linalg.norm(n)
                if np.isclose(norm_n, 0):
                    frame[i, 1] = frame[i - 1, 1]
                    frame[i, 2] = frame[i - 1, 2]
                else:
                    n /= norm_n
                    phi = np.arccos(np.dot(frame[i - 1, 0], frame[i, 0]))
                    frame[i, 1] = (
                        frame[i - 1, 1] * np.cos(phi)
                        + np.cross(n, frame[i - 1, 1]) * np.sin(phi)
                        + n * np.dot(n, frame[i - 1, 1]) * (1 - np.cos(phi))
                    )
                    frame[i, 2] = (
                        frame[i - 1, 2] * np.cos(phi)
                        + np.cross(n, frame[i - 1, 2]) * np.sin(phi)
                        + n * np.dot(n, frame[i - 1, 2]) * (1 - np.cos(phi))
                    )
        else:
            raise ValueError(f"Unkown method '{method}' for moving frame.")
        return frame

    def __call__(self, t, derivative=0):
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
        basis_function_values = self.basis_function(tval, derivative=derivative)
        value = np.matmul(basis_function_values, self.control_points)
        return np.squeeze(value)

    def eval(self, t, derivative=0):
        """
        eval is deprecated use :meth:`splinebox.spline_curves.Spline.__call_` instead.
        """
        warnings.warn(
            "`spline.eval(t)` is deprecated and will be removed in v1 use `spline(t)` instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        return self(t, derivative=derivative)

    def _get_tval(self, t):
        """
        This is a helper method for `__call__`. It is its own method
        to allow :class:`splinebox.spline_curves. HermiteSpline` to
        overwrite the `__call__` method using `_get_tval`.
        It is also used in :meth:`splinebox.spline_curves.Spline.fit`
        """
        if not isinstance(t, collections.abc.Iterable):
            t = np.array([t])
        elif isinstance(t, np.ndarray) and t.shape == ():
            # Array with only one element, e.g. np.array(0.)
            t = np.array([t.item()])
        if self.closed:
            # all knot indices
            k = np.arange(self.M)
            # output array for the helper function _wrap_index
            tval = np.full((len(t), len(k)), np.nan)
            # compute the positions at which the basis functions have to be evaluated
            # and save them in tval
            self._wrap_index(t, k, self.half_support, self.M, tval)
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
    def _wrap_index(ts, ks, half_support, M, wrapped_tval):  # pragma: no cover
        """
        Fill the wrapped_tval array whenever a value t
        is affected by the basis function at knot k, taking
        into account that basis functions at the begging/end
        affect positions on the opposite end for closed splines.
        """
        outside_tvalue = half_support + 1
        for i, k in enumerate(ks):
            for j, t in enumerate(ts):
                if t >= M + k - half_support:
                    # t is close enough to the end to be affected by
                    # the k-th basis function from the beginning
                    wrapped_tval[j, i] = t - M - k
                elif t <= half_support - (M - k):
                    # t is close enough to the beginning to be affected
                    # by the k-th basis function, which is close to the end
                    wrapped_tval[j, i] = t + M - k
                elif t > k + half_support or t < k - half_support:
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

    def distance(self, point, return_t=False):
        """
        Computes the distance of point from the spline.

        Parameters
        ----------
        point : numpy.array
            Array with the coordinates of the point.
        return_t : bool
            Whether to return the paramter t of the spline.
            `spline(t)` gives the location on the spline
            closest to the point.

        Returns
        -------
        distance : float
            The distance between the point and the spline.
        t : float
            Only returned if `return_t=True`. This is the parameter
            corresponding to the location on the spline closest
            to the point.
        """
        self._check_control_points()
        if self.control_points.ndim == 1:
            raise RuntimeError("Cannot compute distance for 1D splines.")

        max_t = self.M if self.closed else self.M - 1
        t = np.linspace(0, max_t, self.M * 10)
        points_on_spline = self(t)
        distances = np.linalg.norm(points_on_spline - point[np.newaxis], axis=-1)
        t_initial = t[np.argmin(distances)]

        def _distance(t):
            return np.linalg.norm(self(t) - point)

        result = scipy.optimize.minimize(_distance, np.array((t_initial,)), bounds=((0, max_t),))

        min_distance = np.linalg.norm(self(result.x) - point)

        if return_t:
            return (min_distance, result.x)

        return min_distance

    def mesh(
        self,
        radius=None,
        step_t=0.1,
        step_angle=5,
        mesh_type="surface",
        cap_ends=False,
        frame="bishop",
        initial_vector=None,
    ):
        """
        Create a 3D mesh around the spline curve.

        This method generates a mesh surrounding the spline, with the distance from
        the spline controlled by the `radius` parameter. The mesh can be either a
        surface mesh with triangular cells or a volume mesh with tetrahedral cells.
        The orientation of the mesh is determined using either the Frenet-Serret or
        Bishop frame.

        Parameters
        ----------
        radius : float or callable, optional
            Defines the distance from the spline to the mesh surface. Can be:

            - A float for a constant radius.
            - A callable function that takes the spline parameter `t` and the polar
              angle in the normal plane as arguments and returns a float.

            If None, the mesh will follow the spline without an offset. Default is None.
        step_t : float, optional
            Step size for the spline parameter `t`, controlling the resolution along
            the curve. Smaller values increase mesh resolution. Default is 0.1.
        step_angle : float, optional
            Step size for the polar angle (in degrees), controlling the resolution
            around the spline. Smaller values increase mesh resolution. Default is 5.
        mesh_type : str, optional
            Specifies the type of mesh to create:

            - "surface": A surface mesh with triangular cells.
            - "volume": A volume mesh with tetrahedral cells.

            Default is "surface".
        cap_ends : bool, optional
            If True, the ends of an open surface mesh are capped with orthogonal
            planes. Ignored for closed splines or volume meshes. Default is False.
        frame : str, optional
            The frame to use for orientation of the mesh:
            - "frenet": Uses the Frenet-Serret frame.
            - "bishop": Uses the Bishop frame, requiring an `initial_vector`.
            See :meth:`splinebox.spline_curves.moving_frame`. Default is "bishop".
        initial_vector : numpy array or None, optional
            For the Bishop frame, an initial vector that defines the orientation of
            the frame at the start of the spline (`t[0]`). This vector must be
            orthogonal to the tangent at `t[0]`. Ignored for the Frenet frame. If
            None, a suitable initial vector is computed automatically. Default is None.

        Returns
        -------
        points : numpy array
            A 2D array of shape `(N, 3)`, where `N` is the number of mesh points.
            Each row represents the (x, y, z) coordinates of a point in the mesh.
        connectivity : numpy array
            A 2D array of shape `(M, K)`, where `M` is the number of elements in the
            mesh and `K` is the number of vertices per element (3 for surface meshes
            and 4 for volume meshes). Each row contains the indices of `points` that
            form an element.

        Raises
        ------
        NotImplementedError
            If the spline is not defined in 3D, as meshes are only supported for 3D splines.

        Notes
        -----
        - Surface meshes are useful for visualization, while volume meshes are
          typically used in simulations and finite element analysis.
        - For open splines, capping the ends (`cap_ends=True`) creates closed surfaces,
          which may be useful for some applications.
        - The Bishop frame is recommended for curves with inflection points or
          straight segments where the Frenet frame is undefined.
        - When radius is callable, the Bishop frame is recommend to avoid "drift" of the
          polar angle around the curve.

        Examples
        --------
        Create a surface mesh with constant radius:

        >>> points, connectivity = spline.mesh(radius=0.5, step_t=0.1, step_angle=10, mesh_type="surface")

        Create a volume mesh with variable radius:

        >>> def radius_function(t, angle):
        >>>     return 0.5 + 0.2 * np.sin(np.radians(angle))
        >>> points, connectivity = spline.mesh(radius=radius_function, mesh_type="volume")
        """
        if self.control_points.ndim != 2 or self.control_points.shape[1] != 3:
            raise NotImplementedError("Meshes are only implemented for splines in 3D.")
        t = np.arange(0, self.M if self.closed else self.M - 1 + step_t, step_t)
        if radius is None or radius == 0:
            points = self(t)
            connectivity = np.stack((np.arange(len(points)), np.arange(len(points)) + 1), axis=-1)
            if self.closed:
                # Connect end to the beginning
                connectivity[-1, -1] = 0
            else:
                connectivity = connectivity[:-1]
        else:
            _radius = (lambda t, phi: np.full(t.shape, radius)) if not callable(radius) else radius

            phi = np.arange(0, 360, step_angle)

            if mesh_type == "surface":
                phiphi, tt = np.meshgrid(phi, t)
                tt = tt.flatten()
                phiphi = phiphi.flatten()
                centers = self(tt.flatten())
                rr = _radius(tt, phiphi)
                normals = self.normal(t, frame=frame, initial_vector=initial_vector)

                n_angles = len(phi)
                n_t = len(t)
                normals = (
                    np.repeat(normals[:, 0], n_angles, axis=0) * np.sin(np.deg2rad(phiphi))[:, np.newaxis]
                    + np.repeat(normals[:, 1], n_angles, axis=0) * np.cos(np.deg2rad(phiphi))[:, np.newaxis]
                )

                points = centers + rr[:, np.newaxis] * normals
                n_points = len(points)
                connectivity = self._surface_mesh_connectivity(self.closed, n_angles, n_t, n_points)
                if cap_ends and not self.closed:
                    points = np.concatenate((centers[0].reshape((1, -1)), points, centers[-1].reshape((1, -1))), axis=0)
                    start_connectivity = np.zeros((n_angles, 3), dtype=int)
                    start_connectivity[:, 1] = np.arange(1, n_angles + 1)
                    start_connectivity[:, 2] = np.roll(start_connectivity[:, 1], -1)
                    end_connectivity = np.zeros((n_angles, 3), dtype=int)
                    end_connectivity[:, 0] = n_points + 1
                    end_connectivity[:, 1] = np.arange(n_points, n_points - n_angles, -1)
                    end_connectivity[:, 2] = np.roll(end_connectivity[:, 1], -1)
                    connectivity = np.concatenate((start_connectivity, connectivity + 1, end_connectivity))

            elif mesh_type == "volume":
                phiphi, tt = np.meshgrid(phi, t)
                rr = _radius(tt, phiphi)
                # Add columns for the center points
                rr = np.hstack((np.zeros((rr.shape[0], 1)), rr))
                tt = np.hstack((tt[:, 0][:, np.newaxis], tt))
                phiphi = np.hstack((phiphi[:, 0][:, np.newaxis], phiphi))

                tt = tt.flatten()
                phiphi = phiphi.flatten()
                rr = rr.flatten()

                centers = self(tt.flatten())
                normals = self.normal(t, frame=frame, initial_vector=initial_vector)

                n_angles = len(phi)
                n_t = len(t)
                normals = (
                    np.repeat(normals[:, 0], n_angles + 1, axis=0) * np.sin(np.deg2rad(phiphi))[:, np.newaxis]
                    + np.repeat(normals[:, 1], n_angles + 1, axis=0) * np.cos(np.deg2rad(phiphi))[:, np.newaxis]
                )
                points = centers + rr[:, np.newaxis] * normals
                n_points = len(points)

                connectivity = self._volume_mesh_connectivity(self.closed, n_angles, n_t, n_points)

        return points, connectivity

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def _surface_mesh_connectivity(closed, n_angles, n_t, n_points):  # pragma: no cover
        if closed:
            connectivity = np.zeros((2 * n_angles * n_t, 3), dtype=numba.int64)
        else:
            connectivity = np.zeros((2 * n_angles * (n_t - 1), 3), dtype=numba.int64)
        face = 0
        for i in range(n_t if closed else n_t - 1):
            for j in range(n_angles):
                connectivity[face] = [
                    i * n_angles + j,
                    ((i + 1) * n_angles + j) % n_points,
                    ((i + 1) * n_angles + (j + 1) % n_angles) % n_points,
                ]
                face += 1
                connectivity[face] = [
                    i * n_angles + j,
                    ((i + 1) * n_angles + (j + 1) % n_angles) % n_points,
                    (i * n_angles + (j + 1) % n_angles) % n_points,
                ]
                face += 1
        return connectivity

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def _volume_mesh_connectivity(closed, n_angles, n_t, n_points):  # pragma: no cover
        if closed:
            connectivity = np.zeros((3 * n_angles * n_t, 4), dtype=numba.int64)
        else:
            connectivity = np.zeros((3 * n_angles * (n_t - 1), 4), dtype=numba.int64)
        vol = 0
        for i in range(n_t if closed else n_t - 1):
            for j in range(1, n_angles + 1):
                connectivity[vol] = [
                    i * (n_angles + 1) + j,
                    ((i + 1) * (n_angles + 1) + j) % n_points,
                    ((i + 1) * (n_angles + 1) + 1 + j % n_angles) % n_points,
                    (i + 1) * (n_angles + 1) % n_points,
                ]
                vol += 1
                connectivity[vol] = [
                    i * (n_angles + 1) + j,
                    ((i + 1) * (n_angles + 1) + 1 + j % n_angles) % n_points,
                    (i * (n_angles + 1) + 1 + j % n_angles) % n_points,
                    i * (n_angles + 1),
                ]
                vol += 1
                connectivity[vol] = [
                    i * (n_angles + 1) + j,
                    ((i + 1) * (n_angles + 1) + 1 + j % n_angles) % n_points,
                    (i + 1) * (n_angles + 1) % n_points,
                    i * (n_angles + 1),
                ]
                vol += 1
        return connectivity


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
        attribute has to be true and the :func:`__call__ <splinebox.basis_functions.BasisFunction.__call__>` method has to return two values
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

    def __repr__(self):
        return f"splinebox.spline_curves.HermiteSpline(M={repr(self.M)}, basis_function={repr(self.basis_function)}, closed={repr(self.closed)}, control_points=np.{repr(self.control_points)}, tangents=np.{repr(self.tangents)})"

    def __eq__(self, other):
        return super().__eq__(other) and np.all(self.tangents == other.tangents)

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
        if len(points) < 2 * (self.M + 2 * self.pad):
            raise RuntimeError(
                f"You provided too few points. For a unique solution you need to provide at least 2*({self.M}+{2 * self.pad}) points to match the number of control points and tangents (including padding). You provided {len(points)} points. Consider providing more points or reducing the number of knots M."
            )
        if len(points) < self.M:
            raise RuntimeError(
                "You provided fewer data points than you spline has knots. For the fit to have a unique solution you need to provide at least as many data points as your spline has knots. Consider adding more data or reducing the number of knots M."
            )
        if arc_length_parameterization:
            raise NotImplementedError
        else:
            t = np.linspace(0, self.M, len(points) + 1)[:-1] if self.closed else np.linspace(0, self.M - 1, len(points))
        tval = self._get_tval(t)
        basis_function_values = self.basis_function(tval, derivative=0)
        basis_function_values = np.concatenate([basis_function_values[0], basis_function_values[1]], axis=1)
        solution = np.linalg.lstsq(basis_function_values, points, rcond=None)[0]
        half = self.M if self.closed else self.M + 2 * self.pad
        self.control_points = solution[:half]
        self.tangents = solution[half:]

    def __call__(self, t, derivative=0):
        self._check_control_points_and_tangents()

        tval = self._get_tval(t)
        basis_function_values = self.basis_function(tval, derivative=derivative)
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

    def _to_dict(self, version):
        dictionary_representation = super()._to_dict(version)
        dictionary_representation["tangents"] = self.tangents.tolist()
        return dictionary_representation


def _prepared_dict_for_constructor(data):
    """
    Helper function that processes the dictionaries loaded from
    json files. It ensure all of the values are valid and prepares
    a dictionary that can be passed to the constructor using `**`.

    Parameters
    ----------
    data : dict
        The dictionary from the json file.
    """
    if not isinstance(data["version"], int):
        raise ValueError("version has to be an integer.")

    if not isinstance(data["M"], int):
        raise ValueError("M has to be an integer.")

    data["basis_function"] = splinebox.basis_functions.basis_function_from_name(data["basis_function"], M=data["M"])

    data["closed"] = str(data["closed"]).lower()
    true_strings = ["true", "1", "t", "y", "yes"]
    false_strings = ["false", "0", "f", "n", "no"]
    if data["closed"] not in true_strings and data["closed"] not in false_strings:
        raise ValueError(f"closed should be a string that can be interpreted as a boolean not {data['closed']}.")
    data["closed"] = data["closed"] in true_strings

    if "control_points" in data:
        data["control_points"] = np.array(data["control_points"])
    if "tangents" in data:
        data["tangents"] = np.array(data["tangents"])

    # The version of the json file is only required for parsing
    del data["version"]

    return data


def splines_to_json(path, splines, version=1):
    """
    Saves multiple splines in a single json file.

    Parameters
    ----------
    path : str or pathlib.Path
        The path where the json should be saved.
    splines : iterable
        For instance a list of :class:`splinebox.spline_curves.Spline`
        and :class:`splinebox.spline_curves.HermiteSpline` objects.
    version : int
        The version used to produce the json file.
    """
    dicts = []

    for spline in splines:
        dicts.append(spline._to_dict(version))

    with open(path, "w") as f:
        json.dump(dicts, f, indent=2)


def splines_from_json(path):
    """
    Loades multiple splines from a json file generated using
    :func:`splinebox.spline_curves.splines_to_json`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the json file.

    Returns
    -------
    splines : list
        A list of :class:`splinebox.spline_curves.Spline` and
        :class:`splinebox.spline_curves.HermiteSpline` objects.
    """
    splines = []
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        # this is a json file of a single spline
        data = [data]

    for spline_data in data:
        spline_data = _prepared_dict_for_constructor(spline_data)

        if spline_data["basis_function"].multigenerator:
            # This is a basis function for a Hermite spline
            splines.append(HermiteSpline(**spline_data))
        else:
            splines.append(Spline(**spline_data))

    return splines
