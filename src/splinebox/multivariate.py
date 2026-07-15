"""
Multivariate splines.

.. warning::
    The multivariate spline API is preliminary and may change in future releases.
"""

import collections.abc
import math

import numba
import numpy as np
import scipy

import splinebox.spline_curves

EINSUM_INDICES = "abcdefghijklmnopqrstuvwxyz"


def tensor_product(arrays):
    """
    Compute the tensor product of a list of arrays.

    Parameters
    ----------
    arrays : list of numpy arrays
        The arrays to combine. Each array must be one or two dimensional.
        One-dimensional arrays are treated as column vectors; two-dimensional
        arrays are used as-is, allowing each factor to have an arbitrary
        codomain dimension.

    Returns
    -------
    numpy array
        The tensor product of the input arrays.

    Examples
    --------
    >>> import numpy as np
    >>> import splinebox
    >>> a = np.array([1.0, 2.0])
    >>> b = np.array([3.0, 4.0])
    >>> splinebox.multivariate.tensor_product([a, b])
    array([[[3.],
            [4.]],
    <BLANKLINE>
           [[6.],
            [8.]]])
    """
    for i, array in enumerate(arrays):
        if array.ndim > 2:
            raise ValueError(f"Each array must be one or two dimensional. Array {i} has shape {array.shape}.")
        if array.ndim == 1:
            arrays[i] = array[:, np.newaxis]
    n = len(arrays)
    einsum_str = ""
    for index in EINSUM_INDICES[:n]:
        einsum_str += index + EINSUM_INDICES[n] + ","
    einsum_str = einsum_str[:-1]
    einsum_str += "->" + EINSUM_INDICES[: n + 1]
    return np.einsum(einsum_str, *arrays)


class MultivariateSpline:
    """
    Spline in multiple independent variables.

    A multivariate spline is defined as a tensor product of univariate splines,
    one for each variable. It can be evaluated, fitted to data, and used to
    generate meshes for bivariate splines.

    Parameters
    ----------
    M : iterable of int
        Number of knots for each variable.
    basis_functions : :class:`splinebox.basis_functions.BasisFunction` or iterable
        The basis function(s) used to construct the spline. A single basis
        function is applied to all variables; otherwise an iterable with one
        basis function per variable must be provided.
    closed : bool or iterable of bool
        Whether each variable is closed. A single boolean is broadcast to all
        variables.
    control_points : numpy array, optional
        The control points of the spline. The first ``nvariate`` axes correspond
        to the control point grid, and the last axis is the codomain dimension.
        If ``None``, the spline must be initialized later via ``knots`` or
        ``fit``.
    padding_functions : callable or iterable of callables
        Function(s) used to pad knots for open splines. The default is
        :func:`splinebox.spline_curves.padding_function`.

    Raises
    ------
    ValueError
        If ``M`` is not an iterable of integers, if ``basis_functions`` or
        ``closed`` do not have the expected length, or if ``padding_functions``
        does not match the number of variables.
    RuntimeError
        If any entry of ``M`` is smaller than the support of the corresponding
        basis function.

    Examples
    --------
    Create a bivariate B-spline:

    >>> import numpy as np
    >>> import splinebox
    >>> spline = splinebox.multivariate.MultivariateSpline(
    ...     M=(4, 5),
    ...     basis_functions=splinebox.B3(),
    ...     closed=(False, False),
    ... )

    Set the control points. For open splines the control points have to be
    padded along each open dimension (one extra point on each side for B3):

    >>> spline.control_points = np.random.rand(6, 7, 3)

    Evaluate the spline on a grid of parameter values:

    >>> t0 = np.linspace(0, 4, 5)
    >>> t1 = np.linspace(0, 5, 6)
    >>> t = np.stack(np.meshgrid(t0, t1, indexing="ij"), axis=-1)
    >>> values = spline(t)
    """

    def __init__(
        self,
        M,
        basis_functions,
        closed=False,
        control_points=None,
        padding_functions=splinebox.spline_curves.padding_function,
    ):
        if not isinstance(M, collections.abc.Iterable) or any(not isinstance(m, (int, np.integer)) for m in M):
            raise ValueError("M should be an iterable of integers.")
        self.M = M

        self.nvariate = len(M)

        if not isinstance(basis_functions, collections.abc.Iterable):
            self.basis_functions = [basis_functions] * self.nvariate
        elif len(basis_functions) == self.nvariate:
            self.basis_functions = basis_functions
        else:
            raise ValueError(
                "basis_functions should be a single basis function or an iterable of basis_functions of the same length as M."
            )

        if any(basis_function.support > m for basis_function, m in zip(self.basis_functions, self.M)):
            raise RuntimeError("M must be greater than or equal to the basis function support size.")

        self._half_support = tuple(basis_function.support / 2 for basis_function in self.basis_functions)

        # Number of additional knots used for padding the ends
        # of an open spline
        self._pad = tuple(math.ceil(half_support) - 1 for half_support in self._half_support)

        if not isinstance(closed, collections.abc.Iterable):
            self.closed = [closed] * self.nvariate
        elif len(closed) == self.nvariate:
            self.closed = closed
        else:
            raise ValueError("closed should be a single boolean or an iterable of booleans of the same length as M.")

        self.control_points = control_points

        if not isinstance(padding_functions, collections.abc.Iterable):
            self.padding_functions = [padding_functions] * self.nvariate
        elif len(padding_functions) == self.nvariate:
            self.padding_functions = padding_functions
        else:
            raise ValueError(
                "padding_functions should be a single function or an iterable of functions of the same length as M."
            )

    @property
    def half_support(self):
        """
        Half the support of each basis function.

        Raises
        ------
        RuntimeError
            If the user tries to set this property directly.
        """
        return self._half_support

    @half_support.setter
    def half_support(self, _):
        raise RuntimeError("The half support is determined by the basis function and cannot be set by the user.")

    @property
    def pad(self):
        """
        Number of additional control points used for padding each open end.

        Raises
        ------
        RuntimeError
            If the user tries to set this property directly.
        """
        return self._pad

    @pad.setter
    def pad(self, _):
        raise RuntimeError(
            "The amount of necessary padding is automatically calculated based on the support of the basis function and cannot be changed."
        )

    @property
    def ndim(self):
        """
        Dimensionality of the codomain, i.e. the number of values produced per
        spline evaluation.

        Returns
        -------
        int
            The size of the last axis of :attr:`control_points`.
        """
        return self.control_points.shape[-1]

    @property
    def control_points(self):
        """
        The control points :math:`c[k]` as defined
        in equation :ref:`(1) <theory:eq:1>`.

        Raises
        ------
        ValueError
            If the number of control points doesn't match :attr:`~splinebox.spline_curves.Spline.M` + 2 * :attr:`~splinebox.spline_curves.Spline.pad`.
        """
        return self._control_points

    @control_points.setter
    def control_points(self, values):
        if values is not None:
            if values.ndim != self.nvariate and values.ndim != self.nvariate + 1:
                raise ValueError
            for variate in range(self.nvariate):
                n = values.shape[variate]
                if self.closed[variate] and n != self.M[variate]:
                    raise ValueError(
                        f"The number of control points must match M for a closed spline. You provided {n} control points for a spline with M={self.M[variate]}."
                    )
                padded_M = self.M[variate] + 2 * self.pad[variate]
                if not self.closed[variate] and n != padded_M:
                    raise ValueError(
                        f"Non-closed splines are padded at the ends, i.e. the effective number of control points is M + 2 * (ceil(support/2) - 1). You provided {n} control points for a spline with M={self.M[variate]} and a basis function with support={self.basis_functions[variate].support}, expected {padded_M}."
                    )
        self._control_points = values

    @property
    def knots(self):
        t = []
        for variate in range(self.nvariate):
            if self.padding_functions[variate] is None and not self.closed[variate]:
                t.append(np.arange(-self.pad[variate], self.M[variate] + self.pad[variate]))
            else:
                t.append(np.arange(self.M[variate]))
        t = np.stack(np.meshgrid(*t, indexing="ij"), axis=-1)
        return self(t)

    @knots.setter
    def knots(self, values):
        knots = np.array(values)
        self.fit(knots)

    def _get_tval(self, t, variate):
        """
        This is a helper method for `__call__`. It is its own method
        to allow :class:`splinebox.spline_curves. HermiteSpline` to
        overwrite the `__call__` method using `_get_tval`.
        It is also used in :meth:`splinebox.spline_curves.Spline.fit`
        """
        t = t[..., variate]
        if self.closed[variate]:
            # all knot indices
            k = np.arange(self.M[variate])
            # output array for the helper function _wrap_index
            flat_t = t.flatten()
            # flat_t = np.einsum("ii->i", t[:, :-1])
            tval = np.full((len(flat_t), len(k)), np.nan)
            # compute the positions at which the basis functions have to be evaluated
            # and save them in tval
            self._wrap_index(flat_t, k, self.half_support[variate], self.M[variate], tval)
            tval = tval.reshape(*t.shape, len(k))
        else:
            # take into account the padding with additional basis functions
            # for non-closed splines
            k = np.arange(self.M[variate] + 2 * self.pad[variate]) - self.pad[variate]
            k = k[np.newaxis, :]
            t = t[..., np.newaxis]
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

    def __call__(self, t, derivatives=0):
        """
        Evaluate the multivariate spline at the parameter values ``t``.

        Parameters
        ----------
        t : numpy array
            Array of shape ``(..., nvariate)`` containing one parameter value
            per variable.
        derivatives : int or iterable of int
            Derivative order for each variable. A single integer is broadcast to
            all variables.

        Returns
        -------
        numpy array
            Spline values of shape ``(..., ndim)``.

        Raises
        ------
        ValueError
            If ``t.shape[-1]`` does not match ``nvariate`` or if the length of
            ``derivatives`` does not match ``nvariate``.

        Examples
        --------
        >>> import numpy as np
        >>> import splinebox
        >>> spline = splinebox.multivariate.MultivariateSpline(
        ...     M=(4, 4),
        ...     basis_functions=splinebox.B3(),
        ...     closed=(True, True),
        ... )
        >>> spline.control_points = np.random.rand(4, 4, 2)
        >>> t0 = np.linspace(0, 4, 5)
        >>> t1 = np.linspace(0, 4, 5)
        >>> t = np.stack(np.meshgrid(t0, t1, indexing="ij"), axis=-1)
        >>> values = spline(t)
        """
        if not isinstance(derivatives, collections.abc.Iterable):
            derivatives = [derivatives] * self.nvariate
        elif len(derivatives) != self.nvariate:
            raise ValueError("The length of derivatives must match the number of variables of the spline.")

        if t.shape[-1] != self.nvariate:
            raise ValueError("You have to specify one t per basis_function")

        basis_function_values = []
        for variate in range(self.nvariate):
            t_vals = self._get_tval(t, variate)
            basis_function_values.append(self.basis_functions[variate](t_vals, derivative=derivatives[variate]))

        control_point_indices = EINSUM_INDICES[: self.nvariate + 1]
        t_indices = EINSUM_INDICES[self.nvariate + 1 : self.nvariate + t.shape[-1] + 1]
        einsum_str = control_point_indices
        for variate in range(self.nvariate):
            einsum_str += "," + t_indices + control_point_indices[variate]
        einsum_str += "->" + t_indices + control_point_indices[-1]
        return np.einsum(einsum_str, self.control_points, *basis_function_values)

    def mesh(self, step_t=0.1):
        """
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
        """
        if self.nvariate != 2 or self.ndim not in (1, 3):
            raise NotImplementedError

        t = []
        for variate in range(self.nvariate):
            stop = self.M[variate] if self.closed[variate] else self.M[variate] - 1 + 0.9 * step_t
            t.append(np.arange(0, stop, step_t))

        t = np.stack(np.meshgrid(t[0], t[1], indexing="ij"), axis=-1)
        points = self(t)

        if self.ndim == 1:
            points = np.concatenate([points, t], axis=-1)
        points = points.reshape(-1, 3)

        xx, yy = np.meshgrid(np.arange(t.shape[0]), np.arange(t.shape[1]), indexing="ij")
        coords = np.stack([xx, yy], axis=0)
        step_down = np.zeros_like(coords)
        step_down[0] = 1
        step_right = np.zeros_like(coords)
        step_right[1] = 1

        connectivity = []
        if self.closed[0] and self.closed[1]:
            slce = (slice(None), slice(None), slice(None))
        elif self.closed[0]:
            slce = (slice(None), slice(None), slice(0, -1))
        elif self.closed[1]:
            slce = (slice(None), slice(0, -1), slice(None))
        else:
            slce = (slice(None), slice(0, -1), slice(0, -1))
        connectivity.append(
            np.stack(
                [
                    coords[slce],
                    coords[slce] + step_down[slce],
                    coords[slce] + step_down[slce] + step_right[slce],
                ],
                axis=1,
            )
        )
        connectivity.append(
            np.stack(
                [
                    coords[slce],
                    coords[slce] + step_down[slce] + step_right[slce],
                    coords[slce] + step_right[slce],
                ],
                axis=1,
            )
        )
        connectivity = np.stack(connectivity, axis=-1)
        connectivity = connectivity.reshape(2, 3, -1)
        connectivity[0] = connectivity[0] % t.shape[0]
        connectivity[1] = connectivity[1] % t.shape[1]
        connectivity = np.ravel_multi_index(connectivity, t.shape[:-1])
        connectivity = connectivity.T

        return points, connectivity

    def fit(self, points, t=None):
        """
        Fit the multivariate spline to a set of points by least squares.

        Parameters
        ----------
        points : numpy array
            Data to fit. If ``t`` is ``None`` and ``points`` has
            ``nvariate`` dimensions, it is interpreted as a scalar field and the
            codomain dimension is 1. Otherwise ``points`` must have
            ``nvariate + 1`` dimensions, where the last axis contains the
            codomain values.
        t : numpy array, optional
            Parameter values corresponding to ``points`` with shape
            ``points.shape[:nvariate] + (nvariate,)``. If ``None``, parameter
            values are generated automatically from the shape of ``points``.

        Examples
        --------
        >>> import numpy as np
        >>> import splinebox
        >>> spline = splinebox.multivariate.MultivariateSpline(
        ...     M=(4, 4),
        ...     basis_functions=splinebox.B3(),
        ...     closed=(False, False),
        ... )
        >>> points = np.random.rand(10, 10, 2)
        >>> spline.fit(points)
        """
        control_points_shape = np.zeros(self.nvariate, dtype=int)
        for variate in range(self.nvariate):
            if self.closed[variate]:
                control_points_shape[variate] = self.M[variate]
            else:
                control_points_shape[variate] = self.M[variate] + 2 * self.pad[variate]

        if t is None and points.ndim == self.nvariate:
            points = points[..., np.newaxis]
        elif t is None:
            t = []
            for variate in range(self.nvariate):
                t.append(
                    np.linspace(0, self.M[variate], points.shape[variate] + 1)[:-1]
                    if self.closed[variate]
                    else np.linspace(0, self.M[variate] - 1, points.shape[variate])
                )
            t = np.stack(np.meshgrid(*t, indexing="ij"), axis=-1)
        basis_function_values = []
        for variate in range(self.nvariate):
            tval = self._get_tval(t, variate)
            basis_function_values.append(self.basis_functions[variate](tval, derivative=0))

        einsum_str = ""
        points_indices = EINSUM_INDICES[: points.ndim - 1]
        control_point_indices = EINSUM_INDICES[points.ndim - 1 : points.ndim - 1 + self.nvariate]
        for variate in range(self.nvariate):
            einsum_str += points_indices + control_point_indices[variate] + ","
        # Remove trailing comma
        einsum_str = einsum_str[:-1]
        einsum_str += "->" + points_indices + control_point_indices
        basis_function_values = np.einsum(einsum_str, *basis_function_values)

        # Run least squares
        basis_function_values = basis_function_values.reshape(-1, math.prod(control_points_shape))
        basis_function_values = scipy.sparse.csr_array(basis_function_values)
        points = points.reshape(-1, points.shape[-1])
        control_points = []
        for i in range(points.shape[-1]):
            control_points.append(scipy.sparse.linalg.lsqr(basis_function_values, points[:, i])[0])

        control_points = np.stack(control_points, axis=-1)
        self.control_points = control_points.reshape(*control_points_shape, -1)
