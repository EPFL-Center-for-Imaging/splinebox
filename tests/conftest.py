import itertools
import math
import unittest.mock

import numpy as np
import pytest
import splinebox.basis_functions
import splinebox.spline_curves


@pytest.fixture(
    params=[
        splinebox.basis_functions.B1,
        splinebox.basis_functions.B2,
        splinebox.basis_functions.B3,
        splinebox.basis_functions.Exponential,
        splinebox.basis_functions.CatmullRom,
        splinebox.basis_functions.CubicHermite,
        splinebox.basis_functions.ExponentialHermite,
    ]
)
def basis_function(request, M):
    basis_function = request.param
    params = {}
    if basis_function in [splinebox.basis_functions.Exponential, splinebox.basis_functions.ExponentialHermite]:
        params["M"] = M
    return basis_function(**params)


@pytest.fixture(
    params=[
        (splinebox.basis_functions.B1, {}),
        (splinebox.basis_functions.B2, {}),
        (splinebox.basis_functions.B3, {}),
        (splinebox.basis_functions.Exponential, {"M": 5}),
        (splinebox.basis_functions.CatmullRom, {}),
    ],
    ids=["B1", "B2", "B3", "Exponential-M5", "CatmullRom"],
)
def non_hermite_basis_function(request):
    basis_function, params = request.param
    return basis_function(**params)


@pytest.fixture(
    params=[
        (splinebox.basis_functions.CubicHermite, {}),
        (splinebox.basis_functions.ExponentialHermite, {"M": 5}),
    ],
    ids=["CubicHermite", "ExponentialHermite"],
)
def hermite_basis_function(request):
    basis_function, params = request.param
    return basis_function(**params)


@pytest.fixture
def is_locally_refinable():
    def _is_locally_refinable(basis_function):
        return not isinstance(
            basis_function,
            (
                splinebox.basis_functions.CubicHermite,
                splinebox.basis_functions.ExponentialHermite,
                splinebox.basis_functions.CatmullRom,
            ),
        )

    return _is_locally_refinable


@pytest.fixture(params=[4, 5, 8])
def M(request):
    return request.param


@pytest.fixture(params=[True, False])
def closed(request):
    return request.param


@pytest.fixture(params=[None, splinebox.spline_curves.padding_function])
def padding_function(request):
    return request.param


@pytest.fixture
def spline_curve(basis_function, M, closed, padding_function):
    if isinstance(
        basis_function, (splinebox.basis_functions.CubicHermite, splinebox.basis_functions.ExponentialHermite)
    ):
        spline = splinebox.spline_curves.HermiteSpline(
            M, basis_function, closed=closed, padding_function=padding_function
        )
        spline._check_control_points_and_tangents = unittest.mock.MagicMock()
        spline._check_control_points = unittest.mock.MagicMock()
    else:
        spline = splinebox.spline_curves.Spline(M, basis_function, closed=closed, padding_function=padding_function)
        spline._check_control_points = unittest.mock.MagicMock()
    return spline


@pytest.fixture
def non_hermite_spline_curve(non_hermite_basis_function, M, closed):
    return splinebox.spline_curves.Spline(M, non_hermite_basis_function, closed=closed)


@pytest.fixture
def hermite_spline_curve(hermite_basis_function, M, closed):
    return splinebox.spline_curves.HermiteSpline(M, hermite_basis_function, closed=closed)


@pytest.fixture
def closed_spline_curve(basis_function, M):
    if hasattr(basis_function, "M"):
        M = basis_function.M
    if isinstance(
        basis_function, (splinebox.basis_functions.CubicHermite, splinebox.basis_functions.ExponentialHermite)
    ):
        return splinebox.spline_curves.HermiteSpline(M, basis_function, closed=True)
    return splinebox.spline_curves.Spline(M, basis_function, closed=True)


@pytest.fixture
def open_spline_curve(basis_function, M):
    if hasattr(basis_function, "M"):
        M = basis_function.M
    if isinstance(
        basis_function, (splinebox.basis_functions.CubicHermite, splinebox.basis_functions.ExponentialHermite)
    ):
        return splinebox.spline_curves.HermiteSpline(M, basis_function, closed=False)
    return splinebox.spline_curves.Spline(M, basis_function, closed=False)


@pytest.fixture(params=[0, 1, 2])
def derivative(request):
    return request.param


@pytest.fixture(params=[1.5, np.linspace(0, 10, 1000)])
def eval_positions(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 5])
def codomain_dimensionality(request):
    """
    The number of dimensions the spline output
    has. If the dimension is 2, a curve spline
    describes a line in 2D space parameterized by
    a single parameter.
    """
    return request.param


@pytest.fixture
def coeff_gen(codomain_dimensionality):
    rng = np.random.default_rng(seed=1492)

    def _coeff_gen(M, support, closed):
        if closed:
            points = rng.random((M, codomain_dimensionality)) * 100
        else:
            points = rng.random((M + 2 * (math.ceil(support / 2) - 1), codomain_dimensionality)) * 100
        # remove superfluos dimension if codomain_dimensionlity is 1
        return np.squeeze(points)

    return _coeff_gen


@pytest.fixture
def knot_gen(codomain_dimensionality):
    rng = np.random.default_rng(seed=2657)

    def _knot_gen(M=100):
        return np.squeeze(rng.random((M, codomain_dimensionality)))

    return _knot_gen


@pytest.fixture
def is_spline():
    def _is_spline(obj):
        return isinstance(obj, (splinebox.spline_curves.Spline, splinebox.spline_curves.HermiteSpline))

    return _is_spline


@pytest.fixture
def is_interpolating(is_spline):
    def _is_interpolating(obj):
        basis_function = obj.basis_function if is_spline(obj) else obj
        return np.allclose(basis_function.eval(0), 1)

    return _is_interpolating


@pytest.fixture
def not_differentiable_twice(is_spline):
    def _not_differentiable_twice(obj):
        basis_function = obj.basis_function if is_spline(obj) else obj
        return isinstance(
            basis_function,
            (
                splinebox.basis_functions.B1,
                splinebox.basis_functions.CubicHermite,
                splinebox.basis_functions.ExponentialHermite,
            ),
        )

    return _not_differentiable_twice


@pytest.fixture
def is_hermite_spline():
    def _is_hermite_spline(spline_curve):
        return isinstance(
            spline_curve,
            splinebox.spline_curves.HermiteSpline,
        )

    return _is_hermite_spline


@pytest.fixture
def translation_vector(codomain_dimensionality):
    return np.random.rand(codomain_dimensionality) * 10 - 5


@pytest.fixture
def initialized_spline_curve(spline_curve, is_hermite_spline, coeff_gen):
    support = spline_curve.basis_function.support
    closed = spline_curve.closed
    M = spline_curve.M

    spline_curve.control_points = coeff_gen(M, support, closed)
    if is_hermite_spline(spline_curve):
        spline_curve.tangents = coeff_gen(spline_curve.M, support, closed)

    return spline_curve


@pytest.fixture
def rotation_matrix(codomain_dimensionality):
    rng = np.random.default_rng(seed=9577)

    R = np.eye(codomain_dimensionality)
    if codomain_dimensionality == 1:
        return R
    # Get the axis of all possible rotation planes aligned with the axis
    plane_indices = list(itertools.combinations(np.arange(codomain_dimensionality), 2))
    # randomly choose an angle between -pi and pi for each rotation plane
    thetas = rng.random(len(plane_indices)) * 2 * np.pi - np.pi
    for (i, j), theta in zip(plane_indices, thetas):
        R_theta = np.eye(codomain_dimensionality)
        R_theta[i, i] = np.cos(theta)
        R_theta[i, j] = np.sin(theta)
        R_theta[j, i] = -np.sin(theta)
        R_theta[j, j] = np.cos(theta)
        R = R @ R_theta
    return R


@pytest.fixture(params=[False, True])
def arc_length_parametrization(request):
    return request.param


@pytest.fixture(params=[2, 20, 50])
def n_points(request):
    return request.param


@pytest.fixture
def points(codomain_dimensionality, n_points):
    rng = np.random.default_rng(seed=5544)
    return np.squeeze(rng.random((n_points, codomain_dimensionality)))
