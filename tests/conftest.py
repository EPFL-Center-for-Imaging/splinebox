import numpy as np
import pytest
import splinebox.basis_functions
import splinebox.spline_curves


@pytest.fixture(
    params=[
        (splinebox.basis_functions.B1, {}),
        (splinebox.basis_functions.B2, {}),
        (splinebox.basis_functions.B3, {}),
        (splinebox.basis_functions.Exponential, {"M": 5, "alpha": 0.3}),
        (splinebox.basis_functions.CatmullRom, {}),
        (splinebox.basis_functions.CubicHermite, {}),
        (splinebox.basis_functions.ExponentialHermite, {"alpha": 0.4}),
    ],
    ids=["B1", "B2", "B3", "Exponential-M5-alpha0.3", "CatmullRom", "CubicHermite", "ExponentialHermite"],
)
def basis_function(request):
    basis_function, params = request.param
    return basis_function(**params)


@pytest.fixture(
    params=[
        (splinebox.basis_functions.B1, {}),
        (splinebox.basis_functions.B2, {}),
        (splinebox.basis_functions.B3, {}),
        (splinebox.basis_functions.Exponential, {"M": 5, "alpha": 0.3}),
        (splinebox.basis_functions.CatmullRom, {}),
    ],
    ids=["B1", "B2", "B3", "Exponential-M5-alpha0.3", "CatmullRom"],
)
def non_hermite_basis_function(request):
    basis_function, params = request.param
    return basis_function(**params)


@pytest.fixture(
    params=[
        (splinebox.basis_functions.CubicHermite, {}),
        (splinebox.basis_functions.ExponentialHermite, {"alpha": 0.4}),
    ],
    ids=["CubicHermite", "ExponentialHermite"],
)
def hermite_basis_function(request):
    basis_function, params = request.param
    return basis_function(**params)


@pytest.fixture(params=[4, 5, 8])
def M(request):
    return request.param


@pytest.fixture(params=[True, False])
def closed(request):
    return request.param


@pytest.fixture
def spline_curve(basis_function, M, closed):
    if hasattr(basis_function, "M"):
        M = basis_function.M
    if isinstance(
        basis_function, (splinebox.basis_functions.CubicHermite, splinebox.basis_functions.ExponentialHermite)
    ):
        return splinebox.spline_curves.HermiteSpline(M, basis_function, closed=closed)
    return splinebox.spline_curves.Spline(M, basis_function, closed=closed)


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
def coef_gen(codomain_dimensionality):
    rng = np.random.default_rng(seed=1492)

    def _point_gen(M, support, closed):
        if closed:
            points = rng.random((M, codomain_dimensionality)) * 100
        else:
            points = rng.random((M + support, codomain_dimensionality)) * 100
        # remove superfluos dimension if codomain_dimensionlity is 1
        return np.squeeze(points)

    return _point_gen
