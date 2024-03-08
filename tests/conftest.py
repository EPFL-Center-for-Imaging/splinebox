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


@pytest.fixture(params=[3, 5, 8])
def M(request):
    return request.param


@pytest.fixture(params=[True, False])
def closed(request):
    return request.param


@pytest.fixture
def spline_curve(basis_function, M, closed):
    if hasattr(basis_function, "M"):
        M = basis_function.M
    return splinebox.spline_curves.Spline(M, basis_function, closed=closed)


@pytest.fixture
def coef_gen():
    rng = np.random.default_rng(seed=1492)

    def _point_gen(M):
        return rng.random(M) * 100

    return _point_gen
