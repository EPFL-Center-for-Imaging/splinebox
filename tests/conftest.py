import pytest
import splinebox.basis_functions


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
