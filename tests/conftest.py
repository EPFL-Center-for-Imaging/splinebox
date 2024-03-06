import pytest
import splinebox.basis_functions


@pytest.fixture(
    params=[
        (splinebox.basis_functions.B1, {}),
        (splinebox.basis_functions.B2, {}),
        (splinebox.basis_functions.B3, {}),
        (splinebox.basis_functions.EM, {"M": 5, "alpha": 0.3}),
        (splinebox.basis_functions.Keys, {}),
        (splinebox.basis_functions.H3, {}),
        (splinebox.basis_functions.HE3, {"alpha": 0.4}),
    ],
    ids=["B1", "B2", "B3", "EM-M5-alpha0.3", "Keys", "H3", "HE3"],
)
def basis_function(request):
    basis_function, params = request.param
    return basis_function(**params)
