import numpy as np
import pytest
import splinebox.basis_functions


def _is_interpolating(spline_curve):
    return spline_curve.basis_function.eval(0) == 1


def _not_differentiable_twice(spline_curve):
    return spline_curve.basis_function in [
        splinebox.basis_functions.B1,
        splinebox.basis_functions.CubicHermite,
        splinebox.basis_functions.ExponentialHermite,
    ]


def test_eval(spline_curve, coef_gen):
    for derivative in range(3):
        for x in [1.5, np.linspace(0, spline_curve.M, 1000)]:
            with pytest.raises(RuntimeError):
                # coefficients have not been set
                spline_curve.eval(x, derivative=derivative)

            # Set coefficients
            spline_curve.coefs = coef_gen(spline_curve.M)

            if _is_interpolating(spline_curve) and derivative == 0:
                values = spline_curve.eval(np.arange(spline_curve.M + 1), derivative=derivative)
                assert np.all_close(values, spline_curve.coefs)

            if derivative == 2 and _not_differentiable_twice(spline_curve):
                with pytest.raises(RuntimeError):
                    spline_curve.eval(x, derivative=derivative)
