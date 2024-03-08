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


def test_eval(spline_curve, coef_gen, derivative, eval_positions):
    support = spline_curve.basis_function.support
    half_support = support / 2
    closed = spline_curve.closed

    with pytest.raises(RuntimeError):
        # coefficients have not been set
        spline_curve.eval(eval_positions, derivative=derivative)

    # Set coefficients
    spline_curve.coefs = coef_gen(spline_curve.M, support, closed)

    if _is_interpolating(spline_curve) and derivative == 0:
        values = spline_curve.eval(np.arange(spline_curve.M), derivative=derivative)
        expected = spline_curve.coefs if closed else spline_curve.coefs[int(half_support) : -int(half_support)]
        assert np.allclose(values, expected)

    if derivative == 2 and _not_differentiable_twice(spline_curve):
        with pytest.raises(RuntimeError):
            spline_curve.eval(eval_positions, derivative=derivative)
