import numpy as np
import pytest
import splinebox.basis_functions


def _is_interpolating(spline_curve):
    return np.allclose(spline_curve.basis_function.eval(0), 1)


def _not_differentiable_twice(spline_curve):
    return isinstance(
        spline_curve.basis_function,
        (
            splinebox.basis_functions.B1,
            splinebox.basis_functions.CubicHermite,
            splinebox.basis_functions.ExponentialHermite,
        ),
    )


def _is_hermite_spline(spline_curve):
    return isinstance(
        spline_curve.basis_function,
        (
            splinebox.basis_functions.CubicHermite,
            splinebox.basis_functions.ExponentialHermite,
        ),
    )


def test_eval(spline_curve, coef_gen, derivative, eval_positions):
    support = spline_curve.basis_function.support
    half_support = support / 2
    closed = spline_curve.closed

    with pytest.raises(RuntimeError):
        # coefficients have not been set
        spline_curve.eval(eval_positions, derivative=derivative)

    # Set coefficients
    spline_curve.coefs = coef_gen(spline_curve.M, support, closed)

    if _is_hermite_spline(spline_curve):
        with pytest.raises(RuntimeError):
            # Tangents have not been set
            spline_curve.eval(eval_positions, derivative=derivative)
        spline_curve.tangents = coef_gen(spline_curve.M, support, closed)

    if _is_interpolating(spline_curve) and derivative == 0:
        values = spline_curve.eval(np.arange(spline_curve.M), derivative=derivative)
        expected = spline_curve.coefs if closed else spline_curve.coefs[int(half_support) : -int(half_support)]
        assert np.allclose(values, expected)

    elif derivative == 2 and _not_differentiable_twice(spline_curve):
        with pytest.raises(RuntimeError):
            spline_curve.eval(eval_positions, derivative=derivative)

    else:
        spline_curve.eval(eval_positions, derivative=derivative)


def test_set_coefs(spline_curve):
    expected = spline_curve.M if spline_curve.closed else int(spline_curve.M - spline_curve.basis_function.support)

    for i in range(expected - 2, expected + 2):
        coefs = np.arange(i)
        if len(coefs) != expected:
            with pytest.raises(RuntimeError):
                spline_curve.coefs = coefs
            if _is_hermite_spline(spline_curve):
                with pytest.raises(RuntimeError):
                    spline_curve.tangents = coefs
        else:
            spline_curve.coefs = coefs
            if _is_hermite_spline(spline_curve):
                spline_curve.tangents = coefs
