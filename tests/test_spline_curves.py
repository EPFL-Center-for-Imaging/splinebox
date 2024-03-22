import numpy as np
import pytest
import splinebox.basis_functions


def test_eval(
    spline_curve, coeff_gen, derivative, eval_positions, is_hermite_spline, not_differentiable_twice, is_interpolating
):
    support = spline_curve.basis_function.support
    half_support = support / 2
    closed = spline_curve.closed

    with pytest.raises(RuntimeError):
        # coefficients have not been set
        spline_curve.eval(eval_positions, derivative=derivative)

    # Set coefficients
    spline_curve.coeffs = coeff_gen(spline_curve.M, support, closed)

    if is_hermite_spline(spline_curve):
        with pytest.raises(RuntimeError):
            # Tangents have not been set
            spline_curve.eval(eval_positions, derivative=derivative)
        spline_curve.tangents = coeff_gen(spline_curve.M, support, closed)

    if is_interpolating(spline_curve) and derivative == 0:
        values = spline_curve.eval(np.arange(spline_curve.M), derivative=derivative)
        expected = spline_curve.coeffs if closed else spline_curve.coeffs[int(half_support) : -int(half_support)]
        assert np.allclose(values, expected)

    elif derivative == 2 and not_differentiable_twice(spline_curve):
        with pytest.raises(RuntimeError):
            spline_curve.eval(eval_positions, derivative=derivative)

    else:
        spline_curve.eval(eval_positions, derivative=derivative)


def test_set_coeffs(spline_curve, is_hermite_spline):
    """
    Test that you can only set coefficients and tangents of the right length.
    """
    expected = spline_curve.M if spline_curve.closed else int(spline_curve.M + spline_curve.basis_function.support)

    for i in range(expected - 2, expected + 2):
        coeffs = np.arange(i)
        if len(coeffs) != expected:
            with pytest.raises(ValueError):
                spline_curve.coeffs = coeffs
            if is_hermite_spline(spline_curve):
                with pytest.raises(ValueError):
                    spline_curve.tangents = coeffs
        else:
            spline_curve.coeffs = coeffs
            if is_hermite_spline(spline_curve):
                spline_curve.tangents = coeffs


def test_spline_with_hermite_basis(hermite_basis_function, M):
    """
    Test that you cannot use a Hermite basis function for a
    normal spline.
    """
    with pytest.raises(ValueError):
        splinebox.spline_curves.Spline(M, hermite_basis_function)


def test_hermite_spline_with_normal_basis(non_hermite_basis_function, M):
    """
    Test that you can only construct a hermite spline with a hermite basis
    function.
    """
    with pytest.raises(ValueError):
        splinebox.spline_curves.HermiteSpline(M, non_hermite_basis_function)


def test_closed_splines(closed_spline_curve, derivative, coeff_gen, is_hermite_spline, not_differentiable_twice):
    M = closed_spline_curve.M
    support = closed_spline_curve.basis_function.support
    closed_spline_curve.coeffs = coeff_gen(M, support, closed=True)
    if is_hermite_spline(closed_spline_curve):
        closed_spline_curve.tangents = coeff_gen(M, support, closed=True)
    if not_differentiable_twice(closed_spline_curve) and derivative == 2:
        return
    assert np.allclose(
        closed_spline_curve.eval(0, derivative=derivative),
        closed_spline_curve.eval(M, derivative=derivative),
        equal_nan=True,
    )
