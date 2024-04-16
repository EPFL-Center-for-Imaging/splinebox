import math

import numpy
import numpy as np
import pytest
import scipy
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
    if spline_curve.closed:
        expected = spline_curve.M
    else:
        expected = spline_curve.M + 2 * math.ceil(spline_curve.basis_function.support / 2)

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


def test_draw():
    spline = splinebox.spline_curves.Spline(M=4, basis_function=splinebox.basis_functions.B1(), closed=True)
    spline.knots = np.array([[1, 1], [1, 2], [2, 2], [2, 1]])

    x = np.linspace(0, 3, 31)
    y = np.linspace(0, 3, 31)

    expected = np.zeros((len(x), len(y)))
    expected[11:20, 11:20] = 1
    expected[10, 10:21] = 0.5
    expected[20, 10:21] = 0.5
    expected[10:21, 10] = 0.5
    expected[10:21, 20] = 0.5

    output = spline.draw(x, y)
    assert np.allclose(output, expected)


def test_arc_length():
    # Create circular spline with radius sqrt(2)
    M = 4
    basis_function = splinebox.basis_functions.Exponential(
        M,
        2.0 * numpy.pi / M,
    )
    spline = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=True)
    spline.knots = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])

    t = np.linspace(0, M - 1, 100)
    arc_lengths = spline.arc_length(t)

    expected = np.linspace(0, 2 * np.pi, 100)

    assert np.allclose(arc_lengths, expected)


def test_arc_length_to_parameter():
    # Create circular spline with radius sqrt(2)
    M = 5
    basis_function = splinebox.basis_functions.Exponential(
        M,
        2.0 * numpy.pi / M,
    )
    spline = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=True)
    knots = []
    for t in np.linspace(0, 2 * np.pi, M + 1)[:-1]:
        knots.append([np.sin(t), np.cos(t)])
    knots = np.array(knots)
    spline.knots = knots

    ls = np.linspace(0, 2 * np.pi, 15)
    expected = ls / 2 / np.pi * M
    assert np.allclose(spline.arc_length_to_parameter(ls), expected)

    # Create a sawtooth spline
    M = 7
    basis_function = splinebox.basis_functions.B1()
    spline = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=False)
    spline.knots = np.array([[0, 0], [1, 1], [1, 0], [2, 1], [2, 0], [3, 1], [3, 0]])

    rising_length = np.sqrt(2)
    tooth_length = rising_length + 1
    total_length = 3 * tooth_length
    ls = np.linspace(0, total_length, 15)

    def to_param(length):
        n_tooth = length // tooth_length
        residual = length % tooth_length
        partial_tooth_param = residual / rising_length if residual < rising_length else 1 + residual - rising_length
        # there are two knots per tooth
        return n_tooth * 2 + partial_tooth_param

    expected = map(to_param, ls)
    assert np.allclose(spline.arc_length_to_parameter(ls), expected)


def test_translate(initialized_spline_curve, translation_vector):
    spline = initialized_spline_curve
    spline_copy = spline.copy()
    spline_copy.translate(translation_vector)
    t = np.linspace(0, spline.M - 1, 100) if spline.closed else np.linspace(0, spline.M, 100)
    expected = spline.eval(t) + translation_vector
    assert np.allclose(spline_copy.eval(t), expected)


def test_rotate(initialized_spline_curve, rotation_matrix, is_hermite_spline):
    spline = initialized_spline_curve
    spline_copy = spline.copy()
    if spline.coeffs.ndim == 1:
        with pytest.raises(RuntimeError):
            spline_copy.rotate(rotation_matrix)
    else:
        spline_copy.rotate(rotation_matrix)
        t = np.linspace(0, spline.M - 1, 100) if spline.closed else np.linspace(0, spline.M, 100)
        vals = spline.eval(t)
        if is_hermite_spline(spline):
            expected = []
            for i in range(2):
                expected.append((rotation_matrix @ vals[i].T).T)
            expected = np.stack(expected)
        else:
            expected = (rotation_matrix @ vals.T).T
        assert np.allclose(spline_copy.eval(t), expected)


def test_fit(spline_curve, arc_length_parametrization, points, is_hermite_spline):
    # Bool indicating whether spline_curve is a hermite spline or not
    hermite = is_hermite_spline(spline_curve)

    if len(points) < spline_curve.M:
        # The problem is underdetermined
        with pytest.raises(RuntimeError):
            spline_curve.fit(points)
    else:
        spline_curve.fit(points)

        # Keep a copy of the result of fit
        coeffs0 = spline_curve.coeffs.copy()
        if hermite:
            tangents0 = spline_curve.tangents.copy()
            half = len(tangents0)

        # Calculate the parameter values for the data points
        if spline_curve.closed:
            t = np.linspace(0, spline_curve.M, len(points) + 1)[:-1]
        else:
            t = np.linspace(0, spline_curve.M, len(points))

        # Add a dimension if the codomain dimensionality is 1
        if points.ndim == 1:
            points = points[:, np.newaxis]

        # Define the objective function for the minimization problem
        def difference_func(x, i):
            if hermite:
                spline_curve.coeffs = x[:half]
                spline_curve.tangents = x[half:]
                spline_vals = spline_curve.eval(t)[0]
            else:
                spline_curve.coeffs = x
                spline_vals = spline_curve.eval(t)
            loss = np.linalg.norm(points[:, i] - spline_vals)
            return loss

        # prepare the initial value for the minimization
        x0 = np.concatenate([coeffs0, tangents0], axis=0) if hermite else coeffs0

        if x0.ndim == 1:
            x0 = x0[:, np.newaxis]

        # Minimize for each codomain dimension separately
        x = []
        for i in range(x0.shape[1]):
            x.append(scipy.optimize.minimize(difference_func, x0=x0[:, i], args=(i,)).x)

        # Turn minimization result into coeffs and tangents for the spline
        x = np.stack(x, axis=-1)
        if hermite:
            coeffs = np.squeeze(x[:half])
            tangents = np.squeeze(x[half:])
        else:
            coeffs = np.squeeze(x)

        assert np.allclose(coeffs, coeffs0)
        if hermite:
            assert np.allclose(tangents, tangents0)


def test_knots(spline_curve, knot_gen, is_hermite_spline):
    knots = knot_gen(spline_curve.M)
    spline_curve.knots = knots

    if is_hermite_spline:
        if spline_curve.closed:
            tangents = knots
        else:
            tangents = knots[:, np.newaxis] if knots.ndim == 1 else knots
            pad = math.ceil(spline_curve.basis_function.support / 2)
            tangents = np.pad(tangents, ((pad, pad), (0, 0)), mode="edge")
            tangents = np.squeeze(tangents)
        spline_curve.tangents = tangents

    # This assert makes sense because the knots are not saved in
    # the spline class but are converted to coefficients which are saved.
    if spline_curve.closed:
        assert np.allclose(knots, spline_curve.knots)
    else:
        pad = math.ceil(spline_curve.basis_function.support / 2)
        assert np.allclose(knots, spline_curve.knots[pad:-pad])
