import math
import unittest.mock

import numpy as np
import pytest
import scipy
import splinebox.basis_functions


def test_check_control_points(non_hermite_spline_curve, coeff_gen):
    spline = non_hermite_spline_curve
    with pytest.raises(RuntimeError):
        spline._check_control_points()
    spline.control_points = coeff_gen(spline.M, spline.basis_function.support, spline.closed)
    spline._check_control_points()


def test_check_control_points_and_tangents(hermite_spline_curve, coeff_gen):
    spline = hermite_spline_curve
    with pytest.raises(RuntimeError):
        spline._check_control_points_and_tangents()
    spline.control_points = coeff_gen(spline.M, spline.basis_function.support, spline.closed)
    with pytest.raises(RuntimeError):
        spline._check_control_points_and_tangents()
    spline.tangents = coeff_gen(spline.M, spline.basis_function.support, spline.closed)
    spline._check_control_points_and_tangents()


def test_minimum_number_of_knots(basis_function):
    M = basis_function.support - 1
    with pytest.raises(RuntimeError):
        splinebox.spline_curves.Spline(M, basis_function)


def test_eval(
    spline_curve, coeff_gen, derivative, eval_positions, is_hermite_spline, not_differentiable_twice, is_interpolating
):
    support = spline_curve.basis_function.support
    half_support = support / 2
    closed = spline_curve.closed

    # Set coefficients and tangents
    spline_curve.control_points = coeff_gen(spline_curve.M, support, closed)
    if is_hermite_spline(spline_curve):
        spline_curve.tangents = coeff_gen(spline_curve.M, support, closed)

    if is_interpolating(spline_curve) and derivative == 0:
        values = spline_curve.eval(np.arange(spline_curve.M), derivative=derivative)
        expected = spline_curve.control_points
        if not closed:
            pad = math.ceil(half_support) - 1
            if pad != 0:
                expected = expected[pad:-pad]
        assert np.allclose(values, expected)

    elif derivative == 2 and not_differentiable_twice(spline_curve):
        with pytest.raises(RuntimeError):
            spline_curve.eval(eval_positions, derivative=derivative)

    else:
        spline_curve.eval(eval_positions, derivative=derivative)

    # Check that the presence of the coefficients and tangents was verified
    if is_hermite_spline(spline_curve):
        spline_curve._check_control_points_and_tangents.assert_called_once()
    else:
        spline_curve._check_control_points.assert_called_once()


def test_set_control_points(spline_curve, is_hermite_spline):
    """
    Test that you can only set coefficients and tangents of the right length.
    """
    if spline_curve.closed:
        expected = spline_curve.M
    else:
        expected = spline_curve.M + 2 * (math.ceil(spline_curve.basis_function.support / 2) - 1)

    for i in range(expected - 2, expected + 2):
        control_points = np.arange(i)
        if len(control_points) != expected:
            with pytest.raises(ValueError):
                spline_curve.control_points = control_points
            if is_hermite_spline(spline_curve):
                with pytest.raises(ValueError):
                    spline_curve.tangents = control_points
        else:
            spline_curve.control_points = control_points
            if is_hermite_spline(spline_curve):
                spline_curve.tangents = control_points


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
    closed_spline_curve.control_points = coeff_gen(M, support, closed=True)
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
    spline._check_control_points = unittest.mock.MagicMock()
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

    # Check that the presence of the coefficients was verified
    spline._check_control_points.assert_called()

    # You can only draw closed splines
    spline.closed = False
    with pytest.raises(RuntimeError):
        spline.draw(x, y)

    # You can only draw splines in 2D
    spline = splinebox.spline_curves.Spline(M=4, basis_function=splinebox.basis_functions.B1(), closed=True)
    spline.knots = np.array([[1, 1, 0.5], [1, 2, 2], [2, 2, 1], [2, 1, 0]])
    with pytest.raises(RuntimeError):
        spline.draw(x, y)


def test_is_inside():
    # The function itself is already tested via test_draw.
    # Here, we will just test that it raises the correct errors

    # Only makes sense for closed splines.
    spline = splinebox.spline_curves.Spline(M=4, basis_function=splinebox.basis_functions.B1(), closed=False)
    with pytest.raises(RuntimeError):
        spline.is_inside(0, 0)

    # Only makes sense for 2D curves
    spline.closed = True
    spline.knots = np.array([[1, 2, 3], [4, 3, 2], [6, 5, 1], [4, 7, 9]])
    with pytest.raises(RuntimeError):
        spline.is_inside(0, 0)

    # Single value input
    spline.knots = np.array([[1, 1], [1, 2], [2, 2], [2, 1]])
    assert not spline.is_inside(0, 0)
    assert spline.is_inside(1.5, 1.5)

    # x and y vectors with different shapes
    with pytest.raises(ValueError):
        spline.is_inside(np.arange(10), np.arange(15))


def test_arc_length():
    # Create circular spline with radius sqrt(2)
    M = 10
    basis_function = splinebox.basis_functions.Exponential(M)
    spline = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=True)
    spline._check_control_points = unittest.mock.MagicMock()
    phi = np.linspace(0, 2 * np.pi, M + 1)[:-1]
    knots = np.stack([np.cos(phi), np.sin(phi)], axis=-1)
    spline.knots = knots

    ts = np.linspace(0, M, 100)

    # Reset mock in case it was called by any of the methods befor
    spline._check_control_points.reset_mock()

    arc_lengths = [spline.arc_length(t) for t in ts]
    arc_lengths = np.array(arc_lengths)

    # Check that the presence of the coefficients was verified
    spline._check_control_points.assert_called()

    expected = np.linspace(0, 2 * np.pi, 100)
    assert np.allclose(arc_lengths, expected)

    # Check that it is working with a vector for stop
    permutation = np.random.permutation(len(ts))
    arc_lengths = spline.arc_length(ts[permutation])
    assert np.allclose(arc_lengths, expected[permutation])

    # Check that is works with vectors for start and stop
    arc_lengths = spline.arc_length(ts[permutation], np.zeros(len(ts)))
    assert np.allclose(arc_lengths, expected[permutation])


def test_arc_length_to_parameter():
    # Create circular spline with radius sqrt(2)
    M = 100
    basis_function = splinebox.basis_functions.Exponential(M)
    spline = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=True)
    spline._check_control_points = unittest.mock.MagicMock()
    knots = []
    for t in np.linspace(0, 2 * np.pi, M + 1)[:-1]:
        knots.append([np.cos(t), np.sin(t)])
    knots = np.array(knots)
    spline.knots = knots

    ls = np.linspace(0, 2 * np.pi, 15)
    expected = np.linspace(0, M, 15)  # ls / 2 / np.pi * M
    atol = 1e-5

    # Reset mock in case it was called by any of the methods befor
    spline._check_control_points.reset_mock()

    results = spline.arc_length_to_parameter(ls, atol=atol)

    # Check that the presence of the coefficients was verified
    spline._check_control_points.assert_called()

    assert np.allclose(results, expected, atol=1e-3, rtol=0)

    # Create a sawtooth spline
    M = 7
    basis_function = splinebox.basis_functions.B1()
    spline = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=False)
    spline._check_control_points = unittest.mock.MagicMock()
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

    expected = np.array(list(map(to_param, ls)))
    atol = 1e-4

    # Reset mock in case it was called by any of the methods befor
    spline._check_control_points.reset_mock()

    results = spline.arc_length_to_parameter(ls, atol=atol)

    # Check that the presence of the coefficients was verified
    spline._check_control_points.assert_called()

    assert np.allclose(results, expected, atol=atol)

    # Check that is works for a single value
    expected = to_param(ls[10])
    result = spline.arc_length_to_parameter(ls[10], atol=atol)
    assert np.isclose(result, expected, atol=atol)


def test_translate(initialized_spline_curve, translation_vector):
    spline = initialized_spline_curve
    spline_copy = spline.copy()
    spline_copy._check_control_points = unittest.mock.MagicMock()
    spline_copy.translate(translation_vector)
    # Check that the presence of the coefficients was verified
    spline_copy._check_control_points.assert_called()
    t = np.linspace(0, spline.M, 100) if spline.closed else np.linspace(0, spline.M - 1, 100)
    expected = spline.eval(t) + translation_vector
    assert np.allclose(spline_copy.eval(t), expected)


def test_rotate(initialized_spline_curve, rotation_matrix, is_hermite_spline):
    spline = initialized_spline_curve
    spline_copy = spline.copy()
    spline_copy._check_control_points = unittest.mock.MagicMock()
    if spline.control_points.ndim == 1:
        with pytest.raises(RuntimeError):
            spline_copy.rotate(rotation_matrix)
    else:
        spline_copy.rotate(rotation_matrix, centred=False)
        # Check that the presence of the coefficients was verified
        spline_copy._check_control_points.assert_called()
        t = np.linspace(0, spline.M, 100) if spline.closed else np.linspace(0, spline.M - 1, 100)
        vals = spline.eval(t)
        expected = (rotation_matrix @ vals.T).T
        assert np.allclose(spline_copy.eval(t), expected)

        spline_copy = spline.copy()
        spline_copy._check_control_points = unittest.mock.MagicMock()
        spline_copy.rotate(rotation_matrix, centred=True)
        # Check that the presence of the coefficients was verified
        spline_copy._check_control_points.assert_called()
        t = np.linspace(0, spline.M, 100) if spline.closed else np.linspace(0, spline.M - 1, 100)
        vals = spline.eval(t)
        centring_vector = np.mean(spline.control_points, axis=0)
        vals = vals - centring_vector
        expected = (rotation_matrix @ vals.T).T
        expected = expected + centring_vector
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
        control_points_fit = spline_curve.control_points.copy()
        if hermite:
            tangents_fit = spline_curve.tangents.copy()
            half = len(tangents_fit)

        # Calculate the parameter values for the data points
        if spline_curve.closed:
            t = np.linspace(0, spline_curve.M, len(points) + 1)[:-1]
        else:
            t = np.linspace(0, spline_curve.M - 1, len(points))

        # Add a dimension if the codomain dimensionality is 1
        if points.ndim == 1:
            points = points[:, np.newaxis]

        # Define the objective function for the minimization problem
        def difference_func(x, i):
            if hermite:
                spline_curve.control_points = x[:half]
                spline_curve.tangents = x[half:]
            else:
                spline_curve.control_points = x
            spline_vals = spline_curve.eval(t)
            loss = np.linalg.norm(points[:, i] - spline_vals)
            return loss

        # prepare the initial value for the minimization
        x0 = np.concatenate([control_points_fit, tangents_fit], axis=0) if hermite else control_points_fit

        if x0.ndim == 1:
            x0 = x0[:, np.newaxis]

        # Minimize for each codomain dimension separately
        x = []
        for i in range(x0.shape[1]):
            x.append(scipy.optimize.minimize(difference_func, x0=x0[:, i], args=(i,)).x)

        # Turn minimization result into control_points and tangents for the spline
        x = np.stack(x, axis=-1)
        if hermite:
            control_points_expected = np.squeeze(x[:half])
            tangents_expected = np.squeeze(x[half:])
        else:
            control_points_expected = np.squeeze(x)

        assert np.allclose(control_points_expected, control_points_fit)
        if hermite:
            assert np.allclose(tangents_expected, tangents_fit)


def test_knots(spline_curve, knot_gen, is_hermite_spline, request):
    if isinstance(spline_curve.basis_function, splinebox.basis_functions.B2):
        # The filter_symmetric and filter_periodic are not implemented for B2
        request.node.add_marker(pytest.mark.xfail)

    knots = knot_gen(spline_curve.M)
    pad = math.ceil(spline_curve.basis_function.support / 2) - 1
    if spline_curve.padding_function is None and not spline_curve.closed:
        knots = splinebox.spline_curves.padding_function(knots, pad)
        knots = np.squeeze(knots)
    spline_curve.knots = knots

    if is_hermite_spline(spline_curve):
        # Use finite differences as tangents
        if spline_curve.closed:
            if knots.ndim == 1:
                diff = np.diff(knots, axis=0, append=knots[0])
            else:
                diff = np.diff(knots, axis=0, append=knots[0][np.newaxis, :])
            tangents = (diff + np.roll(diff, 1)) / 2
        else:
            diff = np.diff(knots, axis=0)
            pad = math.ceil(spline_curve.basis_function.support / 2) - 1
            if diff.ndim == 1:
                diff = diff[:, np.newaxis]
            diff = np.pad(diff, ((pad + 1, pad + 1), (0, 0)), mode="constant", constant_values=(0,))
            tangents = (diff[:-1] + diff[1:]) / 2
            tangents = np.squeeze(tangents)
        spline_curve.tangents = tangents

    # This assert makes sense because the knots are not saved in
    # the spline class but are converted to coefficients which are saved.
    if spline_curve.padding_function is None and not spline_curve.closed and pad != 0:
        # We only compare the actual knots because padded knots will not
        # be the same because the of the filtering to find the control points.
        assert np.allclose(knots[pad:-pad], spline_curve.knots[pad:-pad])
    else:
        assert np.allclose(knots, spline_curve.knots)


def test_number_of_knots(spline_curve, knot_gen):
    for n_knots in (spline_curve.M - 1, spline_curve.M + 1):
        knots = knot_gen(n_knots)
        with pytest.raises(ValueError):
            spline_curve.knots = knots

    if not spline_curve.closed and spline_curve.pad > 0:
        knots = knot_gen(spline_curve.M)

        # Test with no padding function
        spline_curve.padding_function = None
        with pytest.raises(ValueError):
            spline_curve.knots = knots

        # Test with custom padding function
        spline_curve.padding_function = lambda knots, pad: knots
        with pytest.raises(ValueError):
            spline_curve.knots = knots


def test_centroid(spline_curve, coeff_gen, is_hermite_spline):
    support = spline_curve.basis_function.support
    closed = spline_curve.closed
    M = spline_curve.M

    control_points = coeff_gen(M, support, closed)
    spline_curve.control_points = control_points
    if is_hermite_spline(spline_curve):
        tangents = coeff_gen(spline_curve.M, support, closed)
        spline_curve.tangents = tangents

    expected = np.mean(control_points, axis=0)
    assert np.allclose(spline_curve._control_points_centroid(), expected)

    spline_curve._check_control_points.assert_called()


def test_scale(initialized_spline_curve, is_hermite_spline):
    spline = initialized_spline_curve
    spline_copy = spline.copy()

    factor = 1.5

    centroid = spline_copy._control_points_centroid()
    spline_copy.translate(-centroid)
    spline_copy.control_points = spline_copy.control_points * factor
    spline_copy.translate(centroid)

    if is_hermite_spline(spline_copy):
        spline_copy.tangents = spline_copy.tangents * factor

    t = np.linspace(0, spline.M, 100) if spline.closed else np.linspace(0, spline.M - 1, 100)

    # Reset mocks in case they were called by another method already
    if is_hermite_spline(spline_copy):
        spline._check_control_points_and_tangents.reset_mock()
    else:
        spline._check_control_points.reset_mock()

    spline.scale(factor)

    # Check that coefficients and tangents were checked
    if is_hermite_spline(spline_copy):
        spline._check_control_points_and_tangents.assert_called()
    else:
        spline._check_control_points.assert_called()

    assert np.allclose(spline.eval(t), spline_copy.eval(t))


def test_curvilinear_reparametrization_energy():
    M = 4
    spline = splinebox.spline_curves.Spline(M, splinebox.B1())

    spline.knots = np.array([[1, 0], [2, 0], [3, 0], [4, 0]])

    # Because of the spacing of the knots the derivative should be constant so we can just compute one value in the middle of the spline.
    derivative_val = spline.eval((spline.M - 1) / 2, derivative=1)

    # To compute the expected value we exploit the constant derivative.
    # We can replace the integration in the definition of the curvilinear reparametrization energy with a multiplication because the derivative is constant.
    length = spline.arc_length()
    c = (length / spline.M) ** 2
    e_curv = (np.linalg.norm(derivative_val) ** 2 - c) ** 2
    expected_val = e_curv * (spline.M - 1) / length**4

    val = spline.curvilinear_reparametrization_energy()

    assert np.isclose(val, expected_val)


def test_curvilinear_reparametrization_energy_translation(initialized_spline_curve, translation_vector):
    """
    Test if the curvilinear reparametrization energy is invariant to translation.
    """
    spline = initialized_spline_curve
    expected = spline.curvilinear_reparametrization_energy()
    spline.translate(translation_vector)
    val = spline.curvilinear_reparametrization_energy()
    assert np.isclose(val, expected)


def test_curvilinear_reparametrization_energy_scale_invariance(initialized_spline_curve):
    """
    The curvilinear reparametrization energy should not change when the entire spline is scaled.
    """
    spline = initialized_spline_curve
    expected = spline.curvilinear_reparametrization_energy()
    spline.scale(0.1)
    val = spline.curvilinear_reparametrization_energy()
    assert np.isclose(val, expected)


def test_curvature():
    # Create a circular spline
    M = 4
    spline = splinebox.spline_curves.Spline(M=M, basis_function=splinebox.basis_functions.Exponential(M), closed=True)
    t = np.linspace(0, M, 100)
    for r in (1, 1.5, 2):
        # 2D
        spline.knots = np.array([[0, r], [r, 0], [0, -r], [-r, 0]])
        curvature = spline.curvature(t)
        assert np.allclose(curvature, 1 / r)

        # If we reverse the direction, the curvature should change sign
        spline.knots = np.array([[0, r], [-r, 0], [0, -r], [r, 0]])
        curvature = spline.curvature(t)
        assert np.allclose(curvature, -1 / r)

        # 3D
        spline.knots = np.array([[0, 0, r], [0, r, 0], [0, 0, -r], [0, -r, 0]])
        curvature = spline.curvature(t)
        assert np.allclose(curvature, 1 / r)

    # 1D
    spline = splinebox.spline_curves.Spline(M=M, basis_function=splinebox.basis_functions.B3(), closed=False)
    spline.knots = np.array([1, 5, 4, 2])
    expected = spline.eval(t, derivative=2) / (1 + spline.eval(t, derivative=1) ** 2) ** (3 / 2)
    result = spline.curvature(t)
    assert np.allclose(result, expected)


def test_normal():
    # Create a circular spline
    M = 4
    spline = splinebox.spline_curves.Spline(M=M, basis_function=splinebox.basis_functions.Exponential(M), closed=True)
    t = np.linspace(0, M, 100)

    theta = np.linspace(0, 2 * np.pi, 100)
    expected = np.stack([np.sin(theta), np.cos(theta)], axis=-1)
    for r in (1, 1.5, 2):
        spline.knots = np.array([[0, r], [r, 0], [0, -r], [-r, 0]])
        normals = spline.normal(t)
        assert np.allclose(normals, expected)

        # Reversing the direction change the direction of the normal vector
        spline.knots = np.array([[0, r], [-r, 0], [0, -r], [r, 0]])
        normals = spline.normal(t[::-1])
        assert np.allclose(normals, -1 * expected)

        # Check that they are normal vectors
        assert np.allclose(np.linalg.norm(normals, axis=1), np.ones(len(t)))
