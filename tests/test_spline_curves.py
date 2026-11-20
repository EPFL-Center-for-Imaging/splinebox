import json
import math

import array_api_compat
import numpy as np
import pytest
import scipy
import splinebox


def test_get_control_points_xp_and_device_raises(non_hermite_spline_curve):
    spline = non_hermite_spline_curve
    # Before setting the control points
    with pytest.raises(RuntimeError):
        spline._get_control_points_xp_and_device()


def test_get_control_points_xp_and_device(non_hermite_spline_curve, coeff_gen, xp, device):
    spline = non_hermite_spline_curve
    spline.control_points = coeff_gen(spline.M, spline.basis_function.support, spline.closed)
    returned_xp, returned_device = spline._get_control_points_xp_and_device()
    assert returned_xp is xp
    assert returned_device == device


def test_get_control_points_and_tangents_xp_and_device_raises(hermite_spline_curve, coeff_gen):
    spline = hermite_spline_curve.copy()

    # No control points and no tangents
    with pytest.raises(RuntimeError):
        spline._get_control_points_and_tangents_xp_and_device()

    # No tangents
    spline.control_points = coeff_gen(spline.M, spline.basis_function.support, spline.closed)
    with pytest.raises(RuntimeError):
        spline._get_control_points_and_tangents_xp_and_device()

    # No control points
    spline = hermite_spline_curve.copy()
    spline.tangents = coeff_gen(spline.M, spline.basis_function.support, spline.closed)
    with pytest.raises(RuntimeError):
        spline._get_control_points_and_tangents_xp_and_device()


def test_get_control_points_and_tangents_xp_and_device(hermite_spline_curve, coeff_gen, xp, device):
    spline = hermite_spline_curve
    spline.control_points = coeff_gen(spline.M, spline.basis_function.support, spline.closed)
    spline.tangents = coeff_gen(spline.M, spline.basis_function.support, spline.closed)
    returned_xp, returned_device = spline._get_control_points_and_tangents_xp_and_device()
    assert returned_xp is xp
    assert returned_device == device


def test_minimum_number_of_knots(basis_function):
    M = basis_function.support - 1
    with pytest.raises(RuntimeError):
        splinebox.spline_curves.Spline(M, basis_function)


def test_call(
    spline_curve, coeff_gen, derivative, call_positions, is_hermite_spline, is_interpolating, xp, device, dtype
):
    support = spline_curve.basis_function.support
    half_support = support / 2
    closed = spline_curve.closed

    expected_dtype = xp.asarray(1.0).dtype if dtype is int else dtype

    # Set coefficients and tangents
    spline_curve.control_points = coeff_gen(spline_curve.M, support, closed)
    if is_hermite_spline(spline_curve):
        spline_curve.tangents = coeff_gen(spline_curve.M, support, closed)

    if is_interpolating(spline_curve) and derivative == 0:
        t = xp.arange(spline_curve.M, dtype=dtype, device=device)
        values = spline_curve(t, derivative=derivative)

        assert array_api_compat.is_array_api_obj(values)
        assert array_api_compat.array_namespace(values) is xp
        assert xp.isdtype(values.dtype, kind="real floating")
        assert values.dtype == expected_dtype

        expected = xp.astype(spline_curve.control_points, expected_dtype)
        if not closed:
            pad = math.ceil(half_support) - 1
            if pad != 0:
                expected = expected[pad:-pad]
        assert xp.allclose(values, expected)

    else:
        output = spline_curve(call_positions, derivative=derivative)
        assert array_api_compat.is_array_api_obj(output)
        assert array_api_compat.array_namespace(output) is xp
        assert xp.isdtype(output.dtype, kind="real floating")
        assert output.dtype == expected_dtype


def test_raises_without_control_points(spline_curve, call_positions, translation_vector):
    """
    Methods that rely on the control points raise a RuntimeError
    if they are called before the control points (and tangents) are set.
    """
    # The spline_curve fixture returns splines without control points set
    with pytest.raises(RuntimeError):
        spline_curve(call_positions)
    with pytest.raises(RuntimeError):
        spline_curve.arc_length(1.0)
    with pytest.raises(RuntimeError):
        spline_curve._control_points_centroid()
    with pytest.raises(RuntimeError):
        spline_curve.translate(translation_vector)
    with pytest.raises(RuntimeError):
        spline_curve.scale(1.5)


def test_set_control_points(spline_curve, is_hermite_spline):
    """
    Test that you can only set coefficients and tangents of the right length.
    """
    if spline_curve.closed:
        expected = spline_curve.M
    else:
        expected = spline_curve.M + 2 * (math.ceil(spline_curve.basis_function.support / 2) - 1)

    # Array has to be 2D
    with pytest.raises(ValueError):
        # 1D
        spline_curve.control_points = np.arange(expected)
    with pytest.raises(ValueError):
        # 3D
        spline_curve.control_points = np.zeros((expected, 3, 3))
    if is_hermite_spline(spline_curve):
        with pytest.raises(ValueError):
            # 1D
            spline_curve.tangents = np.arange(expected)
        with pytest.raises(ValueError):
            # 3D
            spline_curve.tangents = np.zeros((expected, 3, 3))

    for i in range(expected - 2, expected + 2):
        control_points = np.arange(i)[:, np.newaxis]
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


def test_closed_splines(initialised_closed_spline_curve, derivative, xp):
    spline = initialised_closed_spline_curve
    assert xp.allclose(
        spline(0, derivative=derivative),
        spline(spline.M, derivative=derivative),
        equal_nan=True,
    )


def test_draw(closed, xp, device):
    spline = splinebox.spline_curves.Spline(M=4, basis_function=splinebox.basis_functions.B1(), closed=closed)
    spline.knots = xp.asarray([[1, 1], [1, 2], [2, 2], [2, 1]], dtype=xp.float64)

    x = xp.linspace(0, 3, 16, dtype=xp.float64)
    y = xp.linspace(0, 3, 16, dtype=xp.float64)

    # You can only draw closed splines
    if not closed:
        with pytest.raises(RuntimeError):
            spline.draw(x, y)
    else:
        expected = xp.zeros((len(x), len(y)))
        if array_api_compat.is_jax_namespace(xp):
            expected = expected.at[6:10, 6:10].set(1)
            expected = expected.at[5, 5:11].set(0.5)
            expected = expected.at[10, 5:11].set(0.5)
            expected = expected.at[5:11, 5].set(0.5)
            expected = expected.at[5:11, 10].set(0.5)
        else:
            expected[6:10, 6:10] = 1
            expected[5, 5:11] = 0.5
            expected[10, 5:11] = 0.5
            expected[5:11, 5] = 0.5
            expected[5:11, 10] = 0.5

        output = spline.draw(x, y)

        assert xp.allclose(output, expected)

        # You can only draw splines in 2D
        spline = splinebox.spline_curves.Spline(M=4, basis_function=splinebox.basis_functions.B1(), closed=closed)
        spline.knots = xp.asarray([[1, 1, 0.5], [1, 2, 2], [2, 2, 1], [2, 1, 0]])
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


def test_arc_length(xp, device):
    # Create circular spline with radius sqrt(2)
    M = 10
    basis_function = splinebox.basis_functions.Exponential(M)
    spline = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=True)
    phi = xp.linspace(0, 2 * np.pi, M + 1, device=device)[:-1]
    knots = xp.stack([xp.cos(phi), xp.sin(phi)], axis=-1)
    spline.knots = knots

    ts = xp.linspace(0, M, 100, device=device)

    arc_lengths = [spline.arc_length(t.item()) for t in ts]
    arc_lengths = xp.asarray(arc_lengths, device=device)

    expected = xp.linspace(0, 2 * xp.pi, 100, device=device)
    assert xp.allclose(arc_lengths, expected)

    # Check that it is working with a vector for stop
    permutation = np.random.permutation(len(ts))
    permutation = xp.asarray(permutation, device=device)
    arc_lengths = spline.arc_length(ts[permutation])
    assert xp.allclose(arc_lengths, expected[permutation])

    # Check that is works with vectors for start and stop
    arc_lengths = spline.arc_length(ts[permutation], xp.zeros(len(ts), device=device))
    assert xp.allclose(arc_lengths, expected[permutation])


def test_arc_length_to_parameter(xp, device):
    # Create circular spline with radius sqrt(2)
    M = 100
    basis_function = splinebox.basis_functions.Exponential(M)
    spline = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=True)
    knots = []
    for t in xp.linspace(0, 2 * xp.pi, M + 1, device=device)[:-1]:
        knots.append([xp.cos(t), xp.sin(t)])
    knots = xp.asarray(knots, device=device)
    spline.knots = knots

    ls = xp.linspace(0, 2 * xp.pi, 15, device=device)
    expected = xp.linspace(0, M, 15, device=device)  # ls / 2 / xp.pi * M
    atol = 1e-5

    results = spline.arc_length_to_parameter(ls, atol=atol)

    assert xp.allclose(results, expected, atol=1e-3, rtol=0)

    # Check that it works with an array of integer inputs
    ls = xp.arange(0, int(2 * xp.pi), dtype=int, device=device)
    expected = ls / 2 / xp.pi * M
    results = spline.arc_length_to_parameter(ls, atol=atol)
    assert xp.allclose(results, expected, atol=1e-3, rtol=0)
    results = spline.arc_length_to_parameter(ls, atol=atol)

    # Create a sawtooth spline
    M = 7
    basis_function = splinebox.basis_functions.B1()
    spline = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=False)
    spline.knots = xp.asarray([[0, 0], [1, 1], [1, 0], [2, 1], [2, 0], [3, 1], [3, 0]], device=device)

    rising_length = math.sqrt(2)
    tooth_length = rising_length + 1
    total_length = 3 * tooth_length
    ls = xp.linspace(0, total_length, 15, device=device)

    def to_param(length):
        n_tooth = length // tooth_length
        residual = length % tooth_length
        partial_tooth_param = residual / rising_length if residual < rising_length else 1 + residual - rising_length
        # there are two knots per tooth
        return n_tooth * 2 + partial_tooth_param

    expected = xp.asarray(list(map(to_param, ls)), device=device)
    atol = 1e-4

    results = spline.arc_length_to_parameter(ls, atol=atol)

    assert xp.allclose(results, expected, atol=atol)

    # Check that is works for a single value
    expected = to_param(ls[10])
    result = spline.arc_length_to_parameter(ls[10], atol=atol)
    assert xp.isclose(result, expected, atol=atol)

    # Check that length smaller than 0 and larger than the total length of the spline raise and error
    with pytest.raises(ValueError):
        spline.arc_length_to_parameter(-0.1)
    with pytest.raises(ValueError):
        spline.arc_length_to_parameter(total_length + 0.01)


def test_translate(initialised_spline_curve, translation_vector, xp, dtype, device):
    spline = initialised_spline_curve
    spline_copy = spline.copy()
    spline_copy.translate(translation_vector)
    t_dtype = xp.asarray(1.0).dtype if dtype is int else dtype
    t = (
        xp.linspace(0, spline.M, 100, dtype=t_dtype, device=device)
        if spline.closed
        else xp.linspace(0, spline.M - 1, 100, dtype=t_dtype, device=device)
    )
    expected = spline(t) + translation_vector
    atol = 1e-5 if expected.dtype == xp.float32 else 1e-8
    assert xp.allclose(spline_copy(t), expected, atol=atol)


def test_rotate(initialised_spline_curve, rotation_matrix, is_hermite_spline, xp, device):
    spline = initialised_spline_curve
    dtype = (
        xp.asarray(1.0).dtype
        if xp.isdtype(spline.control_points.dtype, kind="integral")
        else spline.control_points.dtype
    )
    if spline.ndim == 1:
        with pytest.raises(RuntimeError):
            spline.rotate(rotation_matrix)
    else:
        spline_copy = spline.copy()
        spline_copy.rotate(rotation_matrix, centred=False)
        t = (
            xp.linspace(0, spline.M, 100, dtype=dtype, device=device)
            if spline.closed
            else xp.linspace(0, spline.M - 1, 100, dtype=dtype, device=device)
        )
        vals = spline(t)
        expected = (rotation_matrix @ vals.T).T
        atol = 1e-4 if expected.dtype == xp.float32 else 1e-8
        assert xp.allclose(spline_copy(t), expected, atol=atol)

        spline_copy = spline.copy()
        spline_copy.rotate(rotation_matrix, centred=True)
        t = (
            xp.linspace(0, spline.M, 100, dtype=dtype, device=device)
            if spline.closed
            else xp.linspace(0, spline.M - 1, 100, dtype=dtype, device=device)
        )
        vals = spline(t)
        # torch mean only works for floating point arrays
        centring_vector = xp.mean(xp.asarray(spline.control_points, dtype=dtype), axis=0)
        vals = vals - centring_vector
        expected = (rotation_matrix @ vals.T).T
        expected = expected + centring_vector
        atol = 1e-3 if expected.dtype == xp.float32 else 1e-8
        assert xp.allclose(spline_copy(t), expected, atol=atol)


def test_fit(spline_curve, points, is_hermite_spline, xp, device):
    # Bool indicating whether spline_curve is a hermite spline or not
    hermite = is_hermite_spline(spline_curve)

    n_control_points = spline_curve.M + 2 * spline_curve.pad
    if hermite and len(points) < 2 * n_control_points or len(points) < n_control_points:
        # The problem is underdetermined
        with pytest.raises(RuntimeError):
            spline_curve.fit(points)
    else:
        spline_curve.fit(points)

        assert array_api_compat.array_namespace(spline_curve.control_points) is xp
        assert array_api_compat.device(spline_curve.control_points) == device

        # To compare to the reference we use numpy
        if array_api_compat.is_jax_namespace(xp):
            import jax

            cpu_device = jax.devices("cpu")[0]
        else:
            cpu_device = "cpu"
        points = np.array(array_api_compat.to_device(points, cpu_device))
        spline_curve.control_points = np.array(array_api_compat.to_device(spline_curve.control_points, cpu_device))
        if hermite:
            spline_curve.tangents = np.array(array_api_compat.to_device(spline_curve.tangents, cpu_device))

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
                spline_curve.control_points[:, i] = x[:half]
                spline_curve.tangents[:, i] = x[half:]
            else:
                spline_curve.control_points[:, i] = x
            spline_vals = spline_curve(t)
            loss = np.linalg.norm(points[:, i] - spline_vals[:, i])
            return loss

        # prepare the initial value for the minimization
        x0 = np.concatenate([control_points_fit, tangents_fit], axis=0) if hermite else control_points_fit

        # Minimize for each codomain dimension separately
        x = []
        for i in range(x0.shape[1]):
            x.append(scipy.optimize.minimize(difference_func, x0=x0[:, i], args=(i,)).x)

        # Turn minimization result into control_points and tangents for the spline
        x = np.stack(x, axis=-1)
        if hermite:
            control_points_expected = x[:half]
            tangents_expected = x[half:]
        else:
            control_points_expected = x

        assert np.allclose(control_points_expected, control_points_fit)
        if hermite:
            assert np.allclose(tangents_expected, tangents_fit)


def test_fit_clamped(open_spline_curve, points, is_hermite_spline, xp, device):
    spline = open_spline_curve

    # Bool indicating whether spline is a hermite spline or not
    hermite = is_hermite_spline(spline)

    n_control_points = spline.M + 2 * spline.pad
    if hermite and len(points) < 2 * n_control_points or len(points) < n_control_points:
        # The problem is underdetermined
        with pytest.raises(RuntimeError):
            spline.fit(points, "clamped")

    elif spline.basis_function == splinebox.B1():
        # B1 is not differentiable at the knots
        with pytest.raises(RuntimeError):
            spline.fit(points, "clamped")
    else:
        spline.fit(points, "clamped")

        # Check that the boundary conditions are fulfilled
        assert xp.allclose(spline(0, derivative=1), xp.asarray([0], dtype=points.dtype, device=device))
        assert xp.allclose(spline(spline.M - 1, derivative=1), xp.asarray([0], dtype=points.dtype, device=device))


def test_fit_natural(open_spline_curve, points, is_hermite_spline, xp, device):
    spline = open_spline_curve

    # Bool indicating whether spline is a hermite spline or not
    hermite = is_hermite_spline(spline)

    n_control_points = spline.M + 2 * spline.pad
    if hermite and len(points) < 2 * n_control_points or len(points) < n_control_points:
        # The problem is underdetermined
        with pytest.raises(RuntimeError):
            spline.fit(points, "natrual")

    elif spline.basis_function in (
        splinebox.B1(),
        splinebox.CatmullRom(),
        splinebox.CubicHermite(),
        splinebox.ExponentialHermite(spline.M),
    ):
        # Not twice differentiable at the knots
        with pytest.raises(RuntimeError):
            spline.fit(points, "natural")
    else:
        spline.fit(points, "natural")

        # Check that the boundary conditions are fulfilled
        assert xp.allclose(spline(0, derivative=2), xp.asarray([0], dtype=points.dtype, device=device))
        assert xp.allclose(spline(spline.M - 1, derivative=2), xp.asarray([0], dtype=points.dtype, device=device))


def test_knots(spline_curve, knot_gen, is_hermite_spline, request, xp, dtype, device):
    if isinstance(spline_curve.basis_function, splinebox.basis_functions.B2):
        # The filter_symmetric and filter_periodic are not implemented for B2
        request.node.add_marker(pytest.mark.xfail)

    knots = knot_gen(spline_curve.M)
    pad = math.ceil(spline_curve.basis_function.support / 2) - 1
    if spline_curve.padding_function is None and not spline_curve.closed:
        knots = splinebox.spline_curves.padding_function(knots, pad)
    spline_curve.knots = knots

    if is_hermite_spline(spline_curve):
        # Use finite differences as tangents
        if spline_curve.closed:
            if knots.ndim == 1:
                diff = xp.diff(knots, axis=0, append=knots[0])
            else:
                diff = xp.diff(knots, axis=0, append=knots[0][np.newaxis, :])
            tangents = (diff + xp.roll(diff, 1)) / 2
        else:
            diff = xp.diff(knots, axis=0)
            pad = math.ceil(spline_curve.basis_function.support / 2) - 1
            if diff.ndim == 1:
                diff = diff[:, xp.newaxis]
            padding = xp.tile(xp.zeros(diff.shape[1], dtype=diff.dtype, device=device), (pad + 1, 1))
            diff = xp.concat((padding, diff, padding), axis=0)
            tangents = (diff[:-1] + diff[1:]) / 2
        spline_curve.tangents = xp.astype(tangents, dtype)

    if dtype is int:
        # For comparison the knots have to be of the same type as the returned values.
        # For int the default floating point type is returned.
        knots = xp.astype(knots, xp.asarray(1.0).dtype)

    # This assert makes sense because the knots are not saved in
    # the spline class but are converted to coefficients which are saved.
    atol = 1e-5 if knots.dtype == xp.float32 else 1e-8
    if spline_curve.padding_function is None and not spline_curve.closed and pad != 0:
        # We only compare the actual knots because padded knots will not
        # be the same because of the filtering to find the control points.
        assert xp.allclose(knots[pad:-pad], spline_curve.knots[pad:-pad], atol=atol)
    else:
        assert xp.allclose(knots, spline_curve.knots, atol=atol)


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


def test_centroid(spline_curve, coeff_gen, is_hermite_spline, xp, dtype):
    support = spline_curve.basis_function.support
    closed = spline_curve.closed
    M = spline_curve.M

    control_points = coeff_gen(M, support, closed)
    spline_curve.control_points = control_points
    if is_hermite_spline(spline_curve):
        tangents = coeff_gen(spline_curve.M, support, closed)
        spline_curve.tangents = tangents

    output_dtype = xp.asarray(1.0).dtype if dtype is int else dtype
    expected = xp.mean(control_points, axis=0, dtype=output_dtype)
    assert xp.allclose(spline_curve._control_points_centroid(), expected)


def test_scale(initialised_spline_curve, is_hermite_spline, xp, device):
    spline = initialised_spline_curve
    spline_copy = spline.copy()

    factor = 1.5

    centroid = spline_copy._control_points_centroid()
    spline_copy.translate(-centroid)
    spline_copy.control_points = spline_copy.control_points * factor
    spline_copy.translate(centroid)

    if is_hermite_spline(spline_copy):
        spline_copy.tangents = spline_copy.tangents * factor

    t = (
        xp.linspace(0, spline.M, 100, device=device)
        if spline.closed
        else xp.linspace(0, spline.M - 1, 100, device=device)
    )

    spline.scale(factor)

    assert xp.allclose(spline(t), spline_copy(t))


def test_curvilinear_reparametrisation_energy(as_xp_array, xp, device, request):
    if hasattr(device, "type") and device.type in ("cuda", "xpu", "mps"):
        # Currently not working for torch tensors on gpu
        request.node.add_marker(pytest.mark.xfail)

    M = 4
    spline = splinebox.spline_curves.Spline(M, splinebox.B1())

    spline.knots = as_xp_array([[1, 0], [2, 0], [3, 0], [4, 0]])

    # Because of the spacing of the knots the derivative should be constant so we can just compute one value in the middle of the spline.
    derivative_val = spline((spline.M - 1) / 2, derivative=1)

    # To compute the expected value we exploit the constant derivative.
    # We can replace the integration in the definition of the curvilinear reparametrisation energy with a multiplication because the derivative is constant.
    length = spline.arc_length()
    c = (length / spline.M) ** 2
    e_curv = (xp.linalg.vector_norm(derivative_val) ** 2 - c) ** 2
    expected_val = e_curv * (spline.M - 1) / length**4

    val = spline.curvilinear_reparametrisation_energy()

    assert xp.isclose(val, expected_val)


def test_curvilinear_reparametrisation_energy_user_c(as_xp_array, xp, device, request):
    if hasattr(device, "type") and device.type in ("cuda", "xpu", "mps"):
        # Currently not working for torch tensors on gpu
        request.node.add_marker(pytest.mark.xfail)

    M = 4
    spline = splinebox.spline_curves.Spline(M, splinebox.B1())

    spline.knots = as_xp_array([[1, 0], [2, 0], [3, 0], [4, 0]])

    # Because of the spacing of the knots the derivative should be constant so we can just compute one value in the middle of the spline.
    derivative_val = spline((spline.M - 1) / 2, derivative=1)

    # For a user provided c the unnormalised integral (equation 25) is returned.
    c = 1.5
    expected_val = (xp.linalg.vector_norm(derivative_val) ** 2 - c) ** 2 * (spline.M - 1)

    val = spline.curvilinear_reparametrisation_energy(c=c)

    assert np.isclose(val, expected_val)


def test_curvilinear_reparametrisation_energy_translation(
    initialised_spline_curve, translation_vector, xp, device, request
):
    """
    Test if the curvilinear reparametrisation energy is invariant to translation.
    """
    if hasattr(device, "type") and device.type in ("cuda", "xpu", "mps"):
        # Currently not for torch tensors on gpu
        request.node.add_marker(pytest.mark.xfail)

    atol = 1e-3
    spline = initialised_spline_curve
    expected = spline.curvilinear_reparametrisation_energy(atol=atol)
    spline.translate(translation_vector)
    val = spline.curvilinear_reparametrisation_energy(atol=atol)
    assert xp.isclose(val, expected, atol=atol)


def test_curvilinear_reparametrisation_energy_scale_invariance(initialised_spline_curve, xp, device, request):
    """
    The curvilinear reparametrisation energy should not change when the entire spline is scaled.
    """
    if hasattr(device, "type") and device.type in ("cuda", "xpu", "mps"):
        # Currently not for torch tensors on gpu
        request.node.add_marker(pytest.mark.xfail)

    spline = initialised_spline_curve
    expected = spline.curvilinear_reparametrisation_energy()
    spline.scale(0.1)
    val = spline.curvilinear_reparametrisation_energy()
    assert xp.isclose(val, expected)


def test_curvature(as_xp_array, xp, device):
    # Create a circular spline
    M = 4
    spline = splinebox.spline_curves.Spline(M=M, basis_function=splinebox.basis_functions.Exponential(M), closed=True)
    t = xp.linspace(0, M, 100, device=device)
    for r in (1, 1.5, 2):
        # 2D
        spline.knots = as_xp_array([[0, r], [r, 0], [0, -r], [-r, 0]])
        curvature = spline.curvature(t)
        assert xp.allclose(curvature, xp.asarray(1 / r, dtype=curvature.dtype))

        # If we reverse the direction, the curvature should change sign
        spline.knots = as_xp_array([[0, r], [-r, 0], [0, -r], [r, 0]])
        curvature = spline.curvature(t)
        assert xp.allclose(curvature, xp.asarray(-1 / r, dtype=curvature.dtype))

        # 3D
        spline.knots = as_xp_array([[0, 0, r], [0, r, 0], [0, 0, -r], [0, -r, 0]])
        curvature = spline.curvature(t)
        assert xp.allclose(curvature, xp.asarray(1 / r, dtype=curvature.dtype))

    # 1D
    spline = splinebox.spline_curves.Spline(M=M, basis_function=splinebox.basis_functions.B3(), closed=False)
    spline.knots = as_xp_array([[1], [5], [4], [2]])
    expected = spline(t, derivative=2) / (1 + spline(t, derivative=1) ** 2) ** (3 / 2)
    expected = xp.squeeze(expected, axis=1)
    result = spline.curvature(t)
    assert xp.allclose(result, expected)


def test_normal_2D(as_xp_array, xp, device):
    # Create a circular spline
    M = 4
    spline = splinebox.spline_curves.Spline(M=M, basis_function=splinebox.basis_functions.Exponential(M), closed=True)
    t = xp.linspace(0, M, 100, device=device)

    theta = xp.linspace(0, 2 * xp.pi, 100, device=device)
    expected = xp.stack([xp.sin(theta), xp.cos(theta)], axis=-1)
    atol = 1e-5 if expected.dtype == xp.float32 else 1e-8
    for r in (1, 1.5, 2):
        spline.knots = as_xp_array([[0, r], [r, 0], [0, -r], [-r, 0]])
        normals = spline.normal(t)
        assert xp.allclose(normals, expected, atol=atol)

        # Reversing the direction change the direction of the normal vector
        spline.knots = as_xp_array([[0, r], [-r, 0], [0, -r], [r, 0]])
        normals = spline.normal(xp.flip(t))
        assert xp.allclose(normals, -1 * expected, atol=atol)

        # Check that they are normal vectors
        assert xp.allclose(xp.linalg.vector_norm(normals, axis=1), xp.ones(len(t), device=device), atol=atol)


def test_normal_3D(as_xp_array, xp, device):
    M = 4
    spline = splinebox.spline_curves.Spline(M=M, basis_function=splinebox.basis_functions.B3(), closed=False)
    t = xp.linspace(0, M - 1, 100, device=device)

    # This is a straight line
    spline.control_points = xp.stack(
        [xp.arange(M + 2, device=device), xp.arange(M + 2, device=device), xp.ones(M + 2, device=device)]
    ).T
    normals = spline.normal(t, frame="bishop", initial_vector=as_xp_array([0.0, 0.0, 1.0]))
    atol = 1e-5 if normals.dtype == xp.float32 else 1e-8
    # check that the normals don't change
    assert xp.isclose(xp.sum(xp.diff(normals, axis=0)), xp.asarray(0, dtype=normals.dtype), atol=atol)

    spline.knots = as_xp_array([[1, 0, 0], [0, 1, 1], [-1, 0, 2], [0, -1, 3]])
    t = t[1:-1]
    normals = spline.normal(t, frame="frenet")
    tangents = spline(t, derivative=1)
    # Check that both normals are orthogonal t the tangents (dot product)
    dot = xp.sum(normals * tangents[:, xp.newaxis], axis=-1)
    assert xp.allclose(dot, xp.asarray(0, dtype=dot.dtype), atol=atol)
    # Check that the normals are orthogonal to each other
    dot = xp.sum(normals[:, 0] * normals[:, 1])
    assert xp.isclose(dot, xp.asarray(0, dtype=dot.dtype), atol=atol)


def test_repr(initialised_spline_curve, is_hermite_spline, xp, dtype):
    spline = initialised_spline_curve

    if dtype is not int:
        # The rounding is necessary because the precision is limited in the string representation
        xp.set_printoptions(precision=16)
        spline.control_points = xp.round(spline.control_points, decimals=1)
        if is_hermite_spline(spline):
            spline.tangents = xp.round(spline.tangents, decimals=1)
    if array_api_compat.is_jax_namespace(xp):
        pass
    assert eval(repr(spline)) == spline


def test_eq_spline_basis_function(M, non_hermite_basis_function_combinations, closed, coeff_gen):
    basis_function, other_basis_function = non_hermite_basis_function_combinations
    spline = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=closed)
    other = splinebox.spline_curves.Spline(M=M, basis_function=other_basis_function, closed=closed)

    # Before initialising the spline with control points
    if basis_function == other_basis_function:
        assert spline == other
    else:
        assert spline != other

    # Set identical control points
    if basis_function.support == other_basis_function.support:
        spline.control_points = coeff_gen(M, basis_function.support, closed)
        other.control_points = spline.control_points
        if basis_function == other_basis_function:
            assert spline == other
        else:
            assert spline != other


def test_eq_hermite_spline_basis_function(M, hermite_basis_function_combinations, closed, coeff_gen):
    basis_function, other_basis_function = hermite_basis_function_combinations
    spline = splinebox.spline_curves.HermiteSpline(M=M, basis_function=basis_function, closed=closed)
    other = splinebox.spline_curves.HermiteSpline(M=M, basis_function=other_basis_function, closed=closed)

    # Before initialising the spline with control points
    if basis_function == other_basis_function:
        assert spline == other
    else:
        assert spline != other

    # Set identical control points and tangents
    if basis_function.support == other_basis_function.support:
        spline.control_points = coeff_gen(M, basis_function.support, closed)
        spline.tangents = coeff_gen(M, basis_function.support, closed)
        other.control_points = spline.control_points
        other.tangents = spline.tangents
        if basis_function == other_basis_function:
            assert spline == other
        else:
            assert spline != other


def test_eq_spline_M():
    basis_function = splinebox.basis_functions.B1()
    closed = False

    spline = splinebox.spline_curves.Spline(M=10, basis_function=basis_function, closed=closed)
    other = splinebox.spline_curves.Spline(M=9, basis_function=basis_function, closed=closed)
    assert spline != other

    other = splinebox.spline_curves.Spline(M=10, basis_function=basis_function, closed=closed)
    assert spline == other


def test_eq_spline_closed():
    basis_function = splinebox.basis_functions.B3()
    M = 7

    spline = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=True)
    other = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=False)
    assert spline != other

    other = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=True)
    assert spline == other

    # Identical control points for open and closed spline
    spline = splinebox.spline_curves.Spline(M=M, basis_function=splinebox.basis_functions.B1(), closed=True)
    other = splinebox.spline_curves.Spline(M=M, basis_function=spline.basis_function, closed=False)
    spline.control_points = np.arange(M)[:, np.newaxis]
    other.control_points = spline.control_points
    assert spline != other


def test_eq_spline_control_points(coeff_gen):
    M = 7
    basis_function = splinebox.basis_functions.B3()
    closed = False

    spline = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=closed)
    other = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=closed)

    # Only one spline with control points
    spline.control_points = coeff_gen(M, basis_function.support, closed)
    assert spline != other

    # Different control points
    other.control_points = coeff_gen(M, basis_function.support, closed)
    assert spline != other

    # Same control points
    other.control_points = spline.control_points
    assert spline == other


def test_eq_hermite_spline_tangents(coeff_gen):
    M = 7
    basis_function = splinebox.basis_functions.CubicHermite()
    closed = False

    spline = splinebox.spline_curves.HermiteSpline(M=M, basis_function=basis_function, closed=closed)
    other = splinebox.spline_curves.HermiteSpline(M=M, basis_function=basis_function, closed=closed)

    # Only one spline with tangents
    spline.tangents = coeff_gen(M, basis_function.support, closed)
    assert spline != other

    # Different tangents
    other.tangents = coeff_gen(M, basis_function.support, closed)
    assert spline != other

    # Same tangents
    other.tangents = spline.tangents
    assert spline == other

    # Same tangents different control points
    spline.control_points = coeff_gen(M, basis_function.support, closed)
    other.control_points = coeff_gen(M, basis_function.support, closed)
    assert spline != other

    # Same control points different tangents
    other.control_points = spline.control_points
    other.tangents = coeff_gen(M, basis_function.support, closed)
    assert spline != other


def test_str(initialised_spline_curve):
    assert isinstance(str(initialised_spline_curve), str)
    assert str(initialised_spline_curve.M) in str(initialised_spline_curve)
    assert str(initialised_spline_curve.basis_function) in str(initialised_spline_curve)
    if initialised_spline_curve.closed:
        assert "closed" in str(initialised_spline_curve)


def test_to_json_and_from_json(initialised_spline_curve, tmpdir, is_hermite_spline, xp, dtype, device):
    spline = initialised_spline_curve
    path = tmpdir / "spline.json"
    spline.to_json(path)
    if is_hermite_spline(initialised_spline_curve):
        loaded_spline_curve = splinebox.spline_curves.HermiteSpline.from_json(path, xp=xp, dtype=dtype, device=device)
    else:
        loaded_spline_curve = splinebox.spline_curves.Spline.from_json(path, xp=xp, dtype=dtype, device=device)
    assert spline == loaded_spline_curve


def test_from_json(tmpdir):
    # Check float M
    data = {"version": 1, "M": 4.5, "basis_function": str(splinebox.basis_functions.B3()), "closed": True}
    path = tmpdir / "spline.json"
    with open(path, "w") as f:
        json.dump(data, f)
    with pytest.raises(ValueError):
        splinebox.spline_curves.Spline.from_json(path)

    # Check wrong version
    data = {
        "version": "none_existent_version",
        "M": 4.5,
        "basis_function": str(splinebox.basis_functions.B3()),
        "closed": True,
    }
    path = tmpdir / "spline.json"
    with open(path, "w") as f:
        json.dump(data, f)
    with pytest.raises(ValueError):
        splinebox.spline_curves.Spline.from_json(path)

    # Check non existent basis function name
    data = {"version": 1, "M": 4, "basis_function": "FakeBasisFunction", "closed": True}
    path = tmpdir / "spline.json"
    with open(path, "w") as f:
        json.dump(data, f)
    with pytest.raises(ValueError):
        splinebox.spline_curves.Spline.from_json(path)

    # Check wrong entry for "closed"
    data = {"version": 1, "M": 4, "basis_function": str(splinebox.basis_functions.B3()), "closed": "random_string"}
    path = tmpdir / "spline.json"
    with open(path, "w") as f:
        json.dump(data, f)
    with pytest.raises(ValueError):
        splinebox.spline_curves.Spline.from_json(path)


def test_saving_and_loading_of_multiple_splines(
    is_hermite_basis_function, tmpdir, coeff_gen, M, closed, xp, dtype, device
):
    splines = []
    for basis_function_name in splinebox.basis_functions.inventory():
        basis_function = splinebox.basis_functions.basis_function_from_name(basis_function_name, M=M)
        if is_hermite_basis_function(basis_function):
            spline = splinebox.spline_curves.HermiteSpline(M=M, basis_function=basis_function, closed=closed)
            spline.control_points = coeff_gen(spline.M, spline.basis_function.support, spline.closed)
            spline.tangents = coeff_gen(spline.M, spline.basis_function.support, spline.closed)
        else:
            spline = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=closed)
            spline.control_points = coeff_gen(spline.M, spline.basis_function.support, spline.closed)
        splines.append(spline)

    path = tmpdir / "splines.json"
    splinebox.spline_curves.splines_to_json(path, splines)

    loaded_splines = splinebox.spline_curves.splines_from_json(path, xp=xp, dtype=dtype, device=device)

    for spline, loaded_spline in zip(splines, loaded_splines):
        assert spline == loaded_spline


def test_distance(initialised_spline_curve, xp, device, request):
    spline = initialised_spline_curve

    if not array_api_compat.is_numpy_namespace(xp) and spline.ndim != 1:
        # Currently only implemented for numpy
        request.node.add_marker(pytest.mark.xfail)

    if spline.ndim == 1:
        with pytest.raises(RuntimeError):
            spline.distance(xp.asarray((1.0,), device=device))
        return

    point = xp.asarray([1] * spline.control_points.shape[1], device=device)
    distance = spline.distance(point)

    # Check that a set of equaly spaced points on the spline are at least as far away as the distance
    t = xp.linspace(0, spline.M if spline.closed else spline.M - 1, spline.M * 10, device=device)
    points_on_spline = spline(t)
    distances = xp.linalg.vector_norm(points_on_spline - point[xp.newaxis], axis=-1)
    assert xp.all(distances >= distance)

    # Check that the returned t corresponds to the correct distance
    distance, t = spline.distance(point, return_t=True)
    point_on_spline = spline(t)
    assert xp.isclose(distance, xp.linalg.vector_norm(point - point_on_spline))


def test_distance_multiple_points(initialised_spline_curve, as_xp_array, xp, request):
    spline = initialised_spline_curve

    if not array_api_compat.is_numpy_namespace(xp) and spline.ndim != 1:
        # Currently only implemented for numpy
        request.node.add_marker(pytest.mark.xfail)

    if spline.ndim == 1:
        return

    np.random.seed(45)
    points = np.random.rand(10, spline.control_points.shape[1]) * 10
    points = as_xp_array(points)

    distances, t = spline.distance(points, return_t=True)

    points_on_spline = spline(t)
    assert xp.allclose(distances, xp.linalg.vector_norm(points - points_on_spline, axis=1))


def test_moving_frame(initialised_spline_curve, xp, device, dtype):
    spline = initialised_spline_curve
    t = xp.linspace(0, spline.M if spline.closed else spline.M - 1, spline.M * 10, dtype=dtype, device=device)

    # The dtype actually used. If dtype is int,
    # the default float precision, otherwise the just dtype.
    effective_dtype = xp.asarray(1.0, device=device).dtype if dtype is int else dtype

    atol = 1e-5 if effective_dtype == xp.float32 else 1e-8

    # pytorch requires a tensor for comparison
    zero = xp.asarray(0.0, dtype=effective_dtype, device=device)

    if spline.ndim != 3:
        with pytest.raises(RuntimeError):
            frame = spline.moving_frame(t)
    else:
        ########
        # Frenet
        ########
        if isinstance(spline.basis_function, splinebox.basis_functions.B1):
            # The Frenet frame is not defined for splines with straight segments
            with pytest.raises(RuntimeError):
                frame = spline.moving_frame(t, method="frenet")
        else:
            # The second derivative of CatmullRom splines at the knots is not defined so we
            # remove all of the integers from t
            t_frenet = t[(t % 1) != 0] if isinstance(spline.basis_function, splinebox.basis_functions.CatmullRom) else t
            # When dtype is int no t values are left
            if len(t_frenet) == 0:
                return

            frame = spline.moving_frame(t_frenet, method="frenet")

            # Check that you can pass a float value
            assert xp.allclose(frame[5], spline.moving_frame(t_frenet[5], method="frenet"), atol=atol)

            # Check that they are unit vectors
            assert xp.allclose(
                xp.linalg.vector_norm(frame, axis=-1), xp.asarray(1.0, dtype=frame.dtype, device=device), atol=atol
            )

            # Check if the frame forms an orthogonal basis using the dot product
            assert xp.allclose(xp.sum(frame[:, 0] * frame[:, 1], axis=-1), zero, atol=atol)
            assert xp.allclose(xp.sum(frame[:, 0] * frame[:, 2], axis=-1), zero, atol=atol)
            assert xp.allclose(xp.sum(frame[:, 1] * frame[:, 2], axis=-1), zero, atol=atol)

            # Check that T is parallel to the first derivative
            assert xp.allclose(xp.cross(frame[:, 0], spline(t_frenet, derivative=1)), zero, atol=atol)

            # N = dT/ds and T = dr/ds so N should be parallel to d^2r/ds^2 - proj_T(d^2/ds^2)
            # Removing the projection on T is equivalent to removing the part of the
            # derivative that changes the length of the first derivative.
            # This part should not be present in N since it is the derivative of the
            # unit vector T, which has constant length.
            deriv2 = spline(t_frenet, derivative=2)
            # use Gram Schmidt to remove the scaling component
            deriv2 -= xp.sum(deriv2 * frame[:, 0], axis=-1)[:, xp.newaxis] * frame[:, 0]
            assert xp.allclose(
                xp.cross(frame[:, 1], deriv2), zero, atol=atol * float(xp.max(xp.linalg.vector_norm(deriv2, axis=-1)))
            )

        ########
        # Bishop
        ########
        if isinstance(spline.basis_function, splinebox.basis_functions.B1):
            # The derivative of B1 splines at the knots is not defined so we
            # remove all of the integers from t
            t = t[(t % 1) != 0]
            # When dtype is int no t values are left
            if len(t) == 0:
                return

        with pytest.raises(ValueError):
            # Non-orthogonal initial vector
            frame = spline.moving_frame(t, method="bishop", initial_vector=xp.ones(3, dtype=dtype, device=device))

        tangent = spline(t[0].item(), derivative=1)
        initial_vector = xp.zeros(3, dtype=tangent.dtype, device=device)
        if array_api_compat.is_jax_namespace(xp):
            initial_vector = initial_vector.at[1].set(-tangent[2])
            initial_vector = initial_vector.at[2].set(tangent[1])
        else:
            initial_vector[1] = -tangent[2]
            initial_vector[2] = tangent[1]
        frame = spline.moving_frame(t, method="bishop", initial_vector=initial_vector)

        # Check that you can pass a float value
        assert xp.allclose(frame[5], spline.moving_frame(t[5], method="bishop", initial_vector=frame[5, 1]), atol=atol)

        # Check that they are unit vectors
        assert xp.allclose(
            xp.linalg.vector_norm(frame, axis=-1), xp.asarray(1.0, dtype=frame.dtype, device=device), atol=atol
        )

        # Check if the frame forms an orthogonal basis using the dot product
        # Since the Bishop frame is build iteratively, it accumulates more error.
        atol_bishop = 1e-3 if effective_dtype == xp.float32 else 1e-8
        assert xp.allclose(xp.sum(frame[:, 0] * frame[:, 1], axis=-1), zero, atol=atol_bishop)
        assert xp.allclose(xp.sum(frame[:, 0] * frame[:, 2], axis=-1), zero, atol=atol_bishop)
        assert xp.allclose(xp.sum(frame[:, 1] * frame[:, 2], axis=-1), zero, atol=atol_bishop)

        # Check that T is parallel to the first derivative
        assert xp.allclose(xp.cross(frame[:, 0], spline(t, derivative=1)), zero, atol=atol)


def test_moving_frame_raises(as_xp_array, xp, device):
    M = 6
    spline = splinebox.spline_curves.Spline(M=M, basis_function=splinebox.basis_functions.B3(), closed=False)
    t = xp.linspace(0, M - 1, 100, device=device)

    spline.knots = as_xp_array(np.random.rand(6, 3))

    # Wrong method
    with pytest.raises(ValueError):
        spline.moving_frame(t, method="madeup method")

    # Vanishing tangent at the end because of padding
    with pytest.raises(RuntimeError):
        spline.moving_frame(t, method="frenet")

    # Straight segment
    spline.control_points = xp.stack(
        [xp.arange(M + 2, device=device), xp.zeros(M + 2, device=device), xp.zeros(M + 2, device=device)], axis=-1
    )
    with pytest.raises(RuntimeError):
        spline.moving_frame(t, method="frenet")


def test_moving_frame_unsorted(frame, as_xp_array, xp):
    np.random.seed(499)

    spline = splinebox.spline_curves.Spline(M=7, basis_function=splinebox.basis_functions.B3(), closed=False)
    spline.knots = as_xp_array(np.random.rand(7, 3))

    t = as_xp_array(np.random.rand(100)) * (spline.M - 1)

    sort_indices = xp.argsort(t)
    sorted_t = t[sort_indices]

    initial_vector = xp.cross(spline(t[0], derivative=1), spline(sorted_t[0], derivative=1))
    if initial_vector.shape[0] == 1:
        initial_vector = xp.squeeze(initial_vector, axis=0)

    sorted_frame = spline.moving_frame(sorted_t, method=frame, initial_vector=initial_vector)
    unsorted_frame = spline.moving_frame(t, method=frame, initial_vector=initial_vector)

    assert xp.allclose(sorted_frame[xp.argsort(sort_indices)], unsorted_frame)


def test_mesh(initialised_spline_curve, radius, step_t, step_angle, mesh_type, cap_ends, frame, xp, request):
    spline = initialised_spline_curve

    if not array_api_compat.is_numpy_namespace(xp) and spline.ndim == 3:
        # Currently only implemented for numpy
        request.node.add_marker(pytest.mark.xfail)

    if spline.ndim != 3:
        # mesh is only implemented for splines in 3D
        with pytest.raises(RuntimeError):
            spline.mesh(
                radius=radius, step_t=step_t, step_angle=step_angle, mesh_type=mesh_type, cap_ends=cap_ends, frame=frame
            )
    elif radius is not None and frame == "frenet" and isinstance(spline.basis_function, splinebox.basis_functions.B1):
        # The Frenet frame is not defined because the second derivative of B1 is zero everywhere.
        with pytest.raises(RuntimeError):
            spline.mesh(
                radius=radius, step_t=step_t, step_angle=step_angle, mesh_type=mesh_type, cap_ends=cap_ends, frame=frame
            )
    else:
        points, connectivity = spline.mesh(
            radius=radius, step_t=step_t, step_angle=step_angle, mesh_type=mesh_type, cap_ends=cap_ends, frame=frame
        )

        # Check that all points are part of at least one element
        assert xp.all(xp.arange(len(points)) == xp.sort(xp.unique(connectivity)))

        if mesh_type == "surface" and (cap_ends is not None or spline.closed):
            # Compute the Euler characteristic to check that the mesh is closed

            # Number of verices
            V = len(points)
            # Number of faces
            F = len(connectivity) if connectivity.shape[1] > 2 else 0
            # Collect all in a set (no duplicates)
            edges = set()
            for i in range(len(connectivity)):
                edges.add(tuple(sorted([connectivity[i, 0], connectivity[i, 1]])))
                if radius is not None:
                    edges.add(tuple(sorted([connectivity[i, 0], connectivity[i, 2]])))
                    edges.add(tuple(sorted([connectivity[i, 1], connectivity[i, 2]])))
            # Number of edges
            E = len(edges)

            if radius is None and not spline.closed:
                assert V + F - E == 1
            elif spline.closed:
                # toroidal polyhedra have Euler characteristic 0
                assert V + F - E == 0
            elif cap_ends is not None:
                # spherical polyhedra have Euler characteristic 2
                assert V + F - E == 2


def test_mesh_cap_ends_false_alias_matches_none(as_xp_array, xp, request):
    if not array_api_compat.is_numpy_namespace(xp):
        # Currently only implemented for numpy
        request.node.add_marker(pytest.mark.xfail)

    spline = splinebox.spline_curves.Spline(M=4, basis_function=splinebox.basis_functions.B3(), closed=False)
    spline.knots = as_xp_array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.5, 0.0], [3.0, 0.5, 0.5]])

    points_none, connectivity_none = spline.mesh(radius=0.2, step_t=0.5, step_angle=30, cap_ends=None)
    with pytest.warns(DeprecationWarning, match="cap_ends=False"):
        points_false, connectivity_false = spline.mesh(radius=0.2, step_t=0.5, step_angle=30, cap_ends=False)

    assert xp.allclose(points_none, points_false)
    assert xp.array_equal(connectivity_none, connectivity_false)


def test_mesh_cap_ends_true_raises_value_error(as_xp_array, xp, request):
    if not array_api_compat.is_numpy_namespace(xp):
        # Currently only implemented for numpy
        request.node.add_marker(pytest.mark.xfail)

    spline = splinebox.spline_curves.Spline(M=4, basis_function=splinebox.basis_functions.B3(), closed=False)
    spline.knots = as_xp_array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.5, 0.0], [3.0, 0.5, 0.5]])

    with pytest.raises(ValueError, match="cap_ends='flat'"):
        spline.mesh(radius=0.2, step_t=0.5, step_angle=30, cap_ends=True)


def test_spherical_caps_add_more_geometry_than_flat_caps(as_xp_array, xp, request):
    if not array_api_compat.is_numpy_namespace(xp):
        # Currently only implemented for numpy
        request.node.add_marker(pytest.mark.xfail)

    spline = splinebox.spline_curves.Spline(M=4, basis_function=splinebox.basis_functions.B3(), closed=False)
    spline.knots = as_xp_array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.5, 0.0], [3.0, 0.5, 0.5]])

    flat_points, flat_connectivity = spline.mesh(radius=0.2, step_t=0.5, step_angle=30, cap_ends="flat")
    sphere_points, sphere_connectivity = spline.mesh(radius=0.2, step_t=0.5, step_angle=30, cap_ends="sphere")

    assert len(sphere_points) > len(flat_points)
    assert len(sphere_connectivity) > len(flat_connectivity)


def test_flat_caps_use_exact_open_spline_endpoints_when_step_t_does_not_divide_range(as_xp_array, xp, request):
    if not array_api_compat.is_numpy_namespace(xp):
        # Currently only implemented for numpy
        request.node.add_marker(pytest.mark.xfail)

    spline = splinebox.spline_curves.Spline(M=4, basis_function=splinebox.basis_functions.B3(), closed=False)
    spline.knots = as_xp_array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.5, 0.0], [3.0, 0.5, 0.5]])

    points, _ = spline.mesh(radius=0.2, step_t=0.4, step_angle=30, cap_ends="flat")

    assert xp.allclose(points[-2], spline(0))
    assert xp.allclose(points[-1], spline(spline.M - 1))


def test_protected_spline_attributes(spline_curve, coeff_gen, basis_function):
    # spline = splinebox.spline_curves.Spline(M=4, basis_function=splinebox.basis_functions.B1(), closed=True)
    spline = spline_curve

    # Test that all attributes can be changed before the control points are set
    spline.M = spline.M + 1
    spline.basis_function = basis_function
    spline.closed = not spline.closed

    # half_support and pad can never be changed
    with pytest.raises(RuntimeError):
        spline.half_support = spline.half_support + 1
    with pytest.raises(RuntimeError):
        spline.pad = spline.pad + 1

    spline.control_points = coeff_gen(spline.M, spline.basis_function.support, spline.closed)

    # Check that we cannot change M anymore
    with pytest.raises(RuntimeError):
        spline.M = spline.M + 2

    # Check that basis_function can only be changed to basis functions with the same support
    if basis_function.support != spline.basis_function.support:
        with pytest.raises(RuntimeError):
            spline.basis_function = basis_function
    else:
        spline.basis_function = basis_function

    # Check that closed cannot be changed anymore
    with pytest.raises(RuntimeError):
        spline.closed = not spline.closed

    # half_support and pad can be changed after setting the control points either
    with pytest.raises(RuntimeError):
        spline.half_support = spline.half_support + 1
    with pytest.raises(RuntimeError):
        spline.pad = spline.pad + 1


def test_single_value_input_basis_matrix(initialised_spline_curve, is_hermite_spline, single_value_t, xp, dtype):
    spline = initialised_spline_curve
    output = spline.basis_matrix(single_value_t)

    # For hermite spline two matrices are returned for the two parts of the basis function.
    # To not duplicate the tests, we iterate over the outpus.
    outputs = output if is_hermite_spline(spline) else [output]
    for output in outputs:
        if array_api_compat.is_numpy_namespace(xp):
            output = output.todense()
        assert array_api_compat.is_array_api_obj(output)
        assert array_api_compat.array_namespace(output) is xp
        expected_dtype = xp.asarray(1.0).dtype if dtype is int else dtype
        assert output.dtype == expected_dtype
        assert output.shape == (1, spline.control_points.shape[0])


def test_array_input_basis_matrix(initialised_spline_curve, is_hermite_spline, array_t, xp, dtype):
    spline = initialised_spline_curve
    output = spline.basis_matrix(array_t)

    # For hermite spline two matrices are returned for the two parts of the basis function.
    # To not duplicate the tests, we iterate over the outpus.
    outputs = output if is_hermite_spline(spline) else [output]
    for output in outputs:
        if array_api_compat.is_numpy_namespace(xp):
            output = output.todense()
        assert array_api_compat.is_array_api_obj(output)
        assert array_api_compat.array_namespace(output) is xp
        expected_dtype = xp.asarray(1.0).dtype if dtype is int else dtype
        assert output.dtype == expected_dtype
        assert output.shape == (len(array_t), spline.control_points.shape[0])


def test_single_value_input_call(initialised_spline_curve, single_value_t, xp, dtype):
    spline = initialised_spline_curve
    output = spline(single_value_t)
    assert array_api_compat.is_array_api_obj(output)
    assert array_api_compat.array_namespace(output) is xp
    expected_dtype = xp.asarray(1.0).dtype if dtype is int else dtype
    assert output.dtype == expected_dtype
    assert output.shape == (spline.ndim,)


def test_array_input_call(initialised_spline_curve, array_t, xp, dtype):
    spline = initialised_spline_curve
    output = spline(array_t)
    assert array_api_compat.is_array_api_obj(output)
    assert array_api_compat.array_namespace(output) is xp
    expected_dtype = xp.asarray(1.0).dtype if dtype is int else dtype
    assert output.dtype == expected_dtype
    assert output.shape == (len(array_t), spline.ndim)


def test_single_value_input_arc_length(initialised_spline_curve, single_value_t, xp, dtype):
    spline = initialised_spline_curve
    expected_dtype = xp.asarray(1.0).dtype if dtype is int else dtype

    arc_length = spline.arc_length(single_value_t)
    assert xp.isdtype(arc_length.dtype, expected_dtype)

    output = spline.arc_length_to_parameter(arc_length)
    assert xp.isdtype(output.dtype, expected_dtype)


def test_array_input_arc_length(initialised_spline_curve, array_t, dtype, xp):
    spline = initialised_spline_curve

    arc_lengths = spline.arc_length(array_t)
    assert array_api_compat.is_array_api_obj(arc_lengths)
    assert array_api_compat.array_namespace(arc_lengths) is xp
    expected_dtype = xp.asarray(1.0).dtype if dtype is int else dtype
    assert arc_lengths.dtype == expected_dtype
    assert arc_lengths.shape == (len(array_t),)

    output = spline.arc_length_to_parameter(arc_lengths, atol=0.1)
    assert array_api_compat.is_array_api_obj(output)
    assert array_api_compat.array_namespace(output) is xp
    expected_dtype = xp.asarray(1.0).dtype if dtype is int else dtype
    assert output.dtype == expected_dtype
    assert output.shape == (len(array_t),)


def test_single_value_input_curvature(initialised_twice_differentiable_spline_curve, single_value_t, xp, dtype):
    spline = initialised_twice_differentiable_spline_curve
    expected_dtype = xp.asarray(1.0).dtype if dtype is int else dtype

    curvature = spline.curvature(single_value_t)
    assert curvature.dtype == expected_dtype


def test_array_input_curvature(initialised_twice_differentiable_spline_curve, array_t, xp, dtype):
    spline = initialised_twice_differentiable_spline_curve

    curvatures = spline.curvature(array_t)
    assert array_api_compat.is_array_api_obj(curvatures)
    assert array_api_compat.array_namespace(curvatures) is xp
    expected_dtype = xp.asarray(1.0).dtype if dtype is int else dtype
    assert curvatures.dtype == expected_dtype
    assert curvatures.shape == (len(array_t),)


def test_single_value_input_dtheta(initialised_twice_differentiable_2D_spline_curve, single_value_t, xp, dtype):
    spline = initialised_twice_differentiable_2D_spline_curve
    dtheta = spline.dtheta(single_value_t)
    expected_dtype = xp.asarray(1.0).dtype if dtype is int else dtype
    assert dtheta.dtype == expected_dtype


def test_array_input_dtheta(initialised_twice_differentiable_2D_spline_curve, array_t, xp, dtype):
    spline = initialised_twice_differentiable_2D_spline_curve

    dthetas = spline.dtheta(array_t)
    assert array_api_compat.is_array_api_obj(dthetas)
    assert array_api_compat.array_namespace(dthetas) is xp
    expected_dtype = xp.asarray(1.0).dtype if dtype is int else dtype
    assert dthetas.dtype == expected_dtype
    assert dthetas.shape == (len(array_t),)


def test_single_value_input_normal_2D(initialised_2D_spline_curve, single_value_t, xp, dtype):
    spline = initialised_2D_spline_curve
    output = spline.normal(single_value_t, shift=1e-4)
    assert array_api_compat.is_array_api_obj(output)
    assert array_api_compat.array_namespace(output) is xp
    expected_dtype = xp.asarray(1.0).dtype if dtype is int else dtype
    assert output.dtype == expected_dtype
    assert output.shape == (2,)


def test_array_input_normal_2D(initialised_2D_spline_curve, array_t, xp, dtype):
    spline = initialised_2D_spline_curve
    output = spline.normal(array_t, shift=1e-4)
    assert array_api_compat.is_array_api_obj(output)
    assert array_api_compat.array_namespace(output) is xp
    expected_dtype = xp.asarray(1.0).dtype if dtype is int else dtype
    assert output.dtype == expected_dtype
    assert output.shape == (len(array_t), 2)


def test_single_value_input_normal_3D(
    initialised_twice_differentiable_3D_spline_curve, single_value_t, frame, xp, dtype
):
    spline = initialised_twice_differentiable_3D_spline_curve
    output = spline.normal(single_value_t, frame=frame, shift=1e-4)
    assert array_api_compat.is_array_api_obj(output)
    assert array_api_compat.array_namespace(output) is xp
    expected_dtype = xp.asarray(1.0).dtype if dtype is int else dtype
    assert output.dtype == expected_dtype
    assert output.shape == (2, 3)


def test_array_input_normal_3D(initialised_twice_differentiable_3D_spline_curve, array_t, frame, xp, dtype):
    spline = initialised_twice_differentiable_3D_spline_curve
    output = spline.normal(array_t, frame=frame, shift=1e-4)
    assert array_api_compat.is_array_api_obj(output)
    assert array_api_compat.array_namespace(output) is xp
    expected_dtype = xp.asarray(1.0).dtype if dtype is int else dtype
    assert output.dtype == expected_dtype
    assert output.shape == (len(array_t), 2, 3)


def test_single_value_input_moving_frame_3D(
    initialised_twice_differentiable_3D_spline_curve, single_value_t, frame, xp, dtype
):
    spline = initialised_twice_differentiable_3D_spline_curve
    output = spline.moving_frame(single_value_t, method=frame, shift=1e-4)
    assert array_api_compat.is_array_api_obj(output)
    assert array_api_compat.array_namespace(output) is xp
    expected_dtype = xp.asarray(1.0).dtype if dtype is int else dtype
    assert output.dtype == expected_dtype
    assert output.shape == (3, 3)


def test_array_input_moving_frame_3D(initialised_twice_differentiable_3D_spline_curve, array_t, frame, xp, dtype):
    spline = initialised_twice_differentiable_3D_spline_curve
    output = spline.moving_frame(array_t, method=frame, shift=1e-4)
    assert array_api_compat.is_array_api_obj(output)
    assert array_api_compat.array_namespace(output) is xp
    expected_dtype = xp.asarray(1.0).dtype if dtype is int else dtype
    assert output.dtype == expected_dtype
    assert output.shape == (len(array_t), 3, 3)


def test_ndim(initialised_spline_curve, as_xp_array):
    spline = initialised_spline_curve
    expected = spline(as_xp_array([0.5])).shape[-1]
    assert spline.ndim == expected


def test_basis_matrix_B1(as_xp_array, xp, device):
    control_points = as_xp_array(np.random.rand(5, 3))
    M = 5
    spline = splinebox.spline_curves.Spline(
        M=M,
        basis_function=splinebox.basis_functions.B1(),
        closed=False,
        control_points=control_points,
    )
    t = xp.arange(5, device=device)
    result = spline.basis_matrix(t)
    if array_api_compat.is_numpy_namespace(xp):
        result = result.todense()
    expected = xp.eye(M, dtype=control_points.dtype, device=device)[: len(t)]
    assert xp.allclose(result, expected)


def test_basis_matrix(as_xp_array, xp, device, derivative):
    control_points = as_xp_array(np.random.rand(7, 3))
    M = 5
    spline = splinebox.spline_curves.Spline(
        M=M,
        basis_function=splinebox.basis_functions.B3(),
        closed=False,
        control_points=control_points,
    )
    t = xp.asarray([-2.0, 0.5, 1.9, 2.4, 6.1], device=device, dtype=control_points.dtype)
    result = spline.basis_matrix(t, derivative=derivative)
    if array_api_compat.is_numpy_namespace(xp):
        result = result.todense()
    t_expected = xp.asarray(
        [
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0],
            [1.5, 0.5, -0.5, -1.5, -2.5, -3.5, -4.5],
            [2.9, 1.9, 0.9, -0.1, -1.1, -2.1, -3.1],
            [3.4, 2.4, 1.4, 0.4, -0.6, -1.6, -2.6],
            [7.1, 6.1, 5.1, 4.1, 3.1, 2.1, 1.1],
        ],
        device=device,
        dtype=control_points.dtype,
    )
    expected = spline.basis_function(t_expected, derivative=derivative)
    assert xp.allclose(result, expected)


def test_basis_matrix_closed(as_xp_array, xp, device, derivative):
    control_points = as_xp_array(np.random.rand(5, 3))
    M = 5
    spline = splinebox.spline_curves.Spline(
        M=M,
        basis_function=splinebox.basis_functions.B3(),
        closed=True,
        control_points=control_points,
    )
    t = xp.asarray([0.5, 1.9, 3.4], device=device, dtype=control_points.dtype)
    result = spline.basis_matrix(t, derivative=derivative)
    if array_api_compat.is_numpy_namespace(xp):
        result = result.todense()
    t_expected = xp.asarray(
        [
            [0.5, -0.5, -1.5, -2.5, 1.5],
            [1.9, 0.9, -0.1, -1.1, -2.1],
            [-1.6, 2.4, 1.4, 0.4, -0.6],
        ],
        device=device,
        dtype=control_points.dtype,
    )
    expected = spline.basis_function(t_expected, derivative=derivative)
    assert xp.allclose(result, expected)


def test_derivative_wrt_control_points_matches_basis_matrix(
    initialised_spline_curve, partial_derivative_positions, xp, is_hermite_spline
):
    spline = initialised_spline_curve
    t = partial_derivative_positions
    expected = spline.basis_matrix(t)[0] if is_hermite_spline(spline) else spline.basis_matrix(t)
    result = spline.derivative_wrt_control_points(t)
    if array_api_compat.is_numpy_namespace(xp):
        expected = expected.todense()
        result = result.todense()
    assert xp.allclose(result, expected, equal_nan=True)


def test_derivative_wrt_control_points_derivative_argument(
    initialised_spline_curve, partial_derivative_positions, xp, is_hermite_spline
):
    spline = initialised_spline_curve
    t = partial_derivative_positions
    for derivative in [0, 1, 2]:
        expected = (
            spline.basis_matrix(t, derivative=derivative)[0]
            if is_hermite_spline(spline)
            else spline.basis_matrix(t, derivative=derivative)
        )
        result = spline.derivative_wrt_control_points(t, derivative=derivative)
        if array_api_compat.is_numpy_namespace(xp):
            expected = expected.todense()
            result = result.todense()
        assert xp.allclose(result, expected, equal_nan=True)


def test_derivative_wrt_control_points_finite_difference(initialised_spline_curve, partial_derivative_positions, xp):
    spline = initialised_spline_curve
    t = partial_derivative_positions

    delta = np.random.randn(*spline.control_points.shape) * 1e-7
    dtype = (
        xp.asarray(1.0).dtype
        if xp.isdtype(spline.control_points.dtype, kind="integral")
        else spline.control_points.dtype
    )
    delta = xp.asarray(delta, dtype=dtype, device=spline.control_points.device)
    spline_perturbed = spline.copy()
    spline_perturbed.control_points = spline.control_points + delta

    jacobian = spline.derivative_wrt_control_points(t)
    if array_api_compat.is_numpy_namespace(xp):
        jacobian = jacobian.todense()
    predicted = jacobian @ delta
    actual = spline_perturbed(t) - spline(t)
    assert xp.allclose(predicted, actual, atol=1e-5)


def test_derivative_of_norm_squared_wrt_control_points_shape(
    initialised_spline_curve, partial_derivative_positions, xp
):
    spline = initialised_spline_curve
    n_control_points = spline.M if spline.closed else spline.M + 2 * spline.pad
    t = partial_derivative_positions
    result = spline.derivative_of_norm_squared_wrt_control_points(t)
    assert result.shape == (len(t), n_control_points, spline.ndim)


def test_derivative_of_norm_squared_wrt_control_points_finite_difference(
    initialised_spline_curve, partial_derivative_positions, xp
):
    spline = initialised_spline_curve
    t = partial_derivative_positions

    delta = np.random.randn(*spline.control_points.shape) * 1e-7
    dtype = (
        xp.asarray(1.0).dtype
        if xp.isdtype(spline.control_points.dtype, kind="integral")
        else spline.control_points.dtype
    )
    delta = xp.asarray(delta, dtype=dtype, device=spline.control_points.device)
    spline_perturbed = spline.copy()
    spline_perturbed.control_points = spline.control_points + delta

    def total_norm_squared(s):
        vals = s(t)
        return xp.sum(vals * vals)

    gradient = xp.sum(spline.derivative_of_norm_squared_wrt_control_points(t), axis=0)
    predicted = xp.sum(gradient * delta)
    actual = total_norm_squared(spline_perturbed) - total_norm_squared(spline)
    atol = 1e-2 if predicted.dtype == xp.float32 else 1e-5
    assert xp.isclose(predicted, actual, atol=atol)


def test_derivative_wrt_tangents_matches_basis_matrix(
    hermite_spline_curve, coeff_gen, partial_derivative_positions, xp
):
    spline = hermite_spline_curve
    spline.control_points = coeff_gen(spline.M, spline.basis_function.support, spline.closed)
    spline.tangents = coeff_gen(spline.M, spline.basis_function.support, spline.closed)
    t = partial_derivative_positions
    expected = spline.basis_matrix(t)[1]
    result = spline.derivative_wrt_tangents(t)
    if array_api_compat.is_numpy_namespace(xp):
        expected = expected.todense()
        result = result.todense()
    assert xp.allclose(result, expected, equal_nan=True)


def test_derivative_wrt_tangents_finite_difference(
    hermite_spline_curve, coeff_gen, partial_derivative_positions, xp, as_xp_array
):
    spline = hermite_spline_curve
    spline.control_points = coeff_gen(spline.M, spline.basis_function.support, spline.closed)
    spline.tangents = coeff_gen(spline.M, spline.basis_function.support, spline.closed)
    t = partial_derivative_positions

    delta = np.random.randn(*spline.tangents.shape) * 1e-7
    dtype = (
        xp.asarray(1.0).dtype
        if xp.isdtype(spline.control_points.dtype, kind="integral")
        else spline.control_points.dtype
    )
    delta = xp.asarray(delta, dtype=dtype, device=spline.control_points.device)
    spline_perturbed = spline.copy()
    spline_perturbed.tangents = spline.tangents + delta

    jacobian = spline.derivative_wrt_tangents(t)
    if array_api_compat.is_numpy_namespace(xp):
        jacobian = jacobian.todense()
    predicted = jacobian @ delta
    actual = spline_perturbed(t) - spline(t)
    assert xp.allclose(predicted, actual, atol=1e-5)


def test_h1_finite_difference(spline_curve, is_hermite_spline, request, codomain_dimensionality):
    # h1 appears in the derivative of the norm of the
    # first derivative of the spline to the fourth power.
    # For details, see the Active contour model sections of
    # in the Theory part of the documentation.

    spline = spline_curve
    rng = np.random.default_rng(seed=1492)
    spline.control_points = (
        rng.random((spline.M, codomain_dimensionality)) * 100
        if spline.closed
        else rng.random((spline.M + 2 * (math.ceil(spline.basis_function.support / 2) - 1), codomain_dimensionality))
        * 100
    )

    if is_hermite_spline(spline):
        # Currently not implemented for Hermite spline curves
        request.node.add_marker(pytest.mark.xfail)

    delta = np.random.randn(*spline.control_points.shape) * 1e-7
    spline_perturbed = spline.copy()
    spline_perturbed.control_points = spline.control_points + delta

    t = np.linspace(0, spline.M if spline.closed else spline.M - 1, 10000)
    r1 = spline(t, derivative=1)
    r1 = np.nan_to_num(r1)
    r1_perturbed = spline_perturbed(t, derivative=1)
    r1_perturbed = np.nan_to_num(r1_perturbed)
    actual = np.trapezoid(np.linalg.norm(r1_perturbed, axis=1) ** 4, t) - np.trapezoid(
        np.linalg.norm(r1, axis=1) ** 4, t
    )

    gradient = 4 * np.einsum(
        "kx,mx,ny,kmnl->ly", spline.control_points, spline.control_points, spline.control_points, spline._h1
    )
    predicted = np.sum(gradient * delta)
    assert np.allclose(predicted, actual, atol=1e-3)


def test_h2_finite_difference(spline_curve, is_hermite_spline, request, codomain_dimensionality):
    # h2 appears in the derivative of the squared norm of the
    # first derivative of the spline. For details, see the
    # Active contour model sections of in the Theory part of the
    # documentation.

    spline = spline_curve
    rng = np.random.default_rng(seed=1492)
    spline.control_points = (
        rng.random((spline.M, codomain_dimensionality)) * 100
        if spline.closed
        else rng.random((spline.M + 2 * (math.ceil(spline.basis_function.support / 2) - 1), codomain_dimensionality))
        * 100
    )

    if is_hermite_spline(spline):
        # Currently not implemented for Hermite spline curves
        request.node.add_marker(pytest.mark.xfail)

    delta = np.random.randn(*spline.control_points.shape) * 1e-7
    spline_perturbed = spline.copy()
    spline_perturbed.control_points = spline.control_points + delta

    t = np.linspace(0, spline.M if spline.closed else spline.M - 1, 10000)
    r1 = spline(t, derivative=1)
    r1 = np.nan_to_num(r1)
    r1_perturbed = spline_perturbed(t, derivative=1)
    r1_perturbed = np.nan_to_num(r1_perturbed)
    actual = np.trapezoid(np.sum(r1_perturbed**2, axis=1), t) - np.trapezoid(np.sum(r1**2, axis=1), t)

    gradient = 2 * spline._h2 @ spline.control_points
    predicted = np.sum(gradient * delta)
    assert np.allclose(predicted, actual, atol=1e-7)


def _numerical_reparametrization_energy(spline, c, n=10000):
    t = np.linspace(0, spline.M if spline.closed else spline.M - 1, n)
    r1 = spline(t, derivative=1)
    r1 = np.nan_to_num(r1)
    speed = np.linalg.norm(r1, axis=1)
    return np.trapezoid((speed**2 - c) ** 2, t)


def _numerical_normalized_reparametrization_energy(spline, n=10000):
    # Mirrors curvilinear_reparametrisation_energy with c derived from the arc
    # length, but uses a deterministic quadrature for the integral so that the
    # finite difference of two evaluations is not dominated by the noise of the
    # adaptive quadrature.
    t = np.linspace(0, spline.M if spline.closed else spline.M - 1, n)
    r1 = spline(t, derivative=1)
    r1 = np.nan_to_num(r1)
    squared_speed = np.linalg.norm(r1, axis=1) ** 2
    arc_length = spline.arc_length()
    c = (arc_length / spline.M) ** 2
    return np.trapezoid((squared_speed - c) ** 2, t) / arc_length**4


def _numerical_curvature_energy(spline, n=10000):
    t = np.linspace(0, spline.M if spline.closed else spline.M - 1, n)
    r2 = spline(t, derivative=2)
    r2 = np.nan_to_num(r2)
    return np.trapezoid(np.linalg.norm(r2, axis=1) ** 2, t)


def test_derivative_of_curvilinear_reparametrisation_energy_wrt_control_points_shape(
    initialised_spline_curve, is_hermite_spline, request, xp
):
    spline = initialised_spline_curve
    if array_api_compat.is_numpy_namespace(xp):
        if is_hermite_spline(spline):
            # Currently not implemented for Hermite spline curves
            request.node.add_marker(pytest.mark.xfail)
        result = spline.derivative_of_curvilinear_reparametrisation_energy_wrt_control_points(1.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == spline.control_points.shape
    else:
        with pytest.raises(NotImplementedError):
            spline.derivative_of_curvilinear_reparametrisation_energy_wrt_control_points(1.0)


def test_derivative_of_curvilinear_reparametrisation_energy_zero_for_uniform_speed(
    spline_curve, is_hermite_spline, codomain_dimensionality, request
):
    spline = spline_curve
    if is_hermite_spline(spline):
        # Currently not implemented for Hermite spline curves
        request.node.add_marker(pytest.mark.xfail)

    c = 2.0

    if not spline.closed and not isinstance(spline.basis_function, splinebox.Exponential):
        # For a diagonal line with equal coordinates, |r'|^2 = ndim * spacing^2
        # The energy is minimised when |r'|^2 = c, hence spacing = sqrt(c / ndim).
        # The speed is not constant for an exponential spline.
        spacing = np.sqrt(c / codomain_dimensionality)
        spline.control_points = np.array(
            [np.full(codomain_dimensionality, i * spacing) for i in range(spline.M + 2 * spline.pad)], dtype=float
        )

    elif spline.closed and isinstance(spline.basis_function, splinebox.Exponential) and codomain_dimensionality > 1:
        # Create a circle
        r = np.sqrt(c) * spline.M / 2 / np.pi
        knots = np.zeros((spline.M, codomain_dimensionality))
        theta = np.linspace(0, 2 * np.pi, spline.M, endpoint=False)
        knots[:, 0] = r * np.cos(theta)
        knots[:, 1] = r * np.sin(theta)
        spline.knots = knots

    elif (
        spline.closed
        and isinstance(spline.basis_function, splinebox.B1)
        and codomain_dimensionality == 1
        and spline.M % 2 == 0
    ):
        # Use a 1D B1 spline to build a line that goes back and forth
        length = np.sqrt(c) * spline.M / 2
        a = np.linspace(0, length, math.ceil(spline.M / 2) + 1)
        spline.control_points = np.concatenate([a, a[-2::-1]])[: spline.M, np.newaxis]

    elif spline.closed and isinstance(spline.basis_function, splinebox.B1) and codomain_dimensionality > 1:
        # Use a multi dimensional B1 spline to build a polygon
        control_points = np.zeros((spline.M, codomain_dimensionality))
        segment_length = np.sqrt(c)
        theta = np.arange(0, 2 * np.pi, 2 * np.pi / spline.M)
        for i in range(1, spline.M):
            control_points[i, 0] = control_points[i - 1, 0] + np.cos(theta[i - 1]) * segment_length
            control_points[i, 1] = control_points[i - 1, 1] + np.sin(theta[i - 1]) * segment_length
        spline.control_points = control_points

    elif not is_hermite_spline(spline):
        # It is not trivial to construct a spline with uniform speed for the remaining cases
        return

    result = spline.derivative_of_curvilinear_reparametrisation_energy_wrt_control_points(c)

    assert np.allclose(result, 0, atol=1e-8)


def test_derivative_of_curvilinear_reparametrisation_energy_wrt_control_points_finite_difference(
    initialised_spline_curve, is_hermite_spline, request, xp
):
    spline = initialised_spline_curve
    c = 1.5

    if array_api_compat.is_numpy_namespace(xp):
        if is_hermite_spline(spline):
            # Currently not implemented for Hermite spline curves
            request.node.add_marker(pytest.mark.xfail)

        delta = np.random.randn(*spline.control_points.shape) * 1e-7
        spline_perturbed = spline.copy()
        spline_perturbed.control_points = spline.control_points + delta

        gradient = spline.derivative_of_curvilinear_reparametrisation_energy_wrt_control_points(c)
        predicted = np.sum(gradient * delta)
        n = 1000000 if isinstance(spline.basis_function, splinebox.basis_functions.B1) else 10000
        actual = _numerical_reparametrization_energy(spline_perturbed, c, n=n) - _numerical_reparametrization_energy(
            spline, c, n=n
        )
        assert np.isclose(predicted, actual, atol=1e-5)
    else:
        with pytest.raises(NotImplementedError):
            spline.derivative_of_curvilinear_reparametrisation_energy_wrt_control_points(c)


def test_derivative_of_curvilinear_reparametrisation_energy_wrt_control_points_default_c(
    initialised_spline_curve, is_hermite_spline, request, xp
):
    spline = initialised_spline_curve

    if array_api_compat.is_numpy_namespace(xp):
        if is_hermite_spline(spline):
            # Currently not implemented for Hermite spline curves
            request.node.add_marker(pytest.mark.xfail)

        # When c is not provided, the gradient is the exact gradient of the scale
        # invariant curvilinear_reparametrisation_energy, including the dependence
        # of c and the normalisation on the arc length.
        delta = np.random.randn(*spline.control_points.shape) * 1e-5
        gradient = spline.derivative_of_curvilinear_reparametrisation_energy_wrt_control_points()
        predicted = np.sum(gradient * delta)

        spline_perturbed_plus = spline.copy()
        spline_perturbed_plus.control_points = spline.control_points + delta
        spline_perturbed_minus = spline.copy()
        spline_perturbed_minus.control_points = spline.control_points - delta

        n = 1000000 if isinstance(spline.basis_function, splinebox.basis_functions.B1) else 10000
        actual = (
            _numerical_normalized_reparametrization_energy(spline_perturbed_plus, n=n)
            - _numerical_normalized_reparametrization_energy(spline_perturbed_minus, n=n)
        ) / 2

        assert np.isclose(predicted, actual, rtol=1e-3, atol=1e-9)
    else:
        with pytest.raises(NotImplementedError):
            spline.derivative_of_curvilinear_reparametrisation_energy_wrt_control_points()


def test_derivative_of_curvature_energy_wrt_control_points_shape(
    initialised_spline_curve, is_hermite_spline, request, xp
):
    spline = initialised_spline_curve
    if array_api_compat.is_numpy_namespace(xp):
        if is_hermite_spline(spline):
            # Currently not implemented for Hermite spline curves
            request.node.add_marker(pytest.mark.xfail)
        result = spline.derivative_of_curvature_energy_wrt_control_points()
        assert isinstance(result, np.ndarray)
        assert result.shape == spline.control_points.shape
    else:
        with pytest.raises(NotImplementedError):
            spline.derivative_of_curvilinear_reparametrisation_energy_wrt_control_points()


def test_derivative_of_curvature_energy_wrt_control_points_zero_for_straight_line(
    open_spline_curve, codomain_dimensionality, is_hermite_spline, request, xp
):
    spline = open_spline_curve
    if array_api_compat.is_numpy_namespace(xp):
        if is_hermite_spline(spline):
            # Currently not implemented for Hermite spline curves
            request.node.add_marker(pytest.mark.xfail)
        if isinstance(spline.basis_function, splinebox.Exponential):
            # Not possible to construct a straight line with an exponential spline
            return

        direction = np.arange(1, codomain_dimensionality + 1)
        spline.control_points = np.array([i * direction for i in range(spline.M + 2 * spline.pad)], dtype=float)
        result = spline.derivative_of_curvature_energy_wrt_control_points()
        assert np.allclose(result, 0, atol=1e-10)
    else:
        direction = xp.arange(1, codomain_dimensionality + 1)
        spline.control_points = xp.stack([i * direction for i in range(spline.M + 2 * spline.pad)], axis=0)
        with pytest.raises(NotImplementedError):
            spline.derivative_of_curvilinear_reparametrisation_energy_wrt_control_points()


def test_derivative_of_curvature_energy_wrt_control_points_radial_for_circle(M, codomain_dimensionality):
    # For control points on a circle, the gradient of the curvature energy is
    # radial, i.e. parallel to the control points themselves. The energy is
    # invariant under rotations of the circle, so the gradient has no
    # tangential component. Note that the gradient does not vanish: shrinking
    # the circle reduces the energy.
    control_points = np.zeros((M, codomain_dimensionality))
    for i in range(M):
        control_points[i, 0] = np.sin(2 * np.pi * i / M)
        if codomain_dimensionality > 1:
            control_points[i, 1] = np.cos(2 * np.pi * i / M)

    spline = splinebox.Spline(M=M, basis_function=splinebox.Exponential(M), closed=True, control_points=control_points)

    result = spline.derivative_of_curvature_energy_wrt_control_points()

    # Subtract the radial component; only the tangential remainder must vanish.
    radial_coefficient = np.sum(result * control_points) / np.sum(control_points**2)
    assert np.allclose(result - radial_coefficient * control_points, 0, atol=1e-10)
    # The energy decreases when the circle shrinks, so the gradient points radially outward.
    assert radial_coefficient > 0


def test_derivative_of_curvature_energy_wrt_control_points_finite_difference(
    initialised_spline_curve, is_hermite_spline, request, xp
):
    spline = initialised_spline_curve
    if array_api_compat.is_numpy_namespace(xp):
        if is_hermite_spline(spline):
            # Currently not implemented for Hermite spline curves
            request.node.add_marker(pytest.mark.xfail)

        delta = np.random.randn(*spline.control_points.shape) * 1e-7
        spline_perturbed = spline.copy()
        spline_perturbed.control_points = spline.control_points + delta

        gradient = spline.derivative_of_curvature_energy_wrt_control_points()
        predicted = np.sum(gradient * delta)
        actual = _numerical_curvature_energy(spline_perturbed) - _numerical_curvature_energy(spline)
        assert np.isclose(predicted, actual, atol=1e-5)
    else:
        with pytest.raises(NotImplementedError):
            spline.derivative_of_curvilinear_reparametrisation_energy_wrt_control_points()
