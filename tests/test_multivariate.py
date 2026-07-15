import numpy as np
import pytest

import splinebox
import splinebox.basis_functions
import splinebox.multivariate


@pytest.fixture(
    params=[
        (splinebox.basis_functions.B1, {}),
        (splinebox.basis_functions.B2, {}),
        (splinebox.basis_functions.B3, {}),
        (splinebox.basis_functions.CatmullRom, {}),
        (splinebox.basis_functions.Exponential, {"M": 8}),
    ],
    ids=["B1", "B2", "B3", "CatmullRom", "Exponential-M8"],
)
def non_hermite_basis_function(request):
    basis_function, params = request.param
    return basis_function(**params)


@pytest.fixture(params=[True, False], ids=["closed", "open"])
def closed(request):
    return request.param


@pytest.fixture(params=[1, 2, 3], ids=["ndim1", "ndim2", "ndim3"])
def codomain_dimensionality(request):
    return request.param


def _control_points_shape(M, basis_functions, closed):
    shape = []
    for m, bf, c in zip(M, basis_functions, closed):
        pad = int(np.ceil(bf.support / 2)) - 1
        shape.append(m if c else m + 2 * pad)
    return tuple(shape)


def test_tensor_product_vectors():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    result = splinebox.multivariate.tensor_product([a, b])
    expected = np.multiply.outer(a, b)[..., np.newaxis]
    assert np.allclose(result, expected)


def test_tensor_product_matrices():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0]])
    result = splinebox.multivariate.tensor_product([a, b])
    expected = np.einsum("ij,kj->ikj", a, b)
    assert np.allclose(result, expected)


def test_tensor_product_high_dim_raises():
    with pytest.raises(ValueError):
        splinebox.multivariate.tensor_product([np.zeros((2, 2, 2))])


def test_multivariate_spline_construction():
    bf = splinebox.basis_functions.B3()
    spline = splinebox.multivariate.MultivariateSpline(
        M=(4, 5),
        basis_functions=bf,
        closed=(False, False),
    )
    assert spline.nvariate == 2
    assert all(isinstance(b, splinebox.basis_functions.B3) for b in spline.basis_functions)
    assert list(spline.closed) == [False, False]


def test_multivariate_spline_per_variable_basis():
    bf1 = splinebox.basis_functions.B1()
    bf2 = splinebox.basis_functions.B3()
    spline = splinebox.multivariate.MultivariateSpline(
        M=(4, 5),
        basis_functions=[bf1, bf2],
        closed=(True, False),
    )
    assert spline.basis_functions == [bf1, bf2]
    assert spline.pad == (0, 1)


def test_multivariate_spline_M_not_iterable_raises():
    with pytest.raises(ValueError):
        splinebox.multivariate.MultivariateSpline(
            M=4,
            basis_functions=splinebox.basis_functions.B1(),
            closed=True,
        )


def test_multivariate_spline_M_non_integer_raises():
    with pytest.raises(ValueError):
        splinebox.multivariate.MultivariateSpline(
            M=(4.5, 5),
            basis_functions=splinebox.basis_functions.B1(),
            closed=(True, False),
        )


def test_multivariate_spline_basis_functions_wrong_length_raises():
    with pytest.raises(ValueError):
        splinebox.multivariate.MultivariateSpline(
            M=(4, 5),
            basis_functions=[splinebox.basis_functions.B1()],
            closed=(True, False),
        )


def test_multivariate_spline_closed_wrong_length_raises():
    with pytest.raises(ValueError):
        splinebox.multivariate.MultivariateSpline(
            M=(4, 5),
            basis_functions=splinebox.basis_functions.B1(),
            closed=(True,),
        )


def test_multivariate_spline_padding_functions_wrong_length_raises():
    with pytest.raises(ValueError):
        splinebox.multivariate.MultivariateSpline(
            M=(4, 5),
            basis_functions=splinebox.basis_functions.B1(),
            closed=(True, False),
            padding_functions=[lambda k, p: k],
        )


def test_multivariate_spline_M_too_small_raises():
    with pytest.raises(RuntimeError):
        splinebox.multivariate.MultivariateSpline(
            M=(3, 5),
            basis_functions=splinebox.basis_functions.B3(),
            closed=(False, False),
        )


def test_read_only_properties():
    spline = splinebox.multivariate.MultivariateSpline(
        M=(4, 5),
        basis_functions=splinebox.basis_functions.B3(),
        closed=(False, False),
    )
    with pytest.raises(RuntimeError):
        spline.half_support = (1.0, 2.0)
    with pytest.raises(RuntimeError):
        spline.pad = (1, 1)


def test_pad_matches_support():
    spline = splinebox.multivariate.MultivariateSpline(
        M=(8, 8),
        basis_functions=[splinebox.basis_functions.B1(), splinebox.basis_functions.B3()],
        closed=(False, False),
    )
    assert spline.pad == (0, 1)


def test_ndim():
    spline = splinebox.multivariate.MultivariateSpline(
        M=(4, 5),
        basis_functions=splinebox.basis_functions.B3(),
        closed=(False, False),
    )
    spline.control_points = np.random.rand(6, 7, 3)
    assert spline.ndim == 3


def test_set_control_points_closed_open():
    spline = splinebox.multivariate.MultivariateSpline(
        M=(4, 5),
        basis_functions=splinebox.basis_functions.B3(),
        closed=(True, False),
    )
    spline.control_points = np.random.rand(4, 7, 2)
    assert spline.control_points.shape == (4, 7, 2)


def test_set_control_points_wrong_shape_raises():
    spline = splinebox.multivariate.MultivariateSpline(
        M=(4, 5),
        basis_functions=splinebox.basis_functions.B3(),
        closed=(False, False),
    )
    with pytest.raises(ValueError):
        spline.control_points = np.random.rand(5, 7, 2)


def test_set_control_points_ndim_mismatch_raises():
    spline = splinebox.multivariate.MultivariateSpline(
        M=(4, 5),
        basis_functions=splinebox.basis_functions.B3(),
        closed=(False, False),
    )
    # ndim neither nvariate nor nvariate + 1
    with pytest.raises(ValueError):
        spline.control_points = np.random.rand(6, 7, 2, 3)


@pytest.mark.parametrize(
    "M,basis,closed",
    [
        ((4, 5), splinebox.basis_functions.B1(), (False, False)),
        ((4, 5), splinebox.basis_functions.B1(), (True, True)),
        ((6, 7), splinebox.basis_functions.B3(), (False, False)),
        ((6, 7), splinebox.basis_functions.B3(), (True, True)),
    ],
)
def test_call_output_shape(M, basis, closed):
    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=basis,
        closed=closed,
    )
    shape = _control_points_shape(M, spline.basis_functions, closed) + (2,)
    spline.control_points = np.random.rand(*shape)

    t = np.stack(
        np.meshgrid(
            np.linspace(0, M[0] - (0 if closed[0] else 1), 8),
            np.linspace(0, M[1] - (0 if closed[1] else 1), 9),
            indexing="ij",
        ),
        axis=-1,
    )
    values = spline(t)
    assert values.shape == t.shape[:-1] + (2,)


def test_call_wrong_t_dimension_raises():
    spline = splinebox.multivariate.MultivariateSpline(
        M=(4, 5),
        basis_functions=splinebox.basis_functions.B1(),
        closed=(True, True),
    )
    spline.control_points = np.random.rand(4, 5, 2)
    with pytest.raises(ValueError):
        spline(np.array([0.0, 1.0, 2.0]))


def test_call_wrong_derivatives_length_raises():
    spline = splinebox.multivariate.MultivariateSpline(
        M=(4, 5),
        basis_functions=splinebox.basis_functions.B1(),
        closed=(True, True),
    )
    spline.control_points = np.random.rand(4, 5, 2)
    with pytest.raises(ValueError):
        spline(np.array([[0.0, 1.0]]), derivatives=[0, 0, 0])


def test_call_interpolating_at_knots():
    """For an interpolating basis the spline evaluated at knot positions equals the knots."""
    M = (4, 5)
    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.basis_functions.B1(),
        closed=(False, False),
    )
    knots = np.random.rand(*M, 2)
    spline.knots = knots
    t = np.stack(np.meshgrid(np.arange(M[0]), np.arange(M[1]), indexing="ij"), axis=-1)
    assert np.allclose(spline(t), knots)


def test_call_periodic_boundary():
    M = (6, 7)
    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.basis_functions.B3(),
        closed=(True, False),
    )
    shape = _control_points_shape(M, spline.basis_functions, (True, False)) + (2,)
    spline.control_points = np.random.rand(*shape)

    t1 = np.linspace(0, M[1] - 1, 15)
    t_start = np.stack(np.meshgrid(np.array([0.0]), t1, indexing="ij"), axis=-1)
    t_end = np.stack(np.meshgrid(np.array([float(M[0])]), t1, indexing="ij"), axis=-1)
    assert np.allclose(spline(t_start), spline(t_end))


def test_call_derivatives_finite_difference():
    M = (8, 9)
    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.basis_functions.B3(),
        closed=(False, False),
    )
    shape = _control_points_shape(M, spline.basis_functions, (False, False)) + (1,)
    spline.control_points = np.random.rand(*shape)

    t0 = np.linspace(2, M[0] - 2, 10)
    t1 = np.linspace(2, M[1] - 2, 10)
    t = np.stack(np.meshgrid(t0, t1, indexing="ij"), axis=-1)

    analytical = spline(t, derivatives=[1, 0])
    eps = 1e-5
    t_plus = t.copy()
    t_plus[..., 0] += eps
    t_minus = t.copy()
    t_minus[..., 0] -= eps
    finite_diff = (spline(t_plus) - spline(t_minus)) / (2 * eps)
    assert np.allclose(analytical, finite_diff, atol=1e-6)


@pytest.mark.xfail(reason="__call__ assumes t.ndim == nvariate + 1")
def test_call_single_parameter_vector():
    spline = splinebox.multivariate.MultivariateSpline(
        M=(4, 5),
        basis_functions=splinebox.basis_functions.B1(),
        closed=(True, True),
    )
    spline.control_points = np.random.rand(4, 5, 2)
    t = np.array([1.5, 2.0])
    spline(t)


def test_knots_property_open():
    M = (4, 5)
    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.basis_functions.B3(),
        closed=(False, False),
    )
    shape = _control_points_shape(M, spline.basis_functions, (False, False)) + (2,)
    control_points = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            control_points[i, j] = [i, j]
    spline.control_points = control_points
    knots = spline.knots
    assert knots.shape == (*M, 2)
    # Add one because of the padding for B3 splines
    assert np.allclose(knots[..., 0], np.arange(M[0])[:, None] + 1)
    assert np.allclose(knots[..., 1], np.arange(M[1])[None, :] + 1)


def test_knots_setter_reconstructs_interpolating():
    M = (4, 5)
    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.basis_functions.B1(),
        closed=(True, True),
    )
    knots = np.random.rand(*M, 2)
    # This makes sense because the knots setter calls fit under the hood
    spline.knots = knots
    assert np.allclose(spline.knots, knots)


def test_knots_setter_open():
    M = (4, 5)
    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.basis_functions.B3(),
        closed=(False, False),
    )
    knots = np.random.rand(*M, 2)
    spline.knots = knots
    shape = _control_points_shape(M, spline.basis_functions, (False, False)) + (2,)
    assert spline.control_points.shape == shape


def test_fit_reconstructs_vector_field():
    """Fitting a vector field whose grid equals M should reconstruct it (for B1)."""
    M = (8, 7)
    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.basis_functions.B1(),
        closed=(True, True),
    )
    points = np.random.rand(*M, 2)
    spline.fit(points)

    t0 = np.linspace(0, M[0], M[0], endpoint=False)
    t1 = np.linspace(0, M[1], M[1], endpoint=False)
    t = np.stack(np.meshgrid(t0, t1, indexing="ij"), axis=-1)
    assert np.allclose(spline(t), points)


def test_fit_open_b3_approximate():
    M = (8, 8)
    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.basis_functions.B3(),
        closed=(False, False),
    )
    points = np.random.rand(*M, 2)
    spline.fit(points)

    t0 = np.linspace(0, M[0] - 1, M[0])
    t1 = np.linspace(0, M[1] - 1, M[1])
    t = np.stack(np.meshgrid(t0, t1, indexing="ij"), axis=-1)
    reconstructed = spline(t)
    assert reconstructed.shape == points.shape
    assert np.allclose(reconstructed, points, atol=1e-4)


def test_fit_with_explicit_t():
    M = (4, 4)
    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.basis_functions.B3(),
        closed=(False, False),
    )
    points = np.random.rand(10, 10, 1)
    t0 = np.linspace(0, M[0] - 1, 10)
    t1 = np.linspace(0, M[1] - 1, 10)
    t = np.stack(np.meshgrid(t0, t1, indexing="ij"), axis=-1)
    spline.fit(points, t=t)
    assert spline.control_points.shape == (6, 6, 1)


def test_fit_scalar_field():
    M = (8, 8)
    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.B1(),
        closed=(True, True),
    )
    points = np.random.rand(*M)
    spline.fit(points)
    assert spline.control_points.shape == (*M, 1)

    t = np.stack(
        np.meshgrid(
            np.linspace(0, M[0], M[0], endpoint=False),
            np.linspace(0, M[1], M[1], endpoint=False),
            indexing="ij",
        ),
        axis=-1,
    )
    assert np.allclose(spline(t).squeeze(), points)


def test_fit_scalar_field_with_explicit_t():
    M = (4, 4)
    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.B3(),
        closed=(False, False),
    )
    points = np.random.rand(10, 10)
    t = np.stack(
        np.meshgrid(np.linspace(0, M[0] - 1, 10), np.linspace(0, M[1] - 1, 10), indexing="ij"),
        axis=-1,
    )
    spline.fit(points, t=t)
    assert spline.control_points.shape == (6, 6, 1)


def test_mesh_closed():
    M = (4, 4)
    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.basis_functions.B3(),
        closed=(True, True),
    )
    spline.control_points = np.random.rand(*M, 3)
    points, connectivity = spline.mesh(step_t=1.0)
    assert points.shape[1] == 3
    assert connectivity.shape[1] == 3
    assert len(points) == M[0] * M[1]
    assert len(connectivity) == 2 * M[0] * M[1]


@pytest.mark.parametrize(
    "closed",
    [(True, True), (True, False), (False, True), (False, False)],
    ids=["TT", "TF", "FT", "FF"],
)
def test_mesh_matches_evaluation(closed):
    M = (4, 4)
    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.basis_functions.B3(),
        closed=closed,
    )
    shape = _control_points_shape(M, spline.basis_functions, closed) + (3,)
    spline.control_points = np.random.rand(*shape)
    step_t = 1.0
    points, _ = spline.mesh(step_t=step_t)

    stop0 = M[0] if closed[0] else M[0] - 1 + 0.9 * step_t
    stop1 = M[1] if closed[1] else M[1] - 1 + 0.9 * step_t
    t0 = np.arange(0, stop0, step_t)
    t1 = np.arange(0, stop1, step_t)
    t = np.stack(np.meshgrid(t0, t1, indexing="ij"), axis=-1)
    expected = spline(t).reshape(-1, 3)
    assert np.allclose(points, expected)


def test_mesh_scalar_field():
    M = (4, 4)
    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.basis_functions.B3(),
        closed=(True, True),
    )
    spline.control_points = np.random.rand(*M, 1)
    points, connectivity = spline.mesh(step_t=1.0)

    stop = M[0]
    t = np.stack(np.meshgrid(np.arange(0, stop, 1.0), np.arange(0, stop, 1.0), indexing="ij"), axis=-1)
    expected_values = spline(t).reshape(-1, 1)
    expected = np.concatenate([expected_values, t.reshape(-1, 2)], axis=-1)
    assert np.allclose(points, expected)
    assert connectivity.shape[1] == 3


def test_mesh_not_bivariate_raises():
    spline = splinebox.multivariate.MultivariateSpline(
        M=(3, 3, 3),
        basis_functions=splinebox.basis_functions.B1(),
        closed=(True, True, True),
    )
    spline.control_points = np.random.rand(3, 3, 3, 3)
    with pytest.raises(NotImplementedError):
        spline.mesh()


def test_mesh_wrong_codomain_raises():
    spline = splinebox.multivariate.MultivariateSpline(
        M=(4, 4),
        basis_functions=splinebox.basis_functions.B3(),
        closed=(True, True),
    )
    spline.control_points = np.random.rand(4, 4, 2)
    with pytest.raises(NotImplementedError):
        spline.mesh()


def test_translation_invariance():
    M = (4, 5)
    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.basis_functions.B3(),
        closed=(False, False),
    )
    shape = _control_points_shape(M, spline.basis_functions, (False, False)) + (3,)
    control_points = np.random.rand(*shape)
    spline.control_points = control_points

    translated = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.basis_functions.B3(),
        closed=(False, False),
    )
    translated.control_points = control_points + np.array([1.0, -2.0, 3.0])

    t = np.stack(
        np.meshgrid(np.linspace(0, M[0] - 1, 6), np.linspace(0, M[1] - 1, 7), indexing="ij"),
        axis=-1,
    )
    assert np.allclose(translated(t), spline(t) + np.array([1.0, -2.0, 3.0]))


def test_rotation_invariance():
    M = (4, 5)
    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.basis_functions.B3(),
        closed=(False, False),
    )
    shape = _control_points_shape(M, spline.basis_functions, (False, False)) + (3,)
    control_points = np.random.rand(*shape)
    spline.control_points = control_points

    angle = np.pi / 7
    R = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    rotated = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.basis_functions.B3(),
        closed=(False, False),
    )
    rotated.control_points = (R @ control_points.reshape(-1, 3).T).T.reshape(control_points.shape)

    t = np.stack(
        np.meshgrid(np.linspace(0, M[0] - 1, 6), np.linspace(0, M[1] - 1, 7), indexing="ij"),
        axis=-1,
    )
    expected = (R @ spline(t).reshape(-1, 3).T).T.reshape(spline(t).shape)
    assert np.allclose(rotated(t), expected)


def test_equals_univariate_product():
    """A separable bivariate spline equals the product of two univariate splines."""
    M = (4, 5)
    sx = splinebox.spline_curves.Spline(M=M[0], basis_function=splinebox.basis_functions.B3(), closed=False)
    sy = splinebox.spline_curves.Spline(M=M[1], basis_function=splinebox.basis_functions.B3(), closed=False)
    cpx = np.random.rand(6, 3)
    cpy = np.random.rand(7, 3)
    sx.control_points = cpx
    sy.control_points = cpy

    control_points = np.zeros((6, 7, 3))
    for i in range(6):
        for j in range(7):
            control_points[i, j] = cpx[i] * cpy[j]

    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.basis_functions.B3(),
        closed=(False, False),
    )
    spline.control_points = control_points

    t0 = np.linspace(0, M[0] - 1, 8)
    t1 = np.linspace(0, M[1] - 1, 9)
    t = np.stack(np.meshgrid(t0, t1, indexing="ij"), axis=-1)
    multivariate_values = spline(t)
    univariate_product = sx(t0)[:, None, :] * sy(t1)[None, :, :]
    assert np.allclose(multivariate_values, univariate_product)


def test_three_dimensional_eval():
    M = (4, 5, 6)
    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.basis_functions.B3(),
        closed=(False, False, False),
    )
    shape = _control_points_shape(M, spline.basis_functions, (False, False, False)) + (2,)
    spline.control_points = np.random.rand(*shape)
    t = np.stack(
        np.meshgrid(
            np.linspace(0, M[0] - 1, 4),
            np.linspace(0, M[1] - 1, 5),
            np.linspace(0, M[2] - 1, 6),
            indexing="ij",
        ),
        axis=-1,
    )
    values = spline(t)
    assert values.shape == t.shape[:-1] + (2,)


def test_three_dimensional_fit():
    M = (4, 4, 4)
    spline = splinebox.multivariate.MultivariateSpline(
        M=M,
        basis_functions=splinebox.basis_functions.B1(),
        closed=(True, True, True),
    )
    points = np.random.rand(*M, 2)
    spline.fit(points)
    t = np.stack(
        np.meshgrid(
            np.linspace(0, M[0], M[0], endpoint=False),
            np.linspace(0, M[1], M[1], endpoint=False),
            np.linspace(0, M[2], M[2], endpoint=False),
            indexing="ij",
        ),
        axis=-1,
    )
    assert np.allclose(spline(t), points)
