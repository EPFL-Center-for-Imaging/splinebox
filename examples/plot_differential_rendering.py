import jax
import jax.numpy
import matplotlib
import numpy as np
import pyvista
import splinebox

DEVICE = jax.devices("gpu")[0] if any(device.platform == "gpu" for device in jax.devices()) else jax.devices("cpu")[0]

M = 5
BASIS_FUNCTION = splinebox.B3()
CLOSED = False
RADIUS = 4

# The origins and targets of the rays the are cast for the projections
IMG_SIZE = 128
UU, VV = jax.numpy.meshgrid(
    jax.numpy.arange(IMG_SIZE, device=DEVICE), jax.numpy.arange(IMG_SIZE, device=DEVICE), indexing="ij"
)
ORIGINS_ZX = jax.numpy.stack([UU, jax.numpy.full(UU.shape, -3 * IMG_SIZE, device=DEVICE), VV], axis=-1)
ORIGINS_ZY = jax.numpy.stack([jax.numpy.full(UU.shape, -3 * IMG_SIZE, device=DEVICE), VV, UU], axis=-1)
TARGETS_ZX = jax.numpy.stack([UU, jax.numpy.full(UU.shape, 3 * IMG_SIZE, device=DEVICE), VV], axis=-1)
TARGETS_ZY = jax.numpy.stack([jax.numpy.full(UU.shape, 3 * IMG_SIZE, device=DEVICE), VV, UU], axis=-1)

# Epsilon for the Carbonnier norm
EPSILON = 1e-8


@jax.jit
def ray_casting(coords, origins, targets):
    """
    coords : points along the spline
    origins : the origins of the rays (IMG_SIZE, IMG_SIZE, 3)
    targets : the targets of the rays (IMG_SIZE, IMG_SIZE, 3)
    """
    directions = targets - origins
    alpha = (
        jax.numpy.einsum("ij,klj->ikl", coords, directions) - jax.numpy.sum(directions * origins, axis=-1)
    ) / jax.numpy.linalg.norm(directions, axis=-1) ** 2
    closest_points = alpha[..., None] * directions + origins
    distance_vectors = closest_points - coords[:, None, None]
    # Use differentiable Charbonnier norm to compute the distance
    distances = jax.numpy.sqrt(jax.numpy.sum(distance_vectors * distance_vectors, axis=-1) + EPSILON**2)
    # Use soft min
    # distances = -jax.nn.logsumexp(-distances, axis=0)
    distances = distances.min(axis=0)
    return jax.nn.sigmoid((RADIUS - distances) / 1)


def render(spline):
    """
    Render two orthogonal projections of the spline into 2D images.

    Parameters
    ----------
    spline : splinebox.spline_curves.Spline
        The spline to render.

    Returns
    -------
    coords : jax array
        The points on the curve sampled at parameters between 0 and ``M - 1``.
    zx_image : jax array
        The view along y: rows index x, columns index z.
    zy_image : jax array
        The view along x: rows index z, columns index y.
    """
    t = jax.numpy.linspace(0, spline.M - 1, 10000)
    coords = spline(t)

    zx_image = ray_casting(coords, ORIGINS_ZX, TARGETS_ZX)
    zy_image = ray_casting(coords, ORIGINS_ZY, TARGETS_ZY)

    return coords, zx_image, zy_image


def to_texture(image):
    values = np.array(image)
    clim = np.max(np.abs([values.min(), values.max()]))
    values = (values + clim) / (2 * clim) if clim > 0 else np.full_like(values, 0.5)
    rgba = matplotlib.colormaps["bwr"](values)
    return pyvista.Texture((rgba * 255).astype(np.uint8))


def loss_func(control_points, target_zx_image, target_zy_image):
    spline = splinebox.Spline(
        M=M,
        basis_function=BASIS_FUNCTION,
        closed=CLOSED,
        control_points=control_points,
    )
    _, zx_image, zy_image = render(spline)
    return jax.numpy.sum((target_zx_image - zx_image) ** 2) + jax.numpy.sum((target_zy_image - zy_image) ** 2)


grad_func = jax.grad(loss_func)


# %%
# 1. Construct a random spline
# ----------------------------
#
# The control points are placed along the diagonal of a cube with random
# offsets perpendicular to the diagonal.

np.random.seed(0)
offset_from_diag = jax.numpy.array(np.random.rand(7, 3) / 2 - 1 / 4, device=DEVICE) * IMG_SIZE
diag_vector = jax.numpy.ones(3, device=DEVICE)
diag_vector = diag_vector / jax.numpy.linalg.vector_norm(diag_vector)
offset_from_diag = offset_from_diag - jax.numpy.linalg.outer(offset_from_diag @ diag_vector, diag_vector)
target_control_points = (
    jax.numpy.tile(jax.numpy.linspace(20, IMG_SIZE - 20, 7, device=DEVICE)[:, None], (1, 3)) + offset_from_diag
)
target_spline = splinebox.Spline(
    M=M, basis_function=BASIS_FUNCTION, closed=CLOSED, control_points=target_control_points
)

target_coords, target_zx_image, target_zy_image = render(target_spline)
target_zx_image.block_until_ready()
target_zy_image.block_until_ready()
target_spline_mesh = pyvista.MultipleLines(points=np.array(target_coords)).tube(radius=RADIUS, absolute=True)
target_control_point_mesh = pyvista.PolyData(np.array(target_spline.control_points))

DISTANCE = 3 * IMG_SIZE
ZX_PLANE_CORNERS = np.array(
    [[0, DISTANCE, 0], [IMG_SIZE, DISTANCE, 0], [IMG_SIZE, DISTANCE, IMG_SIZE], [0, DISTANCE, IMG_SIZE]], dtype=float
)
ZY_PLANE_CORNERS = np.array(
    [[DISTANCE, 0, 0], [DISTANCE, IMG_SIZE, 0], [DISTANCE, IMG_SIZE, IMG_SIZE], [DISTANCE, 0, IMG_SIZE]], dtype=float
)
zx_plane = pyvista.PolyData(ZX_PLANE_CORNERS).delaunay_2d()
zy_plane = pyvista.PolyData(ZY_PLANE_CORNERS).delaunay_2d()
zx_plane.texture_map_to_plane(
    origin=(IMG_SIZE, DISTANCE, 0),
    point_u=(IMG_SIZE, DISTANCE, IMG_SIZE),
    point_v=(0, DISTANCE, 0),
    inplace=True,
)
zy_plane.texture_map_to_plane(
    origin=(DISTANCE, 0, IMG_SIZE),
    point_u=(DISTANCE, IMG_SIZE, IMG_SIZE),
    point_v=(DISTANCE, 0, 0),
    inplace=True,
)

control_points = jax.numpy.tile(jax.numpy.linspace(20, IMG_SIZE - 20, 7, device=DEVICE)[:, None], (1, 3))
spline = splinebox.Spline(M=M, basis_function=BASIS_FUNCTION, closed=CLOSED, control_points=control_points)

losses = []
lr = 0.5
for i in range(85):
    if (i % 1) == 0:
        coords, zx_image, zy_image = render(spline)
        zx_texture = to_texture(zx_image - target_zx_image)
        zy_texture = to_texture(zy_image - target_zy_image)

        spline_mesh = pyvista.MultipleLines(points=np.array(coords)).tube(radius=RADIUS, absolute=True)
        control_point_mesh = pyvista.PolyData(np.array(spline.control_points))

        plotter = pyvista.Plotter(off_screen=True)
        plotter.camera.position = (-392.769, -348.183, 474.868)
        plotter.camera.roll = 67.511
        plotter.camera.focal_point = (124.80555661565425, 123.81243010926669, 91.2096170882193)
        plotter.add_mesh(spline_mesh, color="red", label="spline")
        plotter.add_mesh(target_spline_mesh, color="blue", label="target spline")
        plotter.add_mesh(
            control_point_mesh, color="red", point_size=10, render_points_as_spheres=True, label="control points"
        )
        plotter.add_mesh(
            target_control_point_mesh,
            color="blue",
            point_size=10,
            render_points_as_spheres=True,
            label="target control points",
        )
        plotter.add_mesh(zx_plane, texture=zx_texture, label="z-x projection")
        plotter.add_mesh(zy_plane, texture=zy_texture, label="z-y projection")
        plotter.add_legend()
        plotter.add_axes()
        # plotter.show()
        plotter.screenshot(f"differential_rendering/{i}.png")
        plotter.close()

    loss = loss_func(spline.control_points, target_zx_image, target_zy_image)
    losses.append(loss)
    if len(losses) > 2 and losses[-2] < losses[-1]:
        lr /= 2
        print(f"{lr = }")
    print(f"{i}: {loss}")
    grad_jax = grad_func(spline.control_points, target_zx_image, target_zy_image)
    spline.control_points -= lr * grad_jax
