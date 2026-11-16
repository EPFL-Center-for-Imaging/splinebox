import jax
import jax.numpy
import matplotlib
import numpy as np
import pyvista
import splinebox
import tqdm

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
POSITION_CAM_1 = jax.numpy.array([IMG_SIZE / 2, -350.0, IMG_SIZE / 2], device=DEVICE)
POSITION_CAM_2 = jax.numpy.array([-350.0, IMG_SIZE / 2, IMG_SIZE / 2], device=DEVICE)
IMG_PLANE_CAM_1 = jax.numpy.stack([UU, jax.numpy.full(UU.shape, 2.0 * IMG_SIZE, device=DEVICE), VV], axis=-1)
IMG_PLANE_CAM_2 = jax.numpy.stack([jax.numpy.full(UU.shape, 2.0 * IMG_SIZE, device=DEVICE), VV, UU], axis=-1)

# Epsilon for the Charbonnier norm
EPSILON = 1e-8


@jax.jit
@jax.checkpoint
def ray_casting(coords, camera_position, img_plane):
    """
    coords : points along the spline
    origins : the origins of the rays (IMG_SIZE, IMG_SIZE, 3)
    targets : the targets of the rays (IMG_SIZE, IMG_SIZE, 3)
    """
    directions = img_plane - camera_position
    alpha = (
        jax.numpy.einsum("ij,klj->ikl", coords, directions) - jax.numpy.sum(directions * camera_position, axis=-1)
    ) / jax.numpy.linalg.norm(directions, axis=-1) ** 2
    closest_points = alpha[..., None] * directions + camera_position
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

    zx_image = ray_casting(coords, POSITION_CAM_1, IMG_PLANE_CAM_1)
    zy_image = ray_casting(coords, POSITION_CAM_2, IMG_PLANE_CAM_2)

    return coords, zx_image, zy_image


def to_texture(image):
    values = np.array(image)
    clim = np.max(np.abs([values.min(), values.max()]))
    values = (values + clim) / (2 * clim) if clim > 0 else np.full_like(values, 0.5)
    rgba = matplotlib.colormaps["bwr"](values)
    return pyvista.Texture((rgba * 255).astype(np.uint8))


def loss_func(control_points, target_img_cam_1, target_img_cam_2):
    spline = splinebox.Spline(
        M=M,
        basis_function=BASIS_FUNCTION,
        closed=CLOSED,
        control_points=control_points,
    )
    _, img_cam_1, img_cam_2 = render(spline)
    return jax.numpy.sum((target_img_cam_1 - img_cam_1) ** 2) + jax.numpy.sum((target_img_cam_2 - img_cam_2) ** 2)


value_and_grad_of_loss = jax.value_and_grad(loss_func)


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

target_coords, target_img_cam_1, target_img_cam_2 = render(target_spline)
target_img_cam_1.block_until_ready()
target_img_cam_2.block_until_ready()
target_spline_mesh = pyvista.MultipleLines(points=np.array(target_coords)).tube(radius=RADIUS, absolute=True)
target_control_point_mesh = pyvista.PolyData(np.array(target_spline.control_points))

DISTANCE = 3 * IMG_SIZE
img_plane_cam_1_corners = np.array(
    [IMG_PLANE_CAM_1[0, 0], IMG_PLANE_CAM_1[-1, 0], IMG_PLANE_CAM_1[-1, -1], IMG_PLANE_CAM_1[0, -1]]
)
img_plane_cam_2_corners = np.array(
    [IMG_PLANE_CAM_2[0, 0], IMG_PLANE_CAM_2[-1, 0], IMG_PLANE_CAM_2[-1, -1], IMG_PLANE_CAM_2[0, -1]]
)
img_plane_cam_1 = pyvista.PolyData(img_plane_cam_1_corners).delaunay_2d()
img_plane_cam_2 = pyvista.PolyData(img_plane_cam_2_corners).delaunay_2d()
img_plane_cam_1.texture_map_to_plane(
    origin=img_plane_cam_1_corners[1],
    point_u=img_plane_cam_1_corners[2],
    point_v=img_plane_cam_1_corners[0],
    inplace=True,
)
img_plane_cam_2.texture_map_to_plane(
    origin=img_plane_cam_2_corners[1],
    point_u=img_plane_cam_2_corners[2],
    point_v=img_plane_cam_2_corners[0],
    inplace=True,
)

control_points = jax.numpy.tile(jax.numpy.linspace(20, IMG_SIZE - 20, 7, device=DEVICE)[:, None], (1, 3))
spline = splinebox.Spline(M=M, basis_function=BASIS_FUNCTION, closed=CLOSED, control_points=control_points)

losses = []
lr = 0.5
for i in tqdm.tqdm(np.arange(85)):
    if (i % 1) == 0:
        coords, img_cam_1, img_cam_2 = render(spline)
        zx_texture = to_texture(img_cam_1 - target_img_cam_1)
        zy_texture = to_texture(img_cam_2 - target_img_cam_2)

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
        plotter.add_mesh(img_plane_cam_1, texture=zx_texture, label="z-x projection")
        plotter.add_mesh(img_plane_cam_2, texture=zy_texture, label="z-y projection")
        plotter.add_lines(
            np.array(
                [
                    POSITION_CAM_1,
                    img_plane_cam_1_corners[0],
                    POSITION_CAM_1,
                    img_plane_cam_1_corners[1],
                    POSITION_CAM_1,
                    img_plane_cam_1_corners[2],
                    POSITION_CAM_1,
                    img_plane_cam_1_corners[3],
                    POSITION_CAM_2,
                    img_plane_cam_2_corners[0],
                    POSITION_CAM_2,
                    img_plane_cam_2_corners[1],
                    POSITION_CAM_2,
                    img_plane_cam_2_corners[2],
                    POSITION_CAM_2,
                    img_plane_cam_2_corners[3],
                ]
            ),
            color="black",
            width=1,
        )
        plotter.add_legend()
        plotter.add_axes()
        # plotter.show()
        plotter.screenshot(f"differential_rendering/{i}.png")
        plotter.close()

    loss, grad_jax = value_and_grad_of_loss(spline.control_points, target_img_cam_1, target_img_cam_2)
    spline.control_points -= lr * grad_jax

    losses.append(loss)
    if len(losses) > 2 and losses[-2] < losses[-1]:
        lr /= 2
        print(f"{lr = }")
    print(f"{i}: {loss}")
