"""
Spline to mesh
==============
This example demonstrates how you can turn a spline curve in 3D into a mesh.
"""

# sphinx_gallery_thumbnail_number = 4

import numpy as np
import pyvista
import splinebox

# %%
# 1. Construct a spline
# ---------------------
# We begin by constructing a circular spline.
M = 4
spline = splinebox.Spline(M=M, basis_function=splinebox.Exponential(M), closed=True)
spline.knots = np.array([[0, 0, 1], [0, 1, 0], [0, 0, -1], [0, -1, 0]])

# %%
# 2. Generate a mesh with no radius
# ---------------------------------
# The resolution determines how fine the resulting mesh is an corresponds
# to the step size in the spline parameter space t.
# Setting the radius to :code:`None` or 0 results in points along the spline connected by lines.
points, connectivity = spline.mesh(resolution=0.1, radius=None)

# %%
# To visulainze the "mesh" with pyvista we have to preprend the number of points
# that are connected by a specific connection. In this case the number of points
# is two since we are simply connecing a two points with a line.
# Lastly, we have to turn the points and connectivity into a pyvista mesh using the
# :code:`PolyData` class.
connectivity = np.hstack((np.full((connectivity.shape[0], 1), 2), connectivity))
mesh = pyvista.PolyData(points, lines=connectivity)
mesh.plot()

# %%
# 3. Fixed radius
# ---------------
# Next, we will use a fixed radius to generate a surface mesh ("tube").
# We choose the Frenet-Serret frame in order to avoid having to choose an initial vector.
points, connectivity = spline.mesh(resolution=0.1, radius=0.2, frame="frenet")
# %%
# Once again, we have to prepend the number how points in each connection.
# Since, our mesh consists of triangles we choose 3.
connectivity = np.hstack((np.full((connectivity.shape[0], 1), 3), connectivity))
mesh = pyvista.PolyData(points, faces=connectivity)
mesh.plot(show_edges=True)

# %%
# 3. Eliptical cross section
# --------------------------
# In order to create, a spline with a custom cross section we can specify
# the radius as a function. The function was to take the spline parameter :code:`t`
# and the angle polar angle :code:`phi` as arguments in that order.
# For this example we create an elliptical cross section.


def elliptical_radius(t, phi):
    a = 0.1
    b = 0.05
    phi = np.deg2rad(phi)
    r = (a * b) / np.sqrt((b * np.cos(phi)) ** 2 + (a * np.sin(phi)) ** 2)
    return r


points, connectivity = spline.mesh(resolution=0.1, angular_resolution=36, radius=elliptical_radius, frame="frenet")
connectivity = np.hstack((np.full((connectivity.shape[0], 1), 3), connectivity))
mesh = pyvista.PolyData(points, faces=connectivity)
mesh.plot(show_edges=True)

# %%
# Or you can change the radius along the spline.


def radius(t, phi):
    return 0.1 + 0.03 * np.sin(t / spline.M * 16 * np.pi)


points, connectivity = spline.mesh(resolution=0.1, angular_resolution=36, radius=radius, frame="frenet")
connectivity = np.hstack((np.full((connectivity.shape[0], 1), 3), connectivity))
mesh = pyvista.PolyData(points, faces=connectivity)
mesh.plot(show_edges=True)

# %%
# 4. Bishop frame
# ---------------
# The Frenet-Serret frame is not defined on straight segments and at inflections point.
# In those cases we can use the Bishop frame instead. Another advantage of the Bishop frame
# is that it does not twist around the spline.
# Let's create a spline with some unwated twist and see how the Bishop frame fixes it.
spline = splinebox.Spline(M=M, basis_function=splinebox.B3(), closed=False)
spline.control_points = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ]
)

points, connectivity = spline.mesh(resolution=0.1, angular_resolution=36, radius=elliptical_radius, frame="frenet")
connectivity = np.hstack((np.full((connectivity.shape[0], 1), 3), connectivity))
mesh = pyvista.PolyData(points, faces=connectivity)
mesh.plot(show_edges=True)

# %%
# You can clearly see how the ellipse twists around the spline.
# The Bishop frame eliminates this twist but requires an initial orientation for the ellipse, given by an initial vector.

initial_vector = np.zeros(3)
tangent = spline.eval(0, derivative=1)
initial_vector[1] = tangent[2]
initial_vector[2] = -tangent[1]

points, connectivity = spline.mesh(
    resolution=0.1, angular_resolution=36, radius=elliptical_radius, frame="bishop", initial_vector=initial_vector
)
connectivity = np.hstack((np.full((connectivity.shape[0], 1), 3), connectivity))
mesh = pyvista.PolyData(points, faces=connectivity)
mesh.plot(show_edges=True)

# %%
# Changing the initial vector rotates the ellipse, because the initial vector is the reference for phi=0.

initial_vector = np.array([0.5, -0.5, 1])

points, connectivity = spline.mesh(
    resolution=0.1, angular_resolution=36, radius=elliptical_radius, frame="bishop", initial_vector=initial_vector
)
connectivity = np.hstack((np.full((connectivity.shape[0], 1), 3), connectivity))
mesh = pyvista.PolyData(points, faces=connectivity)
mesh.plot(show_edges=True)

# %%
# 5. Volume mesh
# --------------
# Besides a surface mesh, we can also turn the spline into a volume mesh.

points, connectivity = spline.mesh(
    radius=radius, resolution=0.5, angular_resolution=72, initial_vector=initial_vector, mesh_type="volume"
)
connectivity = np.hstack((np.full((connectivity.shape[0], 1), 4), connectivity))
cell_types = np.full(len(connectivity), fill_value=pyvista.CellType.TETRA, dtype=np.uint8)

mesh = pyvista.UnstructuredGrid(connectivity, cell_types, points)
mesh.plot(show_edges=True)

# %%
# For a better understanding of the volume mesh we can explode it.
# This allows us to see the individual tetrahedra.
mesh.explode(factor=0.5).plot(show_edges=True)

# %%
# Tips
# ----
# * To save the meshes for visulalisation in ParaView, you can use pyvista: :code:`mesh.save("mesh.vtk")`
# * By default the surface meshes are open at the end. You can close them using the :code:`cap_ends` keyword argument of :meth:`splinebox.spline_curves.Spline.mesh`.
