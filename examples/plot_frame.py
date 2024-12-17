"""
Moving Frames
=============

`Moving frames`_ can be used to to implment a local coordinate system on a curve.
This coordinate system can them be used to interogate the space around a point on the spline,
for instance orthogonal image can be extracted from a volume as in the :ref:`Dendrite-Centric Coordinate System`.
Splinebox implements two different frames, the `Frenet-Serret`_ and the `Bishop`_ [Bishop1975]_ frame.
The Bishop frame has the advantage that it does not twist around the curve.

.. _Moving frames: https://en.wikipedia.org/wiki/Moving_frame
.. _Frenet-Serret: https://en.wikipedia.org/wiki/Frenet-Serret_formulas
.. _Bishop: https://www.jstor.org/stable/2319846
"""

import numpy as np
import pyvista
import splinebox

# %%
# Let's construct a spline and visualize the two frames.

spline = splinebox.Spline(M=4, basis_function=splinebox.B3(), closed=False)
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

t = np.linspace(0, spline.M - 1, spline.M * 3)

# %%
# First, we will compute the Frenet-Serret frame.
frenet_frame = spline.moving_frame(t, kind="frenet")
print(frenet_frame.shape)

# %%
# The returned array contains three vectors that form an orthonormal basis
# for each t.
# To compute the Bishop frame we need an initial vector that fixes the roation
# around the curve. This vector has to be orthogonal to the tangent at :code:`t[0]`.

tangent = spline.eval(t[0], derivative=1)
initial_vector = np.zeros(3)
initial_vector[1] = -tangent[2]
initial_vector[2] = tangent[1]

bishop_frame = spline.moving_frame(t, kind="bishop", initial_vector=initial_vector)

# %%
# To visualize the frames we can use a pyvista mesh.
# In the plot below it is clearly visible how the Frenet-Serret frame on the left
# twists around the curve.

spline_mesh = pyvista.MultipleLines(points=spline.eval(t))

spline_mesh["frenet0"] = frenet_frame[:, 0] * 0.2
spline_mesh["frenet1"] = frenet_frame[:, 1] * 0.2
spline_mesh["frenet2"] = frenet_frame[:, 2] * 0.2
spline_mesh["bishop0"] = bishop_frame[:, 0] * 0.2
spline_mesh["bishop1"] = bishop_frame[:, 1] * 0.2
spline_mesh["bishop2"] = bishop_frame[:, 2] * 0.2

plotter = pyvista.Plotter(shape=(1, 2), border=False)
plotter.subplot(0, 0)
spline_mesh.set_active_vectors("frenet0")
plotter.add_mesh(spline_mesh.arrows, lighting=False, color="blue")
spline_mesh.set_active_vectors("frenet1")
plotter.add_mesh(spline_mesh.arrows, lighting=False, color="red")
spline_mesh.set_active_vectors("frenet2")
plotter.add_mesh(spline_mesh.arrows, lighting=False, color="green")
plotter.add_mesh(spline_mesh, line_width=10, color="black")
plotter.subplot(0, 1)
spline_mesh.set_active_vectors("bishop0")
plotter.add_mesh(spline_mesh.arrows, lighting=False, color="blue")
spline_mesh.set_active_vectors("bishop1")
plotter.add_mesh(spline_mesh.arrows, lighting=False, color="red")
spline_mesh.set_active_vectors("bishop2")
plotter.add_mesh(spline_mesh.arrows, lighting=False, color="green")
plotter.add_mesh(spline_mesh, line_width=10, color="black", copy_mesh=True)
plotter.link_views()
plotter.show()
