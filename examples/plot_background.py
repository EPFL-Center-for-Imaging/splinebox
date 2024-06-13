"""
Generate the splinebox documentation background
-----------------------------------------------

This example shows how the background of the splinebox documentation is made.
"""

import matplotlib.pyplot as plt
import numpy as np
import splinebox
import splinebox.basis_functions
import splinebox.spline_curves

knots = np.array([[0, 1], [0.1, 0.9], [0.2, 0.8], [0.6, 0.7], [0.35, 0.4], [0.8, 0.2], [0.9, 0.1], [1, 0]])
M = len(knots)

spline = splinebox.Spline(M=M, basis_function=splinebox.B3(), closed=False)
spline.knots = knots

ts = np.linspace(0, M - 1, 100)

fig = plt.figure(frameon=False)
fig.set_size_inches(2, 2)
ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
ax.axis("equal")
ax.set_axis_off()
fig.add_axes(ax)
control_points = np.clip(spline.control_points, 0, 1)
# for the actual background alpha=0.07 was used
alpha = 1
ax.plot(
    control_points[1:-1, 0],
    control_points[1:-1, 1],
    "-o",
    color="forestgreen",
    linewidth=1,
    markersize=1,
    alpha=alpha,
)
vals = spline.eval(ts)
ax.plot(vals[:, 0], vals[:, 1], "-", color="forestgreen", linewidth=1, alpha=alpha)
plt.show()
