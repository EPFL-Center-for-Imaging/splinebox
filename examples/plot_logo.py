"""
Generate the splinebox logo
---------------------------

This example shows how the `S` of the splinebox logo is
constructed using a B3 spline.
"""

import matplotlib.pyplot as plt
import numpy as np
import splinebox.basis_functions
import splinebox.spline_curves

points = np.array([[1.5, 1.5], [1, 2], [0, 2], [0, 1], [1, 1], [1, 0], [0, 0], [-0.5, 0.5]])
points = points[:, ::-1]
M = len(points)
padded_points = np.concatenate(
    [
        np.array([[1.5, 1.5]]),
        points,
        np.array([[0.5, -0.5]]),
    ]
)
basis_function = splinebox.basis_functions.B3()
spline = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=False, control_points=padded_points)

t = np.linspace(0.8, M - 1.5, 100)

vals = spline.eval(t)
curvature = spline.curvature(t)
normals = spline.normal(t)

# You can limit the max height of the curvature comb using the factor d.
max_comb_height = 0.3
d = max_comb_height / np.max(np.abs(curvature))

comb = vals + d * curvature[:, np.newaxis] * normals

# color1 = "#f5bc47"
color1 = "#568b22"
color2 = "#568b22"
linewidth1 = 3
linewidth2 = 6

plt.plot(comb[:, 1], comb[:, 0], label="Curvature comb", color=color1, linewidth=linewidth1)
for p in range(len(comb)):
    plt.plot([vals[p, 1], comb[p, 1]], [vals[p, 0], comb[p, 0]], color=color1, linewidth=linewidth1)
plt.plot(vals[:, 1], vals[:, 0], linewidth=linewidth2, color=color2)
plt.gca().set_aspect("equal", "box")
plt.savefig("box.pdf")
plt.show()
