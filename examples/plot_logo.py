"""
Generate the splinebox logo
---------------------------

This example shows how the `S` of the splinebox logo is
constructed using a B3 spline.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import splinebox.basis_functions
import splinebox.spline_curves

points = np.array([[1, 2], [0, 2], [0, 1], [1, 1], [1, 0], [0, 0]])
M = len(points)
padded_points = np.concatenate([np.array([[1.5, 1.5], [1.5, 1.5]]), points, np.array([[-0.5, 0.5], [-0.5, 0.5]])])
basis_function = splinebox.basis_functions.B3()
spline = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=False, coeffs=padded_points)
ts = np.linspace(0, M - 1, 100)
vals = spline.eval(ts)

b3_max = 2 / 3
for t, val in zip(ts, vals):
    for k, point in enumerate(points):
        weight = basis_function.eval(t - k)
        color = matplotlib.cm.viridis(weight / b3_max)
        plt.plot([val[0], point[0]], [val[1], point[1]], linewidth=5 * weight, color=color)
plt.plot(points[:, 0], points[:, 1], "-o", color="gray", linewidth=7, markersize=10)
plt.plot(vals[:, 0], vals[:, 1], "-", color="red", linewidth=7)
plt.gca().axis("equal")
plt.show()
