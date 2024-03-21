"""
Closed interpolating splines
============================

This example shows different closed interpolating splines.
"""

import matplotlib.pyplot as plt
import numpy as np
import splinebox.basis_functions
import splinebox.spline_curves

n, m = 2, 2
fig, axes = plt.subplots(n, m, sharex=True, sharey=True)

# Some pixel coordinates to interpolate
coordinates = np.array([[0, 100], [50, 100], [100, 50], [100, 0], [100, -100], [50, -50], [0, -100], [-100, 0]])

for i in range(n):
    for j in range(m):
        axes[i, j].scatter(coordinates[:, 0] - 1, coordinates[:, 1])

# The parameter values at which the spline is evaluated
x = np.linspace(0, len(coordinates), 1000)

for i, (name, basis_function) in enumerate(
    (
        ("B1", splinebox.basis_functions.B1()),
        ("B3", splinebox.basis_functions.B3()),
        # ("Exponential", splinebox.basis_functions.Exponential(len(coordinates), 2.0 * math.pi / len(coordinates))),
        ("CatmullRom", splinebox.basis_functions.CatmullRom()),
    )
):
    curve = splinebox.spline_curves.Spline(len(coordinates), basis_function, True)
    curve.getCoefsFromKnots(coordinates)
    discreteContour_original = curve.original_eval(x)
    discreteContour = curve.eval(x)
    discreteContour_jit = curve.eval_jit(x)
    discreteContour_jit_parallel = curve.eval_jit_parallel(x)
    discreteContour_jit2 = curve.eval_jit2(x)
    discreteContour_jit2_parallel = curve.eval_jit2_parallel(x)
    discreteContour_jit_no_vectorize = curve.eval_jit_no_vectorize(x)
    discreteContour_jit_parallel_no_vectorize = curve.eval_jit_parallel_no_vectorize(x)
    discreteContour_jit2_no_vectorize = curve.eval_jit2_no_vectorize(x)
    discreteContour_jit2_parallel_no_vectorize = curve.eval_jit2_parallel_no_vectorize(x)

    axes[i // n, i % n].plot(discreteContour[:, 0], discreteContour[:, 1])
    axes[i // n, i % n].plot(discreteContour_jit[:, 0], discreteContour_jit[:, 1])
    axes[i // n, i % n].plot(discreteContour_jit_parallel[:, 0], discreteContour_jit_parallel[:, 1])
    axes[i // n, i % n].plot(discreteContour_jit2[:, 0], discreteContour_jit2[:, 1])
    axes[i // n, i % n].plot(discreteContour_jit2_parallel[:, 0], discreteContour_jit2_parallel[:, 1])
    axes[i // n, i % n].plot(discreteContour_original[:, 0], discreteContour_original[:, 1])

    axes[i // n, i % n].plot(discreteContour_jit_no_vectorize[:, 0], discreteContour_jit_no_vectorize[:, 1])
    axes[i // n, i % n].plot(
        discreteContour_jit_parallel_no_vectorize[:, 0], discreteContour_jit_parallel_no_vectorize[:, 1]
    )
    axes[i // n, i % n].plot(discreteContour_jit2_no_vectorize[:, 0], discreteContour_jit2_no_vectorize[:, 1])
    axes[i // n, i % n].plot(
        discreteContour_jit2_parallel_no_vectorize[:, 0], discreteContour_jit2_parallel_no_vectorize[:, 1]
    )

    axes[i // n, i % n].set_title(name)
    axes[i // n, i % n].set_aspect("equal", adjustable="box")

plt.savefig("example1.pdf")
plt.show()
