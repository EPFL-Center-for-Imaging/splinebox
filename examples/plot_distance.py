"""
Distance between splines
------------------------

In this example, we compute the euclidean distance between two splines.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
import splinebox

# %%
# For simplicity we create a function that can plot the two splines and the
# distance between the splines at parameter values t_min and s_min.


def plot_splines(spline1, spline2, t_min=None, s_min=None):
    vals1 = spline1(np.linspace(0, spline1.M, 1000))
    vals2 = spline2(np.linspace(0, spline2.M, 1000))
    knots1 = spline1.knots
    knots2 = spline2.knots

    plt.plot(vals1[:, 1], vals1[:, 0])
    plt.plot(vals2[:, 1], vals2[:, 0])
    plt.scatter(knots1[:, 1], knots1[:, 0])
    plt.scatter(knots2[:, 1], knots2[:, 0])

    if t_min is not None and s_min is not None:
        point1 = spline1(t_min)
        point2 = spline2(s_min)
        plt.plot([point1[1], point2[1]], [point1[0], point2[0]], color="k", linestyle="--")

    plt.gca().set_aspect("equal", "box")

    plt.show()


# %%
# We start by constructing two arbitrary closed splines.

basis_function = splinebox.basis_functions.B3()
M = 5
spline1 = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=True)
spline2 = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=True)

spline1.control_points = np.array(
    [
        [1, 2],
        [2, 2],
        [3, 2.5],
        [2.2, 3],
        [1.3, 2.2],
    ]
)

spline2.control_points = np.array(
    [
        [2, -2],
        [2.5, -1],
        [2, -1.5],
        [1.5, -1],
        [1, -2],
    ]
)

# %%
# Plot the splines
plot_splines(spline1, spline2)

# %%
# To get an initial guess of the spline parameter pair with the
# smallest distance, we perform a brute force search.
# Note: This can be made more accurate by increasing the number
# of parameters we interogate.

t = np.linspace(0, spline1.M, 5)
s = np.linspace(0, spline2.M, 5)
vals1 = spline1(t)
vals2 = spline2(s)
distance_vectors = vals1[:, np.newaxis] - vals2[np.newaxis, :]
distances = np.linalg.norm(distance_vectors, axis=-1)
indices = np.unravel_index(np.argmin(distances), distances.shape)
t_min = t[indices[0]]
s_min = s[indices[1]]

plot_splines(spline1, spline2, t_min, s_min)

# %%
# To further refine the estimate we can
# run an optimization.


def distance(parameters):
    val1 = spline1(parameters[0])
    val2 = spline2(parameters[1])
    return np.linalg.norm(val1 - val2)


result = scipy.optimize.minimize(distance, np.array([t_min, s_min]), bounds=((0, spline1.M), (0, spline2.M)))
t_min, s_min = result.x

plot_splines(spline1, spline2, t_min, s_min)
