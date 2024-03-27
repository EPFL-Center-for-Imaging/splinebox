"""
Active contours
===============

This example shows a basic active contours implementation using splinebox.
The goal is to segments the astronaut's head in the example image.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
import splinebox.basis_functions
import splinebox.spline_curves

# %%
# Let's load the astronaut example image from skimage
img = skimage.data.astronaut()

# %%
# We want our contours to stick to the edges, so we have to
# compute an edge map. To do that we first convert the image
# to gray scale, smooth it to make the edge map less noisy, and
# apply the sobel filter for edge detection.
gray = skimage.color.rgb2gray(img)
smooth = skimage.filters.gaussian(gray, 3, preserve_range=False)
edge = skimage.filters.sobel(smooth)


# %%
# To make calculating the image gradients on the edge map easier
# we use a surface spline on a regular grid.
external_energy = scipy.interpolate.RectBivariateSpline(
    np.arange(edge.shape[1]), np.arange(edge.shape[0]), edge.T, kx=2, ky=2, s=1
)


# %%
# The external force on our spline is the gradient of the edge map.
def external_force(knots):
    y = knots[:, 0]
    x = knots[:, 1]
    force_x = external_energy(x, y, dx=1, grid=False)
    force_y = external_energy(x, y, dy=1, grid=False)
    return np.stack((force_y, force_x), axis=-1)


# %%
# Let's initialize our spline with 50 knots that form a circle
# around the astronouts head.
s = np.linspace(0, 2 * np.pi, 50)
s = s[:-1]
y = 100 + 100 * np.sin(s)
x = 220 + 100 * np.cos(s)
knots = np.array([y, x]).T

# %%
# Now, we can construct a B3 spline and adjust its control points (coefficients)
# using the knots we generated above.
spline = splinebox.spline_curves.Spline(M=len(knots), basis_function=splinebox.basis_functions.B3(), closed=True)
spline.getCoefsFromKnots(knots)


# %%
# The internal force on our spline tries to keep it straight to keep its shape
# simple. This is done by minimizing its first and second derivative at the knots.
def internal_force(alpha, beta):
    t = np.arange(spline.M)
    second_derivative = spline.eval(t, derivative=2)
    fourth_derivative = 0  # spline.eval(t, derivative=4)
    return alpha * second_derivative + beta * fourth_derivative


# %%
# Here, we set the necessary paramters for active contours:
#
# * :math:`\alpha` controls the contribution of the first derivative to the internal force
# * :math:`\beta` controls the contribution of the second derivative to the internal force
# * :math:`\gamma` scales the forces
# * `max_step` in the maximum allowed step size per iteration in pixels
alpha = 0.0007
beta = 0
gamma = 300
max_step = 2

# %%
# We keep a copy of the initial knots so we can plot them later.
initial_knots = knots.copy()

# %%
# The active contours approach consists of iteratively updating our knots and control points according
# to the internal and external forces on the knots.
for _i in range(2000):
    knots = spline.getKnotsFromCoefs()
    delta = max_step * np.tanh(gamma * (internal_force(alpha, beta) + external_force(knots)))
    new_knots = knots + delta
    spline.getCoefsFromKnots(new_knots)

# %%
# Inorder to plot the spline as a smooth line, we have to evaluate it
# more densly than just at the knots.
samples = spline.eval(np.linspace(0, len(knots), 400))

# %%
# Finaly, we can plot the result.
plt.imshow(img)
plt.scatter(initial_knots[:, 1], initial_knots[:, 0], marker="x", color="black", label="initial knots")
plt.scatter(knots[:, 1], knots[:, 0], marker="o", color="blue", label="final knots")
plt.plot(samples[:, 1], samples[:, 0], label="spline")
plt.legend()
plt.show()
