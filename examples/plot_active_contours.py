"""
Active Contours
===============

This example demonstrates a basic active contours (snakes) implementation using splinebox. The goal is to segment the astronaut's head in an image by iteratively refining a spline contour.
"""

# sphinx_gallery_thumbnail_number = 4

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
import splinebox.basis_functions
import splinebox.spline_curves

# %%
# Let's load the astronaut example image from skimage.
img = skimage.data.astronaut()

plt.imshow(img)
plt.show()

# %%
# To make the contour stick to the edges of the image, we compute an edge map.
# First, convert the image to grayscale, apply Gaussian smoothing to reduce noise, and then apply the Sobel filter for edge detection.
gray = skimage.color.rgb2gray(img)
smooth = skimage.filters.gaussian(gray, 3, preserve_range=False)
edge = skimage.filters.sobel(smooth)

plt.imshow(edge)
plt.show()

# %%
# The control points of our spline will be updated based on the edge image.
# In order to decide in which direction to move the control points, we will used the spatial derivatives (gradients)
# of the edge image.
SIGMA = 3
edge_gy = scipy.ndimage.gaussian_filter(edge, SIGMA, order=[1, 0], mode="nearest", output=float)
edge_gx = scipy.ndimage.gaussian_filter(edge, SIGMA, order=[0, 1], mode="nearest", output=float)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(edge_gy)
axes[1].imshow(edge_gx)
plt.show()

# %%
# Initialize the spline with 30 knots that form a circle around the astronaut's head.
M = 30
s = np.linspace(0, 2 * np.pi, M + 1)[:-1]
y = 100 + 100 * np.sin(s)
x = 220 + 100 * np.cos(s)
knots = np.array([y, x]).T

# %%
# We keep a copy of the initial knots so we can plot them later.
initial_knots = knots.copy()

# %%
# Now, we can construct a B3 spline and set its control points (coefficients)
# using the knots we generated above.
spline = splinebox.spline_curves.Spline(M=M, basis_function=splinebox.basis_functions.B3(), closed=True)
spline.knots = knots

t = np.linspace(0, M, 400)
contour = spline(t)
plt.imshow(img)
plt.scatter(knots[:, 1], knots[:, 0])
plt.plot(contour[:, 1], contour[:, 0])
plt.show()


# %%
# Next, we will fit the spline around the head using the active contour approach.
# The control points are updated in every iteration minimising the energy function.
# The energy function consist of two terms:
#   (1) Image energy: measures the pixel values below the spline.
#   (2) Internal energy: penalises curvy spline
# A detailed description of the math can be found in :ref:`theory/active_contour`.

# Store intermediate contours
contours = []

t = np.linspace(0, spline.M, 3000)[:-1]

for _ in range(1000):
    contour = spline(t)
    contours.append(contour.copy())

    dy = scipy.ndimage.map_coordinates(edge_gy, contour.T, order=1)
    dx = scipy.ndimage.map_coordinates(edge_gx, contour.T, order=1)
    img_gradients = np.stack([dy, dx], axis=-1)

    partial_derivs = spline.control_points_derivatives(t)

    image_energy_gradients = partial_derivs.T @ img_gradients

    internal_energy_gradients = np.mean(spline.control_points_derivatives_of_norm_squared(t, derivative=2), axis=0)

    gradients = -image_energy_gradients + internal_energy_gradients / 2

    # Take a step toward minimizing the energy
    spline.control_points = spline.control_points - gradients

# %%
# In order to plot the spline as a smooth line, we have to evaluate it
# more densely than just at the knots.
samples = spline(t)

final_knots = spline.knots

# %%
# Finally, we can plot the result.
plt.imshow(img)
plt.scatter(initial_knots[:, 1], initial_knots[:, 0], marker="x", color="black", label="initial knots")
contours = contours[::20]
colors = matplotlib.colormaps["viridis"](np.linspace(0, 1, len(contours)))
for contour, color in zip(contours, colors):
    plt.plot(contour[:, 1], contour[:, 0], color=color, alpha=0.2)
plt.scatter(final_knots[:, 1], final_knots[:, 0], marker="o", label="final final_knots")
plt.plot(samples[:, 1], samples[:, 0], label="spline")
plt.legend()
plt.show()
