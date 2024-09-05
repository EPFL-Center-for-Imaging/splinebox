"""
Tracking the Undulating Motion of C. elegans
============================================
This example demonstrates how to track the characteristic undulating pattern of a moving C. elegans using splinebox.

Data Source: `WormSwin <https://zenodo.org/records/7456803>`_

Data Used: csb-1_dataset 24_2_1_1

Preprocessing:

* Background: Maximum projections of frames 75-187 (to account for minor movements).
* Background Subtraction: Subtracted from each frame.
* Cropping: Frames 141-187, focusing on a single worm centered at the bottom.
* Image Adjustment: Rotation, contrast adjustment, and conversion to 8-bit.
"""

# sphinx_gallery_thumbnail_number = 3

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import skimage
import splinebox

# %%
# Load the data and inspect it
# ----------------------------

stack = skimage.io.imread("celegans.tif")


def _update0(i):
    mpl_img.set_array(stack[i])
    return (mpl_img,)


fig, ax = plt.subplots(figsize=(7, 3))
mpl_img = ax.imshow(stack[0], cmap="Greys_r")
ax.set(xlim=(0, stack.shape[2]), ylim=(0, stack.shape[1]))
animation = matplotlib.animation.FuncAnimation(fig, _update0, len(stack), interval=100, blit=True)
plt.show()

# %%
# Segmentation of the Worm
# ------------------------
# Next, we segment the worm using Otsu's thresholding method to prepare for spline fitting.

thresh = skimage.filters.threshold_otsu(stack)
mask = stack < thresh


def _update1(i):
    mpl_img.set_array(stack[i])
    mpl_mask.set_array(mask[i])
    return (mpl_img, mpl_mask)


fig, ax = plt.subplots(figsize=(7, 3))
mpl_img = ax.imshow(stack[0], cmap="Greys_r")
mpl_mask = ax.imshow(mask[0], cmap="Reds", alpha=0.5)
ax.set(xlim=(0, stack.shape[2]), ylim=(0, stack.shape[1]))
animation = matplotlib.animation.FuncAnimation(fig, _update1, len(stack), interval=100, blit=True)
plt.show()

# %%
# Fitting Splines to the Worm's Shape
# -----------------------------------
# We fit a spline to the worm in each frame. The spline is created by skeletonizing the worm and then ordering the skeleton points based on their x-position. For simplicity, this method assumes that the worm is moving from left to right. In more complex scenarios, a graph-based approach (e.g., using the skan package) would be more appropriate for ordering the points.
# We use seven knots in our spline, balancing accuracy and robustness against segmentation imperfections. An exponential basis function is selected for the spline to enable the generation of trigonometric functions.

M = 7
basis_function = splinebox.basis_functions.Exponential(M)

splines = []
for i in range(len(mask)):
    label_img = skimage.measure.label(mask[i])
    label_biggest = np.argmax(np.bincount(label_img.flatten())[1:]) + 1
    mask[i] = label_img == label_biggest
    mask[i] = skimage.morphology.binary_opening(mask[i])
    mask[i] = skimage.morphology.remove_small_holes(mask[i], area_threshold=7)
    skeleton = skimage.morphology.skeletonize(mask[i])
    spline = splinebox.spline_curves.Spline(M=M, basis_function=basis_function, closed=False)
    coords = np.stack(np.where(skeleton), axis=-1)
    coords = coords[np.argsort(coords[:, 1])]
    spline.fit(coords)
    splines.append(spline)

t = np.linspace(0, M - 1, 500)


def _update2(i):
    mpl_img.set_array(stack[i])
    vals = splines[i].eval(t)
    mpl_line.set_data(vals[:, 1], vals[:, 0])
    return (mpl_img, mpl_line)


fig, ax = plt.subplots(figsize=(7, 3))
mpl_img = ax.imshow(stack[0], cmap="Greys_r")
vals = splines[0].eval(t)
(mpl_line,) = ax.plot(vals[:, 1], vals[:, 0])
ax.set(xlim=(0, stack.shape[2]), ylim=(0, stack.shape[1]))
animation = matplotlib.animation.FuncAnimation(fig, _update2, len(stack), interval=100, blit=True)
plt.show()

# %%
# Analyzing the Undulating Motion
# -------------------------------
# To focus on the undulating motion, we translate the splines to their center of mass and rotate them to align horizontally.

for spline in splines:
    start_point = np.squeeze(spline.eval(0))
    stop_point = np.squeeze(spline.eval(spline.M - 1))
    angle = np.arctan2(*(stop_point - start_point))
    c = np.cos(angle)
    s = np.sin(angle)
    rot_matrix = np.array([[c, -s], [s, c]])
    spline.rotate(rot_matrix)
    com = np.mean(spline.eval(t), axis=0)
    spline.translate(-com)

# %%
# Finally, we visualize the isolated undulating movement and its associated curvature.


def _update3(i):
    vals = splines[i].eval(t)
    curvature = splines[i].curvature(t)
    normals = splines[i].normal(t)
    comb = vals + d * curvature[:, np.newaxis] * normals
    for p, mpl_tooth in zip(ps, mpl_teeth):
        mpl_tooth.set_data([[vals[p, 1], comb[p, 1]], [vals[p, 0], comb[p, 0]]])
    mpl_curvature.set_data(comb[:, 1], comb[:, 0])
    mpl_line.set_data(vals[:, 1], vals[:, 0])
    return (*mpl_teeth, mpl_curvature, mpl_line)


fig, ax = plt.subplots(figsize=(7, 3))
vals = splines[0].eval(t)
curvature = splines[0].curvature(t)
normals = splines[0].normal(t)
d = 50
comb = vals + d * curvature[:, np.newaxis] * normals
mpl_teeth = []
ps = np.arange(0, len(comb), 10)
for p in ps:
    mpl_teeth.extend(ax.plot([vals[p, 1], comb[p, 1]], [vals[p, 0], comb[p, 0]], color="#568b22"))
(mpl_curvature,) = ax.plot(comb[:, 1], comb[:, 0])
(mpl_line,) = ax.plot(vals[:, 1], vals[:, 0])
ax.set(xlim=(-60, 60), ylim=(-25, 25))
animation = matplotlib.animation.FuncAnimation(fig, _update3, len(stack), interval=100, blit=True)
plt.show()
