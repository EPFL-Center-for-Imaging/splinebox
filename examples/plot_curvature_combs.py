"""
Curvature combs
---------------

In this example, we plot the curvature comb of a 2D spline.
Calculating curvature requires computing the second derivative, which can be sensitive to noise when using numerical methods.
Splines are particularly useful in this context because their derivatives can be computed analytically using the chain rule, resulting in smoother and more accurate curvature calculations.
"""

import matplotlib.pyplot as plt
import numpy as np
import splinebox

# %%
# We begin by generating an arbitrary B3 spline with 7 knots.
# Any twice-differentiable basis function can be used here.

M = 7
spline = splinebox.Spline(M=M, basis_function=splinebox.B3(), closed=False)
spline.knots = np.array([[0, 0], [1, 1], [2.5, 0.5], [3.5, 1.5], [3, 3], [2, 3.5], [1, 2]])

# %%
# Next, we select finely spaced parameter values along the spline for evaluation and plotting.

t = np.linspace(0, M - 1, 1000)

# %%
# To plot the curvature comb, we need to compute the spline's values, curvature, and normal vectors at each t.
# Normal vectors are unit vectors perpendicular to the spline at each point.

vals = spline.eval(t)

curvature = spline.curvature(t)
normals = spline.normal(t)

# %%
# To construct the curvature comb, we scale the normal vectors by the curvature and a factor `d`.
# The factor `d` controls the height of the comb, allowing us to adjust its visual appearance.
# The scaled normal vectors are then added to the spline values.

max_comb_height = 1
d = max_comb_height / np.max(np.abs(curvature))

comb = vals + d * curvature[:, np.newaxis] * normals

# %%
# Finally, we plot the curvature comb. The backbone of the comb and the spline itself are plotted as lines,
# while the "teeth" of the comb are plotted individually using a loop.

plt.plot(comb[:, 1], comb[:, 0], label="Curvature comb", alpha=0.5)
for p in range(0, len(comb), 7):
    plt.plot([vals[p, 1], comb[p, 1]], [vals[p, 0], comb[p, 0]], alpha=0.5)
plt.plot(vals[:, 1], vals[:, 0], label="spline")
plt.gca().set_aspect("equal", "box")
plt.legend()
plt.show()
