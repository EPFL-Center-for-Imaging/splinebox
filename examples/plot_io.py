"""
Saving and loading splines
--------------------------
"""

import sys

import numpy as np
import splinebox.basis_functions
import splinebox.spline_curves

# %%
# We start by creating a random spline.

spline = splinebox.spline_curves.Spline(
    M=5, basis_function=splinebox.basis_functions.B3(), closed=True, control_points=np.random.rand(5, 3)
)

# %%
# Let's save the spline:

spline.to_json("spline.json")

# %%
# Here is what the json file looks like:

with open("spline.json") as f:
    sys.stdout.write(f.read())

# %%
# Next, we will create a new spline based on the json file.

loaded_spline = splinebox.spline_curves.Spline.from_json("spline.json")
