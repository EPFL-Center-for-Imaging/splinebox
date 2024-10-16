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

# %%
# You can also save multiple splines in a single json file.

splines = []
for _ in range(3):
    spline = splinebox.spline_curves.Spline(
        M=4, basis_function=splinebox.basis_functions.B3(), closed=True, control_points=np.random.rand(4, 1)
    )
    splines.append(spline)

splinebox.spline_curves.splines_to_json("splines.json", splines)

# %%
# Here is what a json file with multiple splines looks like:

with open("splines.json") as f:
    sys.stdout.write(f.read())

# %%
# Lastly, we load multiple splines from a single json file.

splines = splinebox.spline_curves.splines_from_json("splines.json")
print(splines)
