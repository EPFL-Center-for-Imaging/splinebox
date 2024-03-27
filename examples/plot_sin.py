"""
Approximate a noisy signal
--------------------------

Here we add noise to a sinusoidal signal and approximate it.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import splinebox

M = 50
values = np.sin(np.linspace(0, 15, M)) + np.random.rand(M) / 3
x = np.linspace(0, M - 1, 1000)

basis_function = splinebox.B3()
spline = splinebox.Spline(M, basis_function, closed=False)

pad_width = math.ceil(basis_function.support / 2)
spline.coeffs = np.pad(values, pad_width=pad_width, mode="edge")
y = spline.eval(x)
plt.scatter(np.arange(M), values, color="orange")
plt.plot(x, y)
plt.show()
