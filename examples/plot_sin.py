"""
Approximate a noisy signal
--------------------------

Here we add noise to a sinusoidal signal and approximate it.
"""

import matplotlib.pyplot as plt
import numpy as np
import splinebox

# Number of control points/knots
M = 40
# Number of data points
N = 100

x = np.linspace(0, 15, N)
values = np.sin(x) + np.random.rand(N) / 3

basis_function = splinebox.B3()
spline = splinebox.Spline(M, basis_function, closed=False)
spline.fit(values)

ts = np.linspace(0, M - 1, 1000)
spline_y = spline.eval(ts)
spline_x = np.linspace(x.min(), x.max(), len(ts))

plt.scatter(x, values, color="orange")
plt.plot(spline_x, spline_y)
plt.show()
