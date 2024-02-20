import math

import matplotlib.pyplot as plt
import numpy as np
import splinebox.basis as basis
import splinebox.curves as scm

# Some pixel coordinates to interpolate
coordinates = np.array([[0, 100], [100, 0], [0, -100], [-100, 0]])

# A Sampling rate to represent the continuous spline
samplingRate = 100

# Linear B-spline interpolation
curve = scm.SplineCurve(len(coordinates), basis.B1(), True)
curve.getCoefsFromKnots(coordinates)
discreteContour = curve.sample(samplingRate)

plt.scatter(discreteContour[:, 0], discreteContour[:, 1])
plt.show()

# Cubic B-spline interpolation
curve = scm.SplineCurve(len(coordinates), basis.B3(), True)
curve.getCoefsFromKnots(coordinates)
discreteContour = curve.sample(samplingRate)

plt.scatter(discreteContour[:, 0], discreteContour[:, 1])
plt.show()

# Exponential (circular) Spline interpolation
curve = scm.SplineCurve(
    len(coordinates),
    basis.EM(len(coordinates), 2.0 * math.pi / len(coordinates)),
    True,
)
curve.getCoefsFromKnots(coordinates)
discreteContour = curve.sample(samplingRate)

plt.scatter(discreteContour[:, 0], discreteContour[:, 1])
plt.show()
