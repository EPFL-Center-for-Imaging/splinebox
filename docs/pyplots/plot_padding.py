import matplotlib.pyplot as plt
import numpy as np
import splinebox

spline = splinebox.Spline(M=10, basis_function=splinebox.B3(), closed=False)

delta = np.pi / spline.M
theta = np.linspace(-delta, np.pi + delta, spline.M + 2)
control_points = np.stack([np.cos(theta), np.sin(theta)], axis=-1)

spline.control_points = control_points

t = np.linspace(0, spline.M - 1, 100)
vals = spline.eval(t)

plt.figure(figsize=(6, 3))
plt.scatter(spline.control_points[1:-1, 0], spline.control_points[1:-1, 1], label="control points")
plt.plot(vals[:, 0], vals[:, 1], label="spline")
plt.gca().set_aspect("equal", "box")
plt.legend()
plt.tight_layout()
plt.show()
