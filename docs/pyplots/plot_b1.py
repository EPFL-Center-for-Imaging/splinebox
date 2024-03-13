import matplotlib.pyplot as plt
import numpy as np
import splinebox.basis_functions

fig, axes = plt.subplots(2, 1, sharex=True)

b1 = splinebox.basis_functions.B1()

x = np.linspace(-2, 2, 100)

b1_0th = b1.eval(x)
b1_1st = b1.eval(x, derivative=1)

fig.suptitle("B1 basis function and its derivatives")
axes[0].plot(x, b1_0th, label=r"$f(x)$")
axes[0].legend()
axes[1].plot(x, b1_1st, label=r"$\frac{df}{dx}(x)$")
axes[1].legend()
axes[1].set_xlabel(r"$x$")
plt.tight_layout()

plt.show()
