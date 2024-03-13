import matplotlib.pyplot as plt
import numpy as np
import splinebox.basis_functions

fig, axes = plt.subplots(3, 1, sharex=True)

b3 = splinebox.basis_functions.B3()

x = np.linspace(-3, 3, 100)

b3_0th = b3.eval(x)
b3_1st = b3.eval(x, derivative=1)
b3_2nd = b3.eval(x, derivative=2)

fig.suptitle("B3 basis function and its derivatives")
axes[0].plot(x, b3_0th, label=r"$f(x)$")
axes[0].legend()
axes[1].plot(x, b3_1st, label=r"$\frac{df}{dx}(x)$")
axes[1].legend()
axes[2].plot(x, b3_2nd, label=r"$\frac{d^2f}{dx^2}(x)$")
axes[2].legend()
axes[2].set_xlabel(r"$x$")
plt.tight_layout()

plt.show()
