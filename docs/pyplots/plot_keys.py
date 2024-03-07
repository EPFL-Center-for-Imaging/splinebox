import matplotlib.pyplot as plt
import numpy as np
import splinebox.basis_functions

fig, axes = plt.subplots(3, 1, sharex=True)

keys = splinebox.basis_functions.Keys()

x = np.linspace(-3, 3, 100)

keys_0th = keys.eval(x)
keys_1st = keys.eval_1st_derivative(x)
keys_2nd = keys.eval_2nd_derivative(x)

fig.suptitle("Keys basis function and its derivatives")
axes[0].plot(x, keys_0th, label=r"$f(x)$")
axes[0].legend()
axes[1].plot(x, keys_1st, label=r"$\frac{df}{dx}(x)$")
axes[1].legend()
axes[2].plot(x, keys_2nd, label=r"$\frac{d^2f}{dx^2}(x)$")
axes[2].legend()
axes[2].set_xlabel(r"$x$")
plt.tight_layout()

plt.show()
