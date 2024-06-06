import matplotlib.pyplot as plt
import numpy as np
import splinebox.basis_functions

fig, axes = plt.subplots(3, 1, sharex=True)

b3 = splinebox.basis_functions.B3()

t = np.linspace(-3, 3, 100)

b3_0th = b3.eval(t)
b3_1st = b3.eval(t, derivative=1)
b3_2nd = b3.eval(t, derivative=2)

fig.suptitle("B3 basis function and its derivatives")
axes[0].plot(t, b3_0th, label=r"$\Phi(t)$")
axes[0].legend()
axes[1].plot(t, b3_1st, label=r"$\frac{d\Phi}{dt}(t)$")
axes[1].legend()
axes[2].plot(t, b3_2nd, label=r"$\frac{d^2\Phi}{dt^2}(t)$")
axes[2].legend()
axes[2].set_xlabel(r"$t$")
plt.tight_layout()

plt.show()
