import matplotlib.pyplot as plt
import numpy as np
import splinebox.basis_functions

fig, axes = plt.subplots(3, 1, sharex=True)

b2 = splinebox.basis_functions.B2()

t = np.linspace(-2, 2, 100)

b2_0th = b2.eval(t)
b2_1st = b2.eval(t, derivative=1)
b2_2nd = b2.eval(t, derivative=2)


fig.suptitle("B2 basis function and its derivatives")
axes[0].plot(t, b2_0th, label=r"$\Phi(t)$")
axes[0].legend()
axes[1].plot(t, b2_1st, label=r"$\frac{d\Phi}{dt}(t)$")
axes[1].legend()
axes[2].plot(t, b2_2nd, label=r"$\frac{d^2\Phi}{dt^2}(t)$")
axes[2].legend()
axes[2].set_xlabel(r"$t$")
plt.tight_layout()

plt.show()
