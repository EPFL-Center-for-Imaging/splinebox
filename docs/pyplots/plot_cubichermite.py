import matplotlib.pyplot as plt
import numpy as np
import splinebox.basis_functions

fig, axes = plt.subplots(2, 2, sharex=True)

basis = splinebox.basis_functions.CubicHermite()

t = np.linspace(-3, 3, 100)

basis_0th = basis.eval(t)
basis_1st = basis.eval(t, derivative=1)

fig.suptitle("Cubic Hermite basis function and its derivatives")
axes[0][0].plot(t, basis_0th[0], label=r"$\Phi_1(t)$")
axes[0][0].legend()
axes[1][0].plot(t, basis_1st[0], label=r"$\frac{d\Phi_1}{dt}(t)$")
axes[1][0].legend()
axes[1][0].set_xlabel(r"$t$")
axes[0][1].plot(t, basis_0th[1], label=r"$\Phi_2(t)$")
axes[0][1].legend()
axes[1][1].plot(t, basis_1st[1], label=r"$\frac{d\Phi_2}{dt}(t)$")
axes[1][1].legend()
axes[1][1].set_xlabel(r"$t$")
plt.tight_layout()

plt.show()
