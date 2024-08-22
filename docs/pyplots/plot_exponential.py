import matplotlib.pyplot as plt
import numpy as np
import splinebox.basis_functions

fig, axes = plt.subplots(3, 1, sharex=True)

M = 5
basis = splinebox.basis_functions.Exponential(M=M)

t = np.linspace(-3, 3, 100)

basis_0th = basis.eval(t)
basis_1st = basis.eval(t, derivative=1)
basis_2nd = basis.eval(t, derivative=2)

fig.suptitle(f"Exponential basis function and its derivatives for $M={M}$")
axes[0].plot(t, basis_0th, label=r"$\Phi(t)$")
axes[0].legend()
axes[1].plot(t, basis_1st, label=r"$\frac{d\Phi}{dt}(t)$")
axes[1].legend()
axes[2].plot(t, basis_2nd, label=r"$\frac{d^2\Phi}{dt^2}(t)$")
axes[2].legend()
axes[2].set_xlabel(r"$t$")
plt.tight_layout()

plt.show()
