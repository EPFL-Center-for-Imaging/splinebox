import matplotlib.pyplot as plt
import numpy as np
import splinebox.basis_functions

fig, axes = plt.subplots(3, 1, sharex=True)

M = 5
alpha = 0.5
basis = splinebox.basis_functions.Exponential(M=M, alpha=alpha)

x = np.linspace(-3, 3, 100)

basis_0th = basis.eval(x)
basis_1st = basis.eval_1st_derivative(x)
basis_2nd = basis.eval_2nd_derivative(x)

fig.suptitle(f"EM basis function and its derivatives for $M={M}$, $\\alpha={alpha}$")
axes[0].plot(x, basis_0th, label=r"$f(x)$")
axes[0].legend()
axes[1].plot(x, basis_1st, label=r"$\frac{df}{dx}(x)$")
axes[1].legend()
axes[2].plot(x, basis_2nd, label=r"$\frac{d^2f}{dx^2}(x)$")
axes[2].legend()
axes[2].set_xlabel(r"$x$")
plt.tight_layout()

plt.show()
