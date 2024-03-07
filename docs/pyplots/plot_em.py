import matplotlib.pyplot as plt
import numpy as np
import splinebox.basis_functions

fig, axes = plt.subplots(3, 1, sharex=True)

M = 5
alpha = 0.5
em = splinebox.basis_functions.EM(M=M, alpha=alpha)

x = np.linspace(-3, 3, 100)

em_0th = em.eval(x)
em_1st = em.eval_1st_derivative(x)
em_2nd = em.eval_2nd_derivative(x)

fig.suptitle(f"EM basis function and its derivatives for $M={M}$, $\\alpha={alpha}$")
axes[0].plot(x, em_0th, label=r"$f(x)$")
axes[0].legend()
axes[1].plot(x, em_1st, label=r"$\frac{df}{dx}(x)$")
axes[1].legend()
axes[2].plot(x, em_2nd, label=r"$\frac{d^2f}{dx^2}(x)$")
axes[2].legend()
axes[2].set_xlabel(r"$x$")
plt.tight_layout()

plt.show()
