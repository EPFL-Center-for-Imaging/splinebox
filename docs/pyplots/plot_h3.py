import matplotlib.pyplot as plt
import numpy as np
import splinebox.basis_functions

fig, axes = plt.subplots(2, 2, sharex=True)

h3 = splinebox.basis_functions.H3()

x = np.linspace(-3, 3, 100)

h3_0th = h3.eval(x)
h3_1st = h3.eval_1st_derivative(x)

fig.suptitle("H3 basis function and its derivatives")
axes[0][0].plot(x, h3_0th[0], label=r"$f_1(x)$")
axes[0][0].legend()
axes[1][0].plot(x, h3_1st[0], label=r"$\frac{df_1}{dx}(x)$")
axes[1][0].legend()
axes[1][0].set_xlabel(r"$x$")
axes[0][1].plot(x, h3_0th[1], label=r"$f_2(x)$")
axes[0][1].legend()
axes[1][1].plot(x, h3_1st[1], label=r"$\frac{df_2}{dx}(x)$")
axes[1][1].legend()
axes[1][1].set_xlabel(r"$x$")
plt.tight_layout()

plt.show()
