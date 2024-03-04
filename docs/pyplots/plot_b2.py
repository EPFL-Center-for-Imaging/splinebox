import matplotlib.pyplot as plt
import numpy as np
import splinebox.basis

fig, axes = plt.subplots(3, 1, sharex=True)

b2 = splinebox.basis.B2()

x = np.linspace(-2, 2, 100)

b2_0th = []
b2_1st = []
b2_2nd = []

for xx in x:
    b2_0th.append(b2.eval(xx))
    b2_1st.append(b2.eval_1st_derivative(xx))
    b2_2nd.append(b2.eval_2nd_derivative(xx))

fig.suptitle("B2 basis function and its derivatives")
axes[0].plot(x, b2_0th, label=r"$f(x)$")
axes[0].legend()
axes[1].plot(x, b2_1st, label=r"$\frac{df}{dx}(x)$")
axes[1].legend()
axes[2].plot(x, b2_2nd, label=r"$\frac{d^2f}{dx^2}(x)$")
axes[2].legend()
axes[2].set_xlabel(r"$x$")
plt.tight_layout()

plt.show()
