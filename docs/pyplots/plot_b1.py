import matplotlib.pyplot as plt
import numpy as np
import splinebox.basis

fig, axes = plt.subplots(2, 1, sharex=True)

b1 = splinebox.basis.B1()

x = np.linspace(-2, 2, 100)

b1_0th = []
b1_1st = []

for xx in x:
    b1_0th.append(b1.eval(xx))
    b1_1st.append(b1.eval_1st_derivative(xx))

fig.suptitle("B1 basis function and its derivatives")
axes[0].plot(x, b1_0th, label=r"$f(x)$")
axes[0].legend()
axes[1].plot(x, b1_1st, label=r"$\frac{df}{dx}(x)$")
axes[1].legend()
axes[1].set_xlabel(r"$x$")
plt.tight_layout()

plt.show()
