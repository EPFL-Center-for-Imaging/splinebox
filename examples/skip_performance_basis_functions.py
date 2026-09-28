import collections
import time

import jax.numpy as jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import splinebox
import torch

results = collections.defaultdict(list)

for length in [10, 100, 1000, 10000, 100000]:
    for replicate in np.arange(11):
        x = np.random.rand(length)
        for array_namespace in ["numpy", "torch", "jax"]:
            if array_namespace == "torch":
                x = torch.tensor(x)
            elif array_namespace == "jax":
                x = jax.asarray(x)
            for basis_function_name, basis_function_class in splinebox.basis_functions.inventory().items():
                if "Exponential" in basis_function_name:
                    basis_function = basis_function_class(5)
                else:
                    basis_function = basis_function_class()
                for deriv in range(3):
                    for jit in [True, False]:
                        start = time.perf_counter_ns()
                        basis_function(x, derivative=deriv, jit=jit)
                        stop = time.perf_counter_ns()

                        results["length"].append(length)
                        results["Replicate"].append(replicate)
                        # results["Array namespace"].append(array_namespace)
                        results["Basis function"].append(basis_function_name)
                        results["Derivative"].append(deriv)
                        results["JIT"].append(jit)
                        results["time [ns]"].append(stop - start)

df = pd.DataFrame(results)

g = sns.relplot(
    data=df.loc[df["Replicate"] > 0],
    x="length",
    y="time [ns]",
    col="Basis function",
    hue="JIT",
    row="Array namespace",
    kind="line",
)
g.set(xscale="log")
g.set(yscale="log")
plt.show()
