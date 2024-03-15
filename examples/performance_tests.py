import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import splinebox

df = pd.DataFrame(columns=["# knots", "# samples", "Basis function", "implementation", "time [s]"])

for M in [4, 10, 100, 1000]:
    knots = np.sin(np.linspace(0, 15, M)) + np.random.rand(M) / 5
    for bf_name, basis_function in [
        ("B1", splinebox.B1()),
        ("B3", splinebox.B3()),
        ("CatmullRom", splinebox.CatmullRom()),
    ]:  # ("Exponential", splinebox.Exponential(M, 2 * np.pi / M)),]:
        spline = splinebox.Spline(M, basis_function, closed=False)
        spline.getCoefsFromKnots(knots)
        for n_samples in [10, 100, 1000, 10000, 100000]:
            x = np.linspace(0, M - 1, n_samples)
            for replica in range(6):
                print(f"M={M} {bf_name} #samples {n_samples} replica {replica}")
                print("eval")
                start = time.process_time()
                y = spline.eval(x)
                dt = time.process_time() - start
                current_df_eval = pd.DataFrame(
                    {
                        "# knots": [
                            M,
                        ],
                        "# samples": [
                            n_samples,
                        ],
                        "Basis function": [
                            bf_name,
                        ],
                        "implementation": "eval",
                        "time [s]": dt,
                    }
                )

                print("eval_jit")
                start = time.process_time()
                y = spline.eval_jit(x)
                dt = time.process_time() - start
                current_df_eval_jit = pd.DataFrame(
                    {
                        "# knots": [
                            M,
                        ],
                        "# samples": [
                            n_samples,
                        ],
                        "Basis function": [
                            bf_name,
                        ],
                        "implementation": "eval_jit",
                        "time [s]": dt,
                    }
                )
                if replica > 0:
                    df = pd.concat([df, current_df_eval, current_df_eval_jit])

df.to_csv("performance.csv")

df = pd.read_csv("performance.csv")
print(df)

g = sns.FacetGrid(df, col="# knots", row="Basis function")
g.map(sns.lineplot, "# samples", "time [s]", "implementation")
g.set(xscale="log")
g.add_legend()
plt.show()
