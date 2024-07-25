import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import splinebox

n_repetitions = 10
M = 10
closed = True
ndim = 10
nt = 1000


results = []


for repetition in range(n_repetitions + 1):
    knots = np.random.rand(M, ndim)
    start = time.perf_counter_ns()
    spline = splinebox.Spline(M, splinebox.B3(), closed=closed)
    spline.knots = knots
    stop = time.perf_counter_ns()
    if repetition != 0:
        results.append(
            [
                "splinebox",
                f"{ndim}D",
                M,
                "closed" if closed else "open",
                "Creation",
                stop - start,
            ]
        )

    t = np.linspace(0, M if closed else M - 1, nt)
    start_eval = time.perf_counter_ns()
    spline.eval(t)
    stop_eval = time.perf_counter_ns()
    if repetition != 0:
        results.append(
            [
                "splinebox",
                f"{ndim}D",
                M,
                "closed" if closed else "open",
                "Evaluation",
                stop - start,
            ]
        )

    if closed:
        start = time.perf_counter_ns()
        t_knots = np.arange(M + 1)
        knots_periodic = np.concatenate([knots, knots[0][np.newaxis, :]], axis=0)
        spline = scipy.interpolate.make_interp_spline(t_knots, knots_periodic, k=3, bc_type="periodic")
        stop = time.perf_counter_ns()
    else:
        start = time.perf_counter_ns()
        t_knots = np.arange(M)
        spline = scipy.interpolate.make_interp_spline(t_knots, knots, k=3, bc_type="natural")
        stop = time.perf_counter_ns()

    if repetition != 0:
        results.append(
            [
                "scipy",
                f"{ndim}D",
                M,
                "closed" if closed else "open",
                "Creation",
                stop - start,
            ]
        )

    t = np.linspace(0, M if closed else M - 1, nt)
    start_eval = time.perf_counter_ns()
    spline(t)
    stop_eval = time.perf_counter_ns()
    if repetition != 0:
        results.append(
            [
                "scipy",
                f"{ndim}D",
                M,
                "closed" if closed else "open",
                "Evaluation",
                stop - start,
            ]
        )

    data = np.random.rand(M * 10, ndim)

    start = time.perf_counter_ns()
    spline = splinebox.Spline(M, splinebox.B3(), closed=closed)
    spline.fit(data)
    stop = time.perf_counter_ns()

    if repetition != 0:
        results.append(
            [
                "splinebox",
                f"{ndim}D",
                M,
                "closed" if closed else "open",
                "Fitting",
                stop - start,
            ]
        )

    k = 3
    if closed:
        start = time.perf_counter_ns()
        N = len(data)
        t = np.arange(-k, M + k + 1)
        u = np.linspace(0, M, N + 1)[:-1]
        tck, u = scipy.interpolate.splprep(data.T, k=k, u=u, t=t, task=-1, s=0, per=N)
        stop = time.perf_counter_ns()

    else:
        start = time.perf_counter_ns()
        N = len(data)
        t = np.arange(-k, M + k)
        u = np.linspace(0, M - 1, N)
        tck, u = scipy.interpolate.splprep(data.T, k=k, u=u, t=t, task=-1, s=0)
        stop = time.perf_counter_ns()

    if repetition != 0:
        results.append(
            [
                "scipy",
                f"{ndim}D",
                M,
                "closed" if closed else "open",
                "Fitting",
                stop - start,
            ]
        )


df = pd.DataFrame(results, columns=["Package", "Dimensionality", "Number of knots", "closed", "Task", "time [ns]"])
df["time [ms]"] = df["time [ns]"] / 1000

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set(font_scale=1.5)
fig, ax = plt.subplots(figsize=(7, 6))
ax.set(yscale="log")
sns.set_palette(sns.color_palette(["#2cb42c", "#0053a6"]))
g = sns.barplot(df, x="Task", y="time [ms]", hue="Package", ax=ax)
g.set(xlabel=None)
plt.tight_layout()
plt.show()
