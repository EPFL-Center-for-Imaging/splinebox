"""
Code to measure performance for tabel in publication.
"""

import time

import numpy as np
import pandas as pd
import scipy
import splinebox

# Number of times each task is repeated for more reliable results
n_repetitions = 1000
# Number of knots
M = 100
# Dimensionality
ndim = 3

closed = True

# Number of evaluation points
nt = 1000

results_creation = []
results_evaluation = []
results_fitting = []

for repetition in range(n_repetitions + 1):
    knots = np.random.rand(M, ndim)

    # Measure splinebox creation time
    start = time.perf_counter_ns()
    spline = splinebox.Spline(M, splinebox.B3(), closed=closed)
    spline.knots = knots
    stop = time.perf_counter_ns()

    if repetition != 0:
        results_creation.append(
            [
                "splinebox",
                (stop - start) / 1000,
            ]
        )

    # Measure splinebox evaluation time
    t = np.linspace(0, M if closed else M - 1, nt)
    start_eval = time.perf_counter_ns()
    spline(t)
    stop_eval = time.perf_counter_ns()

    if repetition != 0:
        results_evaluation.append(
            [
                "splinebox",
                (stop_eval - start_eval) / 1000,
            ]
        )

    # Measure splinebox fitting time

    data = np.random.rand(M * 10, ndim)

    start = time.perf_counter_ns()
    spline = splinebox.Spline(M, splinebox.B3(), closed=closed)
    spline.fit(data)
    stop = time.perf_counter_ns()

    if repetition != 0:
        results_fitting.append(
            [
                "splinebox",
                (stop - start) / 1000,
            ]
        )

    # Measure scipy spline creation time
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
        results_creation.append(
            [
                "scipy",
                (stop - start) / 1000,
            ]
        )

    # Measure scipy spline evaluation time
    t = np.linspace(0, M if closed else M - 1, nt)
    start_eval = time.perf_counter_ns()
    spline(t)
    stop_eval = time.perf_counter_ns()
    if repetition != 0:
        results_evaluation.append(
            [
                "scipy",
                (stop_eval - start_eval) / 1000,
            ]
        )

    # Measure scipy fitting time
    # The `splprep` is declared as legacy by scipy.
    # However, neither `make_splprep` nor `make_lsq_spline` seem to be able to do the same thing.
    k = 3
    if closed:
        start = time.perf_counter_ns()
        N = len(data)
        t = np.arange(-k, M + k + 1)
        u = np.linspace(0, M, N + 1)[:-1]
        tck, u = scipy.interpolate.splprep(data.T, k=k, u=u, t=t, task=-1, s=0, per=N)
        t, c, k = tck
        c = np.stack(c, axis=-1)
        spline = scipy.interpolate.BSpline(t, c, k)
        stop = time.perf_counter_ns()

    else:
        start = time.perf_counter_ns()
        N = len(data)
        t = np.arange(-k, M + k)
        u = np.linspace(0, M - 1, N)
        tck, u = scipy.interpolate.splprep(data.T, k=k, u=u, t=t, task=-1, s=0)
        t, c, k = tck
        c = np.stack(c, axis=-1)
        spline = scipy.interpolate.BSpline(t, c, k)
        stop = time.perf_counter_ns()

    if repetition != 0:
        results_fitting.append(
            [
                "scipy",
                (stop - start) / 1000,
            ]
        )

creation_df = pd.DataFrame(results_creation, columns=["Package", "time [micros]"])
creation_df = creation_df.groupby("Package", as_index=False).agg(["mean", "std"])
print()
print("##########################################")
print("Creation")
print("##########################################")
print(creation_df)
evaluation_df = pd.DataFrame(results_evaluation, columns=["Package", "time [micros]"])
evaluation_df = evaluation_df.groupby("Package", as_index=False).agg(["mean", "std"])
print()
print("##########################################")
print("Evaluation")
print("##########################################")
print(evaluation_df)
fitting_df = pd.DataFrame(results_fitting, columns=["Package", "time [micros]"])
fitting_df = fitting_df.groupby("Package", as_index=False).agg(["mean", "std"])
print()
print("##########################################")
print("Fitting")
print("##########################################")
print(fitting_df)
