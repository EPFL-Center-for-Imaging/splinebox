"""
Fixtures for doctesting
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import splinebox


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace["np"] = np
    np.set_printoptions(suppress=True, precision=3)
    if np.__version__ >= "2.0.0":
        np.set_printoptions(legacy="1.25")


@pytest.fixture(autouse=True)
def add_splinebox(doctest_namespace):
    doctest_namespace["splinebox"] = splinebox


@pytest.fixture(autouse=True)
def add_plt(doctest_namespace):
    doctest_namespace["plt"] = plt


@pytest.fixture(autouse=True)
def saving_and_loading(doctest_namespace, tmpdir):
    doctest_namespace["path_to_some_directory"] = tmpdir

    path_to_single_spline_json = tmpdir / "single_spline.json"
    spline = splinebox.Spline(M=3, basis_function=splinebox.B1(), closed=True)
    spline.control_points = np.array([[0.8, 1.2], [0.7, 1.5], [1.1, 0.3]])
    spline.to_json(path_to_single_spline_json)
    doctest_namespace["path_to_single_spline_json"] = path_to_single_spline_json
