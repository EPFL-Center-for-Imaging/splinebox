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
    if np.__version__ >= "2.0.0":
        np.set_printoptions(legacy="1.25")


@pytest.fixture(autouse=True)
def add_splinebox(doctest_namespace):
    doctest_namespace["splinebox"] = splinebox


@pytest.fixture(autouse=True)
def add_plt(doctest_namespace):
    doctest_namespace["plt"] = plt
