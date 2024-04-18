import math

import numpy as np
import pytest
import splinebox.basis_functions


def test_base_class_eval(derivative):
    basis_function = splinebox.basis_functions.BasisFunction(False, 2)
    x = np.arange(10)
    with pytest.raises(NotImplementedError):
        basis_function.eval(x, derivative=derivative)


def test_base_class_filters_and_refinement_mask():
    basis_function = splinebox.basis_functions.BasisFunction(False, 2)
    s = np.arange(10)
    with pytest.raises(NotImplementedError):
        basis_function.filter_symmetric(s)
    with pytest.raises(NotImplementedError):
        basis_function.filter_periodic(s)
    with pytest.raises(NotImplementedError):
        basis_function.refinement_mask()


def test_filters(basis_function, is_interpolating, knot_gen):
    s = knot_gen()
    if is_interpolating(basis_function):
        assert np.allclose(basis_function.filter_symmetric(s), s)
        assert np.allclose(basis_function.filter_periodic(s), s)


def test_derivatives(basis_function, derivative, not_differentiable_twice):
    if derivative == 0:
        return

    support = basis_function.support
    x = np.linspace(-support / 2 - 1, support / 2 + 1, 100000)
    y = basis_function.eval(x, derivative=derivative - 1)

    dx = np.diff(x)
    dy = np.diff(y)
    estimated_derivative = dy / dx

    if not_differentiable_twice(basis_function) and derivative == 2:
        # B1, CubicHermite, and ExponentialHermite basis functions are not differentiable twice.
        with pytest.raises(RuntimeError):
            basis_function.eval(x[:-1] + dx / 2, derivative=2)
    else:
        # Check where the estimated derivative is close to the
        # derivative value returned by the method
        close = np.isclose(
            estimated_derivative,
            basis_function.eval(x[:-1] + dx / 2, derivative=derivative),
        )

        # Since the basis functions are not continously differentiable, the
        # estimated_derivative is inaccurate whenever there is a kink in the
        # basis function. These should be isolated values since there are never
        # two kinks right next to each other.
        kernel = np.ones(2)
        if close.ndim == 2:
            # the output of CubicHermite and ExponentialHermite basis functions is 2D
            for i in range(2):
                close[i] = np.convolve(close[i], kernel, mode="same")
        else:
            close = np.convolve(close, kernel, mode="same")

        assert np.all(close > 0)


def test_partition_of_unity(basis_function):
    support = math.ceil(basis_function.support)
    x = np.linspace(0, 1, 10000)
    summed = np.zeros_like(x)
    for k in range(-support, support):
        vals = basis_function.eval(x - k)
        if basis_function.multigenerator:
            vals = vals[0]
        if vals.ndim == 2:
            vals = vals[:, 0]
        summed += vals
    assert np.allclose(summed, np.ones_like(summed))
