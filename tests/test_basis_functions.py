import numpy as np
import pytest
import splinebox.basis_functions


def test_1st_derivatives(basis_function):
    support = basis_function.support
    x = np.linspace(-support / 2 - 1, support / 2 + 1, 100000)
    y = basis_function.eval(x)

    dx = np.diff(x)
    dy = np.diff(y)
    estimated_derivative = dy / dx

    # Check where the estimated derivative is close to the
    # derivative value returned by the method
    close = np.isclose(
        estimated_derivative,
        basis_function.eval_1st_derivative(x[:-1] + dx / 2),
    )

    # Since the basis functions are not continously differentiable, the
    # estimated_derivative is inaccurate whenever there is a kink in the
    # basis function. These should be isolated values since there are never
    # two kinks right next to each other.
    kernel = np.ones(2)
    if close.ndim == 2:
        # the output of H3 and HE3 is 2D
        for i in range(2):
            close[i] = np.convolve(close[i], kernel, mode="same")
    else:
        close = np.convolve(close, kernel, mode="same")

    assert np.all(close > 0)


def test_2nd_derivatives(basis_function):
    support = basis_function.support
    x = np.linspace(-support / 2 - 1, support / 2 + 1, 100000)
    y = basis_function.eval_1st_derivative(x)

    dx = np.diff(x)
    dy = np.diff(y)
    estimated_derivative = dy / dx

    if isinstance(
        basis_function, (splinebox.basis_functions.B1, splinebox.basis_functions.H3, splinebox.basis_functions.HE3)
    ):
        # B1, H3, and HE3 are not differentiable twice.
        with pytest.raises(RuntimeError):
            basis_function.eval_2nd_derivative(x[:-1] + dx / 2)
    else:
        derivative = basis_function.eval_2nd_derivative(x[:-1] + dx / 2)

        # Check where the estimated derivative is close to the
        # derivative value returned by the method
        close = np.isclose(estimated_derivative, derivative)

        # For an explenation see test_1st_derivatives
        close = np.convolve(close, np.ones(2), mode="same")
        assert np.all(close > 0)
