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


def test_derivatives(basis_function, derivative):
    if derivative == 0:
        return

    support = basis_function.support
    x = np.linspace(-support / 2 - 1, support / 2 + 1, 100000)
    y = basis_function.eval(x, derivative=derivative - 1)

    dx = np.diff(x)
    dy = np.diff(y)
    estimated_derivative = dy / dx

    if (
        isinstance(
            basis_function,
            (
                splinebox.basis_functions.B1,
                splinebox.basis_functions.CubicHermite,
                splinebox.basis_functions.ExponentialHermite,
            ),
        )
        and derivative == 2
    ):
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
