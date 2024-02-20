import splinebox.basis


def test_B1():
    basis = splinebox.basis.B1()
    assert basis.value(0) == 1
