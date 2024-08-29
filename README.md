<img style="float: right;" src="https://imaging.epfl.ch/resources/logo-for-gitlab.svg">

# splinebox
A python package for fitting splines.
Developed by the [EPFL Center for Imaging](https://imaging.epfl.ch/) as part of a collaboration with the [Uhlmann Group at EMBL-EBI](https://www.ebi.ac.uk/research/uhlmann/) in Feb 2024.

[![Documentation Status](https://readthedocs.org/projects/splinebox/badge/?version=latest)](https://splinebox.readthedocs.io/en/latest/?badge=latest)
[![License BSD-3](https://img.shields.io/pypi/l/splinebox.svg?color=green)](https://github.com/EPFL-Center-for-Imaging/splinebox/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/splinebox.svg?color=green)](https://pypi.org/project/splinebox)
[![Python Version](https://img.shields.io/pypi/pyversions/splinebox.svg?color=green)](https://python.org)
[![tests](https://github.com/EPFL-Center-for-Imaging/splinebox/workflows/tests/badge.svg)](https://github.com/EPFL-Center-for-Imaging/splinebox/actions)
[![codecov](https://codecov.io/gh/EPFL-Center-for-Imaging/splinebox/branch/main/graph/badge.svg)](https://codecov.io/gh/EPFL-Center-for-Imaging/splinebox)
[![DOI](https://zenodo.org/badge/759892900.svg)](https://zenodo.org/doi/10.5281/zenodo.13358354)

## Features
* Flexible spline fitting for various applications.
* Support for many spline types in any dimesnionality.
* High-perfomance implementation in Python
* Extensive [documentation](https://splinebox.readthedocs.io/en/latest/?badge=latest) with [examples](https://splinebox.readthedocs.io/en/latest/auto_examples/index.html)

## Installation

You can install `splinebox` via pip:
```
pip install splinebox
```

## Usage

Here is a minimal example of a cubic B-spline in 2D with 3 knots.

```python
import splinebox
import numpy as np
import matplotlib.pyplot as plt

n_knots = 4
spline = splinebox.spline_curves.Spline(M=n_knots, basis_function=splinebox.basis_functions.B3(), closed=True)
spline.knots = np.array([[1, 2], [3, 2], [4, 3], [1, 1]])

t = np.linspace(0, n_knots, 100)
vals = spline.eval(t, derivative=0)

plt.scatter(spline.knots[:, 0], spline.knots[:, 1])
plt.plot(vals[:, 0], vals[:, 1])
plt.show()
```

## Support

If you encounter any problems, please [file and issue](https://github.com/EPFL-Center-for-Imaging/splinebox/issues/new) describing the issue and include minimal example to reproduce the issue.

## Contributing

We welcome contributions! Before you submit a pull request, please ensure that the tests are passing. You can run the tests with [pytest](https://docs.pytest.org/en/stable/). If you are unsure how to implement something, feel free to open an issue to discuss.

## Citing splinebox

If you use splinebox in the context of scientific publication, please cite it as follows.
Note, that you will have to fill in the version yourself. If you are unsure what version you are running,
you can find out by running

```python
import splinebox
print(splinbox.__version__)`
```

BibTeX:

```
@misc{splinebox,
  author = {Aymanns, Florian and Andò, Edward and Uhlmann, Virginie},
  title = {{S}pline{B}ox},
  url = {https://pypi.org/project/splinebox/},
  doi = {10.5281/zenodo.13358354},
  note = {{V}ersion V.V.Vb1},
  year = 2024,
}
```


## License

This is an open source project licensed under the BSD-3-Clause License.
