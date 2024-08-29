Getting Started
===============

.. toctree::
   :maxdepth: 2
   :hidden:

   padding.rst
   citation.rst

Installation
------------
SplineBox can be installed using pip:

.. code-block::

   pip install splinebox

This will install the minimal dependencies.
To install SplineBox with the dependencies required for running examples, use:

.. code-block::

   pip install "splinebox[examples]"

Similarly, you can install dependencies for building the docs or running the test suites:

.. code-block::

   pip install "splinebox[test]"

or

.. code-block::

   pip install "splinebox[docs]"

To install all dependencies:

.. code-block::

   pip install "splinebox[all]"

Constructing Your First Spline
------------------------------

**Note**: If you are unfamiliar with spline terminology, such as knots and control points, refer to our :ref:`theory introduction <Theory>`.

Constructing a spline with SplineBox is straightforward. You need to decide:

- The number of knots/control points your spline should have.
- The type of spline you want to construct, i.e., choose a basis function.
- Whether your spline should be closed or open (i.e., whether the two ends are connected to form a loop).

For this example, we will construct a cubic B-spline with 5 knots that is not closed.

.. code-block:: python

   import splinebox
   spline = splinebox.Spline(M=5, basis_function=splinebox.B3(), closed=False)

At this point, we haven't set any control points or knots.
To shape the spline in `ndim` dimensions, you have three options:

1. **Directly provide the knots**:

   .. code-block:: python

      spline.knots = np.random.rand(5, ndim)

2. **Directly provide the control points**:

   .. code-block:: python

      spline.control_points = np.random.rand(7, ndim)

   Note that 7 control points are needed instead of 5 because splines are padded at the end.

3. **Approximate data with a least squares fit**:

   .. code-block:: python

      spline.fit(np.random.rand(100, ndim))

   Learn more about how ``fit`` works in the theory section on :ref:`Data approximation` or by checking the API :meth:`splinebox.spline_curves.Spline.fit`.

Evaluating/Sampling a Spline
----------------------------

Knots are conventionally placed at integer values of the parameter `t` starting from zero.
To sample an open spline with `M` knots:

.. code-block:: python

    t = np.linspace(0, M-1, 100)
    vals = spline.eval(t)

For a closed spline, continue past the last knot to sample all the way around:

.. code-block:: python

    t = np.linspace(0, M, 100)
    vals = spline.eval(t)
