Getting Started
===============

Installation
------------
SplineBox can be installed using pip by running:

.. code-block::

   pip install splinebox

This will only install the minimal dependencies.
If you wish to install splinebox with the dependencies necessary to run the examples,
you can do so with the following command:

.. code-block::

   pip install "splinebox[examples]"

Similarly, you can install the depencies for building the docs or running the test suits.

.. code-block::

   pip install "splinebox[test]"

or

.. code-block::

   pip install "splinebox[docs]"

To install all dependencies run:

.. code-block::

   pip install "splinebox[all]"

Constructing your first spline
------------------------------
Constructing a spline with spline box is easy.
You just have to decide:
- how many knots/controlpoints your spline should have
- what type of spline you want to construct, i.e. choose a basis function
- whether your spline should be closed or open, i.e. are the two ends connected to each other to form a loop

For this example, we will construc a cubic B-spline with, 5 knots, that is closed.

.. code-block:: python

   import splinebox
   spline = splinebox.Spline(M=5, basis_function=basis_function, closed=True)

For now, we haven't set any control points or knots yet.
TODO: Mention ndim
To give the spline it's shape you have three options:

1. Directly provide the knots:

   .. code-block:: python

      spline.knots = np.random.rand(size=(ndim, 5))

2. Directly provide the control points:

   .. code-block:: python

      spline.control_points = np.random.rand(size=(ndim, 7))

   Note that we need to provide 7 control points instead of 5.
   This is because the splines are padded at the end.

3. Approximating some data with a least squared fit:

   .. code-block:: python

      spline.fit(np.random.rand(size=(ndim, 100))

   You can learn more about how ``fit`` works in the theory section on :ref:`Data approximation` theory section or by checking the API :meth:`splinebox.spline_curves.Spline.fit`.

Fitting a spline to pixel values
--------------------------------

Padding
-------
Open splines have to be padded at the ends if the support of the basis function is larger than two.
This can be easily understood by looking at eq1.

If they are not padded at the ends if the

Controlling boundary conditions
-------------------------------
