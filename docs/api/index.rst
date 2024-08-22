API
===

.. toctree::
   :maxdepth: 2
   :hidden:

   basis_functions.rst
   spline_curves.rst

SplineBox's API is divided into two main submodules: :ref:`basis_functions` and :ref:`spline_curves`.

- **basis_functions**: This submodule offers a variety of basis function classes, each tailored to a specific type of basis function. This modularity allows you to easily experiment with different spline types by simply swapping the basis function object when creating your spline.

- **spline_curves**: This submodule provides the classes necessary for constructing splines, along with all the methods required for spline fitting and inference.
