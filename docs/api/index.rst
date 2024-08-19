API
===

.. toctree::
   :maxdepth: 2
   :hidden:

   basis_functions.rst
   spline_curves.rst

Splinbox's API is split into two submodules: `basis_functions`, `spline_curves`.
basis_functions provides a plethora or basis function classes that implement everything that is specific to the type of basis function.
This makes it easy to try different kind of splines as you only have to replace the basis function object used when you create your spline.
The second submodule, `spline_curves`, provides the classes used for constructing splines with all of the methods for fitting and inference of the spline.
