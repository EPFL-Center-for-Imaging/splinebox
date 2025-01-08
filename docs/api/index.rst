API
===

SplineBox is organized into two primary modules.
The first module contains the :code:`Spline` class and related functionalities for creating and manipulating spline objects.
The second module is a collection of basis functions, which are essential for generating various types of splines.

While these modules provide clear organization within the library, all components are also accessible using a unified namespace.
For instance, you can improve readability of your code by using :code:`splinebox.X` instead of using the specific module path like :code:`splinebox.basis_functions.X` or :code:`splinebox.spline_curves.X` respectively.

.. autosummary::
   :toctree: _generated
   :template: custom-autosummary-module-template.rst
   :recursive:

   splinebox.basis_functions
   splinebox.spline_curves
