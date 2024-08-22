basis_functions
===============

This submodule provides a wide range of basis functions implemented as classes.
Each of these classes inherites from the the abstrac base class :class:`splinebox.basis_functions.BasisFunction` described below and
should overwrite the following class methods:

- :func:`splinebox.basis_functions.BasisFunction.eval`
- :func:`splinebox.basis_functions.BasisFunction.filter_periodic`
- :func:`splinebox.basis_functions.BasisFunction.filter_summetric`

Currently the following basis functions are implemented: :ref:`B1`, :ref:`B2`, :ref:`B2`, :ref:`Exponential`, :ref:`Catmull Rom`, :ref:`Cubic Hermite`, :ref:`Exponential Hermite`

.. autoclass:: splinebox.basis_functions.BasisFunction
   :members:
   :undoc-members:
   :exclude-members: refinement_mask


.. toctree::
   :maxdepth: 1
   :hidden:

   b1.rst
   b2.rst
   b3.rst
   exponential.rst
   catmullrom.rst
   cubichermite.rst
   exponentialhermite.rst
