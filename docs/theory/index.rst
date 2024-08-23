Theory
======

.. toctree::
   :maxdepth: 2
   :hidden:

   basis_function.rst
   data_approximation.rst
   ../references.rst

SplineBox implements uniform splines of the form

.. math::
   :name: theory:eq:1

   r(t) = \sum_0^{M-1} c[k]\phi(t-k),

where

* :math:`r: \mathbb{R} \mapsto \mathbb{R}` is a piecewise-continuous function referred to as **spline**
* :math:`t \in \mathbb{R}` is a continuous **parameter**
* :math:`c[k] \in \mathbb{R}, k = 0,...,M-1` are coefficients referred to as **control points**
* :math:`\phi: \mathbb{R} \mapsto \mathbb{R}` is a function referred to as **basis**

One can intuitively think of :math:`r` being built by summing :math:`M` copies of :math:`\phi` centered at integer locations :math:`k=0,...,M-1`, each of them scaled by corresponding weights :math:`c[k]`.
We refer to each :math:`t \in [k, k+1[, k=0,...,M-1` as **intervals**.

We call junction points between the intervals knots :math:`n[k] = r(k)`. The relationship bewtween knots and control points is discussed on the page about the :ref:`Basis function`.

The function :math:`r` can be made *periodic* by either :math:`M`-periodizing the sequence of :math:`\{ c[k] \}_{k=0,...,M-1}` such that :math:`c[0]=c[M]`, or by :math:`M`-periodizing :math:`\phi` and replacing it in :ref:`(1) <theory:eq:1>` by its periodized version

.. math::
   :name: theory:eq:2

   \phi_M(t) = \sum_{i\in\mathbb{Z}} \phi(t-Mi).

The model :ref:`(1) <theory:eq:1>` allows building 1D functions, planar parametric curves, and parametric curves embedded in 3D with the following modifications:

* :math:`c[k] \in \mathbb{R}` results in :math:`r: \mathbb{R} \mapsto \mathbb{R}` (function);
* :math:`\mathbf{c}[k] \in \mathbb{R}^2` results in :math:`\mathbf{r}: \mathbb{R} \mapsto \mathbb{R}^2` (planar parametric curve);
* :math:`\mathbf{c}[k] \in \mathbb{R}^3` results in :math:`\mathbf{r}: \mathbb{R} \mapsto \mathbb{R}^3` (parametric curve in 3D).
