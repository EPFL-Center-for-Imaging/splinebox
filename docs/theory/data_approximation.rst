Data approximation
==================

In order to build :math:`r`, one can either use a pre-defined sequence of control points :math:`\{ c[k] \}_{k=0,...,M-1}` or of knots :math:`\{ n[k] \}_{k=0,...,M-1}`. Alternatively, one can also attempt to retreive the control points that best approximate a set of data points, as in the classical spline approximation setting.

The problem is framed as follows. We consider a set :math:`\{ p[i] \}_{i=0,...,N-1}` of :math:`N` points to be approximated with the spline model :ref:`(1) <theory:eq:1>` of :math:`M` control points. Hereafter, we will assume a periodic spline model, but a similar derivation can easily be done for the non-periodic case.

We obtain an approximation by ensuring that the samples of the spline :math:`r` match the data points :math:`p`, which translates to

.. math::
   :name: approx:eq:1

   p[i] = \sum_{k=0}^{M-1}c[k]\phi\left(\frac{Mi}{N}-k\right).

Since :math:`\phi` is of finite support, we can re-write :ref:`(1) <approx:eq:1>` as

.. math::
   :name: approx:eq:2

   \mathbf{\Phi}\mathbf{C} = \mathbf{P},

with the basis matrix :math:`\mathbf{\Phi}` (size :math:`N \times M`), the control point matrix :math:`\mathbf{C}` (size :math:`M \times 1`), and the data points matrix :math:`\mathbf{P}` (size :math:`N \times 1`) given by

.. math::
   :name: approx:eq:3

   \mathbf{\Phi} = \begin{bmatrix}
    \phi(0) &  \phi(-1) & \dots & \ \phi(-(M-1)) \\
    \phi\left(\frac{M}{N}\right) &  \phi\left(\frac{M}{N}-1\right) & \dots & \ \phi\left(\frac{M}{N}-(M-1)\right) \\
    \vdots & \vdots & \ddots & \vdots \\
    \phi\left(\frac{(N-1)M}{N}\right) &  \phi\left(\frac{(N-1)M}{N}-1\right) & \dots & \ \phi\left(\frac{(N-1)M}{N}-(M-1)\right)
   \end{bmatrix}

.. math::
   :name: approx:eq:4

   \mathbf{C}  =  \begin{bmatrix}
    c[0] \\
    \vdots  \\
    c[M-1]
   \end{bmatrix}

.. math::
   :name: approx:eq:5

   \mathbf{P}  =  \begin{bmatrix}
    p[0] \\
    \vdots  \\
    p[N-1]
   \end{bmatrix}.

The control points :math:`\mathbf{C}` can then be retreived by finding the least-square best solution that minimizes

.. math::
   :name: approx:eq:6

   \| \mathbf{P} - \mathbf{\Phi} \mathbf{C} \|^2_2.
