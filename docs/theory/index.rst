Theory
======

SplineBox implements uniform splines of the form

.. math::
   :name: eq:1

   r(t) = \sum_0^{M-1} c[k]\phi(t-k),

where

* :math:`r: \mathbb{R} \mapsto \mathbb{R}` is a piecewise-continuous function referred to as **spline**
* :math:`t \in \mathbb{R}` is a continuous **parameter**
* :math:`c[k] \in \mathbb{R}, k = 0,...,M-1` are coefficients referred to as **control points**
* :math:`\phi: \mathbb{R} \mapsto \mathbb{R}` is a function referred to as **basis**

One can intuitively think of :math:`r` being built by summing :math:`M` copies of :math:`\phi` centered at integer locations :math:`k=0,...,M-1`, each of them scaled by corresponding weights :math:`c[k]`.
We refer to each :math:`t \in [k, k+1[, k=0,...,M-1` as **intervals**.

The function :math:`r` can be made *periodic* by either :math:`M`-periodizing the sequence of :math:`\{ c[k] \}_{k=0,...,M-1}` such that :math:`c[0]=c[M]`, or by :math:`M`-periodizing :math:`\phi` and replacing it in :ref:`(1) <eq:1>` by its periodized version

.. math::
   :name: eq:2

   \phi_M(t) = \sum_{i\in\mathbb{Z}} \phi(t-Mi).

The model :ref:`(1) <eq:1>` allows building 1D functions, planar parametric curves, and parametric curves embedded in 3D with the following modifications:

* :math:`c[k] \in \mathbb{R}` results in :math:`r: \mathbb{R} \mapsto \mathbb{R}` (function);
* :math:`\mathbf{c}[k] \in \mathbb{R}^2` results in :math:`\mathbf{r}: \mathbb{R} \mapsto \mathbb{R}^2` (planar parametric curve);
* :math:`\mathbf{c}[k] \in \mathbb{R}^3` results in :math:`\mathbf{r}: \mathbb{R} \mapsto \mathbb{R}^3` (parametric curve in 3D).

Basis function
--------------
The basis function :math:`\phi` has the following important properties.

* **Support.** The *support* is the size of the largest interval in which :math:`\phi` is non-zero. For instance, the function

  .. math::
     :name: eq:3

     \phi(t)=\begin{cases}
     1, \quad t\in [-\frac{1}{2}, \frac{1}{2}] \\
     0, \quad  \mathrm{elsewhere}
     \end{cases}

  has a support of size :math:`1`.
  If :math:`\phi` has a support of size :math:`L`, then :math:`\phi(t-k)` will be zero outside of :math:`[k-\frac{L}{2}, k+\frac{L}{2}]` and it will only occupy the :math:`\lceil \frac{L}{2} \rceil` intervals on each side of :math:`k`. The support thus dictates how many neighboring intervals each basis acts upon. Relying of basis that have a small support size means that each control point "controls" only a very localized portion of the entire function.

* **Interpolatory behaviour.** The basis function :math:`\phi` is said to be *interpolatory* if

  .. math::
     :name: eq:4

     \phi(k) =\begin{cases}
     1, \quad k=0 \\
     0, \quad k \in \mathbb{Z}_{\ne 0}.
     \end{cases}

  When this is the case, we therefore have that :math:`r(k)=c[k]`, *i.e.* the spline and the control points coincide at integer values of :math:`k`.
  Most basis are not interpolatory and, as a consequence, spline functions constructed with them do not go through their own control points (*i.e.*, :math:`r(k) \neq c[k]`.
  Because of this, it is sometimes easier to think in terms of *knots* as opposed to control points.
  The knots :math:`n` correspond to the values of the spline at integer locations, *i.e.*

  .. math::
     :name: eq:5

     n[k] = r(k).

  Therefore, if :math:`\phi` is interpolatory, the knots and the control points coincide (*i.e.*, :math:`n[k] = c[k]`).
  The knots also correspond to the junction points of the different intervals of the spline.

  Knots and control points are related through the so-called *inverse filter* of :math:`\phi`.
  Details can be found in [1], but for the purpose of using the \texttt{splinebox} library it is sufficient to understand that control points and knots are related, and that each of these sequences can be transformed into one another.

* **Degree and regularity.** The degree and regularity of :math:`\phi` dictates respectively the degree of the intervals and the regularity at the knots in the resulting spline.
  For instance, if :math:`\phi` is a cubic polynomial, then :math:`r` will be piecewise cubic and twice differentiable at the knots.

Data approximation
------------------
In order to build :math:`r`, one can either use a pre-defined sequence of control points :math:`\{ c[k] \}_{k=0,...,M-1}` or of knots :math:`\{ n[k] \}_{k=0,...,M-1}`. Alternatively, one can also attempt to retreive the control points that best approximate a set of data points, as in the classical spline approximation setting.

The problem is framed as follows. We consider a set :math:`\{ p[i] \}_{i=0,...,N-1}` of :math:`N` points to be approximated with the spline model :ref:`(1) <eq:1>` of :math:`M` control points. Hereafter, we will assume a periodic spline model, but a similar derivation can easily be done for the non-periodic case.

We obtain an approximation by ensuring that the samples of the spline :math:`r` match the data points :math:`p`, which translates to

.. math::
   :name: eq:6

   p[i] = \sum_{k=0}^{M-1}c[k]\phi\left(\frac{Mi}{N}-k\right).

Since :math:`\phi` is of finite support, we can re-write :ref:`(6) <eq:6>` as

.. math::
   :name: eq:7

   \mathbf{\Phi}\mathbf{C} = \mathbf{P},

with the basis matrix :math:`\mathbf{\Phi}` (size :math:`N \times M`), the control point matrix :math:`\mathbf{C}` (size :math:`M \times 1`), and the data points matrix :math:`\mathbf{P}` (size :math:`N \times 1`) given by

.. math::
   :name: eq:8

   \mathbf{\Phi} = \begin{bmatrix}
    \phi(0) &  \phi(-1) & \dots & \ \phi(-(M-1)) \\
    \phi\left(\frac{M}{N}\right) &  \phi\left(\frac{M}{N}-1\right) & \dots & \ \phi\left(\frac{M}{N}-(M-1)\right) \\
    \vdots & \vdots & \ddots & \vdots \\
    \phi\left(\frac{(N-1)M}{N}\right) &  \phi\left(\frac{(N-1)M}{N}-1\right) & \dots & \ \phi\left(\frac{(N-1)M}{N}-(M-1)\right)
   \end{bmatrix}

.. math::
   :name: eq:9

   \mathbf{C}  =  \begin{bmatrix}
    c[0] \\
    \vdots  \\
    c[M-1]
   \end{bmatrix}

.. math::
   :name: eq:10

   \mathbf{P}  =  \begin{bmatrix}
    p[0] \\
    \vdots  \\
    p[N-1]
   \end{bmatrix}.

The control points :math:`\mathbf{C}` can then be retreived by finding the least-square best solution that minimizes

.. math::
   :name: eq:11

   \| \mathbf{P} - \mathbf{\Phi} \mathbf{C} \|^2_2.

References
----------
[1] M. Unser, “Splines: A perfect fit for signal and image processing,” IEEE Signal processing
magazine, vol. 16, no. 6, pp. 22–38, 1999.
