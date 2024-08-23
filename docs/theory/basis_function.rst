Basis function
==============

.. toctree::
   :maxdepth: 2
   :hidden:

   polynomial.rst
   exponential.rst
   catmullrom.rst
   cubichermite.rst
   exponentialhermite.rst


The basis function :math:`\phi` has the following important properties.

* **Support.** The *support* is the size of the largest interval in which :math:`\phi` is non-zero. For instance, the function

  .. math::
     :name: basis:eq:1

     \phi(t)=\begin{cases}
     1, \quad t\in [-\frac{1}{2}, \frac{1}{2}] \\
     0, \quad  \mathrm{elsewhere}
     \end{cases}

  has a support of size :math:`1`.
  If :math:`\phi` has a support of size :math:`L`, then :math:`\phi(t-k)` will be zero outside of :math:`[k-\frac{L}{2}, k+\frac{L}{2}]` and it will only occupy the :math:`\lceil \frac{L}{2} \rceil` intervals on each side of :math:`k`. The support thus dictates how many neighboring intervals each basis acts upon. Relying of basis that have a small support size means that each control point "controls" only a very localized portion of the entire function.

* **Interpolatory behaviour.** The basis function :math:`\phi` is said to be *interpolatory* if

  .. math::
     :name: basis:eq:2

     \phi(k) =\begin{cases}
     1, \quad k=0 \\
     0, \quad k \in \mathbb{Z}_{\ne 0}.
     \end{cases}

  When this is the case, we therefore have that :math:`r(k)=c[k]`, *i.e.* the spline and the control points coincide at integer values of :math:`k`.
  Most basis are not interpolatory and, as a consequence, spline functions constructed with them do not go through their own control points (*i.e.*, :math:`r(k) \neq c[k]`.
  Because of this, it is sometimes easier to think in terms of *knots* as opposed to control points.
  The knots :math:`n` correspond to the values of the spline at integer locations, *i.e.*

  .. math::
     :name: basis:eq:3

     n[k] = r(k).

  Therefore, if :math:`\phi` is interpolatory, the knots and the control points coincide (*i.e.*, :math:`n[k] = c[k]`).
  The knots also correspond to the junction points of the different intervals of the spline.

  Knots and control points are related through the so-called *inverse filter* of :math:`\phi`.
  Details can be found in [Unser1999]_, but for the purpose of using SplineBox it is sufficient to understand that control points and knots are related, and that each of these sequences can be transformed into one another.

* **Degree and regularity.** The degree and regularity of :math:`\phi` dictates respectively the degree of the intervals and the regularity at the knots in the resulting spline.
  For instance, if :math:`\phi` is a cubic polynomial, then :math:`r` will be piecewise cubic and twice differentiable at the knots.
