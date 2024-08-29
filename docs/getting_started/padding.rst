Padding
-------

Open splines require padding at the ends when the support of the basis function exceeds two.

This concept can be understood by examining equation :ref:`(1) <theory:eq:1>`.
Padding means extending the sum as follows:

.. math::
   r(t) = \sum_{-p}^{M-1+p}c[k]\Phi(t-k),

where :math:`p` is the amount of padding.

Not padding, is equivalent to setting :math:`c[k]=0` for all :math:`k<0` and :math:`k>M-1`.
If the basis function :math:`\Phi(t)` is non-zero at :math:`t>=-1` and :math:`t>=1` (i.e., support larger than two), these control points influence the spline and must be set.
In practice, not padding (setting them to zero) causes the spline's ends to curve toward the origin.

For example, consider a cubic B-spline without padding, where control points are equidistantly placed on a semicircle.
The spline curves inward toward the origin, despite the last control point.

.. plot:: pyplots/plot_no_padding.py

By padding with two additional points on the circle, this behavior is corrected.

.. plot:: pyplots/plot_padding.py

When you directly set control points for a spline, you must handle the padding yourself.
This design choice allows full control over the spline's behavior at the ends.

.. code-block:: python

   spline = splinebox.Spline(M=5, basis_function=splinebox.B3(), closed=False)
   spline.control_points = np.random.rand((7, ndim))

In the example above, seven control points are required instead of five to account for padding.

When setting knots, padding is handled internally.
By default, knots are repeated at the ends, but you can provide custom padding functions to :class:`splinebox.spline_curves.Spline`.
