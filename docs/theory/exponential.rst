Exponential basis
=================

See [Delgado2012]_.

.. math::
   :name: exponential:eq:1

   \phi(t)=\begin{cases}
   \frac{\cos\frac{2\pi|t|}{M}\cos\frac{\pi}{M}-\cos\frac{2\pi}{M}}{1-\cos\frac{2\pi}{M}}, & |t| \in [0,\frac{1}{2}[\\
   \frac{1-\cos\frac{2\pi\left(\frac{3}{2}-|t|\right)}{M}}{2\left(1-\cos\frac{2\pi}{M}\right)}, &  |t| \in [\frac{1}{2}, \frac{3}{2}[\\
   0, & \mathrm{elsewhere}
   \end{cases}

where :math:`M` is the number of control points.

.. plot:: pyplots/plot_exponential.py
