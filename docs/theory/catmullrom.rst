Catmull Rom basis
=================

See [Keys1981]_.

.. math::
   :name: catmullrom:eq:1

   \phi(t)=\begin{cases}
   \frac{3}{2}|t|^3 - \frac{5}{2}|t|^2 + 1, \quad |t| \in [0,1[ \\
   -\frac{1}{2}|t|^3 + \frac{5}{2}|t|^2 - 4|t| + 2, \quad |t| \in [1,2] \\
   0, \quad  \mathrm{elsewhere}
   \end{cases}

.. plot:: pyplots/plot_catmullrom.py
