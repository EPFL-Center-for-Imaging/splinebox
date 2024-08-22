Cubic Hermite basis
===================

See [Uhlmann2016]_.

.. math::
   :name: cubichermite:eq:1

   \phi_1(t)=\begin{cases}
   (2|t|+1)(|t|-1)^2 & |t| \in [0,1] \\
   0 & \mathrm{elsewhere}
   \end{cases}

.. math::
   :name: cubichermite:eq:2

   \phi_2(t)=\begin{cases}
   t(|t|-1)^2 & |t| \in [0,1] \\
   0 & \mathrm{elsewhere}
   \end{cases}

.. plot:: pyplots/plot_cubichermite.py
