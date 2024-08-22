Polynomial basis (B-spline)
===========================

The polynomial spline basis of order :math:`n`, usually denoted as :math:`\beta_n`, is obtained by convolving the :math:`0`-th order basis

.. math::
   :name: poly:eq:1

   \beta_0(t)=\begin{cases}
   1, \quad t\in [-\frac{1}{2}, \frac{1}{2}] \\
   0, \quad  \mathrm{elsewhere}
   \end{cases}

with itself :math:`n` times, \emph{i.e.}

.. math::
   :name: poly:eq:2

   \beta_n(t) = (\smash[b]{\underbrace{\beta_0 \ast ... \ast \beta_0}_\text{$n$ times}}).

See [Unser1999]_ for more details.

Linear (:math:`1^{\text{st}}` order) polynomial basis

.. math::
   :name: poly:eq:3

   \beta_1(t)=\begin{cases}
   1-|t|, \quad |t|\in [0, 1] \\
   0, \quad  \mathrm{elsewhere}
   \end{cases}

.. plot:: pyplots/plot_b1.py

Quadratic (:math:`2^{\text{nd}}` order) polynomial basis

.. math::
   :name: poly:eq:4

   \beta_2(t)=\begin{cases}
   \frac{1}{2}t^2 + \frac{3}{2}t + \frac{9}{8}, \quad t\in [-\frac{3}{2}, -\frac{1}{2}[ \\
    \frac{3}{4}-t, \quad t\in [-\frac{1}{2}, \frac{1}{2}[ \\
   \frac{1}{2}t^2 - \frac{3}{2}t + \frac{9}{8}, \quad t\in [\frac{1}{2}, \frac{3}{2}] \\
   0, \quad  \mathrm{elsewhere}
   \end{cases}

.. plot:: pyplots/plot_b2.py

Cubic (:math:`3^{\text{rd}}` order) polynomial basis

.. math::
   :name: poly:eq:5

   \beta_3(t)=\begin{cases}
   \frac{2}{3} - |t|^2 + \frac{1}{2}|t|^3, \quad |t|\in [0, 1[ \\
   \frac{1}{6}(2 - |t|)^3, \quad |t| \in [1, 2] \\
   0, \quad  \mathrm{elsewhere}
   \end{cases}

.. plot:: pyplots/plot_b3.py
