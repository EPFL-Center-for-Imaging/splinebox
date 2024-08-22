Exponential Hermite basis
=========================

See [Uhlmann2014]_.

.. math::
   :name: exponentialhermite:eq:1

   \phi_{1,M}(t) &= \left\{
       \begin{array}{ll}
          	g_{1,M}(t)  & t \geq 0 \\
           g_{1,M}(-t) &  t < 0
       \end{array}
   \right.

.. math::
   :name: exponentialhermite:eq:2

   \phi_{2,M}(t) &= \left\{
       \begin{array}{lll}
   		g_{2,M}(t) & t \geq 0 \\
           -g_{2,M}(-t) & t < 0
       \end{array}
   \right.

where :math:`M` is the number of control points and

.. math::
   :name: exponentialhermite:eq:3

   g_{1,M}(t)=
   \left\{
   \begin{array}{ll}
   a_1(M) + b_1(M) t + c_1(M) \text{e}^{\text{j}\,\frac{2\,\pi}{M}\,t} + d_1(M) \text{e}^{-\text{j}\,\frac{2\,\pi}{M}\,t} & t \in [0,1[ \\
   0 &  \mbox{elsewhere}
   \end{array}
   \right.

.. math::
   :name: exponentialhermite:eq:4

   g_{2,M}(t)=
   \left\{
   \begin{array}{ll}
   a_2(M) + b_2(M) t + c_2(M) \text{e}^{\text{j}\,\frac{2\,\pi}{M} t} + d_2(M) \text{e}^{-\text{j}\,\frac{2\,\pi}{M} t} & t \in [0,1[ \\
   0 &  \mbox{elsewhere} \end{array}
   \right.

.. math::
   :name: exponentialhermite:eq:5

   \begin{array}{ll}
   a_1(M) = \frac{\text{j}\,\frac{2\,\pi}{M}+1 + \text{e}^{\text{j}\,\frac{2\,\pi}{M}} (\text{j}\,\frac{2\,\pi}{M}-1)}{q(M)} & b_1(M) = -\frac{\text{j}\,\frac{2\,\pi}{M} (\text{e}^{\text{j}\,\frac{2\,\pi}{M}} + 1)}{q(M)} \\
   c_1(M) = \frac{1}{q(M)} & d_1(M) = -\frac{\text{e}^{\text{j}\,\frac{2\,\pi}{M}}}{q(M)} \\
   a_2(M)= \frac{p(M)}{\text{j}\,\frac{2\,\pi}{M} (\text{e}^{\text{j}\,\frac{2\,\pi}{M}}-1) q(M) } & b_2(M)= -\frac{\text{e}^{\text{j}\,\frac{2\,\pi}{M}}-1}{q(M)} \\
   c_2(M)= \frac{\text{e}^{\text{j}\,\frac{2\,\pi}{M}}-\text{j}\,\frac{2\,\pi}{M}-1}{\text{j}\,\frac{2\,\pi}{M}(\text{e}^{\text{j}\,\frac{2\,\pi}{M}}-1)q(M)} &
   d_2(M)= -\frac{\text{e}^{\text{j}\,\frac{2\,\pi}{M}} (\text{e}^{\text{j}\,\frac{2\,\pi}{M}}(\text{j}\,\frac{2\,\pi}{M}-1) + 1)}{\text{j}\,\frac{2\,\pi}{M}(\text{e}^{\text{j}\,\frac{2\,\pi}{M}} - 1) q(M)}
   \end{array}

.. math::
   :name: exponentialhermite:eq:6

   p(M) &= \text{j}\,\frac{2\,\pi}{M}+1+\text{e}^{\text{j}\,\frac{4\,\pi}{M}}(\text{j}\,\frac{2\,\pi}{M}-1)\,  \\
   q(M) &= \text{j}\,\frac{2\,\pi}{M}+2+\text{e}^{\text{j}\,\frac{2\,\pi}{M}}(\text{j}\,\frac{2\,\pi}{M}-2)

.. plot:: pyplots/plot_exponentialhermite.py
