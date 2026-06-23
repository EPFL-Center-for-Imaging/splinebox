Active contour model
====================

The active contour model (also known as a snake) is a method for fitting a spline to image features such as edges, ridges, or object boundaries.
The spline is represented by a set of control points and evolves by minimising an energy functional :math:`E`.

Starting from an initial estimate, the control points are iteratively updated using gradient descent until the energy reaches a local minimum.

.. math::
    :name: active_contour_model:eq:1

    c[l]_x = c[l]_x - \frac{\partial E}{\partial c[l]_x}

This can also be written using the matrix notation and the Jacobian matrix :math:`\mathbf{J_r}`.

.. math::

    \mathbf{C} = \mathbf{C} - \mathbf{J_r}

.. math::

    \mathbf{J_r} = \begin{bmatrix}
     \frac{\partial r_x}{\partial c[0]_x} & \frac{\partial r_x}{\partial c[0]_y} & \frac{\partial r_x}{\partial c[0]_z}  \\
     \phi\left(\frac{M}{N}\right) &  \phi\left(\frac{M}{N}-1\right) & \dots & \ \phi\left(\frac{M}{N}-(M-1)\right) \\
     \vdots & \vdots & \ddots & \vdots \\
     \phi\left(\frac{(N-1)M}{N}\right) &  \phi\left(\frac{(N-1)M}{N}-1\right) & \dots & \ \phi\left(\frac{(N-1)M}{N}-(M-1)\right)
    \end{bmatrix}

The energy typically consists of two components:
    (1) **Image energy**, which attracts the spline toward features in the image.
    (2) **Internal energy**, which regularizes the spline and discourages undesirable deformations.

.. math::
    :name: active_contour_model:eq:2

    E = E_{image} + E_{internal}

Image energy
------------

The image energy measures the image intensity sampled along the spline. By choosing an appropriate image representation (for example, an edge map), this term can be used to attract the spline toward specific image features.

In practice, the continuous integral is approximated by sampling the spline at a finite number of points.

.. math::
    :name: active_contour_model:eq:3

    E_{image} &= \int_0^{M-1}\text{img}[r(t)]dt \\
              &\approx \frac{M-1}{N} \sum_{i=0}^N \text{img}[r(i * \frac{M-1}{N})]

To optimize the spline, we need the derivatives of the image energy with respect to the control points. Applying the chain rule yields an expression involving the image gradient and the derivative of the spline with respect to a control point.

.. math::
    :name: active_contour_model:eq:4

    \frac{\partial E_{image}}{\partial c[l]_x} &= \frac{\partial}{\partial c[l]_x} \frac{M-1}{N} \sum_{i=0}^N \text{img}[r(i \frac{M-1}{N})] \\
                                               &= \frac{M-1}{N} \sum_{i=0}^N \frac{\partial}{\partial c[l]_x} \text{img}[r(i \frac{M-1}{N})] \\
                                               &= \frac{M-1}{N} \sum_{i=0}^N \nabla \text{img}[r(i \frac{M-1}{N})] \frac{\partial r}{\partial c[l]_x}(i \frac{M-1}{N})

The image gradient :math:`\nabla \mathrm{img}` depends only on the image and can therefore be precomputed.
The remaining term is the derivative of the spline itself, which can be derived analytically.

.. math::
    :name: active_contour_model:eq:4

    \frac{\partial r}{\partial c[l]_x}(t) &= \frac{\partial}{\partial c[l]_x} \sum_{k=0}^{M-1}c[k]\Phi(t-k) \\
                                          &= \sum_{k=0}^{M-1}\frac{\partial}{\partial c[l]_x}c[k]\Phi(t-k) \\
                                          &= \sum_{k=0}^{M-1}\delta_{kl}\Phi(t-k) \\
                                          &= \Phi(t-l)

Substituting this result into the expression above gives the gradient of the image energy with respect to each control point.

.. math::
    :name: active_contour_model:eq:5

    \frac{\partial E_{image}}{\partial c[l]_x} &= \frac{M-1}{N} \sum_{i=0}^N \nabla \text{img}[r(i * \frac{M-1}{N})] \frac{\partial r}{\partial c[l]_x}(i * \frac{M-1}{N}) \\
                                               &= \frac{M-1}{N} \sum_{i=0}^N \nabla \text{img}[r(i * \frac{M-1}{N})] \Phi(i\frac{M-1}{N}-l)

This can be written more compactly using matrices.

.. math::

    \nabla_{\mathbf{C}}E_{image} = \frac{M-1}{N} \mathbf{J_C}^\intercal \nabla\mathrm{img}

.. math::

   \mathbf{J_C} = \begin{bmatrix}
     \frac{\partial r}{\partial c[0]}(0) & \frac{\partial r}{\partial c[1]}(0) & \dots & \ \frac{\partial r}{\partial c[M-1]}(0) \\
     \frac{\partial r}{\partial c[0]}(\frac{M-1}{N}) & \frac{\partial r}{\partial c[1]}(\frac{M-1}{N}) & \dots & \ \frac{\partial r}{\partial c[M-1]}(\frac{M-1}{N}) \\
     \vdots & \vdots & \ddots & \vdots \\
     \frac{\partial r}{\partial c[0]}(M - 1) & \frac{\partial r}{\partial c[1]}(M - 1) & \dots & \ \frac{\partial r}{\partial c[M - 1]}(M - 1) \\
    \end{bmatrix}

.. math::

   \nabla\mathrm{img} = \begin{bmatrix}
     \frac{\partial \mathrm{img}}{\partial x}[r(0)] & \dots & \ \frac{\partial \mathrm{img}}{\partial z}[r(0)] \\
     \frac{\partial \mathrm{img}}{\partial x}[r(\frac{M-1}{N})] & \dots & \ \frac{\partial \mathrm{img}}{\partial z}[r(\frac{M-1}{N})] \\
     \vdots & \ddots & \vdots \\
     \frac{\partial \mathrm{img}}{\partial x}[r(M - 1)] & \dots & \ \frac{\partial r}{\partial z}[r(M - 1)] \\
    \end{bmatrix}

Internal energy
---------------

The internal energy acts as a regularization term that controls the shape and parameterization of the spline.
Different choices of internal energy can be used depending on which geometric properties should be encouraged or penalised.

In this implementation, we use the curvilinear reparameterisation energy proposed by [Jacob2004]_.

.. math::
    :name: active_contour_model:eq:6

    E_{internal} &= \int_0^{M-1} ||r'(t)|^2 - c|^2dt \\
                 &= \int_0^{M-1} |r'(t)|^4 - 2c|r'(t)|^2 + c^2 dt


The parameter :math:`c = (\frac{\text{desired arc length}}{M-1})^2` represents the desired squared speed of the spline.

This energy encourages a uniform distribution of control points along the curve by penalising deviations from the desired local speed. As a result, it helps prevent control points from clustering together or becoming unevenly spaced during optimization.

As with the image energy, gradient descent requires the derivatives of the internal energy with respect to the control points.

.. math::
    :name: active_contour_model:eq:8

    \frac{\partial E_{internal}}{\partial c[l]_x} &= \frac{\partial}{\partial c[l]_x} \int_0^{M-1} |r'(t)|^4 - 2c|r'(t)|^2 + c^2 dt \\
                                                  &= \int_0^{M-1} \frac{\partial}{\partial c[l]_x}|r'(t)|^4 - 2c\frac{\partial}{\partial c[l]_x}|r'(t)|^2 dt \\
                                                  &= \int_0^{M-1} \frac{\partial}{\partial c[l]_x}|r'(t)|^4 dt - 2c \int_0^{M-1} \frac{\partial}{\partial c[l]_x}|r'(t)|^2 dt

To compute these derivatives, we first rewrite :math:`|r'(t)|^2` in terms of the spline basis functions and control points.
This representation allows the derivatives to be evaluated analytically.

.. math::
    :name: active_contour_model:eq:9

    |r'(t)|^2 &= \sum_x r'(t)_x^2 \\
              &= \sum_x (\sum_{k=0}^{M-1} c[k]_x \Phi'(t-k))^2 \\
              &= \sum_x \sum_{k=0}^{M-1}\sum_{m=0}^{M-1} c[k]_x c[m]_x \Phi'(t-k) \Phi'(t-m) \\
              &= \sum_{k=0}^{M-1}\sum_{m=0}^{M-1} c[k] c[m] \Phi'(t-k) \Phi'(t-m)

.. math::
    :name: active_contour_model:eq:10

    \frac{\partial}{\partial c[l]_y}|r'(t)|^2 &= \frac{\partial}{\partial c[l]_y} \sum_x \sum_{k=0}^{M-1}\sum_{m=0}^{M-1} c[k]_x c[m]_x \Phi'(t-k) \Phi'(t-m) \\
                                              &= \sum_x \sum_{k=0}^{M-1}\sum_{m=0}^{M-1} \frac{\partial}{\partial c[l]_y} c[k]_x c[m]_x \Phi'(t-k) \Phi'(t-m) \\
                                              &= \sum_{k=0}^{M-1}\sum_{m=0}^{M-1} \frac{\partial}{\partial c[l]_y} c[k]_y c[m]_y \Phi'(t-k) \Phi'(t-m) \\
                                              &= \sum_{m=0}^{M-1}c[m]_y \Phi'(t-l) \Phi'(t-m) + \sum_{k=0}^{M-1}c[k]_y \Phi'(t-k) \Phi'(t-l)\\
                                              &= 2 \sum_{m=0}^{M-1}c[m]_y \Phi'(t-l) \Phi'(t-m)\\
                                              &= 2 \Phi'(t-l) \sum_{m=0}^{M-1}c[m]_y \Phi'(t-m)\\
                                              &= 2 \Phi'(t-l) r_y'(t)

The derivative of :math:`|r'(t)|^2` can then be used to derive the derivative of :math:`|r'(t)|^4` through the chain rule.

.. math::
    :name: active_contour_model:eq:11

    \frac{\partial}{\partial c[l]_y}|r'(t)|^4 &= \frac{\partial}{\partial c[l]_y}(|r'(t)|^2)^2 \\
                                              &= 2|r'(t)|^2\frac{\partial}{\partial c[l]_y}|r'(t)|^2 \\
                                              &= 2(\sum_{k=0}^{M-1}\sum_{m=0}^{M-1} c[k] c[m] \Phi'(t-k) \Phi'(t-m))(2 \sum_{n=0}^{M-1}c[n]_y \Phi'(t-l) \Phi'(t-n)) \\
                                              &= 4\sum_{k=0}^{M-1}\sum_{m=0}^{M-1}\sum_{n=0}^{M-1} c[k] c[m] c[n]_y \Phi'(t-k) \Phi'(t-m) \Phi'(t-l) \Phi'(t-n) \\

For convenience, we introduce the auxiliary quantities :math:`h_1` and :math:`h_2`, which depend only on the spline basis functions.
These quantities can be precomputed once and reused throughout the optimization.

.. math::
    :name: active_contour_model:eq:12

    h_1(k, m, l, n) := \int_0^{M-1}\Phi'(t-k) \Phi'(t-m) \Phi'(t-l) \Phi'(t-n)dt

.. math::
    :name: active_contour_model:eq:13

    h_2(l, m) := \int_0^{M-1}\Phi'(t-l) \Phi'(t-m)dt

Substituting these expressions into the gradient of the internal energy yields the final formula used during gradient descent.

.. math::
    :name: active_contour_model:eq:14

    \frac{\partial E_{internal}}{\partial c[l]_x} &= \int_0^{M-1} \frac{\partial}{\partial c[l]_x}|r'(t)|^4 dt - 2c \int_0^{M-1} \frac{\partial}{\partial c[l]_x}|r'(t)|^2 dt \\
                                                  &= 4\sum_{k=0}^{M-1}\sum_{m=0}^{M-1}\sum_{n=0}^{M-1} c[k] c[m] c[n]_x h_1(k, m, l, n) - 4c \sum_{m=0}^{M-1}c[m]_x h_2(l, m)
