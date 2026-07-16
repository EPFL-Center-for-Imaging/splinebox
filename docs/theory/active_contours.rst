Active contour model
====================

The active contour model (also known as a snake) is a method for fitting a spline to image features such as edges, ridges, or object boundaries.
The spline is represented by a set of control points and evolves by minimising an energy functional :math:`E`.

Starting from an initial estimate, the control points are iteratively updated using gradient descent until the energy reaches a local minimum.

.. math::
    :name: active_contour_model:eq:1

    c[l]_x = c[l]_x - \gamma \frac{\partial E}{\partial c[l]_x}

, where :math:`\gamma` is the step size (learning rate).

The energy typically consists of two components:
    (1) **Image energy**, which attracts the spline toward features in the image.
    (2) **Internal energy**, which regularises the spline and discourages undesirable deformations.

.. math::
    :name: active_contour_model:eq:2

    E = E_{image} + E_{internal}

Image energy
------------

The image energy measures the image intensity sampled along the spline. By choosing an appropriate image representation (for example, an edge map), this term can be used to attract the spline toward specific image features.

In practice, the continuous integral is approximated with a Riemann sum by sampling the spline at a finite number of points :math:`t_i`.

.. math::
    :name: active_contour_model:eq:3

    E_{image} &= \int_0^{M-1}\text{img}[\mathbf{r}(t)]dt \\
              &\approx \sum_{i=0}^{N-1} \text{img}[\mathbf{r}(t_i)] \Delta t_i

To optimise the spline, we need the derivatives of the image energy with respect to the control points. Applying the chain rule yields an expression involving the image gradient and the derivative of the spline with respect to a control point.

.. math::
    :name: active_contour_model:eq:4

    \frac{\partial E_{image}}{\partial c[l]_x} &= \frac{\partial}{\partial c[l]_x} \sum_{i=0}^{N-1} \Delta t_i \text{img}[\mathbf{r}(t_i)] \\
                                               &= \sum_{i=0}^{N-1} \Delta t_i \frac{\partial}{\partial c[l]_x} \text{img}[\mathbf{r}(t_i)] \\
                                               &= \sum_{i=0}^{N-1} \Delta t_i \sum_y \frac{\partial \mathrm{img}}{\partial y}[\mathbf{r}(t_i)] \frac{\partial r_y}{\partial c[l]_x} \\
                                               &= \sum_{i=0}^{N-1} \Delta t_i \nabla \text{img}[\mathbf{r}(t_i)] \frac{\partial \mathbf{r}}{\partial c[l]_x}(t_i)

The image gradient :math:`\nabla \mathrm{img}` depends only on the image and can therefore be precomputed.
The remaining term is the derivative of the spline itself, which can be derived analytically.

.. math::
    :name: active_contour_model:eq:5

    \frac{\partial r_y}{\partial c[l]_x}(t) &= \frac{\partial}{\partial c[l]_x} \sum_{k=0}^{M-1} c[k]_y \varphi(t-k) \\
                                            &= \sum_{k=0}^{M-1}\frac{\partial}{\partial c[l]_x} c[k]_y \varphi(t-k) \\
                                            &= \sum_{k=0}^{M-1}\delta_{xy}\delta_{kl}\varphi(t-k) \\
                                            &= \delta_{xy}\varphi(t-l)

Substituting this result into the expression above gives the gradient of the image energy with respect to each control point.

.. math::
    :name: active_contour_model:eq:6

    \frac{\partial E_{image}}{\partial c[l]_x} &= \sum_{i=0}^{N-1} \Delta t_i \nabla \text{img}[\mathbf{r}(t_i)] \frac{\partial \mathbf{r}}{\partial c[l]_x}(t_i) \\
                                               &= \sum_{i=0}^{N-1} \Delta t_i \nabla \text{img}[\mathbf{r}(t_i)] \varphi(t_i-l)

This can be written more compactly using matrices assuming :math:`\Delta t_i` is the same for all :math:`i`.

.. math::
    :name: active_contour_model:eq:7

    \nabla_{\mathbf{C}}E_{image} = \Delta t \mathbf{\Phi}^\intercal \nabla\mathrm{img}

.. math::
    :name: active_contour_model:eq:8

    \mathbf{\Phi} &= \begin{bmatrix}
     \frac{\partial r}{\partial c[0]}(t_0) & \frac{\partial r}{\partial c[1]}(t_0) & \dots & \ \frac{\partial r}{\partial c[M-1]}(t_0) \\
     \frac{\partial r}{\partial c[0]}(t_1) & \frac{\partial r}{\partial c[1]}(t_1) & \dots & \ \frac{\partial r}{\partial c[M-1]}(t_1) \\
     \vdots & \vdots & \ddots & \vdots \\
     \frac{\partial r}{\partial c[0]}(t_{N-1}) & \frac{\partial r}{\partial c[1]}(t_{N-1}) & \dots & \ \frac{\partial r}{\partial c[M - 1]}(t_{N-1}) \\
    \end{bmatrix} \\
    &= \begin{bmatrix}
     \varphi(t_0) & \varphi(t_0-1) & \dots & \ \varphi(t_0-(M-1)) \\
     \varphi(t_1) & \varphi(t_1 - 1) & \dots & \ \varphi(t_1 - (M-1)) \\
     \vdots & \vdots & \ddots & \vdots \\
     \varphi(t_{N-1}) & \varphi(t_{N-1} - 1) & \dots & \ \varphi(t_{N-1} - (M - 1)) \\
    \end{bmatrix}

.. math::
    :name: active_contour_model:eq:9

    \nabla\mathrm{img} = \begin{bmatrix}
     \frac{\partial \mathrm{img}}{\partial x}[\mathbf{r}(t_0)] & \dots & \ \frac{\partial \mathrm{img}}{\partial z}[\mathbf{r}(t_0)] \\
     \frac{\partial \mathrm{img}}{\partial x}[\mathbf{r}(t_1)] & \dots & \ \frac{\partial \mathrm{img}}{\partial z}[\mathbf{r}(t_1)] \\
     \vdots & \ddots & \vdots \\
     \frac{\partial \mathrm{img}}{\partial x}[\mathbf{r}(t_{N-1})] & \dots & \ \frac{\partial \mathrm{img}}{\partial z}[\mathbf{r}(t_{N-1})] \\
    \end{bmatrix}

Internal energy
---------------

The internal energy acts as a regularisation term that controls the shape and parameterization of the spline.
Different choices of internal energy can be used depending on which geometric properties should be encouraged or penalised.

In this implementation, we use the curvilinear reparameterisation energy proposed by [Jacob2004]_.

.. math::
    :name: active_contour_model:eq:10

    E_{internal} &= \int_0^{M-1} ||\mathbf{r}'(t)|^2 - c|^2dt \\
                 &= \int_0^{M-1} |\mathbf{r}'(t)|^4 - 2c|\mathbf{r}'(t)|^2 + c^2 dt


The parameter :math:`c = (\frac{\text{desired arc length}}{M-1})^2` represents the desired squared speed of the spline.

This energy encourages a uniform distribution of control points along the curve by penalising deviations from the desired local speed. As a result, it helps prevent control points from clustering together or becoming unevenly spaced during optimisation.

As with the image energy, gradient descent requires the derivatives of the internal energy with respect to the control points.

.. math::
    :name: active_contour_model:eq:11

    \frac{\partial E_{internal}}{\partial c[l]_x} &= \frac{\partial}{\partial c[l]_x} \int_0^{M-1} |\mathbf{r}'(t)|^4 - 2c|\mathbf{r}'(t)|^2 + c^2 dt \\
                                                  &= \int_0^{M-1} \frac{\partial}{\partial c[l]_x}|\mathbf{r}'(t)|^4 - 2c\frac{\partial}{\partial c[l]_x}|\mathbf{r}'(t)|^2 dt \\
                                                  &= \int_0^{M-1} \frac{\partial}{\partial c[l]_x}|\mathbf{r}'(t)|^4 dt - 2c \int_0^{M-1} \frac{\partial}{\partial c[l]_x}|\mathbf{r}'(t)|^2 dt

To compute these derivatives, we first rewrite :math:`|r'(t)|^2` in terms of the spline basis functions and control points.
This representation allows the derivatives to be evaluated analytically.

.. math::
    :name: active_contour_model:eq:12

    |\mathbf{r}'(t)|^2 &= \sum_x r'(t)_x^2 \\
              &= \sum_x (\sum_{k=0}^{M-1} c[k]_x \varphi'(t-k))^2 \\
              &= \sum_x \sum_{k=0}^{M-1}\sum_{m=0}^{M-1} c[k]_x c[m]_x \varphi'(t-k) \varphi'(t-m) \\
              &= \sum_{k=0}^{M-1}\sum_{m=0}^{M-1} \langle c[k], c[m] \rangle \varphi'(t-k) \varphi'(t-m)

.. math::
    :name: active_contour_model:eq:13

    \frac{\partial}{\partial c[l]_y}|\mathbf{r}'(t)|^2 &= \frac{\partial}{\partial c[l]_y} \sum_x \sum_{k=0}^{M-1}\sum_{m=0}^{M-1} c[k]_x c[m]_x \varphi'(t-k) \varphi'(t-m) \\
                                                       &= \sum_x \sum_{k=0}^{M-1}\sum_{m=0}^{M-1} \frac{\partial}{\partial c[l]_y} c[k]_x c[m]_x \varphi'(t-k) \varphi'(t-m) \\
                                                       &= \sum_{k=0}^{M-1}\sum_{m=0}^{M-1} \frac{\partial}{\partial c[l]_y} c[k]_y c[m]_y \varphi'(t-k) \varphi'(t-m) \\
                                                       &= \sum_{m=0}^{M-1}c[m]_y \varphi'(t-l) \varphi'(t-m) + \sum_{k=0}^{M-1}c[k]_y \varphi'(t-k) \varphi'(t-l)\\
                                                       &= 2 \sum_{m=0}^{M-1}c[m]_y \varphi'(t-l) \varphi'(t-m)\\
                                                       &= 2 \varphi'(t-l) \sum_{m=0}^{M-1}c[m]_y \varphi'(t-m)\\
                                                       &= 2 \varphi'(t-l) r_y'(t)

The derivative of :math:`|r'(t)|^2` can then be used to derive the derivative of :math:`|r'(t)|^4` through the chain rule.

.. math::
    :name: active_contour_model:eq:14

    \frac{\partial}{\partial c[l]_y}|\mathbf{r}'(t)|^4 &= \frac{\partial}{\partial c[l]_y}(|\mathbf{r}'(t)|^2)^2 \\
                                                       &= 2|\mathbf{r}'(t)|^2\frac{\partial}{\partial c[l]_y}|\mathbf{r}'(t)|^2 \\
                                                       &= 2(\sum_{k=0}^{M-1}\sum_{m=0}^{M-1} \langle c[k], c[m] \rangle \varphi'(t-k) \varphi'(t-m))(2 \sum_{n=0}^{M-1}c[n]_y \varphi'(t-l) \varphi'(t-n)) \\
                                                       &= 4\sum_{k=0}^{M-1}\sum_{m=0}^{M-1}\sum_{n=0}^{M-1} \langle c[k], c[m] \rangle c[n]_y \varphi'(t-k) \varphi'(t-m) \varphi'(t-l) \varphi'(t-n) \\

For convenience, we introduce the auxiliary quantities :math:`h_1` and :math:`h_2`, which depend only on the spline basis functions.
These quantities can be precomputed once and reused throughout the optimisation.

.. math::
    :name: active_contour_model:eq:15

    h_1(k, m, l, n) := \int_0^{M-1}\varphi'(t-k) \varphi'(t-m) \varphi'(t-l) \varphi'(t-n)dt

.. math::
    :name: active_contour_model:eq:16

    h_2(l, m) := \int_0^{M-1}\varphi'(t-l) \varphi'(t-m)dt

Substituting these expressions into the gradient of the internal energy yields the final formula used during gradient descent.

.. math::
    :name: active_contour_model:eq:17

    \frac{\partial E_{internal}}{\partial c[l]_x} &= \int_0^{M-1} \frac{\partial}{\partial c[l]_x}|\mathbf{r}'(t)|^4 dt - 2c \int_0^{M-1} \frac{\partial}{\partial c[l]_x}|\mathbf{r}'(t)|^2 dt \\
                                                  &= 4\sum_{k=0}^{M-1}\sum_{m=0}^{M-1}\sum_{n=0}^{M-1} \langle c[k], c[m] \rangle c[n]_x h_1(k, m, l, n) - 4c \sum_{m=0}^{M-1}c[m]_x h_2(l, m)
