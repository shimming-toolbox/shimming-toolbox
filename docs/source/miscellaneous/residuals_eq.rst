:orphan:

Objective Function Equations for Magnetic Field Shimming Optimization
=====================================================================

This document describes the mathematical objective functions used in optimization algorithms for
magnetic field shimming.

Magnetic field shimming aims to reduce spatial inhomogeneities in a measured magnetic field map by adjusting
the currents in a set of shim coils. Each optimization method minimizes a residual (a numerical measure of the
remaining inhomogeneity) according to a specific objective function.

The variables and notation used in the equations will first be defined, then the residual formulations for each
optimization method (pseudo-inverse, quadratic programming, least squares, BFGS, etc.) will be detailed.

Definitions of symbols
----------------------

The main variables used throughout this document are:

- :math:`A` – *Coil profile matrix* of size (``masked_values``, ``nb_channels``),  where ``masked_values`` (noted :math:`v`) is the number of voxels
  in the region of interest (ROI) and ``nb_channels`` is the number of shim coil channels. Each column corresponds to the spatial
  magnetic field profile (per unit current) for a particular coil channel, evaluated at all voxels in the ROI.

- :math:`u` – *Unshimmed field vector* of size (``masked_values``).
  Contains the measured values of the magnetic field map before shimming, at each voxel in the ROI.

- :math:`m` – *Mask/weight vector* of size (``masked_values``).
  Contains non-zero values for voxels included in the optimization, weighting their contribution to the objective function.

- :math:`x` – *Solution vector* of size (``nb_channels``).
  Contains the coil currents to be induced in each channel.

- :math:`r` – *Residual* (objective function value) to be minimized. This scalar value represents the inhomogeneity after applying the coil currents.

- :math:`f` – Stability factor used to scale the residual term for numerical stability.

- :math:`\lambda` – Regularization vector of size (``nb_channels``).
  Used to penalize large current amplitudes (L2 regularization). Its diagonalized matrix form is :math:`\Lambda`, of size
  (``nb_channels``, ``nb_channels``).

- :math:`m_e` – Eroded mask used in gradient-based signal recovery methods.

- :math:`w_i` – Weighting factor for gradient optimization in direction :math:`i \in \{x, y, z\}`, used in signal recovery.

- :math:`\delta` – Threshold parameter in the pseudo-Huber loss function, controlling transition between quadratic and linear penalty.

Problem formulation
-------------------

The goal is to find :math:`x` that minimizes inhomogeneities in the measured field given by the following linear system :
:math:`m^{T}Ax = -m^{T}u`. This represents the requirement that the contribution from the coils :math:`Ax` exactly cancels the
inhomogeneities present in the field :math:`-u`.

Optimization methods and their objective functions
--------------------------------------------------

1. **Pseudo-inverse method** – No explicit objective function; direct linear algebra solution.
2. **Quadprog method** – Mean squared error (MSE) objective with quadratic programming.
3. **Least squares** and **BFGS** methods – Several possible objective functions:
    1. Mean absolute error (MAE)
    2. Mean squared error (MSE)
    3. MSE with signal recovery
    4. Root mean squared error (RMSE)
    5. RMSE with signal recovery
    6. Mean pseudo-Huber (MPSH)

1. Pseudo-inverse
-----------------

This method solves the system directly:

.. math::

   x = -\sqrt{m}^{T}A^+\sqrt{m}^{T}u

Where :math:`A^+` is the pseudo-inverse of :math:`A` and :math:`\sqrt{m}` is the element-wise square root of :math:`m`.

2. Quadprog
-----------

The objective function is the following :

.. math::

   r = \frac{\sum^v_{i=1} m_i (a^T_i x + u_i)^2}{f \cdot \sum^v_{i=1} m_i}
       + (x^2)^{T} \lambda

The first term is the weighted MSE, scaled by the stability factor :math:`f`, and the second term penalizes large currents. The resulting residual is passed to the
`quadprog <https://github.com/quadprog/quadprog>`_ package to obtain the currents.

3. Least squares and BFGS
-------------------------

These methods share the same objective function formulations, solved using:

- `SciPy's Sequential Least SQuares Programming (SLSQP) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp>`_
- `SciPy's Broyden-Fletcher-Goldfarb-Shanno algorithm (L-BFGS-B) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb>`_

3.1. Mean absolute error (MAE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The objective function is the following :

.. math::

   r = \frac{\sum^v_{i=1} m_i \cdot |a^T_i x + u_i|}{f \cdot \sum^v_{i=1} m_i}
       + (x^2)^T \lambda

3.2. Mean squared error (MSE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A different formulation is used with this objective function to accelerate the compute time. The objective function is the following :

.. math::

   r = x^T a x + x^T b + c

where:

.. math::

   a = \frac{(\sqrt{m}^T A)(\sqrt{m}^T A)}{f \cdot \sum^v_{i=1} m_i} + \Lambda

   b = 2 \cdot \frac{(\sqrt{m}^T u)(\sqrt{m}^T A)}{f \cdot \sum^v_{i=1} m_i}

   c = \frac{(\sqrt{m}^T u)(\sqrt{m}^T u)}{f \cdot \sum^v_{i=1} m_i}

3.3. MSE with signal recovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enhance signal recovery with slice-wise shimming, it is possible to consider the gradients (x-wise, y-wise and z-wise) in the optimization
process. The objective function uses the fast quadratic formulation:

.. math::

   r = x^T a x + x^T b + c

where:

.. math::

   a = a_0 + a_x + a_y + a_z + \Lambda

   b = b_0 + b_x + b_y + b_z

   c = c_0 + c_x + c_y + c_z

The quadratic terms :math:`a_0`, :math:`b_0`, and :math:`c_0` are the same as in the standard MSE objective:

.. math::

   a_0 = \frac{(\sqrt{m}^T A)(\sqrt{m}^T A)}{f \cdot \sum^v_{i=1} m_i} + \Lambda

   b_0 = 2 \cdot \frac{(\sqrt{m}^T u)(\sqrt{m}^T A)}{f \cdot \sum^v_{i=1} m_i}

   c_0 = \frac{(\sqrt{m}^T u)(\sqrt{m}^T u)}{f \cdot \sum^v_{i=1} m_i}

For each gradient direction :math:`i \in \{x, y, z\}`, the additional terms :math:`a_i`, :math:`b_i`, and :math:`c_i` are:

.. math::

   a_i = w_i \cdot \frac{(\sqrt{m_e}^T A_i)(\sqrt{m_e}^T A_i)}{\sum^v_{j=1} m_{e,j}}

   b_i = 2 w_i \cdot \frac{(\sqrt{m_e}^T u_i)(\sqrt{m_e}^T A_i)}{\sum^v_{j=1} m_{e,j}}

   c_i = w_i \cdot \frac{(\sqrt{m_e}^T u_i)(\sqrt{m_e}^T u_i)}{\sum^v_{j=1} m_{e,j}}

Here:

- :math:`A_i` is the coil matrix derived for the field gradient in direction :math:`i`,
- :math:`u_i` is the unshimmed field gradient in direction :math:`i`,

3.4. Root mean squared error (RMSE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The RMSE objective function measures the square root of the weighted MSE, plus a regularization term:

.. math::

   r = \sqrt{\frac{\sum^v_{i=1} m_i (a^T_i x + u_i)^2}{f \cdot \sum^v_{i=1} m_i}}
       + (x^2)^T \lambda

3.5. RMSE with signal recovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This variant extends the RMSE objective by incorporating an additional RMSE term for the through-slice (z-direction) gradient :

.. math::

   r = \sqrt{\frac{\sum^v_{i=1} m_i (a^T_i x + u_i)^2}{f \cdot \sum^v_{i=1} m_i}}
       + w_z \cdot \sqrt{\frac{\sum^v_{i=1} m_{e,i} (a^T_{z,i} x + u_{z,i})^2}{f \cdot \sum^v_{i=1} m_{e,i}}}
       + (x^2)^T \lambda

Here:

- :math:`w_z` controls the contribution from the z-gradient recovery term,
- :math:`A_z` and :math:`u_z` are the z-gradient coil matrix and unshimmed field gradient, respectively.

3.6. Mean pseudo-Huber (MPSH)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this method, a parameter :math:`\delta` determines the threshold between quadratic and linear loss. The objective function behaves quadratically
for residuals smaller than :math:`\delta` and linearly otherwise, resulting in the following expression :

.. math::

   r = \frac{\sum^v_{i=1} m_i \cdot \delta^2
       \left( \sqrt{1 + \left( \frac{a^T_i x + u_i}{\delta} \right)^2} - 1 \right)}
       {f \cdot \sum^v_{i=1} m_i}
       + (x^2)^T \lambda
