
## Problem formulation

Let $A$ be a coil profile matrix, $u$ a vector containing the values of the magnetic field map before shimming (unshimmed),
$m$ a vector containing the non-zero voxel values of the mask defining and weighting the region of interest, and $x$ a
solution vector of the currents to be induced in the coils. These form the linear system $m^{T}Ax = -m^{T}u$ : the goal is to find the current
values that cancel the inhomogeneities in the magnetic field map, which correspond to the opposite values of $u$. The solution is
the one that minimizes the residual $r$ given by a specific objective function.

All optimization methods except the Pseudo-inverse method also penalize the solutions with high currents. To do so, let's consider a stabiliy factor
$f$ and a regularization vector $\lambda$ used in L2 regularization.

## All optimization methods and their objective functions
1. Pseudo-inverse method - No objective function
2. Quadprog method - Mean squared error (MSE) objective function
3. Least squares and BFGS methods
    1. Mean absolute error (MAE) objective function
    2. Mean squared error (MSE) objective function
    3. MSE signal recovery objective function
    4. Root mean squared error (RMSE) objective function
    5. RMSE signal recovery objective function
    6. Mean pseudo-Huber (MPSH) objective function

## 1. Pseudo-inverse
This optimization method consists of solving the linear system presented in the **Problem formulation** section :
$x = -\sqrt{m}^{T}A^+\sqrt{m}^{T}u$

Where $A^+$ is the pseudo-inverse of the coil matrix $A$ and $\sqrt{m}$ is the $m$ mask vector where a coefficient-wise squared root as been applied.

## 2. Quadprog
The objective function is the following :
$r = \frac{\sum^n_{i=1} m_i(a^T_ix+u_i)^2}{f \cdot \sum^n_{i=1} m_i} + {x^2}^{T}\lambda$

Where $x^2$ is the $x$ coefficients vector where each coefficient has been squared. The resulting residual is given to the
[quadprog](https://github.com/quadprog/quadprog) package to obtain the currents.

## 3. Least squares and BFGS
The least squares and the BFGS methods share the same objective functions. All of these objective functions
are given to the [scipy.optimize.minimize()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize)
function to compute the currents. For the least squares method, it's the
[Sequential Least SQuares Programming (SLSQP) Algorithm (method='SLSQP')](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp)
that is used, whereas the
[Broyden-Fletcher-Goldfarb-Shanno algorithm (method='L-BFGS-B')](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb)
is used for the BFGS method.

### 3.1. Mean absolute error (MAE) objective function
The objective function is the following :
$r = \frac{\sum^n_{i=1} m_i\cdot|a^T_ix+u_i|}{f \cdot \sum^n_{i=1} m_i} + {x^2}^{T}\lambda$

### 3.2. Mean squared error (MSE) objective function
A different formulation is used with this objective function to accelerate the compute time. The objective function is the following :
$r = x^{T}ax + x^Tb + c$

Where $a$, $b$ and $c$ are quadratic coefficients computed in the following way :

$a = \frac{(\sqrt{m}^{T}A)(\sqrt{m}^{T}A)}{f \cdot \sum^n_{i=1} m_i} + \Lambda$

$b = 2 \cdot \frac{(\sqrt{m}^{T}u)(\sqrt{m}^{T}A)}{f \cdot \sum^n_{i=1} m_i}$

$c = \frac{(\sqrt{m}^{T}u)(\sqrt{m}^{T}u)}{f \cdot \sum^n_{i=1} m_i}$

Where $\Lambda$ is the diagonalized matrix equivalent of the $\lambda$ vector.

### 3.3. MSE signal recovery objective function
To enhance signal recovery with slice-wise shimming, it is possible to consider the gradients (x-wise, y-wise and z-wise) in the optimization
process. It is an eroded mask $m_e$ that is used in this case. The objective function is then the following :
$r = x^{T}ax + x^Tb + c$

Where $a$, $b$ and $c$ are computed in the following way :

$a = a_0 + a_x + a_y + a_z + \Lambda$

$b = b_0 + b_x + b_y + b_z$

$c = c_0 + c_x + c_y + c_z$

Where $a_0$, $b_0$ and $c_0$ are the same quadratic terms computed in the standard MSE objective function :

$a_0 = \frac{(\sqrt{m}^{T}A)(\sqrt{m}^{T}A)}{f \cdot \sum^n_{i=1} m_i} + \Lambda$

$b_0 = 2 \cdot \frac{(\sqrt{m}^{T}u)(\sqrt{m}^{T}A)}{f \cdot \sum^n_{i=1} m_i}$

$c_0 = \frac{(\sqrt{m}^{T}u)(\sqrt{m}^{T}u)}{f \cdot \sum^n_{i=1} m_i}$

And where $a_i$, $b_i$ and $c_i$ are the quadratic terms for a gradient in the direction $i$ :

$a_i = w_i \cdot \frac{(\sqrt{m_e}^{T}A_i)(\sqrt{m}^{T}A_i)}{\sum^n_{j=1} m_{e,j}} + \Lambda$

$b_i = 2w_i \cdot \frac{(\sqrt{m_e}^{T}u_i)(\sqrt{m}^{T}A_i)}{\sum^n_{j=1} m_{e,j}}$

$c_i = w_i \cdot \frac{(\sqrt{m_e}^{T}u_i)(\sqrt{m}^{T}u_i)}{\sum^n_{j=1} m_{e,j}}$

Where $w_i$ is a signal loss factor.

### 3.4. Root mean squared error (RMSE) objective function
The objective function is the following :
$r = \sqrt{\frac{\sum^n_{i=1} m_i(a^T_ix+u_i)^2}{f \cdot \sum^n_{i=1} m_i}} + {x^2}^{T}\lambda$

### 3.5. RMSE signal recovery objective function
The objective function is the following :
$r = \sqrt{\frac{\sum^n_{i=1} m_i(a^T_ix+u_i)^2}{f \cdot \sum^n_{i=1} m_i}} + w_z \cdot \sqrt{\frac{\sum^n_{i=1} m_{e,i}(a^T_{z,i}x+u_{z,i})^2}{f \cdot \sum^n_{i=1} m_{e,i}}} + {x^2}^{T}\lambda$

### 3.6. Mean pseudo-Huber (MPSH) objective function
In this method, a parameter $\delta$ determines the threshold between quadratic and linear loss. The objective function behaves quadratically
for residuals smaller than $\delta$, and linearly otherwise, resulting in the following expression :
$r = \frac{\sum^n\_{i=1} m_i\cdot\delta^2\left(\sqrt{1 + \left(\frac{a^T_ix+u_i}{\delta}\right)^2}-1\right)}{f \cdot \sum^n\_{i=1} m\_i} + {x^2}^{T}\lambda$
