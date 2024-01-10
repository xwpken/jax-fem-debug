# Automatic differentiation

## Formulation

In this tutorial, we compare the derivatives computed by automatic differentiation and finite difference method. The same hyperelastic body as in our hyperelastic example is considered here, i.e., a unit cube with a neo-Hookean solid model. In addition, we have the following definitions:
* $\Omega=(0,1)\times(0,1)\times(0,1)$ (a unit cube)
*  $b=[0, 0, 0]$
* $\Gamma_{D}=(0,1)\times(0,1)\times0$
* $\boldsymbol{u}_{D}=[0,0,\beta]$ 
* $\Gamma_{N_1}=(0,1)\times(0,1)\times1$
* $\boldsymbol{t}_{N_1}=[0, 0, -1000]$
* $\Gamma_{N_2}=\partial\Omega\backslash(\Gamma_{D}\cup\Gamma_{N_1})$
* $\boldsymbol{t}_{N_2}=[0, 0, 0]$

The response function is defined as:

$$
J= \sum_{i=1}^{N_d}u_i^2(\alpha_1,\alpha_2,...\alpha_N)
$$

where $N_d$ is the total number of degrees of freedom. Here, we set up three variables, $\alpha_1 = E$ the elasticity modulus, $\alpha_2 =\rho$ the material density, and $\alpha_3 =\beta$ the scale factor of the Dirichlet boundary conditions.


To calculate the derivative of the response function to the three variables above, we give a small perturbation $h$. With the finite difference method, the derivative can be written as:
$$
\frac{\partial J}{\partial \alpha_i} = \frac{J(\alpha_i+h\alpha_i)-J(\alpha_i)}{h\alpha_i}
$$

The derivative of automatic differentiation will be calculated by the `jax.grad` with custom vector-jacobian product rules, which is already defined in the JAX-FEM.


## Execution
Run
```bash
python -m demos.inverse.example
```
from the `jax-fem/` directory.


## Results

```bash
Derivative comparison between automatic differentiation (AD) and finite difference (FD)
dE = 4.0641751938577116e-07, dE_fd = 0.0, WRONG results! Please avoid gradients w.r.t self.E
drho[0, 0] = 0.002266954599447443, drho_fd_00 = 0.0022666187078357325
dscale_d = 431.59223609853564, dscale_d_fd = 431.80823609844765
```
