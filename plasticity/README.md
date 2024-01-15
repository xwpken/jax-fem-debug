# Plasticity

## Formulation

For perfect J2-plasticity model [1], we assume that the total strain $\boldsymbol{\varepsilon}^{n-1}$ and stress $\boldsymbol{\sigma}^{n-1}$ from the previous loading step are known, and the problem states that find the displacement field $\boldsymbol{u}^n$ at the current loading step such that

$$
\begin{align*} 
    -\nabla \cdot \big(\boldsymbol{\sigma}^n (\nabla \boldsymbol{u}^n, \boldsymbol{\varepsilon}^{n-1}, \boldsymbol{\sigma}^{n-1}) \big) = \boldsymbol{b} & \quad \textrm{in}  \nobreakspace \nobreakspace \Omega, \nonumber \\
    \boldsymbol{u}^n = \boldsymbol{u}_D &  \quad\textrm{on} \nobreakspace \nobreakspace \Gamma_D,  \nonumber \\
    \boldsymbol{\sigma}^n \cdot \boldsymbol{n} = \boldsymbol{t}  & \quad \textrm{on} \nobreakspace \nobreakspace \Gamma_N.
\end{align*}
$$

The stress $\boldsymbol{\sigma}^n$ is defined with the following relationships:

```math
\begin{align*}
    \boldsymbol{\sigma}_\textrm{trial} &= \boldsymbol{\sigma}^{n-1} + \Delta \boldsymbol{\sigma}, \nonumber\\
    \Delta \boldsymbol{\sigma} &= \lambda \nobreakspace \textrm{tr}(\Delta \boldsymbol{\varepsilon}) \boldsymbol{I} + 2\mu \nobreakspace \Delta \boldsymbol{\varepsilon}, \nonumber \\
    \Delta \boldsymbol{\varepsilon} &= \boldsymbol{\varepsilon}^n  - \boldsymbol{\varepsilon}^{n-1} = \frac{1}{2}\left[\nabla\boldsymbol{u}^n + (\nabla\boldsymbol{u}^n)^{\top}\right] - \boldsymbol{\varepsilon}^{n-1}, \nonumber\\
    \boldsymbol{s} &= \boldsymbol{\sigma}_\textrm{trial} - \frac{1}{3}\textrm{tr}(\boldsymbol{\sigma}_\textrm{trial})\boldsymbol{I},\nonumber\\
    s &= \sqrt{\frac{3}{2}\boldsymbol{s}:\boldsymbol{s}}, \nonumber\\
    f_{\textrm{yield}} &= s - \sigma_{\textrm{yield}}, \nonumber\\
    \boldsymbol{\sigma}^n &= \boldsymbol{\sigma}_\textrm{trial} -  \frac{\boldsymbol{s}}{s} \langle f_{\textrm{yield}} \rangle_{+}, \nonumber
\end{align*}
```

where $`\boldsymbol{\sigma}_\textrm{trial}`$ is the elastic trial stress, $`\boldsymbol{s}`$ is the devitoric part of $`\boldsymbol{\sigma}_\textrm{trial}`$, $`f_{\textrm{yield}}`$ is the yield function, $`\sigma_{\textrm{yield}}`$ is the yield strength, $`{\langle x \rangle_{+}}:=\frac{1}{2}(x+|x|)`$ is the ramp function, and $`\boldsymbol{\sigma}^n`$ is the stress at the currently loading step.


The weak form gives

$$
\begin{align*}
\int_{\Omega}  \boldsymbol{\sigma}^n : \nabla \boldsymbol{v} \nobreakspace \nobreakspace \textrm{d}x = \int_{\Omega} \boldsymbol{b}  \cdot \boldsymbol{v} \nobreakspace \textrm{d}x + \int_{\Gamma_N} \boldsymbol{t} \cdot \boldsymbol{v} \nobreakspace\nobreakspace \textrm{d}s.
\end{align*}
$$

In this example, we consider a displacement-controlled uniaxial tensile loading condition. We assume free traction ($\boldsymbol{t}=[0, 0, 0]$) and ignore body force ($\boldsymbol{b}=[0,0,0]$). We assume quasi-static loadings from 0 to 0.1 mm and then unload from 0.1 mm to 0.


> :ghost: A remarkable feature of *JAX-FEM* is that automatic differentiation is used to enhance the development efficiency. In this example, deriving the fourth-order elastoplastic tangent moduli tensor $\mathbb{C}=\frac{\partial \boldsymbol{\sigma}^n}{\partial \boldsymbol{\varepsilon}^n}$ is usually required by traditional FEM implementation, but is **NOT** needed in our program due to automatic differentiation.


## Implementation

Import some useful modules
```python
import jax
import jax.numpy as np
import os
import matplotlib.pyplot as plt

from jax_fem.core import FEM
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh
```

Define constitutive relationship. The `get_tensor_map` function overrides base class method. Generally, *JAX-FEM* solves $`-\nabla \cdot \boldsymbol{f}(\nabla \boldsymbol{u}, \boldsymbol{\alpha}_1,\boldsymbol{\alpha}_2,...,\boldsymbol{\alpha}_N) = \boldsymbol{b}`$. Here, we have $`\boldsymbol{f}(\nabla \boldsymbol{u}, \boldsymbol{\alpha}_1,\boldsymbol{\alpha}_2,...,\boldsymbol{\alpha}_N)=\boldsymbol{\sigma}^n (\nabla \boldsymbol{u}^n, \boldsymbol{\varepsilon}^{n-1}, \boldsymbol{\sigma}^{n-1})`$, reflected by the function `stress_return_map`.

```python
class Plasticity(FEM):
    def custom_init(self):
        """Initializing total strain and stress.
        Note that 'laplace' is a reserved keyword that speficically refers to the internal variables.
        """
        self.epsilons_old = np.zeros((len(self.cells), self.num_quads, self.vec, self.dim))
        self.sigmas_old = np.zeros_like(self.epsilons_old)
        self.internal_vars['laplace'] = [self.sigmas_old, self.epsilons_old]

    def get_tensor_map(self):
        """Override base class method.
        """
        _, stress_return_map = self.get_maps()
        return stress_return_map

    def get_maps(self):
        def safe_sqrt(x):  
            """np.sqrt is not differentiable at 0.
            """
            safe_x = np.where(x > 0., np.sqrt(x), 0.)
            return safe_x

        def safe_divide(x, y):
            return np.where(y == 0., 0., x/y)

        def strain(u_grad):
            epsilon = 0.5*(u_grad + u_grad.T)
            return epsilon

        def stress(epsilon):
            E = 70e3
            nu = 0.3
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
            return sigma

        def stress_return_map(u_grad, sigma_old, epsilon_old):
            sig0 = 250.
            epsilon_crt = strain(u_grad)
            epsilon_inc = epsilon_crt - epsilon_old
            sigma_trial = stress(epsilon_inc) + sigma_old
            s_dev = sigma_trial - 1./self.dim*np.trace(sigma_trial)*np.eye(self.dim)
            s_norm = safe_sqrt(3./2.*np.sum(s_dev*s_dev))
            f_yield = s_norm - sig0
            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)
            sigma = sigma_trial - safe_divide(f_yield_plus*s_dev, s_norm)
            return sigma
        return strain, stress_return_map

    def stress_strain_fns(self):
        strain, stress_return_map = self.get_maps()
        vmap_strain = jax.vmap(jax.vmap(strain))
        vmap_stress_return_map = jax.vmap(jax.vmap(stress_return_map))
        return vmap_strain, vmap_stress_return_map

    def update_stress_strain(self, sol):
        u_grads = self.sol_to_grad(sol)
        vmap_strain, vmap_stress_rm = self.stress_strain_fns()
        self.sigmas_old = vmap_stress_rm(u_grads, self.sigmas_old, self.epsilons_old)
        self.epsilons_old = vmap_strain(u_grads)
        self.internal_vars['laplace'] = [self.sigmas_old, self.epsilons_old]

    def compute_avg_stress(self):
        """For post-processing only: Compute volume averaged stress.
        """
        # (num_cells*num_quads, vec, dim) * (num_cells*num_quads, 1, 1) -> (vec, dim)
        sigma = np.sum(self.sigmas_old.reshape(-1, self.vec, self.dim) * self.JxW.reshape(-1)[:, None, None], 0)
        vol = np.sum(self.JxW)
        avg_sigma = sigma/vol
        return avg_sigma
```

Specify mesh-related information. We use first-order hexahedron element.
```python
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly, Lz = 10., 10., 10.
meshio_mesh = box_mesh(Nx=10, Ny=10, Nz=10, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir, ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
```

Define boundary locations:
```python
def top(point):
    return np.isclose(point[2], Lz, atol=1e-5)

def bottom(point):
    return np.isclose(point[2], 0., atol=1e-5)
```

Define Dirichlet boundary values. We fix the z-component of the displacement field to be zero on the bottom side, and control the z-component on the top side.
```python
def dirichlet_val_bottom(point):
    return 0.

def get_dirichlet_top(disp):
    def val_fn(point):
        return disp
    return val_fn

disps = np.hstack((np.linspace(0., 0.1, 11), np.linspace(0.09, 0., 10)))

location_fns = [bottom, top]
value_fns = [dirichlet_val_bottom, get_dirichlet_top(disps[0])]
vecs = [2, 2]

dirichlet_bc_info = [location_fns, vecs, value_fns]
```

Define problem, solve it and save solutions to local folder.
```python
problem = Plasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info)
avg_stresses = []
for i, disp in enumerate(disps):
    print(f"\nStep {i} in {len(disps)}, disp = {disp}")
    dirichlet_bc_info[-1][-1] = get_dirichlet_top(disp)
    problem.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
    sol = solver(problem, use_petsc=True)
    problem.update_stress_strain(sol)
    avg_stress = problem.compute_avg_stress()
    print(avg_stress)
    avg_stresses.append(avg_stress)
    vtk_path = os.path.join(data_dir, f'vtk/u_{i:03d}.vtu')
    save_sol(problem, sol, vtk_path)
avg_stresses = np.array(avg_stresses)
```

Plot the volume-averaged stress versus the vertical displacement of the top surface.
```python
fig = plt.figure(figsize=(10, 8))
plt.plot(disps, avg_stresses[:, 2, 2], color='red', marker='o', markersize=8, linestyle='-') 
plt.xlabel(r'Displacement of top surface [mm]', fontsize=20)
plt.ylabel(r'Volume averaged stress (z-z) [MPa]', fontsize=20)
plt.tick_params(labelsize=18)
plt.show()
```

## Execution
Run
```bash
python -m demos.plasticity.example
```
from the `jax-fem/` directory.


## Results

Results can be visualized with *ParaWiew*.

<p align="middle">
  <img src="materials/sol.gif" width="400" />
</p>
<p align="middle">
    <em >Deformation (x50)</em>
</p>

Plot of the $`z-z`$ component of volume-averaged stress versus displacement of the top surface:


<p align="middle">
  <img src="materials/stress_strain.png" width="500" />
</p>
<p align="middle">
    <em >Stress-strain curve</em>
</p>

## References

[1] Simo, Juan C., and Thomas JR Hughes. *Computational inelasticity*. Vol. 7. Springer Science & Business Media, 2006.