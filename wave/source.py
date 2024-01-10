import dolfin
import mshr
import matplotlib.pyplot as plt
import numpy as onp

T = 0.000004 # temporal sampling interval


def FEM_wave_equation(mesh, T, N, neumann_bc=True, c=5000):
    
    # define function space
    V = dolfin.FunctionSpace(mesh, "CG", 1)
    
    # u_D = dolfin.Expression('(x[0]-2)*(x[0]-2)+(x[1]-2)*(x[1]-2) + t*t',degree=2,t=0)
    # # define previous and second-last solution
    # u1 = dolfin.interpolate(u_D, V)
    # u_D.t = T
    # u0 = dolfin.interpolate(u_D, V)
    
    # def boundary(x):
    #     return x[1] < dolfin.DOLFIN_EPS
    
    # bcs = dolfin.DirichletBC(V, u_D, boundary)
    
    
    u1 = dolfin.interpolate(dolfin.Constant(0.0), V)
    u0 = dolfin.interpolate(dolfin.Constant(0.0), V)
    
    bcs = dolfin.DirichletBC(V, dolfin.Constant(1.), "on_boundary")
    
    
    # define variational problem
    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)

    a = dolfin.inner(u, v) * dolfin.dx + dolfin.Constant(T**2 * c**2) * dolfin.inner(dolfin.nabla_grad(u), dolfin.nabla_grad(v)) * dolfin.dx
    L = 2*u1*v * dolfin.dx - u0*v * dolfin.dx

    # compute solution for all time-steps
    u = dolfin.Function(V)
    
    val = onp.zeros((N,2))
    t = T
    for n in range(N):
        # t += T
        # u_D.t = t
        # solve variational problem
        dolfin.solve(a==L, u, bcs)
        u0.assign(u1)
        u1.assign(u)
        plot_soundfield(u,n)
        print(f"No,{n}, Max u = {onp.max(u.vector()[:])}, Min u = {onp.min(u.vector()[:])}")
        val[n,:] = [onp.max(u.vector()[:]), onp.min(u.vector()[:])]
    return u, val


def plot_soundfield(u,i):
    '''plots solution of FEM-based simulation'''
    fig = plt.figure(figsize=(10,10))
    fig = dolfin.plot(u)
    plt.set_cmap('coolwarm')
    plt.grid()
    plt.title(r'$p(\mathbf{x}, t)$')
    plt.xlabel(r'$x$ / m')
    plt.ylabel(r'$y$ / m')
    plt.colorbar(fig, fraction=0.038, pad=0.04);
    plt.savefig(f'./data/{i}.png')
    plt.close()
    
# define geometry and mesh
domain = mshr.Rectangle(dolfin.Point(0, 0), dolfin.Point(1,1))
mesh = mshr.generate_mesh(domain, 200)

# from fe_utils import mshr2txt
# mshr2txt(mesh,1,'wave')


# compute solution
u, val = FEM_wave_equation(mesh, T, 100,neumann_bc=False)



import imageio
import os
def make_gif(dir_path):
    '''
    Reference:
    https://zhuanlan.zhihu.com/p/634614676
    '''
    image_list = []
    file_num = len(os.listdir("./data/"))
    for i in range(0, file_num):
        file_path = os.path.join(dir_path, f'{i}.png')
        im = imageio.imread(file_path)
        image_list.append(im)
    imageio.mimsave('animation.gif',image_list, 'GIF', duration=0.1)

make_gif('./data')
onp.save('./data/val_fenics.npy', val)
