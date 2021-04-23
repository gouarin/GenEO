from __future__ import print_function, division
import os
import sys, petsc4py
petsc4py.init(sys.argv)
import mpi4py.MPI as mpi
from petsc4py import PETSc
import numpy as np
from GenEO import *

def rhs(coords, rhs):
    n = rhs.shape
    rhs[..., 1] = -9.81

OptDB = PETSc.Options()
Lx = OptDB.getInt('Lx', 10)
Ly = OptDB.getInt('Ly', 1)
n  = OptDB.getInt('n', 16)
nx = OptDB.getInt('nx', Lx*n)
ny = OptDB.getInt('ny', Ly*n)
E1 = OptDB.getReal('E1', 10**6)
E2 = OptDB.getReal('E2', 1)
nu1 = OptDB.getReal('nu1', 0.4)
nu2 = OptDB.getReal('nu2', 0.4)
randx0 = OptDB.getBool('random_x0',True)

hx = Lx/(nx - 1)
hy = Ly/(ny - 1)

da = PETSc.DMDA().create([nx, ny], dof=2, stencil_width=1)
da.setUniformCoordinates(xmax=Lx, ymax=Ly)
da.setMatType(PETSc.Mat.Type.IS)
da.setFieldName(0, 'u')
da.setFieldName(1, 'v')

path = './output_2d/'
if mpi.COMM_WORLD.rank == 0:
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        os.system('rm {}/*.vts'.format(path))

def lame_coeff(x, y, v1, v2):
    output = np.empty(x.shape)
    mask = np.logical_or(np.logical_and(.2<=y, y<=.4),np.logical_and(.6<=y, y<=.8))
    output[mask] = v1
    output[np.logical_not(mask)] = v2
    return output

# non constant Young's modulus and Poisson's ratio 
E = buildCellArrayWithFunction(da, lame_coeff, (E1,E2))
nu = buildCellArrayWithFunction(da, lame_coeff, (nu1,nu2))

lamb = (nu*E)/((1+nu)*(1-2*nu)) 
mu = .5*E/(1+nu)

def write_simu_info(da, viewer):
    lamb_petsc = da.createGlobalVec()
    lamb_a = da.getVecArray(lamb_petsc)
    coords = da.getCoordinates()
    coords_a = da.getVecArray(coords)
    E = lame_coeff(coords_a[..., 0], coords_a[..., 1], E1, E2)
    nu = lame_coeff(coords_a[..., 0], coords_a[..., 1], nu1, nu2)

    lamb_a[..., 0] = (nu*E)/((1+nu)*(1-2*nu)) 
    lamb_a[..., 1] = mpi.COMM_WORLD.rank
    lamb_petsc.view(viewer)

class callback:
    def __init__(self, da):
        self.da = da
        ranges = da.getRanges()
        ghost_ranges = da.getGhostRanges()
        
        self.slices = []
        for r, gr in zip(ranges, ghost_ranges):
            self.slices.append(slice(gr[0], r[1]))
        self.slices = tuple(self.slices)

        self.it = 0

    def __call__(self, locals):
        pyKSP = locals['self']
        proj = pyKSP.mpc.proj
        x = locals['x']         #current approximate solution      
        r = locals['r']         #current residual 
        z = locals['z']         #current global search direction 
        p = locals['p']         #current local search directions if multipreconditioned iteration 

        if self.it == 0:
            work, _ = proj.A.getVecs()
            for i, vec in enumerate(proj.coarse_vecs):
                if vec:
                    proj.works = vec.copy()
                else:
                    proj.works.set(0.)
                work.set(0)
                proj.scatter_l2g(proj.works, work, PETSc.InsertMode.ADD_VALUES)

                viewer = PETSc.Viewer().createVTK(path + 'coarse_vec_{}.vts'.format(i), 'w', comm = PETSc.COMM_WORLD)
                tmp = self.da.createGlobalVec()
                tmpl_a = self.da.getVecArray(tmp)
                work_a = self.da.getVecArray(work)
                tmpl_a[:] = work_a[:]
                tmp.view(viewer)
                viewer.destroy()

        viewer = PETSc.Viewer().createVTK(path + 'x_at_iteration_{}.vts'.format(self.it), 'w', comm = PETSc.COMM_WORLD)
        tmp = self.da.createGlobalVec()
        tmpl_a = self.da.getVecArray(tmp)
        work_a = self.da.getVecArray(x)
        tmpl_a[:] = work_a[:]
        tmp.view(viewer)
        write_simu_info(self.da, viewer)
        viewer.destroy()

        viewer = PETSc.Viewer().createVTK(path + 'r_at_iteration_{}.vts'.format(self.it), 'w', comm = PETSc.COMM_WORLD)
        tmp = self.da.createGlobalVec()
        tmpl_a = self.da.getVecArray(tmp)
        work_a = self.da.getVecArray(r)
        tmpl_a[:] = work_a[:]
        tmp.view(viewer)
        write_simu_info(self.da, viewer)
        viewer.destroy()

        viewer = PETSc.Viewer().createVTK(path + 'z_at_iteration_{}.vts'.format(self.it), 'w', comm = PETSc.COMM_WORLD)
        tmp = self.da.createGlobalVec()
        tmpl_a = self.da.getVecArray(tmp)
        work_a = self.da.getVecArray(z)
        tmpl_a[:] = work_a[:]
        tmp.view(viewer)
        write_simu_info(self.da, viewer)
        viewer.destroy()

        if isinstance(p[-1], list):
            for i, vec in enumerate(p[-1]):
                viewer = PETSc.Viewer().createVTK(path + 'p_at_iteration_{}_subdomain_{}.vts'.format(self.it,i), 'w', comm = PETSc.COMM_WORLD)
                tmp = self.da.createGlobalVec()
                tmpl_a = self.da.getVecArray(tmp)
                work_a = self.da.getVecArray(vec)
                tmpl_a[:] = work_a[:]
                tmp.view(viewer)
                write_simu_info(self.da, viewer)
                viewer.destroy()
        self.it += 1

x = da.createGlobalVec()
b = buildRHS(da, [hx, hy], rhs)
A = buildElasticityMatrix(da, [hx, hy], lamb, mu)
A.assemble()
bcApplyWest(da, A, b)

#Setup the preconditioner (or multipreconditioner) and the coarse space
pcbnn = PCBNN(A)

# Set initial guess
if randx0:
  x.setRandom()
  xnorm = b.dot(x)/x.dot(A*x)
  x *= xnorm
else:
  x.set(0.)

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType(ksp.Type.PYTHON)
pyKSP = KSP_AMPCG(pcbnn)
pyKSP.callback = callback(da)
ksp.setPythonContext(pyKSP)
ksp.setFromOptions()
ksp.setInitialGuessNonzero(True)

ksp.solve(b, x)

viewer = PETSc.Viewer().createVTK(path + 'solution_2d.vts', 'w', comm = PETSc.COMM_WORLD)
x.view(viewer)
write_simu_info(da, viewer)
viewer.destroy()
