# Authors:
#     Loic Gouarin <loic.gouarin@cmap.polytechnique.fr>
#     Nicole Spillane <nicole.spillane@cmap.polytechnique.fr>
#
# License: BSD 3 clause
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

#In order to use PETSc's CG we must initialize it with the solution to the coarse problem
proj = pcbnn.proj
proj.project(x)
xtild = proj.coarse_init(b)
x += xtild

ksp = PETSc.KSP().create()
ksp.setOperators(A)

ksp.setType("cg")
ksp.setFromOptions()
ksp.setInitialGuessNonzero(True)
pc = ksp.pc
pc.setType(pc.Type.PYTHON)
pc.setPythonContext(pcbnn)

ksp.solve(b, x)

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


viewer = PETSc.Viewer().createVTK(path + 'solution_2d.vts', 'w', comm = PETSc.COMM_WORLD)
x.view(viewer)
write_simu_info(da, viewer)
viewer.destroy()
