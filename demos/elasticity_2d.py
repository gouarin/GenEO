from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)
import mpi4py.MPI as mpi
from petsc4py import PETSc
import numpy as np
from elasticity import *

# def rhs(coords, rhs):
#     x = coords[..., 0]
#     mask = x > 9.8
#     rhs[mask, 0] = 0
#     rhs[mask, 1] = -10

def rhs(coords, rhs):
    rhs[..., 1] = -9.81

OptDB = PETSc.Options()
Lx = OptDB.getReal('Lx', 10)
Ly = OptDB.getReal('Ly', 1)
n  = OptDB.getInt('n', 16)
nx = OptDB.getInt('nx', Lx*n)
ny = OptDB.getInt('ny', Ly*n)

hx = Lx/(nx - 1)
hy = Ly/(ny - 1)

da = PETSc.DMDA().create([nx, ny], dof=2, stencil_width=1)
da.setUniformCoordinates(xmax=Lx, ymax=Ly)

# constant young modulus
E = 30000
# constant Poisson coefficient
nu = 0.4

lamb = (nu*E)/((1+nu)*(1-2*nu)) 
mu = .5*E/(1+nu)

# lamb = .25
# mu = .1

x = da.createGlobalVec()
b = buildRHS(da, [hx, hy], rhs)
A = buildElasticityMatrix(da, [hx, hy], lamb, mu)
A.assemble()

# nullspace = PETSc.NullSpace().createRigidBody(da.getCoordinates())
# A.setNearNullSpace(nullspace)

bcApplyWest(da, A, b)

ksp = PETSc.KSP().create()
ksp.setOperators(A)
#ksp.setType("preonly")
#pc = ksp.getPC()
#pc.setType("lu")
ksp.setFromOptions()

ksp.solve(b, x)

# viewer = PETSc.Viewer().createVTK('solution_2d_dir.vts', 'w', comm = PETSc.COMM_WORLD)
# x.view(viewer)

# A = buildElasticityMatrix(da, [hx, hy], lamb, mu)
# A.assemble()

# nullspace = PETSc.NullSpace().createRigidBody(da.getCoordinates())
# A.setNearNullSpace(nullspace)

# b = A*x
# viewer = PETSc.Viewer().createVTK('solution_2d_Ax.vts', 'w', comm = PETSc.COMM_WORLD)
# b.view(viewer)

# x1 = da.createGlobalVec()
# ksp = PETSc.KSP().create()
# ksp.setOperators(A)
# ksp.setType("preonly")
# pc = ksp.getPC()
# pc.setType("lu")
# ksp.setFromOptions()

# ksp.solve(b, x1)

# x2 = da.createGlobalVec()
# ksp = PETSc.KSP().create()
# ksp.setOperators(A)
# ksp.setType("cg")
# pc = ksp.getPC()
# pc.setType(None)
# ksp.setFromOptions()
# ksp.solve(b, x2)

# u, v, r = nullspace.getVecs()
# N = np.concatenate([u, v, r])
# N.shape = 3, x.size

# y1 = da.createGlobalVec()
# y1[:] = x1[:] - N.T@(N@x1[:])
# y2 = da.createGlobalVec()
# y2[:] = x2[:] - N.T@(N@x2[:])

# viewer = PETSc.Viewer().createVTK('solution_2d_y1.vts', 'w', comm = PETSc.COMM_WORLD)
# y1.view(viewer)
# viewer = PETSc.Viewer().createVTK('solution_2d_y2.vts', 'w', comm = PETSc.COMM_WORLD)
# y2.view(viewer)
