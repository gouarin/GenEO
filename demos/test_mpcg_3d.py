from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)
import mpi4py.MPI as mpi
from petsc4py import PETSc
import numpy as np
from elasticity import *

def rhs(coords, rhs):
    rhs[..., 1] = -9.81# + rand

OptDB = PETSc.Options()
Lx = OptDB.getInt('Lx', 10)
Ly = OptDB.getInt('Ly', 1)
Lz = OptDB.getInt('Lz', 1)
n  = OptDB.getInt('n', 16)
nx = OptDB.getInt('nx', Lx*n)
ny = OptDB.getInt('ny', Ly*n)
nz = OptDB.getInt('nz', Lz*n)

hx = Lx/(nx - 1)
hy = Ly/(ny - 1)
hz = Lz/(nz - 1)
h = [hx, hy, hz]

da = PETSc.DMDA().create([nx, ny, nz], dof=3, stencil_width=1)
da.setUniformCoordinates(xmax=Lx, ymax=Ly, zmax=Lz)
da.setMatType(PETSc.Mat.Type.IS)

# constant young modulus
E = 30000
# constant Poisson coefficient
nu = 0.4

lamb = (nu*E)/((1+nu)*(1-2*nu)) 
mu = .5*E/(1+nu)

x = da.createGlobalVec()
b = buildRHS(da, h, rhs)
A = buildElasticityMatrix(da, h, lamb, mu)
A.assemble()

bcApplyWest(da, A, b)

RBM = PETSc.NullSpace().createRigidBody(da.getCoordinates())
rbm_vecs = RBM.getVecs()
for rbm_vec in rbm_vecs:
    bcApplyWest_vec(da, rbm_vec)

#proj = projection(da, A, RBM)

asm = MP_ASM(A)
P = PETSc.Mat().createPython(
    [x.getSizes(), b.getSizes()], comm=da.comm)
P.setPythonContext(asm)
P.setUp()

# Set initial guess
xtild = asm.proj.xcoarse(b)
bcopy = b.copy()
b -= A*xtild

x.setRandom()
asm.proj.project(x)
bcApplyWest_vec(da, x)
xnorm = b.dot(x)/x.dot(A*x)
x *= xnorm

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType(ksp.Type.PYTHON)
ksp.setPythonContext(KSP_AMPCG(asm))
ksp.setFromOptions()

ksp.setInitialGuessNonzero(True)

ksp.solve(b, x)

# norm = (A*x-bcopy).norm()
# if mpi.COMM_WORLD.rank == 0:
#     print(norm)

x += xtild
viewer = PETSc.Viewer().createVTK('solution_3d_asm.vts', 'w', comm = PETSc.COMM_WORLD)
x.view(viewer)

# norm = (A*x-bcopy).norm()
# if mpi.COMM_WORLD.rank == 0:
#     print(norm)
