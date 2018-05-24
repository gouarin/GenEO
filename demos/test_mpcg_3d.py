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
E1 = OptDB.getReal('E1', 10**6)
E2 = OptDB.getReal('E2', 1)
nu1 = OptDB.getReal('nu1', 0.4)
nu2 = OptDB.getReal('nu2', 0.4)

hx = Lx/(nx - 1)
hy = Ly/(ny - 1)
hz = Lz/(nz - 1)
h = [hx, hy, hz]

da = PETSc.DMDA().create([nx, ny, nz], dof=3, stencil_width=1)
da.setUniformCoordinates(xmax=Lx, ymax=Ly, zmax=Lz)
da.setMatType(PETSc.Mat.Type.IS)

def lame_coeff(x, y, z, v1, v2):
    output = np.empty(x.shape)
    mask = np.logical_or(np.logical_and(.2<=z, z<=.4),np.logical_and(.6<=z, z<=.8))
    output[mask] = v1
    output[np.logical_not(mask)] = v2
    return output

# non constant Young's modulus and Poisson's ratio 
E = buildCellArrayWithFunction(da, lame_coeff, (E1,E2))
nu = buildCellArrayWithFunction(da, lame_coeff, (nu1, nu2))

lamb = (nu*E)/((1+nu)*(1-2*nu)) 
mu = .5*E/(1+nu)

x = da.createGlobalVec()
b = buildRHS(da, h, rhs)
A = buildElasticityMatrix(da, h, lamb, mu)
A.assemble()

bcApplyWest(da, A, b)
bcopy = b.copy()

pcbnn = PCBNN(A)

# Set initial guess
xtild = pcbnn.proj.coarse_init(b)
b -= A*xtild
x.setRandom()
pcbnn.proj.project(x)
xnorm = b.dot(x)/x.dot(A*x)
x *= xnorm

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType(ksp.Type.PYTHON)
ksp.setPythonContext(KSP_AMPCG(pcbnn))

ksp.setInitialGuessNonzero(True)

ksp.setFromOptions()
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
