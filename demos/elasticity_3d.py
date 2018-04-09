from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)
import mpi4py.MPI as mpi
from petsc4py import PETSc
from elasticity import *

def rhs(coords, rhs):
    rhs[..., 1] = -9.81

OptDB = PETSc.Options()
Lx = OptDB.getInt('Lx', 10)
Ly = OptDB.getInt('Ly', 1)
Lz = OptDB.getInt('Lz', 1)
n  = OptDB.getInt('n', 16)
nx = OptDB.getInt('nx', Lx*n)
ny = OptDB.getInt('ny', Ly*n)
nz = OptDB.getInt('ny', Lz*n)

hx = Lx/(nx - 1)
hy = Ly/(ny - 1)
hz = Lz/(nz - 1)

da = PETSc.DMDA().create([nx, ny, nz], dof=3, stencil_width=1)
da.setUniformCoordinates(xmax=Lx, ymax=Ly, zmax=Lz)

# constant young modulus
E = 30000
# constant Poisson coefficient
nu = 0.4

lamb = (nu*E)/((1+nu)*(1-2*nu)) 
mu = .5*E/(1+nu)

x = da.createGlobalVec()
b = buildRHS(da, [hx, hy, hz], rhs)
A = buildElasticityMatrix(da, [hx, hy, hz], lamb, mu)
A.assemble()

bcApplyWest(da, A, b)

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setFromOptions()

ksp.solve(b, x)

viewer = PETSc.Viewer().createVTK('solution_3d.vts', 'w', comm = PETSc.COMM_WORLD)
x.view(viewer)
