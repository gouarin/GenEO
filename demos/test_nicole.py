from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)
import mpi4py.MPI as mpi
from petsc4py import PETSc
import numpy as np
from elasticity import *

def rhs(coords, rhs):
    n = rhs.shape
    rhs[..., 1] = -9.81# + rand

OptDB = PETSc.Options()
Lx = OptDB.getInt('Lx', 10)
Ly = OptDB.getInt('Ly', 1)
n  = OptDB.getInt('n', 16)
nx = OptDB.getInt('nx', Lx*n)
ny = OptDB.getInt('ny', Ly*n)

hx = Lx/(nx - 1)
hy = Ly/(ny - 1)

da = PETSc.DMDA().create([nx, ny], dof=2, stencil_width=1)
da.setUniformCoordinates(xmax=Lx, ymax=Ly)
da.setMatType(PETSc.Mat.Type.IS)

def lame_coeff(x, y, v1, v2):
    output = np.empty(x.shape)
    mask = np.logical_or(np.logical_and(.2<=y, y<=.4),np.logical_and(.6<=y, y<=.8))
    output[mask] = v1
    output[np.logical_not(mask)] = v2
    return output

# non constant Young's modulus and Poisson's ratio 
E = buildCellArrayWithFunction(da, lame_coeff, (10**6,1))
nu = buildCellArrayWithFunction(da, lame_coeff, (0.4, 0.4))

lamb = (nu*E)/((1+nu)*(1-2*nu)) 
mu = .5*E/(1+nu)

x = da.createGlobalVec()
b = buildRHS(da, [hx, hy], rhs)
A = buildElasticityMatrix(da, [hx, hy], lamb, mu)
A.assemble()

bcApplyWest(da, A, b)

#test number 1: BNN+GenEO for large eigenvalues (default)
pcbnn = PCBNN(A)

##test number 2: classical Additive Schwarz + GenEO for small eigenvalues 
#r, _ = A.getLGMap()
#is_A = PETSc.IS().createGeneral(r.indices)
#A_mpiaij = A.convertISToAIJ()
#A_mpiaij_local = A_mpiaij.createSubMatrices(is_A)[0]
#localksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)                                                                    
#localksp.setOperators(A_mpiaij_local) 
#localksp.setType('preonly')                                                                           
#localpc = localksp.getPC()
#localpc.setType('cholesky')
#localpc.setFactorSolverType('mumps')  
#pcbnn = PCBNN(A,newlocalksp=localksp)

####test number 3: diagonal preconditioner + GenEO for small and large eigenvalues
## the coarse space consists of vectors coming from GenEO for the small eigenvalues. The threshold tauGenEO_lambdamin is not reached: all converged eigenvectors are used for the coarse space 
#localksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)                                                                    
#diagofAlocal = pcbnn.A_mpiaij_local.copy()
#diagofAlocal.zeroEntries()
#diagofAlocal.setDiagonal(pcbnn.A_mpiaij_local.getDiagonal())
#localksp.setOperators(diagofAlocal)
#localksp.setType('preonly')                                                                           
#localpc = localksp.getPC()
#localpc.setType('cholesky')
#localpc.setFactorSolverType('mumps')  
#pcb.setup_preconditioners(newlocalksp=localksp,GenEO=1,tauGenEO_lambdamin = 0.1, tauGenEO_lambdamax = 0.1)


# Set initial guess
xtild = pcbnn.proj.xcoarse(b)
bcopy = b.copy()
b -= A*xtild

x.setRandom()
pcbnn.proj.project(x)
xnorm = b.dot(x)/x.dot(A*x)
x *= xnorm

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType(ksp.Type.PYTHON)
ksp.setPythonContext(KSP_MPCG(pcbnn))

ksp.setInitialGuessNonzero(True)

ksp.setFromOptions()

ksp.solve(b, x)

norm = (A*x-b).norm()
if mpi.COMM_WORLD.rank == 0:
    print(f'norm of the projected residual {norm}')

x += xtild
viewer = PETSc.Viewer().createVTK('solution_2d_asm.vts', 'w', comm = PETSc.COMM_WORLD)
x.view(viewer)

norm = (A*x-bcopy).norm()
if mpi.COMM_WORLD.rank == 0:
    print(f'norm of the complete residual {norm}')
