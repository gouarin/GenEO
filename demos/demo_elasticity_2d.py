from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)
import mpi4py.MPI as mpi
from petsc4py import PETSc
import numpy as np
from elasticity import *

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
E = buildCellArrayWithFunction(da, lame_coeff, (E1,E2))
nu = buildCellArrayWithFunction(da, lame_coeff, (nu1,nu2))

lamb = (nu*E)/((1+nu)*(1-2*nu)) 
mu = .5*E/(1+nu)

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

        if self.it == 0:
            work, _ = proj.A.getVecs()
            for i, vec in enumerate(proj.coarse_vecs):
                if vec:
                    proj.workl = vec.copy()
                else:
                    proj.workl.set(0.)
                work.set(0)
                proj.scatter_l2g(proj.workl, work, PETSc.InsertMode.ADD_VALUES)

                viewer = PETSc.Viewer().createVTK('coarse_vec_{}.vts'.format(i), 'w', comm = PETSc.COMM_WORLD)
                tmp = self.da.createGlobalVec()
                tmpl_a = self.da.getVecArray(tmp)
                work_a = self.da.getVecArray(work)
                tmpl_a[:] = work_a[:]
                tmp.view(viewer)
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
x.setRandom()
xnorm = b.dot(x)/x.dot(A*x)
x *= xnorm

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType(ksp.Type.PYTHON)
pyKSP = KSP_AMPCG(pcbnn)
pyKSP.callback = callback(da)
ksp.setPythonContext(pyKSP)
ksp.setFromOptions()
ksp.setInitialGuessNonzero(True)

ksp.solve(b, x)

viewer = PETSc.Viewer().createVTK('solution_2d_asm.vts', 'w', comm = PETSc.COMM_WORLD)
x.view(viewer)
