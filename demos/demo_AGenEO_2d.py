# Authors:
#     Loic Gouarin <loic.gouarin@cmap.polytechnique.fr>
#     Nicole Spillane <nicole.spillane@cmap.polytechnique.fr>
#
# License: BSD 3 clause
from __future__ import print_function, division
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
Lx = OptDB.getInt('Lx', 4)
Ly = OptDB.getInt('Ly', 1)
n  = OptDB.getInt('n', 16)
nx = OptDB.getInt('nx', Lx*n)
ny = OptDB.getInt('ny', Ly*n)
E1 = OptDB.getReal('E1', 10**12)
E2 = OptDB.getReal('E2', 10**6)
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
            for i, vec in enumerate(proj.V0):
                if vec:
                    proj.works = vec.copy()
                else:
                    proj.works.set(0.)
                work.set(0)
                proj.scatter_l2g(proj.works, work, PETSc.InsertMode.ADD_VALUES)

                viewer = PETSc.Viewer().createVTK('output.d/coarse_vec_{}.vts'.format(i), 'w', comm = PETSc.COMM_WORLD)
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
#pcbnn = PCBNN(A)
pcbnn = PCNew(A)

Apos = pcbnn.Apos
############compute x FOR INITIALIZATION OF PCG
# Random initial guess
print('Random rhs')
b.setRandom()

x.setRandom()

xnorm = b.dot(x)/x.dot(Apos*x)
x *= xnorm

#Pre-compute solution in coarse space
#Required for PPCG (projected preconditioner)
#Doesn't hurt or help the hybrid and additive preconditioners
#the initial guess is passed to the PCG below with the option ksp.setInitialGuessNonzero(True)
pcbnn.proj.project(x)
xtild = pcbnn.proj.coarse_init(b)
tmp = xtild.norm()
if mpi.COMM_WORLD.rank == 0:
    print(f'norm xtild (coarse component of solution) {tmp}')
x += xtild
############END of: compute x FOR INITIALIZATION OF PCG

############SETUP KSP
ksp = PETSc.KSP().create()
ksp.setOperators(Apos)
# ksp.setOptionsPrefix("global")

pc = ksp.pc
pc.setType('python')
pc.setPythonContext(pcbnn)
pc.setFromOptions()

ksp.setType("cg")
# #pyKSP.callback = callback(da)
#ksp.setType(ksp.Type.PYTHON)
#pyKSP = KSP_PCG()
#ksp.setPythonContext(pyKSP)

ksp.setInitialGuessNonzero(True)

ksp.setFromOptions()
#### END SETUP KSP

###### SOLVE:
ksp.solve(b, x)

if ksp.getInitialGuessNonzero() == False:
    x+=xtild

Aposx = x.duplicate()
pcbnn.Apos.mult(x,Aposx)
print(f'norm of Apos x - b = {(Aposx - b).norm()}, norm of b = {b.norm()}')

#print('I arrive here')

viewer = PETSc.Viewer().createVTK('solution_2d_asm.vts', 'w', comm = PETSc.COMM_WORLD)
x.view(viewer)
