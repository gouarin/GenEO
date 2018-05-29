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
x.setRandom()
xnorm = b.dot(x)/x.dot(A*x)
x *= xnorm

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType(ksp.Type.PYTHON)
pyKSP = KSP_AMPCG(pcbnn)
pyKSP.callback = callback(da)
ksp.setPythonContext(pyKSP)
ksp.setInitialGuessNonzero(True)
ksp.setFromOptions()

ksp.solve(b, x)

viewer = PETSc.Viewer().createVTK('solution_3d_asm.vts', 'w', comm = PETSc.COMM_WORLD)
x.view(viewer)

lamb_petsc = da.createGlobalVec()
lamb_a = da.getVecArray(lamb_petsc)
coords = da.getCoordinates()
coords_a = da.getVecArray(coords)
E = lame_coeff(coords_a[:, :, :, 0], coords_a[:, :, :, 1], coords_a[:, :, :, 2], E1, E2)
nu = lame_coeff(coords_a[:, :, :, 0], coords_a[:, :, :, 1], coords_a[:, :, :, 2], nu1, nu2)

lamb_a[:, :, :, 0] = (nu*E)/((1+nu)*(1-2*nu)) 
lamb_a[:, :, :, 1] = mpi.COMM_WORLD.rank
lamb_petsc.view(viewer)

viewer.destroy()
