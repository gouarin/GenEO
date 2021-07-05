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
test_case = OptDB.getString('test_case', 'default')
isPCNew = OptDB.getBool('PCNew', True)
computeRitz  =  OptDB.getBool('computeRitz', True)
stripe_nb = OptDB.getInt('stripe_nb', 0)

hx = Lx/(nx - 1)
hy = Ly/(ny - 1)

da = PETSc.DMDA().create([nx, ny], dof=2, stencil_width=1)
da.setUniformCoordinates(xmax=Lx, ymax=Ly)
da.setMatType(PETSc.Mat.Type.IS)

def lame_coeff(x, y, v1, v2, stripe_nb):
    if stripe_nb == 0:
        if mpi.COMM_WORLD.rank == 0:
            print(f'Test number {stripe_nb} - no stripes E = {E1}')
        mask = False
    elif stripe_nb == 1:
        if mpi.COMM_WORLD.rank == 0:
            print(f'Test number {stripe_nb} - one stripe')
        mask = np.logical_and(1./7<=y, y<=2./7)
    elif stripe_nb == 2:
        if mpi.COMM_WORLD.rank == 0:
            print(f'Test number {stripe_nb} - two stripes')
        mask= np.logical_or(np.logical_and(1./7<=y, y<=2./7),np.logical_and(3./7<=y, y<=4./7))
    elif stripe_nb == 3:
        if mpi.COMM_WORLD.rank == 0:
            print(f'Test number {stripe_nb} - three stripes')
        mask= np.logical_or(np.logical_and(1./7<=y, y<=2./7),np.logical_and(3./7<=y, y<=4./7), np.logical_and(5./7<=y, y<=6./7))
    else:
        if mpi.COMM_WORLD.rank == 0:
            print(f'Test number {stripe_nb} is not implemented, instead I set E={E2}')
        mask = True
    output = np.empty(x.shape)
    output[mask] = v1
    output[np.logical_not(mask)] = v2
    return output

# non constant Young's modulus and Poisson's ratio
E = buildCellArrayWithFunction(da, lame_coeff, (E1,E2,stripe_nb))
nu = buildCellArrayWithFunction(da, lame_coeff, (nu1,nu2,stripe_nb))

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
if isPCNew:
    pcbnn = PCNew(A)
else:
    pcbnn = PCBNN(A)

coords = da.getCoordinates()
pcbnn.scatter_l2g(coords, pcbnn.works_1, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
pcbnn.works_1.name = "coordinates"

# lamb_petsc = da.createGlobalVec()
# lamb_a = da.getVecArray(lamb_petsc)
# coords = da.getCoordinates()
# coords_a = da.getVecArray(coords)
# E = lame_coeff(coords_a[..., 0], coords_a[..., 1], E1, E2)
# nu = lame_coeff(coords_a[..., 0], coords_a[..., 1], nu1, nu2)

# print(lamb_a.shape, pcbnn.works_2.array.shape)
# lamb_a[..., 0] = (nu*E)/((1+nu)*(1-2*nu))
# lamb_a[..., 1] = mpi.COMM_WORLD.rank

# pcbnn.works_2.set(0)
# pcbnn.scatter_l2g(lamb_petsc, pcbnn.works_2, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
# pcbnn.works_2.name = "lambda"

for iv, v in enumerate(pcbnn.V0s):
    viewer = PETSc.Viewer().createHDF5(f'output.d/coarse_vec_{iv}_{mpi.COMM_WORLD.rank}.h5', 'w', comm = PETSc.COMM_SELF)
    v.name = "coarse_vec"
    v.view(viewer)
    pcbnn.works_1.view(viewer)
    # pcbnn.works_2.view(viewer)


    # pcbnn.scatter_l2g(E, pcbnn.works_1, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
    # pcbnn.works_1.name = "E"
    viewer.destroy()

import json

prop = {}
prop['E1'] = E1
prop['E2'] = E2
prop['eigs'] = pcbnn.labs

with open(f'output.d/properties_{mpi.COMM_WORLD.rank}.txt', 'w') as outfile:
    json.dump(prop, outfile)