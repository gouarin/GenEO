from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)
import numpy as np
import mpi4py.MPI as mpi
from petsc4py import PETSc
import sympy
from six.moves import range

from matelem import getMatElemElasticity, getMatElemMass

def getIndices(elem):
    ind = np.empty(2*elem.size, dtype=np.int32)
    ind[::2] = 2*elem
    ind[1::2] = 2*elem + 1
    return ind

def bcApplyWest(da, A, b):
    sizes = da.getSizes()
    dof = da.getDof()
    ranges = da.getGhostRanges()
    sizes = np.empty(2, dtype=np.int32)
    for ir, r in enumerate(ranges):
        sizes[ir] = r[1] - r[0]

    rows = np.empty(0, dtype=np.int32)
    values = np.empty(0)

    if ranges[0][0] == 0:
        rows = np.empty(2*sizes[1], dtype=np.int32)
        rows[::2] = dof*np.arange(sizes[1])*sizes[0]
        rows[1::2] = rows[::2] + 1
        values = np.zeros(2*sizes[1])

    A.zeroRowsLocal(rows)

    # mx, my = da.getSizes()
    # (xs, xe), (ys, ye) = da.getRanges()
    # b = da.getVecArray(B)
    # if xs == 0:
    #     for i in range(ys, ye):
    #         b[xs, i, 0] = 0
    #         b[xs, i, 1] = 0

def bcApplyEast(da, A, B):
    global_sizes = da.getSizes()
    dof = da.getDof()
    ranges = da.getGhostRanges()
    sizes = np.empty(2, dtype=np.int32)
    for ir, r in enumerate(ranges):
        sizes[ir] = r[1] - r[0]

    rows = np.empty(0, dtype=np.int32)
    values = np.empty(0)

    if ranges[0][1] == global_sizes[0]:
        rows = np.empty(2*sizes[1], dtype=np.int32)
        rows[::2] = dof*(np.arange(sizes[1])*sizes[0]+ sizes[0]-1)
        rows[1::2] = rows[::2] + 1
        values = np.zeros(2*sizes[1])
        values[::2] = 1.
        values[1::2] = -1.

    A.zeroRowsLocal(rows)

    mx, my = da.getSizes()
    (xs, xe), (ys, ye) = da.getRanges()
    b = da.getVecArray(B)
    if xe == mx:
        for i in range(ys, ye):
            b[xe-1, i, 0] = 1
            b[xe-1, i, 1] = -1

def buildElasticityMatrix(da, h, lamb, mu):
    Melem = getMatElemElasticity()
    Melem = Melem(h[0], h[1], lamb, mu)

    A = da.createMatrix()
    elem = da.getElements()

    ie = 0
    for e in elem:
        ind = getIndices(e)

        if isinstance(lamb, np.ndarray):
            Melem_num = Melem[..., ie]
        else:
            Melem_num = Melem

        A.setValuesLocal(ind, ind, Melem_num, PETSc.InsertMode.ADD_VALUES)
        ie += 1

    return A

def buildMassMatrix(da, h):
    Melem = getMatElemMass()
    Melem = Melem(h[0], h[1])

    A = da.createMatrix()
    elem = da.getElements()

    for e in elem:
        ind = 2*e
        A.setValuesLocal(ind, ind, Melem, PETSc.InsertMode.ADD_VALUES)
        A.setValuesLocal(ind+1, ind+1, Melem, PETSc.InsertMode.ADD_VALUES)

    return A

def buildRHS(da, h, apply_func):
    b = da.createGlobalVec()
    A = buildMassMatrix(da, h)
    A.assemble()
    tmp = buildVecWithFunction(da, apply_func)
    A.mult(tmp, b)
    #b.scale(-1) # Not sure that we have to do that
    return b

def buildVecWithFunction(da, func, extra_args=()):
    OUT = da.createGlobalVec()
    out = da.getVecArray(OUT)

    coords = da.getVecArray(da.getCoordinates())
    (xs, xe), (ys, ye) = da.getRanges()
    for i in range(xs, xe):
        for j in range(ys, ye):
            x, y = coords[i, j]
            func(coords[i, j], out[i, j], *extra_args)

    return OUT

def buildCellArrayWithFunction(da, func, extra_args=()):
    elem = da.getElements()
    coords = da.getCoordinates()
    
    output = np.empty(elem.shape[0])
    
    ind = 0
    for e in elem:
        # Compute the middle point of a cell
        x = .5*(coords[2*e[0]] + coords[2*e[1]])
        y = .5*(coords[2*e[0]+1] + coords[2*e[3]+1])
        output[ind] = func(x, y, *extra_args)
        ind += 1

    return output

OptDB = PETSc.Options()

Lx, Ly = 10, 1
n  = OptDB.getInt('n', 16)
nx = OptDB.getInt('nx', Lx*n)
ny = OptDB.getInt('ny', Ly*n)
hx = Lx/(nx - 1)
hy = Ly/(ny - 1)

da = PETSc.DMDA().create([nx, ny], dof=2, stencil_width=1)
da.setUniformCoordinates(xmax=Lx, ymax=Ly)

# # constant young modulus
# E = 30000
# # constant Poisson coefficient
# nu = 0.4

def g(x, y, v1, v2):
    if .4<=y<=.6:
        return v1
    else:
        return v2

# non constant young modulus
E = buildCellArrayWithFunction(da, g, (100000, 30000))
# non constant Poisson coefficient
nu = buildCellArrayWithFunction(da, g, (0.4, 0.4))

lamb = (nu*E)/((1+nu)*(1-2*nu)) 
mu = .5*E/(1+nu)

x = da.createGlobalVec()

def f(coords, rhs):
    x, y = coords
    if x > 9.8:
        rhs[0] = 0
        rhs[1] = -10

b = buildRHS(da, [hx, hy], f)
#b = da.createGlobalVec()

import time
t1 = time.time()
A = buildElasticityMatrix(da, [hx, hy], lamb, mu)
t2 = time.time()
print("assembling", t2-t1)  
A.assemble()

bcApplyWest(da, A, b)
#bcApplyEast(da, A, b)

ksp = PETSc.KSP().create()
ksp.setOperators(A)
# ksp.setType('gmres')
# pc = ksp.getPC()
# pc.setType('none')
ksp.setFromOptions()

ksp.solve(b, x)

viewer = PETSc.Viewer().createVTK('solution.vts', 'w', comm = PETSc.COMM_WORLD)
x.view(viewer)
