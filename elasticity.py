from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)
import numpy as np
import mpi4py.MPI as mpi
from petsc4py import PETSc
import sympy
from six.moves import range

import matelem_cython

from matelem import getMatElemMass

def getMatElemElasticity(h, lamb, mu):
    """
    return the elementary matrix of the elasticity operator
    discretized using Q1 finite element (works in 2d and 3d)

    Parameters
    ==========

    h : list
        The space step in each direction.

    lamb: double or ndarray 
        Lame constant. If lamb is an array, the coefficients are 
        the constant values of this lame constant on each cell.

    mu: double or array 
        Lame constant. If mu is an array, the coefficients are 
        the constant values of this lame constant on each cell.

    Returns
    =======
    out: ndarray
        The elementary matrix of the elasticity operator.
    """
    dim = len(h)
    if dim == 2:
        hx, hy = h
        if isinstance(lamb, np.ndarray):
            return matelem_cython.getMatElemElasticity_2d_nonconst(hx, hy, lamb, mu)
        else:
            return matelem_cython.getMatElemElasticity_2d_const(hx, hy, lamb, mu)
    elif dim == 3:
        hx, hy, hz = h
        if isinstance(lamb, np.ndarray):
            return matelem_cython.getMatElemElasticity_3d_nonconst(hx, hy, hz, lamb, mu)
        else:
            return matelem_cython.getMatElemElasticity_3d_const(hx, hy, hz, lamb, mu)

def getIndices(elem, dof):
    """
    Return matrix indices for each dof for a given element
    using the PETSc numbering.

    Parameters
    ==========

    elem : ndarray
        The list of the mesh elements.
    
    dof : int
        Number of dof (2 in 2d and 3 in 3d).

    Returns
    =======

    ind : ndarray
        The list of the entries in the matrix.
    """
    ind = np.empty(dof*elem.size, dtype=np.int32)
    for i in range(dof):
        ind[i::dof] = dof*elem + i
    return ind

def bcApplyWest(da, A, b):
    """
    Apply boundary conditions on the west side.

    Parameters
    ==========

    da : petsc.DMDA
        The mesh structure.
    
    A : petsc.Mat
        The matrix of the elasticity operator.

    b : petsc.Vec
        The second member.
    
    This function sets all the row entries of the matrix A 
    corresponding to the Dirichlet boundary to 0 and 1 on 
    the diagonal and sets the Dirichlet condition on the 
    second member b.
    """
    dim = da.getDim()
    dof = da.getDof()
    ranges = da.getGhostRanges()
    sizes = np.empty(dim, dtype=np.int32)
    for ir, r in enumerate(ranges):
        sizes[ir] = r[1] - r[0]

    rows = np.empty(0, dtype=np.int32)
    values = np.empty(0)

    if ranges[0][0] == 0:
        if dim == 2:
            rows = np.empty(dim*sizes[1], dtype=np.int32)
            rows[::dof] = dof*np.arange(sizes[1])*sizes[0]
        else:
            rows = np.empty(dim*sizes[1]*sizes[2], dtype=np.int32)
            y = np.arange(sizes[1])
            z = np.arange(sizes[2])*sizes[1]
            rows[::dof] = dof*sizes[0]*(y + z[:, np.newaxis]).flatten()
        for i in range(1, dof):
            rows[i::dof] = rows[::dof] + i

    A.zeroRowsLocal(rows)

    # mx, my = da.getSizes()
    # (xs, xe), (ys, ye) = da.getRanges()
    # b = da.getVecArray(B)
    # if xs == 0:
    #     for i in range(ys, ye):
    #         b[xs, i, 0] = 0
    #         b[xs, i, 1] = 0

def bcApplyEast(da, A, B):
    """
    Apply boundary conditions on the east side.

    Parameters
    ==========

    da : petsc.DMDA
        The mesh structure.
    
    A : petsc.Mat
        The matrix of the elasticity operator.

    b : petsc.Vec
        The second member.
    
    This function sets all the row entries of the matrix A 
    corresponding to the Dirichlet boundary to 0 and 1 on 
    the diagonal and sets the Dirichlet condition on the 
    second member b.
    """
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
    """
    Assemble the matrix of the elasticity operator
    using Q1 finite elements.

    Parameters
    ==========

    da : petsc.DMDA
        The mesh structure.

    h : list
        The space step in each direction.

    lamb: double or ndarray 
        Lame constant. If lamb is an array, the coefficients are 
        the constant values of this lame constant on each cell.

    mu: double or array 
        Lame constant. If mu is an array, the coefficients are 
        the constant values of this lame constant on each cell.

    Returns
    =======
    A: petsc.Mat
        The matrix of the elasticity operator.
    """
    Melem = getMatElemElasticity(h, lamb, mu)
    A = da.createMatrix()
    elem = da.getElements()

    ie = 0
    dof = da.getDof()
    for e in elem:
        ind = getIndices(e, dof)

        if isinstance(lamb, np.ndarray):
            Melem_num = Melem[ie]
        else:
            Melem_num = Melem

        A.setValuesLocal(ind, ind, Melem_num, PETSc.InsertMode.ADD_VALUES)
        ie += 1

    return A

def buildMassMatrix(da, h):
    """
    Assemble the mass matrix using Q1 finite elements.

    Parameters
    ==========

    da : petsc.DMDA
        The mesh structure.

    h : list
        The space step in each direction.

    Returns
    =======
    A: petsc.Mat
        The mass matrix.
    """    
    Melem = getMatElemMass(da.getDim())

    if da.getDim() == 2:
        Melem = Melem(h[0], h[1], 0)
    else:
        Melem = Melem(h[0], h[1], h[2])

    A = da.createMatrix()
    elem = da.getElements()

    dof = da.getDof()
    for e in elem:
        ind = dof*e
        for i in range(dof):
            A.setValuesLocal(ind+i, ind+i, Melem, PETSc.InsertMode.ADD_VALUES)

    return A

def buildRHS(da, h, apply_func):
    """
    Construct the second member of the elasticity problem.

    Parameters
    ==========

    da : petsc.DMDA
        The mesh structure.

    h : list
        The space step in each direction.

    apply_func: function
        Function corresponding to the f (rhs) in the 
        elasticity problem.

    Returns
    =======
    b: petsc.Vec
        The second member.
    """
    b = da.createGlobalVec()
    A = buildMassMatrix(da, h)
    A.assemble()
    tmp = buildVecWithFunction(da, apply_func)
    A.mult(tmp, b)
    return b

def buildVecWithFunction(da, func, extra_args=()):
    """
    Construct a vector using a function applied on
    each point of the mesh.

    Parameters
    ==========

    da : petsc.DMDA
        The mesh structure.

    func: function
        Function to apply on each point.

    extra_args: tuple
        extra parameters of the function.

    Returns
    =======
    b: petsc.Vec
        The vector with the function values on each point.
    """
    OUT = da.createGlobalVec()
    out = da.getVecArray(OUT)

    coords = da.getVecArray(da.getCoordinates())
    if da.getDim() == 2:
        (xs, xe), (ys, ye) = da.getRanges()
        func(coords[xs:xe, ys:ye], out[xs:xe, ys:ye], *extra_args)
    else:
        (xs, xe), (ys, ye), (zs, ze) = da.getRanges()
        func(coords[xs:xe, ys:ye, zs:ze], out[xs:xe, ys:ye, zs:ze], *extra_args)

    return OUT

def buildCellArrayWithFunction(da, func, extra_args=()):
    """
    Construct a vector using a function applied on
    each cell of the mesh.

    Parameters
    ==========

    da : petsc.DMDA
        The mesh structure.

    func: function
        Function to apply on each cell.

    extra_args: tuple
        extra parameters of the function.

    Returns
    =======
    b: petsc.Vec
        The vector with the function values on each cell.
    """
    elem = da.getElements()
    coords = da.getCoordinatesLocal()
    dof = da.getDof()

    x = .5*(coords[dof*elem[:, 0]] + coords[dof*elem[:, 1]])
    y = .5*(coords[dof*elem[:, 0] + 1] + coords[dof*elem[:, 3] + 1])

    if da.getDim() == 2:
        return func(x, y, *extra_args)
    else:
        z= .5*(coords[dof*elem[:, 0] + 2] + coords[dof*elem[:, 4] + 2])
        return func(x, y, z, *extra_args)

OptDB = PETSc.Options()

Lx = OptDB.getInt('Lx', 10)
Ly = OptDB.getInt('Ly', 1)
Lz = OptDB.getInt('Lz', 0)
n  = OptDB.getInt('n', 16)
nx = OptDB.getInt('nx', Lx*n)
ny = OptDB.getInt('ny', Ly*n)
nz = OptDB.getInt('nz', Lz*n)
hx = Lx/(nx - 1)
hy = Ly/(ny - 1)
hz = Lz/(nz - 1)

if Lz == 0:
    da = PETSc.DMDA().create([nx, ny], dof=2, stencil_width=1)
else:
    da = PETSc.DMDA().create([nx, ny, nz], dof=3, stencil_width=1)

da.setUniformCoordinates(xmax=Lx, ymax=Ly, zmax=Lz)
if mpi.COMM_WORLD.rank == 0:
    print(da.getSizes())

# # constant young modulus
# E = 30000
# # constant Poisson coefficient
# nu = 0.4

def g(x, y, z, v1, v2):
    output = np.empty(x.shape)
    mask = np.logical_and(.4<=y, y<=.6)
    output[mask] = v1
    output[np.logical_not(mask)] = v2
    return output


import time
t1 = time.time()
# non constant young modulus
E = buildCellArrayWithFunction(da, g, (100000, 30000))
# non constant Poisson coefficient
nu = buildCellArrayWithFunction(da, g, (0.4, 0.4))
t2 = time.time()
print("build E and nu", t2-t1)

lamb = (nu*E)/((1+nu)*(1-2*nu)) 
mu = .5*E/(1+nu)

x = da.createGlobalVec()

def f(coords, rhs):
    x = coords[..., 0]
    mask = x > 9.8
    rhs[mask, 0] = 0
    rhs[mask, 1] = -10

t1 = time.time()
b = buildRHS(da, [hx, hy, hz], f)
t2 = time.time()
print("build RHS", t2-t1)
#b = da.createGlobalVec()

t1 = time.time()
A = buildElasticityMatrix(da, [hx, hy, hz], lamb, mu)
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
