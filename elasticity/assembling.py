from __future__ import print_function, division
import numpy as np
from six.moves import range
from petsc4py import PETSc

from . import matelem_cython
from .matelem import getMatElemMass

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

def buildIdentityMatrix(da):
    """
    Assemble the matrix of the elasticity operator
    using Q1 finite elements.

    Parameters
    ==========

    da : petsc.DMDA
        The mesh structure.

    Returns
    =======
    A: petsc.Mat
        The matrix of the elasticity operator.
    """
    A = da.createMatrix()
    elem = da.getElements()

    dof = da.getDof()
    Melem = np.eye(8)
    for e in elem:
        ind = getIndices(e, dof)
        A.setValuesLocal(ind, ind, Melem, PETSc.InsertMode.INSERT_VALUES)

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
