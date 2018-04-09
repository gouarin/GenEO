from __future__ import print_function, division
import numpy as np
from six.moves import range

def bcApplyWest_vec(da, B):
    b = da.getVecArray(B)
    ranges = da.getRanges()
    dim = len(ranges)
    xs = ranges[0][0]
    if xs == 0:
        for i in range(dim):
            b[xs, ..., i] = 0

    # mx, my = da.getSizes()
    # (xs, xe), (ys, ye) = da.getRanges()
    # if xs == 0:
    #     for i in range(ys, ye):
    #         b[xs, i, 0] = 0
    #         b[xs, i, 1] = 0

def bcApplyWestMat(da, A):
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

def bcApplyWest(da, A, B):
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

    b = da.getVecArray(B)
    ranges = da.getRanges()
    xs = ranges[0][0]
    if  xs == 0:
        for i in range(dim):
            b[xs, ..., i] = 0
        #b[xs, ...] = 0
    # if dim == 2:
    #     mx, my = da.getSizes()
    #     (xs, xe), (ys, ye) = da.getRanges()
    #     b = da.getVecArray(B)
    #     if xs == 0:
    #         for i in range(ys, ye):
    #             b[xs, i, 0] = 0
    #             b[xs, i, 1] = 0
    # elif dim == 3:

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
            b[xe-1, i, 0] = 0
            b[xe-1, i, 1] = 0