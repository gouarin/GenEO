# Authors:
#     Loic Gouarin <loic.gouarin@cmap.polytechnique.fr>
#     Nicole Spillane <nicole.spillane@cmap.polytechnique.fr>
#
# License: BSD 3 clause
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