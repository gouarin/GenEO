from .utils import buildVecWithFunction
from.assembling import buildMassMatrix

def buildRHS(da, h, apply_func):
    """
    Construct the right hand side of the elasticity problem.

    Parameters
    ==========

    da : petsc.DMDA
        The mesh structure.

    h : list
        The space step in each direction.

    apply_func: function
        Function corresponding to the f (rhs, or source term) in the 
        elasticity problem.

    Returns
    =======

    b: petsc.Vec
        The right hand side.

    """
    b = da.createGlobalVec()
    A = buildMassMatrix(da, h)
    A.assemble()
    tmp = buildVecWithFunction(da, apply_func)
    A.mult(tmp, b)
    return b
