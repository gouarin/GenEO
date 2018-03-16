from petsc4py import PETSc
import mpi4py.MPI as mpi
from math import sqrt

def cg(A, b, rtol=1e-5, ite_max=5000):
    x = b.duplicate()
    x.set(0.)

    r = b.copy()
    p = r.copy()
    Ap = p.duplicate()

    rdr = r.dot(r)
    rnorm = sqrt(rdr)
    if rnorm == 0:
        return x

    r0 = rnorm
    ite = 0

    # if mpi.COMM_WORLD.rank == 0:
    #     print(f'ite: {ite} residual -> {rnorm}')

    while rnorm/r0 > rtol and ite < ite_max:
        Ap = A*p
        alpha = rdr/p.dot(Ap)

        x += alpha*p
        r -= alpha*Ap

        beta = 1/rdr
        rdr = r.dot(r)
        beta *= rdr

        p = r + beta*p

        rnorm = sqrt(rdr)
        ite += 1
        # if mpi.COMM_WORLD.rank == 0:
        #     print(f'ite: {ite} residual -> {rnorm}')

    return x