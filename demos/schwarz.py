from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)
import mpi4py.MPI as mpi
from petsc4py import PETSc
import numpy as np
from elasticity import *


def get_nullspace(da, A):
    RBM = PETSc.NullSpace().createRigidBody(da.getCoordinates())
    rbm_vecs = RBM.getVecs()

    (xs, xe), (ys, ye) = da.getRanges()
    (gxs, gxe), (gys, gye) = da.getGhostRanges()

    # Restriction operator
    R = da.createGlobalVec()
    Rlocal = da.createLocalVec()
    Rlocal_a = da.getVecArray(Rlocal)
    Rlocal_a[gxs:xe, gys:ye] = 1

    # multiplicity
    D = da.createGlobalVec()
    Dlocal = da.createLocalVec()
    da.localToGlobal(Rlocal, D, addv=PETSc.InsertMode.ADD_VALUES)
    da.globalToLocal(D, Dlocal)

    work1 = da.createLocalVec()
    work2 = da.createLocalVec()

    vecs= []
    for i in range(mpi.COMM_WORLD.size):
        for ivec, rbm_vec in enumerate(rbm_vecs):
            vecs.append(da.createGlobalVec())
            work1.set(0)
            da.globalToLocal(rbm_vec, work2)
            if i == mpi.COMM_WORLD.rank:
                work1 = work2*Rlocal/Dlocal
            da.localToGlobal(work1, vecs[-1], addv=PETSc.InsertMode.ADD_VALUES)

    # orthonormalize
    Avecs = []
    for vec in vecs:
        bcApplyWest_vec(da, vec)
        Avecs.append(A*vec)

    for i, vec in enumerate(vecs):
        alphas = []
        for vec_ in Avecs[:i]:
            alphas.append(vec.dot(vec_))
        for alpha, vec_ in zip(alphas, vecs[:i]):
            vec.axpy(-alpha, vec_)
        vec.scale(1./np.sqrt(vec.dot(A*vec)))
        Avecs[i] = A*vec

    return D, vecs, Avecs


def rhs(coords, rhs):
    rhs[..., 1] = -9.81

OptDB = PETSc.Options()
Lx = OptDB.getInt('Lx', 10)
Ly = OptDB.getInt('Ly', 1)
n  = OptDB.getInt('n', 16)
nx = OptDB.getInt('nx', Lx*n)
ny = OptDB.getInt('ny', Ly*n)

hx = Lx/(nx - 1)
hy = Ly/(ny - 1)

da = PETSc.DMDA().create([nx, ny], dof=2, stencil_width=1)
da.setUniformCoordinates(xmax=Lx, ymax=Ly)

# constant young modulus
E = 30000
# constant Poisson coefficient
nu = 0.4

lamb = (nu*E)/((1+nu)*(1-2*nu)) 
mu = .5*E/(1+nu)

x = da.createGlobalVec()
b = buildRHS(da, [hx, hy], rhs)
A = buildElasticityMatrix(da, [hx, hy], lamb, mu)
A.assemble()

bcApplyWest(da, A, b)

# build nullspace and multiplicity
D, vecs, Avecs = get_nullspace(da, A)

asm = ASM(da, D, vecs, Avecs, [hx, hy], lamb, mu)
P = PETSc.Mat().createPython(
    [x.getSizes(), b.getSizes()], comm=da.comm)
P.setPythonContext(asm)
P.setUp()

# Set initial guess
ptilde = da.createGlobalVec()
for i in range(len(vecs)):
    ptilde += vecs[i].dot(b)*vecs[i]
x = ptilde
bcApplyWest_vec(da, x)

ksp = PETSc.KSP().create()
ksp.setOperators(A, P)
ksp.setOptionsPrefix("elas_")
ksp.setType("cg")
ksp.setInitialGuessNonzero(True)
pc = ksp.pc
pc.setType(pc.Type.PYTHON)
pc.setPythonContext(PCASM())
ksp.setFromOptions()

ksp.solve(b, x)

viewer = PETSc.Viewer().createVTK('solution_2d_asm.vts', 'w', comm = PETSc.COMM_WORLD)
x.view(viewer)