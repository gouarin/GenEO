from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)
import mpi4py.MPI as mpi
from petsc4py import PETSc
import numpy as np
from elasticity import *


def get_nullspace(da, A):
    RBM = PETSc.NullSpace().createRigidBody(da.getCoordinates())
    u, v, r = RBM.getVecs()

    (xs, xe), (ys, ye) = da.getRanges()
    (gxs, gxe), (gys, gye) = da.getGhostRanges()

    R = da.createGlobalVec()
    D = da.createGlobalVec()
    
    Rlocal = da.createLocalVec()
    Dlocal = da.createLocalVec()

    Rlocal_a = da.getVecArray(Rlocal)
    Rlocal_a[gxs: xe, gys:ye] = 1
    
    da.localToGlobal(Rlocal, D, addv=PETSc.InsertMode.ADD_VALUES)
    da.globalToLocal(D, Dlocal)

    size = mpi.COMM_WORLD.size
    rank = mpi.COMM_WORLD.rank
    vecs= []
    for i in range(3*size):
        vecs.append(da.createGlobalVec())

    ulocal = da.createLocalVec()
    vlocal = da.createLocalVec()

    for i in range(size):
        vlocal.set(0)
        da.globalToLocal(u, ulocal)
        if i == rank:
            vlocal = ulocal*Rlocal/Dlocal
        da.localToGlobal(vlocal, vecs[3*i], addv=PETSc.InsertMode.ADD_VALUES)

        vlocal.set(0)
        da.globalToLocal(v, ulocal)
        if i == rank:
            vlocal = ulocal*Rlocal/Dlocal
        da.localToGlobal(vlocal, vecs[3*i+1], addv=PETSc.InsertMode.ADD_VALUES)

        vlocal.set(0)
        da.globalToLocal(r, ulocal)
        if i == rank:
            vlocal = ulocal*Rlocal/Dlocal
        da.localToGlobal(vlocal, vecs[3*i+2], addv=PETSc.InsertMode.ADD_VALUES)

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
    #rhs[..., 1] = -981

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
#bcApplyEast(da, A, b)

D, vecs, Avecs = get_nullspace(da, A)

asm = ASM(da, D, vecs, Avecs, [hx, hy], lamb, mu)
P = PETSc.Mat().createPython(
    [x.getSizes(), b.getSizes()], comm=da.comm)
P.setPythonContext(asm)
P.setUp()

# ksp = PETSc.KSP().create()
# ksp.setOperators(A)
# ksp.setOptionsPrefix("elas_")
# ksp.setType("cg")
# pc = ksp.pc
# pc.setType(None)
# # pc.setType(pc.Type.PYTHON)
# # pc.setPythonContext(PCASM())
# ksp.setFromOptions()

# ksp.solve(b, x)

# viewer = PETSc.Viewer().createVTK('solution_2d_asm_ref.vts', 'w', comm = PETSc.COMM_WORLD)
# x.view(viewer)


# size = mpi.COMM_WORLD.size

# for i in range(3*size):
#     print(i, end=" ")
#     for j in range(3*size):
#         print(vecs[i].dot(Avecs[j]), end=" ")
#     print()
# nullspace = PETSc.NullSpace().create(vectors=vecs)
# A.setNearNullSpace(nullspace)

# nullspace.remove(b)
ptilde = da.createGlobalVec()
for i in range(len(vecs)):
    ptilde += vecs[i].dot(b)*vecs[i]

x = ptilde
bcApplyWest_vec(da, x)

ksp = PETSc.KSP().create()
#ksp.setOperators(A)
ksp.setOperators(A, P)
ksp.setOptionsPrefix("elas_")
ksp.setType("cg")
ksp.setInitialGuessNonzero(True)
pc = ksp.pc
#pc.setType(None)
pc.setType(pc.Type.PYTHON)
pc.setPythonContext(PCASM())
ksp.setFromOptions()

# x[:] = np.random.random(x.size)
# x.assemble()
ksp.solve(b, x)

viewer = PETSc.Viewer().createVTK('solution_2d_asm.vts', 'w', comm = PETSc.COMM_WORLD)
x.view(viewer)