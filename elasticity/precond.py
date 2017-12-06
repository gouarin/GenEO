
from .assembling import buildElasticityMatrix
from .bc import bcApplyWestMat, bcApplyWest_vec
from petsc4py import PETSc
import mpi4py.MPI as mpi
import numpy as np

class ASM(object):
    def __init__(self, da_global, D_global, vecs, Avecs, h, lamb, mu):
        self.da_global = da_global
        self.vecs = vecs
        self.Avecs = Avecs

        (xs, xe), (ys, ye) = self.da_global.getRanges()
        (gxs, gxe), (gys, gye) = self.da_global.getGhostRanges()
        self.block = (slice(gxs, xe), slice(gys, ye))

        self.da_local = PETSc.DMDA().create([xe-gxs, ye-gys], dof=2, 
                                            stencil_width=1,
                                            comm=PETSc.COMM_SELF)
        mx, my = self.da_local.getSizes()
        self.da_local.setUniformCoordinates(xmax=h[0]*mx, ymax=h[1]*my)

        A = buildElasticityMatrix(self.da_local, h, lamb, mu)
        A.assemble()

        D = self.da_local.createGlobalVec()
        Dlocal_a = self.da_local.getVecArray(D)
        Dlocal = self.da_global.createLocalVec()
        self.da_global.globalToLocal(D_global, Dlocal)
        D_a = self.da_global.getVecArray(Dlocal)

        Dlocal_a[:, :] = D_a[self.block]
        A.diagonalScale(D, D)

        if mpi.COMM_WORLD.rank == 0:
            bcApplyWestMat(self.da_local, A)

        self.nullspace = PETSc.NullSpace().createRigidBody(self.da_local.getCoordinates())
        u, v, r = self.nullspace.getVecs()
        u[:] /= D[:]
        v[:] /= D[:]
        r[:] /= D[:]
        if mpi.COMM_WORLD.rank == 0:
            bcApplyWest_vec(self.da_local, u)
            bcApplyWest_vec(self.da_local, v)
            bcApplyWest_vec(self.da_local, r)

        A.setNullSpace(self.nullspace)

        # build local solvers
        self.ksp = PETSc.KSP().create()
        self.ksp.setOperators(A)
        self.ksp.setOptionsPrefix("myasm_")
        self.ksp.setType('cg')
        pc = self.ksp.getPC()
        pc.setType('none')
        self.ksp.setFromOptions()

        # Construct work arrays
        self.work_global = self.da_global.createLocalVec()
        self.workg_global = self.da_global.createGlobalVec()
        self.work1_local = self.da_local.createGlobalVec()
        self.work2_local = self.da_local.createGlobalVec()

    def mult(self, mat, x, y):
        self.work_global.set(0.)
        self.da_global.globalToLocal(x, self.work_global)

        work_global_a = self.da_global.getVecArray(self.work_global)

        work1_local_a = self.da_local.getVecArray(self.work1_local)
        work1_local_a[:, :] = work_global_a[self.block]

        self.ksp.solve(self.work1_local, self.work2_local)

        self.work_global.set(0.)
        sol_a = self.da_local.getVecArray(self.work2_local)

        work_global_a[self.block] = sol_a[:, :]
        self.da_global.localToGlobal(self.work_global, y, addv=PETSc.InsertMode.ADD_VALUES)

        self.workg_global.set(0)
        for vec, Avec in zip(self.vecs, self.Avecs):
            self.workg_global += Avec.dot(y)*vec
        
        y -= self.workg_global

class PCASM(object):
    def setUp(self, pc):
        B, self.P = pc.getOperators()

    def apply(self, pc, x, y):
        y.array = self.P*x
