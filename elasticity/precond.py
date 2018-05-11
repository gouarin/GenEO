from .assembling import buildElasticityMatrix
from .bc import bcApplyWestMat, bcApplyWest_vec
from .cg import cg
from .projection import projection, newprojection
from petsc4py import PETSc
from slepc4py import SLEPc
import mpi4py.MPI as mpi
import numpy as np

class ASM(object):
    def __init__(self, da_global, projection, h, lamb, mu):
        self.da_global = da_global
        self.proj = projection

        ranges = self.da_global.getRanges()
        ghost_ranges = self.da_global.getGhostRanges()

        self.block = []
        sizes = []
        for r, gr in zip(ranges, ghost_ranges):
            self.block.append(slice(gr[0], r[1]))
            sizes.append(r[1] - gr[0])
        self.block = tuple(self.block)

        self.da_local = PETSc.DMDA().create(sizes, dof=len(sizes),
                                            stencil_width=1,
                                            comm=PETSc.COMM_SELF)

        mx = self.da_local.getSizes()
        if len(mx) == 2:
            self.da_local.setUniformCoordinates(xmax=h[0]*mx[0], ymax=h[1]*mx[1])
        elif len(mx) == 3:
            self.da_local.setUniformCoordinates(xmax=h[0]*mx[0], ymax=h[1]*mx[1], zmax=h[2]*mx[2])

        A = buildElasticityMatrix(self.da_local, h, lamb, mu)
        A.assemble()

        D = self.da_local.createGlobalVec()
        Dlocal_a = self.da_local.getVecArray(D)
        Dlocal = self.da_global.createLocalVec()
        self.da_global.globalToLocal(self.proj.D, Dlocal)
        D_a = self.da_global.getVecArray(Dlocal)

        Dlocal_a[...] = D_a[self.block]
        A.diagonalScale(D, D)

        if mpi.COMM_WORLD.rank == 0:
            bcApplyWestMat(self.da_local, A)

        self.A = A

        # build local solvers
        self.ksp = PETSc.KSP().create()
        self.ksp.setOperators(A)
        self.ksp.setOptionsPrefix("myasm_")
        self.ksp.setType('preonly')
        # self.ksp.setType('cg')
        pc = self.ksp.getPC()
        # pc.setType('none')
        pc.setType('lu')
        pc.setFactorSolverPackage('mumps')
        pc.setMumpsIcntl(24, 1)
        pc.setMumpsIcntl(25, -1)
        self.ksp.setFromOptions()

        # Construct work arrays
        self.work_global = self.da_global.createLocalVec()
        self.work1_local = self.da_local.createGlobalVec()
        self.work2_local = self.da_local.createGlobalVec()

    def mult(self, mat, x, y):
        self.work_global.set(0.)
        self.da_global.globalToLocal(x, self.work_global)

        work_global_a = self.da_global.getVecArray(self.work_global)

        work1_local_a = self.da_local.getVecArray(self.work1_local)
        work1_local_a[:, :] = work_global_a[self.block]

        work2_local_a = self.da_local.getVecArray(self.work2_local)
        self.ksp.solve(self.work1_local, self.work2_local)

        # self.work2_local = cg(self.A, self.work1_local)

        self.work_global.set(0.)
        sol_a = self.da_local.getVecArray(self.work2_local)

        work_global_a[self.block] = sol_a[:, :]
        y.set(0.)
        self.da_global.localToGlobal(self.work_global, y, addv=PETSc.InsertMode.ADD_VALUES)

        self.proj.project(y)

class MP_ASM(object):
    def __init__(self, A):
        r, _ = A.getLGMap()
        is_A = PETSc.IS().createGeneral(r.indices)

        # convert matis to mpiaij
        A_mpiaij = A.convertISToAIJ()
        A_mpiaij_local = A_mpiaij.createSubMatrices(is_A)[0]

        A_scaled = A.copy().getISLocalMat()
        v1 = A_mpiaij_local.getDiagonal()
        v2 = A_scaled.getDiagonal()
        A_scaled.diagonalScale(v1/v2, v1/v2)

        self.proj = projection(A,is_A,A_scaled,A_mpiaij_local)

        Alocal = A_scaled

        # build local solvers
        self.ksp = PETSc.KSP().create()
        self.ksp.setOperators(Alocal)
        self.ksp.setOptionsPrefix("myasm_")
        self.ksp.setType('preonly')
       # self.ksp.setFromOptions()
        pc = self.ksp.getPC()
        #pc.setType('none')
        pc.setType('cholesky')
        pc.setFactorSolverType('mumps')

        self.proj.constructCoarse(self.ksp)

        self.workl_1 = self.proj.workl.copy()
        self.workl_2 = self.proj.workl.copy()
        self.scatter_l2g = self.proj.scatter_l2g

        # # Construct work arrays
        # self.work_global = self.da_global.createLocalVec()
        # self.work1_local = self.da_local.createGlobalVec()
        # self.work2_local = self.da_local.createGlobalVec()
    def MP_mult(self, x, y):
        self.scatter_l2g(x, self.workl_1, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        self.ksp.solve(self.workl_1, self.workl_2)

        for i in range(mpi.COMM_WORLD.size):
            self.workl_1.set(0)
            if mpi.COMM_WORLD.rank == i:
                self.workl_1 = self.workl_2.copy()

            y[i].set(0.)
            self.scatter_l2g(self.workl_1, y[i], PETSc.InsertMode.ADD_VALUES)
            self.proj.project(y[i])

    def mult(self, x, y):
        self.scatter_l2g(x, self.workl_1, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        self.ksp.solve(self.workl_1, self.workl_2)

        y.set(0.)
        self.scatter_l2g(self.workl_2, y, PETSc.InsertMode.ADD_VALUES)
        self.proj.project(y)

    def apply(self, pc, x, y):
        self.mult(x,y)

class deflated_ASM(object):
    def __init__(self, A):
        r, _ = A.getLGMap()
        is_A = PETSc.IS().createGeneral(r.indices)

        # convert matis to mpiaij
        A_mpiaij = A.convertISToAIJ()
        A_mpiaij_local = A_mpiaij.createSubMatrices(is_A)[0]

        A_scaled = A.copy().getISLocalMat()
        v1 = A_mpiaij_local.getDiagonal()
        v2 = A_scaled.getDiagonal()
        A_scaled.diagonalScale(v1/v2, v1/v2)
        vglobal, _ = A.getVecs()
        vlocal, _ = A_scaled.getVecs()

        #The default local solver is Neumann-Neumann
        Alocal = A_scaled
        self.localksp = PETSc.KSP().create()
        self.localksp.setOperators(Alocal)
        self.localksp.setOptionsPrefix("myasm_")
        self.localksp.setType('preonly')
        localpc = self.localksp.getPC()
        localpc.setType('cholesky')
        localpc.setFactorSolverType('mumps')

        self.A = A
        self.A_scaled = A_scaled
        self.A_mpiaij_local = A_mpiaij_local
        self.workl_1 = vlocal.copy()
        self.workl_2 = self.workl_1.copy()
        self.scatter_l2g = PETSc.Scatter().create(vlocal, None, vglobal, is_A)

    def setup_preconditioners(self,newlocalksp=None, GenEO=1, tauGenEO_lambdamin = 0., tauGenEO_lambdamax = 0.1):
        if newlocalksp:
            self.localksp = newlocalksp
            print('the local ksp has been changed')
        self.proj = newprojection(self,GenEO, tauGenEO_lambdamin, tauGenEO_lambdamax)


    def MP_mult(self, x, y):
        self.scatter_l2g(x, self.workl_1, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        self.localksp.solve(self.workl_1, self.workl_2)

        for i in range(mpi.COMM_WORLD.size):
            self.workl_1.set(0)
            if mpi.COMM_WORLD.rank == i:
                self.workl_1 = self.workl_2.copy()

            y[i].set(0.)
            self.scatter_l2g(self.workl_1, y[i], PETSc.InsertMode.ADD_VALUES)
            self.proj.project(y[i])

    def mult(self, x, y):
        self.scatter_l2g(x, self.workl_1, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        self.localksp.solve(self.workl_1, self.workl_2)

        y.set(0.)
        self.scatter_l2g(self.workl_2, y, PETSc.InsertMode.ADD_VALUES)
        self.proj.project(y)

    def apply(self, pc, x, y):
        self.mult(x,y)


class ASM_old(object):
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

        # bcApplyWestMat(self.da_local, A)

        self.nullspace = PETSc.NullSpace().createRigidBody(self.da_local.getCoordinates())
        u, v, r = self.nullspace.getVecs()
        u[:] /= D[:]
        v[:] /= D[:]
        r[:] /= D[:]
        if mpi.COMM_WORLD.rank == 0:
            bcApplyWest_vec(self.da_local, u)
            bcApplyWest_vec(self.da_local, v)
            bcApplyWest_vec(self.da_local, r)

        # bcApplyWest_vec(self.da_local, u)
        # bcApplyWest_vec(self.da_local, v)
        # bcApplyWest_vec(self.da_local, r)

        # if mpi.COMM_WORLD.rank != 0:
        #     A.setNearNullSpace(self.nullspace)
        #A.setNullSpace(self.nullspace)
        self.A = A

        # build local solvers
        self.ksp = PETSc.KSP().create()
        self.ksp.setOperators(A)
        self.ksp.setOptionsPrefix("myasm_")
        self.ksp.setType('preonly')
        # self.ksp.setType('cg')
        pc = self.ksp.getPC()
        # pc.setType('none')
        pc.setType('lu')
        self.ksp.setFromOptions()

        # Construct work arrays
        self.work_global = self.da_global.createLocalVec()
        self.workg_global = self.da_global.createGlobalVec()
        self.work1_local = self.da_local.createGlobalVec()
        self.work2_local = self.da_local.createGlobalVec()

    def mult(self, x, y):
        self.work_global.set(0.)
        self.da_global.globalToLocal(x, self.work_global)

        work_global_a = self.da_global.getVecArray(self.work_global)

        work1_local_a = self.da_local.getVecArray(self.work1_local)
        work1_local_a[:, :] = work_global_a[self.block]

        work2_local_a = self.da_local.getVecArray(self.work2_local)
        self.ksp.solve(self.work1_local, self.work2_local)

        # self.work2_local = cg(self.A, self.work1_local)

        self.work_global.set(0.)
        sol_a = self.da_local.getVecArray(self.work2_local)

        work_global_a[self.block] = sol_a[:, :]
        y.set(0.)
        self.da_global.localToGlobal(self.work_global, y, addv=PETSc.InsertMode.ADD_VALUES)

        self.workg_global.set(0)
        for vec, Avec in zip(self.vecs, self.Avecs):
            self.workg_global += Avec.dot(y)*vec
        
        y -= self.workg_global

class PC_JACOBI(object):
    def setUp(self, pc):
        B, self.P = pc.getOperators()
        self.diag = self.P.getDiagonal()

    def apply(self, pc, x, y):
        y.array = x/self.diag


class PCASM(object):
    def setUp(self, pc):
        B, self.P = pc.getOperators()

    def apply(self, pc, x, y):
        y.array = self.P*x

class PC_MP_ASM(object):
    def setUp(self, pc):
        B, self.P = pc.getOperators()

    def my_apply(self, x, y):
        self.P.my_mult(x, y)
