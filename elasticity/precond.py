
from .assembling import buildElasticityMatrix
from petsc4py import PETSc
import mpi4py.MPI as mpi

class ASM(object):

    def __init__(self, da, h, lamb, mu):
        self.da = da
        self.localX  = self.da.createLocalVec()

        mx, my = self.da.getSizes()
        (xs, xe), (ys, ye) = self.da.getRanges()
        (gxs, gxe), (gys, gye) = self.da.getGhostRanges()

        da = PETSc.DMDA().create([xe-gxs, ye-gys], dof=2, 
                                 stencil_width=1,
                                 comm=PETSc.COMM_SELF)
        self.dal = da

        self.xx  = da.createGlobalVec()
        self.rhs  = da.createGlobalVec()
        A = buildElasticityMatrix(da, h, lamb, mu)
        A.assemble()

        lmx, lmy = da.getSizes()
        D = da.createGlobalVec()
        D.set(1.)

        dof = da.getDof()
        if xs != 0:
            for i in range(lmy):
                for d in range(dof):
                    D[dof*lmx*i + d] *= 2
        if xe != mx:
            for i in range(lmy):
                for d in range(dof):
                    D[dof*(lmx-1)+ i*dof*lmx + d] *= 2

        if ys != 0:
            for i in range(lmx):
                for d in range(dof):
                    D[dof*i + d] *= 2
        if ye != my:
            for i in range(lmx):
                for d in range(dof):
                    D[dof*lmx*(lmy-1)+ dof*i + d] *= 2
        
        A.diagonalScale(D, D)

        nullspace = PETSc.NullSpace().createRigidBody(self.da.getCoordinatesLocal())
        A.setNearNullSpace(nullspace)
        self.ksp = PETSc.KSP().create()
        self.ksp.setOperators(A)
        self.ksp.setType('fgmres')
        pc = self.ksp.getPC()
        pc.setType('none')
        self.ksp.setFromOptions()

    def mult(self, mat, X, Y):
        self.localX.set(0.)
        self.da.globalToLocal(X, self.localX)
        (xs, xe), (ys, ye) = self.da.getRanges()
        (gxs, gxe), (gys, gye) = self.da.getGhostRanges()

        localX = self.da.getVecArray(self.localX)

        # if mpi.COMM_WORLD.rank == 0:
        #     print("X", localX[gxs:xe, gys:ye])

        xx = self.dal.getVecArray(self.xx)
        xx[:, :] = localX[gxs:xe, gys:ye]
        
        # if mpi.COMM_WORLD.rank == 0:
        #     print(xx[:, :])
        self.ksp.solve(self.xx, self.rhs)

        self.localX.set(0.)
        rhs = self.dal.getVecArray(self.rhs)
        # if mpi.COMM_WORLD.rank == 0:
        #     print("rhs", rhs[:, :])
        localX[gxs:xe, gys: ye] = rhs[:, :]
        self.da.localToGlobal(self.localX, Y, addv=PETSc.InsertMode.ADD_VALUES)

class PCASM(object):
    def setUp(self, pc):
        B, P = pc.getOperators()
        self.ksp = PETSc.KSP().create()
        self.ksp.setOperators(P)
        self.ksp.setOptionsPrefix("myasm_")
        self.ksp.setType('cg')
        tmp_pc = self.ksp.getPC()
        tmp_pc.setType('none')
        self.ksp.setFromOptions()

    def apply(self, pc, x, y):
        self.ksp.solve(x, y)