
from .assembling import buildElasticityMatrix
from .bc import bcApplyWestMat, bcApplyWest_vec
from petsc4py import PETSc
import mpi4py.MPI as mpi
import numpy as np

class ASM(object):

    def __init__(self, da, D, vecs, Avecs, h, lamb, mu):
        self.da = da
        self.D = D
        self.vecs = vecs
        self.Avecs = Avecs
        
        self.localX  = self.da.createLocalVec()
        self.localY  = self.da.createLocalVec()

        mx, my = self.da.getSizes()
        (xs, xe), (ys, ye) = self.da.getRanges()
        (gxs, gxe), (gys, gye) = self.da.getGhostRanges()

        da = PETSc.DMDA().create([xe-gxs, ye-gys], dof=2, 
                                 stencil_width=1,
                                 comm=PETSc.COMM_SELF)
        mxl, myl = da.getSizes()
        da.setUniformCoordinates(xmax=h[0]*mxl, ymax=h[1]*myl)

        self.dal = da

        self.xx  = da.createGlobalVec()
        self.rhs  = da.createGlobalVec()
        A = buildElasticityMatrix(da, h, lamb, mu)
        A.assemble()

        D = da.createGlobalVec()
        Dlocal_a = da.getVecArray(D)
        print(Dlocal_a.shape)
        Dlocal = self.da.createLocalVec()
        self.da.globalToLocal(self.D, Dlocal)
        D_a = self.da.getVecArray(Dlocal)

        Dlocal_a[:, :, :] = D_a[gxs:xe, gys:ye, :]
        A.diagonalScale(D, D)

        if mpi.COMM_WORLD.rank == 0:
            bcApplyWestMat(self.dal, A)
        # elif mpi.COMM_WORLD.rank == mpi.COMM_WORLD.size-1:
        #     global_sizes = self.dal.getSizes()
        #     dof = self.dal.getDof()
        #     ranges = self.dal.getGhostRanges()
        #     sizes = np.empty(2, dtype=np.int32)
        #     for ir, r in enumerate(ranges):
        #         sizes[ir] = r[1] - r[0]

        #     rows = np.empty(0, dtype=np.int32)

        #     if ranges[0][1] == global_sizes[0]:
        #         rows = np.empty(2*sizes[1], dtype=np.int32)
        #         rows[::2] = dof*(np.arange(sizes[1])*sizes[0]+ sizes[0]-1)
        #         rows[1::2] = rows[::2] + 1

        #     A.zeroRowsLocal(rows)

        self.nullspace = PETSc.NullSpace().createRigidBody(self.dal.getCoordinates())
        u, v, r = self.nullspace.getVecs()
        u[:] /= D[:]
        v[:] /= D[:]
        r[:] /= D[:]
        if mpi.COMM_WORLD.rank == 0:
            bcApplyWest_vec(self.dal, u)
            bcApplyWest_vec(self.dal, v)
            bcApplyWest_vec(self.dal, r)

        A.setNullSpace(self.nullspace)

        self.ksp = PETSc.KSP().create()
        #self.ksp.setComputeSingularValues(True)
        self.ksp.setOperators(A)
        self.ksp.setOptionsPrefix("myasm_")
        #self.ksp.setType('preonly')
        self.ksp.setType('cg')
        pc = self.ksp.getPC()
        #pc.setType('lu')
        pc.setType('none')
        self.ksp.setFromOptions()

    def mult(self, mat, X, Y):
        self.localX.set(0.)
        self.da.globalToLocal(X, self.localX)
        (xs, xe), (ys, ye) = self.da.getRanges()
        (gxs, gxe), (gys, gye) = self.da.getGhostRanges()

        #X.view()
        localX = self.da.getVecArray(self.localX)

        # if mpi.COMM_WORLD.rank == 0:
        #     print("X", localX[gxs:xe, gys:ye])

        xx = self.dal.getVecArray(self.xx)
        xx[:, :] = localX[gxs:xe, gys:ye]

        # u, v, r = self.nullspace.getVecs()
        # N = np.concatenate([u, v, r])
        # N.shape = 3, self.rhs.size
        # y1 = self.dal.createGlobalVec()
        # y1[:] = self.xx[:] - N.T@(N@self.xx[:])
        # self.xx[:] = y1[:]

        # if mpi.COMM_WORLD.rank == 0:
        #     print(xx[:, :])
        # self.xx.view()
        # self.rhs.view()
        A, P = self.ksp.getOperators()
        #(A*self.rhs).view()
        # self.xx.view()
        # print('norm', self.xx.norm())
        # if mpi.COMM_WORLD.rank == 1:
        #     self.xx.view()

        #self.nullspace.remove(self.xx)

        self.ksp.solve(self.xx, self.rhs)
        #self.nullspace.remove(self.rhs)

        # if mpi.COMM_WORLD.rank == 1:
        #     self.rhs.view()

        # projection
        #self.nullspace.remove(self.rhs)
        # u, v, r = self.nullspace.getVecs()
        # N = np.concatenate([u, v, r])
        # N.shape = 3, self.rhs.size

        # y1 = self.dal.createGlobalVec()
        # y1[:] = self.rhs[:] - N.T@(N@self.rhs[:])

        self.localX.set(0.)
        rhs = self.dal.getVecArray(self.rhs)

        
        # if mpi.COMM_WORLD.rank == 0:
        #     print("rhs", rhs[:, :])
        localX[gxs:xe, gys: ye] = rhs[:, :]
        self.da.localToGlobal(self.localX, Y, addv=PETSc.InsertMode.ADD_VALUES)
        # u, v, r = self.nullspace.getVecs()
        # N = np.concatenate([u, v, r])
        # print(xe-xs, ye-ys)
        # N.shape = 3, (xe-xs)*(ye-ys)*2
        # Y[:] = Y[:] - N.T@(N@Y[:])

        ptilde = self.da.createGlobalVec()
        for i in range(len(self.vecs)):
            ptilde += self.Avecs[i].dot(Y)*self.vecs[i]
        
        Y -= ptilde

class PCASM(object):
    def setUp(self, pc):
        B, self.P = pc.getOperators()
        #import ipdb; ipdb.set_trace()
        # self.ksp = PETSc.KSP().create()
        # self.ksp.setOperators(P)
        # self.ksp.setOptionsPrefix("myasm_")
        # self.ksp.setType('cg')
        # tmp_pc = self.ksp.getPC()
        # tmp_pc.setType('none')
        # self.ksp.setFromOptions()

    def apply(self, pc, x, y):
        # self.ksp.solve(x, y)
        # print(x.getSizes(), y.getSizes())
        y.array = self.P*x
        #y.view()
        # z.view()
        #y.array = z
        # print(z.getSizes())
        # import ipdb; ipdb.set_trace()
        #self.P.applyASM(x, y)