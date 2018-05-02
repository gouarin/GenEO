from petsc4py import PETSc
import mpi4py.MPI as mpi
import numpy as np
from .bc import bcApplyWest_vec

class projection:
    def __init__(self, A):
        r, _ = A.getLGMap()
        # convert matis to mpiaij
        A_mpiaij = A.convertISToAIJ()
        is_A = PETSc.IS().createGeneral(r.indices)

        A_mpiaij_local = A_mpiaij.createSubMatrices(is_A)[0]
        A_scaled = A.copy().getISLocalMat()

        v1 = A_mpiaij_local.getDiagonal()
        v2 = A_scaled.getDiagonal()

        A_scaled.diagonalScale(v1/v2, v1/v2)

        vglobal, _ = A_mpiaij.getVecs()
        vlocal, _ = A_scaled.getVecs()
        self.scatter_l2g = PETSc.Scatter().create(vlocal, None, vglobal, is_A)
        self.A = A
        self.A_scaled = A_scaled

    def constructCoarse(self, ksp, F, nrb):
        # coare_vecs is a list of local vectors
        # coare_Avecs is a list of global vectors

        workl, _ = self.A_scaled.getVecs()

        rbm_vecs = []
        for i in range(nrb):
            F.setMumpsIcntl(25, i+1)
            rbm_vecs.append(workl.duplicate())
            rbm_vecs[i].set(0.)
            ksp.solve(rbm_vecs[i], rbm_vecs[i])

        coarse_vecs= []
        for i in range(mpi.COMM_WORLD.size):
            nrbl = nrb if i == mpi.COMM_WORLD.rank else None
            nrbl = mpi.COMM_WORLD.bcast(nrbl, root=i)
            
            for irbm in range(nrbl):
                coarse_vecs.append(rbm_vecs[irbm] if i == mpi.COMM_WORLD.rank else None)

        n = len(coarse_vecs)

        self.Delta = PETSc.Mat().create(comm=PETSc.COMM_SELF)
        self.Delta.setType(PETSc.Mat.Type.SEQDENSE)
        self.Delta.setSizes([len(coarse_vecs),len(coarse_vecs)])
        self.Delta.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self.Delta.setPreallocationDense(None)

        #scale 
        coarse_Avecs = []
        work, _ = self.A.getVecs()
        for vec in coarse_vecs:
            if vec:
                workl = vec.copy()
            else:
                workl.set(0.)

            work.set(0)
            self.scatter_l2g(workl, work, PETSc.InsertMode.ADD_VALUES)
            coarse_Avecs.append(self.A*work)

            self.scatter_l2g(coarse_Avecs[-1], workl, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
            if vec:
                vec.scale(1./np.sqrt(vec.dot(workl)))
                workl = vec.copy()
            else:
                workl.set(0)

            work.set(0)
            self.scatter_l2g(workl, work, PETSc.InsertMode.ADD_VALUES)
            coarse_Avecs[-1] = self.A*work

        #fill coarse problem matrix
        for i, vec in enumerate(coarse_vecs):
            if vec:
                workl = vec.copy()
            else:
                workl.set(0)

            work.set(0)
            self.scatter_l2g(workl, work, PETSc.InsertMode.ADD_VALUES)
            for j in range(i+1):
                tmp = coarse_Avecs[j].dot(work)
                self.Delta[i, j] = tmp
                self.Delta[j, i] = tmp

        self.Delta.assemble()
        if mpi.COMM_WORLD.rank == 0:
            self.Delta.view()
        self.coarse_vecs = coarse_vecs
        self.coarse_Avecs = coarse_Avecs
        
        self.ksp_Delta = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        self.ksp_Delta.setOperators(self.Delta)
        self.ksp_Delta.setType('preonly')
        pc = self.ksp_Delta.getPC()
        pc.setType('cholesky')

        #self.work = self.da.createGlobalVec()
        self.workl = workl
        self.gamma = PETSc.Vec().create(comm=PETSc.COMM_SELF)
        self.gamma.setType(PETSc.Vec.Type.SEQ)
        self.gamma.setSizes(len(coarse_vecs))
        self.gamma_tmp = self.gamma.duplicate()

    def apply(self, x):
        alpha = self.gamma.duplicate()
        for i, Avec in enumerate(self.coarse_Avecs):
            self.gamma[i] = Avec.dot(x)

        self.ksp_Delta.solve(self.gamma, alpha)

        self.workl.set(0)
        for i, vec in enumerate(self.coarse_vecs):
            if vec:
                self.workl.axpy(-alpha[i], vec)

        self.scatter_l2g(self.workl, x, PETSc.InsertMode.ADD_VALUES)

    def xcoarse(self, rhs):
        alpha = self.gamma.duplicate()

        self.scatter_l2g(rhs, self.workl, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        self.gamma.set(0)
        self.gamma_tmp.set(0)
        for i, vec in enumerate(self.coarse_vecs):
            if vec:
                self.gamma_tmp[i] = vec.dot(self.workl)

        mpi.COMM_WORLD.Allreduce([self.gamma_tmp, mpi.DOUBLE], [self.gamma, mpi.DOUBLE], mpi.SUM)

        self.ksp_Delta.solve(self.gamma, alpha)
        
        self.workl.set(0)
        for i, vec in enumerate(self.coarse_vecs):
            if vec:
                self.workl.axpy(alpha[i], vec)

        out = rhs.duplicate()
        out.set(0)
        self.scatter_l2g(self.workl, out, PETSc.InsertMode.ADD_VALUES)
        return out

    def apply_transpose(self, x):
        alpha = self.gamma.duplicate()

        self.scatter_l2g(x, self.workl, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        self.gamma.set(0)
        self.gamma_tmp.set(0)
        for i, vec in enumerate(self.coarse_vecs):
            if vec:
                self.gamma_tmp[i] = vec.dot(self.workl)

        mpi.COMM_WORLD.Allreduce([self.gamma_tmp, mpi.DOUBLE], [self.gamma, mpi.DOUBLE], mpi.SUM)
        self.ksp_Delta.solve(self.gamma, alpha)
        
        for i in range(len(self.coarse_vecs)):
            x.axpy(-alpha[i], self.coarse_Avecs[i])
