from petsc4py import PETSc
import mpi4py.MPI as mpi
import numpy as np

class projection:
    def __init__(self, da, A, RBM):
        # (xs, xe), (ys, ye) = da.getRanges()
        # (gxs, gxe), (gys, gye) = da.getGhostRanges()
        ranges = da.getRanges()
        ghost_ranges = da.getGhostRanges()

        # Restriction operator
        R = da.createGlobalVec()
        Rlocal = da.createLocalVec()
        Rlocal_a = da.getVecArray(Rlocal)
        slices = []
        for r, gr in zip(ranges, ghost_ranges):
            slices.append(slice(gr[0], r[1]))
        slices = tuple(slices)
        Rlocal_a[slices] = 1

        # Rlocal_a[gxs:xe, gys:ye] = 1

        # multiplicity
        D = da.createGlobalVec()
        Dlocal = da.createLocalVec()
        da.localToGlobal(Rlocal, D, addv=PETSc.InsertMode.ADD_VALUES)
        da.globalToLocal(D, Dlocal)
        self.D = D


        self.Aglobal = A
        self.da = da

    def setRBM(self, ksp, F, size, nrb, block):
        work = self.da.createLocalVec()

        rbm_vecs = []
        for i in range(nrb):
            F.setMumpsIcntl(25, i+1)
            rbm_vecs.append(PETSc.Vec().createSeq(size))
            ksp.solve(rbm_vecs[-1], rbm_vecs[-1])

        coarse_vecs= []
        for i in range(mpi.COMM_WORLD.size):
            work.set(0.)
            if i == mpi.COMM_WORLD.rank:
                nrbl = nrb
            else:
                nrbl = None
            nrbl = mpi.COMM_WORLD.bcast(nrbl, root=i)
            for irbm in range(nrbl):
                coarse_vecs.append(self.da.createGlobalVec())
                if i == mpi.COMM_WORLD.rank:
                    work_a = self.da.getVecArray(work)
                    work_a[block] = rbm_vecs[irbm].array.reshape(work_a[block].shape)
                self.da.localToGlobal(work, coarse_vecs[-1], addv=PETSc.InsertMode.ADD_VALUES)

        self.Delta = PETSc.Mat().create(comm=PETSc.COMM_SELF)
        self.Delta.setType(PETSc.Mat.Type.SEQDENSE)
        self.Delta.setSizes([len(coarse_vecs),len(coarse_vecs)])
        self.Delta.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self.Delta.setPreallocationDense(None)

        #scale 
        coarse_Avecs = []
        for vec in coarse_vecs:
            coarse_Avecs.append(self.Aglobal*vec)
            vec.scale(1./np.sqrt(vec.dot(self.Aglobal*vec)))
            coarse_Avecs[-1] = self.Aglobal*vec
        #fill coarse problem matrix
        for i, vec in enumerate(coarse_vecs):
            for j in range(i+1):
                tmp = coarse_Avecs[j].dot(vec)
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
        #pc.setFactorSolverPackage('mumps')

        self.work = self.da.createGlobalVec()
        self.gamma = PETSc.Vec().create(comm=PETSc.COMM_SELF)
        self.gamma.setType(PETSc.Vec.Type.SEQ)
        self.gamma.setSizes(len(coarse_vecs))

    def apply(self, x):
        alpha = self.gamma.duplicate()
        for i, Avec in enumerate(self.coarse_Avecs):
            self.gamma[i] = Avec.dot(x)

        self.ksp_Delta.solve(self.gamma, alpha)
        
        for i in range(len(self.coarse_vecs)):
            x.axpy(-alpha[i], self.coarse_vecs[i])
 
    def xcoarse(self, rhs):
        alpha = self.gamma.duplicate()
        for i, vec in enumerate(self.coarse_vecs):
            self.gamma[i] = vec.dot(rhs)

        self.ksp_Delta.solve(self.gamma, alpha)
        
        self.work.set(0)
        for i in range(len(self.coarse_vecs)):
            self.work.axpy(alpha[i], self.coarse_vecs[i])

        return self.work.copy()

    def apply_transpose(self, x):
        alpha = self.gamma.duplicate()
        for i, vec in enumerate(self.coarse_vecs):
            self.gamma[i] = vec.dot(x)

        self.ksp_Delta.solve(self.gamma, alpha)
        
        for i in range(len(self.coarse_vecs)):
            x.axpy(-alpha[i], self.coarse_Avecs[i])
