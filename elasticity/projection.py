from petsc4py import PETSc
import mpi4py.MPI as mpi
import numpy as np

class projection:
    def __init__(self, da, A, RBM):
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
        self.D = D

        work1 = da.createLocalVec()
        work2 = da.createLocalVec()

        coarse_vecs= []
        rbm_vecs = RBM.getVecs()
        for i in range(mpi.COMM_WORLD.size):
            for ivec, rbm_vec in enumerate(rbm_vecs):
                coarse_vecs.append(da.createGlobalVec())
                work1.set(0)
                da.globalToLocal(rbm_vec, work2)
                if i == mpi.COMM_WORLD.rank:
                    work1 = work2*Rlocal/Dlocal
                da.localToGlobal(work1, coarse_vecs[-1], addv=PETSc.InsertMode.ADD_VALUES)

        # orthonormalize
        coarse_Avecs = []
        for vec in coarse_vecs:
            coarse_Avecs.append(A*vec)

        for i, vec in enumerate(coarse_vecs):
            alphas = []
            for vec_ in coarse_Avecs[:i]:
                alphas.append(vec.dot(vec_))
            for alpha, vec_ in zip(alphas, coarse_vecs[:i]):
                vec.axpy(-alpha, vec_)
            vec.scale(1./np.sqrt(vec.dot(A*vec)))
            coarse_Avecs[i] = A*vec

        self.coarse_vecs = coarse_vecs
        self.coarse_Avecs = coarse_Avecs
        
        self.work = da.createGlobalVec()

    def apply(self, x):

        self.work.set(0)
        for vec, Avec in zip(self.coarse_vecs, self.coarse_Avecs):
            self.work += Avec.dot(x)*vec
        
        x -= self.work

    def xcoarse(self, rhs):

        self.work.set(0)
        for vec in self.coarse_vecs:
            self.work += vec.dot(rhs)*vec

        return self.work.copy()

    def apply_transpose(self, x):
        self.work.set(0)
        for vec, Avec in zip(self.coarse_vecs, self.coarse_Avecs):
            self.work += vec.dot(x)*Avec
        x -= self.work
