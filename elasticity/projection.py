from petsc4py import PETSc
import mpi4py.MPI as mpi
import numpy as np
from .bc import bcApplyWest_vec
from slepc4py import SLEPc

class projection(object):
    def __init__(self,PCBNN): 
        OptDB = PETSc.Options()                                
        self.nev = OptDB.getInt('PCBNN_GenEO_nev', 10) #for SLEPc: nb of desired eigenvectors to be computed
        self.maxev = OptDB.getInt('PCBNN_GenEO_maxev', 2*self.nev) #max number of eigenvectors allowed to be selected (sometimes SLEPc returns more converged eigenvalues than requested by PCBNN_GenEO_nev)
#        self.condmax = OptDB.getReal('PCBNN_GenEO_condmax', 100)
        self.eigmax = OptDB.getReal('PCBNN_GenEO_eigmax', 10)
        self.GenEO = OptDB.getBool('PCBNN_GenEO', True)
        self.eigmin = OptDB.getReal('PCBNN_GenEO_eigmin', 0.1)

        vglobal, _ = PCBNN.A.getVecs()
        vlocal, _ = PCBNN.A_scaled.getVecs()

        self.scatter_l2g = PCBNN.scatter_l2g 
        self.A = PCBNN.A
        self.A_scaled = PCBNN.A_scaled
        self.A_mpiaij_local = PCBNN.A_mpiaij_local

        workl, _ = self.A_scaled.getVecs()
        self.workl = workl

        self.ksp = PCBNN.localksp
        _,Alocal = self.ksp.getOperators() 
        self.Alocal = Alocal

        if self.ksp.pc.getFactorSolverType() == 'mumps':
            self.ksp.pc.setFactorSetUpSolverType()
            F = self.ksp.pc.getFactorMatrix()
            F.setMumpsIcntl(7, 2)
            F.setMumpsIcntl(24, 1)
            F.setMumpsCntl(3, 1e-6)
            self.ksp.pc.setUp()
            nrb = F.getMumpsInfog(28)

            rbm_vecs = []
            for i in range(nrb):
                F.setMumpsIcntl(25, i+1)
                rbm_vecs.append(workl.duplicate())
                rbm_vecs[i].set(0.)
                self.ksp.solve(rbm_vecs[i], rbm_vecs[i])
            
            F.setMumpsIcntl(25, 0)
            print(f"Subdomain number {mpi.COMM_WORLD.rank} contributes {nrb} coarse vectors as zero energy modes of local solver")
        else:
            rbm_vecs = []
            nrb = 0


        if self.GenEO == True:
            #thresholds for the eigenvalue problems are computed from the user chosen targets for eigmin and eigmax
            tau_eigmin = self.eigmin
            if self.eigmax > 0:
                tau_eigmax = PCBNN.mult_max/self.eigmax 
            else:
                tau_eigmax = 0 
            if self.Alocal != self.A_mpiaij_local:
                self.solve_GenEO_eigmax(rbm_vecs, tau_eigmax)
            else:
                print(f'This is classical additive Schwarz so eigmax = {PCBNN.mult_max}, no eigenvalue problem will be solved for eigmax')
            if self.Alocal != self.A_scaled:
                self.solve_GenEO_eigmin(rbm_vecs, tau_eigmin)
            else:
                print('This is BNN so eigmin = 1, no eigenvalue problem will be solved for eigmin')

        coarse_vecs, coarse_Avecs, Delta, ksp_Delta = self.assemble_coarse_operators(rbm_vecs)

        self.coarse_vecs = coarse_vecs
        self.coarse_Avecs = coarse_Avecs
        self.Delta = Delta
        self.ksp_Delta = ksp_Delta

        self.gamma = PETSc.Vec().create(comm=PETSc.COMM_SELF)
        self.gamma.setType(PETSc.Vec.Type.SEQ)
        self.gamma.setSizes(len(self.coarse_vecs))
        self.gamma_tmp = self.gamma.duplicate()

    def solve_GenEO_eigmax(self, rbm_vecs, tauGenEO_eigmax):
        if tauGenEO_eigmax > 0:
            eps = SLEPc.EPS().create(comm=PETSc.COMM_SELF)
            eps.setDimensions(nev=self.nev)

            eps.setProblemType(SLEPc.EPS.ProblemType.GHIEP)
            eps.setOperators(self.Alocal , self.A_mpiaij_local )
            eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
            eps.setTarget(0.)
            if len(rbm_vecs) > 0 :
                eps.setDeflationSpace(rbm_vecs)
            ST = eps.getST()
            ST.setType("sinvert") 
            ST.setKSP(self.ksp) 
            eps.solve()
            if eps.getConverged() < self.nev:
                print(f"WARNING: Only {eps.getConverged()} eigenvalues converged for GenEO_eigmax in subdomain {mpi.COMM_WORLD.rank} whereas {self.nev} were requested")
            for i in range(min(eps.getConverged(),self.maxev)):
                if(abs(eps.getEigenvalue(i))<tauGenEO_eigmax): #TODO tell slepc that the eigenvalues are real
                    rbm_vecs.append(self.workl.duplicate())
                    eps.getEigenvector(i,rbm_vecs[-1])
                #print(mpi.COMM_WORLD.rank, eps.getEigenvalue(i))        
            self.eps_eigmin=eps #TODO FIX the only reason for this line is to make sure self.ksp and hence PCBNN.ksp is not destroyed  
        print(f"Subdomain number {mpi.COMM_WORLD.rank} contributes {len(rbm_vecs)} coarse vectors after first GenEO")
    
    def solve_GenEO_eigmin(self, rbm_vecs, tauGenEO_eigmin):
        if tauGenEO_eigmin > 0:
            #to compute the smallest eigenvalues of the preconditioned matrix, A_scaled must be factorized
            tempksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            tempksp.setOperators(self.A_scaled)
            tempksp.setType('preonly')
            temppc = tempksp.getPC()
            temppc.setType('cholesky')
            temppc.setFactorSolverType('mumps')
            temppc.setFactorSetUpSolverType()
            tempF = temppc.getFactorMatrix()
            tempF.setMumpsIcntl(7, 2)
            tempF.setMumpsIcntl(24, 1)
            tempF.setMumpsCntl(3, 1e-6)
            temppc.setUp()
            tempnrb = tempF.getMumpsInfog(28)

            for i in range(tempnrb):
                tempF.setMumpsIcntl(25, i+1)
                rbm_vecs.append(self.workl.duplicate())
                rbm_vecs[-1].set(0.)
                tempksp.solve(rbm_vecs[-1], rbm_vecs[-1])
            
            tempF.setMumpsIcntl(25, 0)
            print(f"Subdomain number {mpi.COMM_WORLD.rank} contributes {tempnrb} coarse vectors as zero energy modes of local operator")

            #Eigenvalue Problem for smallest eigenvalues
            eps = SLEPc.EPS().create(comm=PETSc.COMM_SELF)
            eps.setDimensions(nev=self.nev)

            eps.setProblemType(SLEPc.EPS.ProblemType.GHIEP)
            eps.setOperators(self.A_scaled,self.Alocal)
            eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
            eps.setTarget(0.)
            ST = eps.getST()

            ST.setType("sinvert") 
            ST.setKSP(tempksp)

            if len(rbm_vecs) > 0 :
                eps.setDeflationSpace(rbm_vecs)
            eps.solve()
            if eps.getConverged() < self.nev:
                print(f"WARNING: Only {eps.getConverged()} eigenvalues converged for GenEO_eigmin in subdomain {mpi.COMM_WORLD.rank} whereas {self.nev} were requested")
            for i in range(min(eps.getConverged(),self.maxev)):
                if(abs(eps.getEigenvalue(i))<tauGenEO_eigmin): #TODO tell slepc that the eigenvalues are real
                   rbm_vecs.append(self.workl.duplicate())
                   eps.getEigenvector(i,rbm_vecs[-1])
                #print(mpi.COMM_WORLD.rank, eps.getEigenvalue(i))        
            self.eps_eigmax=eps #the only reason for this line is to make sure self.ksp and hence PCBNN.ksp is not destroyed  

    def assemble_coarse_operators(self,rbm_vecs):
        #coarse_vecs is a list of the local contributions to the coarse space: coarse_vecs[i] is either a local vector or None if coarse vector number i belongs to another subdomain
        #the vectors in coarse_vecs have been scaled so that their A-norm is 1
        #coarse_Avecs is a list of the global vectors: A*coarse_vecs
        #Delta is the matrix of the coarse problem. As a result of the scaling of the coarse vectors, its diagonal is 1
        print(f"Subdomain number {mpi.COMM_WORLD.rank} contributes {len(rbm_vecs)} coarse vectors in total")

        coarse_vecs = []
        for i in range(mpi.COMM_WORLD.size):
            nrbl = len(rbm_vecs) if i == mpi.COMM_WORLD.rank else None
            nrbl = mpi.COMM_WORLD.bcast(nrbl, root=i)
            for irbm in range(nrbl):
                coarse_vecs.append(rbm_vecs[irbm] if i == mpi.COMM_WORLD.rank else None)
        coarse_Avecs = []
        work, _ = self.A.getVecs()
        for vec in coarse_vecs:
            if vec:
                self.workl = vec.copy()
            else:
                self.workl.set(0.)
            work.set(0)
            self.scatter_l2g(self.workl, work, PETSc.InsertMode.ADD_VALUES)
            coarse_Avecs.append(self.A*work)
            self.scatter_l2g(coarse_Avecs[-1], self.workl, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
            if vec:
                vec.scale(1./np.sqrt(vec.dot(self.workl)))
                self.workl = vec.copy()
            else:
                self.workl.set(0)
            work.set(0)
            self.scatter_l2g(self.workl, work, PETSc.InsertMode.ADD_VALUES)
            coarse_Avecs[-1] = self.A*work

        #Define, fill and factorize coarse problem matrix
        Delta = PETSc.Mat().create(comm=PETSc.COMM_SELF)
        Delta.setType(PETSc.Mat.Type.SEQDENSE)
        Delta.setSizes([len(coarse_vecs),len(coarse_vecs)])
        Delta.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        Delta.setPreallocationDense(None)
        for i, vec in enumerate(coarse_vecs):
            if vec:
                self.workl = vec.copy()
            else:
                self.workl.set(0)

            work.set(0)
            self.scatter_l2g(self.workl, work, PETSc.InsertMode.ADD_VALUES)
            for j in range(i+1):
                tmp = coarse_Avecs[j].dot(work)
                Delta[i, j] = tmp
                Delta[j, i] = tmp
        Delta.assemble()
        ksp_Delta = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp_Delta.setOperators(Delta)
        ksp_Delta.setType('preonly')
        pc = ksp_Delta.getPC()
        pc.setType('cholesky')
        return coarse_vecs, coarse_Avecs, Delta, ksp_Delta 


    def project(self, x):
        alpha = self.gamma.duplicate()
        for i, Avec in enumerate(self.coarse_Avecs):
            self.gamma[i] = Avec.dot(x)

        self.ksp_Delta.solve(self.gamma, alpha)

        self.workl.set(0)
        for i, vec in enumerate(self.coarse_vecs):
            if vec:
                self.workl.axpy(-alpha[i], vec)

        self.scatter_l2g(self.workl, x, PETSc.InsertMode.ADD_VALUES)

    def xcoarse(self, rhs): #a better name would be coarsesolve
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

    def project_transpose(self, x):
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
