from petsc4py import PETSc
import mpi4py.MPI as mpi
import numpy as np
from .bc import bcApplyWest_vec
from slepc4py import SLEPc

class projection(object):
    def __init__(self,PCBNN): 
        """
        Initialize the coarse space and corresponding projection preconditioner and other coarse operators. 
        The default coarse space is the kernel of the local operators if these have been factorized 
        with MUMPS and no coarse space otherwise. 

        Parameters
        ==========

        PETSc.Options
        =============

        PCBNN_GenEO : Bool
            Default is True. 
            If True then the coarse space is enriched by solving local generalized eigenvalue problems. 
 
        PCBNN_GenEO_eigmin : Real
            Default is 0.1.
            Target for the smallest eigenvalue (eigmin) of the preconditioned operator. This sets the threshold for selecting which eigenvectors from the local generalized eigenvalue problems are selected for the coarse space. There are three cases where we do NOT solve an eigenvalue problem for eigmin:
                - if GenEO = False.
                - if PCBNN_GenEO_eigmin = 0.
                - if PCBNN_switchtoASM = False (this is the case by default) then the preconditioner is BNN and it is already known that eigmin >= 1. In this case the value of PCBNN_GenEO_eigmin is ignored. 

        PCBNN_GenEO_eigmax : Real
            Default is 10.
            Target for the largest eigenvalue (eigmax) of the preconditioned operator. This sets the threshold for selecting which eigenvectors from the local generalized eigenvalue problems are selected for the coarse space. There are three cases where we do NOT solve an eigenvalue problem for eigmin:
                - if GenEO = False.
                - if PCBNN_GenEO_eigmax = 0.
                - if PCBNN_switchtoASM = True then the preconditioner is Additive Schwarz and it is already known that eigmax <= maximal multiplicity of a degree of freedom. In this case the value of PCBNN_GenEO_eigmax is ignored. 

        PCBNN_GenEO_nev : Int
            Default is 10 .
            Number of eigenpairs requested during the eigensolves. This is an option passed to the eigensolver.

        PCBNN_GenEO_maxev : Int
            Default is 2*PCBNN_GenEO_nev.
            Maximal number of eigenvectors from each eigenvalue problem that can be selected for the coarse space. This is relevant because if more eigenvalues than requested by PBNN_GenEO_nev have converged during the eigensolve then they are all returned so setting the value of PCBNN_GenEO_nev does not impose a limitation on the size of the coarse space.  

        """
        OptDB = PETSc.Options()                                
        self.GenEO = OptDB.getBool('PCBNN_GenEO', True)
        self.eigmax = OptDB.getReal('PCBNN_GenEO_eigmax', 10)
        self.eigmin = OptDB.getReal('PCBNN_GenEO_eigmin', 0.1)
        self.nev = OptDB.getInt('PCBNN_GenEO_nev', 10) 
        self.maxev = OptDB.getInt('PCBNN_GenEO_maxev', 2*self.nev) 

        vglobal, _ = PCBNN.A.getVecs()
        vlocal, _ = PCBNN.A_scaled.getVecs()

        self.scatter_l2g = PCBNN.scatter_l2g 
        self.A = PCBNN.A
        self.A_scaled = PCBNN.A_scaled
        self.A_mpiaij_local = PCBNN.A_mpiaij_local
        self.verbose = PCBNN.verbose

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
            if self.verbose:
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
                if self.verbose and mpi.COMM_WORLD.rank == 0:
                    print(f'This is classical additive Schwarz so eigmax = {PCBNN.mult_max}, no eigenvalue problem will be solved for eigmax')
            if self.Alocal != self.A_scaled:
                self.solve_GenEO_eigmin(rbm_vecs, tau_eigmin)
            else:
                if self.verbose and mpi.COMM_WORLD.rank == 0:
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
        """
        Solves the local GenEO eigenvalue problem related to the largest eigenvalue eigmax. 

        Parameters
        ==========

        rbm_vecs : list of local PETSc .vecs
            rbm_vecs may already contain some local coarse vectors. This routine will possibly add more vectors to the list.

        tauGenEO_eigmax: Real.
            Threshold for selecting eigenvectors for the coarse space.

        """
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
                    if self.verbose:
                        print(f'GenEO eigenvalue number {i} for lambdamax in subdomain {mpi.COMM_WORLD.rank}: {eps.getEigenvalue(i)}')
                else:
                    if self.verbose:
                        print(f'GenEO eigenvalue number {i} for lambdamax in subdomain {mpi.COMM_WORLD.rank}: {eps.getEigenvalue(i)} <-- not selected (> {tauGenEO_eigmax})')

            self.eps_eigmax=eps #TODO FIX the only reason for this line is to make sure self.ksp and hence PCBNN.ksp is not destroyed  
        if self.verbose:
            print(f"Subdomain number {mpi.COMM_WORLD.rank} contributes {len(rbm_vecs)} coarse vectors after first GenEO")
    
    def solve_GenEO_eigmin(self, rbm_vecs, tauGenEO_eigmin):
        """
        Solves the local GenEO eigenvalue problem related to the smallest eigenvalue eigmin. 

        Parameters
        ==========

        rbm_vecs : list of local PETSc .vecs
            rbm_vecs may already contain some local coarse vectors. This routine will possibly add more vectors to the list.

        tauGenEO_eigmin: Real.
            Threshold for selecting eigenvectors for the coarse space.

        """
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
            if self.verbose:
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
                   if self.verbose:
                       print(f'GenEO eigenvalue number {i} for lambdamin in subdomain {mpi.COMM_WORLD.rank}: {eps.getEigenvalue(i)}')
                else:
                    if self.verbose:
                        print(f'GenEO eigenvalue number {i} for lambdamin in subdomain {mpi.COMM_WORLD.rank}: {eps.getEigenvalue(i)} <-- not selected (> {tauGenEO_eigmin})')
                #print(mpi.COMM_WORLD.rank, eps.getEigenvalue(i))        
            self.eps_eigmin=eps #the only reason for this line is to make sure self.ksp and hence PCBNN.ksp is not destroyed  

    def assemble_coarse_operators(self,rbm_vecs):
        """
        Assembles the coarse operators from a list of local contributions to the coarse space.  

        Parameters
        ==========

        rbm_vecs : list of local PETSc .vecs
           list of the coarse vectors contributed by the subdomain. 

        Returns
        ==========

        coarse_vecs : list of local vectors or None ? FIX 
            list of the local contributions to the coarse space numbered globally: coarse_vecs[i] is either a scaled local vector from rbm_vecs or None if coarse vector number i belongs to another subdomain. The scaling that is applied to the coarse vectors ensures that their A-norm is 1.

        coarse_Avecs : list of global PETSc.Vecs
            list of the A*coarse_vecs[i]. These are global vectors so not in the same format as the vectors in coarse_vecs. 

        Delta : PETSc.Mat (local)
            matrix of the coarse problem. As a result of the scaling of the coarse vectors, its diagonal is 1. This matrix is duplicated over all subdomains This matrix is duplicated over all subdomains

        ksp_Delta : PETSc.ksp 
            Krylov subspace solver for the coarse problem matrix Delta.

        """
        if self.verbose:
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
 
        if mpi.COMM_WORLD.rank == 0:
            print(f"There are {len(coarse_vecs)} vectors in the coarse space.")

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
        """
        Applies the coarse projection (or projection preconditioner) to x 

        Parameters
        ==========

        x : PETSc.Vec
           Vector to which the projection is applied and in which the result is stored. 

        """
        alpha = self.gamma.duplicate()
        for i, Avec in enumerate(self.coarse_Avecs):
            self.gamma[i] = Avec.dot(x)

        self.ksp_Delta.solve(self.gamma, alpha)

        self.workl.set(0)
        for i, vec in enumerate(self.coarse_vecs):
            if vec:
                self.workl.axpy(-alpha[i], vec)

        self.scatter_l2g(self.workl, x, PETSc.InsertMode.ADD_VALUES)

    def coarse_init(self, rhs): 
        """
        Initialize the projected PCG algorithm or MPCG algorithm with the solution of the problem in the coarse space.  

        Parameters
        ==========

        rhs : PETSc.Vec
           Right hand side vector for which to initialize the problem. 

        Returns
        =======
        
        out : PETSc.Vec
           Solution of the problem projected into the coarse space for the initialization of a projected Krylov subspace solver. 

        """

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
        """
        Applies the transpose of the coarse projection (which is also a projection) to x 

        Parameters
        ==========
    
        x : PETSc.Vec
           Vector to which the projection is applied and in which the result is stored. 

        """
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
