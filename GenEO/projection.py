# Authors:
#     Loic Gouarin <loic.gouarin@cmap.polytechnique.fr>
#     Nicole Spillane <nicole.spillane@cmap.polytechnique.fr>
#
# License: BSD 3 clause
from petsc4py import PETSc
import mpi4py.MPI as mpi
import numpy as np
from .bc import bcApplyWest_vec
from slepc4py import SLEPc

class minimal_V0(object):
    def __init__(self,ksp_Atildes,V0s=[],labs=[]):
        """
        Compute local contributions to the minimal coarse space, i.e., the kernel of Atildes. Only implemented if the solver for Atildes is mumps.
        Parameters
        ==========

        PETSc.Options
        =============
        PCBNN_mumpsCntl3 : Real
        Default is 1e-6
        This is a parameter passed to mumps: CNTL(3) is used to determine if a pivot is null when the null pivot detection option is used (which is the case when ICNTL(24) = 1).
        If PCBNN_mumpsCntl3 is too small, part of the kernel of the local matrices may be missing which will deteriorate convergence significantly or even prevent the algorithm from converging. If it is too large, then more vectors than strictly needed may be incorporated into the coarse space. This leads to a larger coarse problem but will accelrate convergence.

        PCBNN_verbose : Bool
            Default is False.
            If True, some information about the preconditioners is printed when the code is executed.

        """
        OptDB = PETSc.Options()
        self.mumpsCntl3 = OptDB.getReal('PCBNN_mumpsCntl3', 1e-6)
        self.verbose =  OptDB.getBool('PCBNN_GenEO_verbose', False)

        self.comm = mpi.COMM_SELF
        #compute the kernel of the self.ksp_Atildes operator and initialize local coarse space with it
        self.ksp_Atildes = ksp_Atildes
        _,Atildes = self.ksp_Atildes.getOperators()
        self.Atildes = Atildes
        works, _ = self.Atildes.getVecs()
        self.works = works
        if self.ksp_Atildes.pc.getFactorSolverType() == 'mumps':
            self.ksp_Atildes.pc.setFactorSetUpSolverType()
            F = self.ksp_Atildes.pc.getFactorMatrix()
            F.setMumpsIcntl(7, 2)
            F.setMumpsIcntl(24, 1)
            F.setMumpsCntl(3, self.mumpsCntl3)
            self.ksp_Atildes.pc.setUp()
            nrb = F.getMumpsInfog(28)

            for i in range(nrb):
                F.setMumpsIcntl(25, i+1)
                works.set(0.)
                self.ksp_Atildes.solve(works, works)
                V0s.append(works.copy())
                labs.append('Ker Atildes')

            F.setMumpsIcntl(25, 0)
            if self.verbose:
                PETSc.Sys.Print('Subdomain number {} contributes {} coarse vectors as zero energy modes of local solver'.format(mpi.COMM_WORLD.rank, nrb), comm=self.comm)
        else:
            nrb = 0
        self.V0s = V0s
        self.labs = labs
        self.nrb = nrb
    def view(self):
        print('###')
        print(f'view of minimal_V0 in Subdomain {mpi.COMM_WORLD.rank}')
        if mpi.COMM_WORLD.rank == 0:
            print(f'{self.mumpsCntl3=}') 
            print(f'{self.labs=}')
        if (self.ksp_Atildes.pc.getFactorSolverType() == 'mumps'):
            print(f'dim(Ker(Atildes)) = {self.nrb=}')
        else:
            print(f'{self.nrb=}, no kernel computation because pc is not mumps')


class GenEO_V0(object):
    def __init__(self,ksp_Atildes,Ms,As,mult_max,V0s=[],labs=[],ksp_Ms=[]):
        """
        Initialize the coarse space and corresponding projection preconditioner and other coarse operators.
        The default coarse space is the kernel of the local operators if these have been factorized
        with MUMPS and no coarse space otherwise.

        Parameters
        ==========

        PETSc.Options
        =============

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

        PCBNN_mumpsCntl3 : Real
        Default is 1e-6
        This is a parameter passed to mumps: CNTL(3) is used to determine if a pivot is null when the null pivot detection option is used (which is the case when ICNTL(24) = 1).
        If PCBNN_mumpsCntl3 is too small, part of the kernel of the local matrices may be missing which will deteriorate convergence significantly or even prevent the algorithm from converging. If it is too large, then more vectors than strictly needed may be incorporated into the coarse space. This leads to a larger coarse problem but will accelrate convergence.

        PCBNN_verbose : Bool
            Default is False.
            If True, some information about the preconditioners is printed when the code is executed.
        """
        OptDB = PETSc.Options()
        self.tau_eigmax = OptDB.getReal('PCBNN_GenEO_taueigmax', 0.1)
        self.tau_eigmin = OptDB.getReal('PCBNN_GenEO_taueigmin', 0.1)
        self.eigmax = OptDB.getReal('PCBNN_GenEO_eigmax', -1)
        self.eigmin = OptDB.getReal('PCBNN_GenEO_eigmin', -1)
        self.nev = OptDB.getInt('PCBNN_GenEO_nev', 10)
        self.maxev = OptDB.getInt('PCBNN_GenEO_maxev', 2*self.nev)
        self.mumpsCntl3 = OptDB.getReal('PCBNN_mumpsCntl3', 1e-6)
        self.verbose =  OptDB.getBool('PCBNN_GenEO_verbose', False)

        self.comm = mpi.COMM_SELF

        self.As = As
        self.Ms = Ms
        self.mult_max = mult_max

        self.ksp_Atildes = ksp_Atildes
        _,self.Atildes = self.ksp_Atildes.getOperators()

        if ksp_Ms == []:
            self.ksp_Ms = []
        else:
            self.ksp_Ms = ksp_Ms
            _,self.Ms = self.ksp_Ms.getOperators()

        works, _ = self.Ms.getVecs()
        self.works = works

        #thresholds for the eigenvalue problems are computed from the user chosen targets for eigmin and eigmax
        if self.eigmin > 0:
          self.tau_eigmin = self.eigmin

        if self.eigmax > 0:
            self.tau_eigmax = self.mult_max/self.eigmax

        if self.tau_eigmax > 0:
            if self.Atildes != self.As:
                self.solve_GenEO_eigmax(V0s, labs, self.tau_eigmax)
            else:
                self.Lambda_GenEO_eigmax = []
                self.n_GenEO_eigmax = 0
                if self.verbose:
                    PETSc.Sys.Print('This is classical additive Schwarz so eigmax = {} (+1 if fully additive preconditioner), no eigenvalue problem will be solved for eigmax'.format(self.mult_max), comm=mpi.COMM_WORLD)
        else:
            self.Lambda_GenEO_eigmax = []
            self.n_GenEO_eigmax = 0
            if self.verbose:
                PETSc.Sys.Print('Skip GenEO for eigmax as user specified non positive eigmax and taueigmax', comm=mpi.COMM_WORLD)
        if self.tau_eigmin > 0:
            if self.Atildes != self.Ms:
                self.solve_GenEO_eigmin(V0s, labs, self.tau_eigmin)
            else:
                self.Lambda_GenEO_eigmin = []
                self.n_GenEO_eigmin = 0
                self.dimKerMs = [] 
                if self.verbose:
                    PETSc.Sys.Print('This is BNN so eigmin = 1, no eigenvalue problem will be solved for eigmin', comm=mpi.COMM_WORLD)
        else:
            self.Lambda_GenEO_eigmin = []
            self.n_GenEO_eigmin = 0
            self.dimKerMs = [] 
            if self.verbose:
                PETSc.Sys.Print('Skip GenEO for eigmin as user specified non positive eigmin and taueigmin', comm=mpi.COMM_WORLD)


        self.V0s = V0s
        self.labs = labs

    def solve_GenEO_eigmax(self, V0s, labs, tauGenEO_eigmax):
        """
        Solves the local GenEO eigenvalue problem related to the largest eigenvalue eigmax.

        Parameters
        ==========

        V0s : list of local PETSc .vecs
            V0s may already contain some local coarse vectors. This routine will possibly add more vectors to the list.

        tauGenEO_eigmax: Real.
            Threshold for selecting eigenvectors for the coarse space.

        """
        if tauGenEO_eigmax > 0:
            eps = SLEPc.EPS().create(comm=PETSc.COMM_SELF)
            eps.setDimensions(nev=self.nev)

            eps.setProblemType(SLEPc.EPS.ProblemType.GHIEP)
            eps.setOperators(self.Atildes , self.As )
            eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
            eps.setTarget(0.)
            if len(V0s) > 0 :
                eps.setDeflationSpace(V0s)
            ST = eps.getST()
            ST.setType("sinvert")
            ST.setKSP(self.ksp_Atildes)
            eps.solve()
            if eps.getConverged() < self.nev:
                PETSc.Sys.Print('WARNING: Only {} eigenvalues converged for GenEO_eigmax in subdomain {} whereas {} were requested'.format(eps.getConverged(), mpi.COMM_WORLD.rank, self.nev), comm=self.comm)
            if abs(eps.getEigenvalue(eps.getConverged() -1)) < tauGenEO_eigmax:
                PETSc.Sys.Print('WARNING: The largest eigenvalue computed for GenEO_eigmax in subdomain {} is {} < the threshold which is {}. Consider setting PCBNN_GenEO_nev to something larger than {}'.format(mpi.COMM_WORLD.rank, eps.getEigenvalue(eps.getConverged() - 1), tauGenEO_eigmax, eps.getConverged()), comm=self.comm)

            self.Lambda_GenEO_eigmax = []
            self.n_GenEO_eigmax = 0 
            for i in range(min(eps.getConverged(),self.maxev)):
                tmp = eps.getEigenvalue(i)
                if(abs(tmp)<tauGenEO_eigmax): #TODO tell slepc that the eigenvalues are real
                    V0s.append(self.works.duplicate())
                    labs.append(f'\lambda_{i}^\sharp = {tmp}')
                    self.Lambda_GenEO_eigmax.append(tmp) #only for viewing 
                    eps.getEigenvector(i,V0s[-1])
                    if self.verbose:
                        PETSc.Sys.Print('GenEO eigenvalue number {} for lambdamax in subdomain {}: {}'.format(i, mpi.COMM_WORLD.rank, tmp) , comm=self.comm)

                    self.n_GenEO_eigmax += 1 
                else:
                    self.Lambda_GenEO_eigmax.append(tmp)  #only for viewing 
                    if self.verbose:
                        PETSc.Sys.Print('GenEO eigenvalue number {} for lambdamax in subdomain {}: {} <-- not selected (> {})'.format(i, mpi.COMM_WORLD.rank, tmp, tauGenEO_eigmax), comm=self.comm)

            self.eps_eigmax=eps #TODO FIX the only reason for this line is to make sure self.ksp_Atildes and hence PCBNN.ksp is not destroyed
        if self.verbose:
            PETSc.Sys.Print('Subdomain number {} contributes {} coarse vectors after first GenEO'.format(mpi.COMM_WORLD.rank, len(V0s)), comm=self.comm)

    def solve_GenEO_eigmin(self, V0s, labs, tauGenEO_eigmin):
        """
        Solves the local GenEO eigenvalue problem related to the smallest eigenvalue eigmin.

        Parameters
        ==========

        V0s : list of local PETSc .vecs
            V0s may already contain some local coarse vectors. This routine will possibly add more vectors to the list.

        tauGenEO_eigmin: Real.
            Threshold for selecting eigenvectors for the coarse space.

        """
        if tauGenEO_eigmin > 0:
            #to compute the smallest eigenvalues of the preconditioned matrix, Ms must be factorized
            if self.ksp_Ms == []: 
                tempksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
                tempksp.setOperators(self.Ms)
                tempksp.setType('preonly')
                temppc = tempksp.getPC()
                temppc.setType('cholesky')
                temppc.setFactorSolverType('mumps')
                temppc.setFactorSetUpSolverType()
                tempF = temppc.getFactorMatrix()
                tempF.setMumpsIcntl(7, 2)
                tempF.setMumpsIcntl(24, 1)
                tempF.setMumpsCntl(3, self.mumpsCntl3)
                temppc.setUp()
                self.dimKerMs = tempF.getMumpsInfog(28)

                for i in range(self.dimKerMs):
                    tempF.setMumpsIcntl(25, i+1)
                    self.works.set(0.)
                    tempksp.solve(self.works, self.works)
                    V0s.append(self.works.copy())
                    labs.append(f'Ker(Ms)')


                tempF.setMumpsIcntl(25, 0)
                if self.verbose:
                    PETSc.Sys.Print('Subdomain number {} contributes {} coarse vectors as zero energy modes of Ms (in GenEO for eigmin)'.format(mpi.COMM_WORLD.rank, self.dimKerMs), comm=self.comm)
            else:
                tempksp = self.ksp_Ms
                self.dimKerMs = []
            #Eigenvalue Problem for smallest eigenvalues
            eps = SLEPc.EPS().create(comm=PETSc.COMM_SELF)
            eps.setDimensions(nev=self.nev)

            eps.setProblemType(SLEPc.EPS.ProblemType.GHIEP)
            eps.setOperators(self.Ms,self.Atildes)
            eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
            eps.setTarget(0.)
            ST = eps.getST()

            ST.setType("sinvert")
            ST.setKSP(tempksp)

            if len(V0s) > 0 :
                eps.setDeflationSpace(V0s)
            eps.solve()
            if eps.getConverged() < self.nev:
                PETSc.Sys.Print('WARNING: Only {} eigenvalues converged for GenEO_eigmin in subdomain {} whereas {} were requested'.format(eps.getConverged(), mpi.COMM_WORLD.rank, self.nev), comm=self.comm)
            self.n_GenEO_eigmin = 0 
            self.Lambda_GenEO_eigmin = []
            for i in range(min(eps.getConverged(),self.maxev)):
                tmp = eps.getEigenvalue(i)
                if(abs(tmp)<tauGenEO_eigmin): #TODO tell slepc that the eigenvalues are real
                    V0s.append(self.works.duplicate())
                    labs.append(f'\lambda_{i}^ \ flat = {tmp}')
                    self.Lambda_GenEO_eigmin.append(tmp)
                    eps.getEigenvector(i,V0s[-1])
                    self.n_GenEO_eigmin += 1
                    if self.verbose:
                        PETSc.Sys.Print('GenEO eigenvalue number {} for lambdamin in subdomain {}: {}'.format(i, mpi.COMM_WORLD.rank, tmp), comm=self.comm)
                else:
                    self.Lambda_GenEO_eigmin.append(tmp)
                    if self.verbose:
                        PETSc.Sys.Print('GenEO eigenvalue number {} for lambdamin in subdomain {}: {} <-- not selected (> {})'.format(i, mpi.COMM_WORLD.rank, eps.getEigenvalue(i), tauGenEO_eigmin), comm=self.comm)
            self.eps_eigmin=eps #the only reason for this line is to make sure self.ksp_Atildes and hence PCBNN.ksp is not destroyed
    def view(self):
        print('###')
        print(f'view of GenEO in Subdomain {mpi.COMM_WORLD.rank}')
        if mpi.COMM_WORLD.rank == 0:
            print(f'{self.tau_eigmax=}') 
            print(f'{self.tau_eigmin=}') 
            print(f'{self.eigmax=}') 
            print(f'{self.eigmin=}') 
            print(f'{self.nev=}') 
            print(f'{self.maxev=}') 
            print(f'{self.mumpsCntl3=}') 
            print(f'{self.verbose=}') 
            print(f'{self.mult_max=}') 
            print(f'Additive Schwarz ? {(self.Atildes == self.As)}')
            print(f'Neumann Neumann ? {(self.Atildes == self.Ms)}')
        print(f'{self.Lambda_GenEO_eigmax=}')    
        print(f'{self.n_GenEO_eigmax=}')
        print(f'{self.dimKerMs=}') 
        print(f'{self.Lambda_GenEO_eigmin=}')
        print(f'{self.n_GenEO_eigmin=}')
        print(f'{self.labs=}')

class coarse_operators(object):
    def __init__(self,V0s,A,scatter_l2g,works,work,V0_is_global=False):
        """
        V0_is_global is a Boolean that defines whether the coarse space has already been assembled over subdomains. If false, the vectors in V0s are local, will be extended by zero to the whole subdomain to obtain a global coarse space of dimension (sum(len(V0s)). If True then the vectors in V0s are already the local components of some global coarse vectors. The dimension of the global coarse space is len(V0s).
        """
        OptDB = PETSc.Options()
        self.verbose =  OptDB.getBool('PCBNN_GenEO_verbose', False)

        self.comm = mpi.COMM_SELF

        self.scatter_l2g = scatter_l2g
        self.A = A
        self.V0_is_global = V0_is_global
        self.works = works
        self.work = work

        V0, AV0, Delta, ksp_Delta = self.assemble_coarse_operators(V0s)
        #if mpi.COMM_WORLD.rank == 0:
        #    Delta.view()

        self.V0 = V0
        self.AV0 = AV0
        self.Delta = Delta
        self.ksp_Delta = ksp_Delta

        self.gamma = PETSc.Vec().create(comm=PETSc.COMM_SELF)
        self.gamma.setType(PETSc.Vec.Type.SEQ)
        self.gamma.setSizes(len(self.V0))
        self.gamma_tmp = self.gamma.duplicate()

    def assemble_coarse_operators(self,V0s):
        """
        Assembles the coarse operators from a list of local contributions to the coarse space.

        Parameters
        ==========

        V0s : list of local PETSc .vecs
           list of the coarse vectors contributed by the subdomain.

        Returns
        ==========

        V0 : list of local vectors or None ? FIX
            list of the local contributions to the coarse space numbered globally: V0[i] is either a scaled local vector from V0s or None if coarse vector number i belongs to another subdomain. The scaling that is applied to the coarse vectors ensures that their A-norm is 1.

        AV0 : list of global PETSc.Vecs
            list of the A*V0[i]. These are global vectors so not in the same format as the vectors in V0.

        Delta : PETSc.Mat (local)
            matrix of the coarse problem. As a result of the scaling of the coarse vectors, its diagonal is 1. This matrix is duplicated over all subdomains This matrix is duplicated over all subdomains

        ksp_Delta : PETSc.ksp
            Krylov subspace solver for the coarse problem matrix Delta.

        """
        if self.verbose:
            if self.V0_is_global == False:
                PETSc.Sys.Print('Subdomain number {} contributes {} coarse vectors in total'.format(mpi.COMM_WORLD.rank, len(V0s)), comm=self.comm)

        if mpi.COMM_WORLD.rank == 0:
            self.gathered_dimV0s = [] #only for view save dims of V0s from each s
        self.work2 = self.work.duplicate()
        if(self.V0_is_global == False):
            V0 = []
            for i in range(mpi.COMM_WORLD.size):
                nrbl = len(V0s) if i == mpi.COMM_WORLD.rank else None
                nrbl = mpi.COMM_WORLD.bcast(nrbl, root=i)
                if mpi.COMM_WORLD.rank == 0:
                    self.gathered_dimV0s.append(nrbl) #only for view save dims of V0s from each s
                for irbm in range(nrbl):
                    V0.append(V0s[irbm].copy() if i == mpi.COMM_WORLD.rank else None)
        else:
            V0 = V0s.copy()

        AV0 = []
        for vec in V0:
            if(self.V0_is_global == False):
                if vec:
                    self.works = vec.copy()
                else:
                    self.works.set(0.)
                self.work.set(0)
                self.scatter_l2g(self.works, self.work, PETSc.InsertMode.ADD_VALUES)
            else:
                self.work = vec.copy()  
            #debug1 = np.sqrt(self.work.dot(self.work))
            #debug4 = self.work.norm()
            #debug3 = np.sqrt(self.works.dot(self.works))
            #debug5 = self.works.norm()
            #print(f'normworks {debug3} = {debug5} normwork {debug1} = {debug4}')
            self.A.mult(self.work,self.work2)
            AV0.append(self.work2.copy())
            tmp = np.sqrt(self.work.dot(self.work2))
            if(self.V0_is_global == False):
                self.scatter_l2g(AV0[-1], self.works, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
                if vec:
                    vec.scale(1./tmp)
                    self.works = vec.copy()
                else:
                    self.works.set(0)
                self.work.set(0)
                self.scatter_l2g(self.works, self.work, PETSc.InsertMode.ADD_VALUES)
                #self.scatter_l2g(self.works, self.work, PETSc.InsertMode.ADD_VALUES)
                #self.A.mult(self.work,self.work2)
                self.A.mult(self.work,AV0[-1])
            else:
                vec.scale(1./tmp)
                self.A.mult(vec,AV0[-1])
            #self.A.mult(self.work,AV0[-1])
            # AV0[-1] = self.work2.copy()
            # AV0[-1] = xtmp.copy()
            #debug6 = np.sqrt(self.work2.dot(self.work2))
            #debug7 = self.work2.norm()
            #debug2 = np.sqrt(AV0[-1].dot(AV0[-1]))
            #debug5 = AV0[-1].norm()
            # debug6 = np.sqrt(self.work2.dot(self.work2))
            # debug7 = self.work2.norm()
            # debug2 = np.sqrt(AV0[-1].dot(AV0[-1]))
            # debug5 = AV0[-1].norm()
#            if mpi.COMM_WORLD.rank == 0:
#                print(f'norm Acoarsevec {debug2} = {debug5}')
            # print(f'norm Acoarsevec {debug2} = {debug5} = {debug6} = {debug7}')


        self.dim = len(V0)
        PETSc.Sys.Print('There are {} vectors in the coarse space.'.format(self.dim), comm=mpi.COMM_WORLD)

        #Define, fill and factorize coarse problem matrix
        Delta = PETSc.Mat().create(comm=PETSc.COMM_SELF)
        Delta.setType(PETSc.Mat.Type.SEQDENSE)
        Delta.setSizes([len(V0),len(V0)])
        Delta.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        Delta.setPreallocationDense(None)
        for i, vec in enumerate(V0):
            if(self.V0_is_global == False):
                if vec:
                    self.works = vec.copy()
                else:
                    self.works.set(0)

                self.work.set(0)
                self.scatter_l2g(self.works, self.work, PETSc.InsertMode.ADD_VALUES)
            else:
                self.work = vec.copy()
            for j in range(i+1):
                tmp = AV0[j].dot(self.work)
                Delta[i, j] = tmp
                Delta[j, i] = tmp
        Delta.assemble()
        ksp_Delta = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp_Delta.setOperators(Delta)
        ksp_Delta.setType('preonly')
        pc = ksp_Delta.getPC()
        pc.setType('cholesky')
        return V0, AV0, Delta, ksp_Delta


    def project(self, x):
        """
        Applies the coarse projection (or projection preconditioner) to x

        Parameters
        ==========

        x : PETSc.Vec
           Vector to which the projection is applied and in which the result is stored.

        """
        alpha = self.gamma.duplicate()
        for i, Avec in enumerate(self.AV0):
            self.gamma[i] = Avec.dot(x)

        self.ksp_Delta.solve(self.gamma, alpha)

        if(self.V0_is_global == False):
            self.works.set(0)
            for i, vec in enumerate(self.V0):
                if vec:
                    self.works.axpy(-alpha[i], vec)

            self.scatter_l2g(self.works, x, PETSc.InsertMode.ADD_VALUES)
        else:
            for i, vec in enumerate(self.V0):
                x.axpy(-alpha[i], vec)

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

        if(self.V0_is_global == False):
            self.scatter_l2g(rhs, self.works, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
            self.gamma.set(0)
            self.gamma_tmp.set(0)
            for i, vec in enumerate(self.V0):
                if vec:
                    self.gamma_tmp[i] = vec.dot(self.works)

            mpi.COMM_WORLD.Allreduce([self.gamma_tmp, mpi.DOUBLE], [self.gamma, mpi.DOUBLE], mpi.SUM)
        else:
            for i, vec in enumerate(self.V0):
                self.gamma[i] = vec.dot(rhs)

        self.ksp_Delta.solve(self.gamma, alpha)

        if(self.V0_is_global == False):
            self.works.set(0)
            for i, vec in enumerate(self.V0):
                if vec:
                    self.works.axpy(alpha[i], vec)

            out = rhs.duplicate()
            out.set(0)
            self.scatter_l2g(self.works, out, PETSc.InsertMode.ADD_VALUES)
        else:
            out = rhs.duplicate()
            out.set(0)
            for i, vec in enumerate(self.V0):
                    out.axpy(alpha[i], vec)

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
        if(self.V0_is_global == False):
            self.scatter_l2g(x, self.works, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
            self.gamma.set(0)
            self.gamma_tmp.set(0)
            for i, vec in enumerate(self.V0):
                if vec:
                    self.gamma_tmp[i] = vec.dot(self.works)

            mpi.COMM_WORLD.Allreduce([self.gamma_tmp, mpi.DOUBLE], [self.gamma, mpi.DOUBLE], mpi.SUM)
        else:
            for i, vec in enumerate(self.V0):
                self.gamma[i] = vec.dot(x)
            
        self.ksp_Delta.solve(self.gamma, alpha)

        for i in range(len(self.V0)):
            x.axpy(-alpha[i], self.AV0[i])
    def view(self):
        if mpi.COMM_WORLD.rank == 0:
            print('###')
            print(f'view of coarse_operators')# in Subdomain {mpi.COMM_WORLD.rank}')
            print(f'{self.V0_is_global=}') 
            print(f'{self.gathered_dimV0s=}') 


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
            Default is False.
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

        PCBNN_mumpsCntl3 : Real
            Default is 1e-6
            This is a parameter passed to mumps: CNTL(3) is used to determine if a pivot is null when the null pivot detection option is used (which is the case when ICNTL(24) = 1).
            If PCBNN_mumpsCntl3 is too small, part of the kernel of the local matrices may be missing which will deteriorate convergence significantly or even prevent the algorithm from converging. If it is too large, then more vectors than strictly needed may be incorporated into the coarse space. This leads to a larger coarse problem but will accelrate convergence.
        """
        OptDB = PETSc.Options()
        self.GenEO = OptDB.getBool('PCBNN_GenEO', True)
        self.eigmax = OptDB.getReal('PCBNN_GenEO_eigmax', 10)
        self.eigmin = OptDB.getReal('PCBNN_GenEO_eigmin', 0.1)
        self.nev = OptDB.getInt('PCBNN_GenEO_nev', 10)
        self.maxev = OptDB.getInt('PCBNN_GenEO_maxev', 2*self.nev)
        self.comm = mpi.COMM_SELF
        self.mumpsCntl3 = OptDB.getReal('PCBNN_mumpsCntl3', 1e-6)


        self.scatter_l2g = PCBNN.scatter_l2g
        self.A = PCBNN.A
        self.Ms = PCBNN.Ms
        self.As = PCBNN.As
        self.mult_max = PCBNN.mult_max
        self.verbose = PCBNN.verbose

        self.ksp_Atildes = PCBNN.ksp_Atildes
        _,Atildes = self.ksp_Atildes.getOperators()
        self.Atildes = Atildes

        works, _ = self.Ms.getVecs()
        self.works = works


#compute the kernel of the self.ksp_Atildes operator and initialize local coarse space with it
        if self.ksp_Atildes.pc.getFactorSolverType() == 'mumps':
            self.ksp_Atildes.pc.setFactorSetUpSolverType()
            F = self.ksp_Atildes.pc.getFactorMatrix()
            F.setMumpsIcntl(7, 2)
            F.setMumpsIcntl(24, 1)
            F.setMumpsCntl(3, self.mumpsCntl3)
            self.ksp_Atildes.pc.setUp()
            nrb = F.getMumpsInfog(28)

            V0s = []
            for i in range(nrb):
                F.setMumpsIcntl(25, i+1)
                works.set(0.)
                self.ksp_Atildes.solve(works, works)
                V0s.append(works.copy())

            F.setMumpsIcntl(25, 0)
            if self.verbose:
                PETSc.Sys.Print('Subdomain number {} contributes {} coarse vectors as zero energy modes of local solver'.format(mpi.COMM_WORLD.rank, nrb), comm=self.comm)
        else:
            V0s = []
            nrb = 0


        if self.GenEO == True:
            #thresholds for the eigenvalue problems are computed from the user chosen targets for eigmin and eigmax
            tau_eigmin = self.eigmin
            if self.eigmax > 0:
                tau_eigmax = self.mult_max/self.eigmax
            else:
                tau_eigmax = 0
            if self.Atildes != self.As:
                self.solve_GenEO_eigmax(V0s, tau_eigmax)
            else:
                if self.verbose:
                    PETSc.Sys.Print('This is classical additive Schwarz so eigmax = {} (+1 if fully additive preconditioner), no eigenvalue problem will be solved for eigmax'.format(self.mult_max), comm=mpi.COMM_WORLD)
            if self.Atildes != self.Ms:
                self.solve_GenEO_eigmin(V0s, tau_eigmin)
            else:
                if self.verbose:
                    PETSc.Sys.Print('This is BNN so eigmin = 1, no eigenvalue problem will be solved for eigmin', comm=mpi.COMM_WORLD)

        V0, AV0, Delta, ksp_Delta = self.assemble_coarse_operators(V0s)

        self.V0 = V0
        self.AV0 = AV0
        self.Delta = Delta
        self.ksp_Delta = ksp_Delta

        self.gamma = PETSc.Vec().create(comm=PETSc.COMM_SELF)
        self.gamma.setType(PETSc.Vec.Type.SEQ)
        self.gamma.setSizes(len(self.V0))
        self.gamma_tmp = self.gamma.duplicate()

    def solve_GenEO_eigmax(self, V0s, tauGenEO_eigmax):
        """
        Solves the local GenEO eigenvalue problem related to the largest eigenvalue eigmax.

        Parameters
        ==========

        V0s : list of local PETSc .vecs
            V0s may already contain some local coarse vectors. This routine will possibly add more vectors to the list.

        tauGenEO_eigmax: Real.
            Threshold for selecting eigenvectors for the coarse space.

        """
        if tauGenEO_eigmax > 0:
            eps = SLEPc.EPS().create(comm=PETSc.COMM_SELF)
            eps.setDimensions(nev=self.nev)

            eps.setProblemType(SLEPc.EPS.ProblemType.GHIEP)
            eps.setOperators(self.Atildes , self.As )
            eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
            eps.setTarget(0.)
            if len(V0s) > 0 :
                eps.setDeflationSpace(V0s)
            ST = eps.getST()
            ST.setType("sinvert")
            ST.setKSP(self.ksp_Atildes)
            eps.solve()
            if eps.getConverged() < self.nev:
                PETSc.Sys.Print('WARNING: Only {} eigenvalues converged for GenEO_eigmax in subdomain {} whereas {} were requested'.format(eps.getConverged(), mpi.COMM_WORLD.rank, self.nev), comm=self.comm)
            if abs(eps.getEigenvalue(eps.getConverged() -1)) < tauGenEO_eigmax:
                PETSc.Sys.Print('WARNING: The largest eigenvalue computed for GenEO_eigmax in subdomain {} is {} < the threshold which is {}. Consider setting PCBNN_GenEO_nev to something larger than {}'.format(mpi.COMM_WORLD.rank, eps.getEigenvalue(eps.getConverged() - 1), tauGenEO_eigmax, eps.getConverged()), comm=self.comm)

            for i in range(min(eps.getConverged(),self.maxev)):
                if(abs(eps.getEigenvalue(i))<tauGenEO_eigmax): #TODO tell slepc that the eigenvalues are real
                    V0s.append(self.works.duplicate())
                    eps.getEigenvector(i,V0s[-1])
                    if self.verbose:
                        PETSc.Sys.Print('GenEO eigenvalue number {} for lambdamax in subdomain {}: {}'.format(i, mpi.COMM_WORLD.rank, eps.getEigenvalue(i)) , comm=self.comm)
                else:
                    if self.verbose:
                        PETSc.Sys.Print('GenEO eigenvalue number {} for lambdamax in subdomain {}: {} <-- not selected (> {})'.format(i, mpi.COMM_WORLD.rank, eps.getEigenvalue(i), tauGenEO_eigmax), comm=self.comm)

            self.eps_eigmax=eps #TODO FIX the only reason for this line is to make sure self.ksp_Atildes and hence PCBNN.ksp is not destroyed
        if self.verbose:
            PETSc.Sys.Print('Subdomain number {} contributes {} coarse vectors after first GenEO'.format(mpi.COMM_WORLD.rank, len(V0s)), comm=self.comm)

    def solve_GenEO_eigmin(self, V0s, tauGenEO_eigmin):
        """
        Solves the local GenEO eigenvalue problem related to the smallest eigenvalue eigmin.

        Parameters
        ==========

        V0s : list of local PETSc .vecs
            V0s may already contain some local coarse vectors. This routine will possibly add more vectors to the list.

        tauGenEO_eigmin: Real.
            Threshold for selecting eigenvectors for the coarse space.

        """
        if tauGenEO_eigmin > 0:
            #to compute the smallest eigenvalues of the preconditioned matrix, Ms must be factorized
            tempksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            tempksp.setOperators(self.Ms)
            tempksp.setType('preonly')
            temppc = tempksp.getPC()
            temppc.setType('cholesky')
            temppc.setFactorSolverType('mumps')
            temppc.setFactorSetUpSolverType()
            tempF = temppc.getFactorMatrix()
            tempF.setMumpsIcntl(7, 2)
            tempF.setMumpsIcntl(24, 1)
            tempF.setMumpsCntl(3, self.mumpsCntl3)
            temppc.setUp()
            tempnrb = tempF.getMumpsInfog(28)

            for i in range(tempnrb):
                tempF.setMumpsIcntl(25, i+1)
                self.works.set(0.)
                tempksp.solve(self.works, self.works)
                V0s.append(self.works.copy())

            tempF.setMumpsIcntl(25, 0)
            if self.verbose:
                PETSc.Sys.Print('Subdomain number {} contributes {} coarse vectors as zero energy modes of the scaled local operator (in GenEO for eigmin)'.format(mpi.COMM_WORLD.rank, tempnrb), comm=self.comm)

            #Eigenvalue Problem for smallest eigenvalues
            eps = SLEPc.EPS().create(comm=PETSc.COMM_SELF)
            eps.setDimensions(nev=self.nev)

            eps.setProblemType(SLEPc.EPS.ProblemType.GHIEP)
            eps.setOperators(self.Ms,self.Atildes)
            eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
            eps.setTarget(0.)
            ST = eps.getST()

            ST.setType("sinvert")
            ST.setKSP(tempksp)

            if len(V0s) > 0 :
                eps.setDeflationSpace(V0s)
            eps.solve()
            if eps.getConverged() < self.nev:
                PETSc.Sys.Print('WARNING: Only {} eigenvalues converged for GenEO_eigmin in subdomain {} whereas {} were requested'.format(eps.getConverged(), mpi.COMM_WORLD.rank, self.nev), comm=self.comm)
            for i in range(min(eps.getConverged(),self.maxev)):
                if(abs(eps.getEigenvalue(i))<tauGenEO_eigmin): #TODO tell slepc that the eigenvalues are real
                   V0s.append(self.works.duplicate())
                   eps.getEigenvector(i,V0s[-1])
                   if self.verbose:
                       PETSc.Sys.Print('GenEO eigenvalue number {} for lambdamin in subdomain {}: {}'.format(i, mpi.COMM_WORLD.rank, eps.getEigenvalue(i)), comm=self.comm)
                else:
                    if self.verbose:
                        PETSc.Sys.Print('GenEO eigenvalue number {} for lambdamin in subdomain {}: {} <-- not selected (> {})'.format(i, mpi.COMM_WORLD.rank, eps.getEigenvalue(i), tauGenEO_eigmin), comm=self.comm)
            self.eps_eigmin=eps #the only reason for this line is to make sure self.ksp_Atildes and hence PCBNN.ksp is not destroyed

    def assemble_coarse_operators(self,V0s):
        """
        Assembles the coarse operators from a list of local contributions to the coarse space.

        Parameters
        ==========

        V0s : list of local PETSc .vecs
           list of the coarse vectors contributed by the subdomain.

        Returns
        ==========

        V0 : list of local vectors or None ? FIX
            list of the local contributions to the coarse space numbered globally: V0[i] is either a scaled local vector from V0s or None if coarse vector number i belongs to another subdomain. The scaling that is applied to the coarse vectors ensures that their A-norm is 1.

        AV0 : list of global PETSc.Vecs
            list of the A*V0[i]. These are global vectors so not in the same format as the vectors in V0.

        Delta : PETSc.Mat (local)
            matrix of the coarse problem. As a result of the scaling of the coarse vectors, its diagonal is 1. This matrix is duplicated over all subdomains This matrix is duplicated over all subdomains

        ksp_Delta : PETSc.ksp
            Krylov subspace solver for the coarse problem matrix Delta.

        """
        if self.verbose:
            PETSc.Sys.Print('Subdomain number {} contributes {} coarse vectors in total'.format(mpi.COMM_WORLD.rank, len(V0s)), comm=self.comm)

        V0 = []
        for i in range(mpi.COMM_WORLD.size):
            nrbl = len(V0s) if i == mpi.COMM_WORLD.rank else None
            nrbl = mpi.COMM_WORLD.bcast(nrbl, root=i)
            for irbm in range(nrbl):
                V0.append(V0s[irbm] if i == mpi.COMM_WORLD.rank else None)

        AV0 = []
        work, _ = self.work
        for vec in V0:
            if vec:
                self.works = vec.copy()
            else:
                self.works.set(0.)
            work.set(0)
            self.scatter_l2g(self.works, work, PETSc.InsertMode.ADD_VALUES)
            AV0.append(self.A*work)
            self.scatter_l2g(AV0[-1], self.works, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
            if vec:
                vec.scale(1./np.sqrt(vec.dot(self.works)))
                self.works = vec.copy()
            else:
                self.works.set(0)
            work.set(0)
            self.scatter_l2g(self.works, work, PETSc.InsertMode.ADD_VALUES)
            AV0[-1] = self.A*work

        PETSc.Sys.Print('There are {} vectors in the coarse space.'.format(len(V0)), comm=mpi.COMM_WORLD)

        #Define, fill and factorize coarse problem matrix
        Delta = PETSc.Mat().create(comm=PETSc.COMM_SELF)
        Delta.setType(PETSc.Mat.Type.SEQDENSE)
        Delta.setSizes([len(V0),len(V0)])
        Delta.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        Delta.setPreallocationDense(None)
        for i, vec in enumerate(V0):
            if vec:
                self.works = vec.copy()
            else:
                self.works.set(0)

            work.set(0)
            self.scatter_l2g(self.works, work, PETSc.InsertMode.ADD_VALUES)
            for j in range(i+1):
                tmp = AV0[j].dot(work)
                Delta[i, j] = tmp
                Delta[j, i] = tmp
        Delta.assemble()
        ksp_Delta = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp_Delta.setOperators(Delta)
        ksp_Delta.setType('preonly')
        pc = ksp_Delta.getPC()
        pc.setType('cholesky')
        return V0, AV0, Delta, ksp_Delta


    def project(self, x):
        """
        Applies the coarse projection (or projection preconditioner) to x

        Parameters
        ==========

        x : PETSc.Vec
           Vector to which the projection is applied and in which the result is stored.

        """
        alpha = self.gamma.duplicate()
        for i, Avec in enumerate(self.AV0):
            self.gamma[i] = Avec.dot(x)

        self.ksp_Delta.solve(self.gamma, alpha)

        self.works.set(0)
        for i, vec in enumerate(self.V0):
            if vec:
                self.works.axpy(-alpha[i], vec)

        self.scatter_l2g(self.works, x, PETSc.InsertMode.ADD_VALUES)

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

        self.scatter_l2g(rhs, self.works, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        self.gamma.set(0)
        self.gamma_tmp.set(0)
        for i, vec in enumerate(self.V0):
            if vec:
                self.gamma_tmp[i] = vec.dot(self.works)

        mpi.COMM_WORLD.Allreduce([self.gamma_tmp, mpi.DOUBLE], [self.gamma, mpi.DOUBLE], mpi.SUM)

        self.ksp_Delta.solve(self.gamma, alpha)

        self.works.set(0)
        for i, vec in enumerate(self.V0):
            if vec:
                self.works.axpy(alpha[i], vec)

        out = rhs.duplicate()
        out.set(0)
        self.scatter_l2g(self.works, out, PETSc.InsertMode.ADD_VALUES)
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

        self.scatter_l2g(x, self.works, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        self.gamma.set(0)
        self.gamma_tmp.set(0)
        for i, vec in enumerate(self.V0):
            if vec:
                self.gamma_tmp[i] = vec.dot(self.works)

        mpi.COMM_WORLD.Allreduce([self.gamma_tmp, mpi.DOUBLE], [self.gamma, mpi.DOUBLE], mpi.SUM)
        self.ksp_Delta.solve(self.gamma, alpha)

        for i in range(len(self.V0)):
            x.axpy(-alpha[i], self.AV0[i])
