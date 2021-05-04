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

#def test(y,x):
#    print(f'y1 {y}')
#    y +=x
#    print(f'y2 {y}')
#    y= 1
#    print(f'y3 {y}')

class minimal_V0(object):
    def __init__(self,ksp_Atildes,V0s=[]):
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
        self.verbose =  OptDB.getBool('PCBNN_verbose', False)

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

            F.setMumpsIcntl(25, 0)
            if self.verbose:
                PETSc.Sys.Print('Subdomain number {} contributes {} coarse vectors as zero energy modes of local solver'.format(mpi.COMM_WORLD.rank, nrb), comm=self.comm)
        else:
            #V0s = []
            nrb = 0
        self.V0s = V0s
        self.nrb = nrb

class GenEO_V0(object):
    def __init__(self,ksp_Atildes,Ms,As,mult_max,V0s=[]):
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
        self.eigmax = OptDB.getReal('PCBNN_GenEO_eigmax', 10)
        self.eigmin = OptDB.getReal('PCBNN_GenEO_eigmin', 0.1)
        self.nev = OptDB.getInt('PCBNN_GenEO_nev', 10)
        self.maxev = OptDB.getInt('PCBNN_GenEO_maxev', 2*self.nev)
        self.mumpsCntl3 = OptDB.getReal('PCBNN_mumpsCntl3', 1e-6)
        self.verbose =  OptDB.getBool('PCBNN_verbose', False)

        self.comm = mpi.COMM_SELF

        self.Ms = Ms
        self.As = As
        self.mult_max = mult_max

        self.ksp_Atildes = ksp_Atildes
        _,Atildes = self.ksp_Atildes.getOperators()
        self.Atildes = Atildes

        works, _ = self.Ms.getVecs()
        self.works = works

        #thresholds for the eigenvalue problems are computed from the user chosen targets for eigmin and eigmax
        tau_eigmin = self.eigmin
        #V0s = []
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
        self.V0s = V0s

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
            ##### trick because of what I think is a bug in the interface with SLEPc
            import copy
            self.copyksp_Atildes = copy.copy(self.ksp_Atildes)
            # self.copyksp_Atildes = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            # self.copyksp_Atildes.setOptionsPrefix("copyksp_Atildes_")
            # self.copyksp_Atildes.setOperators(self.Atildes)
            # self.copyksp_Atildes.setType('preonly')
            # self.copypc_Atildes = self.copyksp_Atildes.getPC()
            # self.copypc_Atildes.setType('cholesky')
            # self.copypc_Atildes.setFactorSolverType('mumps')
            # self.copyksp_Atildes.setFromOptions()
            #### end of the trick
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
            ST.setKSP(self.copyksp_Atildes)
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

class coarse_operators(object):
    def __init__(self,V0s,A,scatter_l2g,works,work):
        """
        PCBNN_verbose : Bool
            Default is False.
            If True, some information about the preconditioners is printed when the code is executed.
        """
        OptDB = PETSc.Options()
        self.verbose =  OptDB.getBool('PCBNN_verbose', False)

        self.comm = mpi.COMM_SELF

        self.scatter_l2g = scatter_l2g
        self.A = A
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
            PETSc.Sys.Print('Subdomain number {} contributes {} coarse vectors in total'.format(mpi.COMM_WORLD.rank, len(V0s)), comm=self.comm)

        V0 = []
        self.work2 = self.work.duplicate()
        for i in range(mpi.COMM_WORLD.size):
            nrbl = len(V0s) if i == mpi.COMM_WORLD.rank else None
            nrbl = mpi.COMM_WORLD.bcast(nrbl, root=i)
            for irbm in range(nrbl):
                V0.append(V0s[irbm] if i == mpi.COMM_WORLD.rank else None)

        AV0 = []
        for vec in V0:
            if vec:
                self.works = vec.copy()
            else:
                self.works.set(0.)
            self.work.set(0)
            self.scatter_l2g(self.works, self.work, PETSc.InsertMode.ADD_VALUES)
            #debug1 = np.sqrt(self.work.dot(self.work))
            #debug4 = self.work.norm()
            #debug3 = np.sqrt(self.works.dot(self.works))
            #debug5 = self.works.norm()
            #print(f'normworks {debug3} = {debug5} normwork {debug1} = {debug4}')
            self.A.mult(self.work,self.work2)
            AV0.append(self.work2.copy())
            self.scatter_l2g(AV0[-1], self.works, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
            if vec:
                vec.scale(1./np.sqrt(vec.dot(self.works)))
                self.works = vec.copy()
            else:
                self.works.set(0)
            self.work.set(0)
            self.scatter_l2g(self.works, self.work, PETSc.InsertMode.ADD_VALUES)
            #self.A.mult(self.work,self.work2)
            AV0[-1] = self.A * self.work
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

            self.work.set(0)
            self.scatter_l2g(self.works, self.work, PETSc.InsertMode.ADD_VALUES)
            for j in range(i+1):
                tmp = AV0[j].dot(self.work)
                Delta[i, j] = tmp
                Delta[j, i] = tmp
                #print(f'i j Deltaij: {i} {j} {tmp}')
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
                #print(f'i j Deltaij: {i} {j} {tmp}')
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
