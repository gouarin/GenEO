# Authors:
#     Loic Gouarin <loic.gouarin@cmap.polytechnique.fr>
#     Nicole Spillane <nicole.spillane@cmap.polytechnique.fr>
#
# License: BSD 3 clause
from .assembling import buildElasticityMatrix
from .bc import bcApplyWestMat, bcApplyWest_vec
from .cg import cg
from .projection import projection, GenEO_V0, minimal_V0, coarse_operators
from petsc4py import PETSc
from slepc4py import SLEPc
import mpi4py.MPI as mpi
import numpy as np
import scipy as sp
import copy

class PCBNN(object): #Neumann-Neumann and Additive Schwarz with no overlap
    def __init__(self, A_IS):
        """
        Initialize the domain decomposition preconditioner, multipreconditioner and coarse space with its operators

        Parameters
        ==========

        A_IS : petsc.Mat
            The matrix of the problem in IS format. A must be a symmetric positive definite matrix
            with symmetric positive semi-definite submatrices

        PETSc.Options
        =============

        PCBNN_switchtoASM :Bool
            Default is False
            If True then the domain decomposition preconditioner is the BNN preconditioner. If false then the domain
            decomposition precondition is the Additive Schwarz preconditioner with minimal overlap.

        PCBNN_kscaling : Bool
            Default is True.
            If true then kscaling (partition of unity that is proportional to the diagonal of the submatrices of A)
            is used when a partition of unity is required. Otherwise multiplicity scaling is used when a partition
            of unity is required. This may occur in two occasions:
              - to scale the local BNN matrices if PCBNN_switchtoASM=True,
              - in the GenEO eigenvalue problem for eigmin if PCBNN_switchtoASM=False and PCBNN_GenEO=True with
                PCBNN_GenEO_eigmin > 0 (see projection.__init__ for the meaning of these options).

        PCBNN_verbose : Bool
            If True, some information about the preconditioners is printed when the code is executed.

        PCBNN_GenEO : Bool
            Default is False.
            If True then the coarse space is enriched by solving local generalized eigenvalue problems.

        PCBNN_CoarseProjection : Bool
            Default is True.
            If False then there is no coarse projection: Two level Additive Schwarz or One-level preconditioner depending on PCBNN_addCoarseSolve.
            If True, the coarse projection is applied: Projected preconditioner of hybrid preconditioner depending on PCBNN_addCoarseSolve.

        PCBNN_addCoarseSolve : Bool
            Default is True.
            If True then (R0t A0\R0 r) is added to the preconditioned residual.
            False corresponds to the projected preconditioner (need to choose initial guess accordingly) (or the one level preconditioner if PCBNN_CoarseProjection = False).
            True corresponds to the hybrid preconditioner (or the fully additive preconditioner if PCBNN_CoarseProjection = False).
        """
        OptDB = PETSc.Options()
        self.switchtoASM = OptDB.getBool('PCBNN_switchtoASM', False) #use Additive Schwarz as a preconditioner instead of BNN
        self.kscaling = OptDB.getBool('PCBNN_kscaling', True) #kscaling if true, multiplicity scaling if false
        self.verbose = OptDB.getBool('PCBNN_verbose', False)
        self.GenEO = OptDB.getBool('PCBNN_GenEO', True)
        self.addCS = OptDB.getBool('PCBNN_addCoarseSolve', True)
        self.projCS = OptDB.getBool('PCBNN_CoarseProjection', True)
        self.viewPC = OptDB.getBool('PCBNN_view', True)
        self.viewV0 = OptDB.getBool('PCBNN_viewV0', False)
        self.viewGenEOV0 = OptDB.getBool('PCBNN_viewGenEO', False)
        self.viewminV0 = OptDB.getBool('PCBNN_viewminV0', False)

        #extract Neumann matrix from A in IS format
        Ms = A_IS.copy().getISLocalMat()

        # convert A_IS from matis to mpiaij
        A_mpiaij = A_IS.convert('mpiaij')
        r, _ = A_mpiaij.getLGMap() #r, _ = A_IS.getLGMap()
        is_A = PETSc.IS().createGeneral(r.indices)
        # extract exact local solver
        As = A_mpiaij.createSubMatrices(is_A)[0]

        vglobal, _ = A_mpiaij.getVecs()
        vlocal, _ = Ms.getVecs()
        scatter_l2g = PETSc.Scatter().create(vlocal, None, vglobal, is_A)

        #compute the multiplicity of each degree
        vlocal.set(1.)
        vglobal.set(0.)
        scatter_l2g(vlocal, vglobal, PETSc.InsertMode.ADD_VALUES)
        scatter_l2g(vglobal, vlocal, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        NULL,mult_max = vglobal.max()

        # k-scaling or multiplicity scaling of the local (non-assembled) matrix
        if self.kscaling == False:
            Ms.diagonalScale(vlocal,vlocal)
        else:
            v1 = As.getDiagonal()
            v2 = Ms.getDiagonal()
            Ms.diagonalScale(v1/v2, v1/v2)

        # the default local solver is the scaled non assembled local matrix (as in BNN)
        if self.switchtoASM:
            Atildes = As
            if mpi.COMM_WORLD.rank == 0:
                print('The user has chosen to switch to Additive Schwarz instead of BNN.')
        else: #(default)
            Atildes = Ms
        ksp_Atildes = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp_Atildes.setOptionsPrefix("ksp_Atildes_")
        ksp_Atildes.setOperators(Atildes)
        ksp_Atildes.setType('preonly')
        pc_Atildes = ksp_Atildes.getPC()
        pc_Atildes.setType('cholesky')
        pc_Atildes.setFactorSolverType('mumps')
        ksp_Atildes.setFromOptions()

        ksp_Atildes_forSLEPc = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp_Atildes_forSLEPc.setOptionsPrefix("ksp_Atildes_")
        ksp_Atildes_forSLEPc.setOperators(Atildes)
        ksp_Atildes_forSLEPc.setType('preonly')
        pc_Atildes_forSLEPc = ksp_Atildes_forSLEPc.getPC()
        pc_Atildes_forSLEPc.setType('cholesky')
        pc_Atildes_forSLEPc.setFactorSolverType('mumps')
        ksp_Atildes_forSLEPc.setFromOptions()

        self.A = A_mpiaij
        self.Ms = Ms
        self.As = As
        self.ksp_Atildes = ksp_Atildes
        self.ksp_Atildes_forSLEPc = ksp_Atildes_forSLEPc
        self.work = vglobal.copy()
        self.works_1 = vlocal.copy()
        self.works_2 = self.works_1.copy()
        self.scatter_l2g = scatter_l2g
        self.mult_max = mult_max

        self.minV0 = minimal_V0(self.ksp_Atildes)
        if self.viewminV0 == True:
            self.minV0.view()
        if self.GenEO == True:
            self.GenEOV0 = GenEO_V0(self.ksp_Atildes_forSLEPc,self.Ms,self.As,self.mult_max,self.minV0.V0s)
            self.V0s = self.GenEOV0.V0s
            if self.viewGenEOV0 == True:
                self.GenEOV0.view()
        else:
            self.V0s = self.minV0.V0s
        self.proj = coarse_operators(self.V0s,self.A,self.scatter_l2g,vlocal,self.work)
        if self.viewV0 == True:
            self.proj.view()

        #self.proj = projection(self)

        if self.viewPC == True:
            self.view()            
    def mult(self, x, y):
        """
        Applies the domain decomposition preconditioner followed by the projection preconditioner to a vector.

        Parameters
        ==========

        x : petsc.Vec
            The vector to which the preconditioner is to be applied.

        y : petsc.Vec
            The vector that stores the result of the preconditioning operation.

        """
########################
########################
        xd = x.copy()
        if self.projCS == True:
            self.proj.project_transpose(xd)

        self.scatter_l2g(xd, self.works_1, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        self.ksp_Atildes.solve(self.works_1, self.works_2)

        y.set(0.)
        self.scatter_l2g(self.works_2, y, PETSc.InsertMode.ADD_VALUES)
        if self.projCS == True:
            self.proj.project(y)

        if self.addCS == True:
            xd = x.copy()
            ytild = self.proj.coarse_init(xd) # I could save a coarse solve by combining this line with project_transpose
            y += ytild

    def MP_mult(self, x, y):
        """
        Applies the domain decomposition multipreconditioner followed by the projection preconditioner to a vector.

        Parameters
        ==========

        x : petsc.Vec
            The vector to which the preconditioner is to be applied.

        y : FIX
            The list of ndom vectors that stores the result of the multipreconditioning operation (one vector per subdomain).

        """
        self.scatter_l2g(x, self.works_1, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        self.ksp_Atildes.solve(self.works_1, self.works_2)
        for i in range(mpi.COMM_WORLD.size):
            self.works_1.set(0)
            if mpi.COMM_WORLD.rank == i:
                self.works_1 = self.works_2.copy()
            y[i].set(0.)
            self.scatter_l2g(self.works_1, y[i], PETSc.InsertMode.ADD_VALUES)
            self.proj.project(y[i])

    def apply(self,pc, x, y):
        """
        Applies the domain decomposition preconditioner followed by the projection preconditioner to a vector.
        This is just a call to PCBNN.mult with the function name and arguments that allow PCBNN to be passed
        as a preconditioner to PETSc.ksp.

        Parameters
        ==========

        pc: This argument is not called within the function but it belongs to the standard way of calling a preconditioner.

        x : petsc.Vec
            The vector to which the preconditioner is to be applied.

        y : petsc.Vec
            The vector that stores the result of the preconditioning operation.

        """
        self.mult(x,y)
    def view(self):
        self.minV0.gathered_dim = mpi.COMM_WORLD.gather(self.minV0.nrb, root=0)
        if self.GenEO == True:
            self.GenEOV0.gathered_nsharp = mpi.COMM_WORLD.gather(self.GenEOV0.n_GenEO_eigmax, root=0)
            self.GenEOV0.gathered_nflat = mpi.COMM_WORLD.gather(self.GenEOV0.n_GenEO_eigmin, root=0)
            self.GenEOV0.gathered_dimKerMs = mpi.COMM_WORLD.gather(self.GenEOV0.dimKerMs, root=0)
            self.GenEOV0.gathered_Lambdasharp = mpi.COMM_WORLD.gather(self.GenEOV0.Lambda_GenEO_eigmax, root=0)
            self.GenEOV0.gathered_Lambdaflat = mpi.COMM_WORLD.gather(self.GenEOV0.Lambda_GenEO_eigmin, root=0)
        if mpi.COMM_WORLD.rank == 0:
            print('#############################')
            print(f'view of PCBNN')
            print(f'{self.switchtoASM=}')
            print(f'{self.kscaling= }')
            print(f'{self.verbose= }')
            print(f'{self.GenEO= }')
            print(f'{self.addCS= }')
            print(f'{self.projCS= }')
            print(f'{self.viewPC= }')
            print(f'{self.viewV0= }')
            print(f'{self.viewGenEOV0= }')
            print(f'{self.viewminV0= }')
            print(f'### info about minV0.V0s = (Ker(Atildes)) ###')
            print(f'{self.minV0.mumpsCntl3=}') 
            if (self.ksp_Atildes.pc.getFactorSolverType() == 'mumps'):
                print(f'dim(Ker(Atildes)) = {self.minV0.gathered_dim}')
            else:
                print(f'Ker(Atildes) not computed because pc is not mumps')
            if self.GenEO == True:
                print(f'### info about GenEOV0.V0s = (Ker(Atildes)) ###')
                print(f'{self.GenEOV0.tau_eigmax=}') 
                print(f'{self.GenEOV0.tau_eigmin=}') 
                print(f'{self.GenEOV0.eigmax=}') 
                print(f'{self.GenEOV0.eigmin=}') 
                print(f'{self.GenEOV0.nev=}') 
                print(f'{self.GenEOV0.maxev=}') 
                print(f'{self.GenEOV0.mumpsCntl3=}') 
                print(f'{self.GenEOV0.verbose=}') 
                print(f'{self.GenEOV0.mult_max=}') 
                print(f'{self.GenEOV0.gathered_nsharp=}')
                print(f'{self.GenEOV0.gathered_nflat=}') 
                print(f'{self.GenEOV0.gathered_dimKerMs=}')
                #print(f'{np.array(self.GenEOV0.gathered_Lambdasharp)=}')
                #print(f'{np.array(self.GenEOV0.gathered_Lambdaflat)=}')
            print(f'### info about the coarse space ###')
            print(f'{self.proj.V0_is_global=}') 
            print(f'{self.proj.gathered_dimV0s=}') 
            if self.GenEO == True:
                print(f'global dim V0 = {np.sum(self.proj.gathered_dimV0s)} = ({np.sum(self.minV0.gathered_dim)} from Ker(Atildes)) + ({np.sum(self.GenEOV0.gathered_nsharp)} from GenEO_eigmax) + ({np.sum(self.GenEOV0.gathered_nflat)+np.sum(self.GenEOV0.gathered_dimKerMs)} from GenEO_eigmin)') 
            else:
                print(f'global dim V0 = {np.sum(self.proj.gathered_dimV0s)} = ({np.sum(self.minV0.gathered_dim)} from Ker(Atildes))') 

            print('#############################')

class PCNew:
    def __init__(self, A_IS):
        OptDB = PETSc.Options()
        self.switchtoASM = OptDB.getBool('PCNew_switchtoASM', False) #use Additive Schwarz as a preconditioner instead of BNN
        self.switchtoASMpos = OptDB.getBool('PCNew_switchtoASMpos', False) #use Additive Schwarz as a preconditioner instead of BNN
        self.verbose = OptDB.getBool('PCNew_verbose', False)
        self.GenEO = OptDB.getBool('PCNew_GenEO', True)
        #self.H2addCS = OptDB.getBool('PCNew_H2addCoarseSolve', True)
        self.H2projCS = OptDB.getBool('PCNew_H2CoarseProjection', True)
        self.H3addCS = OptDB.getBool('PCNew_H3addCoarseSolve', True)
        self.H3projCS = OptDB.getBool('PCNew_H3CoarseProjection', True)
        self.compute_ritz_apos = OptDB.getBool('PCNew_ComputeRitzApos', False)
        self.nev = OptDB.getInt('PCNew_Bs_nev', 20) #number of vectors asked to SLEPc for cmputing negative part of Bs

        self.viewPC = OptDB.getBool('PCNew_view', True)
        self.viewV0 = OptDB.getBool('PCNew_viewV0', False)
        self.viewGenEOV0 = OptDB.getBool('PCNew_viewGenEO', False)
        self.viewminV0 = OptDB.getBool('PCNew_viewminV0', False)
        self.viewnegV0 = OptDB.getBool('PCNew_viewnegV0', False)

        self.H2addCS = True #OptDB.getBool('PCNew_H2addCoarseSolve', True) (it is currently not an option to use a projected preconditioner for H2)
        # Compute Bs (the symmetric matrix in the algebraic splitting of A)
        # TODO: implement without A in IS format
        ANeus = A_IS.getISLocalMat() #only the IS is used for the algorithm,
        Mu = A_IS.copy()
        Mus = Mu.getISLocalMat() #the IS format is used to compute Mu (multiplicity of each pair of dofs)

        for i in range(ANeus.getSize()[0]):
            col, _ = ANeus.getRow(i)
            Mus.setValues([i], col, np.ones_like(col))
        Mu.restoreISLocalMat(Mus)
        Mu.assemble()
        Mu = Mu.convert('mpiaij')

        A_mpiaij = A_IS.convert('mpiaij')
        B = A_mpiaij.duplicate()
        for i in range(*A_mpiaij.getOwnershipRange()):
            a_cols, a_values = A_mpiaij.getRow(i)
            _, b_values = Mu.getRow(i)
            B.setValues([i], a_cols, a_values/b_values, PETSc.InsertMode.INSERT_VALUES)

        B.assemble()

        # B.view()
        # A_mpiaij.view()
        # (A_mpiaij - B).view()
        # data = ANeus.getArray()
        # if mpi.COMM_WORLD.rank == 0:
        #     print(dir(ANeus))
        #     print(type(ANeus), ANeus.getType())
###################@


        # convert A_IS from matis to mpiaij
        #A_mpiaij = A_IS.convertISToAIJ()
        r, _ = A_mpiaij.getLGMap() #r, _ = A_IS.getLGMap()
        is_A = PETSc.IS().createGeneral(r.indices)
        # extract exact local solver
        As = A_mpiaij.createSubMatrices(is_A)[0]
        Bs = B.createSubMatrices(is_A)[0]

        #mumps solver for Bs
        Bs_ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        Bs_ksp.setOptionsPrefix("Bs_ksp_")
        Bs_ksp.setOperators(Bs)
        Bs_ksp.setType('preonly')
        Bs_pc = Bs_ksp.getPC()
        Bs_pc.setType('cholesky')
        Bs_pc.setFactorSolverType('mumps')
        Bs_pc.setFactorSetUpSolverType()
        Bs_pc.setUp()
        Bs_ksp.setFromOptions()


        #temp = Bs.getValuesCSR()

        work, _ = A_mpiaij.getVecs()
        work_2 = work.duplicate()
        works, _ = As.getVecs()
        works_2 = works.duplicate()
        mus = works.duplicate()
        scatter_l2g = PETSc.Scatter().create(works, None, work, is_A)

        #compute the multiplicity of each dof
        work = Mu.getDiagonal()
        NULL,mult_max = work.max()

        scatter_l2g(work, mus, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        invmus = mus.duplicate()
        invmus = 1/mus
        if mpi.COMM_WORLD.rank == 0:
            print(f'multmax: {mult_max}')


        DVnegs = []
        Vnegs = []
        invmusVnegs = []

        #BEGIN diagonalize Bs
        #Eigenvalue Problem for smallest eigenvalues
        eps = SLEPc.EPS().create(comm=PETSc.COMM_SELF)
        eps.setDimensions(nev=self.nev)
        eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
        eps.setOperators(Bs)

        #print(f'dimension of Bs : {Bs.getSize()}')


        #OPTION 1: works but dense algebra
        eps.setType(SLEPc.EPS.Type.LAPACK)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL) #with lapack this just tells slepc how to order the eigenpairs
        ##END OPTION 1

        ##OPTION 2: default solver (Krylov Schur) but error with getInertia - is there a MUMPS mattype - Need to use MatCholeskyFactor
               #if Which eigenpairs is set to SMALLEST_REAL, some are computed but not all

        ##Bs.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        ##Bs.convert('sbaij')
        ##IScholBs = is_A.duplicate()
        ##Bs.factorCholesky(IScholBs) #not implemented
        #tempksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        #tempksp.setOperators(Bs)
        #tempksp.setType('preonly')
        #temppc = tempksp.getPC()
        #temppc.setType('cholesky')
        #temppc.setFactorSolverType('mumps')
        #temppc.setFactorSetUpSolverType()
        #tempF = temppc.getFactorMatrix()
        #tempF.setMumpsIcntl(13, 1) #needed to compute intertia according to slepcdoc, inertia computation still doesn't work though
        #temppc.setUp()
        ##eps.setOperators(tempF)
        #eps.setWhichEigenpairs(SLEPc.EPS.Which.ALL)
        #eps.setInterval(PETSc.NINFINITY,0.0)
        #eps.setUp()

        ##eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        ##eps.setTarget(0.)

        ##if len(Vnegs) > 0 :
        ##    eps.setDeflationSpace(Vnegs)
        ##if mpi.COMM_WORLD.rank == 0:
        ##    eps.view()
        ##END OPTION 2

        eps.solve()
        if eps.getConverged() < self.nev:
            PETSc.Sys.Print('for Bs in subdomain {}: {} eigenvalues converged (less that the {} requested)'.format(mpi.COMM_WORLD.rank, eps.getConverged(), self.nev), comm=PETSc.COMM_SELF)

        Dnegs = []
        Dposs = []
        for i in range(eps.getConverged()):
            tempscalar = np.real(eps.getEigenvalue(i))
            if tempscalar < 0. :
                Dnegs.append(-1.*tempscalar)
                Vnegs.append(works.duplicate())
                eps.getEigenvector(i,Vnegs[-1])
                DVnegs.append(Dnegs[-1] * Vnegs[-1])
                invmusVnegs.append(invmus * Vnegs[-1])
            else :
                Dposs.append(tempscalar)
        if self.verbose: 
            PETSc.Sys.Print('for Bs in subdomain {}: ncv= {} with {} negative eigs'.format(mpi.COMM_WORLD.rank, eps.getConverged(), len(Vnegs), self.nev), comm=PETSc.COMM_SELF)
            print(f'values of Dnegs {np.array(Dnegs)}')
        nnegs = len(Dnegs)
        #print(f'length of Dnegs {nnegs}')
        #END diagonalize Bs

        if self.viewnegV0: 
            print('###')
            print(f'view of Vneg in Subdomain {mpi.COMM_WORLD.rank}')
            print(f'ncv = {eps.getConverged()} eigenvalues converged')
            print(f'{nnegs=}') 
            print(f'values of Dnegs: {np.array(Dnegs)}')

        works.set(0.)
        RsVnegs = []
        Vneg = []
        Dneg = []
        RsDVnegs = []
        RsDnegs = []
        for i in range(mpi.COMM_WORLD.size):
            nnegi = len(Vnegs) if i == mpi.COMM_WORLD.rank else None
            nnegi = mpi.COMM_WORLD.bcast(nnegi, root=i)
            for j in range(nnegi):
                Vneg.append(Vnegs[j].copy() if i == mpi.COMM_WORLD.rank else works.copy())
                dnegi = Dnegs[j] if i == mpi.COMM_WORLD.rank else None
                dnegi = mpi.COMM_WORLD.bcast(dnegi, root=i)
                Dneg.append(dnegi)
                #print(f'i Dneg[i] = {i} {Dneg[i]}')
        for i, vec in enumerate(Vneg):
            work.set(0)
            scatter_l2g(vec, work, PETSc.InsertMode.ADD_VALUES)
            scatter_l2g(work, works, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)

            if works.norm() != 0:
                RsVnegs.append(works.copy())
                RsDVnegs.append(Dneg[i]*works.copy())
                RsDnegs.append(Dneg[i])
            #TO DO: here implement RsVnegs and RsDVnegs
        #self.Vneg = Vneg

#        self.Vnegs = Vnegs
#        self.DVnegs = DVnegs
#        self.scatterl

#Local Apos and Aneg
        Aneg = PETSc.Mat().createPython([work.getSizes(), work.getSizes()], comm=PETSc.COMM_WORLD)
        Aneg.setPythonContext(Aneg_ctx(Vnegs, DVnegs, scatter_l2g, works, works_2))
        Aneg.setUp()

        Apos = PETSc.Mat().createPython([work.getSizes(), work.getSizes()], comm=PETSc.COMM_WORLD)
        Apos.setPythonContext(Apos_ctx(A_mpiaij, Aneg ))
        Apos.setUp()
        #A pos = A_mpiaij + Aneg so it could be a composite matrix rather than Python type

        Anegs = PETSc.Mat().createPython([works.getSizes(), works.getSizes()], comm=PETSc.COMM_SELF)
        Anegs.setPythonContext(Anegs_ctx(Vnegs, DVnegs))
        Anegs.setUp()

        Aposs = PETSc.Mat().createPython([works.getSizes(), works.getSizes()], comm=PETSc.COMM_SELF)
        Aposs.setPythonContext(Aposs_ctx(Bs, Anegs ))
        Aposs.setUp()

        projVnegs = PETSc.Mat().createPython([works.getSizes(), works.getSizes()], comm=PETSc.COMM_SELF)
        projVnegs.setPythonContext(projVnegs_ctx(Vnegs))
        projVnegs.setUp()

        projVposs = PETSc.Mat().createPython([works.getSizes(), works.getSizes()], comm=PETSc.COMM_SELF)
        projVposs.setPythonContext(projVposs_ctx(projVnegs))
        projVposs.setUp()

        #TODO Implement RsAposRsts, this is the restriction of Apos to the dofs in this subdomain. So it applies to local vectors but has non local operations
        RsAposRsts = PETSc.Mat().createPython([works.getSizes(), works.getSizes()], comm=PETSc.COMM_SELF) #or COMM_WORLD ?
        RsAposRsts.setPythonContext(RsAposRsts_ctx(As,RsVnegs,RsDVnegs))
        RsAposRsts.setUp()

        invAposs = PETSc.Mat().createPython([works.getSizes(), works.getSizes()], comm=PETSc.COMM_SELF)
        invAposs.setPythonContext(invAposs_ctx(Bs_ksp, projVposs ))
        invAposs.setUp()

        ksp_Aposs = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp_Aposs.setOperators(Aposs)
        ksp_Aposs.setType('preonly')
        pc_Aposs = ksp_Aposs.getPC()
        pc_Aposs.setType('python')
        pc_Aposs.setPythonContext(invAposs_ctx(Bs_ksp,projVposs))
        ksp_Aposs.setUp()
        work.set(1.)

        Ms = PETSc.Mat().createPython([works.getSizes(), works.getSizes()], comm=PETSc.COMM_SELF)
        Ms.setPythonContext(scaledmats_ctx(Aposs, mus, mus))
        Ms.setUp()

        ksp_Ms = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp_Ms.setOptionsPrefix("ksp_Ms_")
        ksp_Ms.setOperators(Ms)
        ksp_Ms.setType('preonly')
        pc_Ms = ksp_Ms.getPC()
        pc_Ms.setType('python')
        pc_Ms.setPythonContext(scaledmats_ctx(invAposs,invmus,invmus) )
        ksp_Ms.setFromOptions()

            #once a ksp has been passed to SLEPs it cannot be used again so we use a second, identical, ksp for SLEPc as a temporary fix
        ksp_Ms_forSLEPc = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp_Ms_forSLEPc.setOptionsPrefix("ksp_Ms_")
        ksp_Ms_forSLEPc.setOperators(Ms)
        ksp_Ms_forSLEPc.setType('preonly')
        pc_Ms_forSLEPc = ksp_Ms_forSLEPc.getPC()
        pc_Ms_forSLEPc.setType('python')
        pc_Ms_forSLEPc.setPythonContext(scaledmats_ctx(invAposs,invmus,invmus) )
        ksp_Ms_forSLEPc.setFromOptions()

        # the default local solver is the scaled non assembled local matrix (as in BNN)
        if self.switchtoASM:
            Atildes = As
            if mpi.COMM_WORLD.rank == 0:
                print('Switch to Additive Schwarz instead of BNN.')
            ksp_Atildes = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            ksp_Atildes.setOptionsPrefix("ksp_Atildes_")
            ksp_Atildes.setOperators(Atildes)
            ksp_Atildes.setType('preonly')
            pc_Atildes = ksp_Atildes.getPC()
            pc_Atildes.setType('cholesky')
            pc_Atildes.setFactorSolverType('mumps')
            ksp_Atildes.setFromOptions()
            minV0 = minimal_V0(ksp_Atildes,invmusVnegs)
            minV0s = minV0.V0s
            if self.viewminV0 == True:
                self.minV0.view()

            #once a ksp has been passed to SLEPs it cannot be used again so we use a second, identical, ksp for SLEPc as a temporary fix
            ksp_Atildes_forSLEPc = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            ksp_Atildes_forSLEPc.setOptionsPrefix("ksp_Atildes_")
            ksp_Atildes_forSLEPc.setOperators(Atildes)
            ksp_Atildes_forSLEPc.setType('preonly')
            pc_Atildes_forSLEPc = ksp_Atildes_forSLEPc.getPC()
            pc_Atildes_forSLEPc.setType('cholesky')
            pc_Atildes_forSLEPc.setFactorSolverType('mumps')
            ksp_Atildes_forSLEPc.setFromOptions()
            if self.switchtoASMpos:
                if mpi.COMM_WORLD.rank == 0:
                    print('switchtoASMpos has been ignored in favour of switchtoASM.')
        elif self.switchtoASMpos:
            Atildes = RsAposRsts
            if mpi.COMM_WORLD.rank == 0:
                print('Switch to Apos Additive Schwarz instead of BNN.')
            ksp_Atildes = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            ksp_Atildes.setOptionsPrefix("ksp_Atildes_")
            ksp_Atildes.setOperators(Atildes)
            ksp_Atildes.setType('preonly')
            pc_Atildes = ksp_Atildes.getPC()
            pc_Atildes.setType('python')
            pc_Atildes.setPythonContext(invRsAposRsts_ctx(As,RsVnegs,RsDnegs,works))
            ksp_Atildes.setFromOptions()
            minV0 = minimal_V0(ksp_Atildes,invmusVnegs)
            minV0s = minV0.V0s
            if self.viewminV0 == True:
                self.minV0.view()

            #once a ksp has been passed to SLEPs it cannot be used again so we use a second, identical, ksp for SLEPc as a temporary fix
            ksp_Atildes_forSLEPc = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            ksp_Atildes_forSLEPc.setOptionsPrefix("ksp_Atildes_")
            ksp_Atildes_forSLEPc.setOperators(Atildes)
            ksp_Atildes_forSLEPc.setType('preonly')
            pc_Atildes_forSLEPc = ksp_Atildes_forSLEPc.getPC()
            pc_Atildes_forSLEPc.setType('python')
            pc_Atildes_forSLEPc.setPythonContext(invRsAposRsts_ctx(As,RsVnegs,RsDnegs,works))
            ksp_Atildes_forSLEPc.setFromOptions()
        else: #(default)
            Atildes = Ms
            ksp_Atildes = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            ksp_Atildes.setOptionsPrefix("ksp_Atildes_")
            ksp_Atildes.setOperators(Atildes)
            ksp_Atildes.setType('preonly')
            pc_Atildes = ksp_Atildes.getPC()
            pc_Atildes.setType('python')
            pc_Atildes.setPythonContext(scaledmats_ctx(invAposs,invmus,invmus) )
            ksp_Atildes.setFromOptions()
            minV0 = minimal_V0(ksp_Atildes,invmusVnegs) #won't compute anything more vecause the solver for Atildes is not mumps
            minV0s = minV0.V0s
            if self.viewminV0 == True:
                self.minV0.view()


            #once a ksp has been passed to SLEPs it cannot be used again so we use a second, identical, ksp for SLEPc as a temporary fix
            ksp_Atildes_forSLEPc = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            ksp_Atildes_forSLEPc.setOptionsPrefix("ksp_Atildes_")
            ksp_Atildes_forSLEPc.setOperators(Atildes)
            ksp_Atildes_forSLEPc.setType('preonly')
            pc_Atildes_forSLEPc = ksp_Atildes_forSLEPc.getPC()
            pc_Atildes_forSLEPc.setType('python')
            pc_Atildes_forSLEPc.setPythonContext(scaledmats_ctx(invAposs,invmus,invmus) )
            ksp_Atildes_forSLEPc.setFromOptions()

        self.A = A_mpiaij
        self.Apos = Apos
        self.Ms = Ms
        self.As = As
        self.RsAposRsts = RsAposRsts
        self.ksp_Atildes = ksp_Atildes
        self.ksp_Ms = ksp_Ms
        self.ksp_Atildes_forSLEPc = ksp_Atildes_forSLEPc
        self.ksp_Ms_forSLEPc = ksp_Ms_forSLEPc
        self.work = work
        self.work_2 = work_2
        self.works_1 = works
        self.works_2 = works_2
        self.scatter_l2g = scatter_l2g
        self.mult_max = mult_max
        self.ksp_Atildes = ksp_Atildes
        self.minV0 = minV0
        self.Dnegs = Dnegs
        self.nnegs = nnegs

        self.works_1.set(1.)
        self.RsAposRsts.mult(self.works_1,self.works_2)

        if self.GenEO == True:
          self.GenEOV0 = GenEO_V0(self.ksp_Atildes_forSLEPc,self.Ms,self.RsAposRsts,self.mult_max,minV0s,self.ksp_Ms_forSLEPc)
          self.V0s = self.GenEOV0.V0s
          if self.viewGenEOV0 == True:
              self.GenEOV0.view()
        else:
          self.V0s = minV0s

        self.proj2 = coarse_operators(self.V0s,self.Apos,self.scatter_l2g,self.works_1,self.work)
        if self.viewV0 == True:
            self.proj2.view()
#        work.set(1.)
#        test = work.copy()
#        test = self.proj2.coarse_init(work)
#        testb = work.copy()
#        self.proj2.project(testb)
#        testc = work.copy()
#        self.proj2.project_transpose(testc)
#        testd = work.copy()
#        self.apply([], work,testd)
        self.H2 = PETSc.Mat().createPython([work.getSizes(), work.getSizes()], comm=PETSc.COMM_WORLD)
        self.H2.setPythonContext(H2_ctx(self.H2projCS, self.H2addCS, self.proj2, self.scatter_l2g, self.ksp_Atildes, self.works_1, self.works_2 ))
        self.H2.setUp()

        self.ksp_Apos = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
        self.ksp_Apos.setOptionsPrefix("ksp_Apos_")
        self.ksp_Apos.setOperators(Apos)
        self.ksp_Apos.setType("cg")
        if self.compute_ritz_apos:
            self.ksp_Apos.setComputeEigenvalues(True)
        self.pc_Apos = self.ksp_Apos.getPC()
        self.pc_Apos.setType('python')
        self.pc_Apos.setPythonContext(H2_ctx(self.H2projCS, self.H2addCS, self.proj2, self.scatter_l2g, self.ksp_Atildes, self.works_1, self.works_2 ))
        self.ksp_Apos.setFromOptions()
        self.pc_Apos.setFromOptions()
        #At this point the preconditioner for Apos is ready
        if self.verbose: 
            if mpi.COMM_WORLD.rank == 0:
                print(f'#V0(H2) = rank(Ker(Pi2)) = {len(self.proj2.V0)}')
        works.set(0.)
        Vneg = []
        for i in range(mpi.COMM_WORLD.size):
            nnegi = len(Vnegs) if i == mpi.COMM_WORLD.rank else None
            nnegi = mpi.COMM_WORLD.bcast(nnegi, root=i)
            for j in range(nnegi):
                Vneg.append(Vnegs[j].copy() if i == mpi.COMM_WORLD.rank else works.copy())
        AposinvV0 = []
        self.ritz_eigs_apos = [] 
        for vec in Vneg:
            self.works = vec.copy()
            self.work.set(0)
            self.scatter_l2g(self.works, self.work, PETSc.InsertMode.ADD_VALUES)
            self.ksp_Apos.solve(self.work,self.work_2)
            if self.compute_ritz_apos and self.ritz_eigs_apos == []:
                self.ritz_eigs_apos = self.ksp_Apos.computeEigenvalues()
                self.ksp_Apos.setComputeEigenvalues(False)

            AposinvV0.append(self.work_2.copy())
            ##
            Aposx = self.work.duplicate()
            Apos.mult(self.work_2,Aposx)
            ##
        self.AposinvV0 = AposinvV0
        self.proj3 = coarse_operators(self.AposinvV0,self.A,self.scatter_l2g,self.works_1,self.work,V0_is_global=True)
        self.proj = self.proj3 #this name is consistent with the proj in PCBNN

        if self.viewV0 == True:
            self.proj.view()

        if self.viewPC == True:
            self.view()            


##Debug DEBUG
#        works_3 = works.copy()
##projVnegs is a projection
#        #works.setRandom()
#        works.set(1.)
#        projVnegs.mult(works,works_2)
#        projVnegs.mult(works_2,works_3)
#        print(f'check that projVnegs is a projection {works_2.norm()} = {works_3.norm()} < {works.norm()}')
##projVposs is a projection
##Pythagoras ok
#        works.setRandom()
#        #works.set(1.)
#        projVnegs.mult(works,works_2)
#        projVposs.mult(works,works_3)
#        print(f'{works_2.norm()**2} +  {works_3.norm()**2}= {works_2.norm()**2 +  works_3.norm()**2}  =  {(works.norm())**2}')
#        print(f'0 = {(works - works_2 - works_3).norm()} if the two projections sum to identity')
##Aposs = projVposs Bs projVposs = Bs projVposs  (it is implemented as Bs + Anegs)
#        works_4 = works.copy()
#        works.setRandom()
#        #works.set(1.)
#        projVposs.mult(works,works_2)
#        Bs.mult(works_2,works_3)
#        projVposs.mult(works_3,works_2)
#        Aposs.mult(works,works_4)
#        print(f'check Aposs = projVposs Bs projVposs = Bs projVposs: {works_2.norm()} = {works_3.norm()} = {works_4.norm()}')
#        print(f'norms of diffs (should be zero): {(works_2 - works_3).norm()}, {(works_2 - works_4).norm()}, {(works_3 - works_4).norm()}')
###check that Aposs > 0 and Anegs >0 but Bs is indefinite + "Pythagoras"
#        works_4 = works.copy()
#        works.set(1.) #(with vector full of ones I get a negative Bs semi-norm)
#        Bs.mult(works,works_4)
#        Aposs.mult(works,works_2)
#        Anegs.mult(works,works_3)
#        print(f'|.|_Bs {works_4.dot(works)} (can be neg or pos); |.|_Aposs {works_2.dot(works)} > 0;  |.|_Anegs  {works_3.dot(works)} >0')
#        print(f' |.|_Bs^2 = |.|_Aposs^2 -  |.|_Anegs ^2 = {works_2.dot(works)} - {works_3.dot(works)} = {works_2.dot(works) - works_3.dot(works)} = {works_4.dot(works)} ')##
###check that ksp_Aposs.solve(Aposs *  x) = projVposs x
#        works_4 = works.copy()
#        works.setRandom()
#        #works.set(1.)
#        projVposs.mult(works,works_2)
#        Aposs(works,works_3)
#        ksp_Aposs.solve(works_3,works_4)
#        works_5 = works_2 - works_4
#        print(f'norm x = {works.norm()}; norm projVposs x = {works_2.norm()} = norm Aposs\Aposs*x = {works_4.norm()}; normdiff = {works_5.norm()}')
####check that mus*invmus = vec of ones
#        works.set(1.0)
#        works_2 = invmus*mus
#        works_3 = works - works_2
#        print(f'0 = norm(vec of ones - mus*invmus)   = {works_3.norm()}, mus in [{mus.min()}, {mus.max()}], invmus in [{invmus.min()}, {invmus.max()}]')
###check that Ms*ksp_Ms.solve(Ms*x) = Ms*x
#        works_4 = works.copy()
#        works.setRandom()
#        Atildes.mult(works,works_3)
#        self.ksp_Atildes.solve(works_3,works_4)
#        Atildes.mult(works_4,works_2)
#        works_5 = works_2 - works_3
#        print(f'norm x = {works.norm()}; Atilde*x = {works_3.norm()} = norm Atilde*(Atildes\Atildes)*x = {works_2.norm()}; normdiff = {works_5.norm()}')
###check Apos by implementing it a different way in Apos_debug
#        Apos_debug = PETSc.Mat().createPython([work.getSizes(), work.getSizes()], comm=PETSc.COMM_WORLD)
#        Apos_debug.setPythonContext(Apos_debug_ctx(projVposs, Aposs, scatter_l2g, works, work))
#        Apos_debug.setUp()
#        work.setRandom()
#        test = work.duplicate()
#        test2 = work.duplicate()
#        Apos.mult(work,test)
#        Apos_debug.mult(work,test2)
#        testdiff = test-test2
#        print(f'norm of |.|_Apos = {np.sqrt(test.dot(work))} = |.|_Apos_debug = {np.sqrt(test2.dot(work))} ; norm of diff = {testdiff.norm()}')
###
###check that the projection in proj2 is a self.proj2.A orth projection
        #work.setRandom()
#        work.set(1.)
#        test = work.copy()
#        self.proj2.project(test)
#        test2 = test.copy()
#        self.proj2.project(test2)
#        testdiff = test-test2
#        print(f'norm(Pi x - Pi Pix) = {testdiff.norm()} = 0')
#        self.proj2.A.mult(test,test2)
#        test3 = work.duplicate()
#        self.proj2.A.mult(work,test3)
#        print(f'|Pi x|_A^2 - |x|_A^2 = {test.dot(test2)} - {work.dot(test3)} = {test.dot(test2) - work.dot(test3)} < 0 ')
#        #test2 = A Pi x ( = Pit A Pi x)
#        test3 = test2.copy()
#        self.proj2.project_transpose(test3)
#        test = test3.copy()
#        self.proj2.project_transpose(test)
#        testdiff = test3 - test2
#        print(f'norm(A Pi x - Pit A Pix) = {testdiff.norm()} = 0 = {(test - test3).norm()} = norm(Pit Pit A Pi x - Pit A Pix); compare to norm(A Pi x) = {test2.norm()} ')
#        #work.setRandom()
#        work.set(1.)
#        test2 = work.copy()
#        self.proj2.project_transpose(test2)
#        test2 = -1*test2
#        test2 += work
#
#        test = work.copy()
#        test = self.proj2.coarse_init(work)
#        test3 = work.duplicate()
#        self.proj2.A.mult(test,test3)
###check that the projection in proj3 is a self.proj3.A orth projection whose image includes Ker(Aneg)
#        #work.setRandom()
#        work.set(1.)
#        test = work.copy()
#        self.proj3.project(test)
#        test2 = test.copy()
#        self.proj3.project(test2)
#        testdiff = test-test2
#        print(f'norm(Pi x - Pi Pix) = {testdiff.norm()} = 0')
#        self.proj3.A.mult(test,test2)
#        test3 = work.duplicate()
#        self.proj3.A.mult(work,test3)
#        print(f'|Pi x|_A^2 - |x|_A^2 = {test.dot(test2)} - {work.dot(test3)} = {test.dot(test2) - work.dot(test3)} < 0 ')
#        #test2 = A Pi x ( = Pit A Pi x)
#        test3 = test2.copy()
#        self.proj3.project_transpose(test3)
#        test = test3.copy()
#        self.proj3.project_transpose(test)
#        testdiff = test3 - test2
#        print(f'norm(A Pi x - Pit A Pix) = {testdiff.norm()} = 0 = {(test - test3).norm()} = norm(Pit Pit A Pi x - Pit A Pix); compare to norm(A Pi x) = {test2.norm()} ')
#        #work.setRandom()
#        work.set(1.)
#        test2 = work.copy()
#        self.proj3.project_transpose(test2)
#        test2 = -1*test2
#        test2 += work
#
#        test = work.copy()
#        test = self.proj3.coarse_init(work)
#        test3 = work.duplicate()
#        self.proj3.A.mult(test,test3)
#
#        print(f'norm(A coarse_init(b)) = {test3.norm()} = {test2.norm()} = norm((I-Pit b)); norm diff = {(test2 - test3).norm()}')
#
#        work.set(1.)
#        test = work.copy()
#        test2 = work.copy()
#        self.proj3.project(test2)
#        test3 = work.copy()
#        self.proj3.project(test3)
#        test = work.copy()
#        self.Apos.mult(test2,test)
#        test2 = work.copy()
#        self.A.mult(test3,test2)
#        print(f'norm(Apos Pi3 x) = {test.norm()} = {test2.norm()} = norm(A Pi3 x); norm diff = {(test - test2).norm()}')
#        for vec in self.AposinvV0:
#            test = vec.copy()
#            self.proj3.project(test)
#            print(f'norm(Pi3 AposinvV0[i]) = {test.norm()} compare to norm of the non projected vector norm ={(vec).norm()}')
#
### END Debug DEBUG

    def mult(self, x, y):
        """
        Applies the domain decomposition preconditioner followed by the projection preconditioner to a vector.

        Parameters
        ==========

        x : petsc.Vec
            The vector to which the preconditioner is to be applied.

        y : petsc.Vec
            The vector that stores the result of the preconditioning operation.

        """
########################
########################
        xd = x.copy()
        if self.H3projCS == True:
            self.proj3.project_transpose(xd)

        self.H2.mult(xd,y)
        if self.H3projCS == True:
            self.proj3.project(y)

        if self.H3addCS == True:
            xd = x.copy()
            ytild = self.proj3.coarse_init(xd) # I could save a coarse solve by combining this line with project_transpose
            if ytild.dot(xd) <  0:
                print(f'x.dot(coarse_init(x)) = {ytild.dot(xd)} < 0 ')
            y += ytild

    def MP_mult(self, x, y):
        """
        Applies the domain decomposition multipreconditioner followed by the projection preconditioner to a vector.

        Parameters
        ==========

        x : petsc.Vec
            The vector to which the preconditioner is to be applied.

        y : FIX
            The list of ndom vectors that stores the result of the multipreconditioning operation (one vector per subdomain).

        """
        print('not implemented')

    def apply(self, pc, x, y):
        """
        Applies the domain decomposition preconditioner followed by the projection preconditioner to a vector.
        This is just a call to PCNew.mult with the function name and arguments that allow PCNew to be passed
        as a preconditioner to PETSc.ksp.

        Parameters
        ==========

        pc: This argument is not called within the function but it belongs to the standard way of calling a preconditioner.

        x : petsc.Vec
            The vector to which the preconditioner is to be applied.

        y : petsc.Vec
            The vector that stores the result of the preconditioning operation.

        """
        self.mult(x,y)

    def view(self):
        self.minV0.gathered_dim = mpi.COMM_WORLD.gather(self.minV0.nrb, root=0)
        self.gathered_nneg = mpi.COMM_WORLD.gather(self.nnegs, root=0)
        self.gathered_Dneg = mpi.COMM_WORLD.gather(self.Dnegs, root=0)
        if self.GenEO == True:
            self.GenEOV0.gathered_nsharp = mpi.COMM_WORLD.gather(self.GenEOV0.n_GenEO_eigmax, root=0)
            self.GenEOV0.gathered_nflat = mpi.COMM_WORLD.gather(self.GenEOV0.n_GenEO_eigmin, root=0)
            #self.GenEOV0.gathered_dimKerMs = mpi.COMM_WORLD.gather(self.GenEOV0.dimKerMs, root=0)
            self.GenEOV0.gathered_Lambdasharp = mpi.COMM_WORLD.gather(self.GenEOV0.Lambda_GenEO_eigmax, root=0)
            self.GenEOV0.gathered_Lambdaflat = mpi.COMM_WORLD.gather(self.GenEOV0.Lambda_GenEO_eigmin, root=0)
        if mpi.COMM_WORLD.rank == 0:
            print('#############################')
            print(f'view of PCNew')
            print(f'{self.switchtoASM=}')
            print(f'{self.verbose= }')
            print(f'{self.GenEO= }')
            print(f'{self.H3addCS= }')
            print(f'{self.H3projCS= }')
            print(f'{self.H2projCS= }')
            print(f'{self.viewPC= }')
            print(f'{self.viewV0= }')
            print(f'{self.viewGenEOV0= }')
            print(f'{self.viewnegV0= }')
            print(f'{self.viewminV0= }')
            print(f'{self.compute_ritz_apos=}') 
            print(f'### info about minV0.V0s = (Ker(Atildes)) ###')
            print(f'{self.minV0.mumpsCntl3=}') 
            print(f'###info about Vnegs = rank(Anegs) = coarse components for proj3')
            print(f'{self.gathered_nneg=}')
            print(f'{np.sum(self.gathered_nneg)=}')
            if (self.ksp_Atildes.pc.getFactorSolverType() == 'mumps'):
                print(f'dim(Ker(Atildes)) = {self.minV0.gathered_dim}')
            else:
                print(f'Ker(Atildes) not computed because pc is not mumps')
            if self.GenEO == True:
                print(f'### info about GenEOV0.V0s = (Ker(Atildes)) ###')
                print(f'{self.GenEOV0.tau_eigmax=}') 
                print(f'{self.GenEOV0.tau_eigmin=}') 
                print(f'{self.GenEOV0.eigmax=}') 
                print(f'{self.GenEOV0.eigmin=}') 
                print(f'{self.GenEOV0.nev=}') 
                print(f'{self.GenEOV0.maxev=}') 
                print(f'{self.GenEOV0.mumpsCntl3=}') 
                print(f'{self.GenEOV0.verbose=}') 
                print(f'{self.GenEOV0.mult_max=}') 
                print(f'{self.GenEOV0.gathered_nsharp=}')
                print(f'{self.GenEOV0.gathered_nflat=}') 
                #print(f'{self.GenEOV0.gathered_dimKerMs=}')
                #print(f'{np.array(self.GenEOV0.gathered_Lambdasharp)=}')
                #print(f'{np.array(self.GenEOV0.gathered_Lambdaflat)=}')
            print(f'### info about the preconditioner for Apos ###')
            print(f'{self.proj2.V0_is_global=}') 
            if(self.proj2.V0_is_global == False):
                print(f'{self.proj2.gathered_dimV0s=}') 
            if self.GenEO == True:
                print(f'global dim V0 for Apos = {self.proj2.dim} = ({np.sum(self.gathered_nneg)} from Vneg ) + ({np.sum(self.minV0.gathered_dim)} from Ker(Atildes)) + ({np.sum(self.GenEOV0.gathered_nsharp)} from GenEO_eigmax) + ({np.sum(self.GenEOV0.gathered_nflat) } from GenEO_eigmin)') 
            else:
                print(f'global dim V0 for Apos = {np.sum(self.proj2.gathered_dimV0s)} = ({np.sum(self.minV0.gathered_dim)} from Ker(Atildes))') 
            if self.compute_ritz_apos and self.ritz_eigs_apos != []:
                print(f'Estimated kappa(H2 Apos) = {self.ritz_eigs_apos.max()/self.ritz_eigs_apos.min() }; with lambdamin = {self.ritz_eigs_apos.min()} and   lambdamax = {self.ritz_eigs_apos.max()}')   

            print('#############################')

class Aneg_ctx(object):
    def __init__(self, Vnegs, DVnegs, scatter_l2g, works, works_2):
        self.scatter_l2g = scatter_l2g
        self.works = works
        self.works_2 = works_2
        self.Vnegs = Vnegs
        self.DVnegs = DVnegs
        self.gamma = PETSc.Vec().create(comm=PETSc.COMM_SELF)
        self.gamma.setType(PETSc.Vec.Type.SEQ)
        self.gamma.setSizes(len(self.Vnegs))
    def mult(self, mat, x, y):
        y.set(0)
        self.scatter_l2g(x, self.works, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        self.works_2.set(0)
        for i,vec in enumerate(self.Vnegs):
            self.works_2.axpy(self.works.dot(self.DVnegs[i]) , vec)
        self.scatter_l2g(self.works_2, y, PETSc.InsertMode.ADD_VALUES)

class Apos_debug_ctx(object):
    def __init__(self, projVposs, Aposs, scatter_l2g, works, work):
        self.scatter_l2g = scatter_l2g
        self.work = works
        self.works = works
        self.projVposs = projVposs
        self.Aposs = Aposs
    def mult(self, mat, x, y):
        y.set(0)
        works_2 = self.works.duplicate()
        self.scatter_l2g(x, self.works, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        self.Aposs.mult(self.works,works_2)
        self.scatter_l2g(works_2, y, PETSc.InsertMode.ADD_VALUES)

class Apos_ctx(object):
    def __init__(self,A_mpiaij, Aneg):
        self.A_mpiaij = A_mpiaij
        self.Aneg = Aneg
    def mult(self, mat, x, y):
        xtemp = x.duplicate()
        self.Aneg.mult(x,xtemp)
        self.A_mpiaij.mult(x,y)
        y += xtemp

class Anegs_ctx(object):
    def __init__(self, Vnegs, DVnegs):
        self.Vnegs = Vnegs
        self.DVnegs = DVnegs
    def mult(self, mat, x, y):
        y.set(0)
        for i,vec in enumerate(self.Vnegs):
            y.axpy(x.dot(self.DVnegs[i]), vec)

class RsAposRsts_ctx(object):
    def __init__(self,As,RsVnegs,RsDVnegs):
        self.As = As
        self.RsVnegs = RsVnegs
        self.RsDVnegs = RsDVnegs
    def mult(self, mat, x, y):
        self.As.mult(x,y)
        for i,vec in enumerate(self.RsVnegs):
            y.axpy(x.dot(self.RsDVnegs[i]), vec)

class invRsAposRsts_ctx(object):
    def __init__(self,As,RsVnegs,RsDnegs,works):
        self.As = As
        self.works = works
        self.RsVnegs = RsVnegs
        self.RsDnegs = RsDnegs

        self.ksp_As = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        self.ksp_As.setOptionsPrefix("ksp_As_")
        self.ksp_As.setOperators(self.As)
        self.ksp_As.setType('preonly')
        self.pc_As = self.ksp_As.getPC()
        self.pc_As.setType('cholesky')
        self.pc_As.setFactorSolverType('mumps')
        self.ksp_As.setFromOptions()
        self.AsinvRsVnegs = []
        for i,vec in enumerate(self.RsVnegs):
            self.ksp_As.solve(vec,self.works)
            self.AsinvRsVnegs.append(self.works.copy())
        self.Matwood = PETSc.Mat().create(comm=PETSc.COMM_SELF)
        self.Matwood.setType(PETSc.Mat.Type.SEQDENSE)
        self.Matwood.setSizes([len(self.AsinvRsVnegs),len(self.AsinvRsVnegs)])
        self.Matwood.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self.Matwood.setPreallocationDense(None)
        for i, vec in enumerate(self.AsinvRsVnegs):
            for j in range(i):
                tmp = self.RsVnegs[j].dot(vec)
                self.Matwood[i, j] = tmp
                self.Matwood[j, i] = tmp
            tmp = self.RsVnegs[i].dot(vec)
            self.Matwood[i, i] = tmp + 1/self.RsDnegs[i]
        self.Matwood.assemble()
        self.ksp_Matwood = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        self.ksp_Matwood.setOperators(self.Matwood)
        self.ksp_Matwood.setType('preonly')
        self.pc = self.ksp_Matwood.getPC()
        self.pc.setType('cholesky')
        self.gamma, _ = self.Matwood.getVecs()  
        self.alpha = self.gamma.duplicate()
        
    def mult(self, mat, x, y):
        self.ksp_As.solve(x,y)
        for i, vec in enumerate(self.AsinvRsVnegs):
            self.gamma[i] = vec.dot(x)
        self.ksp_Matwood.solve(self.gamma, self.alpha)
        for i, vec in enumerate(self.AsinvRsVnegs):
            y.axpy(-self.alpha[i], vec)

    def apply(self,pc, x, y):
        self.mult(pc,x,y)


class Aposs_ctx(object):
    def __init__(self,Bs, Anegs):
        self.Bs = Bs
        self.Anegs = Anegs
    def mult(self, mat, x, y):
        xtemp = x.duplicate()
        self.Anegs.mult(x,xtemp)
        self.Bs.mult(x,y)
        y += xtemp

class scaledmats_ctx(object):
    def __init__(self, mats, musl, musr):
        self.mats = mats
        self.musl = musl
        self.musr = musr
    def mult(self, mat, x, y):
        xtemp = x.copy()*self.musr
        self.mats.mult(xtemp,y)
        y *= self.musl
    def apply(self, mat, x, y):
        self.mult(mat, x, y)

class invAposs_ctx(object):
    def __init__(self,Bs_ksp,projVposs):
        self.Bs_ksp = Bs_ksp
        self.projVposs = projVposs
    def apply(self, mat, x, y):
        xtemp1 = y.duplicate()
        xtemp2 = y.duplicate()
        self.projVposs.mult(x,xtemp1)
        self.Bs_ksp.solve(xtemp1,xtemp2)
        self.projVposs.mult(xtemp2,y)
    def mult(self, mat, x, y):
        #xtemp1 = y.duplicate()
        #xtemp2 = y.duplicate()
        #self.projVnegs.mult(x,xtemp1)
        #self.Bs_ksp.solve(xtemp1,xtemp2)
        #self.projVnegs.mult(xtemp2,y)
        self.apply(mat, x, y)

class projVnegs_ctx(object):
    def __init__(self, Vnegs):
        self.Vnegs = Vnegs
    def mult(self, mat, x, y):
        y.set(0)
        for i,vec in enumerate(self.Vnegs):
            y.axpy(x.dot(vec) , vec)

class projVposs_ctx(object):
    def __init__(self, projVnegs):
        self.projVnegs = projVnegs
    def mult(self, mat, x, y):
        self.projVnegs(-x,y)
        y.axpy(1.,x)


class H2_ctx(object):
    def __init__(self, projCS, addCS, proj2, scatter_l2g, ksp_Atildes, works_1, works_2 ):
        self.projCS = projCS
        self.addCS = addCS
        self.proj2 = proj2
        self.scatter_l2g = scatter_l2g
        self.ksp_Atildes = ksp_Atildes
        self.works_1 = works_1
        self.works_2 = works_2

    def mult(self,mat,x,y):
        self.apply([],x,y)
    def apply(self,pc, x, y):
        xd = x.copy()
        if self.projCS == True:
            self.proj2.project_transpose(xd)

        self.scatter_l2g(xd, self.works_1, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        self.ksp_Atildes.solve(self.works_1, self.works_2)

        y.set(0.)
        self.scatter_l2g(self.works_2, y, PETSc.InsertMode.ADD_VALUES)
        if self.projCS == True:
            self.proj2.project(y)

        if self.addCS == True:
            xd = x.copy()
            ytild = self.proj2.coarse_init(xd) # I could save a coarse solve by combining this line with project_transpose
            #print(f'in H2 x.dot(coarse_init(x)) = {ytild.dot(xd)} > 0 ')
            if ytild.dot(xd) < 0:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            y += ytild

