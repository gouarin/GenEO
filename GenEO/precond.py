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

class PCBNN(object): #Neumann-Neumann and Additive Scharz with no overlap
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
            Default is False.
            If True, some information about the preconditioners is printed when the code is executed.

        PCBNN_GenEO : Bool
        Default is False. 
        If True then the coarse space is enriched by solving local generalized eigenvalue problems. 

        PCBNN_CoarseProjection :True 
            Default is True
            If False then there is no coarse projection: Two level Additive Schwarz or One-level preconditioner depending on PCBNN_addCoarseSolve
            If True, the coarse projection is applied: Projected preconditioner of hybrid preconditioner depending on PCBNN_addCoarseSolve 

        PCBNN_addCoarseSolve : False
            Default is True 
            If True then (R0t A0\R0 r) is added to the preconditioned residual  
            False corresponds to the projected preconditioner (need to choose initial guess accordingly) (or the one level preconditioner if PCBNN_CoarseProjection = False)
            True corresponds to the hybrid preconditioner (or the fully additive preconditioner if PCBNN_CoarseProjection = False)
        """
        OptDB = PETSc.Options()
        self.switchtoASM = OptDB.getBool('PCBNN_switchtoASM', False) #use Additive Schwarz as a preconditioner instead of BNN
        self.kscaling = OptDB.getBool('PCBNN_kscaling', True) #kscaling if true, multiplicity scaling if false
        self.verbose = OptDB.getBool('PCBNN_verbose', False) 
        self.GenEO = OptDB.getBool('PCBNN_GenEO', True)
        self.addCS = OptDB.getBool('PCBNN_addCoarseSolve', False) 
        self.projCS = OptDB.getBool('PCBNN_CoarseProjection', True) 

        #extract Neumann matrix from A in IS format
        Ms = A_IS.copy().getISLocalMat() 
      
        # convert A_IS from matis to mpiaij
        A_mpiaij = A_IS.convertISToAIJ()
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

        self.A = A_mpiaij
        self.Ms = Ms
        self.As = As
        self.ksp_Atildes = ksp_Atildes
        self.works_1 = vlocal.copy()
        self.works_2 = self.works_1.copy()
        self.scatter_l2g = scatter_l2g 
        self.mult_max = mult_max

        self.minV0 = minimal_V0(self.ksp_Atildes)
        if self.GenEO == True:
          GenEOV0 = GenEO_V0(self.ksp_Atildes,self.Ms,self.As,self.mult_max,self.minV0.V0s) 
          self.V0s = GenEOV0.V0s
        else:
          self.V0s = self.minV0.V0s
        self.proj = coarse_operators(self.V0s,self.A,self.scatter_l2g,vlocal)
        
        #self.proj = projection(self)
        
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

    def apply(self, pc, x, y):
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


