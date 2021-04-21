# Authors:
#     Loic Gouarin <loic.gouarin@cmap.polytechnique.fr>
#     Nicole Spillane <nicole.spillane@cmap.polytechnique.fr>
#
# License: BSD 3 clause
from .assembling import buildElasticityMatrix
from .bc import bcApplyWestMat, bcApplyWest_vec
from .cg import cg
from .projection import projection
from petsc4py import PETSc
from slepc4py import SLEPc
import mpi4py.MPI as mpi
import numpy as np

class PCBNN(object):
    def __init__(self, A):
        """
        Initialize the domain decomposition preconditioner, multipreconditioner and coarse space with its operators  

        Parameters
        ==========

        A : petsc.Mat
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
        self.addCS = OptDB.getBool('PCBNN_addCoarseSolve', False) 
        self.projCS = OptDB.getBool('PCBNN_CoarseProjection', True) 

        # convert matis to mpiaij, extract local matrices
        r, _ = A.getLGMap()
        is_A = PETSc.IS().createGeneral(r.indices)
        A_mpiaij = A.convertISToAIJ()
        A_mpiaij_local = A_mpiaij.createSubMatrices(is_A)[0]
        A_scaled = A.copy().getISLocalMat()
        vglobal, _ = A.getVecs()
        vlocal, _ = A_scaled.getVecs()
        scatter_l2g = PETSc.Scatter().create(vlocal, None, vglobal, is_A)

        #compute the multiplicity of each degree of freedom and max of the multiplicity
        vlocal.set(1.) 
        vglobal.set(0.) 
        scatter_l2g(vlocal, vglobal, PETSc.InsertMode.ADD_VALUES)
        scatter_l2g(vglobal, vlocal, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        NULL,mult_max = vglobal.max()

        # k-scaling or multiplicity scaling of the local (non-assembled) matrix
        if self.kscaling == False:
            A_scaled.diagonalScale(vlocal,vlocal)
        else:
            v1 = A_mpiaij_local.getDiagonal()
            v2 = A_scaled.getDiagonal()
            A_scaled.diagonalScale(v1/v2, v1/v2)

        # the default local solver is the scaled non assembled local matrix (as in BNN)
        if self.switchtoASM:
            Alocal = A_mpiaij_local
            if mpi.COMM_WORLD.rank == 0:
                print('The user has chosen to switch to Additive Schwarz instead of BNN.')
        else: #(default)
            Alocal = A_scaled
        localksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        localksp.setOptionsPrefix("localksp_") 
        localksp.setOperators(Alocal)
        localksp.setType('preonly')
        localpc = localksp.getPC()
        localpc.setType('cholesky')
        localpc.setFactorSolverType('mumps')
        localksp.setFromOptions()

        self.A = A
        self.A_scaled = A_scaled
        self.A_mpiaij_local = A_mpiaij_local
        self.localksp = localksp
        self.workl_1 = vlocal.copy()
        self.workl_2 = self.workl_1.copy()
        self.scatter_l2g = scatter_l2g 
        self.mult_max = mult_max

        self.proj = projection(self)
        
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
        self.scatter_l2g(x, self.workl_1, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        self.localksp.solve(self.workl_1, self.workl_2)

        y.set(0.)
        self.scatter_l2g(self.workl_2, y, PETSc.InsertMode.ADD_VALUES)
        if self.projCS == True: 
            self.proj.project(y)

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
        self.scatter_l2g(x, self.workl_1, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        self.localksp.solve(self.workl_1, self.workl_2)
        for i in range(mpi.COMM_WORLD.size):
            self.workl_1.set(0)
            if mpi.COMM_WORLD.rank == i:
                self.workl_1 = self.workl_2.copy()
            y[i].set(0.)
            self.scatter_l2g(self.workl_1, y[i], PETSc.InsertMode.ADD_VALUES)
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
        xd = x.copy()
        if self.projCS == True:  
            self.proj.project_transpose(xd)
        self.mult(xd,y)
        if self.addCS == True:  
            xd = x.copy()
            ytild = self.proj.coarse_init(xd)
            #tempxd = xd.dot(xd)
            #print(f'tempxd : {tempxd}')
            #tempytild = ytild.dot(ytild)
            #print(f'tempytild : {tempytild}')
            y += ytild


