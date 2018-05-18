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
        OptDB = PETSc.Options()                                
        self.kscaling = OptDB.getBool('PCBNN_kscaling', True) #kscaling if true, multiplicity scaling if false
        self.switchtoASM = OptDB.getBool('PCBNN_switchtoASM', False) #use Additive Schwarz as a preconditioner instead of BNN

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
                print('The user has chosen to switch to Additive Schwarz instead of BNN')
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
        
    def MP_mult(self, x, y):
        self.scatter_l2g(x, self.workl_1, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        self.localksp.solve(self.workl_1, self.workl_2)
        for i in range(mpi.COMM_WORLD.size):
            self.workl_1.set(0)
            if mpi.COMM_WORLD.rank == i:
                self.workl_1 = self.workl_2.copy()
            y[i].set(0.)
            self.scatter_l2g(self.workl_1, y[i], PETSc.InsertMode.ADD_VALUES)
            self.proj.project(y[i])

    def mult(self, x, y):
        self.scatter_l2g(x, self.workl_1, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        self.localksp.solve(self.workl_1, self.workl_2)

        y.set(0.)
        self.scatter_l2g(self.workl_2, y, PETSc.InsertMode.ADD_VALUES)
        self.proj.project(y)

    def apply(self, pc, x, y):
        self.mult(x,y)


