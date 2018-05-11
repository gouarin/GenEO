from petsc4py import PETSc
import mpi4py.MPI as mpi
import numpy as np
from .bc import bcApplyWest_vec
from slepc4py import SLEPc

class newprojection:
    def __init__(self,ASM,GenEO,tauGenEO_lambdamin,tauGenEO_lambdamax):
        vglobal, _ = ASM.A.getVecs()
        vlocal, _ = ASM.A_scaled.getVecs()

        self.scatter_l2g = ASM.scatter_l2g 
        self.A = ASM.A
        self.A_scaled = ASM.A_scaled
        self.A_mpiaij_local = ASM.A_mpiaij_local

        workl, _ = self.A_scaled.getVecs()

        ksp = ASM.localksp
        ksp.pc.setFactorSetUpSolverType()
        Alocal,_ = ksp.getOperators() 

        if ksp.pc.getFactorSolverType() == 'mumps':
            F = ksp.pc.getFactorMatrix()
            F.setMumpsIcntl(7, 2)
            F.setMumpsIcntl(24, 1)
            F.setMumpsCntl(3, 1e-6)
            ksp.pc.setUp()
            nrb = F.getMumpsInfog(28)

            rbm_vecs = []
            for i in range(nrb):
                F.setMumpsIcntl(25, i+1)
                rbm_vecs.append(workl.duplicate())
                rbm_vecs[i].set(0.)
                ksp.solve(rbm_vecs[i], rbm_vecs[i])
            
            F.setMumpsIcntl(25, 0)
        else:
            rbm_vecs = []
            nrb = 0

        print(f"I am subdomain number {mpi.COMM_WORLD.rank} and the dimension of the kernel of my local solver is  {nrb}")
        # Add the GenEO coarse vectors
        if GenEO and tauGenEO_lambdamax > 0:
            #Eigenvalue Problem for smallest eigenvalues
            eps = SLEPc.EPS().create(comm=PETSc.COMM_SELF)
            eps.setDimensions(nev=10)

            eps.setProblemType(SLEPc.EPS.ProblemType.GHIEP)
            eps.setOperators(Alocal , self.A_mpiaij_local )
            #eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
            eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
            eps.setTarget(0.)
            if nrb > 0 :
                eps.setDeflationSpace(rbm_vecs)
            ST = eps.getST()
            ST.setType("sinvert") 
#TODO: use the factorization already computed by mumps. The following line works fine here but some parameter in the ksp must be changed because the next call to ksp.solve in precond.mult produces an error
#            ST.setKSP(ksp)
#TODO turn off eps.setpurify
            eps.setFromOptions()
#            eps.view()
            eps.solve()
#            print(mpi.COMM_WORLD.rank, eps.getConverged())
            for i in range(eps.getConverged()):
                if(abs(eps.getEigenvalue(i))<tauGenEO_lambdamax): #TODO tell slepc that the eigenvalues are real
                   rbm_vecs.append(workl.duplicate())
                   eps.getEigenvector(i,rbm_vecs[-1])
#                print(mpi.COMM_WORLD.rank, eps.getEigenvalue(i))
        print(f"I am subdomain number {mpi.COMM_WORLD.rank} and after first GenEO, I have contributed {len(rbm_vecs)} vectors to the coarse space")
        if GenEO and tauGenEO_lambdamin > 0:
            #to compute the smallest eigenvalues of the preconditioned matrix, A_scaled must be factorized

            #Eigenvalue Problem for smallest eigenvalues
            eps = SLEPc.EPS().create(comm=PETSc.COMM_SELF)
            eps.setDimensions(nev=10)

            eps.setProblemType(SLEPc.EPS.ProblemType.GHIEP)
            eps.setOperators(self.A_scaled,Alocal)
            eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
            eps.setTarget(0.)
            ST = eps.getST()
            ST.setType("sinvert") 
            STksp = ST.getKSP()
            STksp.setOperators(self.A_scaled)
            STksp.setType('preonly')
            STpc = STksp.getPC()
            STpc.setType('cholesky')
            STpc.setFactorSolverType('mumps')
            STpc.setFactorSetUpSolverType()
            ST_F = STpc.getFactorMatrix()
            ST_F.setMumpsIcntl(7, 2)
            ST_F.setMumpsIcntl(24, 1)
            ST_F.setMumpsCntl(3, 1e-6)
            STpc.setUp()
            STnrb = ST_F.getMumpsInfog(28)

            for i in range(STnrb):
                ST_F.setMumpsIcntl(25, i+1)
                rbm_vecs.append(workl.duplicate())
                rbm_vecs[-1].set(0.)
                STksp.solve(rbm_vecs[-1], rbm_vecs[-1])
            
            ST_F.setMumpsIcntl(25, 0)
            print(f"I am subdomain number {mpi.COMM_WORLD.rank} and STnrb = {STnrb}")


            if len(rbm_vecs) > 0 :
                eps.setDeflationSpace(rbm_vecs)
            eps.solve()
#            print(mpi.COMM_WORLD.rank, eps.getConverged())
            for i in range(eps.getConverged()):
                if(abs(eps.getEigenvalue(i))<tauGenEO_lambdamin): #TODO tell slepc that the eigenvalues are real
                   rbm_vecs.append(workl.duplicate())
                   eps.getEigenvector(i,rbm_vecs[-1])
#                print(mpi.COMM_WORLD.rank, eps.getEigenvalue(i))

        ncoarse = len(rbm_vecs)
        print(f"I am subdomain number {mpi.COMM_WORLD.rank} and I contribute {ncoarse} vectors to the coarse space")

        coarse_vecs= []
        for i in range(mpi.COMM_WORLD.size):
            nrbl = ncoarse if i == mpi.COMM_WORLD.rank else None
            nrbl = mpi.COMM_WORLD.bcast(nrbl, root=i)
            
            for irbm in range(nrbl):
                coarse_vecs.append(rbm_vecs[irbm] if i == mpi.COMM_WORLD.rank else None)

        #n = len(coarse_vecs)

        #scale 
        coarse_Avecs = []
        work, _ = self.A.getVecs()
        for vec in coarse_vecs:
            if vec:
                workl = vec.copy()
            else:
                workl.set(0.)

            work.set(0)
            self.scatter_l2g(workl, work, PETSc.InsertMode.ADD_VALUES)
            coarse_Avecs.append(self.A*work)

            self.scatter_l2g(coarse_Avecs[-1], workl, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
            if vec:
                vec.scale(1./np.sqrt(vec.dot(workl)))
                workl = vec.copy()
            else:
                workl.set(0)

            work.set(0)
            self.scatter_l2g(workl, work, PETSc.InsertMode.ADD_VALUES)
            coarse_Avecs[-1] = self.A*work

        self.Delta = PETSc.Mat().create(comm=PETSc.COMM_SELF)
        self.Delta.setType(PETSc.Mat.Type.SEQDENSE)
        self.Delta.setSizes([len(coarse_vecs),len(coarse_vecs)])
        self.Delta.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self.Delta.setPreallocationDense(None)

        #fill coarse problem matrix
        for i, vec in enumerate(coarse_vecs):
            if vec:
                workl = vec.copy()
            else:
                workl.set(0)

            work.set(0)
            self.scatter_l2g(workl, work, PETSc.InsertMode.ADD_VALUES)
            for j in range(i+1):
                tmp = coarse_Avecs[j].dot(work)
                self.Delta[i, j] = tmp
                self.Delta[j, i] = tmp

        self.Delta.assemble()
#        if mpi.COMM_WORLD.rank == 0:
#            self.Delta.view()
        self.coarse_vecs = coarse_vecs
        self.coarse_Avecs = coarse_Avecs
        
        self.ksp_Delta = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        self.ksp_Delta.setOperators(self.Delta)
        self.ksp_Delta.setType('preonly')
        pc = self.ksp_Delta.getPC()
        pc.setType('cholesky')

        #self.work = self.da.createGlobalVec()
        self.workl = workl
        self.gamma = PETSc.Vec().create(comm=PETSc.COMM_SELF)
        self.gamma.setType(PETSc.Vec.Type.SEQ)
        self.gamma.setSizes(len(coarse_vecs))
        self.gamma_tmp = self.gamma.duplicate()

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

    def apply(self,pc,x, y):
        y = self.xcoarse(x)
        #self.project(y)

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
class projection:
    def __init__(self,A,is_A,A_scaled,A_mpiaij_local):
        vglobal, _ = A.getVecs()
        vlocal, _ = A_scaled.getVecs()

        self.scatter_l2g = PETSc.Scatter().create(vlocal, None, vglobal, is_A)
        self.A = A
        self.A_scaled = A_scaled
        self.A_mpiaij_local = A_mpiaij_local

    def constructCoarse(self, ksp):
        # coarse_vecs is a list of local vectors
        # coarse_Avecs is a list of global vectors

        workl, _ = self.A_scaled.getVecs()

        ksp.pc.setFactorSetUpSolverType()

        if ksp.pc.getFactorSolverType() == 'mumps':
            F = ksp.pc.getFactorMatrix()
            F.setMumpsIcntl(7, 2)
            F.setMumpsIcntl(24, 1)
            F.setMumpsCntl(3, 1e-6)
            ksp.pc.setUp()
            nrb = F.getMumpsInfog(28)

            rbm_vecs = []
            for i in range(nrb):
                F.setMumpsIcntl(25, i+1)
                rbm_vecs.append(workl.duplicate())
                rbm_vecs[i].set(0.)
                ksp.solve(rbm_vecs[i], rbm_vecs[i])
            
            F.setMumpsIcntl(25, 0)
        else:
            rbm_vecs = []
            nrb = 0

        # Add the GenEO coarse vectors
        GenEO = 1
        tauGenEO = 0.1
        if GenEO:
            #HERE
            eps = SLEPc.EPS().create(comm=PETSc.COMM_SELF)
            eps.setDimensions(nev=10)

            eps.setProblemType(SLEPc.EPS.ProblemType.GHIEP)
            eps.setOperators(self.A_scaled , self.A_mpiaij_local )
            #eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
            eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
            eps.setTarget(0.)
            if nrb > 0 :
                eps.setDeflationSpace(rbm_vecs)
            ST = eps.getST()
            ST.setType("sinvert") 
#TODO: use the factorization already computed by mumps. The following line works fine here but some parameter in the ksp must be changed because the next call to ksp.solve in precond.mult produces an error
#            ST.setKSP(ksp)
#TODO turn off eps.setpurify
            eps.setFromOptions()
#            eps.view()
            eps.solve()
#            print(mpi.COMM_WORLD.rank, eps.getConverged())
            for i in range(eps.getConverged()):
                if(abs(eps.getEigenvalue(i))<tauGenEO): #TODO tell slepc that the eigenvalues are real
                   rbm_vecs.append(workl.duplicate())
                   eps.getEigenvector(i,rbm_vecs[-1])
#                print(mpi.COMM_WORLD.rank, eps.getEigenvalue(i))

        ncoarse = len(rbm_vecs)
        print(f"I am subdomain number {mpi.COMM_WORLD.rank} and I contribute {ncoarse} vectors to the coarse space")

        coarse_vecs= []
        for i in range(mpi.COMM_WORLD.size):
            nrbl = ncoarse if i == mpi.COMM_WORLD.rank else None
            nrbl = mpi.COMM_WORLD.bcast(nrbl, root=i)
            
            for irbm in range(nrbl):
                coarse_vecs.append(rbm_vecs[irbm] if i == mpi.COMM_WORLD.rank else None)

        #n = len(coarse_vecs)

        #scale 
        coarse_Avecs = []
        work, _ = self.A.getVecs()
        for vec in coarse_vecs:
            if vec:
                workl = vec.copy()
            else:
                workl.set(0.)

            work.set(0)
            self.scatter_l2g(workl, work, PETSc.InsertMode.ADD_VALUES)
            coarse_Avecs.append(self.A*work)

            self.scatter_l2g(coarse_Avecs[-1], workl, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
            if vec:
                vec.scale(1./np.sqrt(vec.dot(workl)))
                workl = vec.copy()
            else:
                workl.set(0)

            work.set(0)
            self.scatter_l2g(workl, work, PETSc.InsertMode.ADD_VALUES)
            coarse_Avecs[-1] = self.A*work

        self.Delta = PETSc.Mat().create(comm=PETSc.COMM_SELF)
        self.Delta.setType(PETSc.Mat.Type.SEQDENSE)
        self.Delta.setSizes([len(coarse_vecs),len(coarse_vecs)])
        self.Delta.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self.Delta.setPreallocationDense(None)

        #fill coarse problem matrix
        for i, vec in enumerate(coarse_vecs):
            if vec:
                workl = vec.copy()
            else:
                workl.set(0)

            work.set(0)
            self.scatter_l2g(workl, work, PETSc.InsertMode.ADD_VALUES)
            for j in range(i+1):
                tmp = coarse_Avecs[j].dot(work)
                self.Delta[i, j] = tmp
                self.Delta[j, i] = tmp

        self.Delta.assemble()
#        if mpi.COMM_WORLD.rank == 0:
#            self.Delta.view()
        self.coarse_vecs = coarse_vecs
        self.coarse_Avecs = coarse_Avecs
        
        self.ksp_Delta = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        self.ksp_Delta.setOperators(self.Delta)
        self.ksp_Delta.setType('preonly')
        pc = self.ksp_Delta.getPC()
        pc.setType('cholesky')

        #self.work = self.da.createGlobalVec()
        self.workl = workl
        self.gamma = PETSc.Vec().create(comm=PETSc.COMM_SELF)
        self.gamma.setType(PETSc.Vec.Type.SEQ)
        self.gamma.setSizes(len(coarse_vecs))
        self.gamma_tmp = self.gamma.duplicate()

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

    def apply(self,pc,x, y):
        y = self.xcoarse(x)
        #self.project(y)

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
