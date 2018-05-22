from petsc4py import PETSc
import mpi4py.MPI as mpi
from math import sqrt
from sys import getrefcount

from .bc import bcApplyWest_vec

class MyKSP(object):

    def __init__(self):
        pass

    def create(self, ksp):
        self.work = []

    def destroy(self, ksp):
        for v in self.work:
            v.destroy()

    def setUp(self, ksp):
        self.work[:] = ksp.getWorkVecs(right=2, left=None)

    def reset(self, ksp):
        for v in self.work:
            v.destroy()
        del self.work[:]

    def loop(self, ksp, r):
        its = ksp.getIterationNumber()
        rnorm = r.norm()
        ksp.setResidualNorm(rnorm)
        ksp.logConvergenceHistory(rnorm)

        # FIX define a Monitor
        ksp.monitor(its, rnorm)
        reason = ksp.callConvergenceTest(its, rnorm)
        if not reason:
            ksp.setIterationNumber(its+1)
        else:
            ksp.setConvergedReason(reason)
        return reason

class KSP_AMPCG(MyKSP):
    def __init__(self, mpc):
        """
        Initialize the AMPCG (Adaptive Multipreconditioned Conjugate Gradient) solver. 

        Parameters
        ==========
        self: 
            
        mpc : PCBNN Object FIX ?
            the multipreconditioner.  

        PETSc.Options
        =======

        AMPCG_fullMP : Bool
            Default is False
            If True then the algorithm performs only multipreconditioned iterations.

        AMPCG_tau : Real
            Default is 0.1
            This is the threshold to choose automatically between a multipreconditioned iteration and a preconditioned iteration. This automatic choice based on AMPCG_tau is only relevant to `catch' large eigenvalues of the preconditioned operator (for theoretical reasons).
            If AMPCG_tau = 0, the algorithm will always perform a preconditioned iteration, except possibly at initialization if the user changes AMPCG_MPinitit. 
            If AMPCG_fullMP = True then the value of AMPCG_tau is discarded.

        AMPCG_MPinitit : Bool
            Default is (self.tau > 0)
            If True then the initial iteration is multipreconditioned, otherwise it is preconditioned. By default, it is multipreconditioned unless AMPCG_tau = 0 in which case we know that all subsequent iterations will be preconditioned.
            If AMPCG_fullMP = True then the value of AMPCG_MPinitit is discarded.

        AMPCG_verbose : Bool
            Default is False. 
            If True, some information about the iterations is printed when the code is executed.
    
        """

        OptDB = PETSc.Options()
        self.tau = OptDB.getReal('AMPCG_tau', 0.1) 
        self.MPinitit = OptDB.getBool('AMPCG_MPinitit', (self.tau>0)) 
        self.fullMP = OptDB.getBool('AMPCG_fullMP', False) 
        self.verbose = OptDB.getBool('AMPCG_verbose', False) 

        self.mpc = mpc  

    def add_vectors(self):
        """
        Initialize a list of self.ndom global vectors that can be used to store the multipreconditioned residual.  

        Parameters
        ==========
        self: 
            
        Returns 
        =======

            : list of self.ndom PETSc.Vecs
        """
        return [self.work[0].duplicate() for i in range(self.ndom)]

    def setUp(self, ksp):
        """
        Setup of the AMPCG Krylov Subspace Solver.  

        Parameters
        ==========
        self : FIX 

        ksp : FIX
            
        """
        super(KSP_AMPCG, self).setUp(ksp)
        self.ndom = mpi.COMM_WORLD.size
        # FIX: transform p and Ap using block matrices
        p = [self.add_vectors()]
        Ap = [self.add_vectors()]
        self.work += [p, Ap]

        self.Delta = PETSc.Mat().create(comm=PETSc.COMM_SELF)
        self.Delta.setType(PETSc.Mat.Type.SEQDENSE)
        self.Delta.setSizes([self.ndom, self.ndom])
        self.Delta.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self.Delta.setPreallocationDense(None)

        self.gamma = PETSc.Vec().create(comm=PETSc.COMM_SELF)
        self.gamma.setType(PETSc.Vec.Type.SEQ)
        self.gamma.setSizes(self.ndom)

        self.ksp_Delta = []

    def solve(self, ksp, b, x):
        """
        Solve of the AMPCG Krylov Subspace Solver.  

        Parameters
        ==========
        self : FIX 

        ksp : FIX
            
        b : PETSc Vec
        The right hand side for which to solve. 

        x : PETSc Vec
        To store the solution. 

        """
        A, B = ksp.getOperators()
        r, z, p, Ap = self.work

        A.mult(x, r)
        r.aypx(-1, b)

        rnorm = r.dot(r)
        its = 0
        if mpi.COMM_WORLD.rank == 0 and self.verbose :
            print(f'ite: {its} residual -> {rnorm}')

        if self.MPinitit or self.fullMP :
            if mpi.COMM_WORLD.rank == 0 and self.verbose :
                print('multipreconditioning initial iteration')
            self.mpc.MP_mult(r, p[-1])
        else:
            if mpi.COMM_WORLD.rank == 0 and self.verbose :
                print('not multipreconditioning initial iteration')
            self.mpc.mult(r, z)
            p[-1] = z.copy()

        alpha = self.gamma.duplicate()
        beta = self.gamma.duplicate()
        phi = self.gamma.duplicate()
        
        while not self.loop(ksp, r):
            if isinstance(p[-1], list):
                for i in range(self.ndom):
                    self.gamma[i] = p[-1][i].dot(r)
                    Ap[-1][i] = A*p[-1][i]
                    for j in range(i + 1):
                        tmp = Ap[-1][i].dot(p[-1][j])
                        self.Delta[i, j] = tmp
                        self.Delta[j, i] = tmp

                self.Delta.assemble()

                self.ksp_Delta.append(PETSc.KSP().create(comm=PETSc.COMM_SELF))
                self.ksp_Delta[-1].setOperators(self.Delta.copy())
                self.ksp_Delta[-1].setType(ksp.Type.PREONLY)
                pc = self.ksp_Delta[-1].getPC()
                pc.setType(pc.Type.CHOLESKY)
                self.ksp_Delta[-1].solve(self.gamma, alpha)

                for i in range(self.ndom):
                    x.axpy(alpha[i], p[-1][i])
                    r.axpy(-alpha[i], Ap[-1][i])

                ti = self.gamma.dot(alpha)
            else:
                gamma0 = p[-1].dot(r)
                Ap[-1] = A*p[-1]
                delta = Ap[-1].dot(p[-1])

                self.ksp_Delta.append(delta)
                alpha0 = gamma0/delta

                x.axpy(alpha0, p[-1])
                r.axpy(-alpha0, Ap[-1])

                ti = gamma0*alpha0

            self.mpc.mult(r, z)
            rnorm = r.dot(z)
            ti /= rnorm

            its = ksp.getIterationNumber()

            if ti < self.tau or self.fullMP:
                if mpi.COMM_WORLD.rank == 0 and self.verbose :
                    print('multipreconditioning this iteration')
                p.append(self.add_vectors())
                Ap.append(self.add_vectors())
                self.mpc.MP_mult(r, p[-1])
            else:
                p.append(z.copy())
                Ap.append(z.duplicate())
                
            for it in range(its):
                if isinstance(p[-1], list):
                    for i in range(self.ndom):
                        if isinstance(p[it], list):
                            for j in range(self.ndom):
                                phi[j] = Ap[it][j].dot(p[-1][i])
                            self.ksp_Delta[it].solve(phi, beta)
                            for j in range(self.ndom):
                                p[-1][i].axpy(-beta[j], p[it][j])
                        else:
                            phi0 = Ap[it].dot(p[-1][i])
                            beta0 = phi0/self.ksp_Delta[it]
                            p[-1][i].axpy(-beta0, p[it])
                else:
                    if isinstance(p[it], list):
                        for j in range(self.ndom):
                            phi[j] = Ap[it][j].dot(p[-1])
                        self.ksp_Delta[it].solve(phi, beta)
                        for j in range(self.ndom):
                            p[-1].axpy(-beta[j], p[it][j])
                    else:
                        phi0 = Ap[it].dot(p[-1])
                        beta0 = phi0/self.ksp_Delta[it]
                        p[-1].axpy(-beta0, p[it])

            if isinstance(p[-1], list):
                for i in range(self.ndom):
                    self.mpc.proj.project(p[-1][i])
            else:
                self.mpc.proj.project(p[-1])

            if mpi.COMM_WORLD.rank == 0 and self.verbose :
                print(f'ite: {its} residual -> {rnorm} ti -> {ti}')

def cg(A, b, rtol=1e-5, ite_max=5000):
    x = b.duplicate()
    x.set(0.)

    r = b.copy()
    p = r.copy()
    Ap = p.duplicate()

    rdr = r.dot(r)
    rnorm = sqrt(rdr)
    if rnorm == 0:
        return x

    r0 = rnorm
    ite = 0

    # if mpi.COMM_WORLD.rank == 0:
    #     print(f'ite: {ite} residual -> {rnorm}')

    while rnorm/r0 > rtol and ite < ite_max:
        Ap = A*p
        alpha = rdr/p.dot(Ap)

        x += alpha*p
        r -= alpha*Ap

        beta = 1/rdr
        rdr = r.dot(r)
        beta *= rdr

        p = r + beta*p

        rnorm = sqrt(rdr)
        ite += 1
        if mpi.COMM_WORLD.rank == 0:
            print(f'ite: {ite} residual -> {rnorm}')

    return x

class KSP_PCG(MyKSP):

    def setUp(self, ksp):
        super(KSP_PCG, self).setUp(ksp)
        p = self.work[0].duplicate()
        Ap = p.duplicate()
        self.work += [p, Ap]

    def solve(self, ksp, b, x):
        A, B = ksp.getOperators()
        P = ksp.getPC()
        r, z, p, Ap = self.work
        #
        A.mult(x, r)
        r.aypx(-1, b)
        
        P.apply(r, z)

        z.copy(p)
        delta_0 = r.dot(z)
        delta = delta_0

        ite = 0
        if mpi.COMM_WORLD.rank == 0:
            print(f'ite: {ite} residual -> {delta}')
        
        while not self.loop(ksp, r):
            A.mult(p, Ap)
            alpha = delta / p.dot(Ap)
            x.axpy(+alpha, p)
            r.axpy(-alpha, Ap)

            P.apply(r, z)

            delta_old = delta
            delta = r.dot(z)
            beta = delta / delta_old
            p.aypx(beta, z)

            ite += 1
            if mpi.COMM_WORLD.rank == 0:
                print(f'ite: {ite} residual -> {delta}')


