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


class KSP_MPCG(MyKSP):

    def __init__(self, P, tol=.1):
        self.P = P
        self.tol = tol

    def add_vectors(self):
        return [self.work[0].duplicate() for i in range(self.ndom)]

    def setUp(self, ksp):
        super(KSP_MPCG, self).setUp(ksp)
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
        A, B = ksp.getOperators()
        r, z, p, Ap = self.work
        #

        # x.setRandom()
        # self.P.proj.apply(x)
        # bcApplyWest_vec(self.P.da_global, x)
        # xnorm = b.dot(x)/x.dot(A*x)
        # x *= xnorm

        A.mult(x, r)
        r.aypx(-1, b)
        self.P.proj.apply_transpose(r)

        rnorm = r.dot(r)
        its = 0
        if mpi.COMM_WORLD.rank == 0:
            print(f'ite: {its} residual -> {rnorm}')

        self.P.mult(r, p[-1])

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

            self.P.mult_z(r, z)
            rnorm = r.dot(z)
            ti /= rnorm

            its = ksp.getIterationNumber()

            if ti < self.tol:
                if mpi.COMM_WORLD.rank == 0:
                    print('multipreconditioning this iteration')
                p.append(self.add_vectors())
                Ap.append(self.add_vectors())
                self.P.mult(r, p[-1])
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
                    self.P.proj.apply(p[-1][i])
            else:
                self.P.proj.apply(p[-1])

            if mpi.COMM_WORLD.rank == 0:
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
