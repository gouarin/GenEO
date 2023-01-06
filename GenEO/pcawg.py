
# Authors:
#     Loic Gouarin <loic.gouarin@cmap.polytechnique.fr>
#     Nicole Spillane <nicole.spillane@cmap.polytechnique.fr>
#
# License: BSD 3 clause
from petsc4py import PETSc
from slepc4py import SLEPc
import mpi4py.MPI as mpi
import numpy as np
import os

from .projection import projection, GenEO_V0, minimal_V0, coarse_operators

class PCAWG:
    def __init__(self, A):
        print('PCAWG !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
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
        self.test_case = OptDB.getString('test_case', 'default')

        self.H2addCS = True #OptDB.getBool('PCNew_H2addCoarseSolve', True) (it is currently not an option to use a projected preconditioner for H2)

        ksp0 = PETSc.KSP().create()
        ksp0.setOperators(A)
        pc0 = ksp0.pc
        pc0.setType('asm')
        pc0.setASMType(PETSc.PC.ASMType.BASIC)   #0: none (block Jacobi); 1:restrict (non sym); 2: interpolate (non sym); 3: basic
        pc0.setASMOverlap(0)
        pc0.setFromOptions()
        pc0.setUp()
        isb0, isl = pc0.getASMLocalSubdomains()


        ksp1 = PETSc.KSP().create()
        ksp1.setOperators(A)
        pc1 = ksp1.pc
        pc1.setType('asm')
        pc1.setASMType(PETSc.PC.ASMType.BASIC)   #0: none (block Jacobi); 1:restrict (non sym); 2: interpolate (non sym); 3: basic
        pc1.setASMOverlap(1)
        pc1.setFromOptions()
        pc1.setUp()
        isb1, isl = pc1.getASMLocalSubdomains()
        localksp1 = pc1.getASMSubKSP()[0]
        As1 = localksp1.getOperators()[0]

        work, _ = A.getVecs()
        works, _ = As1.getVecs()
        scatter_l2g1 = PETSc.Scatter().create(works, None, work, isb1[0])

        work.array[:] = mpi.COMM_WORLD.rank
        scatter_l2g1(work, works, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)

        lgmap = PETSc.LGMap().create(isb1[0])

        newisbarray = isb1[0].array[mpi.COMM_WORLD.rank <= works.array[:]] 

        isb = [PETSc.IS().createGeneral(newisbarray)]
        #add ddls to overlap only if local rank < rank of owner of global ddl

        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        ksp.setOptionsPrefix("H_")
        pc = ksp.pc
        pc.setType('asm')
        pc.setASMType(PETSc.PC.ASMType.BASIC)   #0: none (block Jacobi); 1:restrict (non sym); 2: interpolate (non sym); 3: basic
        pc.setASMLocalSubdomains(1,isb)
        pc.setFromOptions()
        pc.setUp()



        localksp = pc.getASMSubKSP()[0]

        As = localksp.getOperators()[0]
        global_sizes = A.getSizes() #returns ((m, M), (n, N)) (ie local and global info) 
        local_sizes = As.getSizes()

        works_1, works_2 = As.getVecs()
        work_1, work_2 = A.getVecs()
        scatter_l2g = PETSc.Scatter().create(works_1, None, work_1, isb[0])

##begin test
#        Bs = As.copy()
#        works_3 = works_1.duplicate()
#        works_3.set(0.)
#        for i in range(global_sizes[0][1]):
#            work_1.set(0.)
#            work_1[i] = 1.
#            work_1.assemble()
#            scatter_l2g(work_1, works_1, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
##            A.mult(work_1, work_2) #work_2 = column of A correponding to global dof j
#            As.mult(works_1, works_2) #works_2 = column of As correponding to global dof j
#
##            work_2.abs()
##            sii_glob = work_2.sum() #the 1 norm of work2 
#            works_2.abs()
#            sii_loc = works_2.sum() #the 1 norm of works_2 
#            work_2.set(0.)
#            scatter_l2g(works_2, work_2, PETSc.InsertMode.ADD_VALUES)
#            denom = work_2.sum() #the 1 norm of work_2 (all coefs are non negative) 
#
#            col = np.asarray(np.where(works_1[:] == 1)[0], dtype='int32')
#            if col.size > 0:
#                works_3[col[0]] = sii_loc/denom 
#                #mask_B = As[:, col[0]] != 0
#                #mask_works = works_1[:] != 0
#                #mask = np.logical_and(mask_B, mask_works)
#                #indices = np.asarray(range(local_sizes[0][0]), dtype='int32')
#                #Bs.setValues(indices[mask], col[0], As[:, col[0]][mask]**2/works_1[:][mask], PETSc.InsertMode.INSERT_VALUES)
#                #diff = sii_glob - sii_loc
#                #Bs.setValues(col[0], col[0], As[col[0], col[0]] - (sii_glob - sii_loc), PETSc.InsertMode.INSERT_VALUES)
#        works_3.assemble()
#        work_2.set(0.)
#        scatter_l2g(works_3, work_2, PETSc.InsertMode.ADD_VALUES)
#        print(f'sum of all works_3.min()= {work_2.min()} and sumof_works_3.max()= {work_2.max()}') 
#        #print('begin works_3')
#        print(f'works_3.min()= {works_3.min()} and works_3.max()= {works_3.max()}') 
#        #print('end works_3')
#        Bs.diagonalScale(L=works_3)
#        Bs = 0.5*(Bs + Bs.transpose()) 
##end test

#Bs is multiplicity based (as in the AWG article)
        Bs = As.duplicate()
        for i in range(global_sizes[0][1]):
            work_1.set(0.)
            work_1[i] = 1.
            work_1.assemble()
            scatter_l2g(work_1, works_1, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
            As.mult(works_1, works_2)
            work_2.set(0.)
            scatter_l2g(works_2, work_2, PETSc.InsertMode.ADD_VALUES)

            col = np.asarray(np.where(works_1[:] == 1)[0], dtype='int32')
            scatter_l2g(work_2, works_1, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
            if col.size > 0:
                mask_B = As[:, col[0]] != 0
                mask_works = works_1[:] != 0
                mask = np.logical_and(mask_B, mask_works)
                indices = np.asarray(range(local_sizes[0][0]), dtype='int32')
                Bs.setValues(indices[mask], col[0], As[:, col[0]][mask]**2/works_1[:][mask], PETSc.InsertMode.INSERT_VALUES)

        Bs.assemble()

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

        mus = works_1.duplicate()
        #compute the multiplicity of each dof
        works_1.set(1.)
        work_1.set(0.)
        scatter_l2g(works_1, work_1, PETSc.InsertMode.ADD_VALUES)
        scatter_l2g(work_1, mus, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)

        _, mult_max = work_1.max()

        if self.viewPC:
            _, self.ns = mus.getSizes()
            _, self.nglob = work_1.getSizes()
            tempglobal = work_1.getArray(readonly=True)
            templocal = mus.getArray(readonly=True)
            self.nints = np.count_nonzero(tempglobal == 1) #interor dofs in this subdomain
            self.nGammas = np.count_nonzero(templocal -1) #interor dofs in this subdomain

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
                Vnegs.append(works_1.duplicate())
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

        works_1.set(0.)
        RsVnegs = []
        Vneg = []
        Dneg = []
        RsDVnegs = []
        RsDnegs = []
        for i in range(mpi.COMM_WORLD.size):
            nnegi = len(Vnegs) if i == mpi.COMM_WORLD.rank else None
            nnegi = mpi.COMM_WORLD.bcast(nnegi, root=i)
            for j in range(nnegi):
                Vneg.append(Vnegs[j].copy() if i == mpi.COMM_WORLD.rank else works_1.copy())
                dnegi = Dnegs[j] if i == mpi.COMM_WORLD.rank else None
                dnegi = mpi.COMM_WORLD.bcast(dnegi, root=i)
                Dneg.append(dnegi)
                #print(f'i Dneg[i] = {i} {Dneg[i]}')
        for i, vec in enumerate(Vneg):
            work_1.set(0)
            scatter_l2g(vec, work_1, PETSc.InsertMode.ADD_VALUES)
            scatter_l2g(work_1, works_1, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)

            if works_1.norm() != 0:
                RsVnegs.append(works_1.copy())
                RsDVnegs.append(Dneg[i]*works_1.copy())
                RsDnegs.append(Dneg[i])
            #TO DO: here implement RsVnegs and RsDVnegs
        #self.Vneg = Vneg

#        self.Vnegs = Vnegs
#        self.DVnegs = DVnegs
#        self.scatterl

#Local Apos and Aneg
        Aneg = PETSc.Mat().createPython([work_1.getSizes(), work_1.getSizes()], comm=PETSc.COMM_WORLD)
        Aneg.setPythonContext(Aneg_ctx(Vnegs, DVnegs, scatter_l2g, works_1, works_2))
        Aneg.setUp()

        Apos = PETSc.Mat().createPython([work_1.getSizes(), work_1.getSizes()], comm=PETSc.COMM_WORLD)
        Apos.setPythonContext(Apos_ctx(A, Aneg ))
        Apos.setUp()
        #A pos = A + Aneg so it could be a composite matrix rather than Python type

        Anegs = PETSc.Mat().createPython([works_1.getSizes(), works_1.getSizes()], comm=PETSc.COMM_SELF)
        Anegs.setPythonContext(Anegs_ctx(Vnegs, DVnegs))
        Anegs.setUp()

        Aposs = PETSc.Mat().createPython([works_1.getSizes(), works_1.getSizes()], comm=PETSc.COMM_SELF)
        Aposs.setPythonContext(Aposs_ctx(Bs, Anegs ))
        Aposs.setUp()

        projVnegs = PETSc.Mat().createPython([works_1.getSizes(), works_1.getSizes()], comm=PETSc.COMM_SELF)
        projVnegs.setPythonContext(projVnegs_ctx(Vnegs))
        projVnegs.setUp()

        projVposs = PETSc.Mat().createPython([works_1.getSizes(), works_1.getSizes()], comm=PETSc.COMM_SELF)
        projVposs.setPythonContext(projVposs_ctx(projVnegs))
        projVposs.setUp()

        #TODO Implement RsAposRsts, this is the restriction of Apos to the dofs in this subdomain. So it applies to local vectors but has non local operations
        RsAposRsts = PETSc.Mat().createPython([works_1.getSizes(), works_1.getSizes()], comm=PETSc.COMM_SELF) #or COMM_WORLD ?
        RsAposRsts.setPythonContext(RsAposRsts_ctx(As,RsVnegs,RsDVnegs))
        RsAposRsts.setUp()

        invAposs = PETSc.Mat().createPython([works_1.getSizes(), works_1.getSizes()], comm=PETSc.COMM_SELF)
        invAposs.setPythonContext(invAposs_ctx(Bs_ksp, projVposs ))
        invAposs.setUp()

        ksp_Aposs = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp_Aposs.setOperators(Aposs)
        ksp_Aposs.setType('preonly')
        pc_Aposs = ksp_Aposs.getPC()
        pc_Aposs.setType('python')
        pc_Aposs.setPythonContext(invAposs_ctx(Bs_ksp,projVposs))
        ksp_Aposs.setUp()
        work_1.set(1.)

        Ms = PETSc.Mat().createPython([works_1.getSizes(), works_1.getSizes()], comm=PETSc.COMM_SELF)
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


            #once a ksp has been passed to SLEPs it cannot be used again so we use a second, identical, ksp for SLEPc as a temporary fix
            ksp_Atildes_forSLEPc = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            ksp_Atildes_forSLEPc.setOptionsPrefix("ksp_Atildes_")
            ksp_Atildes_forSLEPc.setOperators(Atildes)
            ksp_Atildes_forSLEPc.setType('preonly')
            pc_Atildes_forSLEPc = ksp_Atildes_forSLEPc.getPC()
            pc_Atildes_forSLEPc.setType('python')
            pc_Atildes_forSLEPc.setPythonContext(scaledmats_ctx(invAposs,invmus,invmus) )
            ksp_Atildes_forSLEPc.setFromOptions()

        labs=[]
        for i, tmp in enumerate(Dnegs):
            labs.append(f'(\Lambda_-^s)_{i} = {-1.*tmp}')
        minV0 = minimal_V0(ksp_Atildes,invmusVnegs,labs) #won't compute anything more vecause the solver for Atildes is not mumps
        minV0s = minV0.V0s
        labs = minV0.labs
        if self.viewminV0 == True:
            minV0.view()

        self.A = A
        self.Apos = Apos
        self.Aneg = Aneg
        self.Ms = Ms
        self.As = As
        self.RsAposRsts = RsAposRsts
        self.ksp_Atildes = ksp_Atildes
        self.ksp_Ms = ksp_Ms
        self.ksp_Atildes_forSLEPc = ksp_Atildes_forSLEPc
        self.ksp_Ms_forSLEPc = ksp_Ms_forSLEPc
        self.work = work_1
        self.work_2 = work_2
        self.works_1 = works_1
        self.works_2 = works_2
        self.scatter_l2g = scatter_l2g
        self.mult_max = mult_max
        self.ksp_Atildes = ksp_Atildes
        self.minV0 = minV0
        self.labs = labs
        self.Dnegs = Dnegs
        self.nnegs = nnegs

        self.works_1.set(1.)
        self.RsAposRsts.mult(self.works_1,self.works_2)

        if self.GenEO == True:
          print(f'{labs=}')
          self.GenEOV0 = GenEO_V0(self.ksp_Atildes_forSLEPc,self.Ms,self.RsAposRsts,self.mult_max,minV0s,labs,self.ksp_Ms_forSLEPc)
          self.V0s = self.GenEOV0.V0s
          if self.viewGenEOV0 == True:
              self.GenEOV0.view()
              print(f'{self.GenEOV0.labs=}')
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
        self.H2 = PETSc.Mat().createPython([work_1.getSizes(), work_1.getSizes()], comm=PETSc.COMM_WORLD)
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
        Vneg = []
        for i in range(mpi.COMM_WORLD.size):
            nnegi = len(Vnegs) if i == mpi.COMM_WORLD.rank else None
            nnegi = mpi.COMM_WORLD.bcast(nnegi, root=i)
            for j in range(nnegi):
                if i == mpi.COMM_WORLD.rank:
                    works_1 = Vnegs[j].copy()
                else:
                    works_1.set(0.)
                self.work.set(0)
                self.scatter_l2g(works_1, self.work, PETSc.InsertMode.ADD_VALUES)
                Vneg.append(self.work.copy())
        AposinvV0 = []
        self.ritz_eigs_apos = None
        for vec in Vneg:
            self.ksp_Apos.solve(vec,self.work_2)
            if self.compute_ritz_apos and self.ritz_eigs_apos is None:
                self.ritz_eigs_apos = self.ksp_Apos.computeEigenvalues()
                self.ksp_Apos.setComputeEigenvalues(False)

            AposinvV0.append(self.work_2.copy())
        self.AposinvV0 = AposinvV0
        self.proj3 = coarse_operators(self.AposinvV0,self.A,self.scatter_l2g,self.works_1,self.work,V0_is_global=True)
        self.proj = self.proj3 #this name is consistent with the proj in PCBNN

###############################
#        ###Alternative to assembling the second coarse operators
#
#        ###
#        self.Id = PETSc.Mat().createPython([work.getSizes(), work.getSizes()], comm=PETSc.COMM_WORLD)
#        self.Id.setPythonContext(Id_ctx())
#        self.Id.setUp()
#
#        #self.Id = PETSc.Mat().create(comm=PETSc.COMM_SELF)
#        #self.Id.setType("constantdiagonal") #I don't know how to set the value to 1
#
#        #self.N = PETSc.Mat().createPython([work.getSizes(), work.getSizes()], comm=PETSc.COMM_WORLD)
#        #self.N.setPythonContext(N_ctx(self.Aneg,self.A,self.ksp_Apos,self.work,self.work_2))
#        #self.N.setUp()
#
#        #self.ksp_N = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
#        #self.ksp_N.setOptionsPrefix("ksp_N_")
#        #self.ksp_N.setOperators(self.N)
#        #self.ksp_N.setType("gmres")
#        #self.ksp_N.setGMRESRestart(151)
##       # if self.compute_ritz_N:
#        #self.ksp_N.setComputeEigenvalues(True)
#        ##self.pc_N = self.ksp_N.getPC()
#        ##self.pc_N.setType('python')
#        ##self.pc_N.setPythonContext(
#        #self.ksp_N.setFromOptions()
#        self.proj4 = coarse_operators(Vneg,self.Id,self.scatter_l2g,self.works_1,self.work,V0_is_global=True)
#
#        self.ProjA = PETSc.Mat().createPython([work.getSizes(), work.getSizes()], comm=PETSc.COMM_WORLD)
#        self.ProjA.setPythonContext(ProjA_ctx(self.proj4,self.A))
#        self.ProjA.setUp()
#        self.work.set(1.)
#        #test = self.work.duplicate()
#        #self.ProjA.mult(self.work,test)
#        #print('self.ProjA works ok')
#
#        self.ksp_ProjA = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
#        self.ksp_ProjA.setOptionsPrefix("ksp_ProjA_")
#        self.ksp_ProjA.setOperators(self.ProjA)
#        self.ksp_ProjA.setType("gmres")
#        self.ksp_ProjA.setGMRESRestart(151)
#        self.ksp_ProjA.setComputeEigenvalues(True)
#        #self.pc_ProjA = self.ksp_N.getPC()
#        #self.pc_ProjA.setType('python')
#        #self.pc_ProjA.setPythonContext(
#        self.ksp_ProjA.setFromOptions()
###############################
        ##
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
        self.gathered_ns = mpi.COMM_WORLD.gather(self.ns, root=0)
        self.gathered_nints = mpi.COMM_WORLD.gather(self.nints, root=0)
        self.gathered_Gammas =  mpi.COMM_WORLD.gather(self.nGammas, root=0)
        self.minV0.gathered_dim = mpi.COMM_WORLD.gather(self.minV0.nrb, root=0)
        self.gathered_labs = mpi.COMM_WORLD.gather(self.labs, root=0)
        self.gathered_nneg = mpi.COMM_WORLD.gather(self.nnegs, root=0)
        self.gathered_Dneg = mpi.COMM_WORLD.gather(self.Dnegs, root=0)
        if self.GenEO == True:
            self.GenEOV0.gathered_nsharp = mpi.COMM_WORLD.gather(self.GenEOV0.n_GenEO_eigmax, root=0)
            self.GenEOV0.gathered_nflat = mpi.COMM_WORLD.gather(self.GenEOV0.n_GenEO_eigmin, root=0)
            self.GenEOV0.gathered_dimKerMs = mpi.COMM_WORLD.gather(self.GenEOV0.dimKerMs, root=0)
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
            print(f'{self.mult_max=}')
            print(f'### info about the subdomains ###')
            self.nint = np.sum(self.gathered_nints)
            self.nGamma = self.nglob - self.nint
            print(f'{self.gathered_ns =}')
            print(f'{self.gathered_nints =}')
            print(f'{self.gathered_Gammas=}')
            print(f'{self.nGamma=}')
            print(f'{self.nint=}')
            print(f'{self.nglob=}')
            print(f'{self.gathered_labs=}')
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
                print(f'### info about GenEOV0.V0s  ###')
                print(f'{self.GenEOV0.tau_eigmax=}')
                print(f'{self.GenEOV0.tau_eigmin=}')
                print(f'{self.GenEOV0.eigmax=}')
                print(f'{self.GenEOV0.eigmin=}')
                print(f'{self.GenEOV0.nev=}')
                print(f'{self.GenEOV0.maxev=}')
                print(f'{self.GenEOV0.mumpsCntl3=}')
                print(f'{self.GenEOV0.verbose=}')
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
            if self.compute_ritz_apos and self.ritz_eigs_apos is not None:
                print(f'Estimated kappa(H2 Apos) = {self.ritz_eigs_apos.max()/self.ritz_eigs_apos.min() }; with lambdamin = {self.ritz_eigs_apos.min()} and   lambdamax = {self.ritz_eigs_apos.max()}')

            print('#############################')
            self.savetofile()

    def savetofile(self):
        if mpi.COMM_WORLD.rank == 0:
            if not os.path.exists(self.test_case):
                os.mkdir(self.test_case)
            np.savez(f'{self.test_case}/init',
            switchtoASM        = self.switchtoASM,
            verbose            = self.verbose,
            GenEO              = self.GenEO,
            H3addCS            = self.H3addCS,
            H3projCS           = self.H3projCS,
            H2projCS           = self.H2projCS,
            viewPC             = self.viewPC,
            viewV0             = self.viewV0,
            viewGenEOV0        = self.viewGenEOV0,
            viewnegV0          = self.viewnegV0,
            viewminV0          = self.viewminV0,
            compute_ritz_apos  = self.compute_ritz_apos,
            mult_max           = self.mult_max,
            gathered_ns        = self.gathered_ns,
            gathered_nints     = self.gathered_nints,
            gathered_Gammas    = self.gathered_Gammas,
            nGamma             = self.nGamma,
            nint               = self.nint,
            nglob              = self.nglob,
            minV0_mumpsCntl3   = self.minV0.mumpsCntl3,
            gathered_labs=  np.asarray(self.gathered_labs,dtype='object'),
            gathered_nneg      = self.gathered_nneg,
            minV0_gathered_dim = self.minV0.gathered_dim,
            ritz_eigs_Apos     = self.ritz_eigs_apos ,
            sum_nneg = np.sum(self.gathered_nneg),
            proj2_V0_is_global = self.proj2.V0_is_global,
            proj2_gathered_dimV0s = np.asarray(self.proj2.gathered_dimV0s),
            proj2_dimV0 = np.sum(self.proj2.gathered_dimV0s),
            proj2_sum_dimminV0 = np.sum(self.minV0.gathered_dim) ,
            )
            if self.GenEO == True:
                np.savez(f'{self.test_case}/GenEO',
                GenEOV0_tau_eigmax      = self.GenEOV0.tau_eigmax,
                GenEOV0_tau_eigmin      = self.GenEOV0.tau_eigmin,
                GenEOV0_eigmax          = self.GenEOV0.eigmax,
                GenEOV0_eigmin          = self.GenEOV0.eigmin,
                GenEOV0_nev             = self.GenEOV0.nev,
                GenEOV0_maxev           = self.GenEOV0.maxev,
                GenEOV0_mumpsCntl3      = self.GenEOV0.mumpsCntl3,
                GenEOV0_verbose         = self.GenEOV0.verbose,
                GenEOV0_gathered_nsharp = np.asarray(self.GenEOV0.gathered_nsharp),
                GenEOV0_gathered_nflat  = np.asarray(self.GenEOV0.gathered_nflat),
                GenEOV0_sum_nsharp = np.sum(self.GenEOV0.gathered_nsharp),
                GenEOV0_sum_nflat = np.sum(self.GenEOV0.gathered_nflat),
                )


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
        for i, vec in enumerate(self.Vnegs):
            self.works_2.axpy(self.works.dot(self.DVnegs[i]), vec)
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
    def __init__(self,A, Aneg):
        self.A = A
        self.Aneg = Aneg
    def mult(self, mat, x, y):
        xtemp = x.duplicate()
        self.Aneg.mult(x,xtemp)
        self.A.mult(x,y)
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

class N_ctx(object): #(I - Aneg *invApos)
    def __init__(self,Aneg,A,ksp_Apos,work,work_2):
        self.Aneg = Aneg
        self.A = A
        self.ksp_Apos = ksp_Apos
        self.work = work
        self.work_2 = work_2
    def mult(self, mat, x, y):
        if mpi.COMM_WORLD.rank == 0:
            print('in N_ctx')
        self.ksp_Apos.solve(x,self.work)
        #self.A.mult(self.work,y)
        self.Aneg.mult(self.work,self.work_2)
        #self.Aneg.mult(self.work,y)
        y.set(0.)
        y.axpy(1.,x)
        y.axpy(-1.,self.work_2)
class Id_ctx(object): # I
#    def __init__(self):
    def mult(self, mat, x, y):
        y.axpy(1.,x)

class ProjA_ctx(object): #(Vneg (Vnegt *Vneg)\Vnegt    *A )
    def __init__(self, proj4, A):
        self.proj4 = proj4
        self.A = A
    def mult(self, mat, x, y):
        xd = x.copy()
        self.proj4.project(xd)
        self.A.mult(xd,y)

###UNFINISHED
#class Psi_ctx(object) #Dneg^{-1} - Vnegt Aneg^{-1} Vneg
#    def __init__(self,Vneg,ksp_Apos,work,work_2):
#        self.Vneg = Vneg
#        self.work = work
#        self.work = work_2
#        #self.gamma = PETSc.Vec().create(comm=PETSc.COMM_SELF)
#        #self.gamma.setType(PETSc.Vec.Type.SEQ)
#        #self.gamma.setSizes(len(self.Vneg))
#        self.ksp_Apos = ksp_Apos
#    def mult(self, mat, x, y):
# part with Dneg inv is not at all implemented yet
#        self.work.set(0.)
#        for i, vec in enumerate(self.Vneg):
#            self.work.axpy(x[i], vec)
#
#        self.ksp_Apos.solve(self.work,self.work_2)
#
#        #y = self.gamma.duplicate()
#        for i, vec in enumerate(self.V0):
#            y[i] = vec.dot(x)
#

