# Authors:
#     Loic Gouarin <loic.gouarin@cmap.polytechnique.fr>
#     Nicole Spillane <nicole.spillane@cmap.polytechnique.fr>
#
# License: BSD 3 clause
from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)
import mpi4py.MPI as mpi
import os
from petsc4py import PETSc
import numpy as np
import json
from GenEO import *

def rhs(coords, rhs):
    n = rhs.shape
    rhs[..., 1] = -9.81

def save_json(path, E1, E2, nu1, nu2, Lx, Ly, stripe_nb, ksp, pc, ritz):
    results = {}

    if mpi.COMM_WORLD.rank == 0:
        results['E1'] = E1
        results['E2'] = E2
        results['nu1'] = nu1
        results['nu2'] = nu2
        results['Lx'] = Lx
        results['Ly'] = Ly
        results['stripe_nb'] = stripe_nb
        results['gathered_ns'] = pc.gathered_ns
        results['gathered_Gammas'] = pc.gathered_Gammas
#        results['nGamma'] = pc.nGamma
        results['nglob'] = pc.nglob

        if hasattr(pc, 'proj2'):
            proj = pc.proj2
        else:
            proj = pc.proj

        if hasattr(proj, 'gathered_dimV0s'):
            results['gathered_dimV0s'] = proj.gathered_dimV0s
            results['V0dim'] = float(np.sum(proj.gathered_dimV0s))

        results['minV0_gathered_dim']  = pc.minV0.gathered_dim
        results['minV0dim'] = float(np.sum(pc.minV0.gathered_dim))
        results['gathered_labs'] =  pc.gathered_labs

        if pc.GenEO == True:
            results['GenEOV0_gathered_nsharp'] = pc.GenEOV0.gathered_nsharp
            results['GenEOV0_gathered_nflat'] = pc.GenEOV0.gathered_nflat
            results['GenEOV0_gathered_dimKerMs'] = pc.GenEOV0.gathered_dimKerMs
            results['GenEOV0_gathered_Lambdasharp'] = [[(d.real, d.imag) for d in l] for l in pc.GenEOV0.gathered_Lambdasharp]
            results['GenEOV0_gathered_Lambdaflat'] =  [[(d.real, d.imag) for d in l] for l in pc.GenEOV0.gathered_Lambdaflat]
            results['sum_nsharp'] = float(np.sum(pc.GenEOV0.gathered_nsharp))
            results['sum_nflat'] = float(np.sum(pc.GenEOV0.gathered_nflat))
            results['sum_dimKerMs'] = float(np.sum(pc.GenEOV0.gathered_dimKerMs))

        if isinstance(pc, PCNew):
            results['Aposrtol'] = pc.ksp_Apos.getTolerances()[0]
            results['gathered_nneg'] = pc.gathered_nneg
            results['sum_gathered_nneg'] = float(np.sum(pc.gathered_nneg))
            if pc.compute_ritz_apos and pc.ritz_eigs_apos is not None:
                rmin, rmax = pc.ritz_eigs_apos.min(), pc.ritz_eigs_apos.max()
                kappa = rmax/rmin
                results['kappa_apos'] = (kappa.real, kappa.imag)
                results['lambdamin_apos'] = (rmin.real, rmin.imag)
                results['lambdamax_apos'] = (rmax.real, rmax.imag)
        else:
            results['sum_gathered_nneg'] = float(0.)


        results['precresiduals'] = ksp.getConvergenceHistory()[:].tolist()
        results['l2normofAxminusb'] = tmp1
        results['l2normofA'] = tmp2
        if ritz is not None:
            rmin, rmax = ritz.min(), ritz.max()
            kappa = rmax/rmin
            results['lambdamin'] = (rmin.real, rmin.imag)
            results['lambdamax'] = (rmax.real, rmax.imag)
            results['kappa'] = (kappa.real, kappa.imag)

        results['taueigmax'] = pc.GenEOV0.tau_eigmax
        results['nev'] = pc.GenEOV0.nev

        with open(f'{path}/results.json', 'w') as f:
            json.dump(results, f, indent=4, sort_keys=True)

def save_coarse_vec(path, da, pcbnn):
    coords = da.getCoordinates()
    pcbnn.scatter_l2g(coords, pcbnn.works_1, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
    pcbnn.works_1.name = "coordinates"

    for iv, v in enumerate(pcbnn.V0s):
        viewer = PETSc.Viewer().createHDF5(f'{path}/coarse_vec_{iv}_{mpi.COMM_WORLD.rank}.h5', 'w', comm = PETSc.COMM_SELF)
        v.name = "coarse_vec"
        v.view(viewer)
        pcbnn.works_1.view(viewer)
        viewer.destroy()

    prop = {}
    prop['eigs'] = pcbnn.labs

    with open(f'{path}/properties_{mpi.COMM_WORLD.rank}.txt', 'w') as outfile:
        json.dump(prop, outfile)

OptDB = PETSc.Options()
Lx = OptDB.getInt('Lx', 4)
Ly = OptDB.getInt('Ly', 1)
n  = OptDB.getInt('n', 16)
nx = OptDB.getInt('nx', Lx*n+1)
ny = OptDB.getInt('ny', Ly*n+1)
E1 = OptDB.getReal('E1', 10**12)
E2 = OptDB.getReal('E2', 10**6)
nu1 = OptDB.getReal('nu1', 0.4)
nu2 = OptDB.getReal('nu2', 0.4)
test_case = OptDB.getString('test_case', 'default')
isPCNew = OptDB.getBool('PCNew', True)
computeRitz  =  OptDB.getBool('computeRitz', True)
stripe_nb = OptDB.getInt('stripe_nb', 3)

#TODO: I did this just so I could save this option to json file. The variable tmp_ksp_Apos_rtol is never used
tmp_ksp_Apos_rtol = OptDB.getReal('ksp_Apos_ksp_rtol', -1)

if mpi.COMM_WORLD.rank == 0:
    if not os.path.exists(test_case):
        os.mkdir(test_case)

hx = Lx/(nx - 1)
hy = Ly/(ny - 1)

da = PETSc.DMDA().create([nx, ny], dof=2, stencil_width=1)
da.setUniformCoordinates(xmax=Lx, ymax=Ly)
da.setMatType(PETSc.Mat.Type.IS)

def lame_coeff(x, y, v1, v2, stripe_nb, Ly):
    if stripe_nb == 0:
        if mpi.COMM_WORLD.rank == 0:
            print(f'Test number {stripe_nb} - no stripes E = {E1}')
        mask = False
    elif stripe_nb == 1:
        if mpi.COMM_WORLD.rank == 0:
            print(f'Test number {stripe_nb} - one stripe')
        mask = np.logical_and(1./7<=y-np.floor(y), y-np.floor(y)<=2./7)
        #mask = np.logical_and(1./7<=y, y<=2./7)
    elif stripe_nb == 2:
        if mpi.COMM_WORLD.rank == 0:
            print(f'Test number {stripe_nb} - two stripes')
        #mask= np.logical_or(np.logical_and(1./7<=y, y<=2./7),np.logical_and(3./7<=y, y<=4./7))
        mask= np.logical_or(np.logical_and(1./7<=y-np.floor(y), y-np.floor(y)<=2./7),np.logical_and(3./7<=y-np.floor(y), y-np.floor(y)<=4./7))
    elif stripe_nb == 3:
        if mpi.COMM_WORLD.rank == 0:
            print(f'Test number {stripe_nb} - three stripes')
        mask= np.logical_or(np.logical_or(np.logical_and(1./7<=y-np.floor(y), y-np.floor(y)<=2./7),np.logical_and(3./7<=y-np.floor(y), y-np.floor(y)<=4./7)), np.logical_and(5./7<=y-np.floor(y), y-np.floor(y)<=6./7))
    else:
        if mpi.COMM_WORLD.rank == 0:
            print(f'Test number {stripe_nb} is not implemented, instead I set E={E2}')
        mask = True
    output = np.empty(x.shape)
    output[mask] = v1
    output[np.logical_not(mask)] = v2
    return output

# non constant Young's modulus and Poisson's ratio
E = buildCellArrayWithFunction(da, lame_coeff, (E1,E2,stripe_nb,Ly))
nu = buildCellArrayWithFunction(da, lame_coeff, (nu1,nu2,stripe_nb,Ly))

lamb = (nu*E)/((1+nu)*(1-2*nu))
mu = .5*E/(1+nu)

class callback:
    def __init__(self, da):
        self.da = da
        ranges = da.getRanges()
        ghost_ranges = da.getGhostRanges()

        self.slices = []
        for r, gr in zip(ranges, ghost_ranges):
            self.slices.append(slice(gr[0], r[1]))
        self.slices = tuple(self.slices)

        self.it = 0

    def __call__(self, locals):
        pyKSP = locals['self']
        proj = pyKSP.mpc.proj

        if self.it == 0:
            work, _ = proj.A.getVecs()
            for i, vec in enumerate(proj.V0):
                if vec:
                    proj.works = vec.copy()
                else:
                    proj.works.set(0.)
                work.set(0)
                proj.scatter_l2g(proj.works, work, PETSc.InsertMode.ADD_VALUES)

                viewer = PETSc.Viewer().createVTK('output.d/coarse_vec_{}.vts'.format(i), 'w', comm = PETSc.COMM_WORLD)
                tmp = self.da.createGlobalVec()
                tmpl_a = self.da.getVecArray(tmp)
                work_a = self.da.getVecArray(work)
                tmpl_a[:] = work_a[:]
                tmp.view(viewer)
                viewer.destroy()
            self.it += 1


x = da.createGlobalVec()
b = buildRHS(da, [hx, hy], rhs)
A = buildElasticityMatrix(da, [hx, hy], lamb, mu)
A.assemble()
bcApplyWest(da, A, b)

#Setup the preconditioner (or multipreconditioner) and the coarse space
# if isPCNew:
#     pcbnn = PCNew(A)
# else:
#     pcbnn = PCBNN(A)

pcbnn = PCAWG(A.convert('mpiaij'))

save_coarse_vec(test_case, da, pcbnn)

##############compute x FOR INITIALIZATION OF PCG
if mpi.COMM_WORLD.rank == 0:
    print('Solve a problem with A and H3')
# Random initial guess
#print('Random rhs')
#b.setRandom()

x.setRandom()

xnorm = b.dot(x)/x.dot(A*x)
x *= xnorm

#Pre-compute solution in coarse space
#Required for PPCG (projected preconditioner)
#Doesn't hurt or help the hybrid and additive preconditioners
#the initial guess is passed to the PCG below with the option ksp.setInitialGuessNonzero(True)

pcbnn.proj.project(x)
xtild = pcbnn.proj.coarse_init(b)
tmp = xtild.norm()
if mpi.COMM_WORLD.rank == 0:
    print(f'norm xtild (coarse component of solution) {tmp}')
x += xtild
############END of: compute x FOR INITIALIZATION OF PCG

#############SETUP KSP
ksp = PETSc.KSP().create()
ksp.setOperators(pcbnn.A)
ksp.setOptionsPrefix("global_ksp_")

pc = ksp.pc
pc.setType('python')
pc.setPythonContext(pcbnn)
pc.setFromOptions()

ksp.setType("cg")
if computeRitz:
    ksp.setComputeEigenvalues(True)
#ksp.setType(ksp.Type.PYTHON)
#pyKSP = KSP_PCG()
#ksp.setPythonContext(pyKSP)
##pyKSP.callback = callback(da)

ksp.setInitialGuessNonzero(True)
ksp.setConvergenceHistory(True)

ksp.setFromOptions()
#### END SETUP KSP

###### SOLVE:
ksp.solve(b, x)

viewer = PETSc.Viewer().createVTK(f'{test_case}/solution_2d.vts', 'w', comm = PETSc.COMM_WORLD)
x.view(viewer)

if computeRitz:
    Ritz = ksp.computeEigenvalues()
    Ritzmin = Ritz.min()
    Ritzmax = Ritz.max()
else:
    Ritz = None
convhistory = ksp.getConvergenceHistory()


if ksp.getInitialGuessNonzero() == False:
    x+=xtild

Ax = x.duplicate()
pcbnn.A.mult(x,Ax)
tmp1 = (Ax - b).norm()
tmp2 = b.norm()
if mpi.COMM_WORLD.rank == 0:
    print(f'norm of A x - b = {tmp1}, norm of b = {tmp2}')
    print('convergence history', convhistory)
    if computeRitz:
        print(f'Estimated kappa(H3 A) = {Ritzmax/Ritzmin}; with lambdamin = {Ritzmin} and lambdamax = {Ritzmax}')

save_json(test_case, E1, E2, nu1, nu2, Lx, Ly, stripe_nb, ksp, pcbnn, Ritz)

#if mpi.COMM_WORLD.rank == 0:
#    print('compare with MUMPS global solution')
#
#ksp_Amumps = PETSc.KSP().create(comm=PETSc.COMM_SELF)
#ksp_Amumps.setOptionsPrefix("ksp_Amumps_")
#ksp_Amumps.setOperators(A)
#ksp_Amumps.setType('preonly')
#pc_Amumps = ksp_Amumps.getPC()
#pc_Amumps.setType('cholesky')
#pc_Amumps.setFactorSolverType('mumps')
#ksp_Amumps.setFromOptions()
#
#ksp_Amumps.solve(b,x)
#if mpi.COMM_WORLD.rank == 0:
#    print('finished computing MUMPS global solution')


