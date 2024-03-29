# Authors:
#     Loic Gouarin <loic.gouarin@cmap.polytechnique.fr>
#     Nicole Spillane <nicole.spillane@cmap.polytechnique.fr>
#
# License: BSD 3 clause
# from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)

from dolfinx import plot
from dolfinx import fem
from dolfinx.fem import Expression, Function, VectorFunctionSpace,FunctionSpace, dirichletbc, Constant, locate_dofs_geometrical
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle
import ufl

import mpi4py.MPI as mpi
from petsc4py import PETSc
import numpy as np
from GenEO import *
import json
import matplotlib.pyplot as plt

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
nx = OptDB.getInt('nx', Lx*n)
ny = OptDB.getInt('ny', Ly*n)
E1 = OptDB.getReal('E1', 10**12)
E2 = OptDB.getReal('E2', 10**6)
nu1 = OptDB.getReal('nu1', 0.4)
nu2 = OptDB.getReal('nu2', 0.4)
test_case = OptDB.getString('test_case', 'default')
isPCNew = OptDB.getBool('PCNew', True)
computeRitz  =  OptDB.getBool('computeRitz', True)
stripe_nb = OptDB.getInt('stripe_nb', 3)

# Create mesh and define function space
mesh = create_rectangle(mpi.COMM_WORLD, ((0., 0.), (Lx, Ly)), (nx, ny), CellType.triangle)

class expr:
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2

    def eval(self, coords):
        x, y = coords[0], coords[1]
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

        return np.full(coords.shape[1], self.v1*mask + self.v2*np.logical_not(mask))

V0 = FunctionSpace(mesh, ("DG", 0))
nu = Function(V0, name='nu')
f_nuk = expr(nu1, nu2)
nu.interpolate(f_nuk.eval)
E = Function(V0, name='E')
f_Ek = expr(E1, E2)
E.interpolate(f_Ek.eval)

lambda_ = (nu*E)/((1+nu)*(1-2*nu))
mu = .5*E/(1+nu)

rho_g = 9.81
f = Constant(mesh, (0, -rho_g))

V = VectorFunctionSpace(mesh, ('Lagrange', 1))

def eps(v):
    return ufl.sym(ufl.grad(v))

def sigma(v):
    return lambda_*ufl.tr(eps(v))*ufl.Identity(len(v)) + 2.0*mu*eps(v)

du = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)

a = fem.form(ufl.inner(sigma(du), eps(u_))*ufl.dx)
l = fem.form(ufl.inner(f, u_)*ufl.dx)

def left(x):
    return np.isclose(x[0], 0.)

bc = dirichletbc(np.zeros(2), locate_dofs_geometrical(V, left), V)

u = Function(V, name="Displacement")

A = fem.petsc.assemble_matrix(a, bcs=[bc])
A.assemble()
b = fem.petsc.assemble_vector(l)
fem.petsc.set_bc(b, [bc])

def set_pcbnn(ksp, A, b, x):
    pcbnn = PCAWG(A)

    x.setRandom()

    xnorm = b.dot(x)/x.dot(A*x)
    x *= xnorm

    pcbnn.proj.project(x)
    xtild = pcbnn.proj.coarse_init(b)
    tmp = xtild.norm()
    if mpi.COMM_WORLD.rank == 0:
        print(f'norm xtild (coarse component of solution) {tmp}')
    x += xtild
    ############END of: compute x FOR INITIALIZATION OF PCG
    ksp.setOperators(pcbnn.A)

    pc = ksp.pc
    pc.setType('python')
    pc.setPythonContext(pcbnn)
    pc.setFromOptions()

    return pcbnn
    #############SETUP KSP

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setOptionsPrefix("global_")

pc = ksp.pc
pc.setType(None)

# A = 0.5*(A + A.transpose())

pcawg = set_pcbnn(ksp, A, b, u.vector)

ksp.setType("cg")
if computeRitz:
    ksp.setComputeEigenvalues(True)

ksp.setInitialGuessNonzero(True)
ksp.setConvergenceHistory(True)

ksp.setFromOptions()
#### END SETUP KSP

###### SOLVE:
ksp.solve(b, u.vector)

if computeRitz:
    Ritz = ksp.computeEigenvalues()
    Ritzmin = Ritz.min()
    Ritzmax = Ritz.max()
else:
    Ritz = None
convhistory = ksp.getConvergenceHistory()


# if ksp.getInitialGuessNonzero() == False:
#     x+=xtild

Ax = b.duplicate()
A.mult(u.vector, Ax)
# pcbnn.A.mult(x,Ax)
tmp1 = (Ax - b).norm()
tmp2 = b.norm()
if mpi.COMM_WORLD.rank == 0:
    print(f'norm of A x - b = {tmp1}, norm of b = {tmp2}')
    print('convergence history', convhistory)
    if computeRitz:
        print(f'Estimated kappa(H3 A) = {Ritzmax/Ritzmin}; with lambdamin = {Ritzmin} and lambdamax = {Ritzmax}')

save_json(test_case, E1, E2, nu1, nu2, Lx, Ly, stripe_nb, ksp, pcawg, Ritz)

def get_rank(x):
    return mpi.COMM_WORLD.rank*np.ones(x.shape[1])

rank_field = Function(V0, name='rank')
rank_field.interpolate(get_rank)

with XDMFFile(mpi.COMM_WORLD, "displacement_2d.xdmf", "w") as ufile_xdmf:
    u.x.scatter_forward()
    ufile_xdmf.write_mesh(mesh)
    ufile_xdmf.write_function(u)
    ufile_xdmf.write_function(E)
    ufile_xdmf.write_function(nu)
    ufile_xdmf.write_function(rank_field)

unorm = u.x.norm()
if mpi.COMM_WORLD.rank == 0:
    print("norm: ", unorm)
