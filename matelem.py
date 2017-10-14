from __future__ import division, print_function
from six.moves import range
import sympy
from sympy import Rational as rat

x, y, z, hx, hy, hz = sympy.symbols('x, y, z, hx, hy, hz')
lamb, mu = sympy.symbols('lambda, mu')

basis_function = {2: # 2d
        [  rat(1.)/(hx*hy)*(x - hx)*(y - hy), # ( 0, 0 )
          -rat(1.)/(hx*hy)*x*(y - hy),        # ( 0, hx)
           rat(1.)/(hx*hy)*x*y,               # (hx, hy)
          -rat(1.)/(hx*hy)*(x - hx)*y],       # ( 0, hy)
        # 3d
        3: [-rat(1.)/(hx*hy*hz)*(x - hx)*(y - hy)*(z - hz), # ( 0,  0,  0)
           rat(1.)/(hx*hy*hz)*x*(y - hy)*(z - hz),        # (hx,  0,  0)
          -rat(1.)/(hx*hy*hz)*x*y*(z - hz),               # (hx, hy,  0)
           rat(1.)/(hx*hy*hz)*(x - hx)*y*(z - hz),        # ( 0, hy,  0)
           rat(1.)/(hx*hy*hz)*(x - hx)*(y - hy)*z,        # ( 0,  0, hz)
          -rat(1.)/(hx*hy*hz)*x*(y - hy)*z,               # (hx,  0, hz)
           rat(1.)/(hx*hy*hz)*x*y*z,                      # (hx, hy, hz)
          -rat(1.)/(hx*hy*hz)*(x - hx)*y*z]               # ( 0, hy, hz)
}

def getMatElemElasticity(dim=2):
    eps_ii = lamb*sympy.ones(dim, dim) + 2*mu*sympy.eye(dim)
    dimC = dim + 1 if dim==2 else 2*dim
    C = sympy.zeros(dimC, dimC)
    C[:dim, :dim] = eps_ii

    if dim == 2:
        C[-1, -1] = mu
    elif dim == 3:
        for i in range(1, dim+1):
            C[-i, -i] = mu

    phi = basis_function[dim]

    B = sympy.zeros(C.shape[0], dim*len(phi))

    deriv = [x, y, z]

    for j in range(dim):
        for i in range(len(phi)):
            B[j, dim*i + j] = phi[i].diff(deriv[j])
    
    if dim == 2:
        for i in range(len(phi)):
            B[-1, dim*i] = phi[i].diff(y)
            B[-1, dim*i + 1] = phi[i].diff(x)
    elif dim == 3:
        for i in range(len(phi)):
            B[-3, dim*i] = phi[i].diff(y)
            B[-3, dim*i + 1] = phi[i].diff(x)
            B[-2, dim*i] = phi[i].diff(z)
            B[-2, dim*i + 2] = phi[i].diff(x)
            B[-1, dim*i + 1] = phi[i].diff(z)
            B[-1, dim*i + 2] = phi[i].diff(y)

    A = B.T*C*B

    output = sympy.zeros(dim*len(phi), dim*len(phi))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            #A[i, j] = A[i, j].together().factor()
            if dim == 2:
                output[i, j] = sympy.integrate(A[i, j], (y, 0., hy), (x, 0., hx)).expand()
            elif dim == 3:
                output[i, j] = sympy.integrate(A[i, j], (z, 0., hz), (y, 0., hy), (x, 0., hx)).expand()
            output[i, j].together().factor()

    return sympy.lambdify((hx, hy, hz, lamb, mu), output, "numpy")

def getMatElemMass(dim=2):
    phi = basis_function[dim]
    phi_size = len(phi)
    output = sympy.zeros(phi_size, phi_size)
    for i in range(phi_size):
        for j in range(phi_size):
            if dim == 2:
                output[i, j] = sympy.integrate(phi[i]*phi[j], (y, 0., hy), (x, 0., hx)).expand()
            elif dim == 3:
                output[i, j] = sympy.integrate(phi[i]*phi[j], (z, 0., hz), (y, 0., hy), (x, 0., hx)).expand()
            output[i, j].together().factor()

    return sympy.lambdify((hx, hy, hz), output, "numpy")

if __name__ == "__main__":
    import time

    t1 = time.time()
    getMatElemElasticity(2)
    t2 = time.time()
    print("2d:", t2-t1)

    t1 = time.time()
    getMatElemElasticity(3)
    t2 = time.time()
    print("3d:", t2-t1)