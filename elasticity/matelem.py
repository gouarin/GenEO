from __future__ import division, print_function
from six.moves import range
import sympy
from sympy import Rational as rat

x, y, z, hx, hy, hz = sympy.symbols('x, y, z, hx, hy, hz')
lamb, mu = sympy.symbols('lamb, mu')

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

    #return sympy.lambdify((hx, hy, hz, lamb, mu), output, "numpy")
    return output

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

def write_function():
    output_2d = getMatElemElasticity(2)

    output_3d = getMatElemElasticity(3)

    with open("matelem_cython.pyx", 'w') as f:
        f.write("# cython: boundscheck = False\n")
        f.write("# cython: wraparound = False\n")
        f.write("# cython: cdivision = True\n")
        f.write("import numpy as np\n")

        f.write("def getMatElemElasticity_2d_nonconst(double hx, double hy, double[:] lamb, double[:] mu):\n")
        f.write("\toutput = np.empty((lamb.size, {}, {}))\n".format(*output_2d.shape))
        f.write("\tcdef:\n")
        f.write("\t\tint ie\n")
        f.write("\t\tint size = lamb.size\n")
        f.write("\t\tdouble[:, :, ::1] vout = output\n")
        f.write("\tfor ie in range(size):\n")
        for i in range(output_2d.shape[0]):
            for j in range(output_2d.shape[1]):
                f.write("\t\tvout[ie, {}, {}] = ".format(i, j) + str(output_2d[i, j]).replace('lamb', 'lamb[ie]').replace('mu', 'mu[ie]') + "\n")
        f.write("\treturn output\n\n")

        f.write("def getMatElemElasticity_2d_const(double hx, double hy, double lamb, double mu):\n")
        f.write("\toutput = np.empty(({}, {}))\n".format(*output_2d.shape))
        f.write("\tcdef:\n")
        f.write("\t\tdouble[:, ::1] vout = output\n")
        for i in range(output_2d.shape[0]):
            for j in range(output_2d.shape[1]):
                f.write("\tvout[{}, {}] = ".format(i, j) + str(output_2d[i, j]) + "\n")
        f.write("\treturn output\n\n")

        f.write("def getMatElemElasticity_3d_nonconst(double hx, double hy, double hz, double[:] lamb, double[:] mu):\n")
        f.write("\toutput = np.empty((lamb.size, {}, {}))\n".format(*output_3d.shape))
        f.write("\tcdef:\n")
        f.write("\t\tint ie\n")
        f.write("\t\tint size = lamb.size\n")
        f.write("\t\tdouble[:, :, ::1] vout = output\n")
        f.write("\tfor ie in range(size):\n")
        for i in range(output_3d.shape[0]):
            for j in range(output_3d.shape[1]):
                f.write("\t\tvout[ie, {}, {}] = ".format(i, j) + str(output_3d[i, j]).replace('lamb', 'lamb[ie]').replace('mu', 'mu[ie]') + "\n")
        f.write("\treturn output\n\n")

        f.write("def getMatElemElasticity_3d_const(double hx, double hy, double hz, double lamb, double mu):\n")
        f.write("\toutput = np.empty(({}, {}))\n".format(*output_3d.shape))
        f.write("\tcdef:\n")
        f.write("\t\tdouble[:, ::1] vout = output\n")
        for i in range(output_3d.shape[0]):
            for j in range(output_3d.shape[1]):
                f.write("\tvout[{}, {}] = ".format(i, j) + str(output_3d[i, j]) + "\n")
        f.write("\treturn output\n\n")


if __name__ == "__main__":
    import time
    from sympy.utilities.autowrap import autowrap
    from sympy.utilities.autowrap import ufuncify

    write_function()

    # t1 = time.time()
    # output = getMatElemElasticity(2)
    # print(output)
    # t2 = time.time()
    # print("2d:", t2-t1)

    # t1 = time.time()
    # getMatElemElasticity(3)
    # t2 = time.time()
    # print("3d:", t2-t1)