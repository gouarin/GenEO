import sympy
from sympy import Rational as rat

def getMatElemElasticity():
    x, y, hx, hy = sympy.symbols('x,y,hx,hy')
    lamb, mu, tmp = sympy.symbols('lambda, mu, tmp')

    phi = [ rat(1.)/(hx*hy)*(x - hx)*(y - hy),
           -rat(1.)/(hx*hy)*x*(y - hy),
            rat(1.)/(hx*hy)*x*y,
           -rat(1.)/(hx*hy)*(x - hx)*y]

    C = sympy.Matrix([[lamb + 2*mu,        lamb,  0],
                      [       lamb, lamb + 2*mu,  0],
                      [          0,           0, mu]])

    B = sympy.zeros(3, 8)

    for i in range(4):
        B[0, 2*i] = phi[i].diff(x)
        B[1, 2*i + 1] = phi[i].diff(y)
        B[2, 2*i] = phi[i].diff(y)
        B[2, 2*i + 1] = phi[i].diff(x)

    A = B.T*C*B

    output = sympy.zeros(8, 8)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            output[i, j] = sympy.integrate(A[i, j], (y, 0., hy), (x, 0., hx)).expand()
            output[i, j].simplify()

    return output