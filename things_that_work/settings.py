import numpy as np
from GaussJacobiQuadRule_V3 import Jacobi

print('settings_lib imported ')


class PROBDEF:

    def __init__(self, omega: tuple, r: int):
        self.omegax, self.omegay = omega
        self.r = r

    def u_exact(self, x, y):
        utemp = (0.1*np.sin(self.omegax*x) + np.tanh(self.r*x)) * \
            np.sin(self.omegay*(y))
        return utemp

    def f_exact(self, x, y):
        gtemp = (-0.1*(self.omegax**2)*np.sin(self.omegax*x) - (2*self.r**2)*(np.tanh(self.r*x))/((np.cosh(self.r*x))**2))*np.sin(self.omegay*(y))\
            + (0.1*np.sin(self.omegax*x) + np.tanh(self.r*x)) * \
            (-self.omegay**2 * np.sin(self.omegay*(y)))
        return gtemp

    def v(self, x, y, r):
        """
        Returns a polynomial of order n evaluated at x, y
        1 + x + y + xy + x^2 + y^2 + x^2y + ... + x^n + y^n
        """
        # powers = [(j, i) for i in range(0, r + 1) for j in range(0, r + 1) if i + j <= r]
        powers = []
        for i in range(r+1):
            for j in range(r+1):
                if i + j <= r:
                    powers.append((j,i))
        # print(powers)
        tot = []
        for a in powers:
            tot.append(x**a[0] * y**a[1])

        # print(test)
        return tot
        


    # @staticmethod
    # def test_func_x_core(n, x):
    #     test = Jacobi(n+1, 0, 0, x) - Jacobi(n-1, 0, 0, x)
    #     return test

    # @staticmethod
    # def test_func_y_core(n, y):
    #     test = Jacobi(n+1, 0, 0, y) - Jacobi(n-1, 0, 0, y)
    #     return test

    def v_x(self, n_test, x):
        test_total = [x**n for n in range(n_test + 1)]
        return np.asarray(test_total)

    def v_y(self, n_test, y):
        test_total = [y**n for n in range(n_test + 1)]
        return np.asarray(test_total)

    def dtest_func(self, n_test, x):
        n = 1
        d1test_total = [((n+2)/2)*Jacobi(n, 1, 1, x)]
        d2test_total = [((n+2)*(n+3)/(2*2))*Jacobi(n-1, 2, 2, x)]
        for n in range(2, n_test+1):
            if n == 2:
                d1test = ((n+2)/2)*Jacobi(n, 1, 1, x) - ((n)/2)*Jacobi(n-2, 1, 1, x)
                d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1, 2, 2, x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)
            elif n > 2:
                d1test = ((n+2)/2)*Jacobi(n, 1, 1, x) - ((n)/2)*Jacobi(n-2, 1, 1, x)
                d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1, 2, 2, x) - ((n)*(n+1)/(2*2))*Jacobi(n-3, 2, 2, x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)
            else:
                raise ValueError("Please check the value for 'n_test'")
        return np.asarray(d1test_total), np.asarray(d2test_total)
    
    
'''
Hyper-parameters: 
    scheme     = is either 'PINNs' or 'VPINNs'
    Net_layer  = the structure of fully connected network
    var_form   = the form of the variational formulation used in VPINNs
                    0, 1, 2: no, once, twice integration-by-parts
    N_el_x, N_el_y     = number of elements in x and y direction
    N_test_x, N_test_y = number of test functions in x and y direction
    N_quad     = number of quadrature points in each direction in each element
    N_bound    = number of boundary points in the boundary loss
    N_residual = number of residual points in PINNs
'''

pb = PROBDEF((2*np.pi, 2*np.pi), 10)
N_tests = 5
N_elements = [5, 5]
params = {'scheme': 'VPINNs',
            'NN_struct': [2] + [5] * 3 + [1],
            'var_form': 1,
            'n_elements': tuple(N_elements),
            'n_test': [N_elements[0]*[N_tests], N_elements[1]*[N_tests]],
            'n_quad': 50,
            'n_bound': 80,
            'n_residual': 100,
            'domain': ((-1, -1), (1, 1)),
            'Opt_Niter': 15000 + 1,
            'delta_test': 0.01}
