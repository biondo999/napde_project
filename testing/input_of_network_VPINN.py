import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyDOE import lhs
from GaussJacobiQuadRule_V3 import Jacobi, DJacobi, GaussLobattoJacobiWeights, GaussJacobiWeights
import time

# from tensorflow import placeholder

np.random.seed(1234)
tf.set_random_seed(1234)

from utils_1d_poisson import *


if __name__ == "__main__":                       # a(u,v)=f(v) u in H1 such that is 0 on diriclet boundary for every v in V -->ah(u,vh)=fh(vh) for every vh in Vh

    
    #++++++++++++++++++++++++++++
    LR = 0.001
    Opt_Niter = 1000 + 1
    Opt_tresh = 2e-32
    var_form  = 1
    N_Element = 1
    Net_layer = [1] + [20] * 4 + [1] # [1 20 20 20 20 1] number of neurons in each later numer of param == (input_layer+1)*output_layer in this case 2*20+3*(21*20)+21*1=1321 total param
    N_testfcn = 60 #number of test function
    N_Quad = 80    #quadrature points 
    N_F = 500      #number of points used for trianing 
    lossb_weight = 1

    params = {'var_form': var_form, 'lossb_weight': lossb_weight, 'LR': LR}
        
    #++++++++++++++++++++++++++++    valuete Test function of order n in a point x(n>=1)
    def Test_fcn(n,x):
       test  = Jacobi(n+1,0,0,x) - Jacobi(n-1,0,0,x)
       return test

    #++++++++++++++++++++++++++++    
    #exact sol and forcing term
    omega = 8*np.pi
    amp = 1
    r1 = 80
    def u_ext(x):
        utemp = 0.1*np.sin(omega*x) + np.tanh(r1*x)
        return amp*utemp

    def f_ext(x):
        gtemp =  -0.1*(omega**2)*np.sin(omega*x) - (2*r1**2)*(np.tanh(r1*x))/((np.cosh(r1*x))**2)
        return -amp*gtemp

    #++++++++++++++++++++++++++++ generate once for all quad nodes and weights in the interval [-1,1]
    NQ_u = N_Quad
    [x_quad, w_quad] = GaussLobattoJacobiWeights(NQ_u, 0, 0)
    testfcn = np.asarray([ Test_fcn(n,x_quad)  for n in range(1, N_testfcn+1)])
    
    #generate grid elements,for each element the number of test fuction is the same  
    NE = N_Element
    [x_l, x_r] = [-1, 1]
    delta_x = (x_r - x_l)/NE
    grid = np.asarray([ x_l + i*delta_x for i in range(NE+1)])
    N_testfcn_total = np.array((len(grid)-1)*[N_testfcn])
 
    if N_Element == 3:
        grid = np.array([-1, -0.1, 0.1, 1])
        NE = len(grid)-1
        N_testfcn_total = np.array([N_testfcn,N_testfcn,N_testfcn])
    
    U_ext_total = []
    F_ext_total = []
    for e in range(NE):
        x_quad_element = grid[e] + (grid[e+1]-grid[e])/2*(x_quad+1)
        jacobian = (grid[e+1]-grid[e])/2
        N_testfcn_temp = N_testfcn_total[e]
        testfcn_element = np.asarray([ Test_fcn(n,x_quad)  for n in range(1, N_testfcn_temp+1)])
        
        #is this my left side ot the variational form ah calculated in the real nodal values ?  
        u_quad_element = u_ext(x_quad_element)
        U_ext_element  = jacobian*np.asarray([sum(w_quad*u_quad_element*testfcn_element[i]) for i in range(N_testfcn_temp)])
        U_ext_element = U_ext_element[:,None]
        U_ext_total.append(U_ext_element)
        #this is fh calculated on the real nodal values summed over each real nodal values(which comes form CGL nodes )
        f_quad_element = f_ext(x_quad_element)
        F_ext_element  = jacobian*np.asarray([sum(w_quad*f_quad_element*testfcn_element[i]) for i in range(N_testfcn_temp)])
        F_ext_element = F_ext_element[:,None]
        F_ext_total.append(F_ext_element)
    
    # at the end i have an array with my all my "real" residues
    U_ext_total = np.asarray(U_ext_total)
    F_ext_total = np.asarray(F_ext_total)

    #++++++++++++++++++++++++++++
    # Training points
    X_u_train = np.asarray([-1.0,1.0])[:,None]
    u_train   = u_ext(X_u_train)
    X_bound = np.asarray([-1.0,1.0])[:,None]
    
    Nf = N_F
    X_f_train = (2*lhs(1,Nf)-1) #generate random number of training points 
    f_train   = f_ext(X_f_train) #evaluete these points on f(forcing therm)

    #++++++++++++++++++++++++++++
    # Quadrature points
    [x_quad, w_quad] = GaussLobattoJacobiWeights(N_Quad, 0, 0)

    X_quad_train = x_quad[:,None] #[[w1],[w2]] none stands for new axis
    W_quad_train = w_quad[:,None]

    #++++++++++++++++++++++++++++
    # Test point
    delta_test = 0.001
    xtest      = np.arange(-1 , 1 + delta_test , delta_test)  #linspace
    data_temp  = np.asarray([ [xtest[i],u_ext(xtest[i])] for i in range(len(xtest))]) # pair input real ouput 
    X_test = data_temp.flatten()[0::2]
    u_test = data_temp.flatten()[1::2]
    #test values 
    X_test = X_test[:,None] 
    u_test = u_test[:,None] 
    f_test = f_ext(X_test)

    u_test_total = []
    for e in range(NE):
        x_test_element = grid[e] + (grid[e+1]-grid[e])/2*(xtest+1)
        u_test_element = u_ext(x_test_element)
        u_test_element = u_test_element[:,None]
        u_test_total.append(u_test_element)

    #++++++++++++++++++++++++++++
    # Model and Training
    model = VPINN(X_u_train, u_train, X_quad_train, W_quad_train, F_ext_total,\
                  grid, X_test, u_test, Net_layer, X_f_train, f_train, params=params)
                  #X_u_train,u_train,               points for training at the boundary 
                  #X_quad_train, W_quad_train,      quadrature weights and points in [-1,1](N_quad is their number) arrays
                  #F_ext_total,                     fh calculated on each elem for each test function on the real nodes array
                  #grid,                            lispace grid between -1 and 1 with N_quad number of points points 
                  #X_test,u_test     		    test points between -1 and 1 depends on delta_test and uex evalueted on these
                  #Net_layer         		    array of integer with number of neurons for each layer
                  #X_f_train,                       Nf points for trianing,then evalueted on f
                  #f_train,                         
                  #params=params                    hyperparameters of the network/loss options 
  
    total_record = model.train(Opt_Niter, Opt_tresh, [])
    u_pred = model.predict(X_test)
    
    
    
    
    
    #dont care for the moment
    # =========================================================================
    #     Plotting
    # =========================================================================    
    x_quad_plot = X_quad_train
    y_quad_plot = np.empty(len(x_quad_plot))
    y_quad_plot.fill(1)
    
    x_train_plot = X_u_train
    y_train_plot = np.empty(len(x_train_plot))
    y_train_plot.fill(1) 
    
    x_f_plot = X_f_train
    y_f_plot = np.empty(len(x_f_plot))
    y_f_plot.fill(1)
    
    fig = plt.figure(0)
    gridspec.GridSpec(3,1)
    
    plt.subplot2grid((3,1), (0,0))
    plt.tight_layout()
    plt.locator_params(axis='x', nbins=6)
    plt.yticks([])
    plt.title('$Quadrature \,\, Points$')
    plt.xlabel('$x$')
    plt.axhline(1, linewidth=1, linestyle='-', color='red')
    plt.axvline(-1, linewidth=1, linestyle='--', color='red')
    plt.axvline(1, linewidth=1, linestyle='--', color='red')
    plt.scatter(x_quad_plot,y_quad_plot, color='green')
    
    plt.subplot2grid((3,1), (1,0))
    plt.tight_layout()
    plt.locator_params(axis='x', nbins=6)
    plt.yticks([])
    plt.title('$Training \,\, Points$')
    plt.xlabel('$x$')
    plt.axhline(1, linewidth=1, linestyle='-', color='red')
    plt.axvline(-1, linewidth=1, linestyle='--', color='red')
    plt.axvline(1, linewidth=1, linestyle='--', color='red')
    plt.scatter(x_train_plot,y_train_plot, color='blue')

    fig.tight_layout()
    fig.set_size_inches(w=10,h=7)
    plt.show()
    # plt.savefig('Train-Quad-pnts.pdf')    
    #++++++++++++++++++++++++++++

    font = 24

    fig, ax = plt.subplots()
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$', fontsize = font)
    plt.ylabel('$loss \,\, values$', fontsize = font)
    plt.yscale('log')
    plt.grid(True)
    iteration = [total_record[i][0] for i in range(len(total_record))]
    loss_his  = [total_record[i][1] for i in range(len(total_record))]
    plt.plot(iteration, loss_his, 'gray')
    plt.tick_params( labelsize = 20)
    fig.set_size_inches(w=11,h=5.5)
    plt.show()
    # plt.savefig('loss.pdf')
    #++++++++++++++++++++++++++++

    pnt_skip = 25
    fig, ax = plt.subplots()
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=8)
    plt.xlabel('$x$', fontsize = font)
    plt.ylabel('$u$', fontsize = font)
    plt.axhline(0, linewidth=0.8, linestyle='-', color='gray')
    for xc in grid:
        plt.axvline(x=xc, linewidth=2, ls = '--')
    plt.plot(X_test, u_test, linewidth=1, color='r', label=''.join(['$exact$']))
    plt.plot(X_test[0::pnt_skip], u_pred[0::pnt_skip], 'k*', label='$VPINN$')
    plt.tick_params( labelsize = 20)
    legend = plt.legend(shadow=True, loc='upper left', fontsize=18, ncol = 1)
    fig.set_size_inches(w=11,h=5.5)
    plt.show()
    # plt.savefig('prediction.pdf')
    #++++++++++++++++++++++++++++

    fig, ax = plt.subplots()
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=8)
    plt.xlabel('$x$', fontsize = font)
    plt.ylabel('point-wise error', fontsize = font)
    plt.yscale('log')
    plt.axhline(0, linewidth=0.8, linestyle='-', color='gray')
    for xc in grid:
        plt.axvline(x=xc, linewidth=2, ls = '--')
    plt.plot(X_test, abs(u_test - u_pred), 'k')
    plt.tick_params( labelsize = 20)
    fig.set_size_inches(w=11,h=5.5)
    plt.show()
    # plt.savefig('error.pdf')
    #++++++++++++++++++++++++++++
