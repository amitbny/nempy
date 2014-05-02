"""
Nematic hydrodynamics in Darcy's approximation
Ingredients:Uniaxial/Biaxial nematic with isotropic/anisotropic derivatives
copyright: amitb@courant.nyu.edu, rjoy@imsc.res.in

"""
import numpy as np  
import scipy as sp  
import sympy as syp

import matplotlib as mpl        
import matplotlib.pyplot as plt 
from pylab import *             

import time                     

from pyddx.fd import fdsuite as fds

# finite difference grid
Nx, Ny  = 256, 256                                # grid points in x and y
N = Nx * Ny                                     # total grid points
hx, hy  = 1, 1                                  # grid spacing 
x, y = np.arange(Nx)*hx, np.arange(Ny)*hy       # 1d grid
xx, yy = np.meshgrid(x, y)                      # meshgrid

# 1d finite difference matrices
D2x = fds.cdif(Nx, 2, 3, hx)                    # 3-point Laplacian in x
D2y = fds.cdif(Ny, 2, 5, hy)                    # 3-point Laplacian in y
Ix = sp.sparse.identity(Nx)                     # sparse identity in x
Iy = sp.sparse.identity(Ny)                     # sparse identity in y

# 2d Laplacian assembly
D2 = sp.sparse.kron(D2x, Iy) + sp.sparse.kron(Ix, D2y)

# Ginzburg-Landau free energy parameters
A = -0.08                                       # A TrQ^2
B = -0.5                                        # B TrQ^3
C = 2.67                                        # C Tr Q^4
Eprime = 0                                      # Eprime Tr Q^6
L1 = 1.0                                        # L1 (\nabla_i Q_ij)^2
L2 = 0.0                                        # L2 (\nabla_i ?)    
Na = 5                                          # order parameter components

# time stepping parameters
Gamma = 0.1;                                    # mobility parameter
dt = 1;                                         # time step 
tsteps = 2e3;                                   # total time steps

# initial condition
a  = 0.1*np.random.randn(N, Na)                 # random order parameter
fx = np.zeros((N, Na))

start = time.clock()

# integration loop
for t in np.arange(0, tsteps, dt):

    # compute AA=A+CTrQ^2 
    AA  = Gamma*(A + C*np.square(a).sum(axis=1));
    
    # compute BB=B+6EprimeTrQ^3    
    BB  = Gamma*(B + Eprime*( sqrt(6.0)*np.square(a[:,0])*a[:,0] + 
          sqrt(27.0/2.0)*a[:,0]*(- 2.0*np.square(a[:,1]) - 
          2.0*np.square(a[:,2]) + np.square(a[:,3]) + np.square(a[:,4])) + 
          9.0/sqrt(2.0)*( 2.0*a[:,2]*a[:,3]*a[:,4] + a[:,1]*( 
          np.square(a[:,3]) - np.square(a[:,4])) )))
          
    # compute the derivatives       
    fa1 = Gamma*L1*D2.dot(a[:,0].T);
    fa2 = Gamma*L1*D2.dot(a[:,1].T);
    fa3 = Gamma*L1*D2.dot(a[:,2].T);
    fa4 = Gamma*L1*D2.dot(a[:,3].T);
    fa5 = Gamma*L1*D2.dot(a[:,4].T);
    
    # assemble local and nonlocal parts together
    fx[:,0] = fa1 - AA*a[:,0] - BB/sqrt(6.0)* \
              (np.square(a[:,0]) - np.square(a[:,1]) - np.square(a[:,2]) 
              + 0.5*(np.square(a[:,3]) + np.square(a[:,4])));

    fx[:,1] = fa2 - AA*a[:,1] - BB*(-sqrt(2.0/3.0)*a[:,0]*a[:,1]  
              + sqrt(1.0/8.0)*(np.square(a[:,3]) - np.square(a[:,4])));

    fx[:,2] = fa3 - AA*a[:,2] - BB*(-2.0/sqrt(6.0)*a[:,0]*a[:,2]  
              + sqrt(1.0/2.0)*a[:,3]*a[:,4]);
    
    fx[:,3] = fa4 - AA*a[:,3] - BB*(sqrt(1.0/6.0)*a[:,0]*a[:,3] 
              + sqrt(1.0/2.0)*(a[:,1]*a[:,3] + a[:,2]*a[:,4]));
    
    fx[:,4] = fa5 - AA*a[:,4] - BB*(sqrt(1.0/6.0)*a[:,0]*a[:,4] 
              + sqrt(1.0/2.0)*(a[:,2]*a[:,3] - a[:,1]*a[:,4]));
    
    # advance with an Euler timestep
    a = a + dt*fx
    
    # plot the order parameter
    if t % (tsteps/100) == 0:
        b = np.square(a).sum(axis=1)
        b = np.reshape(b, (Nx, Ny))    
        clf();
        pcolor(xx,yy,b), axis('equal'), axis('tight'), colorbar()
        pause(0.1)
        
    #print t
    
stop = time.clock()
print stop - start