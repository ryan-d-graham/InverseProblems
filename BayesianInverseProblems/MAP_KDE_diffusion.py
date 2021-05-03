from kde_diffusion import kde1d, kde2d
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

N = 512

f = (lambda x: x*x*x/10.0) #The function to be reconstructed

X = np.random.normal(loc=0, scale=10.0, size=1000000) #Distribution P(X) of Cause variable, X
Y = np.random.normal(loc=f(X), scale=0.2, size=1000000) #Distribution P(Y) of Effect variable, Y: Y ~ f(X)

(density_x, grid_x, bw_x) = kde1d(X, n=N, limits=15) #Kernel Density Estimate of P(X) 
(density_y, grid_y, bw_y) = kde1d(Y, n=N, limits=15) #Kernel Density Estimate of P(Y)
(density_xy, grid_xy, bw_xy) = kde2d(X, Y, n=N, limits=15) #Kernel Density Estimate of P(X,Y)

density_x[density_x < 0] = 1e-07 #get rid of negative densities
density_y[density_y < 0] = 1e-07 #""
density_xy[density_xy < 0] = 1e-07 #""
#density_xy[:,740:] = 1e-07 #clean up artefacts on the boundaries (crude, explore other image filtering techniques like hessian, laplacian, gaussian or fft etc.)
#density_xy[:,:490] = 1e-15

#plt.plot(grid_x, density_x, color='blue') #plot marginal densities
#plt.show()
#plt.plot(grid_y, density_y, color='black')
#plt.show()


density_x_y = density_xy/np.sum(density_xy, axis=0) #Bayes' Theorem

means_x_y = [np.sum(grid_x*density_x_y[:][i], axis=0) for i in range(len(grid_x))] #Pointwise Posterior Mean Estimate (divide by N to get proper scaling)

plt.imshow(density_xy.T, origin='lower') #Plot density with origin in the lower region (mathematical convention)
plt.show()
plt.imshow(density_x_y.T, origin='lower') #""
plt.show()

plt.plot(grid_x, means_x_y, color='red') #Bayesian Mean of Posterior Estimate E[X|Y] = SUM{ X*P(X|Y) }
plt.plot(grid_x, f(grid_x), color='black') #Ground Truth
plt.xlim([-5,5]) #estimate seems only to be reliable within a subdomain  
plt.ylim([-5,5]) #keep range same as domain 
plt.show()

