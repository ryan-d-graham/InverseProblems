import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde # scientific/statistical python gaussian kernel density estimator

# Simulation parameters
Nsamp = 100
H = np.float64(1000.0)
V = np.float64(1000.0)
X_actual = np.float64(500.0)
Y_actual = np.float64(500.0)
S_a = np.float64(1.0)
S_b = np.float64(1.0)
gridSize = 32

def RangeSampler(N, x_actual, y_actual, sa, sb, H, V):
	R_a = np.random.normal(loc=np.sqrt(x_actual**2+y_actual**2), scale=sa, size=N)
	R_b = np.random.normal(loc=np.sqrt((H-x_actual)**2+y_actual**2), scale=sb, size=N)
	X_est = (H**2 + R_a**2)/(2.0*H) + np.random.normal(loc=0.0, scale=0.01, size=N) # Adding noise to regularize COV MATRIX
	Y_est = ((2*H)*R_a**2 - H**2 - R_b**2)/(2.0*H) + np.random.normal(loc=0.0, scale=0.01, size=N) # Deterministic estimate of y coordinate (plug into data likelihood)
	return R_a, R_b, X_est, Y_est


# Collect virtual range data
X_est, Y_est, R_a_est, R_b_est = RangeSampler(Nsamp, X_actual, Y_actual, S_a, S_b, H, V)
# Construct grid and get KDE image
R_a_grid, R_b_grid, X_grid, Y_grid = np.mgrid[1.0:np.sqrt(H**2+V**2):gridSize*1j, 1.0:np.sqrt(H**2+V**2):gridSize*1j, 1.0:H:gridSize*1j, 1.0:V:gridSize*1j]
gridPoints = np.vstack([X_grid.ravel(), Y_grid.ravel(), R_a_grid.ravel(), R_b_grid.ravel()])
Data = np.vstack([X_est, Y_est, R_a_est, R_b_est])
R_AB_kernel = kde(Data, bw_method='scott')
Q_XYAB = np.reshape(R_AB_kernel(gridPoints).T, X_grid.shape) #Q(X,Y,R_a,R_b)
# Compute posterior image Q(X,Y|R_a=r_a,R_b=r_b) = Q(R_a|X=x,Y=y)Q(R_b|X=x,Y=y)Q(X=x,Y=y)/Q(R_a,R_b)
Q_xy_AB = Q_XYAB[:,:,16,16]/np.sum(Q_XYAB[:,:,16,16], axis=(0,1)) # Using Bayes' Theorem

# Plot posterior image
plt.imshow(Q_xy_AB, cmap=plt.cm.gist_heat, extent=[0,H,0,V], aspect="auto", interpolation="gaussian")
plt.show()

