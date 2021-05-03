import numpy as np
from scipy.stats import gaussian_kde as kde
from scipy.stats import entropy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# Exploring non-latent confounding in continuous, 
# linear/nonlinear (mixed) structural causal models 
# via kernel density estimation (kde)

# Mutual Information kernel


# Some functions in the SCM
f = (lambda x: np.pi*np.sinh(x))
g = (lambda x: x*x*x/np.pi)
h = (lambda x,y: x*y/10.0)

def MeasureSCM(n): # Establish SCM or measurement model
	m1 = np.random.normal(loc=0.0, scale=5.0, size=n)
	m2 = np.random.normal(loc=f(m1), scale=2.0, size=n)
	m3 = np.random.normal(loc=g(m2), scale=2.0, size=n)
	return m1, m2, m3
	
m1, m2, m3 = MeasureSCM(100) #get n measurements (z, x, y) from the SCM:
# X <-- X --> Y
# Attempt to show that P(X)P(Y) != P(X,Y), but that P(X|Z)P(Y|Z) = P(X,Y|Z)
# P(X|Z) = P(X,Z)/P(Z); P(Y|Z) = P(Y,Z)/P(Z); P(X,Z) = SUM{Y}[P(X,Y,Z)]; P(Y,Z) = SUM{X}[P(X,Y,Z)]
# P(X,Y,Z) is target of kernel density estimator

# Set up bounds
xmin = m2.min()
xmax = m2.max()
ymin = m3.min()
ymax = m3.max()
zmin = m1.min()
zmax = m1.max()

# Build the grid for the kernel and its image in 3D and perform kde on data
X, Y, Z = np.mgrid[xmin:xmax:16j, ymin:ymax:16j, zmin:zmax:16j]
positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
values = np.vstack([m2, m3, m1])
kernel = kde(values, bw_method='scott')
Q_xyz = np.reshape(kernel(positions).T, X.shape) # Q(X,Y,Z)

# Create marginals
P_xyz = Q_xyz / np.sum(Q_xyz, axis=(0,1,2)) # Normalize Q to P(X,Y,Z)
P_xy = np.sum(P_xyz, axis=(2))
P_xz = np.sum(P_xyz, axis=(1))
P_yz = np.sum(P_xyz, axis=(0))
P_x = np.sum(P_xyz, axis=(1,2))
P_y = np.sum(P_xyz, axis=(0,2))
P_z = np.sum(P_xyz, axis=(0,1))

# Create conditionals via Bayes' Theorem
P_xy_z = P_xyz/P_z
P_x_z = P_xz/P_z
P_y_z = P_yz/P_z

# Check Independence via Mutual Information

fig = plt.figure()

plt.imshow(P_xy,origin='lower')
plt.show()
plt.imshow(P_xz,origin='lower')
plt.show()
plt.imshow(P_yz,origin='lower')
plt.show()

