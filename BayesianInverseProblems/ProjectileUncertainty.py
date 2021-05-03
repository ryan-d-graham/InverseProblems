import numpy as np # numerical python
from scipy.stats import gaussian_kde as kde # scientific/statistical python gaussian kernel density estimator
import matplotlib.pyplot as plt # visualization tools

# Quantifying Uncertainty in a simple Projectile Motion Model
# Instead of plugging uniform densities in directly for unknowns, I decided to include the data in the estimation routine
# to develop the prior probability distributions for the unknowns "empirically".
# While technically we are assuming we don't have access to V and U, it is more convenient to let the KDE generate our distributions
# with 'virtual data' so that the density over the entire joint space is available to the user a la Monte Carlo simulation
# The virtual data is pushed through the physical process and the image of the priors is picked up by the sensors

# Notation: Q is density, P is probability (normalized density)

# Graphical Causal Model representing physical dependencies (causal mechanisms):
# Vx <-- V --> Vy; Vx <-- U --> Vy

# Physical model: Vx = V*COS(U); Vy = V*SIN(U) - g*t, where V is muzzle speed in m/s and U is inclination angle in radians
# -g*t term represents vertical velocity slowing, halting, then reversing due to the force of gravity

# Parameters:
#Number of grid points along each axis
N = 64 # Grid points along each observable's represented axis in the kernel density object
n = 250 # Samples
Sx = 1.0 # SD in Vx measurement
Sy = 1.0 # SD in Vy measurement
g = 9.8 # Gravity accel
t = 0.0 # Time after launch for sensor capture

# Take noisy measurements of vertical Vy and horizontal Vx velocities with imperfect sensors; densities are determined empirically
def measurementModel(ns, sx, sy, g, t):
	V = np.random.uniform(0.0, 100.0, ns) # Muzzle velocity uniformly distributed on (0,100) meters/sec
	U = np.random.uniform(0.0, np.pi/2.0, ns) # Cannon inclination angle uniformly distributed on (0, pi/2) radians
	#V = np.random.normal(loc=m_v, scale=1.0, size=ns) # Muzzle velocity normally distributed with mean 100 m/s and unit variance
	#U = np.random.normal(loc=m_u, scale=1.0, size=ns) # Cannn inclination angle normally distributed with mean pi/4 and unit variance
	Vx = V*np.cos(U) + np.random.normal(loc=0.0, scale=sx, size=ns) # Deterministic relationship with additive gaussian sensor noise 
	Vy = V*np.sin(U) - g*t + np.random.normal(loc=0.0, scale=sy, size=ns) # Gaussian noise represents MEASUREMENT uncertainty
	return V, U, Vx, Vy

# Parameters: (number of observations, Vx sensor noise SD, Vy sensor noise SD, accel. due to gravity, time after launch)
v, u, vx, vy = measurementModel(ns=n, sx=Sx, sy=Sy, g=g, t=t) # Take n measurements of the velocities t seconds after firing in Earth gravity (9.8m/s/s)

# Get bounds for grid construction
vmin = v.min()
vmax = v.max()
umin = u.min()
umax = u.max()
vxmin = vx.min()
vxmax = vx.max()
vymin = vy.min()
vymax = vy.max()

# Generate grid and values for kernel density estimation algorithm
V, U, Vx, Vy = np.mgrid[vmin:vmax:N*1j, umin:umax:N*1j, vxmin:vxmax:N*1j, vymin:vymax:N*1j] # mgrid generates 4D space of points for evaluating kernel
positions = np.vstack([V.ravel(), U.ravel(), Vx.ravel(), Vy.ravel()]) # vertically stack the 4D points for KDE
values = np.vstack([v, u, vx, vy]) # vertically stack the 4D values in the same manner
kernel = kde(values, bw_method='scott') # compute kernel density estimate 
Q_vuxy = np.reshape(kernel(positions).T, V.shape) # unpack the values along the positions, take transpose (.T), retain shape of V (any of the 4 will do)

# Marginalizing over some vars to compute Bayesian updates
Q_vxy = np.sum(Q_vuxy, axis=1) # Q(V,Vx,Vy)
Q_uxy = np.sum(Q_vuxy, axis=0) # Q(U,Vx,Vy)
Q_vu = np.sum(Q_vuxy, axis=(2,3)) # Q(V,U) etc...
Q_xy = np.sum(Q_vuxy, axis=(0,1))
Q_ux = np.sum(Q_vuxy, axis=(0,3))
Q_x = np.sum(Q_vuxy, axis=(0,1,3))
Q_y = np.sum(Q_vuxy, axis=(0,1,2))
Q_v = np.sum(Q_vuxy, axis=(1,2,3))
Q_u = np.sum(Q_vuxy, axis=(0,2,3))

# Conditioning on Vx, Vy (Bayes' Theorem: P(A|B) = P(A,B)/P(B))
Q_vu_xy = Q_vuxy[:,:,23,23]/Q_xy[23,23] # Conditional densities; values in brackets are values on which conditional densities depend
Q_u_vxy = Q_vuxy[31,:,23,23]/Q_vxy[31,23,23] # Eg. Q(U|V=v,Vx=vx,Vy=vy) = Q(V=v,U,Vx=vx,Vy=vy)/Q(V=v,Vx=vx,Vy=vy)
Q_v_uxy = Q_vuxy[:,31,23,23]/Q_uxy[31,23,23] # Q(V|U,Vx,Vy)

# Display joint density over (V,U) and conditional joint density (V,U)|(Vx,Vy); read "|" as "conditional on" or "given"
plt.imshow(Q_vu, cmap=plt.cm.gist_heat, extent=[0,100,0,np.pi/2.0], aspect="auto", interpolation='gaussian')
plt.show()
plt.imshow(Q_vu_xy, cmap=plt.cm.gist_heat, extent=[0,100,0,np.pi/2.0], aspect="auto", interpolation='gaussian')
plt.show()
#plt.imshow(Q_ux, origin='lower')
#plt.show()

# Expectation Values (EV_X := SUM{X}[ X*P(X) ] and EV_X_Y := SUM{X}[ X*P(X|Y=y) ]) etc... Used to get point estimate from probabilities
E_u_vxy = np.sum(np.linspace(0,np.pi/2.0,N)*Q_u_vxy, axis=0)/np.sum(Q_u_vxy) # E[U|V=v,Vx=vx,Vy=vy]
E_v_uxy = np.sum(np.linspace(0,100,N)*Q_v_uxy, axis=0)/np.sum(Q_v_uxy)

# Plot 1D marginal and conditional distributions
"""plt.plot(np.linspace(0,100,N),Q_x) # Q(X)
plt.xlabel("Vx")
plt.ylabel("Q(Vx)")
plt.show()
plt.plot(np.linspace(5,260,N),Q_y) # Q(Y)
plt.xlabel("Vy")
plt.ylabel("Q(Vy)")
plt.show()"""
plt.plot(np.linspace(0,np.pi/2.0, N), Q_u/np.sum(Q_u, axis=0), color='black') # Q(U)
plt.plot(np.linspace(0,np.pi/2.0, N), Q_u_vxy, color='red') # Q(U|V,Vx,Vy)
plt.xlabel("U: Cannon inclination angle (radians)")
plt.ylabel("Q(U) & Q(U|V,Vx,Vy)")
plt.show()
plt.plot(np.linspace(0,100,N), Q_v/np.sum(Q_v, axis=0), color='black') # Q(V)
plt.plot(np.linspace(0,100,N), Q_v_uxy, color='red') # Q(V|U,Vx,Vy)
plt.xlabel("V: Muzzle velocity m/s")
plt.ylabel("Q(V) & Q(V|U,Vx,Vy)")
plt.show()

# Print Expectation Values (sanity check)
print("E[U|V,Vx,Vy] = "+str(E_u_vxy)+" radians")
print("E[V|U,Vx,Vy] = "+str(E_v_uxy)+" m/s")


