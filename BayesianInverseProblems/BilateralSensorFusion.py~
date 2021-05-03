import numpy as np
import matplotlib.pyplot as plt

# Exploring uncertainty in (x,y) position based on 2 noisy range-finding measurements (R_a,R_b)
# R_a = reported range from station A, R_b = reported range from station B
# The objective is to generate a probabilistic map over a search grid based on unreliable range reporting from two sources

# Prior PDFs over X and Y
phi = (lambda x,m,s: (1.0/np.sqrt(2.0*np.pi*s*s))*np.exp(-(x-m)**2/(2.0*s*s)))
f_x = (lambda x,H: (1.0/H))
f_y = (lambda y,V: (1.0/V))

# Likelihood PDFs over A and B
f_AB = (lambda A,B,H,V: 2.0*A*B*(H-1.0)/(H*H*V))
f_A_xy = (lambda A,x,y,sa: phi(A,np.sqrt(x**2+y**2),sa))
f_B_xy = (lambda B,x,y,H,sb: phi(B,np.sqrt((H-x)**2+y**2),sb))

# Posterior PDF over A and B given X=x and Y=y: P(Hypothesis|Data) = P(Data|Hypothesis)/P(Data)
f_xy_AB = (lambda A,B,H,V,x,y,sa,sb: f_A_xy(A,x,y,sa)*f_B_xy(B,x,y,H,sb)*f_x(x,H)*f_y(y,V)/f_AB(A,B,H,V)) # Bayesian Posterior

# Hyperparameters
H = 1000 # Grid width in meters
V = 2000 # Grid height in meters
X_actual = 327 # Ground truth location along x-axis (sanity check)
Y_actual = 981 # Ground truth location along y-axis (sanity check)
S_A = 5 # Standard deviation of sensor A's noise (based on empirical measurements of data standard deviation statistics)
S_B = 5 # Standard deviation of sensor B's noise (based on empirical measurements of data standard deviation statistics)

R_a_actual = np.sqrt(X_actual**2+Y_actual**2) # Ground truth distance to station A (sanity check)
R_b_actual = np.sqrt((H-X_actual)**2+Y_actual**2) # Ground truth distance to station B (sanity check)

R_a_measured = R_a_actual + np.random.normal(loc=0, scale=S_A, size=1) #simulate measurement noise
R_b_measured = R_a_actual + np.random.normal(loc=0, scale=S_B, size=1) #simulate measurement noise

# Grid resolution along each axis
N_points = 32

# Construct the grid
X, Y = np.mgrid[0:H:N_points*1j, 0:V:N_points*1j]
A, B = np.mgrid[0:(H*H+V*V):N_points*1j, 0:(H*H+V*V):N_points*1j]

plt.imshow(f_xy_AB(R_a_measured,R_b_measured,H,V,X,Y,S_A,S_B).T, 
	origin='lower', 
	cmap=plt.cm.gist_heat, 
	extent=[0,H,0,V], 
	aspect="auto", 
	interpolation='gaussian'
)
plt.xlabel("X Coordinate (meters)")
plt.ylabel("Y Coordinate (meters)")
plt.title("Probability Map of (x,y) Location")
plt.show()

# Print sanity checks
print("Actual X Coordinate: "+str(round(X_actual,2)))
print("Actual Y Coordinate: "+str(round(Y_actual,2)))


