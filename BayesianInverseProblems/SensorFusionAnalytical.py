from numpy import pi as PI
from numpy import log as LOG
from numpy import exp as EXP
from numpy import sqrt as ROOT
from numpy import sum as SUM
from numpy import mgrid as GRID
from numpy import max as MAX
from numpy import min as MIN
from numpy import argmax as ARGMAX
from numpy import linspace as LIN
from numpy import float64 as F64
from numpy.random import normal as Normal
import matplotlib.pyplot as plt

# Exploring uncertainty in (x,y) position based on 2 noisy range-finding measurements (A,B)
# A = reported range from "Station A", B = reported range from "Station B"
# The objective is to generate a probabilistic map over a search grid based on unreliable range reporting from two sources

# Parameters
x_true = F64(3470)
y_true = F64(5400)
SD_A = F64(500)
SD_B = F64(500)
Nsamp = 30 # Too many samples causes blank plot to appear -- still don't know why
H = F64(10000) # Distance between locator beacons
V = F64(10000)
GridSize = 256

# Auxiliary functions
L = (lambda n, SD_A, SD_B: ( (n + 1) * LOG( (H**2 * V) / (4.0 * PI * (H - 1) * SD_A * SD_B) ) - LOG(H * V) ) )
ALPHA = (lambda x, y, A_k, SD_A: (-1.0 / (2.0 * SD_A**2) ) * (A_k - ROOT(x**2 + y**2) )**2 )
BETA = (lambda x, y, B_k, SD_B: (-1.0 / (2.0 * SD_B**2) ) * (B_k - ROOT((H - x)**2 + y**2) )**2 )
GAMMA = (lambda A_k, B_k: - LOG(A_k * B_k) )

# Generate samples
A = Normal(loc=ROOT(x_true**2 + y_true**2), scale=SD_A, size=Nsamp)
B = Normal(loc=ROOT((H-x_true)**2 + y_true**2), scale=SD_B, size=Nsamp)

# Log Posterior
LogPDF = (lambda x, y, n, A, B, SD_A, SD_B: L(n, SD_A, SD_B) + SUM( [ALPHA(x, y, A[k], SD_A) + BETA(x, y, B[k], SD_B) + GAMMA(A[k], B[k]) for k in range(Nsamp)], axis=0) )

# Create X, Y grid
Xgrid, Ygrid = GRID[ 0:H:GridSize*1j, 0:V:GridSize*1j ]

# Evaluate
LogProb = LogPDF(Xgrid, Ygrid, Nsamp, A, B, SD_A, SD_B)
Prob = EXP(LogProb)

# Plot result
plt.imshow(Prob.T, origin='lower', cmap=plt.cm.gist_heat, extent=[0,H,0,V], aspect='auto', interpolation='gaussian')
plt.title("(X,Y) Coordinate Probability Map")
plt.xlabel("X Coordinate (meters)")
plt.ylabel("Y Coordinate (meters)")
plt.show()



