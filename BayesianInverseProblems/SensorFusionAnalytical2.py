from numpy import pi as PI
from numpy import log as LOG
from numpy import exp as EXP
from numpy import cosh as COSH
from numpy import sqrt as ROOT
from numpy import sum as SUM
from numpy import mgrid as GRID
from numpy import max as MAX
from numpy import min as MIN
from numpy import argmax as ARGMAX
from numpy import linspace as LIN
from numpy import float64 as F64
from numpy.random import logistic as Logistic
import matplotlib.pyplot as plt
from sys import argv as ARGV

# Exploring uncertainty in (x,y) position based on 2 noisy range-finding measurements (A,B)
# A = reported range from "Station A", B = reported range from "Station B"
# The objective is to generate a probabilistic map over a search grid based on unreliable range reporting from two sources

# Parameters
R = 1 # Minimum distance measurable from Stations A and B
x_true = F64(347)
y_true = F64(540)
SD_A = F64(100)
SD_B = F64(100)
Nsamp = 25 # Too many samples causes blank plot to appear -- still don't know why
H = F64(1000) # Distance between locator beacons
V = F64(1000)
GridSize = 512

# Auxiliary functions
L = (lambda n, SD_A, SD_B: (n+1) * LOG( (PI**2 * H**2 * V)/(96 * (H-1) * SD_A * SD_B) ) )
ALPHA = (lambda x, y, A_k, SD_A: -2 * LOG(COSH( ( (PI * ROOT(3))/(3 * SD_A) ) * (A_k - ROOT(x**2 + y**2) ) ) ) )
BETA = (lambda x, y, B_k, SD_B: -2 * LOG(COSH( ( (PI * ROOT(3))/(3 * SD_A) ) * (B_k - ROOT( (x-H)**2 + y**2) ) ) ) )
GAMMA = (lambda A_k, B_k: - LOG(A_k * B_k) )

# Generate samples
A = Logistic(loc=ROOT(x_true**2 + y_true**2), scale=SD_A * ROOT(3) / PI, size=Nsamp)
B = Logistic(loc=ROOT( (H-x_true)**2 + y_true**2), scale=SD_B * ROOT(3) / PI, size=Nsamp)
A[A<R] = F64(R) # Drop samples that are less than R and replace them with R
B[B<R] = F64(R)

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
