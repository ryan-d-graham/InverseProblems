# InverseProblems
Bayesian and non-Bayesian Inverse Problems

DeepONetTuringInvert.jl:

This script demonstrates a use case of DeepONet, the most general neural operator architecture invented in 2019 in this paper https://arxiv.org/abs/1910.03193
using Julia's SciML ecosystem. 
In this example a nonlinear pendulum system is formulated and solved using DifferentialEquations.jl. 10,000 random solution trajectories with random initial conditions 
and associated random external forcing functions are recorded as data to train a surrogate model. The model learns the operator mapping the ICs and input
function to the solution trajectory of the ODE system.

The equations governing the toy example are:
x'(t) = v(t); x''(t) = v'(t) = -gsin(x(t)) - cv(t) + f(t, p)

The numbers g and c are fixed and not inferred. Instead, the parameters p of the function f(t, p)
are inferred such that what is being done is function-space Bayesian inference. Note that any combination of solution components may be used in the inference 
procudure. Inference uncertainty can be compared when using both position and velocity, position or velocity only. It is interesting to explore how the posterior
over functions is affected by eliminating or including, in this example, the angle or frequency of the pendulum wrt vertical. 

DeepONetInverseProb.jl:

This script showcases a non-Bayesian but instead a direct inverse problem approach using DeepONet. Instead of surrogatizing the forward problem and accelerating
Bayesian inference, the approach surrogatizes the inverse of the map from external forcing to the solution and provides the underlying external forcing directly
from the solution data as input. This approach loses the ability to quantify uncertainty but provides a way to do online inference and dynamical control. 
For use in dynamical control, simply provide a solution trajectory you would like the system to exhibit which is "within reach" and it will output the necessary 
external forcing required to achieve the desired trajectory.

SGRLD_odemodel.jl:

This script uses Stochastic Gradient Riemann Langevin Dynamics (SGRLD) with the Monge Riemannian metric tensor. See section 3.1 of https://arxiv.org/pdf/2303.05101.pdf
Using the Monge metric allows inference of multi-modal joint posterior densities over paramters unlike vanilla SGLD, which only explores the major posterior mode.
In this example, the parameter's full densities of an ODE model are learned from data. The downside of these SGRLD samplers is tuning the hyperparameters. 

SGRLD_polymodel.jl:

This script uses SGRLD to estimate the full joint distribution over parameters of a truncated Fourier Series for simple polynomial regression. 

SGRLD_dist.jl:

This script showcases the SGRLD's Monge Riemannian metric tensor's ability to infer the full joint (potentially multi-modal) posterior of an arbitrary energy function. 
Possible applications include learning the partition function of thermodynamic systems whose energy function is specified as the log likelihood. 

MarkovGrid.py:

This script sets up a framework to use pgmpy (probabilistic graphical models in python) for unrolling Dynamic Bayesian Networks with arbitrary markov lag orders. 

Create a set of variables with the "Names(chainName="Node", chainLength=3, subScript="")" function. Pass this list to any of the following functions:

MarkovChain(chainList, markovOrder=1) 
This function creates a Markov chain of arbitrary order (defaults to 1). 
Example: markov_lag = 1 with variables X and Y produces the DAG: X_t --> X_t+1; Y_t --> Y_t+1
Using a markov_lag = 2 for the same variables produces the DAG: X_t --> X_t+2; Y_t --> Y_t+2
Note that if you wish to include more than one markov_lag, you can bunch the edges from separate calls to 
MarkovChain function using a different markovOrder each time. 
Etc...

InterProcess(causeChainList, effectChainList, markovLag=0)
This function creates an edge between variables in the chain in the same way as MarkovChain does for each variable's other time-slices. 
Using [X, Y] in the causeChainList and [W, Z] in the effect chain list with markov_lag = 0 produces the DAG: X --> W; Y --> Z
If you wish to create a fully-connected DAG between [X, Y] and [W, Z], simply use the lists [X, Y, X, Y] and [W, Z, Z, W] as 
the cause and effect variables are treated elementwise in the list and not assuming a fully-connected DAG. 
Increasing the markov_lag will work analogously to the previous function. 
Example with markov_lag = 1 and [X, Y] --> [W, Z]: X_t --> W_t+1; Y_t --> Z_t+1

CommonPaths(nodeName, chainList, source=True)
This function generates either a multi-fork or multi-collider of the single variable in "nodeName" using the variables in chainList. 
Example: nodeName = ["X"] and chainList = ["Y", "Z"] with source=True produces the DAG: X_t --> Y_t, Z_t, otherwise it produces 
the DAG: Y_t, Z_t --> X_t

KernelIntegrals includes scripts that use Integrals.jl to integrate an arbitrary function (spline, NN, or analytical) against a Kernel such 
as a Green's function. The examples showcase a setup for integrating the 2D and 3D Poisson Equation in R2 and R3 (free space solution). 
Note that Integrals.jl has the ability, using a special coordinate transformation, to integrate from [-Inf, -Inf...] to [Inf, Inf...] etc. 
These functions can be used to generate data to train DeepONet to surrogatize the IntegralProblem, which is slow on its own. By training 
DeepONet to learn the map from the f(x, y) and f(x, y, z) to the u(x, y) and u(x, y, z) respectively, you can accelerate solves greatly 
by needing only one forward pass of DeepONet instead of having to call an integral solver for each point in the spatial domain.
