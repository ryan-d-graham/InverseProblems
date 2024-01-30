# InverseProblems
Bayesian and non-Bayesian Inverse Problems

DeepONetTuringInvert.jl:

This script demonstrates a use case of DeepONet, the most general neural operator architecture invented in 2019 in this paper https://arxiv.org/abs/1910.03193
using Julia's SciML ecosystem. 
In this example a nonlinear pendulum system is formulated and solved using DifferentialEquations.jl. 
The equations governing the toy example are:
x'(t) = v(t); x''(t) = v'(t) = -gsin(x(t)) - cv(t) + f(t, p)

The numbers g and c are fixed and not inferred. Instead, the parameters p of the function f(t, p)
are inferred such that what is being done is function-space Bayesian inference. 

DeepONetInverseProb.jl:

This script showcases a non-Bayesian but instead a direct inverse problem approach using DeepONet. Instead of surrogatizing the forward problem and accelerating
Bayesian inference, the approach learns the inverse of the map from external forcing to the solution and provides the underlying external forcing directly
from the solution data as input. This approach loses the ability to quantify uncertainty but provides a way to do online inference and dynamical control. 
For use in dynamical control, simply provide a solution trajectory you would like the system to exhibit which is "within reach" and it will output the necessary 
external forcing required to achieve the desired trajectory.

SGRLD_odemodel.jl:

This script uses Stochastic Gradient Riemann Langevin Dynamics (SGRLD) with the Monge Riemannian metric tensor. See section 3.1 of https://arxiv.org/pdf/2303.05101.pdf
Using the Monge metric allows inference of multi-modal joint posterior densities over paramters unlike vanilla SGLD, which only explores the major posterior mode.
In this example, the parameter's full densities of an ODE model are learned from data. The downside of these SGRLD samplers is tuning the hyperparameters. 

SGRLD_polymodel.jl:

This script uses SGRLD to learn the parameters of a truncated Fourier Series for simple polynomial regression. 

SGRLD_dist.jl:

This script showcases the SGRLD's Monge Riemannian metric tensor's ability to infer the full (multi-modal) posterior of an arbitrary energy function. 
Possible applications include learning the partition function of thermodynamic systems whose energy function is specified as the log likelihood. 

MarkovGrid.py:

This script sets up a framework to use pgmpy (probabilistic graphical models in python) for unrolling Dynamic Bayesian Networks with arbitrary markov lag orders. 

Create a set of variables with the "Names" function. Pass this list to any of the following functions:
MarkovChain(chainList, markovOrder=1) creates a Markov chain of arbitrary order (defaults to 1 with 3 unrolled time-steps per variable). This function creates
an edge from the first time slice of each variable to the following time slices of the same variable and to the appropriate slices if markov lag is higher than 1. 
For example, markov_lag = 1 with variables X and Y produces the DAG: X1 --> X2 --> X3; Y1 --> Y2 --> Y3
Using a markov_lag of 2 for the same variables produces the DAG: X1 --> X2 --> X3 --> X4, X1 --> X3, X2 --> X4; Y1 --> Y2 --> Y3 --> Y4, Y1 --> Y3, Y2 --> Y4
Etc...
