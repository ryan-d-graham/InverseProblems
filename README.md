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

SGRLD_dist.jl

This script showcases the SGRLD's Monge Riemannian metric tensor's ability to infer the full (multi-modal) posterior of an arbitrary energy function. 
Possible applications include learning the partition function of thermodynamic systems whose energy function is specified as the log likelihood. 
