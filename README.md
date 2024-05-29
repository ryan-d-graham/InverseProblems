Projects

DeepONetTuringInvert.jl

DeepONetTuringInvert.jl demonstrates the application of DeepONet to learn and generalize the dynamics of nonlinear systems. By training on data generated from a pendulum system, DeepONet provides a surrogate model that rapidly predicts system behavior, significantly accelerating Bayesian inference within Turing.

DeepONetInverseProb.jl

This script implements a non-Bayesian inverse problem approach using DeepONet, directly inferring the external forcing from solution data.

SGRLD_odemodel.jl

Uses Stochastic Gradient Riemann Langevin Dynamics (SGRLD) with the Monge metric for multi-modal posterior inference of an ODE model’s parameters.

MarkovGrid.py

Sets up a framework for dynamic Bayesian networks using pgmpy in Python, with flexible configurations for Markov chains and inter-process relationships.

KernelIntegrals.jl

KernelIntegrals.jl aims to implement a surrogate for the 2D Laplace operator by using kernel integrals to generate quadrature-based numerical integral data. This approach allows for efficient numerical approximations of integrals, facilitating faster and more accurate solutions to PDE problems.

Usage

Running DeepONetTuringInvert.jl

To train and use the DeepONet model with Turing:

julia DeepONetTuringInvert.jl

Example Output

The model will output a set of predicted trajectories for the system, demonstrating its ability to generalize from the training data:

Example output: [Predicted Trajectories]

Contributing

We welcome contributions! Please see our contributing guidelines for more details.

License

This project is licensed under the MIT License. See the LICENSE file for details.