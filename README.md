# InverseProblems

## Overview
This repository leverages DeepONet to learn Banach-space mappings for inverse problems, accelerating Bayesian inference within probabilistic programming languages (PPL) such as Turing. By generalizing dynamics from training data, DeepONet provides a fast surrogate model, reducing the computational load typically associated with ODE/PDE solvers.

## Table of Contents
1. [Introduction](#introduction)
2. [Setup and Installation](#setup-and-installation)
3. [Projects](#projects)
   - [DeepONetTuringInvert.jl](#deeponetturinginvertjl)
   - [DeepONetInverseProb.jl](#deeponetinverseprobjl)
   - [SGRLD_odemodel.jl](#sgrld_odemodeljl)
   - [MarkovGrid.py](#markovgridpy)
   - [KernelIntegrals.jl](#kernelintegralsjl)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)

## Introduction
Inverse problems involve estimating unknown parameters or inputs from observed data. This repository contains several scripts that showcase different approaches to solving these problems using Julia and Python. The acceleration in forward evaluation is used to accelerate Bayesian inference within a PPL like Turing.

## Setup and Installation

### Requirements
- Julia 1.x
- Python 3.x

### Installation
```sh
# Clone the repository
git clone https://github.com/ryan-d-graham/InverseProblems.git

# Navigate to the directory
cd InverseProblems

# Install Julia packages
julia -e 'using Pkg; Pkg.instantiate()'

# Install Python packages
pip install -r requirements.txt

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