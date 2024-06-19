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
6. [LICENSE](#license)

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
