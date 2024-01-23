using NeuralOperators, Flux, DiffEqFlux
using DifferentialEquations
using Plots, Distributions

# Solve inverse operator problem: 
# Forward Problem: G: f(x) → u(x) where G is a nonlinear map between function spaces
# Inverse Problem: G⁻¹: u(x) → f(x) where G⁻¹ is the inverse of the map G such that f(x) = G⁻¹{u(x)}(x)

# Data parameters
NumSolPts = 64 # number of points along which to evaluate solution input functions
NumTraj = 128 # number of solution trajectories to train on
SensorDim = 1 # dimension of trajectory function domain (1 for time in dynamical systems)
NumFrcPts = 64 # number of points along which to evaluate inverse operator for forcing function (output)

# Create Model
Model = DeepONet((NumSolPts, 64, 72), (SensorDim, 64, 72), σ, tanh) # Learn nonlinear Banach space map between function spaces

# create prototype inputs to DeepONet 
SolnPrototype = randn((NumSolPts, NumTraj))
SensorPrototype = randn((SensorDim, NumFrcPts))
Model(SolnPrototype, SensorPrototype)

# number of parameters for Fourier Basis forcing function
σp = 0.1
θDim = 4
θGT = σp * randn(θDim) # ground truth parameters
Force(x, θ) = θ' * FourierBasis(length(θ))(x)

# input/output function grids
t₀ = 0.0
t₁ = 10.0
SensorGrid = range(t₀, t₁, NumFrcPts)
SolutionGrid = range(t₀, t₁, NumSolPts)
# ODE Problem definition
function Pendulum!(du, u, p, t)
    du[1] = u[2]
    du[2] = -9.8 * sin(u[1]) - 0.5 * u[2] + Force(t, p)
end
# trajectory simulation initial conditions and parameters
u₀ = [0.0, 0.0]
tspan = (t₀, t₁)
prob = ODEProblem(Pendulum!, u₀, tspan, θGT)
sol = solve(prob, Tsit5())
GTSolution = sol(SensorGrid, idxs=2).u # ODE solution prototype
plot(SensorGrid, GTSolution, title="GT Solution", legend=false)

# allocate memory for data generation
SolTrajectories = zeros((NumSolPts, NumTraj))
FrcFunctions = zeros((NumTraj, NumFrcPts))
Sensor = reshape(collect(SensorGrid), (SensorDim, NumFrcPts))

# generate training data using ODE solver
for soln ∈ 1:NumTraj # iterate over number of functions to sample
    θ = σp * randn(θDim) # sample a random function 
    FrcFunctions[soln, :] = [force(x, θ) for x ∈ SensorGrid] # put the forcing function sample in the rows of FrcFunctions
    prob = remake(prob, p = θ) # remake a problem using new parameter sample
    sol = solve(prob, Tsit5()) # solve the problem with new parameter sample
    SolTrajectories[:, soln] = sol(SolutionGrid, idxs=2).u # store solution trajectory into columns of SolTrajectories
end

# evaluate model's ability to invert solution into forcing function
FrcPlot = plot(SensorGrid, FrcFunctions', alpha = 0.5, title = "Forcing", legend=false);
SolPlot = plot(SolutionGrid, SolTrajectories, alpha = 0.5, title = "Solutions", legend=false);
PredPlot = plot(SensorGrid, Model(SolTrajectories, Sensor)', alpha = 0.5, title = "Inverse Prediction", legend=false);
plot(FrcPlot, SolPlot, PredPlot)

# Loss function prototype
Loss(xtrain, ytrain, sensor) = Flux.Losses.mse(Model(xtrain, sensor), ytrain)

# training setup
# train neural operator to learn the inverse of the map from forcing function to solution trajectory
# the forward pass of the trained neural operator will infer the forcing function required to generate any given solution trajectory
LearningRate = 1e-3
opt = Adam(LearningRate)
Pars = Flux.params(Model)
Flux.@epochs 1000 Flux.train!(Loss, Pars, [(SolTrajectories, FrcFunctions, Sensor)], opt)
Loss(SolTrajectories, FrcFunctions, Sensor)

# Compare GT with predictions on finer grid to showcase generalization and discretization invariance capabilities
# Plot single predictions based on GT parameters
Refn = 2 # sensor grid refinement multiplier
SensorGridRefined = range(t₀, t₁, Refn * length(SensorGrid)) # grid for plotting
SensorRefined = reshape(collect(SensorGridRefined), (1, length(SensorGridRefined))) # grid for model
# model evaluation
FrcFunctionPred = Model(GTSolution, SensorRefined)'
plot(SensorGridRefined, FrcFunctionPred, color=:red, label = "F̂(t)");
plot!(SensorGridRefined, [Force(x, θGT) for x ∈ SensorGridRefined], color=:black, label = "F(t)", title = "GT(Black); Pred(Red)")
