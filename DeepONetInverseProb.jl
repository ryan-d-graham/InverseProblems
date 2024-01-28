using NeuralOperators, Flux, DiffEqFlux
using DifferentialEquations
using Plots, Distributions

# Solve inverse operator problem: 
# Forward Problem: G: f(x) → u(x) where G is a nonlinear map between function spaces
# Inverse Problem: G⁻¹: u(x) → f(x) where G⁻¹ is the inverse of the map G such that f(x) = G⁻¹{u(x)}(x)

# Data parameters
NumSolPts = 16 # number of points along which to evaluate solution input functions
NumTraj = 10000 # number of solution trajectories to train on
SensorDim = 4 # dimension of trajectory function domain (1 for time in dynamical systems)
NumFrcPts = 16 # number of points along which to evaluate inverse operator for forcing function (output)

# Create Model
Branch = Chain(x -> Flux.flatten(x), Dense(SensorDim*NumSolPts => 64, elu), Dense(64 => 72))
Trunk = Chain(Dense(1 => 32, gelu), Dense(32 => 64, gelu), Dense(64 => 72))
Model = DeepONet(Branch, Trunk) # Learn nonlinear Banach space map between function spaces

# create prototype inputs to DeepONet 
SolnPrototype = randn((NumSolPts, 1, SensorDim, NumTraj))
SensorPrototype = randn((1, NumFrcPts))
Model(SolnPrototype, SensorPrototype)

# number of parameters for Fourier Basis forcing function
σp = 0.5
θDim = 16
θGT = σp * randn(θDim) # ground truth parameters
Force(x, θ) = θ' * FourierBasis(length(θ))(x)

# input/output function grids
t₀ = 0.0
t₁ = 1.0
SensorGrid = range(t₀, t₁, NumFrcPts)
SolutionGrid = range(t₀, t₁, NumSolPts)
# ODE Problem definition
function Pendulum!(du, u, p, t)
    du[1] = u[2]
    du[2] = -9.8 * sin(u[1]) - 0.5 * u[2] + Force(t, p)
end
# trajectory simulation initial conditions and parameters along with sanity checks
x₀ = 1.0 * π * rand(1)[1] - 0.5 * π
v₀ = 1.0 * rand(1)[1] - 0.5
u₀ = [x₀, v₀]
tspan = (t₀, t₁)
prob = ODEProblem(Pendulum!, u₀, tspan, θGT)
sol = solve(prob, Tsit5())
solvect = Array(sol(SensorGrid))
# allocate memory for data generation
SolTrajectories = zeros((NumSolPts, 1, SensorDim, NumTraj))
FrcFunctions = zeros((NumTraj, NumFrcPts))
u₁s = FourierBasis(NumFrcPts)(u₀[1])
u₂s = FourierBasis(NumFrcPts)(u₀[2])
GTSolution = zeros(NumFrcPts, 1, SensorDim, 1)
GTSolution[:, 1, 1, 1] = solvect[1, :]
GTSolution[:, 1, 2, 1] = solvect[2, :]
GTSolution[:, 1, 3, 1] = u₁s
GTSolution[:, 1, 4, 1] = u₂s
GTSolution
Sensor = reshape(collect(SensorGrid), (1, NumFrcPts))

# generate training data using ODE solver
for soln ∈ 1:NumTraj # iterate over number of functions to sample
    θ = σp * randn(θDim) # sample a random function 
    FrcFunctions[soln, :] = [Force(x, θ) for x ∈ SensorGrid] # put the forcing function sample in the rows of FrcFunctions
    x₀ = 1.5 * π * rand(1)[1] - 0.75 * π
    v₀ = 1.0 * rand(1)[1] - 0.5
    u₀ = [x₀, v₀]
    prob = remake(prob, u0 = u₀, p = θ) # remake a problem using new parameter sample
    sol = solve(prob, Tsit5()) # solve the problem with new parameter sample
    solvect = Array(sol(SolutionGrid))
    SolTrajectories[:, 1, 1, soln] = sol(SolutionGrid, idxs=1).u # store solution trajectory into columns of SolTrajectories
    SolTrajectories[:, 1, 2, soln] = sol(SolutionGrid, idxs=2).u
    SolTrajectories[:, 1, 3, soln] = FourierBasis(NumFrcPts)(x₀)
    SolTrajectories[:, 1, 4, soln] = FourierBasis(NumFrcPts)(v₀)
end

# evaluate model's ability to invert solution into forcing function
FrcPlot = plot(SensorGrid, FrcFunctions', alpha = 0.01, title = "Forcing", legend=false);
SolPlot = plot(SolutionGrid, SolTrajectories[:, 1, 1, :], alpha = 0.01, title = "Solutions", legend=false);
PredPlot = plot(SensorGrid, Model(SolTrajectories, Sensor)', alpha = 0.01, title = "Inverse Prediction", legend=false);
plot(FrcPlot, SolPlot, PredPlot)

# Loss function prototype
Loss(xtrain, ytrain, sensor) = Flux.Losses.mse(Model(xtrain, sensor), ytrain)

# training setup
# train neural operator to learn the inverse of the map from forcing function to solution trajectory
# the forward pass of the trained neural operator will infer the forcing function required to generate any given solution trajectory
LearningRate = 1e-4
opt = Adam(LearningRate)
Pars = Flux.params(Model)
Flux.@epochs 1000 Flux.train!(Loss, Pars, [(SolTrajectories, FrcFunctions, Sensor)], opt)
Loss(SolTrajectories, FrcFunctions, Sensor)

# Compare GT with predictions on finer grid to showcase generalization and discretization invariance capabilities
# Plot single predictions based on GT parameters
Refn = 4 # sensor grid refinement multiplier
SensorGridRefined = range(t₀, t₁, Refn * length(SensorGrid)) # grid for plotting
SensorRefined = reshape(collect(SensorGridRefined), (1, length(SensorGridRefined))) # grid for model
# model evaluation
FrcFunctionPred = Model(GTSolution, SensorRefined)'
plot(SensorGridRefined, FrcFunctionPred, color=:red, label = "F̂(t)");
plot!(SensorGridRefined, [Force(x, θGT) for x ∈ SensorGridRefined], color=:black, label = "F(t)", title = "GT(Black); Pred(Red)")
