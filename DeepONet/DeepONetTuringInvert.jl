using NeuralOperators, Flux, DiffEqFlux
using DifferentialEquations
using LinearAlgebra: I
using Plots, Distributions
using Turing, StatsPlots

# use a nn to approximate an operator, eg. an ODE/PDE solver, Nonlinear Solver, Optimization problem etc
# DeepONet approximates a Banach space map using a discretized input function and sensor locations at which to evaluate the operator
# Architecture: 
# Model{f(x)}(ξ) = Branch{f(x)}ᵀ Trunk{ξ}, where f(x) is the discretized input function and ξ are the coordinates at which to evaluate operator
FuncDim = 2+32 # ICs + number of discrete points to approximate input function
SensorDim = 1 # number of dimensions in input domain of neural operator's output function
Model₁ = DeepONet((FuncDim, 32, 72), (SensorDim, 24, 72), σ, tanh) # Learn nonlinear Banach space map between function spaces
Model₂ = DeepONet((FuncDim, 32, 72), (SensorDim, 24, 72), σ, tanh)

NumFuncObs = 10000 # Number of randomly-sampled input functions to use for training
NumSensors = 32 # Number of discrete points at which to evaluate output functions

# create prototype inputs to neural operator model (this forces precompilation and speeds up later evals)
FuncTrain = randn((FuncDim, NumFuncObs))
Sensor = randn((SensorDim, NumSensors))
# evaluate model at prototype inputs
Model₁(FuncTrain, Sensor)
Model₂(FuncTrain, Sensor)

# generate data to train DeepONet using random Fourier Basis functions and an ODE solver
CDim = 16 # number of terms in Fourier Basis expansion
σp = 1.0 # prior standard deviation of parameters
CGT = σp * randn(CDim) # sample some ground truth parameters
Torque(x, C) = C' * FourierBasis(CDim)(x) # use fourier basis functions for external applied torque
t₀ = 0.0 # left endpoint of sensor domain
t₁ = 1.0 # right endpoint of sensor domain
FunctionGrid = range(t₀, t₁, FuncDim-2) # used as grid for nodes in interpolation for integration function
SensorGrid = range(t₀, t₁, NumSensors) # used to keep track of integral upper limit/domain of operator
Sensor = zeros((SensorDim, NumSensors))
Sensor[1, :] = reshape(collect(SensorGrid), (1, NumSensors))
f̂ = [Torque(t, CGT) for t ∈ FunctionGrid] # sanity checks on data generation
SolutionObs₁ = zeros(NumFuncObs, NumSensors)
SolutionObs₂ = zeros(NumFuncObs, NumSensors)
FunctionObs = zeros(FuncDim, NumFuncObs)
#prob = IntegralProblem(force, x0, xf, θ_gt)
#solve(prob, HCubatureJL())

# x'(t) = v(t)
# v'(t) = -gsin(x(t)) - cv(t)
# setup ODE problem
function Pendulum!(du, u, p, t)
    du[1] = u[2]
    du[2] = -9.8 * sin(u[1]) - 0.5 * u[2] + Torque(t, p)
end
# initial conditions and external torque function parameters
x₀ = 1.0 * π * rand(1)[1] - 0.5 * π # uniform sample a random initial angle wrt vertical on [-π/2, π/2]
v₀ = 1.0 * rand(1)[1] # uniform sample a random initial angular frequency on [-1, 1] rad/s
u₀ = [x₀, v₀]
tspan = (t₀, t₁)
prob = ODEProblem(Pendulum!, u₀, tspan, CGT)
#solve problem
sol = solve(prob, Tsit5(), saveat = collect(SensorGrid))
t̂ = (t₁ - t₀) * rand(1)[1] + t₀ #sample a random point in the time domain
sol(t̂, idxs=1) # more sanity checks
sol(t̂, idxs=2)
plot(SensorGrid, sol(SensorGrid)', title="GT Solution", label = ["θ" "ω"]) # visualize angular position andt angular velocity (u₁ and u₂)
Array(sol(SensorGrid))
# sampling ICs
θ₀ = -π / 2.0
θ₁ = π / 2.0
ω₀ = -1.0
ω₁ = 1.0
# Generate data for DeepONet using u(t) = ODESolve{f(u, t, p)}, t ∈ [x₀, x]
for ϕ ∈ 1:NumFuncObs # iterate over number of functions to sample
    C = σp * randn(CDim) # sample a random function 
    θ_ = (θ₁ - θ₀) * rand(1)[1] + θ₀ # sample a random initial angle from x_low to x_high radians
    ω_ = (ω₁ - ω₀) * rand(1)[1] + ω₀ # sample a random initial angular frequency from v_low to v_high rad/s
    FunctionObs[:, ϕ] = vcat(θ_, ω_, [Torque(t, C) for t ∈ FunctionGrid]) # put the ICs followed by the random function in the row of FunctionObs
    prob = remake(prob, u0 = [θ_, ω_], p = C) # remake ode problem with p = θ and u0 = u₀
    sol = solve(prob, Tsit5()) # integrate
    soln = Array(sol(SensorGrid))
    SolutionObs₁[ϕ, :] = soln[1, :] #get u₁ only
    SolutionObs₂[ϕ, :] = soln[2, :]
end

# evaluate model over input functions "fs" at domain coordinates "sensor"
Model₁(FunctionObs, Sensor) # another sanity check
Model₂(FunctionObs, Sensor)

# Loss function for neural operator network
Loss₁(torques, θs, sensor) = Flux.Losses.mse(Model₁(torques, sensor), θs)
Loss₂(torques, ωs, sensor) = Flux.Losses.mse(Model₂(torques, sensor), ωs)

# training setup (manually decrease learning rate as training progresses or stalls below desired loss)
# I like to see a minimal increase in loss once new random data are generated and model has been trained on previous dataset
# A small increase in loss when new data are generated suggests model is generalizing well to new, unseen data
LearningRate₁ = 1e-2
LearningRate₂ = 1e-3
Opt₁ = Adam(LearningRate₁)
Opt₂ = Adam(LearningRate₂)
Pars₁ = Flux.params(Model₁)
Pars₂ = Flux.params(Model₂)
Flux.@epochs 1000 Flux.train!(Loss₁, Pars₁, [(FunctionObs, SolutionObs₁, Sensor)], Opt₁)
Flux.@epochs 1000 Flux.train!(Loss₂, Pars₂, [(FunctionObs, SolutionObs₂, Sensor)], Opt₂)
FinalLoss₁ = Loss₁(FunctionObs, SolutionObs₁, Sensor)
FinalLoss₂ = Loss₂(FunctionObs, SolutionObs₂, Sensor)

# Evaluate model accuracy and generalization performance using different lengthscales of fourier basis samples on same model params
Ref = 10 # enhance resolution by 10x to showcase the discretization-invariance of DeepONet 
SensorGridRefined = range(t₀, t₁, Ref*NumSensors)
NewSensor = reshape(collect(SensorGridRefined), (1, length(SensorGridRefined)))
# compare ground truth solutions with surrogate model
#p1 = plot(FunctionGrid, FunctionObs[3:end, :], alpha = 0.3, show=false, legend=false, title="External Forcings"); # plot forcing functions
#p2 = plot(SensorGrid, SolutionObs', alpha = 0.3, show=false, legend=false, title="True Solutions"); # plot true solutions (from solver)
#p3 = plot(SensorGridRefined, Model(FunctionObs, NewSensor)', alpha = 0.3, show=false, legend=false, title="Surrogate Solutions"); # plot neural surrogate approximations
#plot(p1, p2, p3)

σθl = 1e-1 + sqrt(FinalLoss₁) # standard deviation of data likelihood (measurement noise + 2 sqrt(model final_loss))
σωl = 1e-1 + sqrt(FinalLoss₂)
# setup turing model to infer the parameters of the fourier basis forcing function which generated the noisy solution trace given as data
@model function BayesFunctionalInverse(u₀Data, θData, ωData)
    C ~ MultivariateNormal(zeros(CDim), σp * I(CDim))
    θ₀ = u₀Data[1]
    ω₀ = u₀Data[2]
    Func = reshape(vcat(θ₀, ω₀, [Torque(t, C) for t ∈ FunctionGrid]), (FuncDim, 1))
    θPred = Model₁(Func, Sensor)
    ωPred = Model₂(Func, Sensor)
    for s ∈ 1:NumSensors
        θData[1, s] ~ Normal(θPred[1, s], σθl) #comment out this line to do inference using only ω data
        ωData[1, s] ~ Normal(ωPred[1, s], σωl) #comment out this line to do inference using only θ data
    end
end

# Evaluate model performance visually
Select = 10 # choose a particular solution from the collection generated above
θData = reshape(SolutionObs₁[Select, :], (1, NumSensors)) # the solver data
ωData = reshape(SolutionObs₂[Select, :], (1, NumSensors))
θDataNoisy = θData + σθl * randn(size(θData)) # data with noise added having amplitude equal to the standard deviation of log likelihood
ωDataNoisy = ωData + σωl * randn(size(ωData))
u₀Data = FunctionObs[1:2, Select]

# visualize model performance
torqueplot = plot(FunctionGrid, FunctionObs[3:end, Select], title = "Torque vs Time", label = "Torque(t)");
modelplot = plot(SensorGrid, θData[1, :], label = "θGT", color=:Black);
modelplot = plot!(SensorGrid, ωData[1, :], label = "ωGT", color=:Red);
modelplot = plot!(SensorGrid, Model₁(FunctionObs[:, Select], Sensor)', label = "θPred", color=:Blue);
modelplot = plot!(SensorGrid, Model₂(FunctionObs[:, Select], Sensor)', label = "ωPred", color=:Orange);
modelplot = scatter!(SensorGrid, θDataNoisy[1, :], label = "θData", color=:Purple);
modelplot = scatter!(SensorGrid, ωDataNoisy[1, :], label = "ωData", title = "Model Prediction", color=:Yellow);
plot(torqueplot, modelplot, layout = (2, 1))

# instantiate Turing model with noisy data
InverseProblem = BayesFunctionalInverse(u₀Data, θDataNoisy, ωDataNoisy)

# sample from model conditional on noisy data using No-U-Turn sampler set at 65% acceptance ratio, 100 samples
Iterations = 100
PosteriorSamples = sample(InverseProblem, NUTS(0.65), Iterations)
StatsPlots.plot(PosteriorSamples) # visualize sampler trajectories and marginal densities

plot(); # visualize the posterior distribution over forcing functions generating the noisy trajectory data
for p in eachrow(Array(PosteriorSamples))
    fpred = [Torque(t, p) for t ∈ FunctionGrid]
    plot!(FunctionGrid, fpred, alpha = 0.05, color=:blue, legend=false)
end
plot!(FunctionGrid, FunctionObs[3:end, Select], lw=1.0, color=:red, title="GT Function(Red); Posterior Traces (Blue)")

plot(); # visualize the posterior distribution over solutions generating the noisy trajectory data
for p in eachrow(Array(PosteriorSamples))
    func = reshape(vcat(u₀Data, [Torque(t, p) for t ∈ FunctionGrid]), (FuncDim, 1))
    prob = remake(prob, u0 = u₀Data, p = p)
    sol = solve(prob, Tsit5())
    soln = Array(sol(SensorGrid))
    θSolver = soln[1, :]
    ωSolver = soln[2, :]
    plot!(SensorGrid, θSolver, alpha = 0.01, color=:Black, legend=false)
    plot!(SensorGrid, ωSolver, alpha = 0.01, color=:Red, legend=false)
    plot!(SensorGrid, Model₁(func, Sensor)', alpha = 0.01, color=:Blue, legend=false)
    plot!(SensorGrid, Model₂(func, Sensor)', alpha = 0.01, color=:Orange, legend=false)
end
scatter!(SensorGrid, θDataNoisy[1, :], color=:Purple, legend=false);
scatter!(SensorGrid, ωDataNoisy[1, :], color=:Yellow, legend=false);
plot!(SensorGrid, SolutionObs₁[Select, :], color=:Black, legend=false);
plot!(SensorGrid, SolutionObs₂[Select, :], color=:Red, legend=false)
