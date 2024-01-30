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
Model = DeepONet((FuncDim, 32, 72), (SensorDim, 32, 72), σ, tanh) # Learn nonlinear Banach space map between function spaces

NumFuncObs = 10000 # Number of randomly-sampled input functions to use for training
NumSensors = 32 # Number of discrete points at which to evaluate output functions

# create prototype inputs to neural operator model (this forces precompilation and speeds up later evals)
FuncTrain = randn((FuncDim, NumFuncObs))
Sensor = randn((SensorDim, NumSensors))
# evaluate model at prototype inputs
Model(FuncTrain, Sensor)

# generate data to train DeepONet using random Fourier Basis functions and an ODE solver
θDim = 16 # number of terms in Fourier Basis expansion
σp = 1.0 # prior standard deviation of parameters
θGT = σp * randn(θDim) # sample some ground truth parameters
Force(x, θ) = θ' * FourierBasis(length(θ))(x) # use fourier basis functions for external applied torque
t₀ = 0.0 # left endpoint of sensor domain
t₁ = 1.0 # right endpoint of sensor domain
FunctionGrid = range(t₀, t₁, FuncDim-2) # used as grid for nodes in interpolation for integration function
SensorGrid = range(t₀, t₁, NumSensors) # used to keep track of integral upper limit/domain of operator
Sensor = zeros((SensorDim, NumSensors))
Sensor[1, :] = reshape(collect(SensorGrid), (1, NumSensors))
f̂ = [Force(t, θGT) for t ∈ FunctionGrid] # sanity checks on data generation
SolutionObs = zeros(NumFuncObs, NumSensors)
FunctionObs = zeros(FuncDim, NumFuncObs)
#prob = IntegralProblem(force, x0, xf, θ_gt)
#solve(prob, HCubatureJL())

# x'(t) = v(t)
# v'(t) = -gsin(x(t)) - cv(t)
# setup ODE problem
function Pendulum!(du, u, p, t)
    du[1] = u[2]
    du[2] = -9.8 * sin(u[1]) - 0.5 * u[2] + Force(t, p)
end
# initial conditions and external torque function parameters
x₀ = 1.0 * π * rand(1)[1] - 0.5 * π # uniform sample a random initial angle wrt vertical on [-π/2, π/2]
v₀ = 1.0 * rand(1)[1] # uniform sample a random initial angular frequency on [-1, 1] rad/s
u₀ = [x₀, v₀]
tspan = (t₀, t₁)
prob = ODEProblem(Pendulum!, u₀, tspan, θGT)
#solve problem
sol = solve(prob, Tsit5(), saveat = collect(SensorGrid))
t⋆ = (t₁ - t₀) * rand(1)[1] + t₀ #sample a random point in the time domain
sol(rand(1)[1], idxs=1) # more sanity checks
plot(SensorGrid, sol(SensorGrid, idxs=1), title="GT Solution") # visualize angular position without angular velocity (u₁ and not u₂)

# sampling ICs
x_low = -π / 2.0
x_high = π / 2.0
v_low = -1.0
v_high = 1.0
# Generate data for DeepONet using u(t) = ODESolve{f(u, t, p)}, t ∈ [x₀, x]
for ϕ ∈ 1:NumFuncObs # iterate over number of functions to sample
    θ = σp * randn(θDim) # sample a random function 
    x₀ = (x_high - x_low) * rand(1)[1] + x_high # sample a random initial angle from x_low to x_high radians
    v₀ = (v_high - v_low) * rand(1)[1] + v_high # sample a random initial angular frequency from v_low to v_high rad/s
    FunctionObs[:, ϕ] = vcat(x₀, v₀, [Force(t, θ) for t ∈ FunctionGrid]) # put the ICs followed by the random function in the row of FunctionObs
    prob = remake(prob, u0 = [x₀, v₀], p = θ) # remake ode problem with p = θ and u0 = u₀
    sol = solve(prob, Tsit5()) # integrate
    SolutionObs[ϕ, :] = sol(SensorGrid, idxs=1) #get u₁ only
end

# evaluate model over input functions "fs" at domain coordinates "sensor"
Model(FunctionObs, Sensor) # another sanity check

# Loss function for neural operator network
Loss(fs, sols, sensor) = Flux.Losses.mse(Model(fs, sensor), sols)

# training setup (manually decrease learning rate as training progresses or stalls below desired loss)
# I like to see a minimal increase in loss once new random data are generated and model has been trained on previous dataset
# A small increase in loss when new data are generated suggests model is generalizing well to new, unseen data
LearningRate = 1e-3
Opt = Adam(LearningRate)
Pars = Flux.params(Model)
Flux.@epochs 1000 Flux.train!(Loss, Pars, [(FunctionObs, SolutionObs, Sensor)], Opt)
FinalLoss = Loss(FunctionObs, SolutionObs, Sensor)

# Evaluate model accuracy and generalization performance using different lengthscales of fourier basis samples on same model params
Ref = 10 # enhance resolution by 10x to showcase the discretization-invariance of DeepONet 
SensorGridRefined = range(t₀, t₁, Ref*NumSensors)
NewSensor = reshape(collect(SensorGridRefined), (1, length(SensorGridRefined)))
# compare ground truth solutions with surrogate model
#p1 = plot(FunctionGrid, FunctionObs[3:end, :], alpha = 0.3, show=false, legend=false, title="External Forcings"); # plot forcing functions
#p2 = plot(SensorGrid, SolutionObs', alpha = 0.3, show=false, legend=false, title="True Solutions"); # plot true solutions (from solver)
#p3 = plot(SensorGridRefined, Model(FunctionObs, NewSensor)', alpha = 0.3, show=false, legend=false, title="Surrogate Solutions"); # plot neural surrogate approximations
#plot(p1, p2, p3)

σl = 1e-1 + sqrt(FinalLoss) # standard deviation of data likelihood (measurement noise + 2 sqrt(model final_loss))
# setup turing model to infer the parameters of the fourier basis forcing function which generated the noisy solution trace given as data
@model function BayesFunctionalInverse(u₀Data, SolutionObs)
    θ ~ MultivariateNormal(zeros(θDim), σp * I(θDim))
    x₀ = u₀Data[1]
    v₀ = u₀Data[2]
    Func = reshape(vcat(x₀, v₀, [Force(t, θ) for t ∈ FunctionGrid]), (FuncDim, 1))
    pred = Model(Func, Sensor)
    for s ∈ 1:NumSensors
        SolutionObs[1, s] ~ Normal(pred[1, s], σl)
    end
end

Select = 1 # choose a particular solution from the collection generated above
Data = reshape(SolutionObs[Select, :], (1, NumSensors)) # the solver data
DataNoisy = Data + σl * randn(size(Data)) # data with noise added having amplitude equal to the standard deviation of log likelihood
u₀Data = FunctionObs[1:2, Select]
funcplot = Plots.plot(FunctionGrid, FunctionObs[3:end, Select], title="GT Function");
solplot = Plots.plot(SensorGrid, Data[1, :], title="GT Solution");
solplot = Plots.plot!(SensorGrid, Model(FunctionObs[:, Select], Sensor)', title = "GT(Blue) & Pred(Orange)");
datplot = Plots.scatter(SensorGrid, DataNoisy[1, :], title="Solution Data");
plot(funcplot, solplot, datplot)

# instantiate Turing model with noisy data
InverseProblem = BayesFunctionalInverse(u₀Data, DataNoisy)

# sample from model conditional on noisy data using No-U-Turn sampler set at 65% acceptance ratio, 100 samples
Iterations = 1000
PosteriorSamples = sample(InverseProblem, NUTS(0.65), Iterations)
StatsPlots.plot(PosteriorSamples) # visualize sampler trajectories and marginal densities

plot(); # visualize the posterior distribution over forcing functions generating the noisy trajectory data
for p in eachrow(Array(PosteriorSamples))
    fpred = [Force(t, p) for t ∈ FunctionGrid]
    plot!(FunctionGrid, fpred, alpha = 0.05, color=:blue, legend=false)
end
plot!(FunctionGrid, FunctionObs[3:end, Select], lw=1.0, color=:red, title="GT Function(Red); Posterior Traces (Blue)")

plot(); # visualize the posterior distribution over solutions generating the noisy trajectory data
for p in eachrow(Array(PosteriorSamples))
    func = reshape(vcat(u₀Data, [Force(t, p) for t ∈ FunctionGrid]), (FuncDim, 1))
    prob = remake(prob, u0 = u₀Data, p = p)
    sol = solve(prob, Tsit5())
    pred = sol(SensorGrid, idxs=1).u
    plot!(SensorGrid, pred, alpha = 0.01, color=:blue, legend=false)
    plot!(SensorGrid, Model(func, Sensor)', alpha = 0.01, color=:green, legend=false)
end
scatter!(SensorGrid, DataNoisy[1, :], color=:black);
plot!(SensorGrid, SolutionObs[Select, :], color=:red, title="GT(Red); Solver(Blue); Surrogate(Green); Data(Black)")
