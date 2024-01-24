using NeuralOperators, Flux, DiffEqFlux
using DifferentialEquations
using LinearAlgebra: I
using Plots, Distributions
using Turing, MCMCChains, StatsPlots

# use a nn to approximate an operator, eg. an ODE/PDE solver, Nonlinear Solver, Optimization problem etc
# DeepONet approximates a Banach space map using a discretized input function and sensor locations at which to evaluate the operator
# Architecture: 
# Model{f(x)}(ξ) = Branch{f(x)}ᵀ Trunk{ξ}, where f(x) is the discretized input function and ξ are the coordinates at which to evaluate operator
FuncDim = 32 # number of discrete points to approximate input function
SensorDim = 1 # number of dimensions in input domain of neural operator
Model = DeepONet((FuncDim, 64, 72), (SensorDim, 64, 72), σ, tanh) # Learn nonlinear Banach space map between function spaces

NumFuncObs = 1024 # Number of randomly-sampled input functions to use for training
NumSensors = 32 # Number of discrete points at which to evaluate input functions

# create prototype inputs to neural operator model
FuncTrain = randn((FuncDim, NumFuncObs))
Sensor = randn((SensorDim, NumSensors))
# evaluate model at prototype inputs
Model(FuncTrain, Sensor)


# generate data to train DeepONet using Gaussian Processes, Integrals and Interpolations
θDim = 10 # number of terms in Fourier Basis expansion
σp = 1.0 # prior standard deviation of parameters
θGT = σp * randn(θDim) # sample some ground truth parameters
Force(x, θ) = θ' * FourierBasis(length(θ))(x) # use fourier basis functions for external applied torque
t₀ = 0.0 # left endpoint of sensor domain
t₁ = 1.0 # right endpoint of sensor domain
FunctionGrid = range(t₀, t₁, FuncDim) # used as grid for nodes in interpolation for integration function
SensorGrid = range(t₀, t₁, NumSensors) # used to keep track of integral upper limit/domain of operator
Sensor = reshape(collect(SensorGrid), (SensorDim, NumSensors))
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
u₀ = [0.0, 0.0]
tspan = (t₀, t₁)
prob = ODEProblem(Pendulum!, u₀, tspan, θGT)
#solve problem
sol = solve(prob, Tsit5(), saveat = collect(SensorGrid))
sol(rand(1)[1], idxs=2) # more sanity checks
plot(SensorGrid, sol(SensorGrid, idxs=2), title="GT Solution") # visualize angular position without angular velocity (u₂ and not u₁)

# Generate data for DeepONet using u(t) = ODESolve{f(u, t, p)}, t ∈ [x₀, x]
for ϕ ∈ 1:NumFuncObs # iterate over number of functions to sample
    θ = σp * randn(θDim) # sample a random function 
    FunctionObs[:, ϕ] = [Force(t, θ) for t ∈ FunctionGrid] # put the function in the row of fs
    prob = remake(prob, p = θ) # remake ode problem with p = θ
    sol = solve(prob, Tsit5()) # integrate
    SolutionObs[ϕ, :] = sol(SensorGrid, idxs=2)
end

# evaluate model over input functions "fs" at domain coordinates "sensor"
Model(FunctionObs, Sensor) # another sanity check

# Loss function for neural operator network
Loss(fs, sols, sensor) = Flux.Losses.mse(Model(fs, sensor), sols)

# training setup (manually decrease learning rate as training progresses or stalls below desired loss)
# I like to see a 1.0e-5 or less on final_loss
LearningRate = 1e-4
Opt = Adam(LearningRate)
Pars = Flux.params(Model)
Flux.@epochs 1000 Flux.train!(Loss, Pars, [(FunctionObs, SolutionObs, Sensor)], Opt)
FinalLoss = Loss(FunctionObs, SolutionObs, Sensor)

# Evaluate model accuracy and generalization performance using different lengthscales of fourier basis samples on same model params
Ref = 4 # enhance resolution by 10x to showcase the discretization-invariance of DeepONet 
SensorGridRefined = range(t₀, t₁, Ref*NumSensors)
NewSensor = reshape(collect(SensorGridRefined), (1, length(SensorGridRefined)))
# compare ground truth solutions with surrogate model
p1 = Plots.plot(FunctionGrid, FunctionObs, alpha = 0.3, show=false, legend=false, title="External Forcings"); # plot forcing functions
p2 = Plots.plot(SensorGrid, SolutionObs', alpha = 0.3, show=false, legend=false, title="True Solutions"); # plot true solutions (from solver)
p3 = Plots.plot(SensorGridRefined, Model(FunctionObs, NewSensor)', alpha = 0.3, show=false, legend=false, title="Surrogate Solutions"); # plot neural surrogate approximations
plot(p1, p2, p3)

σl = 2e-2 + FinalLoss # standard deviation of data likelihood (measurement noise + model final_loss)
# setup turing model to infer the parameters of the fourier basis forcing function which generated the noisy solution trace given as data
@model function BayesFunctionalInverse(SolutionObs)
    θ ~ MultivariateNormal(zeros(θDim), σp * I(θDim))
    Func = reshape([Force(t, θ) for t ∈ FunctionGrid], (FuncDim, 1))
    pred = Model(Func, Sensor)
    for s ∈ 1:NumSensors
        SolutionObs[1, s] ~ Normal(pred[1, s], σl)
    end
end

Select = 2 # choose a particular solution from the collection generated above
Data = reshape(SolutionObs[Select, :], (1, NumSensors)) # the solver data
DataNoisy = Data + σl * randn(size(Data)) # data with noise added having amplitude equal to the standard deviation of log likelihood
funcplot = Plots.plot(FunctionGrid, FunctionObs[:, Select], title="GT Function");
solplot = Plots.plot(SensorGrid, Data[1, :], title="GT Solution");
datplot = Plots.plot(SensorGrid, DataNoisy[1, :], title="Solution Data");
plot(funcplot, solplot, datplot)

# instantiate Turing model with noisy data
InverseProblem = BayesFunctionalInverse(DataNoisy)

# sample from model conditional on noisy data using No-U-Turn sampler set at 65% acceptance ratio, 100 samples
Iterations = 100
PosteriorSamples = sample(InverseProblem, NUTS(0.65), Iterations)
StatsPlots.plot(PosteriorSamples) # visualize sampler trajectories and marginal densities

plot(); # visualize the posterior distribution over forcing functions generating the noisy trajectory data
for p in eachrow(Array(PosteriorSamples))
    fpred = [Force(t, p) for t ∈ FunctionGrid]
    plot!(FunctionGrid, fpred, alpha = 0.05, color=:blue, legend=false)
end
plot!(FunctionGrid, FunctionObs[:, Select], lw=1.0, color=:red, title="GT Function(Red); Posterior Traces (Blue)")

plot(); # visualize the posterior distribution over solutions generating the noisy trajectory data
for p in eachrow(Array(PosteriorSamples))
    func = reshape([Force(t, p) for t ∈ FunctionGrid], (FuncDim, 1))
    prob = remake(prob, p = p)
    sol = solve(prob, Tsit5())
    pred = sol(SensorGrid, idxs=2).u
    plot!(SensorGrid, pred, alpha = 0.05, color=:blue, legend=false)
    plot!(SensorGrid, Model(func, Sensor)', alpha = 0.05, color=:green, legend=false)
end
plot!(SensorGrid, SolutionObs[Select, :], lw=1.0, color=:red, title="GT(Red); Solver(Blue); Surrogate(Green)")