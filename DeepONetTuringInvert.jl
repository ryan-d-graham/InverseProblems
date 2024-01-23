using NeuralOperators, Flux, DiffEqFlux
using DifferentialEquations
using LinearAlgebra: I
using Plots, Distributions
using Integrals, IntegralsCubature
using Turing, MCMCChains, StatsPlots

# use a nn to approximate an operator 
xdim = 32 #number of discretized points for input function
sdim = 1 #dimension of domain of neural operator
model = DeepONet((xdim, 64, 72), (sdim, 64, 72), σ, tanh) # Learn nonlinear Banach space map between function spaces

num_func_obs = 1000 # Number of input functions to use 
num_sens_obs = 32 # Number of discrete points at which to evaluate input functions

# create prototype inputs to neural operator model
xtrain = randn((xdim, num_func_obs))
sensor = randn((sdim, num_sens_obs))
# evaluate model at prototype inputs
model(xtrain, sensor)


# generate data to train DeepONet using Gaussian Processes, Integrals and Interpolations
θdim = 4
σθ_prior = 1.0 # prior standard deviation of parameters
θ_gt = σθ_prior * randn(θdim) # sample some ground truth parameters
force(x, θ) = θ' * FourierBasis(length(θ))(x) # use fourier basis functions for external applied torque
x0 = 0.0 # left endpoint of sensor domain
xf = 1.0 # right endpoint of sensor domain
fgrid = range(x0, xf, xdim) # used as grid for nodes in interpolation for integration function
sgrid = range(x0, xf, num_sens_obs) # used to keep track of integral upper limit/domain of operator
sensor = reshape(collect(sgrid), (1, num_sens_obs))
f̂ = [force(x, θ_gt) for x ∈ fgrid] # sanity checks on data generation
Gfs_ξ = zeros(num_func_obs, num_sens_obs)
fs = zeros(xdim, num_func_obs)
#prob = IntegralProblem(force, x0, xf, θ_gt)
#solve(prob, HCubatureJL())

# x'(t) = v(t)
# v'(t) = -gsin(x(t)) - cv(t)
# setup ODE problem
function odefunc(du, u, p, t)
    du[1] = u[2]
    du[2] = -9.8 * sin(u[1]) - 0.5 * u[2] + force(t, p)
end
# initial conditions and external torque function parameters
u₀ = [0.0, 0.0]
tspan = (x0, xf)
p = θ_gt
prob = ODEProblem(odefunc, u₀, tspan, p)
#solve problem
sol = solve(prob, Tsit5(), saveat = collect(sgrid))
sol(0.3, idxs=2) # more sanity checks
plot(sgrid, sol(sgrid, idxs=2), title="GT Solution") # visualize

# Generate data for DeepONet using u(t) = ODESolve{f(u, t, p)}, t ∈ [x₀, x]
for ϕ ∈ 1:num_func_obs # iterate over number of functions to sample
    θ = σθ_prior * randn(θdim) # sample a random function 
    fs[:, ϕ] = [force(x, θ) for x ∈ fgrid] # put the function in the row of fs
    prob = remake(prob, p = θ) # remake ode problem with p = θ
    sol = solve(prob, Tsit5()) # integrate
    Gfs_ξ[ϕ, :] = sol(sgrid, idxs=2)
end

# visualize forcing functions and their forward influence on ODE trajectory (2nd variable only: angular position) 
fplot = plot(fgrid, fs, alpha = 0.2, legend=false, title="force(t)"); # plot of function samples
∫fplot = plot(sgrid, Gfs_ξ', alpha = 0.2, legend=false, title="Sol(t)");
plot(fplot, ∫fplot)

# evaluate model over input functions "fs" at domain coordinates "sensor"
model(fs, sensor) # another sanity check

# Loss function for neural operator network
loss(xtrain, ytrain, sensor) = Flux.Losses.mse(model(xtrain, sensor), ytrain)

# training setup (manually decrease learning rate as training progresses or stalls below desired loss)
# I like to see a 1.0e-5 or less on final_loss
learning_rate = 1e-3
opt = Adam(learning_rate)
parameters = Flux.params(model)
Flux.@epochs 1000 Flux.train!(loss, parameters, [(fs, Gfs_ξ, sensor)], opt)
final_loss = loss(fs, Gfs_ξ, sensor)

# Evaluate model accuracy and generalization performance using different lengthscales of fourier basis samples on same model params
new_res = 10 # enhance resolution by 10x to showcase the discretization-invariance of DeepONet 
new_sgrid = range(x0, xf, new_res*num_sens_obs)
new_sensor = reshape(collect(new_sgrid), (1, length(new_sgrid)))
fplot = plot(fgrid, fs, alpha = 0.5, legend=false, title="force(t)"); # plot forcing functions
∫fplot = plot(sgrid, Gfs_ξ', alpha = 0.5, legend=false, title="Sol(t)"); # plot true solutions (from solver)
Gfsplot = plot(new_sgrid, model(fs, new_sensor)', alpha = 0.5, legend=false, title="G{f(x)}(ξ)"); # plot neural surrogate approximations
plot(fplot, ∫fplot, Gfsplot)

σl = 1e-2 + final_loss # standard deviation of data likelihood (measurement noise + model final_loss)
# setup turing model to infer the parameters of the fourier basis forcing function which generated the noisy solution trace given as data
@model function BayesFunctionalInverse(Gfs_ξ)
    θ ~ MultivariateNormal(zeros(θdim), σθ_prior * I(θdim))
    func = reshape([force(x, θ) for x ∈ fgrid], (xdim, 1))
    pred = model(func, sensor)
    for s ∈ 1:num_sens_obs
        Gfs_ξ[1, s] ~ Normal(pred[1, s], σl)
    end
end

fselect = 1 # choose a particular solution from the collection generated above
data = reshape(Gfs_ξ[fselect, :], (1, num_sens_obs)) # the solver data
data_n = data + σl * randn(size(data)) # data with noise added having amplitude equal to the standard deviation of log likelihood
funcplot = Plots.plot(fgrid, fs[:, fselect], title="GT Function");
solplot = Plots.plot(sgrid, data[1, :], title="GT Solution");
datplot = Plots.plot(sgrid, data_n[1, :], title="Solution Data");
plot(funcplot, solplot, datplot)

# instantiate Turing model with noisy data
InverseProblem = BayesFunctionalInverse(data_n)

# sample from model conditional on noisy data using No-U-Turn sampler set at 65% acceptance ratio, 100 samples
chain = sample(InverseProblem, NUTS(0.65), 100)
plot(chain) # visualize sampler trajectories and marginal densities

plot(); # visualize the posterior distribution over forcing functions generating the noisy trajectory data
for p in eachrow(Array(chain))
    pred = [force(x, p) for x ∈ fgrid]
    plot!(fgrid, pred, alpha = 0.05, color=:blue, legend=false)
end
plot!(fgrid, fs[:, fselect], lw=1.0, color=:red, title="GT Function(Red); Posterior Traces (Blue)")