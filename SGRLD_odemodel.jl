using ForwardDiff
using LinearAlgebra: I
using Plots, Distributions
using OrdinaryDiffEq

# Experiment with gradients
# x''(t) = -gsin(x(t)) - cx'(t)
# x'(t) = v(t)
# v'(t) = -gsin(x(t)) - cv(t)
function odefunc(du, u, p, t)
    du[1] = u[2]
    du[2] = -p[1] * sin(u[1]) - p[2] * u[2]
end

ndata = 100 # number of data points to generate
u0 = [π/2, 0.0] # initial angle and angular velocity
tspan = (0.0, 3.0) # time span over which to solve
ts = collect(range(tspan[1], tspan[2], ndata)) # time grid
p = [9.8, 0.3] # ground truth parameters 
mdim = length(p) # dimension of the inverse problem
prob = ODEProblem(odefunc, u0, tspan, p) # set up the ode problem
sol = solve(prob, Vern9(), saveat = ts) # solve problem
u = Array(sol) # extract solution array
# Generate some toy data produced by an ode system

σl = 1.0 # standard deviation of model likelihood
σp = 10.0 # prior standard deviation of parameters
data = u + σl * randn(size(u)) # add noise with amplitude equal to log likelihood stdev

# visualize solution overlaid on generated data
scatter(ts, data', color=:blue, alpha = 0.4, label = "Data", title = "ODE Model");
plot!(ts, u', color=:red, lw = 2.0, alpha = 0.4, label = "GT Model")

# Create negative log-posterior function using only θ as a parameter and globally-referenced data (important for easy use with hessian)
# Use in ForwardDiff
function Loss(θ)
    pred = Array(solve(prob, Vern9(), p=θ, saveat=ts, dtmin = 1e-16))
    log(2*π*σp^2) + 0.5 * sum(abs2, θ / σp) + ndata * log(2*π*σl^2) + 0.5 * sum(abs2, (data .- pred) ./ σl)
end

# Automatic Gradient of negative log likelihood function
∇Loss(θ) = ForwardDiff.gradient(Loss, θ)
∇p = ∇Loss(σp * rand(mdim))
#SGRLD functions
# Monge non-diagonal Riemannian metric tensor auxiliary functions
f₋₁(x, α) = - α^2 / (1 + α^2 * sum(abs2, x))
f₋₁₂(x, α) = (1.0 / sum(abs2, x)) * ((1.0 / sqrt(1.0 + α^2 * sum(abs2, x))) - 1.0)

# Noise distribution term
Im = I(mdim)
R = MultivariateNormal(zeros(mdim), Im)

# SGRLD timestep function, hyperparameters and initial conditions
# Time-stepping hyperparameters
a, b, c = (100.0, 1000.0, 0.65)
h(t) = 1.0 / (b + a * t)^c

# Initial parameter θ₀
θₜ = zeros(mdim)
λ = 0.5 # Exponential moving average (EMA) gradient updater
α = 1e-2 # tuning parameter for Monge metric
t = 0.0 #time stepper initial condition
∇lₜ = zeros(mdim) #initialize EMA gradient to zero vector

epochs = 25000 # number of iterations for SGRLD
losses = zeros(epochs) # loss bookkeeping 
Θ = zeros(epochs, mdim) # parameter trace bookkeeping 
# Main SGRLD inference loop
for epoch ∈ 1:epochs
    # verbose training
    #print("θₜ: ", θₜ)
    #print("\n")
    Θ[epoch, :] = θₜ # keep track of parameter samples
    L = Loss(θₜ) # compute loss for this iteration
    losses[epoch] = L # keep track of losses
    #print("Loss: ", L)
    #print("\n")

    Rₜ = rand(R) # draw random variate for Langevin update step
    ĝₜ = ∇Loss(θₜ) # compute gradient at current parameter coordinate
    ∇lₜ = λ .* ∇lₜ .+ (1.0 - λ) .* ĝₜ # compute EMA of gradient 
    Mₜ = ∇lₜ * ∇lₜ' # Fisher Information Matrix (outer product of gradient vector)
    G⁻¹ = Im + f₋₁(∇lₜ, α) * Mₜ # Riemannian metric tensor (Monge metric) for gradient
    G⁻¹² = Im + f₋₁₂(∇lₜ, α) * Mₜ # Riemannian metric tensor for noise term
    θₜ += - G⁻¹ * ĝₜ * h(t) + G⁻¹² * sqrt(2.0 * h(t)) * Rₜ # parameter update step
    t += h(t) # time-stepsize update step
end

# Visualize sampler statistics
plot(losses, title = "Loss trace", label = "Loss") # loss trace
plot(Θ, labels = ["θₜ₁" "θₜ₂"], alpha = 0.7); # SGRLD Markov Chain 
plot!(1:epochs, p[1] * ones(length(1:epochs)), label = "θ_gt[1]"); # ground truth parameter values
plot!(1:epochs, p[2] * ones(length(1:epochs)), label = "θ_gt[2]")

# Get density corner plots
burn_in = Int64(5e2) # discard n = burn_in samples from sampler statistics
bins = 128 # number of bins to use in parameter histograms
plot(histogram2d(Θ[burn_in:end, 1], Θ[burn_in:end, 2], bins=bins), title="ρ(θ₁,θ₂|D)") # 2D joint density over parameters
plot(histogram(Θ[burn_in:end, 1], bins=bins), title="ρ(θ₁|D)") # 1D marginal density over first parameter
plot(histogram(Θ[burn_in:end, 2], bins=bins), title="ρ(θ₂|D)") # 1D marginal density over second parameter

skip = Int64(8) # skip this number of samples when drawing posterior solve traces (higher removes more)
ref = Int64(1) # refine the grid by this factor (higher is more refined)
ts_refined = collect(range(tspan[1], tspan[end], ref*length(ts))) # refined time grid
scatter(ts, data', alpha = 0.7, color=:blue, title = "Data(blue) Posterior samples(red) GT(black)"); # plot data
for sample ∈ burn_in:skip:epochs # draw posterior model traces using parameter samples
    u = Array(solve(prob, Vern9(), p = Θ[sample, :], saveat = ts_refined)) # get solution trace
    traces = plot!(ts_refined, u', alpha = 0.1, color=:red, lw=0.1, legend=false) # plot trace
end
u_gt = Array(solve(prob, Vern9(), p = p, saveat = ts_refined)) # compute ground truth solution
plot!(ts_refined, u_gt', alpha = 0.8, color=:black, lw=2.0, legend=false) # plot ground truth solution trace on top of posterior traces and data

# get summary sampler statistics
p̄ = mean(Θ[burn_in:end, :], dims=1) # mean vector of parameters
p̂ = median(Θ[burn_in:end, :], dims=1) # median vector of parameters
CI95 = 2*sqrt.(var(Θ[burn_in:end, :], dims=1)) # standard deviation vector of parameters
# 95% Confidence interval
pᵤ = p̄ + CI95
pₗ = p̄ - CI95
