using ForwardDiff
using LinearAlgebra: I
using Plots, Distributions

nθ = 2
# Use in ForwardDiff
function Loss(θ)
    sum(abs2, θ)
end

# Automatic Gradient of negative log likelihood function
∇Loss(θ) = ForwardDiff.gradient(Loss, θ)
∇θ = ∇Loss(θ_gt)
#SGRLD functions
# Monge non-diagonal Riemannian metric tensor auxiliary functions
f₋₁(x, α) = - α^2 / (1 + α^2 * sum(abs2, x))
f₋₁₂(x, α) = (1.0 / sum(abs2, x)) * ((1.0 / sqrt(1.0 + α^2 * sum(abs2, x))) - 1.0)

# Noise distribution term
mdim = nθ
Im = I(mdim)
R = MultivariateNormal(zeros(mdim), Im)

# SGRLD timestep function, hyperparameters and initial conditions
# Time-stepping hyperparameters
a, b, c = (1e2, 1e3, 0.55)
h(t) = 1.0 / (b + a * t)^c

# Initial parameter θ₀
θₜ = zeros(mdim)
λ = 1e-6 # Exponential moving average (EMA) gradient updater
α = 1e-2 # tuning parameter for Monge metric
t = 0.0 #time stepper initial condition
∇lₜ = zeros(mdim) #initialize EMA gradient to zero vector

epochs = 100000 # number of iterations for SGRLD
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
    θₜ += -G⁻¹ * ĝₜ * h(t) + G⁻¹² * sqrt(2.0 * h(t)) * Rₜ # parameter update step
    t += h(t) # time-stepsize update step
end

# Visualize sampler statistics
plot(losses, title = "Loss trace", label = "Loss") # loss trace1plot(Θ, alpha = 0.7, legend = false) # SGRLD Markov Chain 

# Get density corner plots
burn_in = Int64(1e2) # discard n = burn_in samples from sampler statistics
bins = 64 # number of bins to use in parameter histograms
plot(histogram2d(Θ[burn_in:end, 1], Θ[burn_in:end, 2], bins=bins), title="ρ(θ₁,θ₂|D)") # 2D joint density over parameters
plot(histogram(Θ[burn_in:end, 1], bins=bins), title="ρ(θ₁|D)") # 1D marginal density over first parameter
plot(histogram(Θ[burn_in:end, 2], bins=bins), title="ρ(θ₂|D)") # 1D marginal density over second parameter
