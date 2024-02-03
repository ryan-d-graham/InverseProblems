using ForwardDiff
using LinearAlgebra: I
using Plots, Distributions
using DiffEqFlux


function model(x, θ)
    n = length(θ)
    θ' * FourierBasis(n)(x)
end

# Generate some toy data produced by a truncated Fourier series
mdim = 16
x0, xf = (0.0, 1.0*π)
ndata = 32
npoints = 512
σl = 1e0 # standard deviation of model likelihood
σp = 1e0 # prior standard deviation of parameters
mnoise = 1e0
θ_gt = σp * randn(mdim) # ground truth parameters

# generate data under model
xs_data = collect(range(x0, xf, ndata))
xs_plot = collect(range(x0, xf, npoints))
fs_data = [model(x, θ_gt) for x ∈ xs_data]
fs_plot = [model(x, θ_gt) for x ∈ xs_plot]
data = fs_data + mnoise * randn(ndata)
scatter(xs_data, data, color=:blue, alpha = 0.4, label = "Data", title = "Poly Model");
plot!(xs_plot, fs_plot, color=:red, lw = 2.0, alpha = 0.4, label = "GT Model")

# Create negative log-posterior function using only θ as a parameter and globally-referenced data (important for easy use with hessian)
# Use in ForwardDiff
function Loss(θ)
    ypred = [model(x, θ) for x ∈ xs_data]
    0.5 * mdim * log(2*π*σp^2) + 0.5 * sum(abs2, θ / σp) + 0.5 * ndata * log(2*π*σl^2) + 0.5 * sum(abs2, (data .- ypred) / σl)
end

# Automatic Gradient, hessian and laplacian of loss function
∇Loss(θ) = ForwardDiff.gradient(Loss, θ)
∇Loss(randn(mdim))
#SGRLD functions
# Monge non-diagonal Riemannian metric tensor auxiliary functions
f₋₁(x, α) = - α^2 / (1 + α^2 * sum(abs2, x))
f₋₁₂(x, α) = (1.0 / sum(abs2, x)) * ((1.0 / sqrt(1.0 + α^2 * sum(abs2, x))) - 1.0)

# Noise distribution term
R = MultivariateNormal(zeros(mdim), I(mdim))

# SGRLD functions, hyperparameters and initial conditions
# Time-stepping term
a, b, c = (1e2, 1e3, 0.55)
h(t) = 1.0 / (b + a * t)^c


# Initial parameter θ₀
θₜ = randn(mdim)
λ = 1e-1 # EMA gradient updater
α = 1e-2 # tuning parameter for Monge metric
t = 0.0 #time stepper initial condition
∇lₜ = zeros(mdim) #initialize EMA gradient to zero vector

epochs = 10000
losses = zeros(epochs) # loss bookkeeping 
Θ = zeros(epochs, mdim) # parameter trace bookkeeping 
# Main SGRLD inference loop
for epoch ∈ 1:epochs
    # verbose training
    #print("θₜ: ", θₜ)
    #print("\n")
    Θ[epoch, :] = θₜ
    L = Loss(θₜ)
    losses[epoch] = L
    print("Loss: ", L)
    print("\n")

    Rₜ = rand(R)
    ĝₜ = ∇Loss(θₜ)
    ∇lₜ = λ * ∇lₜ + (1.0 - λ) * ĝₜ
    Mₜ = ∇lₜ * ∇lₜ'
    G⁻¹ = I(mdim) + f₋₁(∇lₜ, α) * Mₜ
    G⁻¹² = I(mdim) + f₋₁₂(∇lₜ, α) * Mₜ 
    θₜ += - G⁻¹ * ĝₜ * h(t) + G⁻¹² * sqrt(2.0 * h(t)) * Rₜ
    t += h(t)
end

# Visualize sampler statistics
plot(losses, title = "Loss trace", label = "Loss")
#plot(Θ, labels = ["θₜ₁" "θₜ₂" "θₜ₃" "θₜ₄"], alpha = 0.1);
#plot!(1:epochs, θ_gt[1] * ones(length(1:epochs)), label = "θ_gt[1]");
#plot!(1:epochs, θ_gt[2] * ones(length(1:epochs)), label = "θ_gt[2]");
#plot!(1:epochs, θ_gt[3] * ones(length(1:epochs)), label = "θ_gt[3]");
#plot!(1:epochs, θ_gt[4] * ones(length(1:epochs)), label = "θ_gt[4]");
#plot!(1:epochs, θ_gt[5] * ones(length(1:epochs)), label = "θ_gt[5]");
#plot!(1:epochs, θ_gt[6] * ones(length(1:epochs)), label = "θ_gt[6]");
#plot!(1:epochs, θ_gt[7] * ones(length(1:epochs)), label = "θ_gt[7]");
#plot!(1:epochs, θ_gt[8] * ones(length(1:epochs)), label = "θ_gt[8]");
#plot!(1:epochs, θ_gt[9] * ones(length(1:epochs)), label = "θ_gt[9]");
#plot!(1:epochs, θ_gt[10] * ones(length(1:epochs)), label = "θ_gt[10]", legend=false)

# Get density corner plots
burn_in = Int64(1e1)
bins = 32
dist1, dist2 = 3, 4
plot(histogram2d(Θ[burn_in:epochs, dist1], Θ[burn_in:epochs, dist2], bins=bins))

skip = Int64(4)
scatter(xs_data, data, alpha = 0.7, color=:blue, title = "Data(blue); Posterior samples(red); GT(black)");
for sample ∈ burn_in:skip:epochs
    preds = [model(x, Θ[sample, :]) for x ∈ xs_plot]
    plot!(xs_plot, preds, alpha = 0.1, color=:red, lw=0.05, legend=false);
end
plot!(xs_plot, fs_plot, alpha = 0.8, color=:black, lw=2.0, legend=false)