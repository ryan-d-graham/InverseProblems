using NeuralOperators, Flux
using Integrals, Dierckx
using LinearAlgebra: I
using Plots, Distributions
using Turing, StatsPlots

# use a nn to approximate an operator, eg. an ODE/PDE solver, Nonlinear Solver, Optimization problem etc
# DeepONet approximates a Banach space map using a discretized input function and sensor locations at which to evaluate the operator
# Architecture: 
# Model{f(x)}(ξ) = Branch{f(x)}ᵀ Trunk{ξ}, where f(x) is the discretized input function and ξ are the coordinates at which to evaluate operator
Nx, Ny = 32, 32
FuncDim = Nx*Ny # number of discrete points to approximate input function
SensorDim = 2 # number of dimensions in input domain of neural operator's output function
Model = DeepONet((FuncDim, 32, 72), (SensorDim, 32, 72), σ, tanh) # Learn nonlinear Banach space map between function spaces

NumFuncObs = 100 # Number of randomly-sampled input functions to use for training
NumSensors = Nx*Ny # Number of discrete points at which to evaluate output functions

# create prototype inputs to neural operator model (this forces precompilation and speeds up later evals)
FuncTrain = randn((FuncDim, NumFuncObs))
Sensor = randn((SensorDim, NumSensors))
# evaluate model at prototype inputs
Model(FuncTrain, Sensor)

# generate data for DeepONet training
#setup solution integral for Poisson Equation ∇²u = -f
f(r) = -exp(-sum(abs2, r)) #forcing function
G(r, r′) = (1.0 / (2*π)) * log(sum(abs, r - r′)) #Green's function for unbounded Poisson Equation
integrand(r′, r) = f(r′) * G(r, r′) #convolution integrand

r₀ = randn(2) #solution eval point
prob = IntegralProblem(integrand, [-Inf, -Inf], [Inf, Inf], p = r₀)
sol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3).u

xl, xh = -25, 25
yl, yh = -25, 25
FunctionGridX = range(xl, xh, Nx)
FunctionGridY = range(yl, yh, Ny)
FunctionObs = zeros((FuncDim, NumFuncObs))
SolutionObs = zeros((NumFuncObs, NumSensors))
Sensor = 10.0 * randn((SensorDim, NumSensors))


for ϕ ∈ 1:NumFuncObs
    c00, c10, c01, c11 = randn((4, 1))
    #print("Coeffs geneated...\n")
    f(r) = -exp(-sum(abs2, r / 2)) * (c00 + c10 * r[1] + c01 * r[2] + c11 * r[1]*r[2]) 
    #print("Density defined...\n")
    integrand(r′, r) = f(r′) * G(r, r′)
    #print("Integrand defined...\n")
    FunctionObs[:, ϕ] = [f([x, y]) for x ∈ FunctionGridX for y ∈ FunctionGridY]
    #print("Density saved...\n")
    for s ∈ 1:NumSensors
        prob = IntegralProblem(integrand, [-Inf, -Inf], [Inf, Inf], p = Sensor[:, s])
        #print("IntegralProblem defined...\n")
        SolutionObs[ϕ, s] = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3).u
        #print("Integral solved...\n")
    end
    print("FunctionObs: ", ϕ, " / ", NumFuncObs, "\n")
end

Predictions = Model(FunctionObs, Sensor)

Loss(densities, fields, sensors) = Flux.Losses.mse(Model(densities, sensors), fields)

LearningRate = 1e-3
Opt = Adam(LearningRate)
Pars = Flux.params(Model)
Flux.@epochs 100 Flux.train!(Loss, Pars, [(FunctionObs, SolutionObs, Sensor)], Opt)
FinalLoss = Loss(FunctionObs, SolutionObs, Sensor)

DensityObs = reshape(FunctionObs, (Nx, Ny, NumFuncObs))

select = 3
heatmap(DensityObs[:, :, select])
s1 = scatter3d(Sensor[1, :], Sensor[2, :], Predictions[select, :], markersize = 0.1, title = "Predictions", legend=false);
s2 = scatter3d(Sensor[1, :], Sensor[2, :], SolutionObs[select, :], markersize = 0.1, title = "Ground Truth", legend=false);
plot(s1, s2, layout = (1, 2))

