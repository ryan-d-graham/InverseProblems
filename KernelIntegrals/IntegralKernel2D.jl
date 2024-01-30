using Integrals
using Plots

#setup solution integral for Poisson Equation ∇²u = -f
f(r) = -exp(-sum(abs2, r)) #forcing function
G(r, r′) = (1.0 / (2*π)) * log(sum(abs, r - r′)) #Green's function for unbounded Poisson Equation
integrand(r′, r) = f(r′) * G(r, r′) #convolution integrand

r₀ = ones(2) #solution eval point
prob = IntegralProblem(integrand, [-Inf, -Inf], [Inf, Inf], p = r₀)
sol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)
sol.u

function evalSol(r)
    prob = IntegralProblem(integrand, [-Inf, -Inf], [Inf, Inf], p = r)
    solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)[1]
end

nx, ny= 100, 100
xs = range(-10, 10, nx)
ys = range(-10, 10, ny)
U = rand(nx, ny)

for i ∈ 1:nx
    for j ∈ 1:ny
        U[i, j] = evalSol([xs[i], ys[j]])
    end
end

Plots.heatmap(U)
Plots.surface(xs, ys, U)