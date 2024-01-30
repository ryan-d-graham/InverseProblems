using Integrals
using PlotlyJS

#setup solution integral for Poisson Equation ∇²u = -f
f(r) = exp(-10.0 * sum(abs2, r)) #forcing function
G(r, r′) = 1.0 / sum(abs, r - r′) #Green's function for unbounded Poisson Equation
integrand(r′, r) = f(r′) * G(r, r′) #convolution integrand

r₀ = ones(3) #solution eval point
prob = IntegralProblem(integrand, [-Inf, -Inf, -Inf], [Inf, Inf, Inf], p = r₀)
sol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)
sol.u

function evalSol(r)
    prob = IntegralProblem(integrand, [-Inf, -Inf, -Inf], [Inf, Inf, Inf], p = r)
    solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)[1]
end

nx, ny, nz = 20, 20, 20
xs = range(-1, 1, nx)
ys = range(-1, 1, ny)
zs = range(-1, 1, nz)
X, Y, Z = mgrid(xs, ys, zs)
U = rand(nx, ny, nz)

for i ∈ 1:nx
    for j ∈ 1:ny
        for k ∈ 1:nz
            U[i, j, k] = evalSol([xs[i], ys[j], zs[k]])
        end
    end
end

PlotlyJS.plot(PlotlyJS.volume(x = X[:], y = Y[:], z = Z[:], value = U[:], isomin = 0.001, isomax = 1.0, opacity = 0.1, surface_count = 40))
