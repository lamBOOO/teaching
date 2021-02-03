### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 98bb2f16-6573-11eb-396a-b9b1244088ba
using PlutoUI, SymPy, PyPlot, LinearAlgebra

# ╔═╡ 17c7a6d8-660d-11eb-3312-8712778733fb
html"<button onclick='present()'>present</button>"

# ╔═╡ 31e52238-6578-11eb-181a-339a1afd7e0d
md"""
# Constrained Optimziation: KKT & LICQ

- Mathe 3 (CES)
- WS20
- Lambert Theisen (```theisen@acom.rwth-aachen.de```)
"""

# ╔═╡ 921a1b84-660b-11eb-0371-73624873f65f
md"""
## Use `SymPy` symbolic library

- Is a wrapper to Python's `SymPy`
- Using Python directly would be (probably) better
- But we don't want to loose the luxury of `Pluto.jl`
"""

# ╔═╡ 83b62740-660b-11eb-3974-4fe196977c37
md"""
## Define Variables and Lagrange Multipliers
"""

# ╔═╡ 2d6c879a-6605-11eb-1c47-457472258b66
x = [
	symbols("x1", real=true),
	symbols("x2", real=true),
]

# ╔═╡ 07f5fc28-660d-11eb-2df5-5531eaa939a6
lambdas = [
	symbols("lambda1", real=true, nonnegative=true),
	symbols("lambda2", real=true, nonnegative=true),
	symbols("lambda3", real=true, nonnegative=true),
	symbols("lambda4", real=true, nonnegative=true),
]

# ╔═╡ 0d04651a-660d-11eb-23a4-e5c7d7b64d25
mus = [
	symbols("mu")
]

# ╔═╡ 621fc036-660c-11eb-267f-dfacf5c3a5f1
md"""
## Define Objective and Constraints
"""

# ╔═╡ a2a1cbd0-6603-11eb-09a8-67dcc6b79a00
f1(x) = (x[1]-1)^2 + (x[2]-2)^2

# ╔═╡ 5c2ceeba-660c-11eb-3dc9-f794a5f63e91
f1(x)

# ╔═╡ 43078772-660c-11eb-02d8-b57d829f1e7f
md"""
**Inequality Constraints**: $g_i(x) \ge 0$
"""

# ╔═╡ 6022e976-6575-11eb-1755-8f7e577f30d5
g = [
	x -> 1 - x[1] - x[2],
	x -> 1 - x[1] + x[2],
	x -> 1 + x[1] - x[2],
	x -> 1 + x[1] + x[2],
]

# ╔═╡ 35dffc98-660c-11eb-29b0-c31142c939d5
md"""
**Equality Constraints**: $h_j(x) = 0$
"""

# ╔═╡ ae4c9730-6603-11eb-3721-877881d04cc7
h = [
	x -> (x[1]-1)^2 - 5*x[2],  # == 0
]

# ╔═╡ 0b45ae42-660c-11eb-07ba-531fe5ed2ed3
md"""
## Define Lagrangian

```math
\mathcal{L}(x,\lambda,\mu) = 
f(x) 
- \sum_{i=1}^{m} \lambda_i g_i(x)
- \sum_{j=1}^{q} \mu_j h_j(x)
```
"""

# ╔═╡ 27a48782-6587-11eb-35bc-45bb830139f1
function lagrangian(x, f, g, h, λs, μs, Ig)
	# TODO: Include only active gs with Ig
	return (
		f(x) 
		# - reduce(+, [g[i](x) * λs[i] for i=1:size(g)[1]], init=0)  # TODO: Include
		- reduce(+, [h[i](x) * μs[i] for i=1:size(h)[1]], init=0)  # sum
	)
end

# ╔═╡ 244ed6ce-660d-11eb-0e19-a17ccd2bac15
lagrangian(x, f1, [], h, lambdas, mus, [])

# ╔═╡ a8041c28-660c-11eb-28e6-4dfb1e6c1e15
md"""
## KKT Points 

KKT points $(x^*, \lambda^*, \mu^*)$ fulfill:

1. $\nabla_{x} \mathcal{L}(x,\lambda,\mu) = 0$
1. $h_j(x) = 0 \quad \forall j=1,\dots,q$
1. $g_i(x) \ge 0 \quad \forall i=1,\dots,m$
1. $\lambda_i \ge 0 \quad \forall i=1,\dots,m$
1. $g_i(x) \lambda_i = 0 \quad \forall i=1,\dots,m$
"""

# ╔═╡ 526e80d4-6605-11eb-3983-753e63780198
function kktpoints(x, f, g, h, λs, μs, Ig)
	lag = lagrangian(x, f, g, h, λs, μs, Ig)
	return solve([
		diff(lag, x[1]),
		diff(lag, x[2]),
		diff(lag, mus[1]),  # <=> h_i(x)==0
		# TODO: Include ineq. constraints g
	])
end

# ╔═╡ ccc3c252-660c-11eb-0d7a-bb55b87645cb
md"""
## Test KKT Points
"""

# ╔═╡ 77d2c740-6605-11eb-3f3b-d79f8fd7faff
kktpoints(x,f1,[],h,lambdas,mus,[])

# ╔═╡ 2b2cb8ec-660f-11eb-247e-af907d1fba56
h[1]([1,0])  # eq constraint fulfilled

# ╔═╡ 39b82cf2-660f-11eb-0ab5-07306353cf5e
f1([1,0])-f1([1.02,0])  # looks promising

# ╔═╡ daa51c5e-660c-11eb-36aa-dba0f0129bb8
md"""
## Visualize KKT Points
"""

# ╔═╡ d5a7791e-6606-11eb-1b71-dde9e7393c05
begin
	
	kktpts = kktpoints(x,f1,[],h,lambdas,mus,[])[1]
	
	# Plot annotations
	clf()
	ax = gca()
	
	Δ = 0.1
	X=collect(-3:Δ:3)
	Y=collect(-4:Δ:4)
	
	F=[f1([X[j],Y[i]]) for i=1:length(Y), j=1:length(X)]
	
	contour(X,Y,F, levels=50)
	
	PyPlot.plot(X, [Float64(solve(h[1](x),x[2]).subs.(x[1], val)[1][1]) for val in X], color="blue")
	scatter(kktpts[x[1]], kktpts[x[2]], color="r", s=500, zorder=3, marker="x")
	legend(["Equality Constraint", "KKT Points"])
	
	PyPlot.title("NLP: KKT Points")
	
	gcf()
end

# ╔═╡ bfede808-6609-11eb-3d74-e73170fa62a3
md"""
## Linear Independence Constraint Quality (LICQ)

Point $x \in \chi$ satisfies LICQ if:

```math
{\left\{ \nabla h_j(x) \right\}}_{j=1}^{q}, {\left\{ \nabla g_i(x) \right\}}_{i \in I_g(x)}
```

are linearly independent. The set of active inequality constraints at point $x$ is labelled with $I_g(x)$.
"""

# ╔═╡ e69b65a4-660c-11eb-1e9e-eb220ba32d68
md"""
**Index Set of Active Constraints**:
"""

# ╔═╡ b84179a6-6575-11eb-23d3-53d1f4121a78
function Ig(x,g) 
	return [i for i=1:size(g)[1] if g[i](x)==0]
end

# ╔═╡ cb8fdbe2-6575-11eb-356a-173604653340
function LICQ(ξ, g, Ig, h)
	set = hcat(
		# [diff(g[i](x), x) for i ∈ Ig(ξ,g)],
		[diff(h[i](x), x) for i ∈ 1:size(h)[1]]
	)
	return set
end

# ╔═╡ 6d7ec488-660f-11eb-37c2-6ff8df87ea96
md"""
## Test LICQ in potential KKT Point
"""

# ╔═╡ c0483e74-660a-11eb-12cc-4d7b7492ddcf
set = LICQ([kktpts[x[1]], kktpts[x[2]]], [], [], h)[1]

# ╔═╡ a9b1aca6-660d-11eb-2196-ed4f9f55744e
set.rank()  # full rank => linearly independent

# ╔═╡ b83f3da8-660e-11eb-376f-8d6c7ed189fb
md"""
## See you next week ✌️

- Questions?
- Homework: Include the inequality constraints into the above code 😉

"""

# ╔═╡ Cell order:
# ╟─17c7a6d8-660d-11eb-3312-8712778733fb
# ╟─31e52238-6578-11eb-181a-339a1afd7e0d
# ╟─921a1b84-660b-11eb-0371-73624873f65f
# ╠═98bb2f16-6573-11eb-396a-b9b1244088ba
# ╟─83b62740-660b-11eb-3974-4fe196977c37
# ╠═2d6c879a-6605-11eb-1c47-457472258b66
# ╠═07f5fc28-660d-11eb-2df5-5531eaa939a6
# ╠═0d04651a-660d-11eb-23a4-e5c7d7b64d25
# ╟─621fc036-660c-11eb-267f-dfacf5c3a5f1
# ╠═a2a1cbd0-6603-11eb-09a8-67dcc6b79a00
# ╠═5c2ceeba-660c-11eb-3dc9-f794a5f63e91
# ╟─43078772-660c-11eb-02d8-b57d829f1e7f
# ╟─6022e976-6575-11eb-1755-8f7e577f30d5
# ╟─35dffc98-660c-11eb-29b0-c31142c939d5
# ╠═ae4c9730-6603-11eb-3721-877881d04cc7
# ╟─0b45ae42-660c-11eb-07ba-531fe5ed2ed3
# ╠═27a48782-6587-11eb-35bc-45bb830139f1
# ╠═244ed6ce-660d-11eb-0e19-a17ccd2bac15
# ╟─a8041c28-660c-11eb-28e6-4dfb1e6c1e15
# ╠═526e80d4-6605-11eb-3983-753e63780198
# ╟─ccc3c252-660c-11eb-0d7a-bb55b87645cb
# ╠═77d2c740-6605-11eb-3f3b-d79f8fd7faff
# ╠═2b2cb8ec-660f-11eb-247e-af907d1fba56
# ╠═39b82cf2-660f-11eb-0ab5-07306353cf5e
# ╟─daa51c5e-660c-11eb-36aa-dba0f0129bb8
# ╟─d5a7791e-6606-11eb-1b71-dde9e7393c05
# ╟─bfede808-6609-11eb-3d74-e73170fa62a3
# ╟─e69b65a4-660c-11eb-1e9e-eb220ba32d68
# ╠═b84179a6-6575-11eb-23d3-53d1f4121a78
# ╠═cb8fdbe2-6575-11eb-356a-173604653340
# ╟─6d7ec488-660f-11eb-37c2-6ff8df87ea96
# ╠═c0483e74-660a-11eb-12cc-4d7b7492ddcf
# ╠═a9b1aca6-660d-11eb-2196-ed4f9f55744e
# ╟─b83f3da8-660e-11eb-376f-8d6c7ed189fb
