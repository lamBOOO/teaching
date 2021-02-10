### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 92b91de6-5fef-11eb-1b89-951c38260bea
begin
	ENV["MPLBACKEND"]="Agg"

	import Pkg
	Pkg.activate(mktempdir())

	# No python env to use Conda.jl
	ENV["PYTHON"]=""
	Pkg.add("PyCall")
	Pkg.build("PyCall")

	Pkg.add("PyPlot")
	using PyPlot
	Pkg.add("Calculus")
	using Calculus
	Pkg.add("PlutoUI")
	using PlutoUI
end

# ╔═╡ 66c10dde-608a-11eb-024e-d7e7ca5c9f53
html"<button onclick='present()'>present</button>"

# ╔═╡ b54c4e40-5fee-11eb-33fd-8da1baf6540d
md"""
# Constrained Optimization: Penalty & Barrier Methods

- Mathe 3 (CES)
- WS20
- Lambert Theisen (```theisen@acom.rwth-aachen.de```)
"""

# ╔═╡ 606a94fc-6af1-11eb-2fcd-eb64c6a490e5
md"""
## System Setup for Binder

- See [@lamBOOO/teaching on Github](https://github.com/lamBOOO/teaching)
"""

# ╔═╡ c8e17df0-6aee-11eb-3c5f-fdf095448056
md"""
## Define optimization problem

```math
\min_{x \in \mathbb{R^n}} f(x)
\; \text{s.t.} \;
\begin{cases}
g_j(x) \le 0  \; \text{for} \; j=1,\dots,m
\\
h_i(x) = 0 \; \text{for} \; i=1,\dots,q
\end{cases}
```
"""

# ╔═╡ 56f36258-6ada-11eb-0d86-37078249bbc9
struct ConstrainedMinimizationProblem
	f::Function
	g::Array{Function,1}
	h::Array{Function,1}
end

# ╔═╡ 63244e3e-6ada-11eb-3ab0-a3e8e98db6f1
p = ConstrainedMinimizationProblem(
	x -> 4*x[1]^2 - x[1] - x[2] - 2.5,
	[
		x -> -(x[2]^2 -1.5*x[1]^2 + 2*x[1] - 1),
		x -> +(x[2]^2 +2*x[1]^2 - 2*x[1] - 4.25),
	],
	[x -> 5*(x[1]+x[2])],
)

# ╔═╡ 17f43d4c-6aef-11eb-375f-b11b1507d7ac
md"""
## Power Penalty Function

```math
P_p(x,\alpha) = f(x) + \alpha r_p(x)
```
with
```math
r_p(x) = \sum_{i=1}^{q} {|h_i(x)|}^p + \sum_{j=1}^{m} {|\max(0, g_j(x))|}^p
```
"""

# ╔═╡ e38a9a46-6adb-11eb-3be3-794a65028b04
function P(x, p::ConstrainedMinimizationProblem, α::Number, pow::Int)
	@assert α>0
	r = (
		reduce(+, [abs(p.h[i](x))^pow for i ∈ 1:length(p.h)], init=0)
		+ reduce(+, [max(0, p.g[i](x))^pow for i ∈ 1:length(p.g)], init=0)
	)
	return p.f(x) + α * r
end

# ╔═╡ b2f8358c-6aef-11eb-3553-3574676ead99
md"""
## Solve and Visualize Convergence History
"""

# ╔═╡ 42a25bd2-6b8a-11eb-30e4-955c53c97eb8
md"""
steps = $(@bind steps Slider(1:1:100; default=50, show_value=true)),
penalty = $(@bind penalty CheckBox(default=false)),
αp = $(@bind αp Slider(0.1:0.1:10; default=1, show_value=true))
"""

# ╔═╡ af2a5a78-6ae2-11eb-1e12-eb4ff27bfc6a
function visualize(
		hists :: Array{Array{Any, 1}, 1},
		p :: ConstrainedMinimizationProblem;
		showotherfunction = nothing,
		mins = [],
		legend = ["history $(i)" for i=1:length(hists)],
		title = "",
	)

	clf()
	fig, ax = PyPlot.subplots()
	Δ = 0.1
	X=collect(-1.5:Δ:2.5)
	Y=collect(-2.5:Δ:2.5)

	# objective
	F = nothing
	if showotherfunction == nothing
		F=[p.f([X[i],Y[j]]) for j=1:length(Y), i=1:length(X)]
	else
		F=[showotherfunction([X[i],Y[j]]) for j=1:length(Y), i=1:length(X)]
	end
	contourf(X, Y, F, levels=20)

	# inequality constraints g
	for gi=1:length(p.g)
		contourf(
			X, Y, [p.g[gi]([X[i],Y[j]]) for j=1:length(Y), i=1:length(X)],
			[-1000,0], alpha=0.2, colors="white"
		)
		CS1 = ax.contour(
			X, Y, [p.g[gi]([X[i],Y[j]]) for j=1:length(Y), i=1:length(X)],
			[-2], colors="white", alpha=0.5, zorder=-3
		)
		ax.clabel(
			CS1, CS1.levels, inline=true, fontsize=10, fmt="g$(gi)<0", zorder=10
		)
		CS2 = ax.contour(
			X, Y, [p.g[gi]([X[i],Y[j]]) for j=1:length(Y), i=1:length(X)],
			[0], colors="white"
		)
		ax.clabel(CS2, CS2.levels, inline=true, fontsize=10, fmt="g$(gi)=0")
	end

	# equality constraints h
	for hi=1:length(p.h)
		CS1 = ax.contour(
			X, Y, [p.h[hi]([X[i],Y[j]]) for j=1:length(Y), i=1:length(X)],
			[0], colors="black"
		)
		ax.clabel(
			CS1, CS1.levels, inline=true, fontsize=10, fmt="h$(hi)=0"
		)
	end

	# history
	hcolors = ["yellow", "lime", "deepskyblue", "fuchsia"]
	for (ihist, hist) in enumerate(hists)
		hist_x = [hist[i][1] for i=1:length(hist)]
		hist_y = [hist[i][2] for i=1:length(hist)]
		plot(hist_x, hist_y, color=hcolors[ihist])
		scatter(hist_x, hist_y, color=hcolors[ihist])
		for i=1:length(hist_x)
			annotate(
				string(i), [hist_x[i], hist_y[i]] + [0.05, 0.05],
				color=hcolors[ihist], zorder=2
			)
		end
		scatter(
			hist_x[end], hist_y[end], color=hcolors[ihist], 
			s=50, edgecolor="black", zorder=5, linewidth=2
		)
	end

	# minima
	for i=1:length(mins)
		ax.scatter(mins[i][1], mins[i][2], color="r", s=500, zorder=6, marker="x")
	end

	# settings
	PyPlot.title(title)
	ax.legend(legend)
	xlabel("x")
	ylabel("y")

	gcf()
end

# ╔═╡ ddb561e0-608c-11eb-0920-074e5a84724e
md"""
## Stepsize Control Algorithm
"""

# ╔═╡ d11545a0-6aef-11eb-015c-b934ec871a7a
function backtracking_linesearch(f, x, d, αmax, cond, β)
	@assert 0 < β < 1
	α = αmax
	while !cond(f, d, x, α)
		α *= β
	end
	return α
end

# ╔═╡ ef6cba94-608c-11eb-06ef-b5af1c5c9662
md"""
## Armijo Stepsize Conditon

- We need to specify a conditon for the backtracking algorithm
- Use Armijo condition, which is the first Wolfe condition

```math
{\displaystyle {\begin{aligned}{\textbf {i)}}&\quad f(\mathbf {x} _{k}+\alpha _{k}\mathbf {p} _{k})\leq f(\mathbf {x} _{k})+c_{1}\alpha _{k}\mathbf {p} _{k}^{\mathrm {T} }\nabla f(\mathbf {x} _{k}),\\[6pt]{\textbf {ii)}}&\quad {-\mathbf {p} }_{k}^{\mathrm {T} }\nabla f(\mathbf {x} _{k}+\alpha _{k}\mathbf {p} _{k})\leq -c_{2}\mathbf {p} _{k}^{\mathrm {T} }\nabla f(\mathbf {x} _{k}),\end{aligned}}}
```

- Also assert that second Wolfe condition is fulfilled
"""

# ╔═╡ e21dfe42-5fef-11eb-22a6-e5db4e09613c
wolfe1(f, d, x, α) = f(x + α*d) <= f(x) + 1E-4 * α * derivative(f, x)' * d

# ╔═╡ 28aa6a7a-6af0-11eb-0880-8ffcba817081
wolfe2(f, d, x, α) = derivative(f, x+α*d)' * d >= 0.99 * derivative(f, x)' * d

# ╔═╡ bfcac302-5fef-11eb-19ef-bdde45ad188f
function backtracking_linesearch_wolfe(f, x, d, αmax, β)
	# @assert wolfe2(f, d, x, backtracking_linesearch(f, x, d, αmax, wolfe1, β))
	return backtracking_linesearch(f, x, d, αmax, wolfe1, β)
end

# ╔═╡ 29ca1ff8-608d-11eb-2b01-b954fcd2de76
md"""
## Use Backtracking Algorithm in Gradient Descent
"""

# ╔═╡ 5c83c5e6-5fef-11eb-1a0e-3d9e19a8874b
function gradient_descent_wolfe(f, x0, kmax)
	x = x0
	hist = []
	push!(hist, x)
	for k=1:kmax
		x = x + backtracking_linesearch_wolfe(
			f, x, -derivative(f, x), 1, 0.9
		) * -derivative(f, x)
		push!(hist, x)
	end
	return x, hist
end

# ╔═╡ 665d6f78-6ae8-11eb-2a9e-1b6d0536cb74
visualize([
		gradient_descent_wolfe(x->P(x, p, 1, 1), [1,2], steps)[2],
		gradient_descent_wolfe(x->P(x, p, 10, 1), [1,2], steps)[2],
], p, legend = ["1-power, α=1", "1-power, α=10"],
	showotherfunction = if penalty x->P(x, p, αp, 1) else nothing end
)

# ╔═╡ b1b0e32e-6af7-11eb-1462-d32d05f84416
md"""
## TODO: Implement Barrier Methods
"""

# ╔═╡ b9370574-608e-11eb-00d2-e134ec05d7db
md"""
## See you $(html\"<span style='color:#FF0000';>NOT</span>\") next week ✌️

Questions?

!!! danger "Exercises over"

    This was the last exercise on Wednesday 

!!! tip "Exam Questions Session"

    15.03.21 15:00 Exam Questions Session (see Moodle)

"""



# ╔═╡ Cell order:
# ╟─66c10dde-608a-11eb-024e-d7e7ca5c9f53
# ╟─b54c4e40-5fee-11eb-33fd-8da1baf6540d
# ╟─606a94fc-6af1-11eb-2fcd-eb64c6a490e5
# ╠═92b91de6-5fef-11eb-1b89-951c38260bea
# ╟─c8e17df0-6aee-11eb-3c5f-fdf095448056
# ╠═56f36258-6ada-11eb-0d86-37078249bbc9
# ╠═63244e3e-6ada-11eb-3ab0-a3e8e98db6f1
# ╟─17f43d4c-6aef-11eb-375f-b11b1507d7ac
# ╠═e38a9a46-6adb-11eb-3be3-794a65028b04
# ╟─b2f8358c-6aef-11eb-3553-3574676ead99
# ╟─42a25bd2-6b8a-11eb-30e4-955c53c97eb8
# ╠═665d6f78-6ae8-11eb-2a9e-1b6d0536cb74
# ╟─af2a5a78-6ae2-11eb-1e12-eb4ff27bfc6a
# ╟─ddb561e0-608c-11eb-0920-074e5a84724e
# ╠═d11545a0-6aef-11eb-015c-b934ec871a7a
# ╟─ef6cba94-608c-11eb-06ef-b5af1c5c9662
# ╠═e21dfe42-5fef-11eb-22a6-e5db4e09613c
# ╠═28aa6a7a-6af0-11eb-0880-8ffcba817081
# ╠═bfcac302-5fef-11eb-19ef-bdde45ad188f
# ╟─29ca1ff8-608d-11eb-2b01-b954fcd2de76
# ╠═5c83c5e6-5fef-11eb-1a0e-3d9e19a8874b
# ╟─b1b0e32e-6af7-11eb-1462-d32d05f84416
# ╟─b9370574-608e-11eb-00d2-e134ec05d7db
