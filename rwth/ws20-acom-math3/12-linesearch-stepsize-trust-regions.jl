### A Pluto.jl notebook ###
# v0.12.18

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
	using PyPlot
	using Calculus, PlutoUI
end

# ╔═╡ 66c10dde-608a-11eb-024e-d7e7ca5c9f53
html"<button onclick='present()'>present</button>"

# ╔═╡ b54c4e40-5fee-11eb-33fd-8da1baf6540d
md"""
# Line Search Stepsize Control and Trust-Region Methods

- Mathe 3 (CES)
- WS20
- Lambert Theisen (```theisen@acom.rwth-aachen.de```)
"""

# ╔═╡ ddb561e0-608c-11eb-0920-074e5a84724e
md"""
## Stepsize Control Algorithm
"""

# ╔═╡ 6c22d348-5f13-11eb-1a98-eb6313fcf858
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
"""

# ╔═╡ e21dfe42-5fef-11eb-22a6-e5db4e09613c
armijo(f, d, x, α) = f(x + α*d) <= f(x) + 1E-4 * α * derivative(f, x)' * d

# ╔═╡ bfcac302-5fef-11eb-19ef-bdde45ad188f
function backtracking_linesearch_armijo1(f, x, d, αmax, β)
	return backtracking_linesearch(f, x, d, αmax, armijo, β)
end

# ╔═╡ 29ca1ff8-608d-11eb-2b01-b954fcd2de76
md"""
## Use Backtracking Algorithm in Gradient Descent

- Same as last week, but with adaptive step size
"""

# ╔═╡ 5c83c5e6-5fef-11eb-1a0e-3d9e19a8874b
function gradient_descent_armijo1(f, x0, kmax)
	x = x0
	hist = []
	push!(hist, x)
	for k=1:kmax
		x = x + backtracking_linesearch_armijo1(
			f, x, -derivative(f, x), 2, 0.5
		) * -derivative(f, x)
		push!(hist, x)
	end
	return x, hist
end

# ╔═╡ 434c3160-608c-11eb-2160-d3405ce05327
md"""
## Rosenbrock: GD with Armijo

- Remember from last week: GD was very sensitive to step width
- Now: Line search automatically choose a valid step size and we have an easy life
"""

# ╔═╡ 82a2e0e0-5fef-11eb-2dfb-b7644e600e48
begin
	# Rosenbrock function with x* = [a,a^2], f(x*)=0
	a = 1
	b = 100
	h = (x -> (a-x[1])^2 + b*(x[2]-x[1]^2)^2)
	
	x0 = [-1.,0.]
	
	# Gradient Descent without Armijo1
	res_gd_2d_rb_arm1 = gradient_descent_armijo1(h, x0, 100)
	res_gd_2d_rb_arm1_x = [
		res_gd_2d_rb_arm1[2][i][1] for i=1:length(res_gd_2d_rb_arm1[2])
	]
	res_gd_2d_rb_arm1_y = [
		res_gd_2d_rb_arm1[2][i][2] for i=1:length(res_gd_2d_rb_arm1[2])
	]
	
	# Gradient Descent with Armijo1
	res_gd_2d_rb_arm1 = gradient_descent_armijo1(h, x0, 100)
	res_gd_2d_rb_arm1_x = [
		res_gd_2d_rb_arm1[2][i][1] for i=1:length(res_gd_2d_rb_arm1[2])
	]
	res_gd_2d_rb_arm1_y = [
		res_gd_2d_rb_arm1[2][i][2] for i=1:length(res_gd_2d_rb_arm1[2])
	]
	
	clf()
	Δ = 0.1
	X=collect(-2:Δ:2)
	Y=collect(-1:Δ:3)
	F=[h([X[j],Y[i]]) for i=1:length(X), j=1:length(Y)]
	contourf(X,Y,F, levels=50)
	PyPlot.title("Rosenbrock: Gradient Descent with Armoji Linesearch")
	
	# res_gd_2d_rb
	PyPlot.plot(res_gd_2d_rb_arm1_x, res_gd_2d_rb_arm1_y, color="yellow")
	scatter(res_gd_2d_rb_arm1_x, res_gd_2d_rb_arm1_y, color="yellow")
	for i=1:length(res_gd_2d_rb_arm1_x)
		annotate(string(i), [res_gd_2d_rb_arm1_x[i], res_gd_2d_rb_arm1_y[i]], color="w", zorder=2)
	end
	
	legend(["Gradient Descent with Armoji"])
	
	xlabel("x")
	ylabel("y")
		
	# Mark minimum
	scatter(a, a^2, color="r", s=500, zorder=3, marker="x")
	
	gcf()
end

# ╔═╡ 2b218df6-608c-11eb-01f7-0bdb99d010e6
md"""
## Still not the Best Convergence...
"""

# ╔═╡ a372c9f0-5ff1-11eb-05a4-23ac4164dbf5
res_gd_2d_rb_arm1[2][end] # still not converged after 100 its 😓

# ╔═╡ 0511ee5e-5ff3-11eb-13d3-1b164a48295e
md"""
## Trust-Region Methods

1. Given ``x^{(k)}``
1. Replace ``f`` by (e.g 2nd order) approximation ``\hat f``
1. Solve ``\hat x = \text{argmin}_{x \in D_k} \hat f(x)`` for a given thrust region ``D_k = \{x \in \mathbb{R}^n \mid \|x-x^{(k)}\|_p \le \delta\}``
1. Test improvement ``\rho = \frac{\text{actual improvement}}{\text{predicted improvement}} = \frac{f(x^k) - f(\hat x)}{f(x^k) - \hat f(\hat x)}``
1. If ``\rho > \rho_\min``, set ``x^{(k+1)} = \hat x``, else decrease thrust region radius ``\delta \leftarrow \sigma \delta``
"""

# ╔═╡ b7ca5aa8-6084-11eb-2a6b-cf5c83ceb476
function trust_region(
	f, fhat, x0, solve_subproblem, kmax, rhomin, delta0, sigma
)
	println("START")
	hist = []
	x = x0
	push!(hist, [x0, 0])
	for k=1:kmax
		@show k
		delta = delta0
		@show delta
		xhatval = nothing
		xhatval = solve_subproblem(x, delta)
		@show xhatval
		rho = (f(x) - f(xhatval)) / (f(x) - fhat(xhatval, x))
		@show rho
		i = 0
		while rho < rhomin && i<10
			delta *= sigma
			@show delta
			xhatval = solve_subproblem(x, delta)
			@show xhatval
			rho = (f(x) - f(xhatval)) / (f(x) - fhat(xhatval, x))
			@show rho
			i += 1
		end
		@show delta
		x = xhatval
		@show x
		push!(hist, [x, delta])
	end
	return x, hist
end

# ╔═╡ 66e3d78a-608d-11eb-13de-bb7401f11cc3
md"""
## Define Problem
"""

# ╔═╡ c76e9826-608b-11eb-29fb-b1bb0aa9791f
md"""
- Define objective: ``f(x,y) = x^2 + y^2(y^2-1)``
- Derive quadratic approximation ``\hat f = \hat f(x) := f(x^{(k)}) + (x- x^{(k)})^T\nabla f(x^{(k)}) + \frac{1}{2} (x- x^{(k)})^T \nabla^2 f(x^{(k)}) (x- x^{(k)})``
- Minima are at ``(0,\pm 1/\sqrt{6})``, saddle point at ``(0,0)``
"""

# ╔═╡ aaecb90a-6084-11eb-179a-075a10e7f696
# objective
f(x) = x[1]^2 + x[2]^2 * (x[2]^2 - 1)

# ╔═╡ af057d1a-6084-11eb-1900-397644eee27b
# quadratic approximation
fhat(x, x0) = (
	f(x0) + (x-x0)' * derivative(f, x0) 
	+ 1/2 * (x-x0)' * hessian(f, x0) * (x-x0)
)

# ╔═╡ 8a7b5452-608d-11eb-1b6e-b5b622ea4c17
md"""
## Define Solution to Subproblem

- Either analytically (see below)
- Or use approximate solutions (Cauchy point, ...)
"""

# ╔═╡ b26e59f4-6084-11eb-027e-13c287224ff5
solve_subproblem(x, delta) = [
	if (abs(x[1]) <= delta) 
		0 
	else 
		x[1] - sign(x[1])*delta 
	end,
	if (x[2] == 0)
		if (abs(x[2]) <= delta)
			delta
		else 
			x[2] + sign(x[2]) * delta
		end
	elseif (x[2]^2 >= 1/6)
		if (abs(x[2] - (4*x[2]^3)/(6*x[2]^2-1)) <= delta)
			(4*x[2]^3)/(6*x[2]^2-1)
		else
			x[2] - sign(x[2] - (4*x[2]^3)/(6*x[2]^2-1)) * delta
		end
	else
		nothing
	end
]

# ╔═╡ a2745374-608d-11eb-337a-11d31e23b587
md"""
## Test Thrust-Region Method with Saddle Point

- We can escape the saddle point ``x^{(0)} = (0,0)`` 👏
"""

# ╔═╡ de79d384-6084-11eb-1fb8-03ed377ca699
trust_region(f, fhat, [0.,0.], solve_subproblem, 5, 0.5, 0.5, 0.5)

# ╔═╡ b971c04a-608b-11eb-361c-0309cf3ddaa0
md"""
## Trust-Region Method in Action 😎
"""

# ╔═╡ 7364adfc-608a-11eb-0592-91691e0fb644
md"""
x01 = $(@bind x01 NumberField(1:0.1:10; default=1))
x02 = $(@bind x02 NumberField(0.4:0.1:10; default=0.5))
k = $(@bind k NumberField(1:10; default=5))
rhomin = $(@bind rhomin NumberField(0.01:0.1:0.99; default=0.99))
delta0 = $(@bind delta0 NumberField(0.1:0.1:2; default=0.3))
sigma = $(@bind sigma NumberField(0.01:0.1:0.99; default=0.5))
"""

# ╔═╡ fb7d613a-6084-11eb-04a8-812c1b633f3e
let
	# Perform Optimization
	tr = trust_region(f, fhat, [Float64(x01),Float64(x02)], solve_subproblem, k, rhomin, delta0, sigma)
	tr_x = [
		tr[2][i][1][1] for i=1:length(tr[2])
	]
	tr_y = [
		tr[2][i][1][2] for i=1:length(tr[2])
	]
	deltas = [
		tr[2][i][2][1] for i=1:length(tr[2])
	]
	
	# Plot annotations
	clf()
	ax = gca()
	Δ = 0.1
	X=collect(-2:Δ:2)
	Y=collect(-2:Δ:2)
	F=[f([X[j],Y[i]]) for i=1:length(Y), j=1:length(X)]
	contourf(X,Y,F, levels=50)
	PyPlot.title("2nd-Order Trust-Region Method (Analytic)")
	
	# Trust Regions
	for i=2:length(tr_x)
		ax.add_patch(PyPlot.matplotlib.pyplot.Rectangle((tr_x[i-1]-deltas[i], tr_y[i-1]-deltas[i]), 2deltas[i], 2deltas[i], facecolor="red", alpha=0.2, edgecolor="black", linewidth=2.))
	end
	
	# Trajectory
	PyPlot.plot(tr_x, tr_y, color="yellow", zorder=2)
	scatter(tr_x, tr_y, color="yellow", zorder=2)
	for i=1:length(tr_x)
		annotate(string(i), [tr_x[i], tr_y[i]], color="w", zorder=3)
	end
	
	# Plot annotations
	legend(["2nd-Order Trust-Region Method (Analytic)"])
	xlabel("x")
	ylabel("y")
		
	# Mark minima
	scatter(0, 1/sqrt(2), color="r", s=500, zorder=3, marker="x")
	scatter(0, -1/sqrt(2), color="r", s=500, zorder=3, marker="x")
	
	gcf()
end

# ╔═╡ b9370574-608e-11eb-00d2-e134ec05d7db
md"""
## See you next week ✌️

Questions?

"""

# ╔═╡ Cell order:
# ╟─66c10dde-608a-11eb-024e-d7e7ca5c9f53
# ╟─b54c4e40-5fee-11eb-33fd-8da1baf6540d
# ╟─92b91de6-5fef-11eb-1b89-951c38260bea
# ╟─ddb561e0-608c-11eb-0920-074e5a84724e
# ╠═6c22d348-5f13-11eb-1a98-eb6313fcf858
# ╟─ef6cba94-608c-11eb-06ef-b5af1c5c9662
# ╠═e21dfe42-5fef-11eb-22a6-e5db4e09613c
# ╠═bfcac302-5fef-11eb-19ef-bdde45ad188f
# ╟─29ca1ff8-608d-11eb-2b01-b954fcd2de76
# ╠═5c83c5e6-5fef-11eb-1a0e-3d9e19a8874b
# ╟─434c3160-608c-11eb-2160-d3405ce05327
# ╟─82a2e0e0-5fef-11eb-2dfb-b7644e600e48
# ╟─2b218df6-608c-11eb-01f7-0bdb99d010e6
# ╠═a372c9f0-5ff1-11eb-05a4-23ac4164dbf5
# ╟─0511ee5e-5ff3-11eb-13d3-1b164a48295e
# ╠═b7ca5aa8-6084-11eb-2a6b-cf5c83ceb476
# ╟─66e3d78a-608d-11eb-13de-bb7401f11cc3
# ╟─c76e9826-608b-11eb-29fb-b1bb0aa9791f
# ╠═aaecb90a-6084-11eb-179a-075a10e7f696
# ╠═af057d1a-6084-11eb-1900-397644eee27b
# ╟─8a7b5452-608d-11eb-1b6e-b5b622ea4c17
# ╠═b26e59f4-6084-11eb-027e-13c287224ff5
# ╟─a2745374-608d-11eb-337a-11d31e23b587
# ╠═de79d384-6084-11eb-1fb8-03ed377ca699
# ╟─b971c04a-608b-11eb-361c-0309cf3ddaa0
# ╟─7364adfc-608a-11eb-0592-91691e0fb644
# ╠═fb7d613a-6084-11eb-04a8-812c1b633f3e
# ╟─b9370574-608e-11eb-00d2-e134ec05d7db
