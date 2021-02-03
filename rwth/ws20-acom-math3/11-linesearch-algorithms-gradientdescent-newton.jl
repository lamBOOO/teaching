### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 3c142764-5a73-11eb-0540-b7b14a6d5d3f
using PlutoUI, Calculus, Gadfly, LinearAlgebra

# ╔═╡ 8999380a-5af8-11eb-047f-7d4006b15f48
begin
	ENV["MPLBACKEND"]="Agg"
	using PyPlot
end

# ╔═╡ 4bb69ec4-5a76-11eb-2c4f-cd9227558dc0
md"""
# Line Search Algorithm for Optimization

- Mathe 3 (CES)
- WS20
- Lambert Theisen (```theisen@acom.rwth-aachen.de```)
"""

# ╔═╡ de02a6b8-5a78-11eb-1fde-53c163af27a2
md"""
# Define Objective

$f(x) = x^2$
"""

# ╔═╡ b404175c-5a73-11eb-18d9-7ba65f2fec52
f = (x -> x[1]^2)

# ╔═╡ f2a8118e-5a78-11eb-2acb-550e4172884d
md"""
## Line Search
1. Given $x^{(0)}$
1. For $k+0,1,2,\dots$ do
    1. Update: $x^{(k+1)} = x^{(k)} + \alpha_k d^{(k)}$
1. End
"""

# ╔═╡ cffc768e-5a73-11eb-0717-0f031026eff4
function line_search(f, x0, α, d, kmax)
	x = x0
	hist = []
	push!(hist, x)
	for k=1:kmax
		x = x + α(x) * d(x)
		push!(hist, x)
	end
	return x, hist
end

# ╔═╡ 40ffbe2c-5a79-11eb-3b1c-177a10981882
md"""
## Check Line Search

- Observe that different step sizes change the result!
"""

# ╔═╡ 0dd16d7a-5a74-11eb-3e89-49bfba6dfb5a
line_search(f, 1, (x->1), (x->-sign(x)), 10)

# ╔═╡ 52ec352c-5a79-11eb-165e-ab8facbea08d
line_search(f, 1, (x->2), (x->-sign(x)), 10)

# ╔═╡ 5cc7b22c-5a79-11eb-0d88-4f59f5ca0811
md"""
## Gradient Descent

- Is basically line search with $d^{(k)} = - \nabla f(x^{(k)})$
"""

# ╔═╡ 7ebce1be-5a7a-11eb-2082-6feaa8539ed1
begin
	# some notation
	∇ = derivative
	∇² = hessian
end

# ╔═╡ 84ab696e-5a74-11eb-3516-9963af72f1fb
function gradient_descent(f, x0, α, kmax)
	return line_search(f, x0, α, (x->-∇(f, x)), kmax)
end

# ╔═╡ 772238e0-5a79-11eb-318b-370716fa392e
md"""
## Check Gradient Descent
"""

# ╔═╡ a496bca8-5a74-11eb-36ab-4d42ff3206e9
gradient_descent(f, 1, (x->0.1), 100)

# ╔═╡ 7faedaae-5a79-11eb-063e-6f051f747ebc
gradient_descent(f, 1, (x->0.9), 100) # slower, oscillating but converging

# ╔═╡ 948ebe32-5a79-11eb-3193-e17be931cba0
md"""
## Newton's Method for Optimization

- Is line search with $d^{(k)} = - {[\nabla^2 f(x^{(k)})]}^{-1} \nabla f(x^{(k)})$
"""

# ╔═╡ ae8f5574-5a74-11eb-378f-d7b2a9c59575
function newton(f, x0, α, kmax)
	return line_search(f, x0, α, (x->-inv(∇²(f, x))*∇(f, x)), kmax)
end

# ╔═╡ a79842b4-5a79-11eb-2b8d-47c8ba6c75dd
md"""
## Check Newton's Method
"""

# ╔═╡ 08ad3e7c-5a75-11eb-208a-a509dbe12e6e
newton(f, 1., (x->1.0), 100) # works well 😎

# ╔═╡ adf08aac-5a79-11eb-15c7-8171dd96e851
newton(f, 1., (x->3.0), 100) # diverged 😓

# ╔═╡ c62a0d8a-5a76-11eb-120f-99169f32365d
md"""
## Visualize Results
"""

# ╔═╡ 00231932-5a77-11eb-1e22-9b0237c2490e
begin
	res_n = newton(f, 1., (x->0.4), 5)
	Gadfly.plot(
		Guide.title("Newton Algorithm"),
		layer(f, minimum(res_n[2]), maximum(res_n[2])),
		layer(x=res_n[2], y=f.(res_n[2]), label=string.(1:length(res_n[2])), Geom.point, Geom.path, Geom.label, Theme(default_color=color("red")))
	)
end

# ╔═╡ e3a00fa6-5a79-11eb-1ada-a1695a4e237b
begin
	res_gd = gradient_descent(f, 1., (x->0.4), 5)
	Gadfly.plot(
		Guide.title("Gradient Descent Algorithm"),
		layer(f, minimum(res_gd[2]), maximum(res_gd[2])),
		layer(x=res_gd[2], y=f.(res_gd[2]), label=string.(1:length(res_gd[2])), Geom.point, Geom.path, Geom.label, Theme(default_color=color("red")))
	)
end

# ╔═╡ b8676c56-5a76-11eb-390f-e9d38cb9c783
md"""
## Two-Dimensional Optimization
"""

# ╔═╡ 09133588-5a7a-11eb-11ec-cfbff9d0a488
md"""
## Define Objective

$g(x,y) = x^2 + y^2$
"""

# ╔═╡ 7f47f314-5a76-11eb-040d-cf1d3b80fabf
g = (x->x[1]^2+x[2]^2)

# ╔═╡ 166d376a-5a7a-11eb-10a0-c950dd79b520
md"""
## Check Methods

- both work
"""

# ╔═╡ ae0bb618-5a76-11eb-31d1-23b0130b1bdf
res_gd_2d = gradient_descent(g, [1.,1.], (x->0.4), 100)

# ╔═╡ 89c29786-5a76-11eb-24aa-e373bde1dd3d
res_n_2d = newton(g, [1.,1.], (x->0.4), 100)

# ╔═╡ 3f9451ea-5a7b-11eb-15c3-f7be154ca0f1
res_n_2d[2][end]

# ╔═╡ 1ad70922-5a7b-11eb-1834-6d781d2e6d0a
norm(res_n_2d[2][end] - [0,0]) < eps(Float64) # is converged to machine-precision?

# ╔═╡ 6ca5ad54-5af9-11eb-3e33-d91c9c54484c
md"""
## Test Gradient Descent vs Newton for 2D Rosenbrock
"""

# ╔═╡ 9b20248a-5af8-11eb-2078-6b7283933b8d
begin
	# Rosenbrock function with x* = [a,a^2], f(x*)=0
	a = 1
	b = 100
	h = (x -> (a-x[1])^2 + b*(x[2]-x[1]^2)^2)
	
	x0 = [-1.,0.]
	
	# Gradient Descent
	res_gd_2d_rb = gradient_descent(h, x0, (x->0.002), 20)
	res_gd_2d_rb_x = [res_gd_2d_rb[2][i][1] for i=1:length(res_gd_2d_rb[2])]
	res_gd_2d_rb_y = [res_gd_2d_rb[2][i][2] for i=1:length(res_gd_2d_rb[2])]
	
	# Newton
	res_n_2d_rb = newton(h, x0, (x->0.5), 20)
	res_n_2d_rb_x = [res_n_2d_rb[2][i][1] for i=1:length(res_n_2d_rb[2])]
	res_n_2d_rb_y = [res_n_2d_rb[2][i][2] for i=1:length(res_n_2d_rb[2])]
	
	clf()
	Δ = 0.1
	X=collect(-2:Δ:2)
	Y=collect(-1:Δ:3)
	F=[h([X[j],Y[i]]) for i=1:length(X), j=1:length(Y)]
	contourf(X,Y,F, levels=50)
	PyPlot.title("Rosenbrock: Gradient Descent vs. Newton")
	
	# res_gd_2d_rb
	PyPlot.plot(res_gd_2d_rb_x, res_gd_2d_rb_y, color="yellow")
	scatter(res_gd_2d_rb_x, res_gd_2d_rb_y, color="yellow")
	for i=1:length(res_gd_2d_rb_x)
		annotate(string(i), [res_gd_2d_rb_x[i], res_gd_2d_rb_y[i]], color="w", zorder=2)
	end
	
	# res_n_2d_rb
	PyPlot.plot(res_n_2d_rb_x, res_n_2d_rb_y, color="red")
	scatter(res_n_2d_rb_x, res_n_2d_rb_y, color="red")
	for i=1:length(res_n_2d_rb_x)
		annotate(string(i), [res_n_2d_rb_x[i], res_n_2d_rb_y[i]], color="w", zorder=2)
	end
	
	legend(["Gradient Descent", "Newton"])
	
	xlabel("x")
	ylabel("y")
		
	# Mark minimum
	scatter(a, a^2, color="r", s=500, zorder=3, marker="x")
	
	gcf()
end

# ╔═╡ 903963e2-5b02-11eb-3518-27fbab536dd9
let
	X=collect(-2:Δ:2)
	Y=collect(-1:Δ:3)
	ff = (x->x[1]^2+x[2]^2)
	F=[h([X[j],Y[i]]) for i=1:length(X), j=1:length(Y)]
	clf()
	surf(X, Y, F, cmap=:summer)
	PyPlot.title("Rosenbrock Function")
	gcf()
end

# ╔═╡ Cell order:
# ╟─4bb69ec4-5a76-11eb-2c4f-cd9227558dc0
# ╠═3c142764-5a73-11eb-0540-b7b14a6d5d3f
# ╟─de02a6b8-5a78-11eb-1fde-53c163af27a2
# ╠═b404175c-5a73-11eb-18d9-7ba65f2fec52
# ╟─f2a8118e-5a78-11eb-2acb-550e4172884d
# ╠═cffc768e-5a73-11eb-0717-0f031026eff4
# ╟─40ffbe2c-5a79-11eb-3b1c-177a10981882
# ╠═0dd16d7a-5a74-11eb-3e89-49bfba6dfb5a
# ╠═52ec352c-5a79-11eb-165e-ab8facbea08d
# ╟─5cc7b22c-5a79-11eb-0d88-4f59f5ca0811
# ╠═7ebce1be-5a7a-11eb-2082-6feaa8539ed1
# ╠═84ab696e-5a74-11eb-3516-9963af72f1fb
# ╟─772238e0-5a79-11eb-318b-370716fa392e
# ╠═a496bca8-5a74-11eb-36ab-4d42ff3206e9
# ╠═7faedaae-5a79-11eb-063e-6f051f747ebc
# ╟─948ebe32-5a79-11eb-3193-e17be931cba0
# ╠═ae8f5574-5a74-11eb-378f-d7b2a9c59575
# ╟─a79842b4-5a79-11eb-2b8d-47c8ba6c75dd
# ╠═08ad3e7c-5a75-11eb-208a-a509dbe12e6e
# ╠═adf08aac-5a79-11eb-15c7-8171dd96e851
# ╟─c62a0d8a-5a76-11eb-120f-99169f32365d
# ╠═00231932-5a77-11eb-1e22-9b0237c2490e
# ╠═e3a00fa6-5a79-11eb-1ada-a1695a4e237b
# ╟─b8676c56-5a76-11eb-390f-e9d38cb9c783
# ╟─09133588-5a7a-11eb-11ec-cfbff9d0a488
# ╠═7f47f314-5a76-11eb-040d-cf1d3b80fabf
# ╟─166d376a-5a7a-11eb-10a0-c950dd79b520
# ╠═ae0bb618-5a76-11eb-31d1-23b0130b1bdf
# ╠═89c29786-5a76-11eb-24aa-e373bde1dd3d
# ╠═3f9451ea-5a7b-11eb-15c3-f7be154ca0f1
# ╠═1ad70922-5a7b-11eb-1834-6d781d2e6d0a
# ╟─6ca5ad54-5af9-11eb-3e33-d91c9c54484c
# ╠═8999380a-5af8-11eb-047f-7d4006b15f48
# ╠═903963e2-5b02-11eb-3518-27fbab536dd9
# ╠═9b20248a-5af8-11eb-2078-6b7283933b8d
