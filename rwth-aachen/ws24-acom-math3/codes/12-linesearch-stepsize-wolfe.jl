### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 92b91de6-5fef-11eb-1b89-951c38260bea
begin
	ENV["MPLBACKEND"]="Agg"
	using PyPlot
	using Calculus, PlutoUI
	using LinearAlgebra
end

# ╔═╡ 66c10dde-608a-11eb-024e-d7e7ca5c9f53
html"<button onclick='present()'>present</button>"

# ╔═╡ b54c4e40-5fee-11eb-33fd-8da1baf6540d
md"""
# Line Search Stepsize Control and Trust-Region Methods

- Mathe 3 (CES)
- WS24/25
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
	@show α
	return α
end

# ╔═╡ ef6cba94-608c-11eb-06ef-b5af1c5c9662
md"""
## Wolfe Stepsize Conditon

- We need to specify a conditon for the backtracking algorithm
- Use Wolfe conditions

```math
{\displaystyle {\begin{aligned}{\textbf {i)}}&\quad f(\mathbf {x} _{k}+\alpha _{k}\mathbf {p} _{k})\leq f(\mathbf {x} _{k})+c_{1}\alpha _{k}\mathbf {p} _{k}^{\mathrm {T} }\nabla f(\mathbf {x} _{k}),\\[6pt]{\textbf {ii)}}&\quad {-\mathbf {p} }_{k}^{\mathrm {T} }\nabla f(\mathbf {x} _{k}+\alpha _{k}\mathbf {p} _{k})\leq -c_{2}\mathbf {p} _{k}^{\mathrm {T} }\nabla f(\mathbf {x} _{k}),\end{aligned}}}
```
"""

# ╔═╡ e21dfe42-5fef-11eb-22a6-e5db4e09613c
armijo(f, d, x, α) = f(x + α*d) <= f(x) + 1E-4 * α * derivative(f, x)' * d

# ╔═╡ fb78b6ea-b798-4f76-90f9-c95cd55b80db
curvature(f, d, x, α) = derivative(f, x + α*d)' * d >= 0.9 * derivative(f, x)' * d

# ╔═╡ bfcac302-5fef-11eb-19ef-bdde45ad188f
function backtracking_linesearch_wolfe(f, x, d, αmax, β)  #TODO
	return backtracking_linesearch(f, x, d, αmax, (f, d, x, α)->(armijo(f, d, x, α)&&curvature(f, d, x, α)), β)
end

# ╔═╡ 29ca1ff8-608d-11eb-2b01-b954fcd2de76
md"""
## Use Backtracking Algorithm in Gradient Descent

- Same as last week, but with adaptive step size
"""

# ╔═╡ 5c83c5e6-5fef-11eb-1a0e-3d9e19a8874b
function gradient_descent_wolfe(f, x0, kmax)
	x = x0
	hist = []
	push!(hist, x)
	for k=1:kmax
		x = x + backtracking_linesearch_wolfe(
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
  - Even divergence for most stepsizes
- Now: Line search automatically choose a valid step size and we have an easy life
"""

# ╔═╡ 82a2e0e0-5fef-11eb-2dfb-b7644e600e48
begin
	# Rosenbrock function with x* = [a,a^2], f(x*)=0
	a = 1
	b = 100
	h = (x -> (a-x[1])^2 + b*(x[2]-x[1]^2)^2)
	
	x0 = [-1.,0.5]
	
	# Gradient Descent with Armijo1
	res_gd_2d_rb_arm1 = gradient_descent_wolfe(h, x0, 1000)
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
	# for i=1:length(res_gd_2d_rb_arm1_x)
	# 	annotate(string(i), [res_gd_2d_rb_arm1_x[i], res_gd_2d_rb_arm1_y[i]], color="w", zorder=2)
	# end
	
	legend(["Gradient Descent with Armoji"])

	PyPlot.text(2-0.2, -1+0.2, "final error: $(norm(res_gd_2d_rb_arm1[2][end]-[1,1]))", size=16,
         ha="right", va="bottom",
		bbox=Dict("boxstyle"=>"square")
	)
	
	xlabel("x")
	ylabel("y")
		
	# Mark minimum
	scatter(a, a^2, color="r", s=500, zorder=3, marker="x")
	
	gcf()
end

# ╔═╡ 2b218df6-608c-11eb-01f7-0bdb99d010e6
md"""
## Works but still not the best convergence...
"""

# ╔═╡ a372c9f0-5ff1-11eb-05a4-23ac4164dbf5
res_gd_2d_rb_arm1 # still not very fast (x*=[1,1]) 😓 (but robust!)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Calculus = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"

[compat]
Calculus = "~0.5.1"
PlutoUI = "~0.7.30"
PyPlot = "~2.10.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.7"
manifest_format = "2.0"
project_hash = "52d815cbe948e5d2968fd7675a0afe5cfaa85935"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "6cdc8832ba11c7695f494c9d9a1c31e90959ce0f"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.6.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "92f91ba9e5941fc781fecf5494ac1da87bdac775"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "5c0eb9099596090bb3215260ceca687b888a1575"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.30"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "71fd4022ecd0c6d20180e23ff1b3e05a143959c2"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.93.0"

[[deps.PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "14c1b795b9d764e1784713941e787e1384268103"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.10.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VersionParsing]]
git-tree-sha1 = "e575cf85535c7c3292b4d89d89cc29e8c3098e47"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─66c10dde-608a-11eb-024e-d7e7ca5c9f53
# ╟─b54c4e40-5fee-11eb-33fd-8da1baf6540d
# ╟─92b91de6-5fef-11eb-1b89-951c38260bea
# ╟─ddb561e0-608c-11eb-0920-074e5a84724e
# ╠═6c22d348-5f13-11eb-1a98-eb6313fcf858
# ╟─ef6cba94-608c-11eb-06ef-b5af1c5c9662
# ╠═e21dfe42-5fef-11eb-22a6-e5db4e09613c
# ╠═fb78b6ea-b798-4f76-90f9-c95cd55b80db
# ╠═bfcac302-5fef-11eb-19ef-bdde45ad188f
# ╟─29ca1ff8-608d-11eb-2b01-b954fcd2de76
# ╠═5c83c5e6-5fef-11eb-1a0e-3d9e19a8874b
# ╟─434c3160-608c-11eb-2160-d3405ce05327
# ╠═82a2e0e0-5fef-11eb-2dfb-b7644e600e48
# ╟─2b218df6-608c-11eb-01f7-0bdb99d010e6
# ╠═a372c9f0-5ff1-11eb-05a4-23ac4164dbf5
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
