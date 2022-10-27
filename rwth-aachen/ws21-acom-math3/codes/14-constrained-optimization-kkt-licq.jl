### A Pluto.jl notebook ###
# v0.17.3

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
- WS21
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

# ╔═╡ 621fc036-660c-11eb-267f-dfacf5c3a5f1
md"""
## Define Objective and Constraints
"""

# ╔═╡ a3eaac27-36eb-4ca3-9879-e3c2dc5fcd85
A = [
	1 5;
	5 1
]

# ╔═╡ fc942da8-0679-47a4-a0ea-09e3dd302663
eigen(A).values

# ╔═╡ a2a1cbd0-6603-11eb-09a8-67dcc6b79a00
f1(x) = (x[1]-1)^2 + (x[2]-2)^2
# f1(x) = x' * A * x

# ╔═╡ 5c2ceeba-660c-11eb-3dc9-f794a5f63e91
f1(x)

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
	return (
		f(x) 
		- sum([g[i](x) * λs[i] for i=1:size(g)[1]]; init=0)
		- sum([h[i](x) * μs[i] for i=1:size(h)[1]]; init=0)
		# - reduce(+, [g[i](x) * λs[i] for i=1:size(g)[1]], init=0)
		# - reduce(+, [h[i](x) * μs[i] for i=1:size(h)[1]], init=0)  # sum
	)
end

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

# ╔═╡ ccc3c252-660c-11eb-0d7a-bb55b87645cb
md"""
## Test KKT Points
"""

# ╔═╡ 43078772-660c-11eb-02d8-b57d829f1e7f
md"""
**Inequality Constraints**: $g_i(x) \ge 0$
"""

# ╔═╡ 6022e976-6575-11eb-1755-8f7e577f30d5
g = [
	x -> x[2] + 1,
	x -> x[2] + x[1] - 0,
]; [gi(x) for gi in g]

# ╔═╡ 07f5fc28-660d-11eb-2df5-5531eaa939a6
lambdas = 
	symbols("lambda:$(length(g))", real=true, nonnegative=true)
	# for i=1:length(g)
# ]

# ╔═╡ 35dffc98-660c-11eb-29b0-c31142c939d5
md"""
**Equality Constraints**: $h_j(x) = 0$
"""

# ╔═╡ ae4c9730-6603-11eb-3721-877881d04cc7
h = [
	x -> (x[1]-1)^2 - 5*x[2],  
	x -> 2-(x[1]-1)^2 - 10*x[2],  
	# x -> x' * x - 1
]; [hi(x) for hi in h]

# ╔═╡ 0d04651a-660d-11eb-23a4-e5c7d7b64d25
mus = [
	symbols("mu$i", real=true)
	for i=1:length(h)
]

# ╔═╡ 526e80d4-6605-11eb-3983-753e63780198
function kktpoints(x, f, g, h, λs, μs, Ig)
	lag = lagrangian(x, f, g, h, λs, μs, Ig)
	eqs = [
		diff(lag, x[1]),
		diff(lag, x[2]),
		[diff(lag, mus[i]) for i=1:length(h)]...,  # <=> h_i(x)==0
		[diff(lag, lambdas[i])*lambdas[i] for i=1:length(g)]...,  # use active g's
		[g[i](x)*lambdas[i] for i=1:length(g)]...,
	]
	sols = solve(eqs, [x...,mus...,lambdas...])
	# filter for "gi > 0" solutions since sympy cannot really solve ineqs...
	return filter(sol->all([gi(sol[1:2])>=0 for gi in g]),sols)
end

# ╔═╡ 244ed6ce-660d-11eb-0e19-a17ccd2bac15
lagrangian(x, f1, g, h, lambdas, mus, [])

# ╔═╡ 77d2c740-6605-11eb-3f3b-d79f8fd7faff
kktpoints(x,f1,g,h,lambdas,mus,[])

# ╔═╡ 1887d96f-1ff9-47f4-a119-072eec387ea1
eigen(A).vectors

# ╔═╡ daa51c5e-660c-11eb-36aa-dba0f0129bb8
md"""
## Visualize KKT Points
"""

# ╔═╡ d5a7791e-6606-11eb-1b71-dde9e7393c05
begin
	
	kktpts = kktpoints(x,f1,g,h,lambdas,mus,[])
	
	# Plot annotations
	clf()
	ax = gca()
	
	Δ = 0.1
	X=collect(-3:Δ:4)
	Y=collect(-4:Δ:4)
	
	F=[f1([X[j],Y[i]]) for i=1:length(Y), j=1:length(X)]
	
	for pt in kktpts
		scatter(pt[1], pt[2], s=250, zorder=3, marker="x")
	end

	# equality constraints h
	for hi=1:length(h)
		CS1 = ax.contour(
			X, Y, [h[hi]([X[i],Y[j]]) for j=1:length(Y), i=1:length(X)],
			[0], colors="white", linestyles="solid"
		)
		PyPlot.plot([], [], color="white", linestyle="solid", zorder=-3)  # to get a legend entry
		ax.clabel(
			CS1, CS1.levels, inline=true, fontsize=10, fmt="h$(hi)=0"
		)
	end

	# inequality constraints g
	for gi=1:length(g)
		contourf(
			X, Y, [g[gi]([X[i],Y[j]]) for j=1:length(Y), i=1:length(X)],
			[0,100], alpha=0.2, colors="white", zorder=2
		)
		CS1 = ax.contour(
			X, Y, [g[gi]([X[i],Y[j]]) for j=1:length(Y), i=1:length(X)],
			[1], colors="white", alpha=0.5, zorder=-3
		)
		ax.clabel(
			CS1, CS1.levels, inline=true, fontsize=10, fmt="g$(gi)>0", zorder=10
		)
		CS2 = ax.contour(
			X, Y, [g[gi]([X[i],Y[j]]) for j=1:length(Y), i=1:length(X)],
			[0], colors="white", linestyles="dotted"
		)
		PyPlot.plot(X, [Float64(solve(g[gi](x),x[2]).subs.(x[1], val)[1][1]) for val in X], color="white", linestyle="dotted", zorder=-3)  # to get a legend entry
		ax.clabel(CS2, CS2.levels, inline=true, fontsize=10, fmt="g$(gi)=0")
		
	end

	contourf(X,Y,F, levels=50)
	legend([["EC h$i" for i=1:length(h)]...,["IC g$i" for i=1:length(g)]...,["KKT Point $i" for i=1:length(kktpts)]...])
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
	set = sympy.Matrix([
		Matrix([diff(g[i](x), x).subs(x[1], ξ[1]).subs(x[2], ξ[2]) for i ∈ Ig(ξ,g)]')...,
		Matrix([diff(h[i](x), x).subs(x[1], ξ[1]).subs(x[2], ξ[2]) for i ∈ 1:size(h)[1]]')...
	])'
	return set
end

# ╔═╡ 6d7ec488-660f-11eb-37c2-6ff8df87ea96
md"""
## Test LICQ in potential KKT Point
"""

# ╔═╡ c0483e74-660a-11eb-12cc-4d7b7492ddcf
set = LICQ(kktpts[1], g, Ig, h)  # check first pt

# ╔═╡ a9b1aca6-660d-11eb-2196-ed4f9f55744e
set.rank()  # full rank <=> linearly independent

# ╔═╡ 3daa0a0b-2d59-4c17-9d8d-812a10c0b5bf
[LICQ([kktpts[i][1], kktpts[i][2]], g, Ig, h).rank() == findmin(size(LICQ([kktpts[i][1], kktpts[i][2]], g, Ig, h)))[1] for i=1:length(kktpts)]

# ╔═╡ b83f3da8-660e-11eb-376f-8d6c7ed189fb
md"""
## See you next week ✌️

- Questions?

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"

[compat]
PlutoUI = "~0.7.32"
PyPlot = "~2.10.0"
SymPy = "~1.1.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "54fc4400de6e5c3e27be6047da2ef6ba355511f8"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

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

[[deps.CommonEq]]
git-tree-sha1 = "d1beba82ceee6dc0fce8cb6b80bf600bbde66381"
uuid = "3709ef60-1bee-4518-9f2f-acd86f176c50"
version = "0.2.0"

[[deps.CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "6cdc8832ba11c7695f494c9d9a1c31e90959ce0f"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.6.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

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

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "22df5b96feef82434b07327e2d3c770a9b21e023"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

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

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "92f91ba9e5941fc781fecf5494ac1da87bdac775"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "ae6145ca68947569058866e443df69587acc1806"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.32"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "2cf929d64681236a2e074ffafb8d568733d2e6af"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.3"

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
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e08890d19787ec25029113e88c34ec20cac1c91e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.0.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.SymPy]]
deps = ["CommonEq", "CommonSolve", "Latexify", "LinearAlgebra", "Markdown", "PyCall", "RecipesBase", "SpecialFunctions"]
git-tree-sha1 = "571bf3b61bcd270c33e22e2e459e9049866a2d1f"
uuid = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
version = "1.1.3"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─17c7a6d8-660d-11eb-3312-8712778733fb
# ╟─31e52238-6578-11eb-181a-339a1afd7e0d
# ╟─921a1b84-660b-11eb-0371-73624873f65f
# ╠═98bb2f16-6573-11eb-396a-b9b1244088ba
# ╟─83b62740-660b-11eb-3974-4fe196977c37
# ╠═2d6c879a-6605-11eb-1c47-457472258b66
# ╟─621fc036-660c-11eb-267f-dfacf5c3a5f1
# ╟─a3eaac27-36eb-4ca3-9879-e3c2dc5fcd85
# ╠═fc942da8-0679-47a4-a0ea-09e3dd302663
# ╠═a2a1cbd0-6603-11eb-09a8-67dcc6b79a00
# ╠═5c2ceeba-660c-11eb-3dc9-f794a5f63e91
# ╠═07f5fc28-660d-11eb-2df5-5531eaa939a6
# ╟─0d04651a-660d-11eb-23a4-e5c7d7b64d25
# ╟─0b45ae42-660c-11eb-07ba-531fe5ed2ed3
# ╠═27a48782-6587-11eb-35bc-45bb830139f1
# ╠═244ed6ce-660d-11eb-0e19-a17ccd2bac15
# ╟─a8041c28-660c-11eb-28e6-4dfb1e6c1e15
# ╠═526e80d4-6605-11eb-3983-753e63780198
# ╟─ccc3c252-660c-11eb-0d7a-bb55b87645cb
# ╟─43078772-660c-11eb-02d8-b57d829f1e7f
# ╠═6022e976-6575-11eb-1755-8f7e577f30d5
# ╟─35dffc98-660c-11eb-29b0-c31142c939d5
# ╠═ae4c9730-6603-11eb-3721-877881d04cc7
# ╠═77d2c740-6605-11eb-3f3b-d79f8fd7faff
# ╠═1887d96f-1ff9-47f4-a119-072eec387ea1
# ╟─daa51c5e-660c-11eb-36aa-dba0f0129bb8
# ╟─d5a7791e-6606-11eb-1b71-dde9e7393c05
# ╟─bfede808-6609-11eb-3d74-e73170fa62a3
# ╟─e69b65a4-660c-11eb-1e9e-eb220ba32d68
# ╠═b84179a6-6575-11eb-23d3-53d1f4121a78
# ╠═cb8fdbe2-6575-11eb-356a-173604653340
# ╟─6d7ec488-660f-11eb-37c2-6ff8df87ea96
# ╠═c0483e74-660a-11eb-12cc-4d7b7492ddcf
# ╠═a9b1aca6-660d-11eb-2196-ed4f9f55744e
# ╠═3daa0a0b-2d59-4c17-9d8d-812a10c0b5bf
# ╟─b83f3da8-660e-11eb-376f-8d6c7ed189fb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
