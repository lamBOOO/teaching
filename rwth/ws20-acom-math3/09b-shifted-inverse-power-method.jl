### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 35c92372-4f6e-11eb-3fce-b1a2317de576
using LinearAlgebra, PlutoUI, Plots, Printf

# ╔═╡ a2c1cb88-500a-11eb-0d82-2f89e4f653a0
md"""
# (Shifted) (Inverse) Power Method

- Mathe 3 (CES)
- WS20
- Lambert Theisen (```theisen@acom.rwth-aachen.de```)
"""

# ╔═╡ a95a3b88-4f74-11eb-3a04-bd00b03800cd
plotly()

# ╔═╡ b1c8c730-500a-11eb-38f9-15affdadecc1
md"""
## Define some Matrices
"""

# ╔═╡ 4221310c-4f6e-11eb-1c3d-ed1601061f66
A = [
	25 -89 68
    -26 148 -52
    -10 77 -29
]

# ╔═╡ a68a0278-4f72-11eb-12d2-b1c177d65a42
B = [
	-139 -85 -125
	182 -64 -178
	-117 -105 79
]

# ╔═╡ c25fbfa4-500a-11eb-3767-f7eca51e4ca0
md"""
## Define the Rayleigh Quotient
```math
\rho_A(x) := \frac{x^T A x}{x^T x}
```
"""

# ╔═╡ d5fc3048-4f6e-11eb-1e76-f9da889f18dc
function ρ(A, x) # Rayleigh quotient
	return x' * A * x / (x' * x)
end

# ╔═╡ f6b280e8-500a-11eb-17d2-0ffd4ec3b348
md"""
## Construct Error History Object

This is used to store all the errors for later plotting.
"""

# ╔═╡ 1c5f7d86-4f71-11eb-15ad-f184885ff599
struct Errorhistory
	errors :: Array{Float64}
end

# ╔═╡ 0cd70740-500b-11eb-0910-7f1ea781bfc4
md"""
## Define the Power Method Algorithm

1. Given $A \in \mathbb{R}^n$
1. Choose start vector $x_0 \in \mathbb{R} \setminus \{0\}$
1. While $k < k_\max \wedge \text{error} > \text{tol}$ do
    1. $x_{k+1} = \frac{A x_k}{{||A x_k||}_2}$
    1. $\lambda_{k+1} = \rho_A(x_{k+1})$
1. Return estimated eigenpair $(\lambda_k,x_k)$
"""

# ╔═╡ 41ef2bee-4f6e-11eb-3d48-a71b0ed5dd36
function PM(A, x0; maxit = 100, tol = 1E-10) # Power Method
	x = x0
	k = 0
	residual = Inf
	λ = nothing
	eh = Errorhistory([])
	while k <= maxit && norm(residual) > tol
		x = A * x
		x = x / norm(x)
		λ = ρ(A, x)
		residual = A * x - λ * x # if λ exact => residual = zeros(n)
		push!(eh.errors, norm(residual))
		k += 1
	end
	return (λ, x, eh)
end

# ╔═╡ 113bd184-500c-11eb-1d6e-21653b218da3
md"""
## Execute PM

Will converge to largest eigenpair.
"""

# ╔═╡ 35969908-4f6f-11eb-31e2-9f825ba00791
pm = PM(A, ones(3))

# ╔═╡ 223edd32-500c-11eb-30a4-3d9a3a853463
md"""
## Residual Error

The normed error of the residual $e = {||x_k - x^*||}_2$ is in $\mathcal{O}(q^k)$ with the eigenvalue ratio $q = \lambda_1 / \lambda_2$ of the considered matrix.
"""

# ╔═╡ bd3f88f6-4f71-11eb-0fa2-05c5abfff416
plot([pm[3].errors, [1 * abs(eigvals(A)[end-1] / eigvals(A)[end])^i for i = 1 : size(pm[3].errors)[1]]], yaxis=:log, title="PM: Norm of Residual", label=["norm(res, 2)" "O(q^i)=O($(@sprintf("%.2f", abs(eigvals(A)[end-1] / eigvals(A)[end])))^i)"])

# ╔═╡ 8cecedf4-500c-11eb-2a30-f75eaf612d00
md"""
## Comparison of Matrices with Different Fundamental Ratios

- Matrix $A$ has ratio $q=$ $(abs(eigvals(A)[end-1]/eigvals(A)[end]))
- Matrix $B$ has ratio $q=$ $(abs(eigvals(B)[end-1]/eigvals(B)[end]))

Therefore, much faster converge for $A$.

Oscillations probably due to $|\lambda_3| = |\lambda_2|$ 🤔.
"""

# ╔═╡ b7b14674-4f72-11eb-3686-c773f4eb30f7
plot(map(X -> PM(X, ones(3), maxit=500)[3].errors, [A, B]), yaxis=:log, label=["A" "B"], title="Residual errors A vs B")

# ╔═╡ 5a6643f2-500d-11eb-257e-391533afac57
md"""
## Define the (Shifted) Inverse Power Method Algorithm

1. Given $A \in \mathbb{R}^n$ and eigenvalue shift $\mu$
1. Choose start vector $x_0 \in \mathbb{R} \setminus \{0\}$
1. While $k < k_\max \wedge \text{error} > \text{tol}$ do
    1. Solve: $(A - \mu I)x_{k+1} = x_k$ (this is like $x_{k+1} = {(A - \mu I)}^{-1} x_k$)
    1. Normalize: $x_{k+1} \mapsto x_{k+1} / {||x_{k+1}||}_2$
    1. Update eigenvalue estimate (with initial matrix $A$, not $A^{-1}$): $\lambda_{k+1} = \rho_A(x_{k+1})$
1. Return estimated eigenpair $(\lambda_k,x_k)$
"""

# ╔═╡ 0fdc3b7c-4f75-11eb-2beb-af291c8e958a
function IPM(A, x0; shift = 0, maxit = 100, tol = 1E-10) # Inverse Power Method
	x = x0
	i = 0
	residual = Inf
	λ = nothing
	eh = Errorhistory([])
	while i <= maxit && norm(residual) > tol
		x = (A - shift * I(size(A)[2])) \ x
		x = x / norm(x)
		λ = ρ(A, x)
		residual = A * x - λ * x
		push!(eh.errors, norm(residual))
		i += 1
	end
	return (λ, x, eh)
end

# ╔═╡ e86fffee-500d-11eb-3ac3-b5566ad0ae0d
md"""
## Execute IPM to find the Smallest Eigenpair
"""

# ╔═╡ b677f062-4f74-11eb-183a-43f3ca78885c
ipm = IPM(A, ones(3))

# ╔═╡ f6348782-500d-11eb-1d08-d1ee9f734ce0
md"""
## Check the Error
"""

# ╔═╡ ea5b4532-4f74-11eb-28b8-d91d3fe98c63
abs(ipm[1] - eigvals(A)[1])

# ╔═╡ fec4b97e-500d-11eb-1cbc-21a0804aa684
md"""
## Check the Convergence Behavior for Different Shifts

Notice that better shifts significantly improve the performance of the algorithm. Shift eight and ten are the same because they have the same absolute distance to the real eigenvalue.
"""

# ╔═╡ 3d0bc656-4f75-11eb-3965-756953306b63
# Compare zeros shift with good estimation
shifts = 0:-2:-10 # real lowest eval is -9

# ╔═╡ 4a0e3e1c-4f75-11eb-0126-3b421029314a
plot(map(X -> IPM(A, ones(3), tol=1E-5, shift=X)[3].errors, shifts), yaxis=:log, label=reshape(map(x -> string("shift=", x), shifts), 1, :), title="Residual Errors A with diferent shifts")

# ╔═╡ Cell order:
# ╟─a2c1cb88-500a-11eb-0d82-2f89e4f653a0
# ╠═35c92372-4f6e-11eb-3fce-b1a2317de576
# ╠═a95a3b88-4f74-11eb-3a04-bd00b03800cd
# ╟─b1c8c730-500a-11eb-38f9-15affdadecc1
# ╠═4221310c-4f6e-11eb-1c3d-ed1601061f66
# ╠═a68a0278-4f72-11eb-12d2-b1c177d65a42
# ╟─c25fbfa4-500a-11eb-3767-f7eca51e4ca0
# ╠═d5fc3048-4f6e-11eb-1e76-f9da889f18dc
# ╟─f6b280e8-500a-11eb-17d2-0ffd4ec3b348
# ╠═1c5f7d86-4f71-11eb-15ad-f184885ff599
# ╟─0cd70740-500b-11eb-0910-7f1ea781bfc4
# ╠═41ef2bee-4f6e-11eb-3d48-a71b0ed5dd36
# ╟─113bd184-500c-11eb-1d6e-21653b218da3
# ╠═35969908-4f6f-11eb-31e2-9f825ba00791
# ╟─223edd32-500c-11eb-30a4-3d9a3a853463
# ╠═bd3f88f6-4f71-11eb-0fa2-05c5abfff416
# ╟─8cecedf4-500c-11eb-2a30-f75eaf612d00
# ╠═b7b14674-4f72-11eb-3686-c773f4eb30f7
# ╟─5a6643f2-500d-11eb-257e-391533afac57
# ╠═0fdc3b7c-4f75-11eb-2beb-af291c8e958a
# ╟─e86fffee-500d-11eb-3ac3-b5566ad0ae0d
# ╠═b677f062-4f74-11eb-183a-43f3ca78885c
# ╟─f6348782-500d-11eb-1d08-d1ee9f734ce0
# ╠═ea5b4532-4f74-11eb-28b8-d91d3fe98c63
# ╟─fec4b97e-500d-11eb-1cbc-21a0804aa684
# ╠═3d0bc656-4f75-11eb-3965-756953306b63
# ╠═4a0e3e1c-4f75-11eb-0126-3b421029314a
