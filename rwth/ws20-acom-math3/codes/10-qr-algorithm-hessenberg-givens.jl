### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ e8b796fa-54d5-11eb-1e71-ffd347c0d39a
begin
	using LinearAlgebra, PlutoUI, Random, SparseArrays, Plots
	plotly()
end

# ╔═╡ 364c54f6-550e-11eb-0725-aff4337a78cf
md"""
# QR Algorithm for Eigenvalue Problems with Hessenberg/Givens Tricks

- Mathe 3 (CES)
- WS20
- Lambert Theisen (```theisen@acom.rwth-aachen.de```)
"""

# ╔═╡ 4acfb882-550e-11eb-1e9c-0d28cf60071f
md"""
## QR Algorithm Native

We use Julia's standard `qr()` function and implement:

1. Given $A \in \mathbb{R}^{n \times n}$
1. Initialize $Q^{(m+1)} = I$
1. For $k=1,\dots,m$:
    1. Calculate QR-Decomposition: $A_k = Q_k R_k$
    1. Update: $A_{k+1} = R_k Q_k$
1. Return diagonal entries of $A_m$ and $Q^{(m)} = Q_m \cdots Q_0 Q_1$
"""

# ╔═╡ 988bfb48-54e7-11eb-39db-affae631adb1
function qra_general(A, m)
	@assert size(A)[1] == size(A)[2] && length(size(A))==2
	n = size(A)[1]
	Qm = I(n)
	for k=1:m
		Q, R = qr(A)
		A = R * Q
		Qm = Qm * Q
	end
	return diag(A), Qm
end

# ╔═╡ 5e14962c-5510-11eb-0068-41e58fe2ba7e
md"""
## Check Validity of Implementation
"""

# ╔═╡ 6de63e92-557f-11eb-0ef9-0518d8f3ba6b
A = 1. * [
	1 2 3
	4 5 6 
	7 8 9
] + 2I(3)

# ╔═╡ c3b2a5d2-5510-11eb-2eca-d7aa43034a89
qra_general(A, 100)

# ╔═╡ 69ac7432-5510-11eb-1977-a578de19784e
eigen(A).values - sort(qra_general(A, 100)[1], rev=false)

# ╔═╡ 6d8daab0-5580-11eb-1198-e18638114f71
md"""
## Check Convergence
"""

# ╔═╡ 737b2c4a-5580-11eb-0316-255c9fc42d8d
begin
	N = 100
	tape = Array[]
	for k=1:N
		push!(tape, sort(qra_general(A, k)[1], rev=false)) 
		# *dont do this, very inefficient*
	end
end

# ╔═╡ dfd96c6c-5580-11eb-1654-7b9a9e50c562
errors = [
	abs.(map(x -> x[1], tape) .- eigen(A).values[1]),
	abs.(map(x -> x[2], tape) .- eigen(A).values[2]),
	abs.(map(x -> x[3], tape) .- eigen(A).values[3]),
]

# ╔═╡ 945baba0-5581-11eb-3b11-3d7135aed14c
plot([errors[1],errors[2],errors[3]], yaxis=:log, label=["lambda_1" "lambda_2" "lambda_3"], title="QR Algorithm Errors", xlabel="itertation", ylabel="abs error")

# ╔═╡ d9483bf8-5510-11eb-1976-e9f2a4397ad8
md"""
## Improve QR Algorithm with Upper Hessenberg Matrix Preconditioning


- Idea: QR decomposition needs $\mathcal{O}(n^3)$, QR for Hessenberg matrices is easier done in $\mathcal{O}(n^2)$. Linear complexity is even possible if $A$ is symmetric. In this case, the QR decomposition only needs $\mathcal{O}(n)$ Givens rotations with constant effort.
- Therefore transform the matix $A$ to upper Hessenberg form with similarity transforms in $\mathcal{O}(n^3)$ (also cubic, but only needs to be done once) and use this matrix for the QR algorithm.

### Upper Hessenberg Shape

```math
H={\begin{pmatrix}h_{{11}}&h_{{12}}&h_{{13}}&\cdots &h_{{1n}}\\h_{{21}}&h_{{22}}&h_{{23}}&\cdots &h_{{2n}}\\0&h_{{32}}&h_{{33}}&\cdots &h_{{3n}}\\\vdots &\ddots &\ddots &\ddots &\vdots \\0&\cdots &0&h_{{nn-1}}&h_{{nn}}\end{pmatrix}}
```

Algorithm [1]:

1. Given: $A \in \mathbb{R}^{n \times n}$
1. For $k=1 \dots n-2$ do:
    1. $[v, \beta] \leftarrow \text{house}(A(k+1:n, k))$
    1. $A(k+1:n, k:n) \leftarrow (I - \beta v v^T) A(k+1:n, k:n)$
    1. $A(1:n, k+1:n) \leftarrow A(1:n, k+1:n) (I - \beta v v^T)$

with Householder reflection vector $v$ and weight $\beta = 2 / (v^T v)$.

[1]: https://www.tu-chemnitz.de/mathematik/numa/lehre/nla-2015/Folien/nla-kapitel6.pdf

"""

# ╔═╡ b3cceda2-5511-11eb-0dbf-b97617c34c62
function householdervec(x)
	@assert size(x)[1]>0 && length(size(x))==1
	n = size(x)[1]
	e1 = I(n)[:,1]
	v = x + norm(x, 2) * e1
	β = 2 / (v' * v)
	return v, β
end

# ╔═╡ ae7a3f94-5511-11eb-0411-f1a99384aa00
function upperhessenberg(A)
	@assert size(A)[1] == size(A)[2] && length(size(A))==2
	n = size(A)[1]
	for k = 1:n-2
		v, β = householdervec(A[k+1:n,k])
		A[k+1:n, k:n] = (I(n-k) - β * v * v') * A[k+1:n, k:n]
		A[1:n, k+1:n] = A[1:n, k+1:n] * (I(n-k) - β * v * v')
	end
	return A
end

# ╔═╡ 6bcf9854-5513-11eb-39e9-e7fc80d7427a
md"""
### Check if Householder ad Upperhessenberg Transformations work
"""

# ╔═╡ 79c70514-5513-11eb-3aad-db76acce1878
BB = 1. * [
	1 2 3
	4 5 6 
	7 8 9
]

# ╔═╡ a1c2a12c-5513-11eb-1069-1fb48f7baa9f
begin
	CC = rand(4,4)
	v, β = householdervec(CC[:,1])
	CC = (I(4) - 2/(v' * v) * v * v') * CC
	# CC should now have zeros in first column except diagonal
	# To get all other columns to zero, repeat with sub blockmatrices
end

# ╔═╡ 831cb3f2-5513-11eb-0e32-cb8882c28a68
upperhessenberg(BB) # Should have only one lower sub diagonal

# ╔═╡ 3092870a-5514-11eb-3446-2deb5169f82c
md"""
## QR Algorithm for symmetric Hessenberg matrices

Symmertric hessenberg matrices are tridiagonal (only diag plus uppe and lower sub-diagonal). For the QR decomposition, we only have to make the lower sub diagonal entries to zero to obtain the upper right triangular matrix. This can be done by using Givens roations:

### Givens Rotations [1]

Given a matrix $A$, we can make to entry $A_{ij}$ to zero with $A_{\text{new}} = G(A, i, j) A$ where

```math
{\displaystyle G(A, i, j) ={\begin{bmatrix}1&\cdots &0&\cdots &0&\cdots &0\\\vdots &\ddots &\vdots &&\vdots &&\vdots \\0&\cdots &c&\cdots &-s&\cdots &0\\\vdots &&\vdots &\ddots &\vdots &&\vdots \\0&\cdots &s&\cdots &c&\cdots &0\\\vdots &&\vdots &&\vdots &\ddots &\vdots \\0&\cdots &0&\cdots &0&\cdots &1\end{bmatrix}},}
```

with 
- $r = \sqrt{A_{jj}^2 + A_{ij}^2}$
- $s = - A_{ij} / r$
- $c = A_{jj} / r$

[1]: https://en.wikipedia.org/wiki/Givens_rotation

"""

# ╔═╡ 555e6fac-5516-11eb-1de4-2b06aa09fbbc
function givens_rotation_matrix(A, i, j)
	n = size(A)[1]
	r = sqrt(A[j,j]^2 + A[i,j]^2)
	s = -A[i,j]/r
	c = A[j,j]/r
	G = sparse(1.0I, n, n)
	G[i,i] = c
	G[j,j] = c
	G[i,j] = s
	G[j,i] = -s
	return G
end

# ╔═╡ b0e098be-5516-11eb-3a9e-3153f4f4fc81
md"""
## Check Givens Rotation
"""

# ╔═╡ bad026e8-5516-11eb-230e-85b6a52c0c40
begin
	AA =	1. * [
		1 2 3
		4 5 6 
		7 8 9
	]
	AA = givens_rotation_matrix(AA, 2, 1) * AA
end

# ╔═╡ 43a62c90-5517-11eb-071b-73ae839fc436
md"""
## Implement QR Decomposition for Symmetric Hessenberg Matrices

- Just iterate over sub-diagonal entries and make them zero to get $R$. Store all Givens rotations to get $Q$.
"""

# ╔═╡ 51894b8a-5517-11eb-37c9-874bd7624ec3
function qr_symm_hess(A)
	# QR decomposition for symmetric Hessenberg matrix <=> tridiagonal matrix
	n = size(A)[1]
	abstol = 1E-14
	isapproxsymmetric = any(isapprox.(-0.5 * (A - A'), zeros(n, n), atol=abstol))
	isapproxtridiagonal = any(isapprox.(A, Tridiagonal(A), atol=abstol))
	@assert isapproxsymmetric && isapproxtridiagonal
	
	Am = A
	Qm = sparse(1.0I, n, n)
	for m=1:n-1
		G = givens_rotation_matrix(Am, m+1,m)
		Am = G * Am
		Qm = Qm * G'
	end
	Q = Qm
	R = Qm' * A
	return Array(Q), R
end

# ╔═╡ cb098a5a-5518-11eb-3149-3186e8a69295
md"""
## Check QR Decomposition for Tridiagonal Matrices
"""

# ╔═╡ d4bd8f2e-5518-11eb-3790-0bce5bfbf37e
EE = [6 5 0; 5 1 4; 0 4 3]

# ╔═╡ e1b5ed2a-5518-11eb-2b73-dd0e6ed30e47
qr(EE).Q * qr(EE).R

# ╔═╡ e30bbd58-5518-11eb-1724-95df526072b1
qr_symm_hess(EE)[1] * qr_symm_hess(EE)[2]

# ╔═╡ 6ab5a6c6-5517-11eb-2725-2318790eb408
md"""
## QR Algorithm for Tridiagonal Matrices (Symmetric Upper Hessenberg)

- Same as native implementation, but first transform to Hessenberg and then use the cheap Givens rotation style to get all the QR decompositions.
"""

# ╔═╡ 553d7aec-54e9-11eb-1373-63dd8650c710
function qra_symm(A, m)
	n = size(A)[1]
	@assert size(A)[1] == size(A)[2] && length(size(A))==2
	isapproxsymmetric = any(isapprox.(-0.5 * (A - A'), zeros(n, n), atol=1E-8))
	@assert isapproxsymmetric
	Qm = I(n)
	A = upperhessenberg(A)
	for k=1:m
		Q, R = qr_symm_hess(A)
		A = R * Q
		Qm = Qm * Q
	end
	return diag(A), Array(Qm)
end

# ╔═╡ d4b20e56-54f0-11eb-34fe-f93dc6a33ec9
md"""
## Check Validity of our QR Algorithm for Tridiagonal Matrice
"""

# ╔═╡ d270bc9c-5517-11eb-3892-9104335a8205
DD = [6 5 0; 5 1 4; 0 4 3]

# ╔═╡ bbd39212-5519-11eb-13a2-6d5dfb79d025
qra_symm(DD, 1000)

# ╔═╡ 209fd328-5518-11eb-1915-c35381b35e92
qra_general(DD, 1000)

# ╔═╡ fc2396f0-54f3-11eb-1a4b-d7503ff393b4
md"""
## Speed Check

We still loose against Julia's native `qr`-method. 😐

- Homework: Improve this.
"""

# ╔═╡ 02636bee-54f4-11eb-29b0-fb4aec4b0ba0
with_terminal() do
	N = 100
	M = 50
	C = Array(Symmetric(rand(N,N)))
	
	# @time qr(C)
	# @time qr_symm_hess(C)
	# @time upperhessenberg(C)
	
	println("QRA General:")
	@time Am1, Qm1 = qra_general(C, M)
	println("QRA Own:")
	@time Am2, Qm2 = qra_symm(C, M)
end 

# ╔═╡ Cell order:
# ╟─364c54f6-550e-11eb-0725-aff4337a78cf
# ╠═e8b796fa-54d5-11eb-1e71-ffd347c0d39a
# ╟─4acfb882-550e-11eb-1e9c-0d28cf60071f
# ╠═988bfb48-54e7-11eb-39db-affae631adb1
# ╟─5e14962c-5510-11eb-0068-41e58fe2ba7e
# ╠═6de63e92-557f-11eb-0ef9-0518d8f3ba6b
# ╠═c3b2a5d2-5510-11eb-2eca-d7aa43034a89
# ╠═69ac7432-5510-11eb-1977-a578de19784e
# ╟─6d8daab0-5580-11eb-1198-e18638114f71
# ╠═737b2c4a-5580-11eb-0316-255c9fc42d8d
# ╠═dfd96c6c-5580-11eb-1654-7b9a9e50c562
# ╠═945baba0-5581-11eb-3b11-3d7135aed14c
# ╟─d9483bf8-5510-11eb-1976-e9f2a4397ad8
# ╠═ae7a3f94-5511-11eb-0411-f1a99384aa00
# ╠═b3cceda2-5511-11eb-0dbf-b97617c34c62
# ╠═6bcf9854-5513-11eb-39e9-e7fc80d7427a
# ╠═79c70514-5513-11eb-3aad-db76acce1878
# ╠═a1c2a12c-5513-11eb-1069-1fb48f7baa9f
# ╠═831cb3f2-5513-11eb-0e32-cb8882c28a68
# ╟─3092870a-5514-11eb-3446-2deb5169f82c
# ╠═555e6fac-5516-11eb-1de4-2b06aa09fbbc
# ╟─b0e098be-5516-11eb-3a9e-3153f4f4fc81
# ╠═bad026e8-5516-11eb-230e-85b6a52c0c40
# ╟─43a62c90-5517-11eb-071b-73ae839fc436
# ╠═51894b8a-5517-11eb-37c9-874bd7624ec3
# ╠═cb098a5a-5518-11eb-3149-3186e8a69295
# ╠═d4bd8f2e-5518-11eb-3790-0bce5bfbf37e
# ╠═e1b5ed2a-5518-11eb-2b73-dd0e6ed30e47
# ╠═e30bbd58-5518-11eb-1724-95df526072b1
# ╟─6ab5a6c6-5517-11eb-2725-2318790eb408
# ╠═553d7aec-54e9-11eb-1373-63dd8650c710
# ╟─d4b20e56-54f0-11eb-34fe-f93dc6a33ec9
# ╠═d270bc9c-5517-11eb-3892-9104335a8205
# ╠═bbd39212-5519-11eb-13a2-6d5dfb79d025
# ╠═209fd328-5518-11eb-1915-c35381b35e92
# ╟─fc2396f0-54f3-11eb-1a4b-d7503ff393b4
# ╠═02636bee-54f4-11eb-29b0-fb4aec4b0ba0
