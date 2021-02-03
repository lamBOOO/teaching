### A Pluto.jl notebook ###
# v0.12.17

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

# ╔═╡ 7eeeb98c-3f20-11eb-220d-41efe1d23b03
using Images, TestImages, LinearAlgebra, PlutoUI, Plots

# ╔═╡ 39f3be04-3f20-11eb-3654-57e887652680
md"""
# Image Compression Using SVD

We consider an grayscale image $ A \in \mathbb{R}^{n \times n} $ with entries $A_{ij} \in [0,1]$ representing the gray intensity.

## Singular Value Decomposition

A square matrix $A \in \mathbb{R}^{n \times n}$ can be written as SVD, defined as:
```math
A = U \Sigma V^T = \sum_{i=1}^{n} u_i \sigma_i v_i^T = u_1 \sigma_1 v_1^T + \dots + u_r \sigma_r v_r^T
```
"""

# ╔═╡ 3169b458-3f20-11eb-2dba-571a9e073819
md"""
Load test image from the `TestImages` package.
"""

# ╔═╡ 25579486-3f00-11eb-1405-c3c724fcaf95
begin
	# img = float.(testimage("lena_gray_512"))
	img = float.(testimage("moonsurface"))
end

# ╔═╡ cca739f8-3f00-11eb-2c80-dff8b51dbe98
function compressed(img, rank)
	U, Σ, Vt = svd(img);
	return Gray.(sum([U[:,i] * Σ[i] * Vt[:,i]' for i=1:rank])) # Gray
end

# ╔═╡ 00770746-3f23-11eb-1f0d-af0b7ad6086d
@bind rank Slider(1:size(img)[1], default=Int(round(size(img)[1]//20)), show_value=true)

# ╔═╡ be3886e8-3f1d-11eb-20e2-2b3003abdc3b
md"""
## Compressed Image

We construct the compressed image `` \tilde{A} \in \mathbb{R}^{n \times n} `` as rank $ r=$(rank) $ approimxation, defined as:
```math
\tilde{A} = \sum_{i=1}^{r} u_i \sigma_i v_i^T = u_1 \sigma_1 v_1^T + \dots + u_r \sigma_r v_r^T
```
with rank r.

## Storage Requirement of Compressed Matrix

Instead of storing $n^2$ matrix entires, we could now only store the $r$-times the summation tuple $\{u_i,\sigma_i,v_i^T\}$ which leads to a size

```math
\text{size}(\tilde{A}) = r(n+1+n) = r(2n+1)\ll n^2 = \text{size}(A) \text{ for } r \ll n
```

"""

# ╔═╡ 59b84df2-3f1d-11eb-1b70-a39292a3fe24
compressed(img, rank)

# ╔═╡ 8e2d0f44-3f24-11eb-1688-f56b164b1959
md"""
## Check the Singular Values

Rule of thumb: *If the decrease of SVs is strong, we have a low rank stucture and can compress.*
"""

# ╔═╡ f0f493a4-3f24-11eb-2fe8-d3b5ca07cca4
plotly()

# ╔═╡ 3c1b396a-3f24-11eb-2fea-8d391221ccda
plot(svd(img).S, title="Singular Values", label="sigma_i")

# ╔═╡ Cell order:
# ╟─39f3be04-3f20-11eb-3654-57e887652680
# ╠═7eeeb98c-3f20-11eb-220d-41efe1d23b03
# ╟─3169b458-3f20-11eb-2dba-571a9e073819
# ╠═25579486-3f00-11eb-1405-c3c724fcaf95
# ╟─be3886e8-3f1d-11eb-20e2-2b3003abdc3b
# ╠═cca739f8-3f00-11eb-2c80-dff8b51dbe98
# ╟─00770746-3f23-11eb-1f0d-af0b7ad6086d
# ╠═59b84df2-3f1d-11eb-1b70-a39292a3fe24
# ╟─8e2d0f44-3f24-11eb-1688-f56b164b1959
# ╠═f0f493a4-3f24-11eb-2fe8-d3b5ca07cca4
# ╠═3c1b396a-3f24-11eb-2fea-8d391221ccda
