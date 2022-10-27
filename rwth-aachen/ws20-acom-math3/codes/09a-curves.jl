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

# ╔═╡ 74f62a8c-5002-11eb-3d83-1d8870060fdc
using PlutoUI, Plots, Calculus

# ╔═╡ 8bfac3c6-5009-11eb-149f-dd26e31bb041
md"""
# Curves

- Mathe 3 (CES)
- WS20
- Lambert Theisen (```theisen@acom.rwth-aachen.de```)
"""

# ╔═╡ 74f61572-5002-11eb-01ba-37fb5ac1d5d3
plotly()

# ╔═╡ f364ad2c-5009-11eb-215f-01fbca4c17c5
md"""
## Define a Curve
Define the curve $\gamma$:
```math
γ:[0, 6\pi] \to \mathbb{R}^2, t \mapsto \gamma(t) = (\sin(t/3), 5\sin(2t))^T
```
"""

# ╔═╡ dd9ed9fc-5006-11eb-1c34-1f7c3f86deba
γ(t) = [sin(t/3), 5sin(2t)]

# ╔═╡ ec83eac2-5004-11eb-12fd-df144b6b2f70
@bind t Slider(0:π/20:6π, show_value=true)

# ╔═╡ a4c575e0-5002-11eb-28b6-8baab01da743
begin
	plot(t->γ(t)[1], t->γ(t)[2], 0, 6π, leg=false)
	scatter!([γ(t)[1]], [γ(t)[2]])
end

# ╔═╡ Cell order:
# ╟─8bfac3c6-5009-11eb-149f-dd26e31bb041
# ╠═74f62a8c-5002-11eb-3d83-1d8870060fdc
# ╠═74f61572-5002-11eb-01ba-37fb5ac1d5d3
# ╟─f364ad2c-5009-11eb-215f-01fbca4c17c5
# ╠═dd9ed9fc-5006-11eb-1c34-1f7c3f86deba
# ╠═ec83eac2-5004-11eb-12fd-df144b6b2f70
# ╠═a4c575e0-5002-11eb-28b6-8baab01da743
