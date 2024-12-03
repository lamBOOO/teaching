# Teaching Archive

## Run Julia Pluto Notebooks

- powered by [Pluto on Binder!](https://pluto-on-binder.glitch.me/)

- e.g. [Math3 WS20 Week 14](https://binder.plutojl.org/open?url=https%253A%252F%252Fgithub.com%252FlamBOOO%252Fteaching%252Fblob%252Fmain%252Frwth%252Fws20-acom-math3%252Fcodes%252F14-constrained-optimization-penalty-barrier.jl%253Fraw%253Dtrue)

## Local execution
```
julia
using Pkg
Pkg.add("Pluto")
using Pluto
Pluto.run(host="0.0.0.0", port=1234)  # allows exposure of socket when using, e.g., Docker
# Goto localhost:1234 in browser and load the notebooks
```
