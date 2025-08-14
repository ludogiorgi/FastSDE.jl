````markdown
## FastSDE

Fast, minimalist SDE/ODE integrators with in-place drift `f!(du,u,p,t)`, explicit parameter passing, preallocation, a StaticArrays fast path, and additive/diagonal/correlated Gaussian noise.

### Install

```julia
import Pkg
Pkg.activate(".")
Pkg.develop(path=PWD())
Pkg.instantiate()
````

### Use

```julia
using FastSDE

# Parameters (can be NamedTuple, struct, or even a neural net)
p = (γ = 1.0,)

# Drift and diffusion must accept `p` explicitly
f!(du, u, p, t) = (du .= -p.γ .* u)
σ = 0.2                 # or fill(0.2, length(u)), or 0.2 .* I

N = 64
u0 = randn(N)
dt = 1e-3
steps = 10_000

# Single trajectory
traj = evolve(u0, dt, steps, f!, σ; params=p, timestepper=:rk4, resolution=10)

# Parallel ensemble
ens  = evolve_ens(u0, dt, steps, f!, σ; params=p, timestepper=:rk4, n_ens=8)
```

### Benchmark (parallel ensembles)

* Setup: identical drift `f!(u)=-u` and identical diffusion matrix `Σ` (N×N) for both methods; `dt=1e-3`, `Nsteps=200`, full saving; `n_ens ≥ 16`.
* Methods: FastSDE uses `:rk4`; StochasticDiffEq uses **SRA3** (highest-order additive-noise method available).

| N   | FastSDE (s) | StochasticDiffEq (s) |
| --- | ----------- | -------------------- |
| 8   | 0.000       | 0.011                |
| 32  | 0.001       | 0.143                |
| 128 | 0.009       | 1.751                |
| 512 | 0.055       | 43.116               |

Notes:

* Both methods receive the same `Σ`, `u0`, and `p`, integrate with the same `dt` and `Nsteps`, save every step, and run `n_ens` trajectories in parallel.
* Explicit `params=p` avoids global captures, ensures type stability for complex drifts (e.g. neural networks), and preserves performance for constant and non-constant parameters alike.

### Why FastSDE

* Speed-first design: in-place APIs, explicit parameter passing, allocation-free inner loops, and StaticArrays/BLAS fast paths.
* Ideal for large ensembles/long trajectories when approximate noise handling is acceptable.
* Flexible: works seamlessly with constant parameters, mutable parameters, and learned models.

### License

MIT

```
```
