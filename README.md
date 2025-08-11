## FastSDE

Fast, minimalist SDE/ODE integrators with in-place drift `f!(du,u,t)`, preallocation, a StaticArrays fast path, and additive/diagonal/correlated Gaussian noise.

### Install

```julia
import Pkg
Pkg.activate(".")
Pkg.develop(path=PWD())
Pkg.instantiate()
```

### Use

```julia
using FastSDE

f!(du,u,t) = (du .= -u)
N=64; u0=randn(N); dt=1e-3; steps=10_000
σ = 0.2                 # or fill(0.2,N), or 0.2.*I

traj = evolve(u0, dt, steps, f!, σ; timestepper=:rk4, resolution=10)
ens  = evolve_ens(u0, dt, steps, f!, σ; timestepper=:rk4, n_ens=8)
```

### Benchmark (parallel ensembles)

- Setup: identical drift `f(u)=-u` and identical diffusion matrix `Σ` (N×N) for both methods; `dt=1e-3`, `Nsteps=200`, full saving; `n_ens ≥ 16`.
- Methods: FastSDE uses `:rk4`; StochasticDiffEq uses **SRA3** (highest-order additive-noise method available).

| N   | FastSDE (s) | StochasticDiffEq (s) |
|-----|-------------|-----------------------|
| 8   | 0.000       | 0.011                 |
| 32  | 0.001       | 0.143                 |
| 128 | 0.009       | 1.751                 |
| 512 | 0.055       | 43.116                |

Notes:
- Both methods receive the same `Σ` and `u0`, integrate with the same `dt` and `Nsteps`, save every step, and run `n_ens` trajectories in parallel.

### Why FastSDE

- Speed-first design: in-place APIs, allocation-free inner loops, and StaticArrays/BLAS fast paths.
- Ideal for large ensembles/long trajectories when approximate noise handling is acceptable.

### License

MIT


# FastSDE
