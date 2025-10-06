# FastSDE.jl

[![CI](https://github.com/ludovico/FastSDE.jl/workflows/CI/badge.svg)](https://github.com/ludovico/FastSDE.jl/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> Fast, minimalist SDE integrator for Julia using the Euler–Maruyama scheme with higher-order deterministic time-stepping (RK2, RK4).

FastSDE.jl provides high-performance stochastic differential equation integration with in-place drift `f!(du, u, p, t)`, explicit parameter passing, automatic dispatch, and support for additive, diagonal, and correlated Gaussian noise.

## Features

- **Multiple time-steppers**: Euler, 2nd-order Runge–Kutta (RK2), and 4th-order Runge–Kutta (RK4)
  - **Note on stochastic order**: When using SDE integration, the stochastic component uses Euler–Maruyama regardless of the deterministic time-stepper. This means the strong convergence order remains 0.5 even when using RK2 or RK4 for the drift term. Higher-order time-steppers improve accuracy of the deterministic component but do not increase the stochastic convergence order.
- **Automatic dispatch**: StaticArrays for small systems (≤ 64 dimensions by default), dynamic arrays with BLAS acceleration for larger systems
- **Parameter support**: Pass parameters explicitly to drift and diffusion functions
- **Flexible noise**: Scalar, diagonal (vector), or correlated (matrix) diffusion
- **Thread-parallel ensembles**: Efficient parallel ensemble integration with per-trajectory RNGs
- **Boundary resets**: Optional automatic reset to initial conditions when crossing specified bounds
- **Allocation-free inner loops**: Optimized for long trajectories and large ensembles

## Installation

```julia
import Pkg
Pkg.add("FastSDE")
```

Or from a local directory:

```julia
import Pkg
Pkg.activate(".")
Pkg.develop(path=pwd())
Pkg.instantiate()
```

## Quick Start

```julia
using FastSDE

# Define drift and diffusion
f!(du, u, p, t) = (du .= -p.γ .* u)
sigma = 0.2  # Scalar diffusion

# Parameters
p = (γ = 1.0,)

# Initial condition
u0 = randn(64)
dt = 1e-3
Nsteps = 10_000

# Single trajectory
traj = evolve(u0, dt, Nsteps, f!, sigma; params=p, timestepper=:rk4, resolution=10)

# Parallel ensemble (8 trajectories)
ens = evolve_ens(u0, dt, Nsteps, f!, sigma; params=p, timestepper=:rk4, n_ens=8, resolution=10)
```

## Usage

### Basic Integration

```julia
using FastSDE

# Linear decay with additive noise
f!(du, u, t) = (du .= -u)
sigma = 0.1

u0 = randn(10)
trajectory = evolve(u0, 0.01, 1000, f!, sigma; resolution=10)
# Returns (10, 101) array
```

### With Parameters

```julia
# Parameters as NamedTuple
params = (α = 0.5, β = 1.0)

f!(du, u, p, t) = (du .= -p.α .* u .+ p.β)
sigma(u, p, t) = p.α  # Time-varying or state-dependent diffusion

traj = evolve(u0, dt, Nsteps, f!, sigma; params=params)
```

### Diagonal and Correlated Noise

```julia
# Diagonal (uncorrelated) noise
sigma_diag = fill(0.1, 10)  # Independent noise per dimension

# Correlated noise
using LinearAlgebra
sigma_corr = 0.1 * I(10)  # Or any positive definite matrix
```

### Ensemble Integration

```julia
# Run 16 trajectories in parallel
ens = evolve_ens(u0, dt, Nsteps, f!, sigma; 
                 n_ens=16, 
                 resolution=10,
                 seed=42)
# Returns (dim, Nsave+1, n_ens) array
```

### Boundary Resets

```julia
# Reset to initial condition when any component leaves [-1, 1]
traj = evolve(u0, dt, Nsteps, f!, sigma; 
              boundary=(-1.0, 1.0),
              verbose=true)  # Prints crossing rate
```

## API Reference

### Main Functions

- `evolve(u0, dt, Nsteps, f!, sigma; kwargs...)`: Integrate a single trajectory
- `evolve_ens(u0, dt, Nsteps, f!, sigma; kwargs...)`: Integrate an ensemble in parallel
- `set_static_threshold!(N)`: Set dimension threshold for StaticArrays path (default: 64)

### Keyword Arguments

- `params=nothing`: Parameters passed to `f!` and `sigma`
- `seed::Integer=123`: Random seed
- `resolution::Integer=1`: Save every `resolution`-th step
- `timestepper::Symbol=:euler`: `:euler`, `:rk2`, or `:rk4`
- `boundary::Union{Nothing,Tuple}=nothing`: `(lo, hi)` bounds for resets
- `n_ens::Integer=1`: Number of ensemble members (`evolve` only, uses threading if > 1)
- `rng::Union{Nothing,AbstractRNG}=nothing`: Custom random number generator
- `verbose::Bool=false`: Print diagnostics
- `manage_blas_threads::Bool=true`: Manage BLAS threading for ensembles
- `sigma_inplace::Bool=true`: Allow in-place diffusion evaluation

## Performance Tips

1. **For small systems** (< 64 dimensions): StaticArrays path is fastest
2. **For large systems**: Dynamic path with BLAS acceleration kicks in automatically
3. **Ensembles**: Use `evolve_ens` for better parallelism than looping over `evolve`
4. **Memory**: Adjust `resolution` to reduce memory footprint for long trajectories
5. **Reproducibility**: Set `seed` for deterministic results across runs

## Why FastSDE?

FastSDE prioritizes speed over mathematical generality:
- In-place APIs minimize allocations
- Explicit parameter passing (no closures)
- Preallocated buffers in inner loops
- StaticArrays for small systems, BLAS for large systems
- Thread-parallel ensembles with independent RNGs

This makes it ideal for:
- Long-running simulations
- Large ensemble studies
- Parameter sweeps and optimization loops
- Integration with machine learning models (e.g., neural ODEs/SDEs)

### Batched NN drift (fast)

When your drift is a neural network, evaluate it **for all ensemble members at once** by enabling `batched_drift=true` and using a batched drift signature:

```julia
using FastSDE, Flux

# 4D MLP drift
nn = Chain(Dense(4,128,tanh), Dense(128,4))
p  = (nn = nn,)

# DU, U are (4, n_ens)
f_nn_batched!(DU, U, p, t) = (DU .= p.nn(U))

σ  = 0.1
u0 = randn(4)

ens = evolve_ens(u0, 1e-3, 100_000, f_nn_batched!, σ;
                 params=p, n_ens=4096, resolution=200, batched_drift=true)
```

This issues large BLAS ops (CPU) or GPU kernels instead of `n_ens` small calls, giving a substantial speedup for ML-based drifts. Noise types `Number` (additive), `Vector` (diagonal), `Matrix` (correlated via cached Cholesky), and in‑place `sigma!(Ξ,U,p,t)` with `sigma_inplace=true` are supported exactly as in the columnwise path.

#### Batched diffusion (in-place)

When using `batched_drift=true` with an in-place diffusion function, the signature must be:

```julia
sigma!(Xi, U, p, t)
```

where:
- `Xi` is a `(D, n_ens)` matrix pre-filled with `N(0,1)` samples
- `U` is the current state matrix `(D, n_ens)`
- The function must transform `Xi` in-place to `Σ^{1/2}(U,t) * Xi`

Example with state-dependent diagonal diffusion:

```julia
# Xi will be modified in-place: Xi[i,j] ← σ(U[i,j]) * Xi[i,j]
function sigma_batched!(Xi, U, p, t)
    @. Xi = p.sigma_scale * sqrt(abs(U)) * Xi
    return nothing
end

ens = evolve_ens(u0, dt, Nsteps, f_batched!, sigma_batched!;
                 params=(sigma_scale=0.1,), n_ens=256, batched_drift=true, sigma_inplace=true)
```

**Note**: Callable (returning) sigma forms are not yet supported in batched mode unless they return a scalar, vector, or constant matrix.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Reporting bugs
- Suggesting enhancements
- Submitting pull requests
- Code style and testing conventions

## Citation

If you use FastSDE.jl in your research, please cite:

```bibtex
@software{fastsde2025,
  author = {Giorgini, Ludovico Theo},
  title = {FastSDE.jl: Fast Stochastic Differential Equation Integration in Julia},
  year = {2025},
  url = {https://github.com/ludovico/FastSDE.jl}
}
```

## Related Packages

- [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) - Comprehensive suite with adaptive methods
- [StochasticDiffEq.jl](https://github.com/SciML/StochasticDiffEq.jl) - Advanced SDE solvers with high-order schemes
- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) - Fast fixed-size arrays (used internally)

## Acknowledgments

FastSDE.jl prioritizes performance for long-running simulations and large ensemble studies. For production applications requiring adaptive time-stepping or higher-order stochastic convergence, consider using StochasticDiffEq.jl from the SciML ecosystem.

## License

MIT License - see [LICENSE](LICENSE) for details.

Copyright (c) 2025 Ludovico Theo Giorgini
