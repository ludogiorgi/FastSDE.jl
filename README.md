# FastSDE

Fast, minimalist SDE integrator for Julia using the Euler–Maruyama scheme with higher-order deterministic time-stepping (RK2, RK4). Features in-place drift `f!(du, u, p, t)`, explicit parameter passing, preallocation, StaticArrays fast path for small systems, and support for additive, diagonal, and correlated Gaussian noise.

## Features

- **Multiple time-steppers**: Euler, 2nd-order Runge–Kutta (RK2), and 4th-order Runge–Kutta (RK4)
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

## License

MIT
