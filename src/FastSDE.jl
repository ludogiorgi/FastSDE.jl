module FastSDE

"""
FastSDE

Lightweight utilities to evolve stochastic and deterministic dynamical systems
defined by user-supplied drift `f!(du, u, t)` and diffusion `sigma(u, t)`.

Features:
- Deterministic steppers: 4th-order Runge–Kutta (`rk4_step!`) and Euler (`euler_step!`).
- Single-trajectory integrator `evolve` for SDEs/ODEs with optional boundary reset.
- Thread-parallel ensemble integrator `evolve_ens` with per-trajectory RNGs and
  preallocated buffers for high performance.
- StaticArrays fast path for small, fixed sizes; dynamic path for large systems.
"""

using LinearAlgebra
using Base.Threads
using Random
using StaticArrays

# Shared helper: avoid duplicate definitions across includes
@inline _call_f!(f!, du, u, t, p) = (p === nothing ? f!(du, u, t) : f!(du, u, p, t))

# Implementation files
include("integrators_dynamic.jl")
include("integrators_static.jl")
include("hybrid.jl")

# Public API
export evolve, evolve_ens, set_static_threshold!

end # module
