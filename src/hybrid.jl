########################
# Hybrid dispatch (public API)
########################

# Default cutoff between StaticArrays and the dynamic path.
# Can be adjusted at runtime via `set_static_threshold!(N)`.
const _STATIC_THRESHOLD = Ref(64)

"""
    set_static_threshold!(N::Integer)

Set the dimension threshold for choosing between StaticArrays and dynamic array paths.

Systems with `length(u0) ≤ N` use the StaticArrays fast path, optimized for small,
fixed-size systems. Larger systems use the dynamic path with in-place operations
and BLAS acceleration where appropriate.

# Arguments
- `N::Integer`: Threshold dimension (must be ≥ 1). Default is 64.

# Returns
- `Int`: The updated threshold value.

# Example
```julia
set_static_threshold!(32)  # Use static arrays for systems up to size 32
```
"""
function set_static_threshold!(N::Integer)
    N < 1 && throw(ArgumentError("Threshold must be ≥ 1"))
    _STATIC_THRESHOLD[] = N
    return _STATIC_THRESHOLD[]
end

# Public API: `evolve`, `evolve_ens`

"""
    evolve(u0, dt, Nsteps, f!, sigma; kwargs...)

Integrate a stochastic or deterministic trajectory using Euler–Maruyama or higher-order
time-stepping methods.

Automatically selects between StaticArrays (small systems) and dynamic arrays (large systems)
based on the current threshold (see [`set_static_threshold!`](@ref)).

# Arguments
- `u0::AbstractVector`: Initial condition.
- `dt`: Time step size.
- `Nsteps::Integer`: Number of integration steps.
- `f!`: In-place drift function. Signature: `f!(du, u, t)` or `f!(du, u, p, t)`.
- `sigma`: Diffusion coefficient. Can be a scalar, vector, matrix, or callable `(u, t)` / `(u, p, t)`.

# Keyword Arguments
- `params=nothing`: Parameters passed to `f!` and `sigma` (when callable).
- `seed::Integer=123`: Random seed for reproducibility.
- `resolution::Integer=1`: Save every `resolution`-th step.
- `n_burnin::Integer=0`: Number of initial steps to discard when `n_ens > 1`.
- `timestepper::Symbol=:euler`: Time-stepping method (`:euler`, `:rk2`, `:rk4`).
- `boundary::Union{Nothing,Tuple}=nothing`: Optional `(lo, hi)` bounds for boundary reset.
- `n_ens::Integer=1`: Number of ensemble members (uses thread parallelism if > 1).
- `rng::Union{Nothing,AbstractRNG}=nothing`: Custom random number generator.
- `verbose::Bool=false`: Print diagnostics (e.g., boundary crossing rate).
- `flatten::Bool=true`: Flatten ensemble output to 2D when `n_ens > 1`.
- `manage_blas_threads::Bool=true`: Disable BLAS threading during ensemble runs.
- `sigma_inplace::Bool=true`: Allow in-place diffusion evaluation when possible.

# Returns
- `Array{T,2}`: Trajectory data of size `(dim, Nsave+1)` for single trajectory, or
  `(dim, Nsave*n_ens)` when `n_ens > 1` and `flatten=true`.

# Example
```julia
using FastSDE

# Define drift and diffusion
f!(du, u, t) = (du .= -u)
sigma = 0.1

u0 = randn(10)
traj = evolve(u0, 0.01, 1000, f!, sigma; timestepper=:rk4, resolution=10)
```
"""
function evolve(u0::AbstractVector, dt, Nsteps::Integer, f!, sigma;
                params::Any = nothing,                       
                seed::Integer=123, resolution::Integer=1, n_burnin::Integer=0,
                timestepper::Symbol=:euler, boundary::Union{Nothing,Tuple}=nothing,
                n_ens::Integer=1, rng::Union{Nothing,AbstractRNG}=nothing,
                verbose::Bool=false, flatten::Bool=true, manage_blas_threads::Bool=true,
                sigma_inplace::Bool=true)

    N = length(u0)
    if n_ens != 1
        arr = evolve_ens(u0, dt, Nsteps, f!, sigma;
                         params=params, seed, resolution, timestepper, boundary,
                         n_ens, rng, verbose, manage_blas_threads, sigma_inplace=sigma_inplace)
        
        dim, timesteps, ensembles = size(arr)
        
        # Apply burn-in if requested
        if n_burnin > 0
            # Convert burn-in steps to saved timesteps (accounting for resolution)
            # Add 1 because first index is the initial condition (t=0)
            burnin_saved = min(fld(n_burnin, resolution) + 1, timesteps - 1)
            
            # Discard first burnin_saved timesteps from each trajectory
            arr_trimmed = arr[:, (burnin_saved+1):end, :]
        else
            arr_trimmed = arr
        end
        _, timesteps_trimmed, _ = size(arr_trimmed)
        
        if flatten
            T = eltype(arr_trimmed)
            out = Array{T}(undef, dim, timesteps_trimmed*ensembles)
            @inbounds for i in 1:ensembles
                @views out[:, (i-1)*timesteps_trimmed+1 : i*timesteps_trimmed] .= arr_trimmed[:, :, i]
            end
            return out
        else
            return arr_trimmed
        end
    end

    if N <= _STATIC_THRESHOLD[]
        return _evolve_static(u0, dt, Nsteps, f!, sigma;
                              params=params, seed, resolution, timestepper, boundary,
                              rng=rng, verbose=verbose, sigma_inplace=sigma_inplace)
    else
        return _evolve_dyn(u0, dt, Nsteps, f!, sigma;
                           params=params, seed, resolution, timestepper, boundary,
                           rng=rng, verbose=verbose, sigma_inplace=sigma_inplace)
    end
end

"""
    evolve_ens(u0, dt, Nsteps, f!, sigma; kwargs...)

Integrate an ensemble of trajectories in parallel using threading.

Each ensemble member uses an independent random number generator for reproducibility.
BLAS threading is automatically disabled during ensemble runs to avoid oversubscription
(controlled by `manage_blas_threads`).

# Arguments
- `u0::AbstractVector`: Initial condition.
- `dt`: Time step size.
- `Nsteps::Integer`: Number of integration steps.
- `f!`: In-place drift function. Signature: `f!(du, u, t)` or `f!(du, u, p, t)`.
- `sigma`: Diffusion coefficient. Can be a scalar, vector, matrix, or callable `(u, t)` / `(u, p, t)`.

# Keyword Arguments
- `params=nothing`: Parameters passed to `f!` and `sigma` (when callable).
- `seed::Integer=123`: Base random seed (each ensemble member gets `seed + ens_idx * 1000`).
- `resolution::Integer=1`: Save every `resolution`-th step.
- `timestepper::Symbol=:rk4`: Time-stepping method (`:euler`, `:rk2`, `:rk4`).
- `boundary::Union{Nothing,Tuple}=nothing`: Optional `(lo, hi)` bounds for boundary reset.
- `n_ens::Integer=1`: Number of ensemble members.
- `rng::Union{Nothing,AbstractRNG}=nothing`: Custom random number generator for seed generation.
- `verbose::Bool=false`: Print diagnostics for first ensemble member.
- `manage_blas_threads::Bool=true`: Disable BLAS threading during ensemble runs.
- `sigma_inplace::Bool=true`: Allow in-place diffusion evaluation when possible.
- `batched_drift::Bool=false`: Enable batched drift evaluation (all ensemble members at once).

# Returns
- `Array{T,3}`: Ensemble data of size `(dim, Nsave+1, n_ens)`.

# Example
```julia
using FastSDE

f!(du, u, t) = (du .= -u)
sigma = 0.1

u0 = randn(10)
ens = evolve_ens(u0, 0.01, 1000, f!, sigma; n_ens=8, resolution=10)
# Result is (10, 101, 8)
```
"""
function evolve_ens(u0::AbstractVector, dt, Nsteps::Integer, f!, sigma;
                    params::Any = nothing,                       # <-- NEW
                    seed::Integer=123, resolution::Integer=1,
                    timestepper::Symbol=:rk4, boundary::Union{Nothing,Tuple}=nothing,
                    n_ens::Integer=1, rng::Union{Nothing,AbstractRNG}=nothing,
                    verbose::Bool=false, manage_blas_threads::Bool=true,
                    sigma_inplace::Bool=true, batched_drift::Bool=false)

    if batched_drift
        return _evolve_ens_batched(u0, dt, Nsteps, f!, sigma;
            params=params, n_ens=n_ens, resolution=resolution, seed=seed,
            timestepper=timestepper, boundary=boundary, rng=rng, verbose=verbose,
            manage_blas_threads=manage_blas_threads, sigma_inplace=sigma_inplace)
    end

    N = length(u0)

    # When running ensembles across threads, disable BLAS threading to avoid
    # oversubscription and make full use of outer parallelism.
    old_blas_threads = nothing
    if manage_blas_threads && n_ens > 1
        old_blas_threads = LinearAlgebra.BLAS.get_num_threads()
        LinearAlgebra.BLAS.set_num_threads(1)
    end

    try
        if N <= _STATIC_THRESHOLD[]
            return _evolve_ens_static(u0, dt, Nsteps, f!, sigma;
                                      params=params, seed, resolution, timestepper, boundary,
                                      n_ens, rng=rng, verbose=verbose, sigma_inplace=sigma_inplace)
        else
            return _evolve_ens_dyn(u0, dt, Nsteps, f!, sigma;
                                   params=params, seed, resolution, timestepper, boundary,
                                   n_ens, rng=rng, verbose=verbose, sigma_inplace=sigma_inplace)
        end
    finally
        if manage_blas_threads && n_ens > 1 && old_blas_threads !== nothing
            LinearAlgebra.BLAS.set_num_threads(old_blas_threads)
        end
    end
end
