########################
# Hybrid dispatch (public API)
########################

# Default cutoff between StaticArrays and the dynamic path.
# Can be adjusted at runtime via `set_static_threshold!(N)`.
const _STATIC_THRESHOLD = Ref(64)

"""
    set_static_threshold!(N::Integer)

Set the dimension threshold under which the StaticArrays path is used.
Returns the updated threshold value.
"""
function set_static_threshold!(N::Integer)
    N < 1 && throw(ArgumentError("Threshold must be ≥ 1"))
    _STATIC_THRESHOLD[] = N
    return _STATIC_THRESHOLD[]
end

# Public API: `evolve`, `evolve_ens`

"""
    evolve(u0, dt, Nsteps, f!, sigma; kwargs...)

Integrate a single trajectory using the StaticArrays path when `length(u0) ≤ threshold`,
otherwise the dynamic path. Returns an array of size `(dim, Nsave+1)`.
"""
function evolve(u0::AbstractVector, dt, Nsteps::Integer, f!, sigma;
                params::Any = nothing,                       # <-- NEW
                seed::Integer=123, resolution::Integer=1,
                timestepper::Symbol=:rk4, boundary::Union{Nothing,Tuple}=nothing,
                n_ens::Integer=1, rng::Union{Nothing,AbstractRNG}=nothing,
                verbose::Bool=false, flatten::Bool=true, manage_blas_threads::Bool=true,
                sigma_inplace::Bool=false)

    N = length(u0)
    if n_ens != 1
        arr = evolve_ens(u0, dt, Nsteps, f!, sigma;
                         params=params, seed, resolution, timestepper, boundary,
                         n_ens, rng, verbose, manage_blas_threads, sigma_inplace=sigma_inplace)
        if flatten
            dim, timesteps, ensembles = size(arr)
            T = eltype(arr)
            out = Array{T}(undef, dim, timesteps*ensembles)
            @inbounds for i in 1:ensembles
                @views out[:, (i-1)*timesteps+1 : i*timesteps] .= arr[:, :, i]
            end
            return out
        else
            return arr
        end
    end

    if N <= _STATIC_THRESHOLD[]
        return evolve_static(u0, dt, Nsteps, f!, sigma;
                             params=params, seed, resolution, timestepper, boundary,
                             rng=rng, verbose=verbose, sigma_inplace=sigma_inplace)
    else
        return evolve_dyn(u0, dt, Nsteps, f!, sigma;
                          params=params, seed, resolution, timestepper, boundary,
                          rng=rng, verbose=verbose, sigma_inplace=sigma_inplace)
    end
end

"""
    evolve_ens(u0, dt, Nsteps, f!, sigma; kwargs...)

Integrate an ensemble using the StaticArrays path when `length(u0) ≤ threshold`,
otherwise the dynamic path. Returns an array of size `(dim, Nsave+1, n_ens)`.
"""
function evolve_ens(u0::AbstractVector, dt, Nsteps::Integer, f!, sigma;
                    params::Any = nothing,                       # <-- NEW
                    seed::Integer=123, resolution::Integer=1,
                    timestepper::Symbol=:rk4, boundary::Union{Nothing,Tuple}=nothing,
                    n_ens::Integer=1, rng::Union{Nothing,AbstractRNG}=nothing,
                    verbose::Bool=false, manage_blas_threads::Bool=true,
                    sigma_inplace::Bool=false)

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
            return evolve_ens_static(u0, dt, Nsteps, f!, sigma;
                                     params=params, seed, resolution, timestepper, boundary,
                                     n_ens, rng=rng, verbose=verbose, sigma_inplace=sigma_inplace)
        else
            return evolve_ens_dyn(u0, dt, Nsteps, f!, sigma;
                                  params=params, seed, resolution, timestepper, boundary,
                                  n_ens, rng=rng, verbose=verbose, sigma_inplace=sigma_inplace)
        end
    finally
        if manage_blas_threads && n_ens > 1 && old_blas_threads !== nothing
            LinearAlgebra.BLAS.set_num_threads(old_blas_threads)
        end
    end
end
