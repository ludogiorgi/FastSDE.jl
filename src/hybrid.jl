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
                seed::Integer=123, resolution::Integer=1,
                timestepper::Symbol=:rk4, boundary::Union{Nothing,Tuple}=nothing,
                n_ens::Integer=1)

    N = length(u0)
    if n_ens != 1
        # Delegate to ensemble, then reshape to 2D for convenience
        arr = evolve_ens(u0, dt, Nsteps, f!, sigma; seed, resolution, timestepper, boundary, n_ens)
        dim, timesteps, ensembles = size(arr)
        T = eltype(arr)
        out = Array{T}(undef, dim, timesteps*ensembles)
        @inbounds for i in 1:ensembles
            @views out[:, (i-1)*timesteps+1 : i*timesteps] .= arr[:, :, i]
        end
        return out
    end

    if N <= _STATIC_THRESHOLD[]
        return evolve_static(u0, dt, Nsteps, f!, sigma; seed, resolution, timestepper, boundary)
    else
        return evolve_dyn(u0, dt, Nsteps, f!, sigma; seed, resolution, timestepper, boundary)
    end
end

"""
    evolve_ens(u0, dt, Nsteps, f!, sigma; kwargs...)

Integrate an ensemble using the StaticArrays path when `length(u0) ≤ threshold`,
otherwise the dynamic path. Returns an array of size `(dim, Nsave+1, n_ens)`.
"""
function evolve_ens(u0::AbstractVector, dt, Nsteps::Integer, f!, sigma;
                    seed::Integer=123, resolution::Integer=1,
                    timestepper::Symbol=:rk4, boundary::Union{Nothing,Tuple}=nothing,
                    n_ens::Integer=1)

    N = length(u0)
    if N <= _STATIC_THRESHOLD[]
        return evolve_ens_static(u0, dt, Nsteps, f!, sigma; seed, resolution, timestepper, boundary, n_ens)
    else
        return evolve_ens_dyn(u0, dt, Nsteps, f!, sigma; seed, resolution, timestepper, boundary, n_ens)
    end
end
