"""
Utility functions shared across FastSDE.jl
"""

"""
    validate_inputs(u0, dt, Nsteps; resolution=1, n_burnin=0)

Validate common input parameters and provide helpful error messages.
"""
function validate_inputs(u0::AbstractVector, dt::Real, Nsteps::Integer;
                        resolution::Integer=DEFAULT_RESOLUTION,
                        n_burnin::Integer=0)
    if dt <= 0
        throw(ArgumentError(
            "Time step dt must be positive, got dt=$dt. " *
            "Common values are 1e-3 to 1e-2 for stiff systems."
        ))
    end

    if Nsteps <= 0
        throw(ArgumentError(
            "Number of steps Nsteps must be positive, got Nsteps=$Nsteps."
        ))
    end

    if resolution < 1
        throw(ArgumentError(
            "resolution must be ≥ 1 (saves every resolution-th step), got $resolution. " *
            "Use resolution=1 to save all steps, resolution=10 for every 10th step, etc."
        ))
    end

    if n_burnin < 0
        throw(ArgumentError(
            "n_burnin must be ≥ 0, got n_burnin=$n_burnin. " *
            "Use n_burnin to discard initial transient steps from ensemble output."
        ))
    end

    if n_burnin >= Nsteps
        throw(ArgumentError(
            "n_burnin must be < Nsteps, got n_burnin=$n_burnin, Nsteps=$Nsteps. " *
            "Cannot discard all or more steps than will be computed."
        ))
    end

    if length(u0) == 0
        throw(ArgumentError(
            "Initial condition u0 must be non-empty."
        ))
    end

    return nothing
end

"""
    estimate_memory_usage(u0, Nsteps; resolution=1, n_ens=1)

Estimate memory usage in bytes for trajectory storage.
Returns (bytes, gigabytes).
"""
function estimate_memory_usage(u0::AbstractVector, Nsteps::Integer;
                               resolution::Integer=1, n_ens::Integer=1)
    dim = length(u0)
    T = eltype(u0)
    Nsave = fld(Nsteps, resolution) + 1
    bytes = sizeof(T) * dim * Nsave * n_ens
    gb = bytes / 1e9
    return bytes, gb
end

"""
    warn_large_memory(u0, Nsteps; resolution=1, n_ens=1, threshold_gb=4.0)

Warn user if estimated memory usage exceeds threshold.
"""
function warn_large_memory(u0::AbstractVector, Nsteps::Integer;
                          resolution::Integer=1, n_ens::Integer=1,
                          threshold_gb::Real=4.0)
    bytes, gb = estimate_memory_usage(u0, Nsteps; resolution, n_ens)

    if gb > threshold_gb && resolution == 1
        dim = length(u0)
        @warn """
        Estimated memory usage: $(round(gb, digits=2)) GB
        Consider increasing 'resolution' to reduce memory footprint.
        For example, resolution=10 would reduce memory by ~10x.
        Current: $(dim) dimensions × $(fld(Nsteps, resolution) + 1) timesteps × $(n_ens) ensemble members
        """ maxlog=1
    end

    return nothing
end

"""
    _crossed(u, lo, hi)

Check if any component of u is outside [lo, hi].
"""
@inline function _crossed(u, lo::Real, hi::Real)
    @inbounds for x in u
        if x < lo || x > hi
            return true
        end
    end
    return false
end

"""
    _should_save(step, next_save)

Check if current step should be saved.
"""
@inline _should_save(step::Integer, next_save::Integer) = step == next_save
