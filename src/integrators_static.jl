########################
# Static-path integrators (StaticArrays) — optimized
########################

using StaticArrays
using Random
using LinearAlgebra: mul!

# --- Call shims (params or no params) ---
@inline _call_f!(f!, du, u, t, p) = (p === nothing ? f!(du, u, t) : f!(du, u, p, t))

# --- Time steppers (in-place, static) ---

"""
    rk4_step_static!(u, dt, f!, t, params, k1, k2, k3, k4, tmp)
"""
@inline function rk4_step_static!(
    u::MVector{N,T}, dt::T, f!, t::T, params,
    k1::MVector{N,T}, k2::MVector{N,T}, k3::MVector{N,T},
    k4::MVector{N,T}, tmp::MVector{N,T},
) where {N,T}
    _call_f!(f!, k1, u, t, params)
    @inbounds @simd for i in 1:N
        tmp[i] = u[i] + 0.5 * dt * k1[i]
    end
    _call_f!(f!, k2, tmp, t + 0.5 * dt, params)
    @inbounds @simd for i in 1:N
        tmp[i] = u[i] + 0.5 * dt * k2[i]
    end
    _call_f!(f!, k3, tmp, t + 0.5 * dt, params)
    @inbounds @simd for i in 1:N
        tmp[i] = u[i] + dt * k3[i]
    end
    _call_f!(f!, k4, tmp, t + dt, params)
    @inbounds @simd for i in 1:N
        u[i] += (dt / 6) * (k1[i] + 2k2[i] + 2k3[i] + k4[i])
    end
    return nothing
end

"""
    rk2_step_static!(u, dt, f!, t, params, k1, k2, tmp)

Second-order Runge–Kutta (midpoint) deterministic step.
"""
@inline function rk2_step_static!(
    u::MVector{N,T}, dt::T, f!, t::T, params,
    k1::MVector{N,T}, k2::MVector{N,T}, tmp::MVector{N,T},
) where {N,T}
    _call_f!(f!, k1, u, t, params)
    @inbounds @simd for i in 1:N
        tmp[i] = u[i] + 0.5 * dt * k1[i]
    end
    _call_f!(f!, k2, tmp, t + 0.5 * dt, params)
    @inbounds @simd for i in 1:N
        u[i] += dt * k2[i]
    end
    return nothing
end

"""
    euler_step_static!(u, dt, f!, t, params, k1)
"""
@inline function euler_step_static!(
    u::MVector{N,T}, dt::T, f!, t::T, params, k1::MVector{N,T},
) where {N,T}
    _call_f!(f!, k1, u, t, params)
    @inbounds @simd for i in 1:N
        u[i] += dt * k1[i]
    end
    return nothing
end

# --- Noise adders (static) ---

@inline function _mulS!(y::MVector{N,T}, A::SMatrix{N,N,T}, x::MVector{N,T}) where {N,T}
    @inbounds for i in 1:N
        s = zero(T)
        @simd for j in 1:N
            s += A[i,j] * x[j]
        end
        y[i] = s
    end
    return nothing
end

@inline function add_noise_static!(
    u::MVector{N,T}, s::T, σ::T, rng::AbstractRNG, ξ::MVector{N,T},
) where {N,T}
    randn!(rng, ξ)
    @inbounds @simd for i in 1:N
        u[i] += s * σ * ξ[i]
    end
    return nothing
end

@inline function add_noise_static!(
    u::MVector{N,T}, s::T, σ::SVector{N,T}, rng::AbstractRNG, ξ::MVector{N,T},
) where {N,T}
    randn!(rng, ξ)
    @inbounds @simd for i in 1:N
        u[i] += s * (σ[i] * ξ[i])
    end
    return nothing
end

@inline function add_noise_static!(
    u::MVector{N,T}, s::T, σ::MVector{N,T}, rng::AbstractRNG, ξ::MVector{N,T},
) where {N,T}
    randn!(rng, ξ)
    @inbounds @simd for i in 1:N
        u[i] += s * (σ[i] * ξ[i])
    end
    return nothing
end

@inline function add_noise_static!(
    u::MVector{N,T}, s::T, Σ::SMatrix{N,N,T},
    rng::AbstractRNG, ξ::MVector{N,T}, tmp::MVector{N,T},
) where {N,T}
    randn!(rng, ξ)
    _mulS!(tmp, Σ, ξ)
    @inbounds @simd for i in 1:N
        u[i] += s * tmp[i]
    end
    return nothing
end

@inline function add_noise_static!(
    u::MVector{N,T}, s::T, Σ::MMatrix{N,N,T},
    rng::AbstractRNG, ξ::MVector{N,T}, tmp::MVector{N,T},
) where {N,T}
    randn!(rng, ξ)
    mul!(tmp, Σ, ξ)
    @inbounds @simd for i in 1:N
        u[i] += s * tmp[i]
    end
    return nothing
end

# Lift constants to callable sigma(u,t) once (static path)
@inline _make_sigma_static(sigma) = sigma
@inline _make_sigma_static(sigma::Real) = (u, t) -> sigma
@inline _make_sigma_static(sigma::SVector) = (u, t) -> sigma
@inline _make_sigma_static(sigma::SMatrix) = (u, t) -> sigma
@inline _make_sigma_static(sigma::AbstractVector) = (u, t) -> sigma
@inline _make_sigma_static(sigma::AbstractMatrix) = (u, t) -> sigma

# --- Noise applier builders ---

function _make_noise_applier_static(sigma_any, u0::MVector{N,T}, t0::T;
                                    sigma_inplace::Bool=true) where {N,T}
    if sigma_any isa Function
        σ_probe = MVector{N,T}(undef)
        try
            sigma_any(σ_probe, u0, t0)             # sigma!(σ, u, t)
            return (u, s, t, rng, ξ, tmp) -> begin
                sigma_any(σ_probe, u, t)
                add_noise_static!(u, s, σ_probe, rng, ξ)
            end
        catch err
            if !(err isa MethodError); rethrow(); end
        end
        Σ_probe = MMatrix{N,N,T}(undef)
        try
            sigma_any(Σ_probe, u0, t0)             # sigma!(Σ, u, t)
            return (u, s, t, rng, ξ, tmp) -> begin
                sigma_any(Σ_probe, u, t)
                add_noise_static!(u, s, Σ_probe, rng, ξ, tmp)
            end
        catch err
            if !(err isa MethodError); rethrow(); end
        end
    end

    if sigma_any isa AbstractVector
        σ_const = SVector{N,T}(sigma_any)
        return (u, s, t, rng, ξ, tmp) -> add_noise_static!(u, s, σ_const, rng, ξ)
    elseif sigma_any isa AbstractMatrix
        Σ_const = SMatrix{N,N,T}(sigma_any)
        return (u, s, t, rng, ξ, tmp) -> add_noise_static!(u, s, Σ_const, rng, ξ, tmp)
    end

    sigma = _make_sigma_static(sigma_any)
    sig0 = sigma(u0, t0)
    if sig0 isa Real
        return (u, s, t, rng, ξ, tmp) -> add_noise_static!(u, s, (sigma(u, t))::T, rng, ξ)
    elseif sig0 isa SVector{N,T}
        return (u, s, t, rng, ξ, tmp) -> add_noise_static!(u, s, (sigma(u, t))::SVector{N,T}, rng, ξ)
    elseif sig0 isa SMatrix{N,N,T}
        return (u, s, t, rng, ξ, tmp) -> add_noise_static!(u, s, (sigma(u, t))::SMatrix{N,N,T}, rng, ξ, tmp)
    elseif sig0 isa AbstractVector
        return (u, s, t, rng, ξ, tmp) -> begin
            σ_vec = SVector{N,T}(sigma(u, t))
            add_noise_static!(u, s, σ_vec, rng, ξ)
        end
    elseif sig0 isa AbstractMatrix
        return (u, s, t, rng, ξ, tmp) -> begin
            Σ_mat = SMatrix{N,N,T}(sigma(u, t))
            add_noise_static!(u, s, Σ_mat, rng, ξ, tmp)
        end
    else
        error("Unsupported σ type for static path: $(typeof(sig0))")
    end
end

"""
With-params builder (preferred when params ≠ nothing).
Supports sigma!(out,u,p,t) and sigma(u,p,t).
"""
function _make_noise_applier_static_with_params(sigma_any, u0::MVector{N,T}, t0::T, params;
                                                sigma_inplace::Bool=true) where {N,T}
    if sigma_any isa Function
        σ_probe = MVector{N,T}(undef)
        try
            sigma_any(σ_probe, u0, params, t0)     # sigma!(σ,u,p,t)
            return (u, s, t, rng, ξ, tmp) -> begin
                sigma_any(σ_probe, u, params, t)
                add_noise_static!(u, s, σ_probe, rng, ξ)
            end
        catch err
            if !(err isa MethodError); rethrow(); end
        end
        Σ_probe = MMatrix{N,N,T}(undef)
        try
            sigma_any(Σ_probe, u0, params, t0)     # sigma!(Σ,u,p,t)
            return (u, s, t, rng, ξ, tmp) -> begin
                sigma_any(Σ_probe, u, params, t)
                add_noise_static!(u, s, Σ_probe, rng, ξ, tmp)
            end
        catch err
            if !(err isa MethodError); rethrow(); end
        end
        # Returning form
        try
            sig0 = sigma_any(u0, params, t0)
            if sig0 isa Real
                return (u, s, t, rng, ξ, tmp) -> add_noise_static!(u, s, (sigma_any(u, params, t))::T, rng, ξ)
            elseif sig0 isa SVector{N,T}
                return (u, s, t, rng, ξ, tmp) -> add_noise_static!(u, s, (sigma_any(u, params, t))::SVector{N,T}, rng, ξ)
            elseif sig0 isa SMatrix{N,N,T}
                return (u, s, t, rng, ξ, tmp) -> add_noise_static!(u, s, (sigma_any(u, params, t))::SMatrix{N,N,T}, rng, ξ, tmp)
            end
        catch err
            if !(err isa MethodError); rethrow(); end
        end
    end
    # Fallback: no-params path
    return _make_noise_applier_static(sigma_any, u0, t0; sigma_inplace=sigma_inplace)
end

# --- Public API: static path (internal) ---

"""
    evolve_static(u0, dt, Nsteps, f!, sigma; ...)
"""
function evolve_static(u0, dt, Nsteps, f!, sigma;
                       params::Any = nothing,                         # <-- NEW
                       seed::Integer=123, resolution::Integer=1,
                       timestepper::Symbol=:rk4, boundary::Union{Nothing,Tuple}=nothing,
                       rng::Union{Nothing,AbstractRNG}=nothing, verbose::Bool=false,
                       sigma_inplace::Bool=true)
    local_rng = rng === nothing ? Random.MersenneTwister(seed) : rng
    return _evolve_static_typed(u0, dt, Nsteps, f!, sigma, local_rng;
                                params=params, resolution, timestepper, boundary, verbose, sigma_inplace)
end

function _evolve_static_typed(u0, dt, Nsteps, f!, sigma, rng::R;
                              params::Any = nothing,                         # <-- NEW
                              resolution::Integer=1, timestepper::Symbol=:rk4,
                              boundary::Union{Nothing,Tuple}=nothing, verbose::Bool=false,
                              sigma_inplace::Bool=true) where {R<:AbstractRNG}
    N = length(u0)
    T = promote_type(eltype(u0), typeof(dt))

    u   = MVector{N,T}(u0)
    k1  = MVector{N,T}(undef); k2 = MVector{N,T}(undef)
    k3  = MVector{N,T}(undef); k4 = MVector{N,T}(undef)
    tmp = MVector{N,T}(undef)
    ξ   = MVector{N,T}(undef)
    z   = MVector{N,T}(undef)  # correlated noise tmp

    s = sqrt(T(dt))
    t = zero(T)
    noise! = (params === nothing ?
                _make_noise_applier_static(sigma, u, t; sigma_inplace=sigma_inplace) :
                _make_noise_applier_static_with_params(sigma, u, t, params; sigma_inplace=sigma_inplace))

    Nsave = fld(Nsteps, resolution)
    results = Array{T}(undef, N, Nsave + 1)
    @inbounds @simd for i in 1:N
        results[i, 1] = u0[i]
    end
    save_idx = 1
    next_save = resolution

    if boundary === nothing
        if timestepper === :rk4
            @inbounds for step in 1:Nsteps
                rk4_step_static!(u, T(dt), f!, t, params, k1, k2, k3, k4, tmp)
                noise!(u, s, t, rng, ξ, z)
                t += dt
                if step == next_save
                    save_idx += 1
                    @inbounds @simd for i in 1:N
                        results[i, save_idx] = u[i]
                    end
                    next_save += resolution
                end
            end
        elseif timestepper === :rk2
            @inbounds for step in 1:Nsteps
                rk2_step_static!(u, T(dt), f!, t, params, k1, k2, tmp)
                noise!(u, s, t, rng, ξ, z)
                t += dt
                if step == next_save
                    save_idx += 1
                    @inbounds @simd for i in 1:N
                        results[i, save_idx] = u[i]
                    end
                    next_save += resolution
                end
            end
        else
            @inbounds for step in 1:Nsteps
                euler_step_static!(u, T(dt), f!, t, params, k1)
                noise!(u, s, t, rng, ξ, z)
                t += dt
                if step == next_save
                    save_idx += 1
                    @inbounds @simd for i in 1:N
                        results[i, save_idx] = u[i]
                    end
                    next_save += resolution
                end
            end
        end
    else
        lo, hi = boundary
        count = 0
        if timestepper === :rk4
            @inbounds for step in 1:Nsteps
                rk4_step_static!(u, T(dt), f!, t, params, k1, k2, k3, k4, tmp)
                noise!(u, s, t, rng, ξ, z)
                t += dt
                if @inbounds any(u[i] < lo || u[i] > hi for i in 1:N)
                    @inbounds @simd for i in 1:N
                        u[i] = u0[i]
                    end
                    count += 1
                end
                if step == next_save
                    save_idx += 1
                    @inbounds @simd for i in 1:N
                        results[i, save_idx] = u[i]
                    end
                    next_save += resolution
                end
            end
        elseif timestepper === :rk2
            @inbounds for step in 1:Nsteps
                rk2_step_static!(u, T(dt), f!, t, params, k1, k2, tmp)
                noise!(u, s, t, rng, ξ, z)
                t += dt
                if @inbounds any(u[i] < lo || u[i] > hi for i in 1:N)
                    @inbounds @simd for i in 1:N
                        u[i] = u0[i]
                    end
                    count += 1
                end
                if step == next_save
                    save_idx += 1
                    @inbounds @simd for i in 1:N
                        results[i, save_idx] = u[i]
                    end
                    next_save += resolution
                end
            end
        else
            @inbounds for step in 1:Nsteps
                euler_step_static!(u, T(dt), f!, t, params, k1)
                noise!(u, s, t, rng, ξ, z)
                t += dt
                if @inbounds any(u[i] < lo || u[i] > hi for i in 1:N)
                    @inbounds @simd for i in 1:N
                        u[i] = u0[i]
                    end
                    count += 1
                end
                if step == next_save
                    save_idx += 1
                    @inbounds @simd for i in 1:N
                        results[i, save_idx] = u[i]
                    end
                    next_save += resolution
                end
            end
        end
        if verbose
            println("Percentage of boundary crossings: ", count / Nsteps)
        end
    end
    return results
end

"""
    evolve_ens_static(u0, dt, Nsteps, f!, sigma; ...)
"""
function evolve_ens_static(u0, dt, Nsteps, f!, sigma;
                           params::Any = nothing,                         # <-- NEW
                           seed::Integer=123, resolution::Integer=1,
                           timestepper::Symbol=:rk4, boundary::Union{Nothing,Tuple}=nothing,
                           n_ens::Integer=1, rng::Union{Nothing,AbstractRNG}=nothing,
                           verbose::Bool=false, sigma_inplace::Bool=true)
    N = length(u0)
    T = promote_type(eltype(u0), typeof(dt))
    Nsave = fld(Nsteps, resolution)
    results = Array{T}(undef, N, Nsave + 1, n_ens)

    Threads.@threads :static for ens_idx in 1:n_ens   # <-- static scheduling
        # Faster per-thread RNG
        seed_val  = (rng === nothing) ? (seed + ens_idx * 1000) : rand(rng, UInt)
        local_rng = Random.TaskLocalRNG()
        Random.seed!(local_rng, seed_val)

        u   = MVector{N,T}(u0)
        @inbounds @simd for i in 1:N
            results[i, 1, ens_idx] = u0[i]
        end
        t = zero(T); s = sqrt(T(dt)); save_idx = 1; next_save = resolution
        k1 = MVector{N,T}(undef); k2 = MVector{N,T}(undef)
        k3 = MVector{N,T}(undef); k4 = MVector{N,T}(undef)
        tmp = MVector{N,T}(undef); ξ = MVector{N,T}(undef); z = MVector{N,T}(undef)
        noise! = (params === nothing ?
                    _make_noise_applier_static(sigma, u, t; sigma_inplace=sigma_inplace) :
                    _make_noise_applier_static_with_params(sigma, u, t, params; sigma_inplace=sigma_inplace))

        if boundary === nothing
            if timestepper === :rk4
                @inbounds for step in 1:Nsteps
                    rk4_step_static!(u, T(dt), f!, t, params, k1, k2, k3, k4, tmp)
                    noise!(u, s, t, local_rng, ξ, z)
                    t += dt
                    if step == next_save
                        save_idx += 1
                        @inbounds @simd for i in 1:N
                            results[i, save_idx, ens_idx] = u[i]
                        end
                        next_save += resolution
                    end
                end
            else
                @inbounds for step in 1:Nsteps
                    euler_step_static!(u, T(dt), f!, t, params, k1)
                    noise!(u, s, t, local_rng, ξ, z)
                    t += dt
                    if step == next_save
                        save_idx += 1
                        @inbounds @simd for i in 1:N
                            results[i, save_idx, ens_idx] = u[i]
                        end
                        next_save += resolution
                    end
                end
            end
        else
            lo, hi = boundary
            count = 0
            if timestepper === :rk4
                @inbounds for step in 1:Nsteps
                    rk4_step_static!(u, T(dt), f!, t, params, k1, k2, k3, k4, tmp)
                    noise!(u, s, t, local_rng, ξ, z)
                    t += dt
                    if @inbounds any(u[i] < lo || u[i] > hi for i in 1:N)
                        @inbounds @simd for i in 1:N
                            u[i] = u0[i]
                        end
                        count += 1
                    end
                    if step == next_save
                        save_idx += 1
                        @inbounds @simd for i in 1:N
                            results[i, save_idx, ens_idx] = u[i]
                        end
                        next_save += resolution
                    end
                end
            else
                @inbounds for step in 1:Nsteps
                    euler_step_static!(u, T(dt), f!, t, params, k1)
                    noise!(u, s, t, local_rng, ξ, z)
                    t += dt
                    if @inbounds any(u[i] < lo || u[i] > hi for i in 1:N)
                        @inbounds @simd for i in 1:N
                            u[i] = u0[i]
                        end
                        count += 1
                    end
                    if step == next_save
                        save_idx += 1
                        @inbounds @simd for i in 1:N
                            results[i, save_idx, ens_idx] = u[i]
                        end
                        next_save += resolution
                    end
                end
            end
            if verbose && ens_idx == 1
                println("Percentage of boundary crossings: ", count / Nsteps)
            end
        end
    end
    return results
end
