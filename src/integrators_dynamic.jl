########################
# Dynamic-path integrators (optimized) — updated
########################

using Random
using LinearAlgebra: mul!, BLAS

# --- Call shims (params or no params) ---

# --- Utilities ---

@inline function crossed(u, lo, hi)
    @inbounds for x in u
        if x < lo || x > hi
            return true
        end
    end
    return false
end

@inline _resolve_stepper_dyn(sym) = sym === :rk4 ? rk4_step! :
                                    sym === :rk2 ? rk2_step! :
                                    sym === :euler ? euler_step! :
                                    error("Invalid timestepper: $sym")

# Lift constants to callable sigma(u,t) once
@inline _make_sigma_dyn(sigma) = sigma
@inline _make_sigma_dyn(sigma::Real) = (u, t) -> sigma
@inline _make_sigma_dyn(sigma::AbstractVector) = (u, t) -> sigma
@inline _make_sigma_dyn(sigma::AbstractMatrix) = (u, t) -> sigma

# --- Time steppers (in-place, dynamic) ---

"""
    rk4_step!(u, dt, f!, t, params, k1, k2, k3, k4, tmp)
"""
function rk4_step!(u, dt, f!, t, params, k1, k2, k3, k4, tmp)
    _call_f!(f!, k1, u, t, params)
    if length(u) >= 256
        @inbounds copyto!(tmp, u); BLAS.axpy!(0.5 * dt, k1, tmp)
    else
        @inbounds @simd for i in eachindex(u, tmp, k1)
            tmp[i] = u[i] + 0.5 * dt * k1[i]
        end
    end
    _call_f!(f!, k2, tmp, t + 0.5 * dt, params)

    if length(u) >= 256
        @inbounds copyto!(tmp, u); BLAS.axpy!(0.5 * dt, k2, tmp)
    else
        @inbounds @simd for i in eachindex(u, tmp, k2)
            tmp[i] = u[i] + 0.5 * dt * k2[i]
        end
    end
    _call_f!(f!, k3, tmp, t + 0.5 * dt, params)

    if length(u) >= 256
        @inbounds copyto!(tmp, u); BLAS.axpy!(dt, k3, tmp)
    else
        @inbounds @simd for i in eachindex(u, tmp, k3)
            tmp[i] = u[i] + dt * k3[i]
        end
    end
    _call_f!(f!, k4, tmp, t + dt, params)

    if length(u) >= 256
        BLAS.axpy!(dt/6, k1, u)
        BLAS.axpy!(dt/3, k2, u)
        BLAS.axpy!(dt/3, k3, u)
        BLAS.axpy!(dt/6, k4, u)
    else
        @inbounds @simd for i in eachindex(u, k1, k2, k3, k4)
            u[i] += (dt / 6) * (k1[i] + 2k2[i] + 2k3[i] + k4[i])
        end
    end
    return nothing
end

"""
    rk2_step!(u, dt, f!, t, params, k1, k2, tmp)

Second-order Runge–Kutta (midpoint) deterministic step.
"""
function rk2_step!(u, dt, f!, t, params, k1, k2, tmp)
    _call_f!(f!, k1, u, t, params)
    if length(u) >= 256
        @inbounds copyto!(tmp, u); BLAS.axpy!(0.5 * dt, k1, tmp)
    else
        @inbounds @simd for i in eachindex(u, tmp, k1)
            tmp[i] = u[i] + 0.5 * dt * k1[i]
        end
    end
    _call_f!(f!, k2, tmp, t + 0.5 * dt, params)
    if length(u) >= 256
        BLAS.axpy!(dt, k2, u)
    else
        @inbounds @simd for i in eachindex(u, k2)
            u[i] += dt * k2[i]
        end
    end
    return nothing
end

"""
    euler_step!(u, dt, f!, t, params, k1)
"""
function euler_step!(u, dt, f!, t, params, k1)
    _call_f!(f!, k1, u, t, params)
    if length(u) >= 256
        BLAS.axpy!(dt, k1, u)
    else
        @inbounds @simd for i in eachindex(u, k1)
            u[i] += dt * k1[i]
        end
    end
    return nothing
end

# --- Noise adders (dynamic) ---

@inline function add_noise!(u::AbstractVector{<:Real}, s, σ::Real,
                            rng::AbstractRNG, noise_buf::AbstractVector{<:AbstractFloat})
    randn!(rng, noise_buf)
    @inbounds @simd for i in eachindex(u, noise_buf)
        u[i] += s * σ * noise_buf[i]
    end
    return nothing
end

@inline function add_noise!(u::AbstractVector{<:Real}, s, σ::AbstractVector,
                            rng::AbstractRNG, noise_buf::AbstractVector{<:AbstractFloat})
    randn!(rng, noise_buf)
    @inbounds @simd for i in eachindex(u, σ, noise_buf)
        u[i] += s * (σ[i] * noise_buf[i])
    end
    return nothing
end

@inline function add_noise!(u::AbstractVector{<:Real}, s, σ::AbstractMatrix,
                            rng::AbstractRNG, noise_buf::AbstractVector{<:AbstractFloat},
                            tmp_vec::AbstractVector{<:Real})
    randn!(rng, noise_buf)
    mul!(tmp_vec, σ, noise_buf)   # correlated noise (no temporary)
    @inbounds @simd for i in eachindex(u, tmp_vec)
        u[i] += s * tmp_vec[i]
    end
    return nothing
end

# --- Noise applier builders ---

"""
No-params builder (existing behavior).
"""
function _make_noise_applier_dyn(sigma_any, u0, t0; sigma_inplace::Bool=true)
    if sigma_inplace && sigma_any isa Function
        T = typeof(t0); dim = length(u0)
        σ_vec = Vector{T}(undef, dim)
        try
            sigma_any(σ_vec, u0, t0)  # sigma!(σ_vec,u,t)
            return (u, s, t, rng, noise_buf, tmp_vec) -> begin
                sigma_any(σ_vec, u, t)
                add_noise!(u, s, σ_vec, rng, noise_buf)
            end
        catch err
            if !(err isa MethodError); rethrow(); end
        end
        Σ_mat = Matrix{T}(undef, dim, dim)
        try
            sigma_any(Σ_mat, u0, t0)  # sigma!(Σ_mat,u,t)
            return (u, s, t, rng, noise_buf, tmp_vec) -> begin
                sigma_any(Σ_mat, u, t)
                add_noise!(u, s, Σ_mat, rng, noise_buf, tmp_vec)
            end
        catch err
            if !(err isa MethodError); rethrow(); end
        end
    end

    sigma = _make_sigma_dyn(sigma_any)
    sig0 = sigma(u0, t0)
    if sig0 isa Real
        return (u, s, t, rng, noise_buf, tmp_vec) -> begin
            σ = sigma(u, t)::Real
            iszero(σ) && return
            add_noise!(u, s, σ, rng, noise_buf)
        end
    elseif sig0 isa AbstractVector
        return (u, s, t, rng, noise_buf, tmp_vec) -> begin
            σ = sigma(u, t)::AbstractVector
            add_noise!(u, s, σ, rng, noise_buf)
        end
    elseif sig0 isa AbstractMatrix
        return (u, s, t, rng, noise_buf, tmp_vec) -> begin
            σ = sigma(u, t)::AbstractMatrix
            add_noise!(u, s, σ, rng, noise_buf, tmp_vec)
        end
    else
        error("Unsupported σ type: $(typeof(sig0))")
    end
end

"""
With-params builder (preferred when params ≠ nothing).
Supports sigma!(out,u,p,t) and sigma(u,p,t).
"""
function _make_noise_applier_dyn_with_params(sigma_any, u0, t0, params; sigma_inplace::Bool=true)
    if sigma_any isa Function
        T = typeof(t0); dim = length(u0)

        # Try in-place vector
        σ_vec = Vector{T}(undef, dim)
        try
            sigma_any(σ_vec, u0, params, t0)  # sigma!(σ_vec,u,p,t)
            return (u, s, t, rng, noise_buf, tmp_vec) -> begin
                sigma_any(σ_vec, u, params, t)
                add_noise!(u, s, σ_vec, rng, noise_buf)
            end
        catch err
            if !(err isa MethodError); rethrow(); end
        end

        # Try in-place matrix
        Σ_mat = Matrix{T}(undef, dim, dim)
        try
            sigma_any(Σ_mat, u0, params, t0)  # sigma!(Σ,u,p,t)
            return (u, s, t, rng, noise_buf, tmp_vec) -> begin
                sigma_any(Σ_mat, u, params, t)
                add_noise!(u, s, Σ_mat, rng, noise_buf, tmp_vec)
            end
        catch err
            if !(err isa MethodError); rethrow(); end
        end

        # Try returning form
        try
            sig0 = sigma_any(u0, params, t0)
            if sig0 isa Real
                return (u, s, t, rng, noise_buf, tmp_vec) -> begin
                    σ = sigma_any(u, params, t)::Real
                    iszero(σ) && return
                    add_noise!(u, s, σ, rng, noise_buf)
                end
            elseif sig0 isa AbstractVector
                return (u, s, t, rng, noise_buf, tmp_vec) -> begin
                    σ = sigma_any(u, params, t)::AbstractVector
                    add_noise!(u, s, σ, rng, noise_buf)
                end
            elseif sig0 isa AbstractMatrix
                return (u, s, t, rng, noise_buf, tmp_vec) -> begin
                    σ = sigma_any(u, params, t)::AbstractMatrix
                    add_noise!(u, s, σ, rng, noise_buf, tmp_vec)
                end
            end
        catch err
            if !(err isa MethodError); rethrow(); end
        end
    end

    # Fallback: use no-params path
    return _make_noise_applier_dyn(sigma_any, u0, t0; sigma_inplace=sigma_inplace)
end

# --- Public API: dynamic path (internal) ---

"""
    evolve_dyn(u0, dt, Nsteps, f!, sigma; ...)
"""
function evolve_dyn(u0, dt, Nsteps, f!, sigma;
                    params::Any = nothing,                          # <-- NEW
                    seed::Integer=123, resolution::Integer=1,
                    timestepper::Symbol=:rk4, boundary::Union{Nothing,Tuple}=nothing,
                    rng::Union{Nothing,AbstractRNG}=nothing, verbose::Bool=false,
                    sigma_inplace::Bool=true)
    local_rng = rng === nothing ? Random.MersenneTwister(seed) : rng
    return _evolve_dyn_typed(u0, dt, Nsteps, f!, sigma, local_rng;
                             params=params, resolution, timestepper, boundary, verbose, sigma_inplace)
end

function _evolve_dyn_typed(u0, dt, Nsteps, f!, sigma, rng::R;
                           params::Any = nothing,                          # <-- NEW
                           resolution::Integer=1, timestepper::Symbol=:rk4,
                           boundary::Union{Nothing,Tuple}=nothing, verbose::Bool=false,
                           sigma_inplace::Bool=true) where {R<:AbstractRNG}

    dim   = length(u0)
    T     = promote_type(eltype(u0), typeof(dt))
    Nsave = fld(Nsteps, resolution)

    u = copy(u0)
    results = Array{T}(undef, dim, Nsave+1)
    @views results[:, 1] .= u0

    ts  = _resolve_stepper_dyn(timestepper)
    t = zero(T); save_index = 1; next_save = resolution
    noise_buf = Vector{T}(undef, dim)
    tmp_vec   = Vector{T}(undef, dim)
    k1 = similar(u, T); k2 = similar(u, T); k3 = similar(u, T); k4 = similar(u, T); tmp = similar(u, T)
    s = sqrt(T(dt))
    noise! = (params === nothing ?
                _make_noise_applier_dyn(sigma, u, t; sigma_inplace=sigma_inplace) :
                _make_noise_applier_dyn_with_params(sigma, u, t, params; sigma_inplace=sigma_inplace))

    if boundary === nothing
        if ts === rk4_step!
            @inbounds for step in 1:Nsteps
                rk4_step!(u, dt, f!, t, params, k1, k2, k3, k4, tmp)
                noise!(u, s, t, rng, noise_buf, tmp_vec)
                t += dt
                if step == next_save
                    save_index += 1
                    @inbounds @simd for i in 1:dim
                        results[i, save_index] = u[i]
                    end
                    next_save += resolution
                end
            end
        elseif ts === rk2_step!
            @inbounds for step in 1:Nsteps
                rk2_step!(u, dt, f!, t, params, k1, k2, tmp)
                noise!(u, s, t, rng, noise_buf, tmp_vec)
                t += dt
                if step == next_save
                    save_index += 1
                    @inbounds @simd for i in 1:dim
                        results[i, save_index] = u[i]
                    end
                    next_save += resolution
                end
            end
        else
            @inbounds for step in 1:Nsteps
                euler_step!(u, dt, f!, t, params, k1)
                noise!(u, s, t, rng, noise_buf, tmp_vec)
                t += dt
                if step == next_save
                    save_index += 1
                    @inbounds @simd for i in 1:dim
                        results[i, save_index] = u[i]
                    end
                    next_save += resolution
                end
            end
        end
    else
        lo, hi = boundary
        count = 0
        if ts === rk4_step!
            @inbounds for step in 1:Nsteps
                rk4_step!(u, dt, f!, t, params, k1, k2, k3, k4, tmp)
                noise!(u, s, t, rng, noise_buf, tmp_vec)
                t += dt
                if crossed(u, lo, hi)
                    @inbounds @simd for i in 1:dim
                        u[i] = u0[i]
                    end
                    count += 1
                end
                if step == next_save
                    save_index += 1
                    @inbounds @simd for i in 1:dim
                        results[i, save_index] = u[i]
                    end
                    next_save += resolution
                end
            end
        elseif ts === rk2_step!
            @inbounds for step in 1:Nsteps
                rk2_step!(u, dt, f!, t, params, k1, k2, tmp)
                noise!(u, s, t, rng, noise_buf, tmp_vec)
                t += dt
                if crossed(u, lo, hi)
                    @inbounds @simd for i in 1:dim
                        u[i] = u0[i]
                    end
                    count += 1
                end
                if step == next_save
                    save_index += 1
                    @inbounds @simd for i in 1:dim
                        results[i, save_index] = u[i]
                    end
                    next_save += resolution
                end
            end
        else
            @inbounds for step in 1:Nsteps
                euler_step!(u, dt, f!, t, params, k1)
                noise!(u, s, t, rng, noise_buf, tmp_vec)
                t += dt
                if crossed(u, lo, hi)
                    @inbounds @simd for i in 1:dim
                        u[i] = u0[i]
                    end
                    count += 1
                end
                if step == next_save
                    save_index += 1
                    @inbounds @simd for i in 1:dim
                        results[i, save_index] = u[i]
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
    evolve_ens_dyn(u0, dt, Nsteps, f!, sigma; ...)
"""
function evolve_ens_dyn(u0, dt, Nsteps, f!, sigma;
                        params::Any = nothing,                          # <-- NEW
                        seed::Integer=123, resolution::Integer=1,
                        timestepper::Symbol=:rk4, boundary::Union{Nothing,Tuple}=nothing,
                        n_ens::Integer=1, rng::Union{Nothing,AbstractRNG}=nothing,
                        verbose::Bool=false, sigma_inplace::Bool=true)

    dim   = length(u0)
    T     = promote_type(eltype(u0), typeof(dt))
    Nsave = fld(Nsteps, resolution)
    results = Array{T}(undef, dim, Nsave+1, n_ens)

    Threads.@threads :static for ens_idx in 1:n_ens   # <-- static scheduling
        # Faster per-thread RNG
        seed_val  = (rng === nothing) ? (seed + ens_idx * 1000) : rand(rng, UInt)
        local_rng = Random.TaskLocalRNG()
        Random.seed!(local_rng, seed_val)

        u = copy(u0)
        @views results[:, 1, ens_idx] .= u0
        ts = _resolve_stepper_dyn(timestepper)
        t = zero(T); save_index = 1; next_save = resolution
        noise_buf = Vector{T}(undef, dim)
        tmp_vec   = Vector{T}(undef, dim)
        k1 = similar(u, T); k2 = similar(u, T); k3 = similar(u, T); k4 = similar(u, T); tmp = similar(u, T)
        s = sqrt(T(dt))
        noise! = (params === nothing ?
                    _make_noise_applier_dyn(sigma, u, t; sigma_inplace=sigma_inplace) :
                    _make_noise_applier_dyn_with_params(sigma, u, t, params; sigma_inplace=sigma_inplace))

        if boundary === nothing
            if ts === rk4_step!
                @inbounds for step in 1:Nsteps
                    rk4_step!(u, dt, f!, t, params, k1, k2, k3, k4, tmp)
                    noise!(u, s, t, local_rng, noise_buf, tmp_vec)
                    t += dt
                    if step == next_save
                        save_index += 1
                        @inbounds @simd for i in 1:dim
                            results[i, save_index, ens_idx] = u[i]
                        end
                        next_save += resolution
                    end
                end
            elseif ts === rk2_step!
                @inbounds for step in 1:Nsteps
                    rk2_step!(u, dt, f!, t, params, k1, k2, tmp)
                    noise!(u, s, t, local_rng, noise_buf, tmp_vec)
                    t += dt
                    if step == next_save
                        save_index += 1
                        @inbounds @simd for i in 1:dim
                            results[i, save_index, ens_idx] = u[i]
                        end
                        next_save += resolution
                    end
                end
            else
                @inbounds for step in 1:Nsteps
                    euler_step!(u, dt, f!, t, params, k1)
                    noise!(u, s, t, local_rng, noise_buf, tmp_vec)
                    t += dt
                    if step == next_save
                        save_index += 1
                        @inbounds @simd for i in 1:dim
                            results[i, save_index, ens_idx] = u[i]
                        end
                        next_save += resolution
                    end
                end
            end
        else
            lo, hi = boundary
            count = 0
            if ts === rk4_step!
                @inbounds for step in 1:Nsteps
                    rk4_step!(u, dt, f!, t, params, k1, k2, k3, k4, tmp)
                    noise!(u, s, t, local_rng, noise_buf, tmp_vec)
                    t += dt
                    if crossed(u, lo, hi)
                        @inbounds @simd for i in 1:dim
                            u[i] = u0[i]
                        end
                        count += 1
                    end
                    if step == next_save
                        save_index += 1
                        @inbounds @simd for i in 1:dim
                            results[i, save_index, ens_idx] = u[i]
                        end
                        next_save += resolution
                    end
                end
            elseif ts === rk2_step!
                @inbounds for step in 1:Nsteps
                    rk2_step!(u, dt, f!, t, params, k1, k2, tmp)
                    noise!(u, s, t, local_rng, noise_buf, tmp_vec)
                    t += dt
                    if crossed(u, lo, hi)
                        @inbounds @simd for i in 1:dim
                            u[i] = u0[i]
                        end
                        count += 1
                    end
                    if step == next_save
                        save_index += 1
                        @inbounds @simd for i in 1:dim
                            results[i, save_index, ens_idx] = u[i]
                        end
                        next_save += resolution
                    end
                end
            else
                @inbounds for step in 1:Nsteps
                    euler_step!(u, dt, f!, t, params, k1)
                    noise!(u, s, t, local_rng, noise_buf, tmp_vec)
                    t += dt
                    if crossed(u, lo, hi)
                        @inbounds @simd for i in 1:dim
                            u[i] = u0[i]
                        end
                        count += 1
                    end
                    if step == next_save
                        save_index += 1
                        @inbounds @simd for i in 1:dim
                            results[i, save_index, ens_idx] = u[i]
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
