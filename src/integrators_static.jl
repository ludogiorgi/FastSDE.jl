########################
# Static-path integrators (StaticArrays)
########################

# Threshold is managed in hybrid.jl; here we only implement the static kernels.

# --- Time steppers (in-place, static) ---

"""
    rk4_step_static!(u, dt, f!, t, k1, k2, k3, k4, tmp)

Fourth-order Runge–Kutta step applied in-place to `u` for the in-place drift
`f!(du, u, t)`. Scratch buffers `k1..k4` and `tmp` must have the same length
as `u`.
"""
function rk4_step_static!(
    u::MVector{N,T},
    dt::T,
    f!,
    t::T,
    k1::MVector{N,T},
    k2::MVector{N,T},
    k3::MVector{N,T},
    k4::MVector{N,T},
    tmp::MVector{N,T},
) where {N,T}
    f!(k1, u, t)
    @inbounds for i in 1:N
        tmp[i] = u[i] + 0.5 * dt * k1[i]
    end
    f!(k2, tmp, t + 0.5 * dt)
    @inbounds for i in 1:N
        tmp[i] = u[i] + 0.5 * dt * k2[i]
    end
    f!(k3, tmp, t + 0.5 * dt)
    @inbounds for i in 1:N
        tmp[i] = u[i] + dt * k3[i]
    end
    f!(k4, tmp, t + dt)
    @inbounds for i in 1:N
        u[i] += (dt / 6) * (k1[i] + 2k2[i] + 2k3[i] + k4[i])
    end
    return nothing
end

"""
    euler_step_static!(u, dt, f!, t, k1)

Forward–Euler step applied in-place to `u` for the in-place drift `f!(du, u, t)`.
Scratch buffer `k1` must have the same length as `u`.
"""
function euler_step_static!(
    u::MVector{N,T},
    dt::T,
    f!,
    t::T,
    k1::MVector{N,T},
) where {N,T}
    f!(k1, u, t)
    @inbounds for i in 1:N
        u[i] += dt * k1[i]
    end
    return nothing
end

# --- Noise adders (static) ---

"""
    add_noise_static!(u, s, σ, rng, ξ)

Add isotropic Gaussian noise in-place to `u` with scalar `σ` using preallocated
buffer `ξ` for standard normal draws.
"""
@inline function add_noise_static!(
    u::MVector{N,T},
    s::T,
    σ::T,
    rng::AbstractRNG,
    ξ::MVector{N,T},
) where {N,T}
    @inbounds for i in 1:N
        ξ[i] = randn(rng)
    end
    @inbounds for i in 1:N
        u[i] += s * σ * ξ[i]
    end
    return nothing
end

"""
    add_noise_static!(u, s, σ_vec, rng, ξ)

Add diagonal (per-component) Gaussian noise in-place to `u` with `σ_vec`.
"""
@inline function add_noise_static!(
    u::MVector{N,T},
    s::T,
    σ::SVector{N,T},
    rng::AbstractRNG,
    ξ::MVector{N,T},
) where {N,T}
    @inbounds for i in 1:N
        ξ[i] = randn(rng)
    end
    @inbounds for i in 1:N
        u[i] += s * (σ[i] * ξ[i])
    end
    return nothing
end

"""
Add diagonal (per-component) Gaussian noise with in-place sigma! using MVector.
"""
@inline function add_noise_static!(
    u::MVector{N,T},
    s::T,
    σ::MVector{N,T},
    rng::AbstractRNG,
    ξ::MVector{N,T},
) where {N,T}
    @inbounds for i in 1:N
        ξ[i] = randn(rng)
    end
    @inbounds for i in 1:N
        u[i] += s * (σ[i] * ξ[i])
    end
    return nothing
end

"""
    add_noise_static!(u, s, Σ, rng, ξ, tmp)

Add correlated Gaussian noise in-place to `u` using covariance-factor `Σ`.
Temporary buffer `tmp` holds the correlated noise `Σ * ξ`.
"""
@inline function add_noise_static!(
    u::MVector{N,T},
    s::T,
    Σ::SMatrix{N,N,T},
    rng::AbstractRNG,
    ξ::MVector{N,T},
    tmp::MVector{N,T},
) where {N,T}
    @inbounds for i in 1:N
        ξ[i] = randn(rng)
    end
    tmp .= Σ * ξ
    @inbounds for i in 1:N
        u[i] += s * tmp[i]
    end
    return nothing
end

"""
Add correlated Gaussian noise with in-place sigma! using MMatrix.
"""
@inline function add_noise_static!(
    u::MVector{N,T},
    s::T,
    Σ::MMatrix{N,N,T},
    rng::AbstractRNG,
    ξ::MVector{N,T},
    tmp::MVector{N,T},
) where {N,T}
    @inbounds for i in 1:N
        ξ[i] = randn(rng)
    end
    tmp .= Σ * ξ
    @inbounds for i in 1:N
        u[i] += s * tmp[i]
    end
    return nothing
end

# Lift constants to callable sigma(u,t) once (static path)
_make_sigma_static(sigma) = sigma
_make_sigma_static(sigma::Real) = (u, t) -> sigma
_make_sigma_static(sigma::SVector) = (u, t) -> sigma
_make_sigma_static(sigma::SMatrix) = (u, t) -> sigma
_make_sigma_static(sigma::AbstractVector) = (u, t) -> sigma
_make_sigma_static(sigma::AbstractMatrix) = (u, t) -> sigma

# Build monomorphic noise applier for static forms (Real, SVector, SMatrix)
function _make_noise_applier_static(sigma_any, u0::MVector{N,T}, t0::T; sigma_inplace::Bool=false) where {N,T}
    # In-place sigma! detection for vector/matrix forms (robust try-call). Always attempt once.
    if sigma_any isa Function
        σ_probe = MVector{N,T}(undef)
        try
            sigma_any(σ_probe, u0, t0)
            return (u, s, t, rng, ξ, tmp) -> begin
                sigma_any(σ_probe, u, t)
                add_noise_static!(u, s, σ_probe, rng, ξ)
            end
        catch err
            if !(err isa MethodError)
                rethrow()
            end
        end
        Σ_probe = MMatrix{N,N,T}(undef)
        try
            sigma_any(Σ_probe, u0, t0)
            return (u, s, t, rng, ξ, tmp) -> begin
                sigma_any(Σ_probe, u, t)
                add_noise_static!(u, s, Σ_probe, rng, ξ, tmp)
            end
        catch err
            if !(err isa MethodError)
                rethrow()
            end
        end
    end

    # Fast-path: constant Vector/Matrix provided directly
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
        # Function returning a Vector: convert at each call
        return (u, s, t, rng, ξ, tmp) -> begin
            σ_vec = SVector{N,T}(sigma(u, t))
            add_noise_static!(u, s, σ_vec, rng, ξ)
        end
    elseif sig0 isa AbstractMatrix
        # Function returning a Matrix: convert at each call
        return (u, s, t, rng, ξ, tmp) -> begin
            Σ_mat = SMatrix{N,N,T}(sigma(u, t))
            add_noise_static!(u, s, Σ_mat, rng, ξ, tmp)
        end
    else
        error("Unsupported σ type for static path. Expected Real, (S)Vector, or (S)Matrix; got $(typeof(sig0))")
    end
end

# --- Public API: static path (internal) ---

"""
    evolve_static(u0, dt, Nsteps, f!, sigma; resolution=1, timestepper=:rk4, boundary=nothing, seed=123)

StaticArrays path for small fixed N. Returns (N, Nsave+1).
"""
function evolve_static(
    u0,
    dt,
    Nsteps,
    f!,
    sigma;
    seed::Integer = 123,
    resolution::Integer = 1,
    timestepper::Symbol = :rk4,
    boundary::Union{Nothing,Tuple} = nothing,
    rng::Union{Nothing,AbstractRNG} = nothing,
    verbose::Bool = false,
    sigma_inplace::Bool=false,
)
    local_rng = rng === nothing ? Random.MersenneTwister(seed) : rng
    return _evolve_static_typed(u0, dt, Nsteps, f!, sigma, local_rng;
                                resolution=resolution, timestepper=timestepper,
                                boundary=boundary, verbose=verbose, sigma_inplace=sigma_inplace)
end

function _evolve_static_typed(
    u0,
    dt,
    Nsteps,
    f!,
    sigma,
    rng::R;
    resolution::Integer = 1,
    timestepper::Symbol = :rk4,
    boundary::Union{Nothing,Tuple} = nothing,
    verbose::Bool = false,
    sigma_inplace::Bool=false,
) where {R<:AbstractRNG}
    N = length(u0)
    T = promote_type(eltype(u0), typeof(dt))

    # Static state and buffers
    u = MVector{N,T}(u0)
    k1 = MVector{N,T}(undef)
    k2 = MVector{N,T}(undef)
    k3 = MVector{N,T}(undef)
    k4 = MVector{N,T}(undef)
    tmp = MVector{N,T}(undef)
    ξ = MVector{N,T}(undef)    # standard-normal draws
    z = MVector{N,T}(undef)    # correlated noise temporary

    ts = (timestepper === :rk4 ? :rk4 : :euler)
    s = sqrt(T(dt))
    t = zero(T)
    noise! = _make_noise_applier_static(sigma, u, t; sigma_inplace=sigma_inplace)

    Nsave = fld(Nsteps, resolution)
    results = Array{T}(undef, N, Nsave + 1)
    @inbounds for i in 1:N
        results[i, 1] = u0[i]
    end
    save_idx = 1

    if boundary === nothing
        @inbounds for step in 1:Nsteps
            if ts === :rk4
                rk4_step_static!(u, T(dt), f!, t, k1, k2, k3, k4, tmp)
            else
                euler_step_static!(u, T(dt), f!, t, k1)
            end
            noise!(u, s, t, rng, ξ, z)
            t += dt
            if step % resolution == 0
                save_idx += 1
                @inbounds for i in 1:N
                    results[i, save_idx] = u[i]
                end
            end
        end
    else
        lo, hi = boundary
        count = 0
        @inbounds for step in 1:Nsteps
            if ts === :rk4
                rk4_step_static!(u, T(dt), f!, t, k1, k2, k3, k4, tmp)
            else
                euler_step_static!(u, T(dt), f!, t, k1)
            end
            noise!(u, s, t, rng, ξ, z)
            t += dt
            if @inbounds any(u[i] < lo || u[i] > hi for i in 1:N)
                @inbounds for i in 1:N
                    u[i] = u0[i]
                end
                count += 1
            end
            if step % resolution == 0
                save_idx += 1
                @inbounds for i in 1:N
                    results[i, save_idx] = u[i]
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
    evolve_ens_static(u0, dt, Nsteps, f!, sigma; resolution=1, timestepper=:rk4, boundary=nothing, seed=123, n_ens=1)

StaticArrays path ensemble. Returns (N, Nsave+1, n_ens).
"""
function evolve_ens_static(
    u0,
    dt,
    Nsteps,
    f!,
    sigma;
    seed::Integer = 123,
    resolution::Integer = 1,
    timestepper::Symbol = :rk4,
    boundary::Union{Nothing,Tuple} = nothing,
    n_ens::Integer = 1,
    rng::Union{Nothing,AbstractRNG} = nothing,
    verbose::Bool = false,
    sigma_inplace::Bool=false,
)
    N = length(u0)
    T = promote_type(eltype(u0), typeof(dt))
    Nsave = fld(Nsteps, resolution)
    results = Array{T}(undef, N, Nsave + 1, n_ens)

    Threads.@threads :dynamic for ens_idx in 1:n_ens
        local_rng = rng === nothing ? Random.MersenneTwister(seed + ens_idx * 1000) : Random.MersenneTwister(rand(Random.default_rng(), UInt))
        u = MVector{N,T}(u0)
        @inbounds for i in 1:N
            results[i, 1, ens_idx] = u0[i]
        end
        ts = (timestepper === :rk4 ? :rk4 : :euler)
        t = zero(T)
        save_idx = 1
        s = sqrt(T(dt))
        # buffers
        k1 = MVector{N,T}(undef)
        k2 = MVector{N,T}(undef)
        k3 = MVector{N,T}(undef)
        k4 = MVector{N,T}(undef)
        tmp = MVector{N,T}(undef)
        ξ = MVector{N,T}(undef)
        z = MVector{N,T}(undef)
        noise! = _make_noise_applier_static(sigma, u, t; sigma_inplace=sigma_inplace)

        if boundary === nothing
            @inbounds for step in 1:Nsteps
                if ts === :rk4
                    rk4_step_static!(u, T(dt), f!, t, k1, k2, k3, k4, tmp)
                else
                    euler_step_static!(u, T(dt), f!, t, k1)
                end
                noise!(u, s, t, local_rng, ξ, z)
                t += dt
                if step % resolution == 0
                    save_idx += 1
                    @inbounds for i in 1:N
                        results[i, save_idx, ens_idx] = u[i]
                    end
                end
            end
        else
            lo, hi = boundary
            count = 0
            @inbounds for step in 1:Nsteps
                if ts === :rk4
                    rk4_step_static!(u, T(dt), f!, t, k1, k2, k3, k4, tmp)
                else
                    euler_step_static!(u, T(dt), f!, t, k1)
                end
                noise!(u, s, t, local_rng, ξ, z)
                t += dt
                if @inbounds any(u[i] < lo || u[i] > hi for i in 1:N)
                    @inbounds for i in 1:N
                        u[i] = u0[i]
                    end
                    count += 1
                end
                if step % resolution == 0
                    save_idx += 1
                    @inbounds for i in 1:N
                        results[i, save_idx, ens_idx] = u[i]
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
