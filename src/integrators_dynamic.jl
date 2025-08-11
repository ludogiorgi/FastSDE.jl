########################
# Dynamic-path integrators (optimized)
########################

# --- Utilities ---

@inline function crossed(u, lo, hi)
    @inbounds for x in u
        if x < lo || x > hi
            return true
        end
    end
    return false
end

@inline function _resolve_stepper_dyn(sym)
    sym === :rk4   && return rk4_step!
    sym === :euler && return euler_step!
    error("Invalid timestepper specified. Use :rk4 or :euler.")
end

# Lift constants to callable sigma(u,t) once
_make_sigma_dyn(sigma) = sigma
_make_sigma_dyn(sigma::Real) = (u, t) -> sigma
_make_sigma_dyn(sigma::AbstractVector) = (u, t) -> sigma
_make_sigma_dyn(sigma::AbstractMatrix) = (u, t) -> sigma

"""
Build a monomorphic noise-applier closure specialized to the form of `sigma`.
"""
function _make_noise_applier_dyn(sigma_any, u0, t0)
    sigma = _make_sigma_dyn(sigma_any)
    sig0 = sigma(u0, t0)
    if iszero(sig0)
        return (u, s, t, rng, noise_buf, tmp_vec) -> nothing
    elseif sig0 isa Real
        return (u, s, t, rng, noise_buf, tmp_vec) -> begin
            σ = sigma(u, t)::Real
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
        error("Unsupported sigma type: $(typeof(sig0))")
    end
end

# --- Time steppers (in-place) ---

"""
    rk4_step!(u, dt, f!, t, k1, k2, k3, k4, tmp)

Fourth-order Runge–Kutta step applied in-place to `u` for the in-place drift
`f!(du, u, t)`. Scratch buffers `k1..k4` and `tmp` must have the same length
as `u`.
"""
function rk4_step!(u, dt, f!, t, k1, k2, k3, k4, tmp)
    f!(k1, u, t)
    if length(u) >= 256
        @inbounds copyto!(tmp, u)
        LinearAlgebra.BLAS.axpy!(0.5 * dt, k1, tmp)
    else
        @inbounds @simd for i in eachindex(u, tmp, k1)
            tmp[i] = u[i] + 0.5 * dt * k1[i]
        end
    end
    f!(k2, tmp, t + 0.5 * dt)
    if length(u) >= 256
        @inbounds copyto!(tmp, u)
        LinearAlgebra.BLAS.axpy!(0.5 * dt, k2, tmp)
    else
        @inbounds @simd for i in eachindex(u, tmp, k2)
            tmp[i] = u[i] + 0.5 * dt * k2[i]
        end
    end
    f!(k3, tmp, t + 0.5 * dt)
    if length(u) >= 256
        @inbounds copyto!(tmp, u)
        LinearAlgebra.BLAS.axpy!(dt, k3, tmp)
    else
        @inbounds @simd for i in eachindex(u, tmp, k3)
            tmp[i] = u[i] + dt * k3[i]
        end
    end
    f!(k4, tmp, t + dt)
    if length(u) >= 256
        LinearAlgebra.BLAS.axpy!(dt/6, k1, u)
        LinearAlgebra.BLAS.axpy!(dt/3, k2, u)
        LinearAlgebra.BLAS.axpy!(dt/3, k3, u)
        LinearAlgebra.BLAS.axpy!(dt/6, k4, u)
    else
        @inbounds @simd for i in eachindex(u, k1, k2, k3, k4)
            u[i] += (dt / 6) * (k1[i] + 2k2[i] + 2k3[i] + k4[i])
        end
    end
    return nothing
end

"""
    euler_step!(u, dt, f!, t, k1)

Forward–Euler step applied in-place to `u` for the in-place drift `f!(du, u, t)`.
Scratch buffer `k1` must have the same length as `u`.
"""
function euler_step!(u, dt, f!, t, k1)
    f!(k1, u, t)
    if length(u) >= 256
        LinearAlgebra.BLAS.axpy!(dt, k1, u)
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
    mul!(tmp_vec, σ, noise_buf)
    @inbounds @simd for i in eachindex(u, tmp_vec)
        u[i] += s * tmp_vec[i]
    end
    return nothing
end

# --- Public API: dynamic path (internal) ---

"""
    evolve_dyn(u0, dt, Nsteps, f!, sigma; seed=123, resolution=1, timestepper=:rk4, boundary=nothing)

Dynamic-path single-trajectory integrator. Returns an array of size `(dim, Nsave+1)`.
"""
function evolve_dyn(u0, dt, Nsteps, f!, sigma;
                    seed::Integer=123, resolution::Integer=1,
                    timestepper::Symbol=:rk4, boundary::Union{Nothing,Tuple}=nothing)

    dim   = length(u0)
    T     = promote_type(Float64, eltype(u0), typeof(dt))
    Nsave = fld(Nsteps, resolution)

    u = copy(u0)
    results = Array{T}(undef, dim, Nsave+1)
    @views results[:, 1] .= u0

    rng = Random.MersenneTwister(seed)
    ts  = _resolve_stepper_dyn(timestepper)
    t = zero(T); save_index = 1
    noise_buf = Vector{T}(undef, dim)
    tmp_vec   = Vector{T}(undef, dim)
    k1 = similar(u, T); k2 = similar(u, T); k3 = similar(u, T); k4 = similar(u, T); tmp = similar(u, T)
    s = sqrt(T(dt))
    noise! = _make_noise_applier_dyn(sigma, u, t)

    if boundary === nothing
        if ts === rk4_step!
            next_save = resolution
            @inbounds for step in 1:Nsteps
                rk4_step!(u, dt, f!, t, k1, k2, k3, k4, tmp)
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
            next_save = resolution
            @inbounds for step in 1:Nsteps
                euler_step!(u, dt, f!, t, k1)
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
            next_save = resolution
            @inbounds for step in 1:Nsteps
                rk4_step!(u, dt, f!, t, k1, k2, k3, k4, tmp)
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
            next_save = resolution
            @inbounds for step in 1:Nsteps
                euler_step!(u, dt, f!, t, k1)
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
        println("Percentage of boundary crossings: ", count / Nsteps)
    end
    return results
end

"""
    evolve_ens_dyn(u0, dt, Nsteps, f!, sigma; seed=123, resolution=1, timestepper=:rk4, boundary=nothing, n_ens=1)

Dynamic-path ensemble integrator. Returns an array of size `(dim, Nsave+1, n_ens)`.
"""
function evolve_ens_dyn(u0, dt, Nsteps, f!, sigma;
                        seed::Integer=123, resolution::Integer=1,
                        timestepper::Symbol=:rk4, boundary::Union{Nothing,Tuple}=nothing,
                        n_ens::Integer=1)

    dim   = length(u0)
    T     = promote_type(Float64, eltype(u0), typeof(dt))
    Nsave = fld(Nsteps, resolution)
    results = Array{T}(undef, dim, Nsave+1, n_ens)

    Threads.@threads :dynamic for ens_idx in 1:n_ens
        rng = Random.MersenneTwister(seed + ens_idx * 1000)
        u = copy(u0)
        @views results[:, 1, ens_idx] .= u0
        ts = _resolve_stepper_dyn(timestepper)
        t = zero(T); save_index = 1
        noise_buf = Vector{T}(undef, dim)
        tmp_vec   = Vector{T}(undef, dim)
        k1 = similar(u, T); k2 = similar(u, T); k3 = similar(u, T); k4 = similar(u, T); tmp = similar(u, T)
        s = sqrt(T(dt))
        noise! = _make_noise_applier_dyn(sigma, u, t)

        if boundary === nothing
            if ts === rk4_step!
                next_save = resolution
                @inbounds for step in 1:Nsteps
                    rk4_step!(u, dt, f!, t, k1, k2, k3, k4, tmp)
                    noise!(u, s, t, rng, noise_buf, tmp_vec)
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
                next_save = resolution
                @inbounds for step in 1:Nsteps
                    euler_step!(u, dt, f!, t, k1)
                    noise!(u, s, t, rng, noise_buf, tmp_vec)
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
                next_save = resolution
                @inbounds for step in 1:Nsteps
                    rk4_step!(u, dt, f!, t, k1, k2, k3, k4, tmp)
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
                            results[i, save_index, ens_idx] = u[i]
                        end
                        next_save += resolution
                    end
                end
            else
                next_save = resolution
                @inbounds for step in 1:Nsteps
                    euler_step!(u, dt, f!, t, k1)
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
                            results[i, save_index, ens_idx] = u[i]
                        end
                        next_save += resolution
                    end
                end
            end
            if ens_idx == 1
                println("Percentage of boundary crossings: ", count / Nsteps)
            end
        end
    end
    return results
end
