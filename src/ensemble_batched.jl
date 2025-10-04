# src/ensemble_batched.jl
# Batched ensemble integration fast path (U, DU, Xi, K* are (D, n_ens))

using LinearAlgebra: BLAS, Symmetric
using Random

# Timesteppers handled via Val-dispatch
const _Timestepper = Union{Val{:euler},Val{:rk2},Val{:rk4}}

@inline _tstep(sym::Symbol) =
    sym === :euler ? Val(:euler) :
    sym === :rk2   ? Val(:rk2)   :
    sym === :rk4   ? Val(:rk4)   :
    throw(ArgumentError("timestepper must be :euler, :rk2 or :rk4"))

# Public entry point from evolve_ens when batched_drift=true
function _evolve_ens_batched(u0, dt, Nsteps, f!, sigma;
                             params=nothing, n_ens::Integer=1, resolution::Integer=1,
                             seed::Integer=123, timestepper::Symbol=:euler, boundary=nothing,
                             rng=nothing, verbose::Bool=false, manage_blas_threads::Bool=true,
                             sigma_inplace::Bool=true)

    D = length(u0)
    T = promote_type(eltype(u0), typeof(dt))
    tval = _tstep(timestepper)

    # allocate state and buffers
    U   = _tile_state(u0, n_ens)     # (D, n_ens)
    DU  = similar(U)
    Tmp = similar(U)
    Xi  = similar(U)

    # RK buffers (unused for :euler but cheap to prealloc)
    K1 = similar(U); K2 = similar(U); K3 = similar(U); K4 = similar(U)

    # saving buffer: (D, Nsave, n_ens)
    Nsave = (Nsteps ÷ resolution) + 1
    out = similar(U, D, Nsave, n_ens)
    out[:, 1, :] .= U

    # RNG (let GPU fall back to array-specific randn! if rng is nothing)
    _rng = isnothing(rng) ? MersenneTwister(seed) : rng

    # correlated diffusion factor (optional)
    Lcorr = nothing
    if sigma isa AbstractMatrix
        Lcorr = cholesky(Symmetric(sigma)).L
    end

    # optional BLAS thread mgmt (avoid nested threading if outer code threads elsewhere)
    old_blas = nothing
    if manage_blas_threads
        old_blas = BLAS.get_num_threads()
        # leave as-is in batched path; users can tune externally if needed
        BLAS.set_num_threads(old_blas)
    end

    t = zero(T)
    saveidx = 2

    @inbounds for k in 1:Nsteps
        _batched_step!(U, DU, Tmp, Xi, K1, K2, K3, K4,
                       f!, sigma, params, t, dt, tval,
                       sigma_inplace, Lcorr, _rng)

        # optional boundary reset to initial u0 when leaving [lo, hi]
        if boundary !== nothing
            lo, hi = boundary
            for j in 1:n_ens
                reset = false
                @simd for i in 1:D
                    @inbounds if (U[i,j] < lo) | (U[i,j] > hi); reset = true; end
                end
                reset && (@views U[:,j] .= @views out[:,1,1])
            end
        end

        t += dt
        if (k % resolution) == 0
            out[:, saveidx, :] .= U
            saveidx += 1
        end
    end

    if manage_blas_threads && old_blas !== nothing
        BLAS.set_num_threads(old_blas)
    end

    return out
end

# ---- helpers ----

@inline function _tile_state(u0::AbstractVector, n_ens::Integer)
    # Make a (D, n_ens) matrix by repeating the column vector
    U = repeat(reshape(u0, :, 1), 1, n_ens)
    return U
end

@inline function _randn!(rng, A)
    if rng === nothing
        randn!(A)
    else
        randn!(rng, A)
    end
    return A
end

# Draw standard normals (all columns), then scale by sigma and sqrt(dt)
@inline function _noise!(rng, Xi, sigma, sqrt_dt, Lcorr)
    _randn!(rng, Xi)
    if sigma isa Number
        @. Xi = sqrt_dt * sigma * Xi
    elseif sigma isa AbstractVector
        @. Xi = sqrt_dt * sigma * Xi    # row-wise scale (length D) broadcast across columns
    elseif sigma isa AbstractMatrix
        mul!(Xi, Lcorr, Xi)             # Xi ← L * Xi
        @. Xi = sqrt_dt * Xi
    else
        throw(ArgumentError("Unsupported sigma type in batched path"))
    end
    return nothing
end

# In-place batched diffusion evaluation: sigma!(Xi, U, p, t) multiplies Xi by Σ^{1/2}(U,t)
@inline function _noise_inplace!(rng, Xi, sigma!, U, p, t, sqrt_dt)
    _randn!(rng, Xi)
    sigma!(Xi, U, p, t)                 # user fills Xi ← Σ^{1/2}(U,t) * Xi
    @. Xi = sqrt_dt * Xi
    return nothing
end

# Euler–Maruyama (batched)
@inline function _batched_step!(U, DU, Tmp, Xi, K1, K2, K3, K4,
                                f!, sigma, p, t, dt, ::Val{:euler},
                                sigma_inplace, Lcorr, rng)
    sqrt_dt = sqrt(dt)
    f!(DU, U, p, t)
    if sigma_inplace === true && sigma isa Function
        _noise_inplace!(rng, Xi, sigma, U, p, t, sqrt_dt)
    else
        _noise!(rng, Xi, sigma, sqrt_dt, Lcorr)
    end
    @. U = U + dt * DU + Xi
    return nothing
end

# RK2 (deterministic Heun) + EM noise
@inline function _batched_step!(U, DU, Tmp, Xi, K1, K2, K3, K4,
                                f!, sigma, p, t, dt, ::Val{:rk2},
                                sigma_inplace, Lcorr, rng)
    sqrt_dt = sqrt(dt)
    f!(K1, U, p, t)
    @. Tmp = U + 0.5 * dt * K1
    f!(K2, Tmp, p, t)
    @. DU = 0.5 * (K1 + K2)
    if sigma_inplace === true && sigma isa Function
        _noise_inplace!(rng, Xi, sigma, U, p, t, sqrt_dt)
    else
        _noise!(rng, Xi, sigma, sqrt_dt, Lcorr)
    end
    @. U = U + dt * DU + Xi
    return nothing
end

# RK4 (deterministic) + EM noise
@inline function _batched_step!(U, DU, Tmp, Xi, K1, K2, K3, K4,
                                f!, sigma, p, t, dt, ::Val{:rk4},
                                sigma_inplace, Lcorr, rng)
    sqrt_dt = sqrt(dt)
    f!(K1, U, p, t)
    @. Tmp = U + 0.5 * dt * K1
    f!(K2, Tmp, p, t)
    @. Tmp = U + 0.5 * dt * K2
    f!(K3, Tmp, p, t)
    @. Tmp = U + dt * K3
    f!(K4, Tmp, p, t)
    @. DU = (K1 + 2K2 + 2K3 + K4) / 6
    if sigma_inplace === true && sigma isa Function
        _noise_inplace!(rng, Xi, sigma, U, p, t, sqrt_dt)
    else
        _noise!(rng, Xi, sigma, sqrt_dt, Lcorr)
    end
    @. U = U + dt * DU + Xi
    return nothing
end

