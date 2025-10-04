# AGENTS.md — Add **Batched Drift** fast path to `FastSDE.jl`

## Objective
Implement an **optional batched drift** path for ensemble integration so neural-network drifts can be evaluated for **all ensemble members at once**. This adds a keyword `batched_drift::Bool=false` to `evolve_ens`. When `true`, the ensemble integrator calls:

```julia
f!(DU::AbstractMatrix, U::AbstractMatrix, p, t)
```

with `U, DU` shaped `(D, n_ens)`. The default columnwise behavior remains unchanged and fully backward-compatible.

---

## Constraints & Non‑Goals
- **Backward compatible**: No breaking changes. Default remains columnwise (`batched_drift=false`).
- **No new dependencies**: Use only `Base`, `Random`, and `LinearAlgebra`.
- **Allocation‑free inner loops**: Preallocate buffers and reuse them each step.
- **Noise parity with current API**: Support `sigma::Number`, `AbstractVector` (diagonal), `AbstractMatrix` (correlated; cache Cholesky), and in‑place diffusion via a callable `sigma!(Ξ, U, p, t)` when `sigma_inplace=true`.
- **Timesteppers**: Support `:euler`, `:rk2`, `:rk4` (deterministic RK with EM noise).
- **GPU friendliness**: Keep code generic over `AbstractArray`. If `u0` is a `CuArray`, buffers should live on GPU; `randn!` will dispatch accordingly.

---

## Deliverables (in this PR)
1. `src/FastSDE.jl`: add new keyword and route to the batched path.
2. `src/ensemble_batched.jl`: new implementation for the batched ensemble integrator.
3. `test/batched_ensemble.jl`: tests for shapes, determinism, OU variance sanity, correlated noise branch.
4. `examples/nn_batched_ensemble.jl`: example with Flux MLP drift (not used in tests).
5. `README.md`: new “Batched NN drift (fast)” section with a minimal example.

---

## Pre‑flight
1. Ensure the repository test suite is green **before** changes:
   ```bash
   julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
   ```
2. Use Julia ≥ 1.9 for consistent RNG performance and BLAS behavior.

---

## Step 1 — Wire the new path in `src/FastSDE.jl`
**Action:** Add the include and the new keyword; route early to `_evolve_ens_batched` when `batched_drift=true`.

Insert near other includes:
```julia
include("ensemble_batched.jl")
```

Modify the signature and add the early return:

```julia
function evolve_ens(u0, dt, Nsteps, f!, sigma;
                    params=nothing, n_ens::Integer=1, resolution::Integer=1,
                    seed::Integer=123, timestepper::Symbol=:euler, boundary=nothing,
                    rng=nothing, verbose::Bool=false, manage_blas_threads::Bool=true,
                    sigma_inplace::Bool=true, batched_drift::Bool=false)

    if batched_drift
        return _evolve_ens_batched(u0, dt, Nsteps, f!, sigma;
            params=params, n_ens=n_ens, resolution=resolution, seed=seed,
            timestepper=timestepper, boundary=boundary, rng=rng, verbose=verbose,
            manage_blas_threads=manage_blas_threads, sigma_inplace=sigma_inplace)
    end

    # ... existing columnwise ensemble implementation ...
end
```

> Do **not** change existing defaults or behavior.

---

## Step 2 — Create `src/ensemble_batched.jl`
**Action:** Add the batched implementation below. **Do not** open a new module here; this file is included inside `module FastSDE`.

```julia
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
```

---

## Step 3 — Add tests in `test/batched_ensemble.jl`
Tests avoid any ML dependency; they use a linear OU drift with known variance.

```julia
# test/batched_ensemble.jl
using Test, Random, LinearAlgebra
using FastSDE

@testset "Batched ensemble" begin
    D, N = 4, 256
    γ = 0.3
    dt = 1e-3
    steps = 50_000
    u0 = randn(D)
    p  = (γ = γ,)

    # linear OU drift: DU .= -γ * U (batched signature)
    f_lin_batched!(DU, U, p, t) = (DU .= .-p.γ .* U)

    # scalar diffusion
    σ = 0.2

    ens = evolve_ens(u0, dt, steps, f_lin_batched!, σ;
                     params=p, n_ens=N, resolution=100,
                     batched_drift=true, seed=42)

    @test size(ens) == (D, (steps ÷ 100) + 1, N)

    # determinism with same seed
    ens2 = evolve_ens(u0, dt, steps, f_lin_batched!, σ;
                      params=p, n_ens=N, resolution=100,
                      batched_drift=true, seed=42)
    @test ens == ens2

    # correlated diffusion sanity
    Σ = 0.04 * Symmetric(I(D))
    ens3 = evolve_ens(u0, dt, steps, f_lin_batched!, Σ;
                      params=p, n_ens=N, resolution=100,
                      batched_drift=true, seed=7)
    @test size(ens3) == size(ens)

    # OU stationary variance: Var[X] ≈ σ^2 / (2γ)
    target_var = σ^2 / (2γ)
    final = ens[:, end, :]
    est_var = mean(var(final; dims=2))
    @test isapprox(est_var, target_var; rtol=0.25)
end
```

Run:
```bash
julia --project -e 'using Pkg; Pkg.test()'
```

---

## Step 4 — Add example in `examples/nn_batched_ensemble.jl`
> Example depends on Flux but is not part of tests.

```julia
using FastSDE, Flux, Random

Random.seed!(0)

D, N = 8, 8192
nn = Chain(Dense(D,128,gelu), Dense(128,D))
p  = (nn = nn,)

# Batched NN drift: DU, U are (D, n_ens)
f_nn_batched!(DU, U, p, t) = (DU .= p.nn(U))

u0 = randn(D)
σ  = 0.15
dt = 1e-3
steps = 50_000

ens = evolve_ens(u0, dt, steps, f_nn_batched!, σ;
                 params=p, n_ens=N, resolution=200,
                 batched_drift=true, manage_blas_threads=true)

@show size(ens)  # (D, Nsave, N)
```

---

## Step 5 — Update `README.md`
Append a short section (keep style consistent with the repo):

```markdown
### Batched NN drift (fast)

When your drift is a neural network, evaluate it **for all ensemble members at once** by enabling `batched_drift=true` and using a batched drift signature:

```julia
using FastSDE, Flux

# 4D MLP drift
nn = Chain(Dense(4,128,tanh), Dense(128,4))
p  = (nn = nn,)

# DU, U are (4, n_ens)
f_nn_batched!(DU, U, p, t) = (DU .= p.nn(U))

σ  = 0.1
u0 = randn(4)

ens = evolve_ens(u0, 1e-3, 100_000, f_nn_batched!, σ;
                 params=p, n_ens=4096, resolution=200, batched_drift=true)
```

This issues large BLAS ops (CPU) or GPU kernels instead of `n_ens` small calls, giving a substantial speedup for ML-based drifts. Noise types `Number` (additive), `Vector` (diagonal), `Matrix` (correlated via cached Cholesky), and in‑place `sigma!(Ξ,U,p,t)` with `sigma_inplace=true` are supported exactly as in the columnwise path.
```

---

## Post‑flight
1. **Run tests**:
   ```bash
   julia --project -e 'using Pkg; Pkg.test()'
   ```
2. (Optional) **Benchmark**: Compare `batched_drift=true` vs. per-trajectory wrapper for `n_ens ∈ {1e2,1e3,1e4}` with a small MLP drift.
3. **Lint/format**: ensure style matches the repo (indentation, UTF‑8 encoding).
4. **Commit**:
   ```bash
   git checkout -b feat/batched-drift-ensemble
   git add src/FastSDE.jl src/ensemble_batched.jl test/batched_ensemble.jl examples/nn_batched_ensemble.jl README.md
   git commit -m "feat(ensemble): add batched drift path for NN-friendly ensemble integration; docs & tests"
   git push origin feat/batched-drift-ensemble
   ```
5. **PR description (suggested)**:
   - What/Why/How, BC statement, docs and tests summary.
   - Mention that stochastic increment uses EM (`+ sqrt(dt)*…`) combined with deterministic RK, consistent with current library semantics.

---

## Notes & Rationale
- **Why batched?** Neural drift evaluation dominates runtime for large ensembles. Batched calls turn many tiny NN evaluations into a handful of matrix ops that saturate CPU BLAS or GPU cores.
- **Noise semantics:** EM increment is `U += dt * DU + sqrt(dt) * ξ_scaled`. Users should pass the Langevin diffusion as `σ = sqrt(2β⁻¹)` (no hidden √2 in the integrator).
- **Correlated noise:** `sigma::AbstractMatrix` uses `cholesky(Symmetric(sigma)).L` once, then multiplies `Ξ ← LΞ` per step.
- **GPU:** Keeping everything `AbstractArray`-generic lets CUDA/AMDGPU work without special casing, provided the user supplies `CuArray` inputs and GPU‑compatible `f!`.

```
