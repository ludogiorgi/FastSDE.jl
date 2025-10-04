# test/batched_ensemble.jl
using Test, Random, LinearAlgebra, Statistics
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

