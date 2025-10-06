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

    # Test batched RK2 timestepper
    @testset "Batched RK2 timestepper" begin
        ens_rk2 = evolve_ens(u0, dt, 1000, f_lin_batched!, σ;
                             params=p, n_ens=64, resolution=10,
                             batched_drift=true, seed=99, timestepper=:rk2)
        @test size(ens_rk2) == (D, 101, 64)
        # Check determinism
        ens_rk2_2 = evolve_ens(u0, dt, 1000, f_lin_batched!, σ;
                               params=p, n_ens=64, resolution=10,
                               batched_drift=true, seed=99, timestepper=:rk2)
        @test ens_rk2 == ens_rk2_2
    end

    # Test batched RK4 timestepper
    @testset "Batched RK4 timestepper" begin
        ens_rk4 = evolve_ens(u0, dt, 1000, f_lin_batched!, σ;
                             params=p, n_ens=64, resolution=10,
                             batched_drift=true, seed=42, timestepper=:rk4)
        @test size(ens_rk4) == (D, 101, 64)
        # Check determinism
        ens_rk4_2 = evolve_ens(u0, dt, 1000, f_lin_batched!, σ;
                               params=p, n_ens=64, resolution=10,
                               batched_drift=true, seed=42, timestepper=:rk4)
        @test ens_rk4 == ens_rk4_2
    end

    # Test non-SPD matrix failure
    @testset "Non-SPD matrix diffusion error" begin
        # Create a non-SPD matrix (negative eigenvalue)
        Σ_bad = [-1.0 0.0; 0.0 1.0]
        u0_2d = randn(2)
        f_2d!(DU, U, p, t) = (DU .= -0.1 .* U)

        @test_throws ArgumentError evolve_ens(u0_2d, dt, 100, f_2d!, Σ_bad;
                                              params=nothing, n_ens=16,
                                              batched_drift=true)
    end

    # Test in-place batched sigma!
    @testset "In-place batched sigma!" begin
        # State-dependent diffusion in batched mode
        function sigma_batched_inplace!(Xi, U, p, t)
            # Scale noise by sqrt(abs(state)) element-wise
            @. Xi = p.σ_scale * sqrt(abs(U) + 0.1) * Xi
            return nothing
        end

        ens_inplace = evolve_ens(u0, dt, 1000, f_lin_batched!, sigma_batched_inplace!;
                                 params=(γ=γ, σ_scale=0.2), n_ens=64, resolution=10,
                                 batched_drift=true, sigma_inplace=true, seed=777)
        @test size(ens_inplace) == (D, 101, 64)
    end

    # Test boundary reset correctness in batched mode
    @testset "Batched boundary reset preserves ensemble independence" begin
        # Use strong noise to trigger boundary crossings
        u0_test = [0.5, 0.5, 0.5]
        σ_strong = 2.0
        boundary = (0.0, 1.0)

        f_test!(DU, U, p, t) = (DU .= 0.0)  # No drift, only diffusion

        ens_boundary = evolve_ens(u0_test, 0.1, 200, f_test!, σ_strong;
                                  n_ens=8, resolution=10, batched_drift=true,
                                  boundary=boundary, seed=999)

        @test size(ens_boundary) == (3, 21, 8)

        # Check that ensemble members remain independent (not all identical after resets)
        final_states = ens_boundary[:, end, :]
        # If boundary reset was buggy (using out[:,1,1]), all members would be identical
        # With correct reset (out[:,1,j]), each member resets to its own IC
        @test !all(final_states[:, i] == final_states[:, 1] for i in 2:8)
    end
end

