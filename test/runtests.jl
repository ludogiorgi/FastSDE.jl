using Test
using Random
using StaticArrays
using FastSDE
using LinearAlgebra

"""
Simple linear drift: du/dt = -u
"""
function linear_decay!(du, u, t)
    @inbounds @views du .= -u
    return nothing
end

"""
Build a constant vector diffusion of length N with value σx per component.
"""
build_sigma_vector(N::Int, σx::Float64) = (u, t) -> fill(σx, N)

"""
Build a constant matrix diffusion Σ = σx * I of size N×N
"""
build_sigma_matrix(N::Int, σx::Float64) = (u, t) -> Matrix(I, N, N) .* σx

@testset "Hybrid dispatch and integrators" begin
    # Common parameters
    Random.seed!(123)
    dt = 0.01
    Nsteps = 200
    resolution = 10
    Nsave = fld(Nsteps, resolution) + 1

    # Deterministic: Euler should match (1 - dt)^Nsteps scaling exactly
    @testset "Deterministic Euler - static path" begin
        set_static_threshold!(1000)  # force static
        u0 = [1.0, 2.0, -1.0]
        sigma = 0.0
        traj = evolve(u0, dt, Nsteps, linear_decay!, sigma; seed=1, resolution=1, timestepper=:euler)
        expected_scale = (1 - dt)^Nsteps
        @test size(traj) == (length(u0), Nsteps + 1)
        @test isapprox(traj[:, end], expected_scale .* u0; rtol=0, atol=1e-12)
    end

    @testset "Deterministic Euler - dynamic path" begin
        set_static_threshold!(1)     # force dynamic for N > 1
        u0 = [1.5, -0.5, 0.25]
        sigma = 0.0
        traj = evolve(u0, dt, Nsteps, linear_decay!, sigma; seed=1, resolution=1, timestepper=:euler)
        expected_scale = (1 - dt)^Nsteps
        @test size(traj) == (length(u0), Nsteps + 1)
        @test isapprox(traj[:, end], expected_scale .* u0; rtol=0, atol=1e-12)
    end

    @testset "Vector σ support - static path" begin
        set_static_threshold!(1000)  # force static
        N = 5
        u0 = randn(N)
        sigma_vec_const = fill(0.1, N)
        # Pass as constant Vector to exercise static conversion path
        traj = evolve(u0, dt, Nsteps, linear_decay!, sigma_vec_const; seed=42, resolution=resolution, timestepper=:rk4)
        @test size(traj) == (N, Nsave)
        # Also test function returning Vector
        sigma_vec_fn = build_sigma_vector(N, 0.05)
        traj2 = evolve(u0, dt, Nsteps, linear_decay!, sigma_vec_fn; seed=42, resolution=resolution)
        @test size(traj2) == (N, Nsave)
    end

    # Check all timesteppers (:euler, :rk2, :rk4) agree with fixed-stage polynomial for du/dt=-u, σ=0
    @testset "All timesteppers - static and dynamic" begin
        amp_factor(sym, dt) = sym === :euler ? (1 - dt) : sym === :rk2 ? (1 - dt + 0.5*dt^2) : (1 - dt + 0.5*dt^2 - (dt^3)/6 + (dt^4)/24)
        for path in ("static", "dynamic")
            if path == "static"
                set_static_threshold!(1000)  # force static
            else
                set_static_threshold!(1)     # force dynamic
            end
            u0 = [1.0, -2.0, 0.5]
            sigma = 0.0
            for method in (:euler, :rk2, :rk4)
                traj = evolve(u0, dt, Nsteps, linear_decay!, sigma; seed=2024, resolution=1, timestepper=method)
                fac = amp_factor(method, dt)^Nsteps
                @test size(traj) == (length(u0), Nsteps + 1)
                @test isapprox(traj[:, end], fac .* u0; rtol=1e-10, atol=1e-10)
            end
        end
    end

    @testset "Vector σ support - dynamic path" begin
        set_static_threshold!(1)     # force dynamic for N > 1
        N = 6
        u0 = randn(N)
        sigma_vec_const = fill(0.2, N)
        traj = evolve(u0, dt, Nsteps, linear_decay!, sigma_vec_const; seed=7, resolution=resolution)
        @test size(traj) == (N, Nsave)
    end

    @testset "Matrix σ support - static path" begin
        set_static_threshold!(1000)  # force static
        N = 4
        u0 = randn(N)
        Σ_const = Matrix(I, N, N) .* 0.1
        traj = evolve(u0, dt, Nsteps, linear_decay!, Σ_const; seed=11, resolution=resolution)
        @test size(traj) == (N, Nsave)
    end

    @testset "Ensemble shapes and flattening" begin
        set_static_threshold!(1)     # force dynamic for N > 1
        N = 8
        u0 = randn(N)
        sigma = 0.0
        n_ens = 3
        traj3d = evolve_ens(u0, dt, Nsteps, linear_decay!, sigma; seed=9, resolution=resolution, n_ens=n_ens)
        @test size(traj3d) == (N, Nsave, n_ens)

        traj2d = evolve(u0, dt, Nsteps, linear_decay!, sigma; seed=9, resolution=resolution, n_ens=n_ens)
        @test size(traj2d) == (N, Nsave * n_ens)
    end

    @testset "Boundary reset keeps state within bounds" begin
        set_static_threshold!(1)     # force dynamic for N > 1
        N = 5
        u0 = fill(0.0, N)
        sigma = 0.5                   # non-zero noise to trigger excursions
        lo, hi = -0.1, 0.1
        traj = evolve(u0, dt, Nsteps, linear_decay!, sigma; seed=1234, resolution=resolution, boundary=(lo, hi))
        @test all((lo .<= traj) .& (traj .<= hi))
    end

    @testset "Threshold setter returns value" begin
        val = set_static_threshold!(64)
        @test val == 64
    end

    @testset "In-place sigma! - vector form (static path)" begin
        set_static_threshold!(1000)  # force static
        N = 4
        u0 = randn(N)
        dt = 0.01
        Nsteps = 100
        resolution = 10

        # Drift with params
        f_with_p!(du, u, p, t) = (du .= -p.decay * u)

        # In-place vector sigma! that scales by state magnitude
        params = (σ_base = 0.1, decay = 1.0)
        function sigma_inplace_vec!(σ_vec, u, p, t)
            @. σ_vec = p.σ_base * (1.0 + 0.1 * abs(u[1]))
            return nothing
        end

        traj = evolve(u0, dt, Nsteps, f_with_p!, sigma_inplace_vec!;
                      params=params, resolution=resolution, sigma_inplace=true)
        @test size(traj) == (N, fld(Nsteps, resolution) + 1)
        # Verify it's not identical to constant sigma (should differ due to state dependence)
        traj_const = evolve(u0, dt, Nsteps, f_with_p!, 0.1;
                            params=(decay=1.0,), resolution=resolution, seed=123)
        @test traj != traj_const  # Should differ due to state-dependent scaling
    end

    @testset "In-place sigma! - matrix form (dynamic path)" begin
        set_static_threshold!(1)  # force dynamic
        N = 5
        u0 = randn(N)
        dt = 0.01
        Nsteps = 100
        resolution = 10

        # Drift with params
        f_with_p!(du, u, p, t) = (du .= -p.decay * u)

        # In-place matrix sigma! with state-dependent correlation
        params = (σ_scale = 0.15, decay = 1.0)
        function sigma_inplace_mat!(Σ_mat, u, p, t)
            # Fill diagonal with scale
            for i in 1:size(Σ_mat, 1)
                for j in 1:size(Σ_mat, 2)
                    Σ_mat[i,j] = (i == j) ? p.σ_scale : 0.0
                end
            end
            return nothing
        end

        traj = evolve(u0, dt, Nsteps, f_with_p!, sigma_inplace_mat!;
                      params=params, resolution=resolution, sigma_inplace=true)
        @test size(traj) == (N, fld(Nsteps, resolution) + 1)
    end

    @testset "Input validation errors" begin
        set_static_threshold!(1)
        N = 3
        u0 = randn(N)
        sigma = 0.1

        # Test negative dt
        @test_throws ArgumentError evolve(u0, -0.01, 100, linear_decay!, sigma)

        # Test zero dt
        @test_throws ArgumentError evolve(u0, 0.0, 100, linear_decay!, sigma)

        # Test invalid resolution (< 1)
        @test_throws ArgumentError evolve(u0, 0.01, 100, linear_decay!, sigma; resolution=0)
        @test_throws ArgumentError evolve(u0, 0.01, 100, linear_decay!, sigma; resolution=-5)

        # Test negative n_burnin
        @test_throws ArgumentError evolve(u0, 0.01, 100, linear_decay!, sigma; n_ens=2, n_burnin=-1)

        # Test n_burnin >= Nsteps
        @test_throws ArgumentError evolve(u0, 0.01, 100, linear_decay!, sigma; n_ens=2, n_burnin=100)
        @test_throws ArgumentError evolve(u0, 0.01, 100, linear_decay!, sigma; n_ens=2, n_burnin=150)

        # Same validations for evolve_ens
        @test_throws ArgumentError evolve_ens(u0, -0.01, 100, linear_decay!, sigma; n_ens=2)
        @test_throws ArgumentError evolve_ens(u0, 0.0, 100, linear_decay!, sigma; n_ens=2)
        @test_throws ArgumentError evolve_ens(u0, 0.01, 100, linear_decay!, sigma; n_ens=2, resolution=0)
    end

    @testset "Static path RK2 ensemble" begin
        set_static_threshold!(1000)  # force static
        N = 4
        u0 = randn(N)
        dt = 0.01
        Nsteps = 100
        n_ens = 8

        # Test RK2 works in static ensemble path
        ens_rk2 = evolve_ens(u0, dt, Nsteps, linear_decay!, 0.1;
                             timestepper=:rk2, n_ens=n_ens, resolution=10, seed=42)
        @test size(ens_rk2) == (N, 11, n_ens)

        # Test determinism
        ens_rk2_2 = evolve_ens(u0, dt, Nsteps, linear_decay!, 0.1;
                               timestepper=:rk2, n_ens=n_ens, resolution=10, seed=42)
        @test ens_rk2 == ens_rk2_2
    end
end

include("batched_ensemble.jl")

