import Pkg
Pkg.activate(@__DIR__)
# Ensure we use the local FastSDE sources for benchmarking
Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.resolve()
Pkg.instantiate()

using BenchmarkTools
using Random
using LinearAlgebra
using StaticArrays
using FastSDE
using StochasticDiffEq

function run()
    println("\nBenchmark: Triad-like 3D system (FastSDE vs StochasticDiffEq.EM)\n")

    Random.seed!(123)

    # Parameters
    p = (dᵤ=0.2, wᵤ=0.4, dₜ=2.0, σ₁=0.3, σ₂=0.3)

    # Problem setup
    dim = 3
    dt = 0.01
    Nsteps = 1_000_000  # 1e6 to keep runtime reasonable here
    u0 = zeros(dim)
    resolution = 10

    # Force static path
    set_static_threshold!(64)

    println("Config: dim=$(dim), dt=$(dt), Nsteps=$(Nsteps), save every $(resolution) steps")

    # Drift function for FastSDE (in-place)
    function drift_fast!(du, u, t)
        du[1] = -p.dᵤ * u[1] - p.wᵤ * u[2] + u[3]
        du[2] = -p.dᵤ * u[2] + p.wᵤ * u[1]
        du[3] = -p.dₜ * u[3]
    end

    # Sigma variants
    function sigma_mvec!(out::MVector{3,Float64}, u::MVector{3,Float64}, t::Float64)
        out[1] = p.σ₁
        out[2] = p.σ₂
        out[3] = 1.5 * (tanh(u[1]) + 1)
    end
    sigma_svec(u, t) = @SVector [p.σ₁, p.σ₂, 1.5 * (tanh(u[1]) + 1)]
    sigma_alloc(u, t) = [p.σ₁, p.σ₂, 1.5 * (tanh(u[1]) + 1)]

    function run_fast(name, sigma)
        print(rpad(name, 18))
        GC.gc()
        local val, t, bytes, gctime, memallocs
        val, t, bytes, gctime, memallocs = @timed evolve(u0, dt, Nsteps, drift_fast!, sigma; timestepper=:euler, resolution=resolution, n_ens=1)
        println(" time=$(round(t; digits=3))s, allocs=$(memallocs), bytes=$(round(bytes/1024^2; digits=1)) MiB")
    end

    println("\nFastSDE variants:")
    run_fast("sigma! MVector", sigma_mvec!)
    run_fast("sigma SVector", sigma_svec)
    run_fast("sigma Vector", sigma_alloc)
    run_fast("sigma = 0.0", (u,t)->0.0)

    # StochasticDiffEq (EM)
    function drift_de(du, u, p, t)
        du[1] = -p.dᵤ * u[1] - p.wᵤ * u[2] + u[3]
        du[2] = -p.dᵤ * u[2] + p.wᵤ * u[1]
        du[3] = -p.dₜ * u[3]
    end
    function diffusion_de(du, u, p, t)
        du[1] = p.σ₁
        du[2] = p.σ₂
        du[3] = 1.5 * (tanh(u[1]) + 1)
    end
    println("\nStochasticDiffEq (EM):")
    tspan = (0.0, Nsteps * dt)
    prob = SDEProblem(drift_de, diffusion_de, copy(u0), tspan, p)
    GC.gc()
    val2, t2, bytes2, gc2, allocs2 = @timed solve(prob, EM(); dt=dt, adaptive=false, saveat=resolution*dt)
    println("EM() time=$(round(t2; digits=3))s, allocs=$(allocs2), bytes=$(round(bytes2/1024^2; digits=1)) MiB")

    println("\nDone.\n")
end

run()

