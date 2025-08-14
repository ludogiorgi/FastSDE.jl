import Pkg
Pkg.activate(@__DIR__)
# Ensure we use the local FastSDE sources for benchmarking
Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.resolve()
Pkg.instantiate()

##
using BenchmarkTools
using Random
using LinearAlgebra
using StaticArrays
using FastSDE
using DifferentialEquations

const p = (dᵤ=0.2, wᵤ=0.4, dₜ=2.0, σ₁=0.3, σ₂=0.3)

function run()
    println("\nBenchmark: Triad-like 3D system (FastSDE vs DifferentialEquations.EM)\n")

    Random.seed!(123)

    # Problem setup
    dim = 3
    dt = 0.01
    Nsteps = 1_000_000
    u0 = [0.0, 0.0, 0.0]
    resolution = 10

    # Force static path
    set_static_threshold!(64)

    println("Config: dim=$(dim), dt=$(dt), Nsteps=$(Nsteps), save every $(resolution) steps")

    # Drift function for FastSDE (in-place)
    function f!(du, u, t)
        du[1] = -p.dᵤ * u[1] - p.wᵤ * u[2] + u[3]
        du[2] = -p.dᵤ * u[2] + p.wᵤ * u[1]
        du[3] = -p.dₜ * u[3]
    end

    # Single diffusion form: in-place MVector, matching the user example
    function sigma!(out::MVector{3,Float64}, u::MVector{3,Float64}, t::Float64)
        out[1] = p.σ₁
        out[2] = p.σ₂
        out[3] = 1.5 * (tanh(u[1]) + 1)
    end

    println("\nFastSDE (evolve, Euler):")
    bench_fast = @benchmark evolve($u0, $dt, $(1*Nsteps), $f!, $sigma!; timestepper=:euler, resolution=$resolution, n_ens=100)
    println(bench_fast)

    # DifferentialEquations setup (EM)
    function drift(du, u, p, t)
        du[1] = -p.dᵤ * u[1] - p.wᵤ * u[2] + u[3]
        du[2] = -p.dᵤ * u[2] + p.wᵤ * u[1]
        du[3] = -p.dₜ * u[3]
    end
    function diffusion(du, u, p, t)
        du[1] = p.σ₁
        du[2] = p.σ₂
        du[3] = 1.5 * (tanh(u[1]) + 1)
    end
    println("\nDifferentialEquations (EM):")
    tspan = (0.0, (100*Nsteps) * dt)
    prob = SDEProblem(drift, diffusion, copy(u0), tspan, p)
    bench_de = @benchmark solve($prob, EM(); dt=$dt, saveat=$(resolution*dt))
    println(bench_de)

    println("\nDone.\n")
end

run()

