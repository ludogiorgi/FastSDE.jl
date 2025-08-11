import Pkg
Pkg.activate(@__DIR__)        # use benchmarks/Project.toml
Pkg.instantiate() 

using Random
using LinearAlgebra
using Statistics
using FastSDE
using StochasticDiffEq

println("\nFastSDE vs StochasticDiffEq (parallel ensembles), fixed dt, full saving\n")

# Problem definition: du = -u dt + Σ dW (additive matrix noise)
struct BenchConfig
    N::Int
    dt::Float64
    Nsteps::Int
    repeats::Int
    sigma::Float64
end

make_problem_fast(N::Int, σ::Float64) = ((du,u,t)->(@inbounds du .= -u; nothing))

function make_problem_sde(N::Int, σ::Float64)
    f4(u, p, t) = -u
    g4(u, p, t) = Matrix(I, N, N) .* σ   # N×N diffusion
    return f4, g4
end

function bench_fast(cfg::BenchConfig, u0::AbstractVector, Σ::AbstractMatrix; n_ens::Int)
    f3! = make_problem_fast(cfg.N, cfg.sigma)
    # scalar sigma
    times = Float64[]
    # Force dispatch mode depending on N for coverage
    set_static_threshold!(64)
    for _ in 1:cfg.repeats
        t = @elapsed begin
            _ = evolve(u0, cfg.dt, cfg.Nsteps, f3!, Σ;
                       seed=123, resolution=1, timestepper=:rk4, n_ens=n_ens)
        end
        push!(times, t)
    end
    return minimum(times)
end

function bench_sde(cfg::BenchConfig, u0::AbstractVector, Σ::AbstractMatrix; n_ens::Int)
    f4, _ = make_problem_sde(cfg.N, cfg.sigma)
    tspan = (0.0, cfg.dt * cfg.Nsteps)
    g4mat(u, p, t) = Σ
    prob = SDEProblem(f4, g4mat, u0, tspan; noise_rate_prototype=zeros(eltype(u0), cfg.N, cfg.N))
    times = Float64[]
    # High-order additive-noise solver (weak order 3)
    alg = SRA3()
    for _ in 1:cfg.repeats
        Random.seed!(123)
        t = @elapsed begin
            # Parallel ensemble: allocate and fill
            Nsave = cfg.Nsteps + 1
            results = Array{Float64}(undef, cfg.N, Nsave, n_ens)
            Threads.@threads :dynamic for ei in 1:n_ens
                # independent RNG seeds
                u0i = copy(u0)
                probi = remake(prob; u0=u0i)
                soli = solve(probi, alg; dt=cfg.dt, adaptive=false, save_everystep=true, save_start=true)
                # Write into results
                @inbounds for k in 1:Nsave
                    @views results[:, k, ei] .= soli.u[k]
                end
            end
            # Flatten to match FastSDE output shape
            dim, timesteps, ensembles = size(results)
            out = Array{Float64}(undef, dim, timesteps*ensembles)
            @inbounds for i in 1:ensembles
                @views out[:, (i-1)*timesteps+1 : i*timesteps] .= results[:, :, i]
            end
            out
        end
        push!(times, t)
    end
    return minimum(times)
end

function run_suite()
    Random.seed!(123)
    Ns = [8, 32, 128, 512]
    dt = 1e-3
    Nsteps = 200
    repeats = 3
    σ = 0.2
    n_ens = max(Threads.nthreads(), 16)

    println(rpad("N", 8), rpad("FastSDE (s)", 16), "StochasticDiffEq (s)")
    println(repeat('-', 40))
    for N in Ns
        cfg = BenchConfig(N, dt, Nsteps, repeats, σ)
        # shared initial condition and sigma for both solvers
        u0 = randn(N)
        Σ = Matrix(I, N, N) .* σ
        tf = bench_fast(cfg, u0, Σ; n_ens=n_ens)
        ts = bench_sde(cfg, u0, Σ; n_ens=n_ens)
        # simple formatted printing without using Printf
        tf_str = lpad(string(round(tf; digits=3)), 8)
        ts_str = lpad(string(round(ts; digits=3)), 8)
        println(rpad(string(N), 8), rpad(tf_str, 16), ts_str)
    end
end

run_suite()


