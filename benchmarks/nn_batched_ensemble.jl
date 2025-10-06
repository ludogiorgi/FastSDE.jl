import Pkg
Pkg.activate(@__DIR__)        # use benchmarks/Project.toml
Pkg.instantiate() 

using FastSDE, Flux, Random


Random.seed!(0)

D, N = 8, 100
nn = Chain(Dense(D,128,gelu), Dense(128,D))
p  = (nn = nn,)

# Batched NN drift: DU, U are (D, n_ens)
f_nn_batched!(DU, U, p, t) = (DU .= p.nn(U))

u0 = randn(D)
σ  = 0.15
dt = 1e-3
steps = 5000

@time ens = evolve(u0, dt, steps, f_nn_batched!, σ;
                 params=p, n_ens=N, resolution=200, boundary=(-10, 10),
                 batched_drift=true, manage_blas_threads=true)

@show size(ens)  # (D, Nsave, N)

