using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Random
using Plots
using FastSDE   # exports: evolve, evolve_ens, set_static_threshold!

# ──────────────────────────────────────────────────────────────────────────────
# Two-scale Lorenz–96: in-place drift f!(du, u, t)
# State layout: u = [X[1:K]; vec(Y[1:J, 1:K])]
# ──────────────────────────────────────────────────────────────────────────────
function build_l96_drift!(K::Int, J::Int, F::Float64, h::Float64, c::Float64, b::Float64)
    mod1K(k) = Base.mod1(k, K)
    _idx(j, k) = (j - 1) + (k - 1) * J + 1

    # Wrap (j,k): when j overflows, move to next/prev k
    wrap_jk(j, k) = begin
        q = fld(j - 1, J)                  # how many times we wrapped in j
        jr = j - q * J                      # wrapped j back into 1:J
        kr = mod1K(k + q)                   # shift k by the same amount
        (jr, kr)
    end

    function f!(du::AbstractVector{<:Real}, u::AbstractVector{<:Real}, t::Real)
        @views begin
            X  = u[1:K];         Y  = u[K+1:end]
            dX = du[1:K];        dY = du[K+1:end]
        end

        # dX/dt with column-mean of Y
        @inbounds for k in 1:K
            acc = 0.0
            base = (k - 1) * J
            @inbounds for j in 1:J
                acc += Y[base + j]
            end
            dX[k] = -X[mod1K(k - 1)] * (X[mod1K(k - 2)] - X[mod1K(k + 1)]) - X[k] + F - h * c * (acc / J)
        end

        # dY/dt
        @inbounds for k in 1:K
            for j in 1:J
                j_p1, k_p1 = wrap_jk(j + 1, k)
                j_p2, k_p2 = wrap_jk(j + 2, k)
                j_m1, k_m1 = wrap_jk(j - 1, k)

                jj   = _idx(j,   k)
                jp1  = _idx(j_p1,k_p1)
                jp2  = _idx(j_p2,k_p2)
                jm1  = _idx(j_m1,k_m1)

                Y_jk   = @inbounds Y[jj]
                Y_jp1k = @inbounds Y[jp1]
                Y_jp2k = @inbounds Y[jp2]
                Y_jm1k = @inbounds Y[jm1]

                @inbounds dY[jj] = c * ( -b * Y_jp1k * (Y_jp2k - Y_jm1k) - Y_jk + (h / J) * X[k] )
            end
        end
        return nothing
    end
    return f!
end

# ──────────────────────────────────────────────────────────────────────────────
# Constant per-component diffusion σ(u,t) as a vector:
# first K entries use sigma_x; remaining J*K entries use sigma_y
# Works with the new FastSDE constant-σ lifting (no allocations in the loop).
# ──────────────────────────────────────────────────────────────────────────────
function build_sigma_fn(K::Int, J::Int, sigma_x::Float64, sigma_y::Float64)
    σvec = vcat(fill(sigma_x, K), fill(sigma_y, J * K))  # built once
    sigma(u, t) = σvec
    return sigma
end

# ──────────────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────────────
K = 36
J = 10
F = 10.0
h = 1.0
c = 10.0
b = 10.0

sigma_x = 0.5
sigma_y = 0.2

Tfinal = 20.0
dt     = 5e-3
saveat = 0.1

# (Optional) pick StaticArrays path cutoff; default is 64 in the package.
# For this high-dimensional example (N = K + J*K = 396), dynamic path will be used anyway.
# FastSDE.set_static_threshold!(64)

# ──────────────────────────────────────────────────────────────────────────────
# Initial condition
# ──────────────────────────────────────────────────────────────────────────────
Random.seed!(123)
N  = K + J * K
X0 = F .+ 0.01 .* randn(K)
Y0 = 0.01 .* randn(J, K)

u0 = Vector{Float64}(undef, N)
@views u0[1:K]        .= X0
@views u0[K+1:end]    .= vec(Y0)

# ──────────────────────────────────────────────────────────────────────────────
# Time discretization
# ──────────────────────────────────────────────────────────────────────────────
Nsteps     = ceil(Int, Tfinal / dt)
save_every = max(1, round(Int, saveat / dt))

# ──────────────────────────────────────────────────────────────────────────────
# Build model and integrate
# ──────────────────────────────────────────────────────────────────────────────
f!    = build_l96_drift!(K, J, F, h, c, b)
sigma = build_sigma_fn(K, J, sigma_x, sigma_y)

traj = evolve(u0, dt, Nsteps, f!, sigma;
                    seed=123, resolution=save_every, timestepper=:rk4)
# ── Heatmap of slow variables X₁..X_K over time ───────────────────────────────
num_saved = fld(Nsteps, save_every) + 1
t = collect(0:num_saved-1) .* (save_every * dt)

# Downsample to a reasonable number of time columns
max_cols = 1200                          # tweak if you want more/less detail
stride   = max(1, ceil(Int, num_saved / max_cols))
idxs     = 1:stride:num_saved

t_sub   = t[idxs]
X_sub   = @view traj[1:K, idxs]          # (K × length(idxs)) as required by Plots.heatmap

heatmap(t_sub, 1:K, X_sub;
    xlabel = "time",
    ylabel = "slow index k",
    colorbar_title = "X_k",
    legend = false,
    size = (1000, 450),
    aspect_ratio = :auto)
