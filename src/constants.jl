"""
Constants used throughout FastSDE.jl
"""

# Threshold for switching to BLAS operations in dynamic path
const BLAS_SWITCH_THRESHOLD = 256

# Default dimension threshold for StaticArrays vs dynamic arrays
# Systems with length(u0) ≤ this value use StaticArrays
const DEFAULT_STATIC_THRESHOLD = 64

# Seed offset for ensemble members (ensures independent RNG streams)
const ENSEMBLE_SEED_OFFSET = 1000

# Default values for integration parameters
const DEFAULT_RESOLUTION = 1
const DEFAULT_SEED = 123
const DEFAULT_TIMESTEPPER = :euler
