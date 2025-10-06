"""
Type definitions and unions for FastSDE.jl
"""

# Parameter types - more specific than Any for type stability
const ParamsType = Union{Nothing, NamedTuple, AbstractDict}

# Boundary specification
const BoundaryType = Union{Nothing, Tuple{Real,Real}}

# Time stepper options
const TimestepperType = Symbol  # :euler, :rk2, or :rk4
