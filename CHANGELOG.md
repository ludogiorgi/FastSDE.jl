# Changelog

All notable changes to FastSDE.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- LICENSE file (MIT License)
- GitHub Actions CI/CD workflow for automated testing
- Type definitions (`ParamsType`, `BoundaryType`) for better type stability
- Constants module for shared configuration values
- Utilities module with input validation and memory estimation
- Comprehensive input validation with helpful error messages
- Memory usage warnings for large trajectory storage

### Changed
- Improved type stability by replacing `params::Any` with `ParamsType`
- Extracted BLAS threshold and other magic numbers to constants
- Standardized default parameter values across all functions
- Enhanced error messages with actionable guidance
- Updated ensemble seed offset to use named constant

### Fixed
- Type instabilities in parameter handling that could impact performance
- Inconsistent default values across different function signatures

## [0.1.0] - 2025-01-06

### Added
- Initial release of FastSDE.jl
- Euler, RK2, and RK4 time-stepping methods for SDEs
- Automatic dispatch between StaticArrays (small systems) and dynamic arrays (large systems)
- Thread-parallel ensemble integration with independent RNGs
- Batched drift evaluation for neural network integration
- Support for scalar, diagonal, and correlated noise
- Optional boundary reset functionality
- Configurable static array threshold
- In-place and returning diffusion function support
- Burn-in support for ensemble simulations

### Features
- Zero-allocation inner loops for optimal performance
- BLAS acceleration for large systems
- Per-trajectory RNG streams for reproducibility
- Flexible parameter passing to drift and diffusion functions
- Multiple noise types (additive, diagonal, correlated)
