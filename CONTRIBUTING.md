# Contributing to FastSDE.jl

Thank you for your interest in contributing to FastSDE.jl! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and collaborative environment.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the behavior
- Expected behavior vs actual behavior
- Julia version and FastSDE.jl version
- Minimal reproducible example (if applicable)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue describing:
- The enhancement and its motivation
- How it would be used
- Any potential implementation approaches

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the code style guidelines below
3. **Add tests** for any new functionality
4. **Update documentation** including docstrings and README if needed
5. **Run tests** to ensure everything passes
6. **Create a pull request** with a clear description of changes

## Development Setup

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FastSDE.jl.git
cd FastSDE.jl

# Start Julia in the project directory
julia --project=.

# Instantiate dependencies
using Pkg
Pkg.instantiate()

# Run tests
Pkg.test()
```

### Running Tests

```julia
using Pkg
Pkg.activate(".")
Pkg.test()
```

Or from the command line:
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

### Running Benchmarks

```julia
cd benchmarks
julia --project=. compare_sde.jl
```

## Code Style Guidelines

### General Principles
- **Clarity over cleverness**: Write code that is easy to understand
- **Type stability**: Avoid type instabilities in hot paths
- **Memory efficiency**: Minimize allocations in inner loops
- **Documentation**: All public functions must have docstrings

### Naming Conventions
- **Functions**: `lowercase_with_underscores` or `camelCase` (prefer underscores)
- **Types**: `CapitalizedCamelCase`
- **Constants**: `UPPERCASE_WITH_UNDERSCORES`
- **Internal functions**: prefix with `_` (e.g., `_internal_helper`)

### Code Formatting
- **Indentation**: 4 spaces (no tabs)
- **Line length**: Aim for ≤92 characters, but can exceed for readability
- **Blank lines**: Use blank lines to separate logical blocks

### Performance Guidelines
- Use `@inbounds` for bounds-checked operations in hot loops (only when safe)
- Use `@simd` for vectorizable loops
- Prefer in-place operations (`!` suffix) for mutating functions
- Preallocate arrays outside loops when possible
- Use StaticArrays for small, fixed-size arrays

### Documentation
All exported functions must include:
```julia
"""
    function_name(arg1, arg2; kwarg1=default)

Brief one-line description.

# Arguments
- `arg1::Type`: Description
- `arg2::Type`: Description

# Keyword Arguments
- `kwarg1=default`: Description

# Returns
- `Type`: Description of return value

# Examples
```julia
# Provide a complete, runnable example
result = function_name(1.0, 2.0; kwarg1=3.0)
```

# Extended Help
Additional details, algorithm description, references, etc.
"""
```

## Testing Guidelines

### Test Organization
- Tests are in `test/runtests.jl` and `test/batched_ensemble.jl`
- Use `@testset` to organize related tests
- Include both correctness and edge-case tests

### What to Test
- **Correctness**: Does it compute the right answer?
- **Edge cases**: Empty arrays, single elements, extreme values
- **Error handling**: Do bad inputs throw appropriate errors?
- **Type stability**: Use `@inferred` for critical paths
- **Performance**: Add benchmarks for performance-critical code

### Example Test
```julia
@testset "New feature" begin
    @test new_function(1.0, 2.0) ≈ 3.0
    @test_throws ArgumentError new_function(-1.0, 2.0)
    @test size(new_function_output) == (10, 100)
end
```

## Commit Guidelines

### Commit Messages
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line: concise summary (≤50 characters)
- Blank line, then detailed explanation if needed

### Good Commit Messages
```
Add memory usage warning for large trajectories

- Estimate memory usage before allocation
- Warn user if exceeds 4GB
- Suggest increasing resolution parameter
```

## Pull Request Process

1. **Update CHANGELOG.md** with your changes under [Unreleased]
2. **Update documentation** if you changed APIs or added features
3. **Ensure all tests pass** on your local machine
4. **Request review** from maintainers
5. **Address feedback** and update your PR as needed

### PR Checklist
- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] All tests pass locally
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No unnecessary changes or files included

## Questions?

If you have questions about contributing, please open an issue with the "question" label, or reach out to the maintainers.

## License

By contributing to FastSDE.jl, you agree that your contributions will be licensed under the MIT License.

---

Thank you for making FastSDE.jl better! 🚀
