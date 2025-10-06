# AGENTS: Minimal Necessary Improvement Guide for FastSDE

This document enumerates ONLY the strictly necessary improvements to ensure correctness, robustness, and performance stability. It intentionally excludes optional nice-to-have refactors or feature expansions.

---
## 1. Correctness & API Clarity

1. **Document batched diffusion expectations**  
   In `ensemble_batched.jl`, the in-place batched diffusion path (`sigma_inplace && sigma isa Function`) assumes signature `sigma!(Xi, U, p, t)` where `Xi` is pre-filled with N(0,1) samples and must be transformed in-place to `Σ^{1/2}(U,t)*Ξ`. This is not stated in README.  
   Action: Add a short "Batched diffusion (in-place)" subsection to README describing required signature and that callable (returning) sigma forms are not yet supported in batched mode unless they are scalar / vector / constant matrix.

2. **Clarify RK + EM order statement**  
   Users may assume RK4 + EM implies higher strong/weak order for SDE noise. Current method applies deterministic RK4 + Euler–Maruyama noise (strong order 0.5).  
   Action: Add a note in README under time-steppers specifying stochastic order is still that of EM unless specialized SDE integrators are added.

---
## 2. Latent Edge / Failure Cases

1. **Matrix diffusion positiveness (batched path)**  
   `_evolve_ens_batched` performs `cholesky(Symmetric(sigma))` without guarding for failure.  
   Action: Wrap in `try`/`catch` and rethrow with message: "Matrix diffusion must be SPD for batched path; got ..." for clearer diagnostics.

2. **Boundary reset branch: bitwise vs boolean OR**  
   In `ensemble_batched.jl` boundary check uses `(U[i,j] < lo) | (U[i,j] > hi)`; using `||` avoids accidental scalar promotion issues (minor but correctness-aligned).  
   Action: Replace `|` with `||` in boundary predicate.

3. **`resolution` validation**  
   Silent acceptance of `resolution <= 0` would produce divide-by-zero or logic errors if misused.  
   Action: At public API entry (`evolve`, `evolve_ens`) add: `resolution < 1 && throw(ArgumentError("resolution must be ≥ 1"))`.

4. **Negative or zero `dt`**  
   Currently not validated; a zero dt causes stagnant state with potential division assumptions downstream (future code).  
   Action: Validate `dt > 0` in public API.

---
## 3. Performance Stabilizers (Necessary)

1. **Avoid repeated `sqrt(dt)`**  
   In batched step functions, `sqrt_dt = sqrt(dt)` recomputed each call.  
   Action: Hoist `sqrt_dt` outside loop in `_evolve_ens_batched` and pass as argument to `_batched_step!`.

2. **Consistent BLAS threading intent**  
   Batched path saves and restores BLAS thread count but calls `BLAS.set_num_threads(old_blas)` without change—redundant call per run.  
   Action: Remove redundant set or implement a policy (e.g., force 1 thread when `n_ens` very large). Minimal necessary: skip resetting if unchanged.

3. **Dynamic path BLAS threshold constant**  
   Hard-coded length check `length(u) >= 256`. This magic number should be a tunable constant for portability.  
   Action: Introduce `const _BLAS_SWITCH_THRESHOLD = 256` near top of `integrators_dynamic.jl`, replace literals to centralize tuning.

---
## 4. Testing Gaps (Necessary)

1. **In-place sigma! coverage**  
   No test exercises vector or matrix in-place diffusion paths (`sigma!(vec, u, t)` / `sigma!(mat, u, p, t)`). Missing coverage risks regressions.  
   Action: Add tests creating custom `sigma!` that scales by state and confirms evolution changes relative to constant sigma.

2. **Batched RK2 / RK4**  
   Batched ensemble tests only default (Euler).  
   Action: Add short tests ensuring shapes and determinism for `timestepper=:rk2` and `:rk4` in batched mode.

3. **Input validation tests**  
   After adding validation (resolution, dt, SPD), add tests asserting errors thrown.

---
## 5. Minimal Code Changes Summary

Implement ONLY these edits:
- Add validation in `evolve`, `evolve_ens` for `dt > 0` and `resolution ≥ 1`.
- Add `_BLAS_SWITCH_THRESHOLD` constant; replace `256` occurrences in dynamic steppers.
- Hoist `sqrt_dt` out of batched loop; pass to `_batched_step!` dispatches.
- Change boundary reset condition `|` to `||` in batched path.
- Wrap Cholesky in `try/catch` with clear error.
- Add README notes: in-place batched diffusion contract; stochastic order note.
- Add tests: in-place sigma! (static + dynamic), batched rk2/rk4, validation error cases, SPD failure case.

No other refactors or feature additions are required for baseline reliability.

---
## 6. Ordering for Implementation

1. Add constant & replace literals (`_BLAS_SWITCH_THRESHOLD`).
2. Add input validation guards in public API.
3. Update dynamic integrator (replace 256) and batched path (`sqrt_dt` hoist, boundary OR, cholesky try/catch).
4. Update README text sections.
5. Add minimal tests (do not expand beyond specified scope).

---
## 7. Acceptance Criteria Checklist

- All existing tests pass unmodified (except newly added).  
- New tests cover added behaviors (in-place sigma!, batched rk2/rk4, validation, SPD failure).  
- Benchmark performance unchanged or improved (micro-delta from removed repeated sqrt).  
- Documentation accurately states diffusion & order semantics.  
- No new allocations introduced in hot loops (verify via `@btime` spot checks).  

---
## 8. Post-Implementation Optional (EXCLUDED from this pass)
These are intentionally deferred: adaptive dt, GPU examples, callback support, separate RNG per batched column, advanced SDE integrators.

---
End of necessary improvements guide.
