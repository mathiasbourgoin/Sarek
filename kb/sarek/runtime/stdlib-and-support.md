# Standard Library And Support Libraries

## Component Inventory

Reviewed support files: `sarek/Sarek_stdlib/**`, `sarek/Sarek_float64/**`, `sarek/Sarek_geometry/**`, `sarek/Visibility_lib/**`, `sarek/sarek/Sarek_float32.ml`, `sarek/sarek/Sarek_float32.mli`, and related documentation.

## Per-File Purpose

- `sarek/Sarek_stdlib/README.md`: standard library overview and supported intrinsic families.
- `sarek/Sarek_stdlib/Gpu.ml`: GPU-oriented builtins, vector helpers, atomic helpers, math/power helpers, and backend code snippets.
- `sarek/Sarek_stdlib/Float32.ml`: float32 intrinsic definitions and backend snippets.
- `sarek/Sarek_stdlib/Int32.ml`: int32 intrinsic definitions and backend snippets.
- `sarek/Sarek_stdlib/Int64.ml`: int64 intrinsic definitions and backend snippets.
- `sarek/Sarek_stdlib/Math.ml`: math intrinsic wrappers.
- `sarek/Sarek_stdlib/Sarek_stdlib.ml`: stdlib package entrypoint.
- `sarek/Sarek_float64/README.md`: float64 support overview.
- `sarek/Sarek_float64/Float64.ml`: float64 intrinsic definitions and backend snippets.
- `sarek/Sarek_geometry/geometry_lib.ml`: geometry support package.
- `sarek/Visibility_lib/visibility_lib.ml`: visibility helper package.
- `sarek/sarek/Sarek_float32.ml` and `.mli`: runtime float32 helpers used by the Sarek library.

## Features/APIs

- Scalar math and type-specific operations for float32, float64, int32, and int64.
- GPU-style thread/block index helpers and memory helpers.
- Atomic add/sub/exchange/CAS-like helpers backed by host mutexes in CPU paths.
- Backend code fragments for CUDA/OpenCL-style code generation.
- Geometry and visibility package stubs/support modules.

## Invariants

- Atomic helpers must release locks even when vector operations fail.
- Backend intrinsic snippets must match the selected backend language and capabilities.
- Float64 operations should only be emitted to devices that support FP64.
- Power/exponent helpers must terminate for all accepted inputs or reject unsupported inputs.
- Runtime float32 helpers must agree with stdlib intrinsic semantics.

## Potential Invariant Violations/Bugs

- Atomic helpers are not exception-safe: `spoc_atomic_add`/related lock and unlock patterns at `sarek/Sarek_stdlib/Gpu.ml:168-179` and CAS-style code at `sarek/Sarek_stdlib/Gpu.ml:183-193` can leave `atomic_mutex` locked if `Vector.get` or `Vector.set` raises.
- `spoc_powint` can run indefinitely or for an impractically long time on negative exponents: it decrements `e` until `0l` at `sarek/Sarek_stdlib/Gpu.ml:235-245`. Uncertain: a typer or frontend may reject negative integer exponents before runtime.
- Backend snippet selection generally distinguishes CUDA from a generic "other" path, for example `Gpu.dev` at `sarek/Sarek_stdlib/Gpu.ml:19-20`. Float32/Int32/Int64/Float64 modules use similar `cuda_or_opencl` style selection. This is a maintainability risk as Vulkan, Metal, native, or interpreter backends grow.
- Float64 support registers double-precision snippets in `sarek/Sarek_float64/Float64.ml:23-24`, while device capability has `allows_fp64` in `sarek/core/Device.ml:157`. This pass did not find a clear runtime enforcement point before emission. Uncertain depending on backend validation outside this scope.

## Performance Or Maintainability Risks

- A single global mutex for atomics serializes all CPU atomic operations and can become a bottleneck.
- Backend code strings are distributed across stdlib modules, making backend feature evolution broad and error-prone.
- Capability checks are not colocated with intrinsic registration.
- Geometry and visibility support are very small packages; their intended runtime contract is not obvious from colocated tests.

## Related Tests

- `sarek/sarek/test/test_sarek_float32.ml`: float32 runtime helper tests.
- No focused tests were found in the scoped support-library directories for `Gpu.ml`, `Float64.ml`, geometry, or visibility behavior.

## Missing Tests

- Atomic helper exception safety, including vector bounds failures while the mutex is held.
- Negative exponent handling in `spoc_powint`.
- Float64 intrinsic use on a device with `allows_fp64 = false`.
- Backend snippet selection for non-CUDA/non-OpenCL backends.
- Consistency tests between `sarek/sarek/Sarek_float32.ml` and `sarek/Sarek_stdlib/Float32.ml`.
- Smoke tests for geometry and visibility library exports.

## Concrete Improvement/Fix Candidates

- Wrap each atomic critical section in `Fun.protect ~finally:(fun () -> Mutex.unlock atomic_mutex)`.
- Reject negative `spoc_powint` exponents or define reciprocal semantics where type-correct.
- Move backend snippet selection behind an explicit backend-language abstraction rather than CUDA-vs-other branching.
- Check `Device.allows_fp64` before accepting or emitting float64 intrinsics.
- Add support-library smoke tests and one runtime test per intrinsic family.
