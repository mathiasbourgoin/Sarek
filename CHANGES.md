## 2026-07

### Added

- Direct PTX emitter completion: records/variants with aligned C-ABI layout,
  match, helper-function inlining, static + dynamic `.shared`, `.local`
  arrays, full atomic set (CAS, wrapping inc/dec, 64-bit add/exch), f64
  softmath (trig, exp/log family, hypot, …), scalar cast matrix
- `Sarek_worklist` — portable dynamic-parallelism (work-queue) library
- `Sarek_gemm` — portable tiled SGEMM library for shared-memory backends
- `Sarek_df64` — float-float extended-precision arithmetic in pure Sarek
- Portable float64: `G`-suffix float64 literals, `sarek.real64` host
  plumbing, complete interpreter float64 math oracle
- Tier-0 defunctionalization pass (first-class kernel-local functions)
- Tuple-typed vectors on device and Native paths (shared shape registry)
- Static tag erasure for kernel-local variant slots
- Aligned C-ABI aggregate host layout (L8), with the `PtxLayout.v` formal
  model restated for the aligned ABI (0 admits)
- float32 `fma` intrinsic on all backends; GLSL `precise` qualifier on
  float locals
- ZLUDA/AMD support for the CUDA/PTX backend (PTX launch ABI fix)
- T3-SEMANTIC milestone lock for both formal projects; conformance +
  mutation tests wired into `dune runtest`

### Fixed

- Indexed kernel-argument container with strict launch validation, honored
  across all six backends (out-of-order/sparse `set_arg` now correct)
- Unambiguous, collision-resistant compile-cache keys (kernel name included,
  digest-per-field); CUDA kernel-cache eviction on context destroy
- e2e test alias actually executes tests; vacuous tests now assert real
  results; negative suite runs in CI
- Vulkan push-constant scalar binding by logical index; 64-bit write
  alignment; buffer-index validation
- Benchmark dashboard HTML escaping at all sinks (stored-XSS hardening)
- CPU data kept authoritative after Interpreter runs

## 2026-06

### Added

- CUDA backend split into CUDA/PTX and CUDA/C devices (PTX is the default)
- `.shared` (DShared) emission in the PTX backend with formal spec
  (`specs/ptx-dshared-formal.md`)
- T3-SEMANTIC theorem work for the convergence-safety and type-safety
  formal projects
- WGSL/WebGPU codegen backend, in-browser playground and Learn course

### Changed

- Breaking: removed first-party `Obj` escape hatches from Sarek/SPOC execution
  paths.
  - Kernel vector arguments now cross framework/plugin boundaries through typed
    existential accessors and runtime type witnesses.
  - Custom value conversion now uses typed helper lookup instead of raw runtime
    representation casts.
  - Native and interpreter plugin buffer copies use typed Bigarray operations.
  - Custom shared-memory arrays are keyed by typed witnesses to reject
    mismatched reuse.
  - Legacy native direct execution now accepts typed `Framework_sig.exec_arg`
    arrays.

## 2026-01

### Changed

- Documentation cleanup and modernization
  - Comprehensive README rewrite with clear SPOC/Sarek distinction
  - Added sarek/ directory navigation guide
  - Created CONTRIBUTING.md with project guidelines
  - Removed pretentious language ("Grade A", "100%", etc.)
  - Added AI assistance acknowledgment
- Code quality improvements
  - Eliminated 49 failwith calls from Native and Interpreter plugins
  - Added structured error handling using Backend_error pattern
  - Created test suites for Native and Interpreter plugins
  - Added READMEs for Native, Interpreter, ppx_intrinsic, Sarek_float64
- CI/CD modernization
  - Added unit test execution (dune runtest)
  - Created fast benchmark suite (~20s) for CI
  - Integrated coverage measurement with bisect_ppx
  - Simplified workflow to single build job
- Repository cleanup
  - Removed AI artifact files (AGENTS.md, etc.) from history
  - Removed tracked build artifacts (*.exe, *.log, etc.)
  - Updated .gitignore for better hygiene

## 2025-12

### Added

- Full PPX rewrite replacing Camlp4
  - PPX-based kernel syntax ([%kernel ...], [%ktype ...])
  - Type registry for cross-module GPU types
  - Intrinsic registration system (Sarek_stdlib)
  - Float32 stdlib module with math intrinsics
  - Pragma support for loop unrolling hints
- Kernel fusion system (Sarek_fusion.ml)
  - Vertical fusion for producer-consumer patterns
  - Reduction fusion (map + reduce)
  - Stencil fusion with radius tracking
  - Auto-fusion heuristics
- Clean intermediate representation (Sarek_ir.ml)
- BSP superstep syntax with barrier synchronization
- Warp convergence tracking (Sarek_convergence.ml)
- Core primitives with convergence information
- Comprehensive test suite (unit, negative, e2e tests)

### Changed

- Removed Camlp4 dependency entirely
- Improved type inference for kernel parameters
- Better compile-time error messages

## 2024-2025

### Changed

- GPU Backend Overhaul (all 4 backends: CUDA, OpenCL, Vulkan, Metal)
  - Eliminated all failwith calls (94 total removed)
  - Implemented Backend_error.Make functor pattern for structured errors
  - Refactored code generators for maintainability
  - Added comprehensive unit test suites (19-20 tests per backend)
  - Created professional documentation (9-26KB per backend)
  - Improved error messages with context
- OCaml 5.x migration
  - Updated to OCaml 5.4.0
  - Migrated from Domains to Effect handlers where appropriate
  - Maintained backward compatibility

## 20210823

- Add PPX extension to declare external GPGPU kernels
- Update Samples to use PPX instead of Camlp4 extension
- Update dune/opam files for opam release

## 20210816

### Added

- Build with dune
- Compatible with OCaml 4.12
- Switch to github actions for CI
