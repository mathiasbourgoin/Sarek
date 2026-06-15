## Unreleased

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
