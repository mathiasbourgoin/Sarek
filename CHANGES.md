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
- `docs/fp-contraction-policy.md` — cross-backend floating-point contraction
  policy: what each backend may contract, what actually prevents it, and
  whether that mechanism is verified or merely believed. The interpreter is
  named as the cross-backend oracle. Linked from every site that previously
  carried an ad-hoc contraction comment.
- FP-conformance guard on the CUDA/nvrtc path: `-use_fast_math`, `-ftz=true`,
  `--prec-div=false` and `--prec-sqrt=false` are now REJECTED at the point an
  option array reaches `nvrtcCompileProgram` (`--fmad=true` warns). They flush
  binary32 subnormals or downgrade div/sqrt, and no later flag undoes that.
  `sarek-cuda/test/test_cuda_fp_conformance.ml` reproduces the hazard
  host-side (`-ftz=true` turns `FMUL`/`FADD` into `FMUL.FTZ`/`FADD.FTZ` at
  sm_90, CUDA 13.3) and each of its four cases was proved red by mutation.
- ZLUDA/AMD support for the CUDA/PTX backend (PTX launch ABI fix)
- T3-SEMANTIC milestone lock for both formal projects; conformance +
  mutation tests wired into `dune runtest`

### Changed

- The CUDA branch of `sarek_f32_barrier` no longer emits
  `asm volatile("" : "+f"(x))`. That barrier was inert: the assembly template
  is empty, NVVM erases it, and the cubins are byte-identical with and without
  it — re-measured on CUDA 13.3 for sm_75 through sm_121. What keeps the f32
  multiply out of the f16 narrowing on NVIDIA is `ptxas`, machine-checked by
  `test_cuda_f16_sass`. The AMDGPU `"+v"` barrier, which IS load-bearing, is
  unchanged. Removing a no-op that read as protection; no behaviour change.

### Fixed

- `Sarek_df64` silently ran at plain float32 precision on real NVIDIA
  hardware (CUDA/PTX and NVIDIA OpenCL): `ptxas` contracted the multiply in
  `two_prod` into the `add`/`sub` of the `quick_two_sum` closing `df64_mul`,
  rebuilding the exact product and cancelling the TwoProd error term.
  `mul`/`div`/`sqrt` degraded from ~2^-47 to ~2^-24 with no error reported.
  `two_prod` now forms its product with `fma a b 0.0`, which cannot be fused
  again. Measured on a GTX 1070 (sm_61, CUDA 12.9): mul 5.92e-08 → 9.07e-15,
  div 5.64e-08 → 5.08e-15, throughput unchanged. A CPU-only regression guard
  (`test_df64_no_contraction`) asserts the emitted PTX contains no
  contractable `mul.f32`. The df64 per-backend precision table now names the
  device and toolchain behind every figure — the previous table generalised
  AMD-only measurements to "CUDA/PTX", which is why this went unseen. NOTE
  `df64_sqrt` on NVIDIA remains above tolerance; see the `KNOWN RESIDUAL`
  block in `sarek/Sarek_df64/Sarek_df64.ml`.
- df64 precision gates could not distinguish a working df64 from a collapsed
  one. `test_df64`, `test_real64` and `test_real64_single_source` all widened
  their tolerance to `0x1p-22` (2.38e-07 — four times the float32 unit
  roundoff) on the backends with a documented deviation, so a df64 that had
  degraded all the way to plain float32 (measured 5.84e-08 on RADV) and one
  meeting its contract (9.07e-15) both read PASS. The widening also keyed on
  the bare `Vulkan` framework tag, sweeping in NVIDIA Vulkan, which meets the
  full contract. All three tests now hold every device to the derived contract
  bound (2^-47 add/sub, 2^-46 mul/div/sqrt) and express the documented
  deviations as an explicit expected-failure band — upper end 2^-23, twice the
  float32 unit roundoff — keyed on driver identity (Mesa RADV, Mesa ANV) in
  `Test_helpers.df64_known_deviation`. Degrading past plain float32 now FAILs
  even on an allowlisted device, and so does an allowlisted device that starts
  MEETING the contract (strict XPASS, as in pytest's `xfail(strict=True)`) —
  the run goes red naming the match arm to delete, so the allowlist cannot rot
  behind a green exit code. Red-proved twice: replacing `df64_mul` with a plain
  float32 multiply (old gate PASS on both RADV devices and Native at 1.43e-07
  vs tol 2.38e-07, new gate FAIL), and adding a bogus allowlist entry for an op
  that already meets the contract (all three tests exit 1 with the stale-entry
  message).
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
