# Implementation Brief — pure-codegen-extraction · Phase 0A (spike)

**Date:** 2026-06-02
**Mode:** full (gated spike)
**Status:** COMPLETED — all 5 steps; GO on the riskiest (Step 4) and Step 5.

## Modified / added files

| File | Change | Reason |
|---|---|---|
| `spoc/framework_error/{dune,Backend_error.ml}` | new pure lib `sarek_backend_error` (no ctypes) | Step 2 — extract error layer |
| `spoc/framework/Backend_error.ml` + `dune` | re-export stub → `Sarek_backend_error` | preserve compat |
| `sarek-{cuda,opencl,metal,vulkan}/{*_error.ml,dune}` | repoint to `Sarek_backend_error` | Step 2 |
| `sarek-cuda/Sarek_ir_cuda.ml` + `dune` | `current_device:Device.t` → `current_framework:string`; `generate_for_device` wrapper; drop `open Spoc_core` | Step 3 |
| `spoc/ir/Sarek_pure_registry.ml` + `dune` | new pure intrinsic registry (`framework:string->string`, no Device.t) | Step 4 |
| `sarek/tests/codegen_golden/*` (9 files) + `sarek/tests/dune` | golden-snapshot harness, 5 kernels × 4 backends = 20 byte-exact tests | Step 1 |
| `tools/transpile_smoke/*` (2 files) | `(modes byte)` FFI-free target: `sarek_ir + sarek_backend_error` only | Step 5 |

## Decisions / deviations
- Pure registry runs **in parallel** to the FFI registry (path-qualified intrinsics query it first) — minimal, non-breaking; full PPX dual-registration is 0B.
- Step 5 FFI-free surface proven = generator-emit logic against `sarek_ir + sarek_backend_error`; the CUDA *module* still lives in `sarek_cuda` (ctypes) — full generator extraction is 0B (documented).
- Golden capture mode via `GOLDEN_CAPTURE=1`; metal/cuda goldens captured locally, CI uses stub variants (determinism still exercised).

## Quality gates
- [x] `dune build @sarek/tests/runtest` ✅ (20 golden + all unit/e2e)
- [x] `dune build @sarek-vulkan/all` ✅
- [x] `dune build` ✅ (except pre-existing `new_runtime` `-lnvrtc` link — CUDA toolkit absent)
- [x] `ocamlformat --check` changed files ✅
- [x] `tools/transpile_smoke` links FFI-free + runs PASS ✅
- [x] Golden perturbation test (1-char change fails, revert restores) ✅

## GO/NO-GO verdict (the spike's purpose)
- **Step 4 (registry/stdlib decouple): GO** — a `Device.t`-free pure registry resolves
  `Float32.sin` and the CUDA generator emits byte-identical output via it.
- **Step 5 (FFI-free compile): GO** — the pure slice links with no ctypes/spoc_core.
- ⇒ Phase 0B (full rollout) is **feasible**.

## Points of attention for review
- Goldens must be genuinely byte-exact + deterministic (mutable refs reset).
- `Backend_error` extraction: pure lib ctypes-free; the exception identity preserved across
  the re-export stub (existing `Spoc_framework.Backend_error.Backend_error` catchers).
- CUDA `generate_for_device` wrapper must keep `SNative` output byte-identical.
- Confirm nothing outside 0A scope changed (no stdlib conversion, no lib split, no
  `Sarek_transpile`).

## Out-of-scope (0B, documented)
Full generator extraction to a pure lib; PPX dual-registration of intrinsics;
`Kirc_Ast.Native` decouple; opencl/metal/glsl `current_device` decouple; `sarek_ppx_lib`
split; `Sarek_transpile.of_source`.
