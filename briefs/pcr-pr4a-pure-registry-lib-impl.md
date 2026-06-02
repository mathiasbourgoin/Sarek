# PR-4a Implementation Report — pure-registry-lib

**Branch:** phase0b/pr4a-pure-registry-lib
**Commits:** 0362dc20, 6bafb83d

## Files Modified

| File | Change |
|---|---|
| `spoc/registry/Sarek_registry.ml` | Retyped `ti_device`/`fi_device` to `string -> string`; `fun_device_template` passes `"generic"` directly; `cuda_or_opencl` takes `(framework : string)` |
| `spoc/registry/dune` | Removed `spoc_framework` from `(libraries)` |
| `spoc/registry/test/test_sarek_registry.ml` | Replaced `Device_type.t` construction with framework strings in `test_type_device_code`, `test_fun_device_code`, `test_cuda_or_opencl` |
| `sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml` | Changed generated type annotations from `Spoc_core.Device.t -> string` to `string -> string` (both binding and ref) |
| `sarek/Sarek_stdlib/Gpu.ml` | Changed `dev` helper from explicit `Device.t` match to `(framework : string)` match, preserving CUDA/_ semantics |

## Decisions

- **Gpu.ml not in brief's caller list** — Gpu.ml had its own `dev` helper with `d.framework` field access, requiring an update. Changed to `(framework : string)` with identical match arms (`"CUDA" -> cuda | _ -> opencl`), not the canonical `cuda_or_opencl`, to preserve the original "non-CUDA defaults to opencl" behavior for Gpu intrinsics.
- **Sarek_ppx_intrinsic not in brief's scope** — The PPX generates type annotations for device closures. Without updating these from `Spoc_core.Device.t -> string` to `string -> string`, the stdlib callers fail to typecheck. This is a necessary co-change, not a scope expansion.

## Gate Results

| Gate | Result |
|---|---|
| `dune build @sarek/tests/runtest` | PASS (20/20 golden, all unit tests pass) |
| `git diff origin/main -- sarek/tests/codegen_golden/` | EMPTY (byte-identical) |
| `dune build @spoc/registry/runtest` | PASS (17/17 tests) |
| `dune build` | Only pre-existing `-lnvrtc` link error |
| `dune build @sarek-vulkan/all` | PASS |
| `test_math_intrinsics --interpreter` | PASS |
| `ocamlformat --check` on changed files | PASS |
| `git grep Spoc_framework\|Device_type spoc/registry/Sarek_registry.ml` | EMPTY |
| `spoc/registry/dune (libraries …)` | `(libraries)` — no spoc_framework |

## Residual Risks

- Metal/CUDA backends not available in this environment — verified OpenCL/Vulkan/Native/Interpreter paths only.
- The pre-existing uncommitted changes in `sarek/ppx/Sarek_ppx_registry.ml`, `sarek/ppx/Sarek_quote.ml`, and `sarek/tests/unit/test_*.ml` were left unstaged as they belong to earlier work not covered by this PR.
