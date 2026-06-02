# Reviewer Sub-brief — pure-codegen-rollout PR-4b (extract sarek_codegen)

**Status:** VALIDATED
**Parent:** `briefs/pure-codegen-rollout-intake.md` (Phase 0B, PR-4 of 5, part b)

## What was implemented
The 4 generators (`Sarek_ir_{cuda,opencl,metal,glsl}`) moved from their ctypes backend packages
into a new pure `sarek_codegen` lib (deps `sarek_ir sarek_registry sarek_backend_error`). Dead
`generate_for_device` removed (last `Spoc_core` ref); `<Backend>_error` swapped for a local
`Backend_error.Make`; each backend re-exports `Sarek_ir_<b>` (`include Sarek_codegen.Sarek_ir_<b>`).

## Load-bearing GO/NO-GO hinges (verify first)
1. **`sarek_codegen` is FFI-free.** Read its dune `(libraries …)` — must be exactly
   `sarek_ir sarek_registry sarek_backend_error`, NO `ctypes`/`ctypes.foreign`/`spoc_core`/
   `spoc_framework`. `git grep -n 'Spoc_core\|Ctypes\|Spoc_framework' <the 4 generator sources>`
   must be empty. `dune build sarek_codegen` succeeds standalone. If FFI leaked → NO-GO.
2. **Byte-identical goldens.** `git diff origin/main -- sarek/tests/codegen_golden/` EMPTY. Any diff → NO-GO.

## Verify
- **Consumers unchanged & build:** `Cuda/Opencl/Metal_plugin.ml`, `Vulkan_plugin.ml` reference
  `Sarek_ir_<b>.generate{,_with_types}` unqualified — confirm those call sites are UNEDITED and
  resolve via the re-export. `benchmarks/generate_backend_code.ml` uses `Sarek_cuda.Sarek_ir_cuda.generate`
  — confirm the wrapped path still resolves. Full `dune build` (only pre-existing `-lnvrtc` acceptable).
- **`generate_for_device` removal is behavior-neutral:** confirm it had no caller and the SNative
  branch behavior is unchanged (nothing sets `current_framework` to `Some`, so the error path is
  as before). The `current_framework`/`current_variants` refs and SNative match must be byte-identical.
- **Error swap is mechanical:** the local `Backend_error.Make(name="CUDA"|…)` produces the same
  `unknown_intrinsic`/`unsupported_construct`/`invalid_arg_count`/`raise_error` and raises the same
  shared `Backend_error` exception. No error message text changed (goldens don't trigger errors, so
  check by reading: the `~backend` name strings match the old `<Backend>_error` names).
- **Re-export correctness:** each backend's `Sarek_ir_<b>.ml` re-export exposes the same surface
  (`generate`, `generate_with_types`, `current_framework`, `current_variants`, etc.) the golden
  shims (`gen_*.real.ml`) and plugins use. Golden shims build and run on all 4 backends.
- **Vulkan e2e:** `test_float32_sin_pure --vulkan` PASSES (RX 7900 XTX) — proves the moved GLSL gen
  + re-export still produce a runnable shader.
- **No scope creep** — no `Sarek_transpile`, no generator logic change beyond delete-dead-fn /
  swap-error-module / move-files.

Return GO/NO-GO + findings. NO-GO if: `sarek_codegen` pulls any FFI dep, OR a golden changed, OR a
consumer needed a source edit to its `Sarek_ir_<b>` call sites (means the re-export is incomplete),
OR a non-pre-existing build failure, OR the Vulkan e2e regressed.
