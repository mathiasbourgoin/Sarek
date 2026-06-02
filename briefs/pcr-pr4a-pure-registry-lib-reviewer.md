# Reviewer Sub-brief — pure-codegen-rollout PR-4a (make sarek_registry pure)

**Status:** VALIDATED
**Parent:** `briefs/pure-codegen-rollout-intake.md` (Phase 0B, PR-4 of 5, part a)

## What was implemented
`sarek_registry` (`spoc/registry/Sarek_registry.ml`) device-codegen closures retyped
`Device_type.t -> string` → `string -> string` (framework name); `fun_device_template` now passes
`"generic"` instead of a fake `Device_type.t`; `cuda_or_opencl` matches a framework string; the 5
stdlib `dev` helpers + registry tests updated; `spoc_framework` dropped from `spoc/registry/dune`.

## Load-bearing GO/NO-GO hinges
1. **`sarek_registry` is spoc_framework-free.** Read `spoc/registry/dune` `(libraries …)` — must
   NOT contain `spoc_framework`. `git grep -n 'Spoc_framework\|Device_type' spoc/registry/Sarek_registry.ml`
   must return nothing. If either fails, NO-GO.
2. **Byte-identical goldens.** `git diff origin/main -- sarek/tests/codegen_golden/` must be EMPTY.
   This is the proof the line-384 generator query output is unchanged. Any diff → NO-GO.

## Verify
- **`fun_device_template` behavior preserved:** old code passed a `minimal_dev` with
  `framework="generic"`; new code passes `"generic"`. Confirm `cuda_or_opencl "generic"` →
  cuda_code branch (the `_` arm), i.e. identical result. Confirm the `cuda_or_opencl` arms are
  unchanged: `"OpenCL"`→opencl, `"CUDA"|"Native"|"Interpreter"|_`→cuda.
- **Registry unit tests** (`spoc/registry/test/test_sarek_registry.ml`) pass and their assertions
  (CUDA→cuda_code, OpenCL→opencl_code, Native→cuda_code) are preserved, not weakened.
- **5 stdlib callers** updated mechanically (signature only; body forwards the framework arg). No
  logic change. Native/interpreter intrinsic tests pass (`test_math_intrinsics --interpreter`).
- **Full build** — all consumers build (only pre-existing `-lnvrtc` acceptable). `@sarek-vulkan/all` builds.
- **No scope creep** — no generator edits, no `sarek_codegen`, no `Sarek_transpile`.

Return GO/NO-GO. NO-GO if: `sarek_registry` still pulls `spoc_framework`/`Device_type`, OR any
golden changed, OR a registry/native test regressed, OR a consumer fails to build (non-pre-existing).
