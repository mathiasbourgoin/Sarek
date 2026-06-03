# Reviewer Sub-brief — pure-codegen-rollout PR-5b (Sarek_transpile + FFI-free proof)

**Status:** VALIDATED
**Parent:** `briefs/pure-codegen-rollout-intake.md` (Phase 0B, PR-5 of 5, part b) — FINAL

## What was implemented
New pure `sarek_transpile` lib with `of_source : backend -> string -> (string, error) result`
(parse string → frontend → IR → `sarek_codegen` emit), rejecting `[%native]` with a structured
error; stdlib-intrinsic path resolution fixed so transpiled `Float32.sin` codegens correctly; a
proof exe transpiling a `Float32.sin` kernel FFI-free to all 4 backends; a bytecode (and attempted
jsoo) compile gate.

## Load-bearing GO/NO-GO hinges
1. **FFI-free, stdlib-intrinsic transpile WORKS.** Run the proof exe — it must transpile a
   `Float32.sin` kernel and assert CUDA output has `sinf(`, OpenCL/Metal/GLSL have `sin(`. Read the
   proof's dune `(libraries …)` and confirm its link closure has NO `spoc_core`/`ctypes` (only
   sarek_transpile→frontend/codegen/stdlib_meta/ppxlib). If it pulls FFI, or the intrinsic doesn't
   resolve (wrong/empty emit), NO-GO — this is the entire deliverable.
2. **Bytecode FFI-free compile passes.** The byte-mode target builds with no ctypes. (jsoo is bonus —
   if it failed, confirm the implementer DOCUMENTED why and that bytecode-FFI-free still passes.)
3. **Byte-identical goldens.** `git diff origin/main -- sarek/tests/codegen_golden/` EMPTY (no
   generator change). → NO-GO if changed.

## Verify
- **`of_source` error handling is total:** no `Location.Error`/`Sarek_error`/exception escapes —
  parse errors, type errors, convergence errors, and `[%native]` all map to the structured `error`
  and `string_of_error`. Confirm the `[%native]`-rejection test passes (returns `Unsupported_native`,
  not a crash).
- **Path-resolution fix is sound:** if `map_stdlib_path` gained `Sarek_stdlib_meta` arms (or lowering
  was changed), confirm it routes identically to the existing `Sarek_stdlib`/short-path arms — and
  that this did NOT change any golden (the existing PPX path still emits the same code). Read the diff.
- **No generator/behavior change:** `sarek_codegen`, the 4 generators, and the existing PPX flow are
  untouched except the additive path-resolution arms. Native/interpreter + Vulkan e2e pass.
- **Float64:** confirm whether Float64 transpile is covered or escalated to a documented follow-up
  (the proof uses Float32). If escalated, that's acceptable per the brief; flag it as tracked.
- **No scope creep:** purely additive (new lib + proof + path arms); no refactor of merged work.

Return GO/NO-GO + findings. NO-GO if: the FFI-free stdlib-intrinsic transpile doesn't work or pulls
FFI, OR the bytecode target isn't FFI-free, OR a golden changed, OR an exception escapes `of_source`,
OR native/interpreter/Vulkan regressed.
