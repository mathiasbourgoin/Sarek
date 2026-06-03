# PR-5b Implementation Report — Sarek_transpile + FFI-free Proof

**Branch:** `phase0b/pr5b-transpile`
**Date:** 2026-06-03
**Status:** COMPLETE — all hard gates passed

## Commits

| Hash | Description |
|------|-------------|
| `cf31d706` | feat(transpile): add pure sarek_transpile library (step 1) |
| `48eb2240` | fix(registry): add Sarek_stdlib_meta path aliases in pure registry (step 2) |
| `247db175` | test(transpile): add FFI-free proof exe (step 4) |
| `69eed737` | style(transpile): ocamlformat 0.28.1 pass + opam regeneration (step 6) |

## New Files

- `sarek/transpile/Sarek_transpile.ml` — pure pipeline orchestrator
- `sarek/transpile/Sarek_transpile.mli` — public interface with docstrings
- `sarek/transpile/Sarek_ir_conv.ml` — `Sarek_ir_ppx.kernel → Sarek_ir_types.kernel` converter
- `sarek/transpile/dune` — `sarek_transpile` library (no spoc_core/ctypes)
- `sarek/transpile/test/test_transpile_proof.ml` — FFI-free proof exe
- `sarek/transpile/test/dune` — `(modes byte exe)` proof exe

## Modified Files

- `spoc/ir/Sarek_pure_registry.ml` — added `Sarek_stdlib_meta.*` path aliases

## `of_source` Design

The `of_source : backend -> string -> (string, error) result` function:

1. Calls `Sarek_stdlib_meta.force_init()` — idempotent, registers stdlib intrinsics
2. Sets `current_framework` in all 4 codegen modules for correct device names
3. Parses the source string via `Ppxlib.Parse.expression` (avoids compiler-libs dependency)
4. Converts to `Sarek_ast.kernel` via `Sarek_parse.parse_payload`
5. Rejects `[%native]` nodes (walks AST, returns `Unsupported_native`)
6. Types via `Sarek_env.(empty |> with_stdlib)` + `Sarek_typer.infer_kernel`
7. Checks convergence, monomorphizes, tail-rec transforms, lowers to IR
8. Converts `Sarek_ir_ppx.kernel` → `Sarek_ir_types.kernel` via `Sarek_ir_conv.conv_kernel`
9. Emits GPU source via `Sarek_ir_cuda.generate` / `Sarek_ir_opencl.generate` etc.

All exceptions caught and converted to structured `error` variants.

### Key: kernel source syntax

`Ppxlib.Parse.expression` parses bare OCaml expressions. The `.[]` vector syntax
(`b.[i] <- x`) is NOT valid as a standalone expression in OCaml 5.4 — it fails
parsing. The correct form is `b.(i) <- x` which the OCaml parser desugars to
`Array.set b i x` (Pexp_apply), which `Sarek_parse` already handles. The proof
uses `b.(i) <- Float32.sin a.(i)`.

## Path Resolution Fix

The integration risk was confirmed: `Sarek_stdlib_meta.Float32.sin` registers in
`Sarek_ppx_registry` under `ii_module = ["Sarek_stdlib_meta"; "Float32"]`. The
lower_kernel pass preserves this path verbatim in `Ir.EIntrinsic`. The pure
registry only had `["Float32"]` registered.

**Fix:** Added alias registrations for `["Sarek_stdlib_meta"; "Float32"]`,
`["Sarek_stdlib_meta"; "Float64"]`, `["Sarek_stdlib_meta"; "Math"; "Float32"]`,
and `["Sarek_stdlib_meta"; "Math"; "Float64"]` in `Sarek_pure_registry.ml`.

## IR Type Bridge

`Sarek_lower_ir` lowers to `Sarek_ir_ppx.kernel` (PPX compile-time types).
The codegen generators expect `Sarek_ir_types.kernel` (runtime types). These
are nominally separate modules with identical structure (only `kern_native_fn`
differs: `unit option` vs `native_fn_t option`).

`Sarek_ir_conv.conv_kernel` provides the mechanical conversion. SNative nodes
raise `Invalid_argument` (unreachable since `[%native]` is rejected before lowering).

## dune Libraries (FFI-free proof)

```
; sarek_transpile library
(libraries ppxlib sarek_frontend sarek_codegen sarek_stdlib_meta)

; proof exe
(libraries sarek_transpile sarek_stdlib_meta)
(modes byte exe)
```

No `spoc_core`, no `ctypes`, no `nvrtc`. The native exe links only:
- `libzstd.so.1`, `libm.so.6`, `libc.so.6`

## Proof Exe Output

```
=== PR-5b proof: FFI-free Float32.sin transpile ===
[INFO] CUDA output:
extern "C" {
__global__ void sarek_kern(float* __restrict__ a, int sarek_a_length, float* __restrict__ b, int sarek_b_length) {
  int i = (threadIdx.x + blockIdx.x * blockDim.x);
  b[i] = sinf(a[i]);
}
}
[PASS] CUDA contains "sinf("
[PASS] CUDA: Float32.sin -> sinf()
[PASS] OpenCL contains "sin("
[PASS] OpenCL: Float32.sin -> sin()
[PASS] Metal contains "sin("
[PASS] Metal: Float32.sin -> sin()
[PASS] GLSL contains "sin("
[PASS] GLSL: Float32.sin -> sin()
[PASS] [%native] kernel correctly rejected with Unsupported_native
[PASS] CUDA does NOT contain " sin(" (correct)

=== PASS: all 4 backends transpile Float32.sin FFI-free ===
```

## Bytecode/jsoo Result

- **Bytecode:** `test_transpile_proof.bc` builds and passes (FFI-free, hard gate PASSED)
- **jsoo:** `js_of_ocaml` NOT installed in this switch — deferred, documented here

## Golden Tests

`git diff origin/main -- sarek/tests/codegen_golden/` is empty (1 byte = trailing newline only).
All 20 golden test cases pass byte-identically.

## Float64 Escalation

`Float64` is in `Sarek_float64` (separate lib with FFI). The proof uses only `Float32`,
and `sarek_stdlib_meta` covers `Float32/Int32/Int64/Math/Gpu`. Float64 FFI-free coverage
would require a `sarek_float64_meta` analog. **Tracking as follow-up** — does not affect
this PR since the hard proof uses Float32.

## Gate Results

| Gate | Result |
|------|--------|
| `dune build @sarek/tests/runtest` (goldens) | PASS (byte-identical) |
| `dune build sarek_transpile` | PASS (pure, no FFI) |
| `dune exec sarek/transpile/test/test_transpile_proof.exe` | PASS (all 4 backends + [%native] rejection) |
| `dune build sarek/transpile/test/test_transpile_proof.bc` | PASS (bytecode FFI-free) |
| `dune build` (full) | PASS (only -lnvrtc pre-existing fail) |
| `dune build @fmt` | PASS (clean) |
