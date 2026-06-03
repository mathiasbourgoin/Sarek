# PR-5a Implementation Report — pure stdlib metadata lib

**Status:** COMPLETE  
**Branch:** `phase0b/pr5a-pure-stdlib-meta`  
**Date:** 2026-06-03

## Commits

1. `597abb5a` — `refactor(ppx_intrinsic): make ctype optional in type%sarek_intrinsic`
2. `6cf0e46e` — `feat(stdlib_meta): add pure sarek_stdlib_meta library (FFI-free)`
3. `777a4261` — `refactor(stdlib): add sarek_stdlib_meta as dependency of sarek_stdlib`
4. `fadf9307` — `test(stdlib_meta): add FFI-free proof exe for PR-5a`

## Mechanism

### Step 1 — PPX change (ctype optional)

`sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml` gains:

- `size_of_type_name` pure helper: float32=4, float64=8, int32=4, int64=8,
  bool=4, char=1, unit=0 (matching Ctypes.sizeof values exactly).
- `expand_sarek_intrinsic_type` now reads `ctype` as an optional field:
  - **Present (FFI path):** emits both `Sarek_registry.register_type` (JIT) and
    `Sarek_ppx_registry.register_type` using `Ctypes.sizeof ctype`. Unchanged
    behaviour for existing `sarek_stdlib`.
  - **Absent (pure path):** emits only `Sarek_ppx_registry.register_type` with
    derived size. No Ctypes dependency in generated code.
- Dispatcher updated: routes `Ppat_var` + record to the type handler regardless
  of whether `ctype` is present (previously required `ctype` for routing).

### Step 2 — New `sarek_stdlib_meta` library

`sarek/Sarek_stdlib_meta/` — new library with modules: Float32, Int32, Int64,
Math, Gpu, Sarek_stdlib_meta (wrapper). All use `%sarek_intrinsic` without
`ctype` field.

`dune (libraries …)` line: **`sarek_registry sarek_ppx_lib`** — no spoc_core,
no ctypes.

Special handling in Gpu.ml: `atomic_add_global_int32` and
`atomic_inc_global_int32` use `failwith` stubs for the `ocaml` field (the real
`Spoc_core.Vector` implementations remain in `sarek_stdlib.Gpu`).

### Step 3 — sarek_stdlib depends on sarek_stdlib_meta

`sarek/Sarek_stdlib/dune` `(libraries …)` now includes `sarek_stdlib_meta`.
The existing `%sarek_intrinsic` declarations with `ctype` remain unchanged and
produce duplicate-but-harmless `Hashtbl.replace` calls. Single-source is
deferred but the dependency DAG is correct.

### Step 4 — Proof exe

`sarek/tests/e2e/test_stdlib_meta_proof.ml` links `sarek_frontend +
sarek_stdlib_meta` only. Asserts 96 intrinsics registered, including:
Float32.sin, cos, sqrt, exp, add_float32; Int32.add_int32; Int64.add_int64;
Math.xor, pow; Gpu.thread_idx_x, block_idx_x, global_thread_id. Three types
with correct sizes: float32=4, int32=4, int64=8.

## Coverage

Full coverage: Float32, Int32, Int64, Math, Gpu — all modules in Sarek_stdlib.
Float64 (sarek_float64, separate library) was out of scope per the brief.

No escalation needed. The PPX change was contained (16 lines added, dispatcher
refactor straightforward).

## Gate Results

| Gate | Result |
|------|--------|
| `dune build @sarek/tests/runtest` | PASS — goldens byte-identical (`git diff origin/main -- sarek/tests/codegen_golden/` empty) |
| `dune build sarek/Sarek_stdlib_meta` | PASS — no ctypes/spoc_core in libraries |
| `sarek_stdlib_meta (libraries …)` | `sarek_registry sarek_ppx_lib` only |
| Proof exe: `test_stdlib_meta_proof.exe` | PASS — 96 intrinsics, 3 types, all FFI-free |
| `test_math_intrinsics --native` | PASS |
| `test_math_intrinsics --interpreter` | PASS |
| `dune build` (full) | PASS (only pre-existing `-lnvrtc` link error) |
| `dune build @sarek-vulkan/all` | PASS |
| ocamlformat check | PASS |

## Files Changed

- `sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml` — ctype-optional type expansion
- `sarek/Sarek_stdlib_meta/dune` — new (no spoc_core/ctypes in libraries)
- `sarek/Sarek_stdlib_meta/Float32.ml` — new (pure metadata)
- `sarek/Sarek_stdlib_meta/Int32.ml` — new (pure metadata)
- `sarek/Sarek_stdlib_meta/Int64.ml` — new (pure metadata)
- `sarek/Sarek_stdlib_meta/Math.ml` — new (pure metadata)
- `sarek/Sarek_stdlib_meta/Gpu.ml` — new (pure metadata; global atomic stubs)
- `sarek/Sarek_stdlib_meta/Sarek_stdlib_meta.ml` — new wrapper
- `sarek/Sarek_stdlib/dune` — added sarek_stdlib_meta to libraries
- `sarek/tests/e2e/test_stdlib_meta_proof.ml` — new proof exe
- `sarek/tests/e2e/dune` — added proof exe stanza

## Residual Risks

- **Double registration:** sarek_stdlib still re-registers all metadata that
  sarek_stdlib_meta registered. This is harmless (Hashtbl.replace) but not
  single-source. Refactoring sarek_stdlib modules to delegate to sarek_stdlib_meta
  is deferred to a follow-up.
- **Qualified name divergence:** sarek_stdlib_meta registers under
  `Sarek_stdlib_meta.Float32.sin` while sarek_stdlib registers under
  `Sarek_stdlib.Float32.sin`. PR-5b typer must resolve against the meta path
  when using sarek_stdlib_meta alone.
- **GPU backends (CUDA/Metal/OpenCL/Vulkan):** backend-specific execution not
  verified (no GPU hardware in CI). Vulkan build passes; CUDA not available.
