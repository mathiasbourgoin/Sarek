# Implementer Sub-brief — pure-codegen-extraction · Phase 0A (spike)

**Status:** VALIDATED
**Scope:** Phase 0A ONLY (the de-risking spike). Phase 0B is gated behind the 0A
go/no-go and gets its own implementer brief after the gate.

## Goal
Cheaply prove the Phase 0 carve-out is feasible before the full rollout: a real
byte-identical gate, `Backend_error` extraction, a minimal Device.t→framework decouple,
a minimal registry/stdlib pure-metadata proof, and an FFI-free bytecode compile.

## Steps (do in order; build + goldens green after each)

1. **Golden harness.** Add `sarek/tests/e2e/` (or unit) golden snapshots: pick a small
   kernel set covering scalars, records, variants, and a stdlib call (`Float32.sin`), for
   each of cuda/opencl/metal/glsl. Capture byte-exact generated source to committed golden
   files; reset `current_variants`/`current_device` between captures; double-capture to
   assert determinism. Wire into `@sarek/tests/runtest`.
   - Verify a deliberate 1-char change to a generator makes it fail.
2. **Extract `Backend_error`.** New pure lib (no ctypes), e.g. `spoc/framework_error/` →
   `sarek_backend_error`. Move `Backend_error.ml` verbatim. Repoint `Cuda_error`/
   `Opencl_error`/`Metal_error`/`Vulkan_error` to it. `Framework_sig`/`Device_type`/
   `Typed_value` stay in `spoc_framework`.
3. **CUDA device-decouple.** In `sarek-cuda/Sarek_ir_cuda.ml` replace
   `current_device : Device.t option ref` → `current_framework : string option ref`; the
   `SNative` branch reads the string; `generate_for_device ~device` becomes a backend-side
   wrapper setting `current_framework := Some device.framework`. CUDA goldens must be
   byte-identical.
4. **Minimal registry/stdlib proof (Float32.sin).** Introduce a pure intrinsic-metadata
   form with `device : framework:string -> string` (no `Spoc_core.Device.t`); register
   `Float32.sin` via a jsoo-clean path; confirm the typer resolves it and CUDA emits the
   same bytes for a `Float32.sin` kernel. Do NOT yet convert the whole stdlib.
5. **FFI-free bytecode proof.** Define a throwaway dune target that compiles the minimal
   pure slice (frontend subset + a `sarek_codegen` skeleton holding the decoupled CUDA
   generator + `Backend_error`) with NO `ctypes`/`spoc_core` in `(libraries …)`. It must
   link. (Bytecode is enough; jsoo is Phase 2.)

## Hard constraints
- Generated GPU source BYTE-IDENTICAL throughout (the goldens are the oracle).
- Native/interpreter behaviour unchanged; do NOT touch `Sarek_native_gen_*`/`Kirc_Ast.Native`
  in 0A beyond what Step 4 minimally needs (do not convert the whole stdlib).
- No new public API surfaced yet (`Sarek_transpile` is 0B).
- SPDX headers on new files; `dune fmt` clean.

## Quality gates
```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest   # incl. new goldens
opam exec --switch=/home/mathias/dev/SPOC -- dune build
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek-vulkan/all
opam exec --switch=/home/mathias/dev/SPOC -- dune fmt --auto-promote
# FFI-free proof target builds (Step 5)
```

## Points of attention (from dual-voice)
- The registry/stdlib coupling (Step 4) is the riskiest unknown — if it can't be done
  without rewriting the stdlib, STOP and report for the go/no-go, do not force it.
- Mutable global refs in generators → goldens must reset them or determinism is fake.
- `generate_for_device` wrapper must keep SNative output identical.
