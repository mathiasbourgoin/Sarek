# Intake Brief — pure-codegen-rollout (Phase 0B)

**Date:** 2026-06-02
**Status:** VALIDATED
**Type:** refactor

## Goal

Complete the pure-codegen carve-out begun in Phase 0A (golden harness, `sarek_backend_error`,
CUDA decouple — all merged to `main`). End state: a pure, FFI-free
`OCaml source → IR → per-backend GPU source` pipeline exposed as
`Sarek_transpile.of_source : string -> (backend, string) result`, compilable to bytecode
(and ultimately js_of_ocaml for the Phase 2 playground), with GPU backend *runtimes*
unchanged. The Phase 0A golden harness on `main` is the byte-identical regression gate
throughout.

**This is large — to be delivered as 5 sequential PRs, each its own roster cycle** (see
"PR split"). This brief covers the whole Phase 0B; each PR gets a focused implementer
sub-brief at its own plan stage.

## Scope Boundary (OUT)
- Phase 2 playground UI / the actual jsoo web bundle (a bytecode FFI-free compile is the 0B
  acceptance proof; a jsoo smoke is a stretch goal, not required).
- Backend runtime behaviour (memory/launch/device mgmt) — unchanged.
- The native-CPU generator (`Sarek_native_gen_*`) — stays FFI-linked; native/interpreter
  output must remain identical.
- `[%native]` in `of_source` — excluded (ENative carries unevaluated `Ppxlib.expression`);
  returns a structured error.

## Decisions (resolved at intake)
- **`sinf` (path-qualified `Float32` math): ADOPT.** `main` emits `sin` for
  `EIntrinsic(["Float32"],"sin")` (fell through to the unqualified branch) — wrong for a
  `float32` op. The pure registry's `sinf` is correct single-precision. 0B adopts it as a
  deliberate fix, updating the affected goldens and adding a **GPU-validated** `Float32.sin`
  e2e (RX 7900 XTX available) to confirm numerically. Mirror for `cos/sqrt/exp/...` and the
  OpenCL/Metal/GLSL equivalents.
- Device decouple mechanism: `current_framework : string option ref` (same as CUDA 0A).
- GLSL logging: injected `?log` no-op hook.
- Parser: ppxlib parse + `Ppxlib_ast.Selected_ast.of_ocaml` migration.

## PR split (proposed)
1. **opencl+metal+glsl decouple** (plan steps 6–7). Mirror CUDA `current_device`→
   `current_framework` in OpenCL + Metal; GLSL `?log` hook. Byte-identical; low risk.
2. **pure registry + PPX dual-registration + `sinf`** (step 8). Promote `Sarek_pure_registry`
   from the spike; make `%sarek_intrinsic` emit pure (`framework:string->string`) entries
   alongside the FFI ones across `Float32/Float64/Int32/Int64/Math/Gpu`; adopt `sinf` with
   golden updates + GPU e2e. Behaviour-affecting; medium risk.
3. **split `sarek_ppx_lib`** (step 9) → `sarek_frontend` (pure) + `sarek_native_gen` (FFI) +
   re-export façade; decouple `Kirc_Ast.Native (Spoc_core.Device.t -> string)` → framework.
   Structural; medium-high risk (wrapped-lib paths).
4. **create `sarek_codegen`** (step 10): extract the 4 device-decoupled generators +
   `sarek_backend_error` into a pure lib (deps `sarek_ir` only); backends re-export
   `Sarek_ir_<b>` under their wrapped namespace.
5. **`Sarek_transpile.of_source` + FFI-free proof** (steps 11–12): the orchestrator; final
   gates incl. all goldens byte-identical and a bytecode FFI-free compile of
   `sarek_frontend + sarek_codegen + Sarek_transpile`.

## Relevant Files
| File | Role | Fact |
|---|---|---|
| `sarek-opencl/Sarek_ir_opencl.ml` | OpenCL gen | `current_device:Device.t option ref` (l.35, used l.531, set l.770) — mirror CUDA |
| `sarek-metal/Sarek_ir_metal.ml` | Metal gen | same pattern (l.24/644/997) |
| `sarek-vulkan/Sarek_ir_glsl.ml` | GLSL gen | `Spoc_core.Log.debugf` ×2 (l.947, l.1034) — inject `?log` |
| `spoc/ir/Sarek_pure_registry.ml` (spike branch) | pure registry | proven; promote + extend |
| `sarek/ppx/Sarek_ppx_registry.ml` | FFI registry | `*_device : Spoc_core.Device.t -> string` |
| `sarek/Sarek_stdlib/*.ml`, `sarek/ppx_intrinsic/` | `%sarek_intrinsic` defs + PPX | source of dual-registration |
| `sarek/ppx/Sarek_lower_ir.ml` | typed AST → IR | `IntrinsicRef(path,name)→EIntrinsic(path,name,..)` (l.383–390) — path-qualified intrinsics reach codegen |
| `sarek/ppx/dune` (`sarek_ppx_lib`) | mixed lib | split into frontend/native-gen |
| `sarek/ppx/Kirc_Ast.ml` | legacy AST | `Native of (Spoc_core.Device.t -> string)` (l.122) — decouple |
| `sarek/tests/codegen_golden/` (main) | golden harness | the byte-identical gate; extend with stdlib-math goldens |
| `sarek-*/Cuda|Opencl|Metal|Vulkan_plugin.ml` | consumers | reference `Sarek_ir_<b>` unqualified (wrapped) — re-export must preserve |

## Quality Gates
```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest   # goldens byte-identical
opam exec --switch=/home/mathias/dev/SPOC -- dune build
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek-vulkan/all
opam exec --switch=/home/mathias/dev/SPOC -- dune fmt --auto-promote
# PR2: GPU e2e for Float32 math — dune exec sarek/tests/e2e/test_*.exe -- --vulkan (RX 7900 XTX)
# PR5: bytecode FFI-free compile of sarek_frontend + sarek_codegen + Sarek_transpile
```
(`-lnvrtc` full-build link error is pre-existing; CI `build` e2e-fast matrix-mul segfault is known-flaky OpenCL-CPU — re-run, don't treat a lone hit as regression.)

## Open Questions
- [ ] **PR2 numerical validation:** adopting `sinf` changes CUDA/OpenCL output for
  `Float32` math. Beyond byte-identical goldens (which we deliberately update), confirm a
  `Float32.sin` kernel produces correct results on the Vulkan/OpenCL GPU before merging PR2.
  (Resolve in PR2's QA, not assumed.)
- [ ] **PR3 façade completeness:** after splitting `sarek_ppx_lib`, must `sarek_ppx_lib`
  re-export everything its current external consumers use? Enumerate consumers at PR3 plan
  time (grep `Sarek_ppx_lib`/the modules) — the brief flags it; the PR3 plan resolves it.
- [ ] **jsoo stretch:** is a real `js_of_ocaml` compile (not just bytecode) required for 0B
  acceptance, or deferred to Phase 2? (Default: bytecode FFI-free proof suffices for 0B;
  jsoo is Phase 2.)
