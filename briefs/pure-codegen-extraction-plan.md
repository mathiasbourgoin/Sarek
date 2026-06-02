# Plan — pure-codegen-extraction (Phase 0)

**Date:** 2026-06-02
**Status:** VALIDATED
**Source brief:** `briefs/pure-codegen-extraction-intake.md` (v2, VALIDATED)

Dual-voice analysis (architect + skeptic) was performed on v1 and folded into the v2 brief;
not re-spawned. Q1 resolved (Backend_error is ctypes-free → extractable). Q2 resolved with
expanded scope (the registry + every stdlib intrinsic carry `Spoc_core.Device.t` device
closures; population is FFI-coupled). The plan front-loads a **spike (0A)** to validate the
riskiest unknowns cheaply, with a **go/no-go** before the full rollout **(0B)**.

## Strategy

The whole carve-out hinges on removing `Spoc_core.Device.t` from the compile path
(generators + registry + stdlib intrinsic defs) and on having a real byte-identical gate.
Prove both on a *minimal slice* before touching the full surface.

## Phase 0A — De-risking spike (small, gated)

1. **Golden-snapshot harness.** Add a test that captures byte-exact generated source for a
   representative kernel set across cuda/opencl/metal/glsl, resetting the mutable codegen
   refs (`current_variants`, `current_device`) between captures. This is the regression
   detector that does not exist today. Establishes the baseline goldens on current `main`.
   *Completion:* goldens committed; a deliberate whitespace change makes the test fail.
2. **Extract `Backend_error` to a pure leaf lib** (`sarek_backend_error`, no ctypes).
   Repoint the four `*_error` modules to it. *Completion:* `@sarek/tests/runtest` +
   `@sarek-vulkan/all` green; goldens unchanged.
3. **Minimal device-decouple proof.** In ONE generator (CUDA) replace
   `current_device : Device.t option ref` → `current_framework : string option ref`;
   `generate_for_device` becomes a backend-side wrapper. *Completion:* CUDA goldens
   byte-identical; build green.
4. **Minimal registry/stdlib-split proof.** For ONE intrinsic (`Float32.sin`): introduce a
   pure intrinsic-metadata representation with `device : framework:string -> string` (no
   `Device.t`), register it via a jsoo-clean path, and confirm the typer resolves it and the
   CUDA generator emits the same bytes. Proves Q2(a)+(b) tractable. *Completion:* goldens
   byte-identical for a `Float32.sin` kernel.
5. **Bytecode FFI-free proof.** Compile the minimal pure slice (frontend subset +
   `sarek_codegen` skeleton + `Backend_error`) with no ctypes/spoc_core link (bytecode is
   sufficient; jsoo target is Phase 2). *Completion:* it links FFI-free.

   **GO/NO-GO gate (human):** if 0A succeeds, proceed to 0B. If Step 4 or 5 reveals the
   registry/stdlib decoupling is intractable without rewriting the stdlib, STOP and
   re-decide scope (the spike has then cheaply saved the full rollout).

## Phase 0B — Full rollout (after GO)

6. Apply the device→framework decouple to **opencl + metal** generators (mirror Step 3).
7. Apply the GLSL `?log` no-op hook (decouple `Spoc_core.Log`).
8. Extend the registry `*_device` decouple + pure-metadata registration to **all** stdlib
   intrinsics/types/consts (mirror Step 4 across `Float32/Float64/Int32/Int64/Math/Gpu`),
   splitting pure metadata from FFI runtime bodies. Keep native/interpreter behaviour
   identical (their runtime path still links the FFI stdlib).
9. **Split `sarek_ppx_lib`** → `sarek_frontend` (pure: parse/typer/convergence/mono/tailrec/
   lower/lower_ir/env/registry/core_primitives; deps `ppxlib`+`sarek_ir`) and
   `sarek_native_gen` (Kirc_Ast, Sarek_native_gen_*; keeps spoc_core). Keep `sarek_ppx_lib`
   as a re-export façade. Decouple `Kirc_Ast.Native (Spoc_core.Device.t -> string)` →
   framework-name (so `Sarek_lower`'s path is pure).
10. **Create `sarek_codegen`** = the 4 device-decoupled generators + `Backend_error`; deps
    `sarek_ir` only. Backend packages re-export `Sarek_ir_<b>` under their wrapped namespace
    (preserve unqualified call sites in `*_plugin.ml`).
11. **Write `Sarek_transpile.of_source`** — ppxlib parse (compiler-libs `Parse.expression` +
    `Ppxlib_ast.Selected_ast.of_ocaml` migration) → pipeline (parse→type→convergence→mono→
    tailrec→lower_ir) → per-backend `generate_with_types`. Returns `(backend, string) result`;
    `[%native]` → structured error.
12. **Final gates:** full `dune build`; `@sarek/tests/runtest`; `@sarek-vulkan/all`; all
    goldens byte-identical; `dune fmt`; bytecode FFI-free compile of `sarek_frontend` +
    `sarek_codegen` + `Sarek_transpile`.

## Dependencies

1 (goldens) precedes everything (it's the gate). 2 unblocks 10. 3 precedes 6. 4 precedes 8.
5 gates 0B. 9 depends on 8 (Kirc_Ast decouple). 10 depends on 2,6,7,9. 11 depends on 9,10.

## Identified risks

| Risk | Prob | Impact | Mitigation |
|---|---|---|---|
| Registry/stdlib decouple intractable without stdlib rewrite | Med | High | Spike Step 4 proves it on one intrinsic before the full surface; GO/NO-GO gate |
| Golden gate misses non-determinism (mutable refs) | Med | High | Step 1 resets refs between captures; assert determinism by double-capture |
| `Selected_ast` migration / version pinning (ocaml 5.x ↔ ppxlib) | Med | Med | Validate the parse bridge in spike (fold into Step 5 if `of_source` skeleton is touched) |
| Wrapped-lib re-export breaks unqualified `Sarek_ir_<b>` call sites | Med | Low (compile error) | re-export module named exactly `Sarek_ir_<b>` per backend |
| Native/interpreter output changes when stdlib is split | Low | High | their runtime path keeps the FFI stdlib; goldens + runtest gate |
| Scope/effort overrun (pervasive Device.t threading) | High | Med | spike-first + go/no-go; 0B only after cheap proof |

## Decisions made

| Point | Decision | Reason |
|---|---|---|
| Device decouple | `current_framework : string option ref` (generators) + `framework:string->string` (registry/stdlib closures) | minimal-diff, lowest byte-identical risk; only `dev.framework` is read |
| GLSL logging | inject `?log` no-op hook | preserves bytes; no return value used |
| Parser | ppxlib parse + `Selected_ast.of_ocaml` migration | robust to version skew (skeptic) |
| `[%native]` | excluded from `of_source` (structured error) | ENative carries unevaluated AST |
| Lib split | mandatory: `sarek_frontend` / `sarek_native_gen` + façade | single lib mixes pure+FFI |
| Sequencing | spike (0A) + go/no-go before full rollout (0B) | scope grew at every layer; de-risk cheaply |

## Assumptions

- The device closures (`*_device`, generators' SNative) read only `dev.framework` — verified
  for generators; to re-verify across stdlib defs in Step 8.
- `kern.kern_types` is the correct `~types` source (all plugins pass it).
- Native/interpreter execution is out of scope and keeps the FFI stdlib.
- Byte-identical is judged by the new goldens (Step 1), not the existing fragment asserts.
