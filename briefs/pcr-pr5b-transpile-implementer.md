# Implementer Sub-brief — pure-codegen-rollout PR-5b (Sarek_transpile.of_source + FFI-free proof)

**Status:** VALIDATED
**Parent:** `briefs/pure-codegen-rollout-intake.md` (Phase 0B, PR-5 of 5, part b) — FINAL PR
**Type:** feature (new pure orchestrator + the FFI-free compile proof)

## Goal
Tie the pure libs together into `Sarek_transpile.of_source : backend -> string -> (string, error)
result` — parse an OCaml kernel source STRING, run the frontend (parse → type → lower → IR), and
emit backend source via `sarek_codegen` — and PROVE the whole slice (`sarek_frontend` +
`sarek_codegen` + `sarek_stdlib_meta` + `sarek_transpile`) compiles to **FFI-free bytecode**
(and attempt js_of_ocaml) while transpiling a **stdlib-using kernel** (`Float32.sin`). This closes
Phase 0 — the in-browser-transpiler enabler.

## Decided scope (do NOT re-litigate)
"Full": the proof MUST transpile a kernel that calls a stdlib intrinsic (`Float32.sin`) FFI-free
(PR-5a made stdlib metadata FFI-free). Basic-only was rejected.

## Pipeline (verified — reuse these exact entrypoints)
1. String → ppxlib expression: `Lexing.from_string src |> Parse.expression` (compiler AST) then
   migrate via the existing helper pattern in `Sarek_parse_helpers` (`From_502.copy_expression |>
   Ppxlib_ast.Selected_ast.of_ocaml Expression`) — reuse `Sarek_parse_helpers.expr_of_502` if exposed.
2. `Sarek_parse.parse_payload (expr : expression) : Sarek_ast.kernel`.
3. **Reject `[%native]`**: `parse_payload` yields `Sarek_ast.ENative {gpu; ocaml}` for `[%native]`
   (Sarek_parse.ml:311). Walk the AST (or detect during lowering) and return a structured
   `Error (Unsupported_native …)` — `[%native]` escape hatches can't be transpiled purely.
4. `let env = Sarek_env.(empty |> with_stdlib)` then `Sarek_typer.infer_kernel env ast : tkernel result`.
5. Mirror the PPX's passes for correct IR: `Sarek_convergence.check_kernel`, `Sarek_mono.monomorphize`,
   `Sarek_tailrec.transform_kernel`, then `Sarek_lower_ir.lower_kernel tkernel : Ir.kernel * _`.
6. Emit: `match backend with CUDA -> Sarek_ir_cuda.generate kernel | OpenCL -> Sarek_ir_opencl.generate …`
   (from `sarek_codegen`).

## Steps (build green after each; COMMIT after each step on `phase0b/pr5b-transpile`)
1. **New `sarek_transpile` lib** (`sarek/transpile/`): `(library (name sarek_transpile)
   (public_name sarek.transpile) (libraries ppxlib sarek_frontend sarek_codegen sarek_stdlib_meta)
   (preprocess (pps ppxlib.metaquot)) — NO spoc_core/ctypes). Define `type backend = CUDA | OpenCL
   | Metal | GLSL`, a structured `error` variant (parse error w/ location, type error, unsupported
   `[%native]`, convergence error), and `of_source : backend -> string -> (string, error) result`
   plus `string_of_error`. Wrap the pipeline; convert the frontend's exception/`result` errors into
   the structured `error` (do NOT let `Location.Error`/`Sarek_error` escape — catch and convert).
   COMMIT.
2. **Resolve stdlib-intrinsic path in codegen.** Verify a transpiled `Float32.sin` lowers to an IR
   intrinsic whose path the GPU generators resolve (`Sarek_pure_registry`/`Sarek_registry`
   keyed by `["Float32"]`). If the lowered path carries the wrapped `Sarek_stdlib_meta.Float32`
   prefix and the generator lookup misses, FIX the resolution (e.g. add `["Sarek_stdlib_meta";…]`
   arms to `Sarek_native_intrinsics.map_stdlib_path` AND/OR ensure lowering uses the short module
   path) so CUDA emits `sinf(...)` and OpenCL/Metal/GLSL emit `sin(...)`. The PR-5a reviewer flagged
   `map_stdlib_path` lacks `Sarek_stdlib_meta` arms — handle it here. COMMIT.
3. **Float64 (completeness).** If `of_source` must also resolve `Float64.*` FFI-free, add a pure
   `sarek_float64_meta` analog (Float64 lives in the separate `Sarek_float64` lib, absent from
   `sarek_stdlib_meta`). If this balloons, ESCALATE: ship Float32/Int32/Int64/Math/Gpu coverage and
   document Float64 as a tracked follow-up (the proof uses Float32). COMMIT.
4. **FFI-free proof** (`sarek/transpile/test/` or `sarek/tests/`): a test exe depending ONLY on
   `sarek_transpile` (+ alcotest) — NO spoc_core/ctypes in its link closure — that:
   - transpiles `"fun (a:float32 vector) (b:float32 vector) -> let i = global_thread_id in
     b.[i] <- Float32.sin a.[i]"` (or the kernel form `parse_payload` expects) to all 4 backends;
   - asserts CUDA output contains `sinf(`, OpenCL/Metal/GLSL contain `sin(` — proving stdlib-intrinsic
     resolution works FFI-free;
   - asserts `of_source` on a `[%native]` kernel returns the structured `Unsupported_native` error.
   COMMIT.
5. **Bytecode + jsoo gate.** Add a dune rule/alias building `sarek_transpile`'s proof in
   `(modes byte)` to prove FFI-free bytecode linking, and ATTEMPT a `js_of_ocaml` build of it. If
   jsoo links cleanly, great; if it needs polyfills/fails on a residual, DOCUMENT exactly what
   (don't block the PR on jsoo — bytecode-FFI-free is the hard gate, jsoo is the bonus). COMMIT.
6. **Final gates.** `dune fmt` (ocamlformat 0.28.1 is now in the switch — run `dune build @fmt`). COMMIT.

## Hard constraints
- **BYTE-IDENTICAL goldens** unchanged (`@sarek/tests/runtest`) — you add code, don't change generators.
- `sarek_transpile` + its proof have NO `spoc_core`/`ctypes` in their dune `(libraries …)` or link
  closure. The bytecode proof must link with zero FFI. This is the PR's whole point.
- Native/interpreter/Vulkan e2e still pass.
- SPDX headers; `dune build @fmt` clean. **COMMIT after each step. Do NOT push — I will push.**

## Quality gates
```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest        # goldens byte-identical
opam exec --switch=/home/mathias/dev/SPOC -- dune build sarek_transpile              # pure, no FFI
opam exec --switch=/home/mathias/dev/SPOC -- dune exec sarek/transpile/test/<proof>.exe  # transpiles Float32.sin to 4 backends, FFI-free
opam exec --switch=/home/mathias/dev/SPOC -- dune build <bytecode/jsoo target>
opam exec --switch=/home/mathias/dev/SPOC -- dune build                              # full (only -lnvrtc may fail)
opam exec --switch=/home/mathias/dev/SPOC -- dune build @fmt
```
(`-lnvrtc` full-build link error pre-existing; CI e2e-fast matrix-mul segfault known-flaky.)
