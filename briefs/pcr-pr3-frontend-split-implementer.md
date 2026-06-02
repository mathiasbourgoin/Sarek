# Implementer Sub-brief — pure-codegen-rollout PR-3 (split sarek_ppx_lib → frontend + native_gen)

**Status:** VALIDATED
**Parent:** `briefs/pure-codegen-rollout-intake.md` (Phase 0B, PR-3 of 5)
**Type:** refactor (structural; pure — NO behavior/codegen change)

## Goal
Split the mixed `sarek_ppx_lib` (`sarek/ppx/dune`, 33 modules, deps `ppxlib spoc_core`)
into a **pure `sarek_frontend`** (deps `ppxlib` + `sarek_ir`, NO `spoc_core`/ctypes) and an
**FFI `sarek_native_gen`** (keeps `spoc_core`), and decouple `Kirc_Ast.Native` from
`Spoc_core.Device.t`. A re-export façade keeps all current consumers building unchanged.
This is the structural enabler for PR-4 (`sarek_codegen`) and PR-5 (`Sarek_transpile`,
FFI-free bytecode/jsoo).

**This is a pure refactor: generated source must be BYTE-IDENTICAL.** The
`sarek/tests/codegen_golden/` harness is the oracle. Any golden diff = a real regression → STOP.

## Evidence (dependency map, already done — do NOT re-investigate)
- The 23 pure-frontend modules have **zero** `Spoc_core` references (verified by grep).
- The `Spoc_core` coupling lives in only: `Kirc_Ast` (the `Native` variant), the 6 native_gen
  modules, `Sarek_quote`, and `Sarek_ppx_registry` (dead `*_device` closures).
- `Kirc_Ast.Native of (Spoc_core.Device.t -> string)` (`Kirc_Ast.ml:122`) is **vestigial**:
  the ONLY construction is `Sarek_quote.ml:334` `[%expr Sarek.Kirc_Ast.Native (fun _dev -> "")]`
  (a device-ignoring no-op); it is never applied to a device. Printer `Kirc_Ast.ml:277` and
  `Ir_compare.ml:109,185` only match `Native _`.
- `Sarek_ppx_registry.{ti,ii,ci}_device : Spoc_core.Device.t -> string` (lines ~24/35/46) are
  **dead** in V2 — consumers (`Sarek_env`, `Sarek_ppx`) read only metadata fields, never the
  device closures.

## Module partition (exact)
**`sarek_frontend` (PURE — 23 modules):**
Sarek_types, Sarek_scheme, Sarek_core_primitives, Sarek_ppx_registry, Sarek_env, Sarek_error,
Sarek_reserved, Sarek_parse, Sarek_parse_helpers, Sarek_ast, Sarek_typed_ast, Sarek_typer,
Sarek_convergence, Sarek_lower_ir, Sarek_quote_ir, Sarek_ir_ppx, Sarek_mono, Sarek_debug,
Sarek_tailrec_analysis, Sarek_tailrec_elim, Sarek_tailrec_bounded, Sarek_tailrec_pragma,
Sarek_tailrec.

**`sarek_native_gen` (FFI — keeps `spoc_core`):**
Kirc_Ast, Sarek_lower, Sarek_quote, Sarek_native_helpers, Sarek_native_intrinsics,
Sarek_native_gen_base, Sarek_native_gen_expr, Sarek_native_gen, Sarek_native_gen_kernel.

> NOTE on `Kirc_Ast`/`Sarek_lower`: after the `Native` retype (below) they no longer touch
> `Spoc_core` directly, but they belong to the native/legacy codegen path and are consumed by
> `Sarek_quote` → keep them in `sarek_native_gen` (do NOT force them into the pure lib; the
> goal is a Spoc_core-free `sarek_frontend`, not minimizing native_gen). If you find a frontend
> module (the 23) transitively needs Kirc_Ast or Sarek_lower, STOP and report — that breaks the
> cut and the map says it won't happen.

## Steps (build + goldens green after EACH; COMMIT after each step on your worktree branch)
1. **Decouple `Kirc_Ast.Native`.** Change `Native of (Spoc_core.Device.t -> string)`
   (`Kirc_Ast.ml:122`) → `Native of (string -> string)` (framework-string closure). Update the
   sole constructor `Sarek_quote.ml:334` `(fun _dev -> "")` → `(fun _framework -> "")`. Confirm
   `Kirc_Ast` has no other `Spoc_core` use; if clean, `Kirc_Ast` no longer needs `spoc_core`.
   Build + goldens green. COMMIT.
2. **Make `Sarek_ppx_registry` pure.** Drop the dead `ti_device/ii_device/ci_device` fields
   (and their constructors/usages if any remain) so the registry record carries metadata only and
   the module no longer references `Spoc_core`. If removal ripples beyond the registry, STOP and
   report (do not leave `spoc_core` in the pure set). Build + goldens green. COMMIT.
3. **Create `sarek_frontend`** (`sarek/ppx/frontend/` or via dune `(modules ...)` selection —
   your call, lowest churn): `(library (name sarek_frontend) (public_name sarek.frontend)
   (libraries ppxlib sarek_ir) (preprocess (pps ppxlib.metaquot)) (modules <the 23>))`.
   It must build with NO `spoc_core` dependency. Build + goldens green. COMMIT.
4. **Create `sarek_native_gen`** `(library (name sarek_native_gen) (public_name sarek.native_gen)
   (libraries ppxlib spoc_core sarek_ir sarek_frontend) (preprocess (pps ppxlib.metaquot))
   (modules <the 9>))`. Build + goldens green. COMMIT.
5. **Façade.** Keep `sarek_ppx_lib` as a thin re-export so the 7 consumers below build UNCHANGED:
   `(library (name sarek_ppx_lib) (public_name sarek.ppx.lib)
   (libraries (re_export sarek_frontend) (re_export sarek_native_gen)) (modules))`.
   If `(modules)` empty + `(re_export …)` does not transparently expose the modules to consumers,
   FALL BACK to updating the 7 consumers' dune `libraries` to depend on `sarek_frontend` and/or
   `sarek_native_gen` directly, and delete `sarek_ppx_lib` — and SAY SO in the report. Either way,
   all 7 consumers must build. Build everything + goldens green. COMMIT.
6. **Final gates** (below). COMMIT any format fixes.

## Consumers that MUST still build (the 7)
`sarek/ppx/dune` (sarek_ppx rewriter), `sarek/ppx_intrinsic/dune`, `sarek/ppx/test/dune`,
`sarek/tests/common/dune`, `sarek/tests/unit/dune`, `sarek/tests/e2e/dune`, `sarek/Sarek_stdlib/dune`.

## Hard constraints
- **BYTE-IDENTICAL goldens** — `@sarek/tests/runtest` unchanged. A golden diff is a regression, STOP.
- `sarek_frontend` MUST NOT depend on `spoc_core` (verify: its `(libraries …)` has no `spoc_core`,
  and `dune build @sarek/tests/runtest` passes). This is the load-bearing outcome of the PR.
- NO new `sarek_codegen` (PR-4), NO `Sarek_transpile` (PR-5), NO generator changes.
- SPDX headers on any new files; `dune fmt` clean.
- **COMMIT after each step** (do not leave changes uncommitted — prior PR work was lost this way).

## Quality gates
```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest   # goldens byte-identical
opam exec --switch=/home/mathias/dev/SPOC -- dune build                         # all 7 consumers + libs
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek-vulkan/all
opam exec --switch=/home/mathias/dev/SPOC -- dune build sarek_frontend          # must link with NO spoc_core
/home/mathias/.opam/octez-setup/bin/ocamlformat --check <changed .ml/dune>
```
(`-lnvrtc` full-build link error pre-existing; CI e2e-fast matrix-mul segfault known-flaky.)
