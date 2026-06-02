# Reviewer Sub-brief — pure-codegen-rollout PR-3 (split sarek_ppx_lib)

**Status:** VALIDATED
**Parent:** `briefs/pure-codegen-rollout-intake.md` (Phase 0B, PR-3 of 5)

## What was implemented
`sarek_ppx_lib` (33 modules, deps `ppxlib spoc_core`) split into pure `sarek_frontend`
(23 modules, deps `ppxlib sarek_ir`, NO spoc_core) + FFI `sarek_native_gen` (9 modules), with
`Kirc_Ast.Native` retyped `(Device.t -> string)` → `(string -> string)` and the dead
`Sarek_ppx_registry.*_device` closures dropped. A façade keeps consumers building.

## The two load-bearing checks (GO/NO-GO hinges on these)
1. **`sarek_frontend` is genuinely Spoc_core-free.** Read its dune `(libraries …)` — must have
   NO `spoc_core`. Then confirm it builds standalone. If `spoc_core` leaked in (e.g. a frontend
   module still references it, or a dep pulls it transitively), NO-GO — the entire PR's purpose
   is this purity (it unblocks PR-5's FFI-free bytecode/jsoo compile).
2. **Byte-identical goldens.** `git diff main -- sarek/tests/codegen_golden/` must be EMPTY. This
   is a pure structural refactor; ANY golden change is a regression → NO-GO.

## Also verify
- **All 7 consumers build:** `sarek/ppx`, `sarek/ppx_intrinsic`, `sarek/ppx/test`,
  `sarek/tests/common`, `sarek/tests/unit`, `sarek/tests/e2e`, `sarek/Sarek_stdlib`. Run
  `dune build` (full) — known-pre-existing `-lnvrtc` link error is the ONLY acceptable failure.
- **Module partition matches the brief** — the 23 named modules in `sarek_frontend`, the 9 in
  `sarek_native_gen`, none duplicated/dropped. No module silently left in a dead `sarek_ppx_lib`.
- **`Kirc_Ast.Native` retype is complete** — no residual `Device.t` in the variant; the sole
  constructor (`Sarek_quote.ml:~334`) updated; printer/`Ir_compare` still compile (match `Native _`).
- **Façade correctness** — if `sarek_ppx_lib` kept as `(re_export …)`, confirm consumers resolve
  the modules they reference (they `open`/qualify `Sarek_typer`, `Sarek_quote`, etc.). If the
  implementer fell back to updating consumer dunes + deleting `sarek_ppx_lib`, confirm every
  reference site was updated and report that as the chosen approach.
- **No scope creep** — no `sarek_codegen`, no `Sarek_transpile`, no generator edits, no behavior change.

## Gates
```bash
dune build @sarek/tests/runtest      # goldens byte-identical
dune build                            # all consumers (only -lnvrtc may fail, pre-existing)
dune build @sarek-vulkan/all
ocamlformat --check <changed>
```

Return GO/NO-GO. NO-GO if: `sarek_frontend` pulls `spoc_core`, OR any golden changed, OR a
consumer fails to build for a non-pre-existing reason.
