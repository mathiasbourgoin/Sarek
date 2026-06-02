# Implementation Report — Phase 0B PR-3: split sarek_ppx_lib → frontend + native_gen

**Branch:** `phase0b/pr3-frontend-split`
**Status:** COMPLETE — all hard gates pass

## Commits (on branch, ordered)

| Hash | Subject |
|------|---------|
| `e7dcf39d` | refactor(kirc_ast): retype Native variant from Device.t->string to string->string |
| `62a53363` | refactor(ppx_registry): drop dead {ti,ii,ci}_device closure fields |
| `f5a09404` | refactor(ppx): split sarek_ppx_lib into sarek_frontend + sarek_native_gen |
| `3e675a58` | style: apply dune fmt to sarek/ppx/dune |

## Files Modified

**Step 1 — Kirc_Ast.Native retype:**
- `sarek/ppx/Kirc_Ast.ml` line 122: `Native of (Spoc_core.Device.t -> string)` → `Native of (string -> string)`
- `sarek/ppx/Sarek_quote.ml` line 334: `(fun _dev -> "")` → `(fun _framework -> "")`

**Step 2 — Sarek_ppx_registry pure:**
- `sarek/ppx/Sarek_ppx_registry.ml`: removed `ti_device`, `ii_device`, `ci_device` fields; removed `~device` param from `make_{type,intrinsic,const}_info` helpers
- `sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml`: removed `~device` arg from both PPX call sites
- `sarek/tests/unit/test_ppx_registry.ml`: removed `test_device_gen` helper and all `{ti,ii,ci}_device = ...` fields

**Steps 3–5 — Library split + façade + consumer updates:**
- `sarek/ppx/dune`: four stanzas: `sarek_frontend` (23 modules, wrapped false, `libraries ppxlib sarek_ir`), `sarek_native_gen` (9 modules, wrapped false, `libraries ppxlib spoc_core sarek_ir sarek_frontend`), `sarek_ppx_lib` (empty modules, re_export both), `sarek_ppx` (unchanged)
- `sarek/ppx/Sarek_quote.ml`: removed 81 `Sarek_ppx_lib.` qualifications in quoted expressions
- `sarek/ppx/Sarek_ppx.ml`: removed bare `open Sarek_ppx_lib`
- `sarek/ppx_intrinsic/Sarek_ppx_intrinsic.ml`: removed all `Sarek_ppx_lib.` prefixes
- `sarek/tests/common/Ir_compare.ml`, `sarek/ppx/test/` (3 files), `sarek/tests/unit/` (20+ files): removed `Sarek_ppx_lib.` prefixes and bare `open Sarek_ppx_lib` statements

**Step 6 — Format:**
- `sarek/ppx/dune`: blank line added by `dune fmt --auto-promote`

## Façade Decision: FALLBACK TAKEN

Dune's `(re_export)` with `(modules)` empty does NOT create `Sarek_ppx_lib.X`
submodule aliases. Both sub-libraries use `(wrapped false)`. `sarek_ppx_lib` is
an empty re-export façade — consumers get all modules at top level. All
`Sarek_ppx_lib.X` qualified paths were removed from consumer source files.

## sarek_frontend final dune (libraries …) line

```
(libraries ppxlib sarek_ir)
```

No `spoc_core`.

## Quality Gate Results

| Gate | Result |
|------|--------|
| `dune build @sarek/tests/runtest` — goldens byte-identical | PASS |
| `dune build` — full build | PASS (pre-existing -lnvrtc only) |
| `dune build @sarek-vulkan/all` | PASS |
| `sarek_frontend (libraries ppxlib sarek_ir)` — no spoc_core | PASS |
| `ocamlformat --check` on changed files | PASS |

## Golden Diff vs pre-PR3 (94011d2e)

```
git diff 94011d2e -- sarek/tests/codegen_golden/
(empty — byte-identical)
```

## Residual Risks

- CUDA/Metal backends not available — verified OpenCL, Vulkan, and stub paths only.
- Both sub-libs use `(wrapped false)` — all 32 modules at top-level namespace. Future name collisions possible.
- Generated code in `Sarek_quote.ml` and `Sarek_ppx_intrinsic.ml` now references modules by short name; requires consumers to have `sarek_ppx_lib` (or the sub-libs) in their transitive deps at user-compile-time.
