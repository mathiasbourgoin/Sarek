# Reviewer Sub-brief — Phase 2 MVP: in-browser transpiler playground

**Status:** VALIDATED
**Plan of record:** `docs/plans/phase2-playground-mvp-2026-06-03.md`

## What was implemented
A jsoo web shim (`sarek/transpile/web/transpile_js.ml`) exporting `SarekTranspile.transpile`, a
GitHub Pages playground page (`gh-pages/playground.html`, backend dropdown + single output), a CI
jsoo-compat gate (`ci.yml`), and a Pages deploy step building the bundle into `_site` (`docs.yml`).

## Load-bearing GO/NO-GO hinges
1. **jsoo bundle builds + runs FFI-free.** `dune build sarek/transpile/web/transpile_js.bc.js` →
   exit 0. Node smoke test: load the bundle, call `SarekTranspile.transpile(<Float32.sin kernel>,
   "cuda")` → `{ok:true, code}` where `code` contains `sinf(`. Confirm the shim's dune `(libraries …)`
   is only `sarek_transpile sarek_stdlib_meta js_of_ocaml` — no spoc_core/ctypes. If it pulls FFI or
   the transpile returns wrong/empty → NO-GO.
2. **No regression to existing build/CI.** `dune build @install` and `dune build @sarek/tests/runtest`
   succeed WITHOUT building the js target (verify the js artifact is NOT produced by them). Goldens
   `git diff origin/main -- sarek/tests/codegen_golden/` empty. `sarek_transpile`'s own dune still has
   NO js_of_ocaml dep (purity preserved). → NO-GO if any of these regress.
3. **Generated `.js` is NOT committed.** `git ls-files | grep -i 'sarek_transpile.*\.js'` must be
   empty — the bundle is built in CI, never checked in. → NO-GO if a generated bundle is committed.

## Verify
- **CI gate (`ci.yml`):** the build job adds `js_of_ocaml-compiler` to `opam install` and a step
  building `transpile_js.bc.js`. Confirm it's ADDITIVE (existing `@install`/`runtest` steps unchanged)
  and that a jsoo/FFI regression would actually fail this step.
- **Deploy (`docs.yml`):** adds `js_of_ocaml-compiler`, builds the bundle, copies it into the Jekyll
  site BEFORE the Jekyll build so it lands in `_site`. Confirm the copy path matches what
  `playground.html` loads (`javascripts/sarek_transpile.js`).
- **Playground page:** valid HTML; the prefilled kernel uses `b.(i)` (not `b.[i]`); the `<select>`
  covers all 4 backends; output shows `code` on success and the `error` string on failure; the page
  is linked from the site index. (Static review — no need to run a browser; the node smoke covers the JS API.)
- **Error path:** the shim maps `Error e` to `{ok:false, error: string_of_error e}` and an unknown
  backend to a clear error — not an exception/crash. Spot-check by calling transpile with a syntax
  error and with `"bogus"` backend in the node smoke.
- **No scope creep:** no editor/highlighting/persistence; no change to transpile/codegen behavior.

Return GO/NO-GO + findings. NO-GO if: the bundle doesn't build or isn't FFI-free, OR existing
build/CI regresses, OR a generated `.js` is committed, OR `sarek_transpile` gained a jsoo dep, OR the
deploy copy path mismatches the page's load path.
