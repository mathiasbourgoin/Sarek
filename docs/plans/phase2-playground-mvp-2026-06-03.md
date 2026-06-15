# Plan — Phase 2 MVP: in-browser Sarek transpiler playground

**Date:** 2026-06-03
**Status:** VALIDATED (UI: backend dropdown/single output; bundle built in CI, never committed)
**Type:** feature (jsoo web bundle + CI jsoo-compat gate + GitHub Pages playground)

## Goal
Ship the FFI-free `Sarek_transpile.of_source` (Phase 0B) as a **client-side, in-browser
playground** on the GitHub Pages site: a user types a Sarek kernel, picks a backend, and sees the
generated CUDA/OpenCL/Metal/GLSL live — no server, all js_of_ocaml. AND make **CI enforce
jsoo-compatibility** so a future change that reintroduces FFI / breaks the jsoo build fails CI.

## Two deliverables (the user's two asks)
1. **jsoo bundle in the website** — the transpiler compiled to JS and served by the Pages site.
2. **CI jsoo-compat gate** — CI builds the jsoo target; a regression breaking FFI-freedom fails CI.

## Design

### A. JS API entry (`sarek/transpile/web/transpile_js.ml`, new)
A thin module (depends `sarek_transpile sarek_stdlib_meta js_of_ocaml`) exporting to JS:
```
globalThis.SarekTranspile.transpile(source: string, backend: string)
  => { ok: true, code: string } | { ok: false, error: string }
```
`backend` ∈ `"cuda" | "opencl" | "metal" | "glsl"`. Wraps `Sarek_transpile.of_source`, maps the
structured `error` to `string_of_error`. Built as `(modes js)` → `sarek_transpile.bc.js`.
`sarek_transpile` itself stays pure (no js_of_ocaml dep); only this web shim links jsoo.

### B. Playground page (`gh-pages/playground.html` + `gh-pages/javascripts/` asset)
Minimal, self-contained, client-side: a source `<textarea>` prefilled with the `Float32.sin`
example (using `b.(i) <- …`), a backend selector (default: show all 4 in tabbed/stacked `<pre>`
panes), transpile-on-input (debounced), error pane. Loads `sarek_transpile.js`. Plain HTML/CSS/JS,
Jekyll front-matter so it fits the existing site theme. Linked from the site nav/index.

### C. CI jsoo-compat gate (`.github/workflows/ci.yml`)
In the existing build job, add `js_of_ocaml-compiler` to the `opam install` line and add a step
`dune build sarek/transpile/web/transpile_js.bc.js`. If the transpile slice ever pulls FFI or
breaks jsoo, this step fails. (The bytecode-FFI-free gate from PR-5b remains too.)

### D. Pages deploy (`.github/workflows/docs.yml`)
Add `js_of_ocaml-compiler` to its `opam install`; build the jsoo bundle; copy
`sarek_transpile.bc.js` into the Jekyll site before the Jekyll build (so it lands in `_site`).
The bundle is **built in CI, never committed** (a generated artifact in git would drift from source).

## Sequential steps
1. `transpile_js.ml` + dune (`(modes js)`); local `dune build … .bc.js` + run under node to confirm.
2. Playground HTML/CSS/JS in `gh-pages/`; test locally against the built bundle (open in a browser).
3. CI jsoo-compat gate in `ci.yml`.
4. Pages bundle build + copy in `docs.yml`; link the page from the site index/nav.
5. Verify: full `dune build` still green sans jsoo (gate is additive, not default); goldens untouched.

## Scope boundary (OUT)
- No syntax highlighting / Monaco / fancy editor (plain textarea for MVP).
- No share-links, no persistence, no multi-kernel.
- No Float64 kernels in the default example (Float64 isn't FFI-free yet — PR-5a follow-up).
- No change to `sarek_transpile` purity (the lib stays jsoo-clean; only the web shim links jsoo).

## Risks
| Risk | Mitigation |
|---|---|
| jsoo bundle pulls a residual FFI/Unix dep and fails in-browser | the web shim depends only on the proven-FFI-free slice; built+node-run in step 1 |
| Adding js mode breaks the default `dune build`/CI | gate is a SEPARATE explicit target; verified `@install`/`@runtest` don't build it |
| Committing the generated `.js` causes source/artifact drift | build it in CI only; never commit the bundle |
| Pages deploy clobbers existing site | docs.yml already deploys `_site` wholesale; we only ADD a page + asset |

## Quality gates
```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build sarek/transpile/web/transpile_js.bc.js
node _build/default/sarek/transpile/web/transpile_js.bc.js   # smoke (if it has a main) / or load in a test harness
opam exec --switch=/home/mathias/dev/SPOC -- dune build @install   # unaffected, green
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest  # goldens byte-identical
opam exec --switch=/home/mathias/dev/SPOC -- dune build @fmt
```
