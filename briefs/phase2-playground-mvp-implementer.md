# Implementer Sub-brief — Phase 2 MVP: in-browser transpiler playground

**Status:** VALIDATED
**Plan of record:** `docs/plans/phase2-playground-mvp-2026-06-03.md`
**Type:** feature (jsoo web bundle + CI jsoo-compat gate + GitHub Pages playground)

## Goal
Ship `Sarek_transpile.of_source` (pure, FFI-free) as a **client-side in-browser playground** on the
GitHub Pages site, and make **CI enforce jsoo-compatibility** so any future FFI/jsoo regression fails CI.

## Validated decisions (do NOT re-litigate)
- **Bundle delivery:** built in CI on each deploy; the generated `.js` is **never committed** to the repo.
- **Build impact:** the jsoo target is a SEPARATE explicit target — `dune build @install`/`@runtest`
  must NOT build it (verified). Existing CI jobs stay green without js_of_ocaml.
- **Playground UI:** prefilled kernel + a **backend `<select>` dropdown showing ONE backend's output
  at a time** (NOT all-4-at-once).
- `sarek_transpile` stays pure (no js_of_ocaml dep). Only the new web shim links jsoo.

## Environment (already done by lead)
`js_of_ocaml`, `js_of_ocaml-ppx`, `js_of_ocaml-compiler` (6.3.2) are installed in the SPOC switch.
You are on branch `phase2/playground-mvp`.

## Sarek_transpile API (verified)
```ocaml
type backend = CUDA | OpenCL | Metal | GLSL
val of_source : backend -> string -> (string, error) result
val string_of_error : error -> string
```

## Steps (build green after each; COMMIT after each on `phase2/playground-mvp`)
1. **JS shim** `sarek/transpile/web/transpile_js.ml` + `dune`:
   `(executable (name transpile_js) (modules transpile_js) (libraries sarek_transpile
   sarek_stdlib_meta js_of_ocaml) (preprocess (pps js_of_ocaml-ppx)) (modes js))`.
   Export `globalThis.SarekTranspile` with `transpile(source: string, backend: string)` returning a
   JS object `{ ok: bool, code: string|null, error: string|null }`. `backend` ∈
   `cuda|opencl|metal|glsl` (case-insensitive; unknown → `{ok:false, error:"unknown backend …"}`).
   Map `Ok code`→`{ok:true, code}`, `Error e`→`{ok:false, error: string_of_error e}`. Also expose
   a `backends` array `["cuda","opencl","metal","glsl"]`. SPDX header.
   Verify: `dune build sarek/transpile/web/transpile_js.bc.js` succeeds AND the produced JS links
   FFI-free (no spoc_core/ctypes in the closure — it depends only on the proven-pure slice).
   COMMIT.
2. **Playground page** `gh-pages/playground.html` (Jekyll front-matter `--- layout: default
   title: Playground ---` so it matches the site theme): a `<textarea>` prefilled with a working
   kernel using `b.(i) <- …` syntax (NOT `b.[i]`, which OCaml 5.4 can't parse standalone), e.g.
   `fun (a : float32 vector) (b : float32 vector) -> let i = global_thread_id in b.(i) <- Float32.sin a.(i)`;
   a backend `<select>` (cuda/opencl/metal/glsl); a single `<pre>` output pane; debounced
   transpile-on-input + on-backend-change calling `SarekTranspile.transpile(...)`, showing `code`
   or the `error` string. Loads `javascripts/sarek_transpile.js`. Plain HTML/CSS/vanilla JS, no
   framework. Link it from `gh-pages/index.md` nav. COMMIT.
3. **CI jsoo-compat gate** `.github/workflows/ci.yml`: in the existing build job's `opam install`
   line, add `js_of_ocaml-compiler`; add a build step `dune build sarek/transpile/web/transpile_js.bc.js`
   so a jsoo/FFI regression fails CI. Keep it additive — do not change the existing `@install`/`runtest`
   steps. COMMIT.
4. **Pages deploy** `.github/workflows/docs.yml`: add `js_of_ocaml-compiler` to its `opam install`;
   before the Jekyll build, `dune build sarek/transpile/web/transpile_js.bc.js` and copy the result
   to `gh-pages/javascripts/sarek_transpile.js` (so Jekyll includes it in `_site`). Do NOT commit the
   `.js`. COMMIT.
5. **Verify no regression:** `dune build @install` and `dune build @sarek/tests/runtest` still green
   WITHOUT building the js target; goldens byte-identical; `dune build @fmt` clean. COMMIT fmt.

## Hard constraints
- The generated `.js` is NEVER committed (built in CI). Add `*.bc.js`-in-gh-pages to `.gitignore` if needed.
- `sarek_transpile` keeps zero js_of_ocaml dependency; only `sarek/transpile/web/` links jsoo.
- Existing CI (`@install`, `dune runtest`) must not require js_of_ocaml — the gate is a separate step.
- Goldens byte-identical; no change to `sarek_transpile`/`sarek_codegen` behavior.
- SPDX headers on new `.ml`; `dune build @fmt` clean.

## Quality gates
```bash
opam exec --switch=/home/mathias/dev/SPOC -- dune build sarek/transpile/web/transpile_js.bc.js   # jsoo builds
node -e 'global.globalThis=global; require("./_build/default/sarek/transpile/web/transpile_js.bc.js"); const r=globalThis.SarekTranspile.transpile("fun (a:float32 vector) (b:float32 vector) -> let i = global_thread_id in b.(i) <- Float32.sin a.(i)","cuda"); console.log(r.ok, r.code||r.error)'  # node smoke -> expect ok + sinf(
opam exec --switch=/home/mathias/dev/SPOC -- dune build @install            # green, no jsoo needed
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest # goldens byte-identical
opam exec --switch=/home/mathias/dev/SPOC -- dune build @fmt
```
