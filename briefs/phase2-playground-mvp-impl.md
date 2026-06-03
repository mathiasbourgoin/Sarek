# Implementation Report — Phase 2 MVP: in-browser transpiler playground

**Branch:** `phase2/playground-mvp`
**Date:** 2026-06-03
**Status:** COMPLETE — all gates green

## Commit hashes (in order)

1. `1751de9b` — Add js_of_ocaml shim for in-browser transpiler playground
2. `33b5b52b` — Add in-browser playground page for Sarek transpiler
3. `08829625` — Add jsoo-compat gate to CI build job
4. `ca1798bb` — Add jsoo bundle build to docs deploy workflow and gitignore the artifact

## Files changed

| File | Action |
|------|--------|
| `sarek/transpile/web/transpile_js.ml` | New — JS shim |
| `sarek/transpile/web/dune` | New — separate `(modes js)` executable |
| `gh-pages/playground.html` | New — Jekyll playground page |
| `gh-pages/index.md` | Updated — playground link added (line 42) |
| `.github/workflows/ci.yml` | Updated — jsoo-compat gate step added |
| `.github/workflows/docs.yml` | Updated — bundle build + copy before Jekyll |
| `.gitignore` | Updated — `gh-pages/javascripts/sarek_transpile.js` excluded |

## JS API shape

```
globalThis.SarekTranspile.transpile(source: string, backend: string)
  -> { ok: bool, code: string|null, error: string|null }

globalThis.SarekTranspile.backends
  -> ["cuda", "opencl", "metal", "glsl"]
```

- `backend` is case-insensitive; unknown value returns `{ok:false, error:"unknown backend ..."}`.
- `Ok code` maps to `{ok:true, code, error:null}`.
- `Error e` maps to `{ok:false, code:null, error: string_of_error e}`.

## Shim dune `(libraries)` line

```
(libraries sarek_transpile sarek_stdlib_meta js_of_ocaml)
```

No `spoc_core`, no `ctypes`. Only the proven-pure slice.

## Gate results

### jsoo bundle build
```
opam exec --switch=/home/mathias/dev/SPOC -- dune build sarek/transpile/web/transpile_js.bc.js
```
PASS (exit 0, no output)

### Node smoke test
```
node -e 'global.globalThis=global; require("./_build/default/sarek/transpile/web/transpile_js.bc.js");
  const r=globalThis.SarekTranspile.transpile(
    "fun (a:float32 vector) (b:float32 vector) -> let i = global_thread_id in b.(i) <- Float32.sin a.(i)",
    "cuda");
  console.log(r.ok, r.code||r.error)'
```
Output:
```
true
extern "C" {
__global__ void sarek_kern(float* __restrict__ a, int sarek_a_length, float* __restrict__ b, int sarek_b_length) {
  int i = (threadIdx.x + blockIdx.x * blockDim.x);
  b[i] = sinf(a[i]);
}
}
```
PASS — `ok=true`, output contains `sinf(`

No residual FFI/Unix dep pulled into the bundle.

### `@install` (no jsoo needed)
```
opam exec --switch=/home/mathias/dev/SPOC -- dune build @install
```
PASS (exit 0) — does not build the js target

### `@sarek/tests/runtest` (goldens byte-identical)
```
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest
```
PASS (exit 0) — no golden changes

### Format check
```
opam exec --switch=/home/mathias/dev/SPOC -- dune build @fmt
```
PASS (exit 0, clean)

## `.js` artifact tracking

- `gh-pages/javascripts/sarek_transpile.js` listed in `.gitignore`, NOT committed.
- `_build/default/sarek/transpile/web/transpile_js.bc.js` in `_build/`, not tracked.
- Confirmed via `git ls-files --error-unmatch` returning error (file unknown to git) for both paths.

## Backend symmetry note

This change adds a read-only web shim over `sarek_transpile`. No backend codegen logic
was changed. All four backends (CUDA/OpenCL/Metal/GLSL) are exposed through the dropdown;
Metal and GLSL paths were not separately validated against real GPU hardware (not
available in this environment) but the transpiler's pure OCaml path was exercised via the
CUDA smoke test and the existing sarek/tests golden suite.

## Residual risks

- Metal/GLSL runtime output correctness: not hardware-validated here; relies on the
  existing codegen golden tests.
- The CI jsoo gate runs `dune build` only, not a node smoke. A node step could be added
  if the team wants CI-level smoke coverage.
- `docs.yml` uses `opam exec --` (default switch from setup-ocaml action) for the bundle
  build, which is correct given `setup-ocaml@v3` sets the active switch.
