# Plan — Sarek/SPOC Website & Docs Overhaul

**Date:** 2026-06-02
**Status:** DRAFT — for approval
**Theme:** make the site self-maintaining (docs derived from tested code), then add a
flagship in-browser transpiler, a real book, and an interactive benchmarks portal.

Builds on PR #142 (audit + dropdown a11y + device-selection/troubleshooting docs).

---

## Guiding principle: auto-sync or it rots

Every audit finding about *drift* (outdated API examples, scattered capability info,
hidden internals) has the same root cause: docs hand-written separately from code. The
overhaul's spine is **generate from ground truth at build time** — tested example
sources, the intrinsic registry, backend capability flags, the KB.

---

## Phase 0 — Enabler: extract a pure frontend+codegen library

Architectural prerequisite for the playground (Phase 2) and good design regardless.

- Create `sarek_codegen` (FFI-free): move the pure `Sarek_ir_{cuda,opencl,metal,glsl}.ml`
  out of the `ctypes`-linked backend packages; depend only on `sarek_ir`.
- Isolate the 1 FFI ref in `Sarek_ir_types.ml` (native-vector helper) behind a boundary so
  the IR types are jsoo-clean.
- Narrow `sarek_ppx_lib`'s `spoc_core` dependency to types-only (confirm parse/typer don't
  touch FFI).
- Expose a pure entry point: `Sarek_transpile.of_source : string -> (backend * string) list`
  (parse OCaml → Sarek AST → type → IR → per-backend source), usable from native AND jsoo.
- **Validates:** existing build + e2e still green; the backends keep working (they just
  consume `sarek_codegen`). Pure code move — same review discipline as the recent refactor.

## Phase 1 — Quick, high-trust wins (ship first)

**1a. Docs from tested examples (#2).** A build step renders the example pages
(vector_add, matrix_mul, reduction, transpose) from `sarek/tests/e2e/*.ml` sources +
their kernels, so published examples always compile and match the current API. Kills the
outdated-example drift (audit C1/D3) structurally.

**1b. Backend × feature compatibility matrix (#3).** Generate a single table — backends ×
{fp64, atomics, shared mem, variants, records, barriers, subgroups} × platforms — from
backend capability flags + `kb/backends/*`. Answers the perennial "does Metal do fp64?".

## Phase 2 — Flagship: in-browser transpiler playground (#1)

- Compile `Sarek_transpile.of_source` (Phase 0) to JS with **js_of_ocaml** (parsing OCaml
  source in-browser via compiler-libs/ppxlib — proven by existing OCaml playgrounds).
- UI: editor pane + tabbed output (CUDA / OpenCL / GLSL / MSL), live on keystroke,
  shareable via URL-encoded source. Reuses the site's existing code-tabs.
- Replaces the dead `live_transpiler.html` / `live_mandelbrot.html` placeholders with a
  real feature. Source→source only (no execution) — no GPU, no backend runtime needed.
- **Risks:** jsoo bundle size (compiler-libs is large — mitigate with lazy load); error
  reporting for invalid kernels (surface the structured parse/type errors).

## Phase 3 — Depth

**3a. "The Sarek Book" (#4, reframed).** A structured, accessible **book/tutorial** — not a
KB dump. Progressive track: first kernel → memory & transfers → reductions → custom types
(records/variants) → multi-backend → performance/convergence → writing a backend. Rich and
comprehensive, but **auto-synced**: code snippets from tested sources (Phase 1a), the DSL
reference (intrinsics, syntax, types) generated from the intrinsic registry, architecture
chapters distilled (curated, rewritten for readability) from `kb/` — with a build check
that flags when referenced symbols/files drift. The KB stays the internal source of truth;
the Book is the readable, public, synced surface over it.

**3b. Interactive benchmarks portal (#5).** Turn the JSON dashboard into an explorable
performance story: cross-device/backend comparison, hardware filters, historical trends,
per-kernel generated-code view, "reproduce this" commands. Fix the `benchmark-viewer.js`
`innerHTML` stored-XSS (audit F1) as part of the rebuild.

---

## Sequencing & rationale

1. **Phase 1 first** — fastest payoff, makes the site trustworthy, no prerequisites.
2. **Phase 0** — needed before Phase 2; do it as its own reviewed pure-refactor PR.
3. **Phase 2** — the flagship/demo, once Phase 0 lands.
4. **Phase 3** — the long game; the Book reuses Phase 1a/2 machinery (snippets, transpiler).

Each phase is its own PR(s) with the usual review+QA gates. Phase 0 and the
docs-generation steps must keep `@sarek/tests/runtest` green; Phase 2/3 are gated by the
Jekyll `deploy-preview`.

## Decisions (made 2026-06-02)
- **Start with Phase 1** (quick wins: docs-from-tested-examples + compatibility matrix).
- **The Book (3a) → dedicated docs engine (mdBook / Docusaurus)** for first-class book UX
  (search, sidebar, versioning), accepting a second toolchain/deploy path alongside Jekyll.
  Engine choice (mdBook vs Docusaurus) to be finalized at Phase 3 kickoff; lean mdBook if
  the Book stays Markdown-centric, Docusaurus if it needs React interactivity (e.g.
  embedding the Phase-2 playground).
