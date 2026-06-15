# Website / Documentation Audit — 2026-06-02

Audit of the Sarek/SPOC site (`gh-pages/` Jekyll source → deployed on the `gh-pages`
branch by `docs.yml` on push to `main`, `clean: true`). Two independent passes:
deployed-docs content, and UI/UX + source structure.

**Correction to an earlier assumption:** the docs ARE source-controlled (19 `.md` files
in `gh-pages/docs/`); an earlier `ls` showing them empty was a tooling artifact. Source ↔
deploy alignment is sound — changes to `gh-pages/` reach the live site on the next `main`
deploy.

Legend: **[H]** high / **[M]** medium / **[L]** low. ✅ = fixed in this PR. 🔧 = proposed.

---

## A. UI / UX

| # | Finding | Sev | Status |
|---|---|---|---|
| A1 | Docs dropdown was unreachable by mouse — 8px `margin-top` gap dropped `:hover` mid-travel | H | ✅ CSS `::before` bridge |
| A2 | Dropdown is hover-only: no keyboard, no touch, `<a href="#">` toggle (mobile users can't open Docs) | H | ✅ button + ARIA + JS + focus |
| A3 | `_layouts/default.html` references missing `css/main.css` / `css/syntax.css`; dead layout that breaks if used | M | 🔧 (see note) |
| A4 | Duplicate ~30-line nav markup in `page.html` and `index.html` (DRY) | M | 🔧 `_includes/navbar.html` (Liquid; not applied — no local Jekyll to validate) |
| A5 | No active-nav indicator (user can't tell which page they're on) | L | 🔧 |
| A6 | `--primary-color` `#f39c12` link contrast fails WCAG AA: ~2.2:1 on white, ~3.8:1 on dark nav | M | 🔧 (brand-color change — needs owner sign-off; report only) |
| A7 | No mobile nav collapse (<768px links reflow/overflow); hamburger needed | M | 🔧 |
| A8 | Copy-code button hover-only (hidden on touch); theme toggle is a 32px emoji-only tap target | L | 🔧 |

**A3 note:** I did not delete `default.html` in this pass because I could not run Jekyll
locally to confirm no page resolves `layout: default`. Safe follow-up: confirm no
front-matter uses it, then delete (or point it at `modern.css`).

## B. Missing docs

| # | Finding | Sev | Status |
|---|---|---|---|
| B1 | No backend/device-selection guide: `SPOC_DISABLE_{GPU,CUDA,OPENCL,VULKAN,METAL}`, per-test `--vulkan/--native/--interpreter/-d`, device enumeration | H | ✅ new `docs/device_selection.md` |
| B2 | FAQ lacks operational troubleshooting (driver faults, "context lost", ICD setup, backend disabling) | H | ✅ troubleshooting section added to `faq.md` |
| B3 | No user-facing Sarek DSL reference (intrinsics, `[%kernel]`/`let%shared`/`let%superstep`, types). "API" nav points to low-level odoc only | M | 🔧 |
| B4 | `CONTRIBUTING.md` and the rich `kb/` are not linked from the site | M | 🔧 |

## C. Deprecated / outdated docs

| # | Finding | Sev | Status |
|---|---|---|---|
| C1 | `getting_started.md` example uses `Device.get_default` and an `Execute.run … [Vec a;…]` form not present in the current API (e2e uses `Execute.run_vectors ~device ~ir ~args ~block ~grid`) | H | 🔧 (NOT auto-fixed — correct user-facing API must be confirmed first; fixing blindly would replace wrong docs with wrong docs) |
| C2 | `backends.md` lists CUDA "Dynamic Parallelism" with no support in `sarek-cuda/`; Vulkan "Android" support is aspirational | M | 🔧 |
| C3 | Clone URL drift: docs use `…/Sarek.git`, the git remote is `…/SPOC.git` (and PRs resolve under `mathiasbourgoin/Sarek`) — the canonical name needs confirming, then align everywhere | M | 🔧 |
| C4 | `live_transpiler.md` / `live_mandelbrot.md` advertise interactive elements that are placeholders | L | 🔧 (clarify status) |

## D. Inconsistent docs

| # | Finding | Sev | Status |
|---|---|---|---|
| D1 | SPOC vs Sarek vs Spoc used interchangeably; intro pages conflate "DSL" (Sarek) and "runtime/SDK" (SPOC) | M | 🔧 (one-paragraph "Sarek = DSL, SPOC = runtime" framing on intro pages) |
| D2 | "API" nav → odoc with no signpost ("writing kernels → Concepts; extending SPOC → API") | M | 🔧 |
| D3 | Example code style varies (`open Sarek` vs qualified; differing `Execute` arg forms) | L | 🔧 (resolve once C1's API is confirmed) |

## E. Additional docs worth adding

| # | Finding | Sev | Status |
|---|---|---|---|
| E1 | Device-selection guide | H | ✅ (B1) |
| E2 | Troubleshooting / known issues (incl. the Renoir+Mesa-clover OpenCL "context lost" fault → use rusticl / Vulkan / CPU) | H | ✅ (B2) |
| E3 | "For developers" section linking `CONTRIBUTING.md`, backend READMEs, `kb/` | M | 🔧 |

## F. Infrastructure / security (from KB cross-reference)

| # | Finding | Sev | Status |
|---|---|---|---|
| F1 | `benchmark-viewer.js` writes benchmark-JSON-derived fields (descriptions, hostname, device name) via `innerHTML` → stored-XSS if untrusted data is accepted (`kb/support/docs-site.md`) | H | 🔧 (use `textContent`/escaping; separate security follow-up) |
| F2 | `dependencies/{CL,Cuda}/LICENCE` labeled GPLv3 but are Khronos / NVIDIA terms — provenance mismatch (`kb/support/dependencies.md`) | M | 🔧 |
| F3 | Two benchmark dashboards (`index.md` + `dashboard.md`); the older `dashboard.js` knows only 4 benchmarks — stale, should unify/deprecate | L | 🔧 |

---

## Implemented in this PR
- **A1** dropdown hover bridge (CSS).
- **A2** accessible dropdown — `<button>` toggle, `aria-expanded`/`aria-controls`, click + keyboard (Enter/Space/Esc) + outside-click close, `:focus-visible`; CSS `:hover` retained for mouse.
- **B1/E1** new `docs/device_selection.md`.
- **B2/E2** troubleshooting section in `docs/faq.md` (covers the real-world Renoir/clover OpenCL fault).

## Deferred (proposed, not applied)
Everything marked 🔧. The larger reasons not to auto-apply now:
- **No local Jekyll** → Liquid-template changes (`_includes` DRY, `default.html` removal) can't be build-validated; deferred to avoid a deploy break.
- **C1/C3/D3** require confirming the canonical user-facing API and repo name before editing, or the "fix" just substitutes one inaccuracy for another.
- **A6** changes the brand color — owner decision.
- **F1** is a security fix deserving its own focused, tested change.
