# Documentation Site and Notebooks

## Component Inventory

- `gh-pages/`: Jekyll site sources, benchmark dashboard, docs, examples, layouts, static assets, legacy notebooks, talks, images, and generated benchmark descriptions.
- `docs/odoc_custom.css`: style overlay appended to generated odoc CSS.
- `notebooks/Introduction_to_Sarek.ipynb`: current Binder/Jupyter introduction notebook.
- `gh-pages/static/**`: vendored/static browser assets for legacy IOCaml/Jupyter pages.
- `gh-pages/docs/talks/**`, `gh-pages/pres_resources/**`, images and PDFs: static archival media.
- **Missing from prior inventory, added 2026-07-02:**
  - `gh-pages/learn/`: interactive GPU course — `index.md` plus 9 lesson pages (`01-vector-add.html`, `02-scale-saxpy.html`, `03-map-square.html`, `04-bounds-if.html`, `05-mandelbrot.html`, `06-image-filter.html`, `07-compose.html`, `08-shared-barrier.html`, `09-reduction.html`) and `gh-pages/learn/test/` (Playwright GPU acceptance scripts run by `make lessons-gpu-test` / `make compose-gpu-test`).
  - `gh-pages/playground.html`: in-browser Sarek-to-WGSL transpiler playground.
  - `gh-pages/javascripts/sarek_lesson.js`: lesson harness that injects kernel bodies into a lesson page and grades PASS/FAIL against expected output.
  - `gh-pages/javascripts/sarek_webgpu_runner.js`: runs transpiled WGSL kernels on the visitor's own GPU via WebGPU.

## Per-File Purpose

- `gh-pages/_config.yml`: Jekyll markdown/baseurl/path configuration and includes benchmark data.
- `gh-pages/Gemfile`: Jekyll dependency entrypoint.
- `gh-pages/_layouts/*.html`: site page/index/default layouts.
- `gh-pages/index.md`: documentation home page.
- `gh-pages/docs/*.md`: Sarek docs, backend guides, FAQ, architecture, publications, and redirect stubs.
- `gh-pages/examples/*.md`: example pages for vector add, matrix mul, transpose, reduction, mandelbrot.
- `gh-pages/benchmarks/index.md`: primary interactive benchmark viewer page.
- `gh-pages/benchmarks/dashboard.md`: older/alternate multi-chart dashboard page.
- `gh-pages/javascripts/benchmark-viewer.js`: main benchmark dashboard, markdown description loader, charts, filters, system-info rendering.
- `gh-pages/javascripts/benchmark-dashboard.js`: older four-chart dashboard implementation.
- `gh-pages/javascripts/*.js`: theme toggle, code tabs, syntax helpers, copy-code, relative-link processing, Mermaid init, IOCaml saver, MathJax/screenfull wrappers.
- `gh-pages/css/*.css`, `gh-pages/stylesheets/*.css`, `docs/odoc_custom.css`: site, syntax, modern theme, and odoc styling.
- `gh-pages/benchmarks/descriptions/*.md`: benchmark prose and markers for generated backend code tabs.
- `gh-pages/benchmarks/descriptions/generated/*.md`: generated backend code snapshots copied from `benchmarks/descriptions/generated`.
- `gh-pages/benchmarks/data/latest.json`: generated web benchmark data.
- `gh-pages/static/**`: vendored legacy Jupyter/IOCaml dependencies, marked static/vendor.
- `gh-pages/notebooks/*.ipynb`: legacy IOCaml/WebSPOC demo notebooks.
- `notebooks/Introduction_to_Sarek.ipynb`: current minimal OCaml 5.4 Sarek notebook.
- `gh-pages/docs/talks/*.pdf`, `gh-pages/pres_resources/*`, `gh-pages/docs/lena.png`, favicon and benchmark images: static media.

## Features and APIs

- Benchmark page loads Chart.js and Prism from CDNs, maps CUDA/OpenCL/Metal aliases, then calls `loadBenchmarkData('{{ site.baseurl }}/benchmarks/data/latest.json')` (`gh-pages/benchmarks/index.md:457-473`).
- `benchmark-viewer.js` supports single chart, comparison, ranking, and matrix views with system/backend filters.
- Benchmark descriptions are fetched from markdown files and generated code snippets are fetched from `descriptions/generated/*_generated.md` (`gh-pages/javascripts/benchmark-viewer.js:315-353`, `gh-pages/javascripts/benchmark-viewer.js:472-491`).
- `docs.yml` combines Jekyll output and odoc output under `_site/spoc_docs` (`.github/workflows/docs.yml:273-289`).
- Legacy notebooks preserve historical WebSPOC/IOCaml browser demos and static presentation content.

## Invariants

- `gh-pages/spoc_docs/**` is generated odoc and excluded from semantic review.
- Benchmark selector options in `gh-pages/benchmarks/index.md` must match `BENCHMARK_CONFIGS` keys in `benchmark-viewer.js`.
- `BENCHMARK_CONFIGS` variants must match result JSON `benchmark.name` values.
- All untrusted benchmark JSON string fields and markdown-derived content should be escaped before insertion into DOM.
- Static/vendor assets should be separated from first-party JS/CSS so updates and license checks are tractable.

## Potential Invariant Violations or Bugs

- **fixed 2026-07-02 (merged), `benchmark-viewer.js` only:** `markdownToHtml` previously performed regex replacement into HTML without escaping general markdown text, then wrote it through `descDiv.innerHTML`, letting a malicious benchmark description inject HTML/JS. Current source (`gh-pages/javascripts/benchmark-viewer.js`) now calls `escapeHtml(markdown)` as the *first* step inside `markdownToHtml` (`:546-547`, comment: "escapeHtml() ... every substitution below runs on top of that"), so every subsequent regex substitution operates on already-escaped text; `escapeHtml` itself is defined at `:45`. Link handling is now http(s)-only: URLs are scheme-checked before being rendered as a real `<a>` tag, otherwise rendered as plain escaped text (`:543-583`, `\/\//i.test(url.trim())` guard plus the surrounding comment "(http/https only ...) links ... renders as plain escaped text").
- **fixed 2026-07-02 (merged), `benchmark-viewer.js` only:** benchmark JSON system and device fields were previously interpolated into HTML without escaping before `infoDiv.innerHTML`/`container.innerHTML` assignment. Current source wraps every such field in `escapeHtml(...)`: system/device info block (`gh-pages/javascripts/benchmark-viewer.js:1083-1101`) and matrix-view device labels (`gh-pages/javascripts/benchmark-viewer.js:1835`).
- **NOT fixed — separate file, out of scope for #213/#214 (verified 2026-07-02):** `gh-pages/javascripts/benchmark-dashboard.js` (the older four-chart dashboard, loaded by `gh-pages/benchmarks/dashboard.md:210`) has zero `escapeHtml` calls against 12 raw `innerHTML` writes (`benchmark-dashboard.js:71,126,156,170,274,291,381,393,481,497,719,726`). Its XSS sinks are unfixed. **Correction to the prior KB framing:** this file was previously described only as "stale" (four benchmark groups vs. the larger suite `benchmark-viewer.js` covers) with an implicit "probably dead code, consider deleting" framing. That undersold the risk — `dashboard.md` builds a real, URL-addressable page (see `kb/support/README.md` for the full orphan-page analysis: reachable by direct URL, not linked from any site nav found in `gh-pages/_layouts/`, `gh-pages/_config.yml`, or other `.md` pages). It is live-but-orphaned, not dead, and its unescaped `innerHTML` sinks are a real open XSS surface. Whether to link it into navigation (raising priority for a fix) or remove/archive it (removing the surface) is an open question for a human — this KB does not decide it.
- CDN scripts/styles in `gh-pages/benchmarks/index.md:457-463` and `gh-pages/benchmarks/dashboard.md:209` do not include Subresource Integrity attributes.
- `gh-pages/_config.yml` uses `path: http://mathiasbourgoin.github.io/Sarek` (`gh-pages/_config.yml:3`) while most modern pages are HTTPS; mixed-scheme links can drift.

## Performance and Maintainability Risks

- `benchmark-viewer.js` is over 2k lines and duplicates benchmark metadata in multiple places, including a comment warning about duplication in chart config (`gh-pages/javascripts/benchmark-viewer.js:639-645`).
- Client-side markdown parsing by regex is brittle for tables, nested lists, links with parentheses, and HTML escaping.
- Chart rendering can become crowded with many systems/devices; top-20 ranking truncation is ad hoc.
- Legacy notebook JSON files are one-line nbformat/old-worksheet dumps in `gh-pages/notebooks`, making review diffs noisy.
- Vendored Jupyter/CodeMirror/jQuery assets under `gh-pages/static/**` are old and static; security posture depends on the site not using them in active trusted contexts.

## Related Tests and Checks

- Docs deploy workflow builds Jekyll and odoc on pushes to `main` (`.github/workflows/docs.yml`).
- PR preview workflow builds the Jekyll site for same-repo PRs touching docs/benchmarks (`.github/workflows/deploy-pr-preview.yml`).
- Generated benchmark descriptions are checked by CI (`.github/workflows/ci.yml:117-191`).
- No browser automation or static-site sanitizer tests were found.

## Missing Tests

- Browser smoke test for `gh-pages/benchmarks/index.md` loading `latest.json`.
- DOM-safety tests for benchmark JSON and markdown inputs containing `<script>`, event attributes, `javascript:` URLs, and malformed tables.
- Link checker for internal docs/examples/benchmark description links.
- SRI/pinned CDN policy check.
- Notebook validation/nbformat normalization check.

## Concrete Improvement Candidates

- ~~Replace `markdownToHtml` with a maintained markdown parser configured with sanitization, or sanitize output with a strict allowlist before `innerHTML`.~~ **DONE 2026-07-02 for `benchmark-viewer.js`** — escape-first `markdownToHtml` plus http(s)-only link gating. **Still open for `benchmark-dashboard.js`.**
- ~~Use `textContent` or DOM creation for all system/device/benchmark JSON fields.~~ **DONE 2026-07-02 for `benchmark-viewer.js`** via `escapeHtml()`. **Still open for `benchmark-dashboard.js`.**
- Delete or clearly archive `benchmark-dashboard.js` if `benchmark-viewer.js` is canonical. **Now a live decision, not a cleanup nicety:** the page is reachable by direct URL today with unfixed XSS sinks — see the orphan-page open question in `kb/support/README.md` and the corrected framing above. Recommend resolving this explicitly (link it in, or remove it) rather than leaving it ambiguous.
- Add SRI and version review for CDN dependencies, or vendor modern copies with license metadata.
- Normalize legacy notebooks to stable nbformat or move them under an explicit archive path.
