# Documentation Site and Notebooks

## Component Inventory

- `gh-pages/`: Jekyll site sources, benchmark dashboard, docs, examples, layouts, static assets, legacy notebooks, talks, images, and generated benchmark descriptions.
- `docs/odoc_custom.css`: style overlay appended to generated odoc CSS.
- `notebooks/Introduction_to_Sarek.ipynb`: current Binder/Jupyter introduction notebook.
- `gh-pages/static/**`: vendored/static browser assets for legacy IOCaml/Jupyter pages.
- `gh-pages/docs/talks/**`, `gh-pages/pres_resources/**`, images and PDFs: static archival media.

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

- `markdownToHtml` performs regex replacement into HTML without escaping general markdown text (`gh-pages/javascripts/benchmark-viewer.js:507-558`), then writes it through `descDiv.innerHTML` (`gh-pages/javascripts/benchmark-viewer.js:491`). A malicious benchmark description can inject HTML/JS.
- Benchmark JSON system and device fields are interpolated into HTML without escaping (`gh-pages/javascripts/benchmark-viewer.js:1054-1067`) and then assigned with `infoDiv.innerHTML` (`gh-pages/javascripts/benchmark-viewer.js:1080`). A malicious `hostname`, OS, CPU model, or device name in submitted benchmark JSON can execute markup.
- Matrix rendering builds rows from device/backend labels and writes `container.innerHTML` (`gh-pages/javascripts/benchmark-viewer.js:1795-1837`), also trusting benchmark data strings.
- `benchmark-dashboard.js` appears stale: it knows only four benchmark groups (`gh-pages/javascripts/benchmark-dashboard.js:33-59`) while `benchmark-viewer.js` and `gh-pages/benchmarks/index.md` cover the larger suite.
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

- Replace `markdownToHtml` with a maintained markdown parser configured with sanitization, or sanitize output with a strict allowlist before `innerHTML`.
- Use `textContent` or DOM creation for all system/device/benchmark JSON fields.
- Delete or clearly archive `benchmark-dashboard.js` if `benchmark-viewer.js` is canonical.
- Add SRI and version review for CDN dependencies, or vendor modern copies with license metadata.
- Normalize legacy notebooks to stable nbformat or move them under an explicit archive path.
