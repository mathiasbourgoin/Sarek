Jekyll setup for benchmark preview

This file explains how to set up Jekyll locally to preview the benchmark web viewer used by `make bench-preview`.

Prerequisites
- Ruby (>= 3.0)
- Bundler

Steps
1. Install bundler and dependencies:

```bash
cd gh-pages
gem install bundler
bundle install
```

2. Build the site locally and preview the benchmarks page:

```bash
bundle exec jekyll build --destination _site
bundle exec jekyll serve --watch --destination _site --baseurl="/Sarek"
# open http://127.0.0.1:4000/Sarek/benchmarks/
```

3. If you want to preview a PR using the preview workflow, push a branch and open the PR; the Deploy PR Preview workflow will generate a preview under `/preview/pr-<N>/benchmarks/`.

Notes
- The repository uses GitHub Pages with `baseurl: /Sarek` so use the `--baseurl` flag when serving locally.
- If you only need to regenerate the web data after running benchmarks, run `dune exec benchmarks/to_web.exe -- gh-pages/benchmarks/data/latest.json benchmarks/results/*.json` from the repository root.
