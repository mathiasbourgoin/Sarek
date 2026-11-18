# GitHub Pages Sources

This directory contains the Jekyll sources for Sarek's GitHub Pages documentation site.

## Structure

- Jekyll site sources (markdown, layouts, assets)
- Combined with odoc API documentation during CI build
- Deployed to `gh-pages` branch automatically by CI

## Local Development

To preview the Jekyll site locally:

```bash
cd gh-pages
bundle install --path ../vendor/bundle
bundle exec jekyll serve
```

Visit `http://localhost:4000/Sarek/`

## Deployment

The `.github/workflows/docs.yml` workflow automatically:
1. Builds the Jekyll site from this directory
2. Generates odoc API documentation
3. Combines both into `spoc_docs/`
4. Deploys to `gh-pages` branch

## Making Changes

- Edit markdown files in `gh-pages/docs/` for documentation updates
- Edit layouts in `gh-pages/_layouts/` for template changes
- Changes merged to `main` branch will trigger automatic deployment

## Talk PDFs

The 2013-2016 talk PDFs (about 38 MB) are not in `main`. They live only on the
`gh-pages` branch under `docs/talks/` and are served at
`https://mathiasbourgoin.github.io/Sarek/docs/talks/<name>.pdf`;
`docs/publications.md` links to those URLs. `docs.yml` deploys with
`clean: true`, so it lists `docs/talks` in `clean-exclude` to keep them. To add
or replace one, commit it to the `gh-pages` branch directly (never to `main`).
