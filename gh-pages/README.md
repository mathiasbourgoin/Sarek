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
