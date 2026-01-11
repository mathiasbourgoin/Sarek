# GitHub Actions Workflows

This directory contains CI/CD workflows for the SPOC/Sarek project.

## Benchmark PR Preview Deployment

### `deploy-pr-preview.yml`

Automatically deploys a preview of benchmark results when PRs modify:
- `gh-pages/**` - Documentation changes
- `benchmarks/results/**` - New benchmark results
- `benchmarks/*.ml` or `benchmarks/dune` - Benchmark tool changes

**What it does:**

1. **Triggers** on PR open/update/reopen
2. **Builds** the `to_web.exe` converter tool
3. **Converts** any benchmark JSON files to web format
4. **Deploys** to `gh-pages` branch under `preview/pr-XXX/`
5. **Comments** on the PR with preview links

**Preview URL format:**
```
https://mathiasbourgoin.github.io/Sarek/preview/pr-123/benchmarks/
```

**Comment includes:**
- Link to preview deployment
- Summary of benchmark results (if any)
- Updates automatically on new commits

### `cleanup-pr-preview.yml`

Automatically cleans up preview deployments when PRs are closed.

**What it does:**

1. **Triggers** when a PR is closed (merged or not)
2. **Removes** the `preview/pr-XXX/` directory from gh-pages
3. **Commits** the cleanup to keep the repo tidy

## Workflow Permissions

Both workflows need:
- `contents: write` - To deploy to gh-pages branch
- `pull-requests: write` - To comment on PRs (deploy only)

## Testing Locally

To test the conversion locally before pushing:

```bash
# Install dependencies
opam install -y yojson

# Build tools
dune build benchmarks/to_web.exe

# Convert results
dune exec benchmarks/to_web.exe -- \
  gh-pages/benchmarks/data/latest.json \
  benchmarks/results/*.json

# Preview locally
cd gh-pages
python3 -m http.server 8000
# Visit http://localhost:8000/benchmarks/
```

## Troubleshooting

### Preview not deploying

Check the Actions tab for workflow run errors. Common issues:

1. **OCaml version mismatch** - Workflow uses 5.4.x
2. **Missing dependencies** - yojson should be installed automatically
3. **Build failures** - Check compilation errors in the logs

### Preview link not appearing

The bot comment requires:
- `pull-requests: write` permission (should be default)
- Successful workflow run
- GitHub Actions not restricted in repo settings

### Old previews not cleaning up

The cleanup workflow needs:
- `contents: write` permission
- Access to gh-pages branch
- Runs on PR close event

## Future Enhancements

Possible improvements:
- [ ] Add preview expiration (e.g., 30 days)
- [ ] Preview size limits for large result sets
- [ ] Diff view comparing PR results vs main
- [ ] Automatic benchmark quality checks
- [ ] Performance regression detection
