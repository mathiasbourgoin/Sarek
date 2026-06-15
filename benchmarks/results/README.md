# Benchmark Results

This directory contains benchmark results from different machines and configurations.

## File Format

Benchmark results are saved as JSON files with the naming pattern:
```
{hostname}_{benchmark_name}_{size}_{timestamp}.json
```

## Submitting Results

**Contributors are encouraged to submit benchmark results!**

When adding results in a PR:
1. Run benchmarks on your machine: `opam exec -- dune exec benchmarks/bench_vector_add.exe`
2. Results are automatically saved here
3. Commit and submit your PR with the new result files
4. Reviewers will deduplicate before merging

## Deduplication (For Reviewers)

Before merging PRs with benchmark results, use the deduplication tool:

```bash
# Preview what would be removed (dry run)
opam exec -- dune exec benchmarks/deduplicate_results.exe -- --dry-run

# Remove duplicates, keeping oldest result (default)
opam exec -- dune exec benchmarks/deduplicate_results.exe

# Remove duplicates, keeping newest result
opam exec -- dune exec benchmarks/deduplicate_results.exe -- --keep-latest
```

The tool identifies duplicates based on:
- Same hostname
- Same benchmark name  
- Same size/parameters
- Same device name and backend

This prevents accumulation while still allowing users to submit comprehensive results.

## CI/CD Integration

When a PR is opened or updated:
1. The CI workflow converts all JSON files in this directory to `gh-pages/benchmarks/data/latest.json`
2. The Jekyll site is built with the aggregated results
3. A preview is deployed to `https://mathiasbourgoin.github.io/Sarek/preview/pr-{number}/benchmarks/`
