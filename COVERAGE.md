# Code Coverage Measurement

This project uses [bisect_ppx](https://github.com/aantron/bisect_ppx) for code coverage measurement.

## Requirements

- bisect_ppx 2.8.3.1~alpha-repo (for OCaml 5.4 compatibility)
- Installed from the opam alpha repository: https://github.com/kit-ty-kate/opam-alpha-repository

## Setup

bisect_ppx is already configured in the dune files for the `sarek` library and will automatically instrument code when built with the `--instrument-with bisect_ppx` flag.

## Running Coverage

Four scripts are provided for different test scenarios:

### 1. Unit Tests Only
```bash
./scripts/coverage-unit.sh
```
Runs unit tests in `sarek/tests/unit/` and generates coverage report at `_coverage/unit-report/index.html`.

### 2. E2E Tests
```bash
./scripts/coverage-e2e.sh
```
Runs end-to-end tests with the Native backend and generates coverage report at `_coverage/e2e-report/index.html`.

### 3. Benchmarks
```bash
./scripts/coverage-benchmarks.sh
```
Runs benchmark tests with small problem sizes using the Native backend and generates coverage report at `_coverage/benchmarks-report/index.html`.

Note: Does not use `--benchmark` flag to avoid running on all devices (which can fail on some backends).

### 4. Aggregate (All Tests)
```bash
./scripts/coverage-aggregate.sh
```
Runs all three test suites sequentially and generates:
- Individual reports for each test type
- Aggregate report combining all coverage data at `_coverage/aggregate-report/index.html`

## Understanding Coverage Reports

- HTML reports show source files with line-by-line coverage
- Green lines were executed, red lines were not
- Summary shows overall coverage percentage
- Coverage data files (`.coverage`) are stored in `_coverage/`

## Notes

- E2E and benchmark scripts use `--native` backend for consistent, reproducible coverage
- Benchmark scripts use small problem sizes to reduce execution time
- Coverage reports are generated in `_coverage/` which is git-ignored
- All scripts automatically clean previous coverage data before running
