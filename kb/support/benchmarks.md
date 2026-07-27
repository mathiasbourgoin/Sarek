# Benchmarks Support

## Component Inventory

- Shared benchmark libraries: `benchmarks/common.ml`, `benchmarks/system_info.ml`, `benchmarks/output.ml`, `benchmarks/benchmark_runner.ml`.
- Backend selectors: `benchmarks/backend_loader.ml`, `benchmarks/backend_{cuda,opencl,vulkan,metal}.{available,unavailable}.ml`.
- Workload executables: `benchmarks/bench_*.ml`.
- Result tooling: `benchmarks/aggregate.ml`, `benchmarks/to_web.ml`, `benchmarks/to_csv.ml`, `benchmarks/deduplicate_results.ml`.
- Generated-code tooling: `benchmarks/generate_backend_code.ml`, `benchmarks/descriptions/generated/*.md`.
- Runner/docs: `benchmarks/run_all_benchmarks.sh`, `benchmarks/dune`, `benchmarks/README.md`, `CONTRIBUTING.md`, `INFRASTRUCTURE.md`, `OUTPUT_FORMATS.md`, `TODO.md`, `JEKYLL_SETUP.md`, and `benchmarks/descriptions/*.md`.

## Per-File Purpose

- `benchmarks/common.ml`: statistics, timing, git/timestamp helpers, array comparison, and GPU timing helpers.
- `benchmarks/system_info.ml`: collects host, OS, CPU, memory, and device metadata for result files.
- `benchmarks/output.ml`: canonical benchmark result OCaml types, JSON/CSV writers, failed-result filtering, and filename generation.
- `benchmarks/benchmark_runner.ml`: shared CLI parser and multi-device/size runner used by newer benchmarks.
- `benchmarks/backend_loader.ml`: initializes native/interpreter and optional GPU backends.
- `benchmarks/backend_*.available.ml`: calls the matching backend plugin init, gated on `SPOC_DISABLE_GPU`/`SPOC_DISABLE_<BACKEND>` env vars (added 2026-07-02 to KB — e.g. `benchmarks/backend_cuda.available.ml:8-9` checks `SPOC_DISABLE_GPU` and `SPOC_DISABLE_CUDA` before calling `Sarek_cuda.Cuda_plugin.init ()`; same pattern for `_opencl`/`_vulkan`/`_metal`); `backend_*.unavailable.ml` is a no-op stub.
- `benchmarks/bench_matrix_mul.ml`, `bench_matrix_mul_tiled.ml`, `bench_vector_add.ml`, `bench_vector_copy.ml`, `bench_stream_triad.ml`, `bench_reduction.ml`, `bench_reduction_max.ml`, `bench_dot_product.ml`, `bench_transpose.ml`, `bench_transpose_tiled.ml`, `bench_mandelbrot.ml`, `bench_nbody.ml`, `bench_conv2d.ml`, `bench_stencil_2d.ml`, `bench_scan.ml`, `bench_bitonic_sort.ml`, `bench_histogram.ml`, `bench_gather_scatter.ml`, `bench_radix_sort.ml`: individual workloads and verification logic.
- `benchmarks/aggregate.ml`: wraps many result JSON files into a single aggregate JSON.
- `benchmarks/to_web.ml`: wraps result JSON files into `{"results": [...], "updated_at": ...}` for the website.
- `benchmarks/to_csv.ml`: converts one result JSON or an aggregate file to CSV.
- `benchmarks/deduplicate_results.ml`: scans `benchmarks/results` for duplicate hostname/benchmark/size/device/backend keys and removes extras unless `--dry-run`.
- `benchmarks/generate_backend_code.ml`: emits generated CUDA/OpenCL/Vulkan/Metal snippets into markdown descriptions.
- `benchmarks/run_all_benchmarks.sh`: builds and executes all workloads, moves JSON into `benchmarks/results`, updates web data, and optionally regenerates backend code.
- `benchmarks/descriptions/*.md`: human-authored benchmark descriptions; `descriptions/generated/*.md` are generated code snapshots.
- `benchmarks/dune`: Dune libraries/executables and optional backend `select` rules.

## Features and APIs

- CLI flags include `--sizes`, `--iterations`, `--warmup`, `--block-size`, `--output`, and `--all-backends` in the shared runner (`benchmarks/benchmark_runner.ml:39-69`).
- Result JSON has `benchmark`, `system`, and `results` top-level fields (`benchmarks/output.ml:67-89`).
- Failed device results are filtered before writing (`benchmarks/output.ml:91-103`).
- Filename convention is `<output>/<hostname>_<benchmark>_<size>_<timestamp>.json` (`benchmarks/output.ml:164-173`).
- Web conversion keeps raw benchmark result objects under `results` (`benchmarks/to_web.ml:46-68`).
- Dune builds all workload executables and tools, with GPU backends selected only when matching packages are installed (`benchmarks/dune:1-42`, `benchmarks/dune:41-220`).

## Invariants

- Timing should exclude first compilation/driver warmup work; `benchmark_gpu` does a compile-trigger run, warmups, then timed iterations with device synchronization (`benchmarks/common.ml:200-237`).
- JSON consumers expect a `results` array whose entries carry per-device `framework`, timing fields, `throughput_gflops`, and `verified` when available.
  **The array is not guaranteed non-empty, and this file used to say it was.**
  `Output.write_json` (`benchmarks/output.ml:101-103`) filters through
  `filter_valid_results` (`:93-99`) before writing, so a run in which *every*
  device result failed — empty `iterations`, or `median_ms = 0.0` with
  `verified = Some false` — writes `"results": []`. That is a reachable state, not
  a hypothetical, and it is the same one the `to_web` failure at `:54` below
  describes from the consumer end (`benchmarks/to_web.ml:41-44`, "No valid results
  to convert", `exit 1`).
  Stated as an invariant the way it was, the sentence was **unsatisfiable in that
  case** and would have had a reader believe a guarantee the writer does not make.
  The accurate contract today is: *at least one successful device result is
  required for a non-empty array, and nothing enforces that there is one.*
  Deliberately qualified rather than tightened: this file is a descriptive audit of
  what the code does, and writing a contract the code does not implement would make
  it wrong in the more dangerous direction. The fix, if a real non-empty guarantee
  is wanted, is already recorded below under Concrete Improvement Candidates —
  write failed device records with error messages instead of filtering them out —
  which also closes the postmortem gap noted at `:63`.
- Benchmark names in generated JSON must match `BENCHMARK_CONFIGS` variants in `gh-pages/javascripts/benchmark-viewer.js`.
- `run_all_benchmarks.sh` expects each benchmark to honor `--output` and produce JSON in the temporary run directory (`benchmarks/run_all_benchmarks.sh:109-122`, move-to-`benchmarks/results/` and web-data update at `:174-203`). **Citation fix (2026-07-02):** the previous line refs (`:2328-2341`, `:2393-2421`) were byte offsets mistaken for line numbers — the script is 259 lines total, so those numbers pointed past the end of the file.
- Generated backend descriptions should be reproducible from `benchmarks/generate_backend_code.ml`; CI enforces this.

## Potential Invariant Violations or Bugs

- `Common.make_result` drops the first timing value from statistics (`benchmarks/common.ml:85-99`), but many benchmarks compute `Common.mean`, `Common.median`, and `Common.min` directly on the raw `times` arrays. This means statistics are inconsistent across workloads.
- Several older benchmarks return `device_id = 0` inside per-device results and rely on later list indexing to overwrite it (`benchmarks/bench_matrix_mul.ml:153`, `benchmarks/bench_vector_add.ml:136`, `benchmarks/bench_reduction.ml:201`, `benchmarks/bench_transpose.ml:198`). This is harmless only when every caller remembers to rewrite IDs.
- `benchmarks/bench_vector_add.ml` records `block_size = 0` in JSON despite using a 256-thread block (`benchmarks/bench_vector_add.ml:82`, `benchmarks/bench_vector_add.ml:227`).
- `benchmarks/deduplicate_results.ml` deduplicates a multi-device JSON file using only the first device result (`benchmarks/deduplicate_results.ml:297-308`). Files with the same first device but different additional devices can be collapsed incorrectly; files with duplicate second devices may be missed.
- `Output.write_csv_row` writes CSV fields without escaping names/hostnames/frameworks (`benchmarks/output.ml:128-146`), unlike `benchmarks/to_csv.ml` which has `escape_csv_field` (`benchmarks/to_csv.ml:10-19`).
- `run_all_benchmarks.sh` always runs `to_web` after the move phase (`benchmarks/run_all_benchmarks.sh:198-201`, real line numbers; corrected 2026-07-02 from the byte-offset citation `:2417-2421`). If no JSON exists, shell glob handling can pass a literal `benchmarks/results/*.json`, and `to_web` exits with "No valid results" (`benchmarks/to_web.ml:41-44`). This glob at `:201` runs after `set -e` is re-enabled at `:146`, so it is also the "unguarded glob under `set -e`" gap noted separately in [scripts-ci.md](scripts-ci.md)/[README.md](README.md).
- **fixed 2026-07-02 (merged)** — unknown `--flag` values previously put the CLI arg parser into an infinite loop (the `while` loop's default case only called `shift` when the argument did *not* start with `--`, so an unrecognized `--foo` flag matched neither the known flags nor the "not a `--` flag" guard and was never consumed). Commit `90eb3b2f` ("error on unknown flags instead of looping") added an explicit `--*)` case to the `case` statement that prints "Unknown option: $1" plus usage guidance to stderr and `exit 2` (confirmed in current `benchmarks/run_all_benchmarks.sh`); the previous fallthrough `*)` case now unconditionally treats its argument as `OUTPUT_DIR` and shifts, since the ambiguous "unrecognized `--` flag vs. positional arg" case is now handled separately above it.
- `benchmarks/generate_backend_code.ml` creates only one directory level (`benchmarks/generate_backend_code.ml:456`). It assumes the parent path exists.

## Performance and Maintainability Risks

- Workloads duplicate runner code instead of consistently using `Benchmark_runner.run_benchmark`, so defaults differ (`output_dir = "results"` in `benchmark_runner.ml:30` and `bench_matrix_mul_tiled.ml:34`, but `"benchmarks/results"` in several older files).
- Throughput is always serialized as `throughput_gflops` (`benchmarks/output.ml:55-57`) even when the unit is GB/s, MElements/s, M pixels/s, or G interactions/s. The website handles labels separately, but CSV/schema names are misleading.
- Large default sizes can allocate very large vectors and matrices without explicit memory-capability checks before launch.
- Result filtering removes failed device entries before JSON persistence (`benchmarks/output.ml:101-103`), making postmortem analysis harder unless console logs are preserved.

## Related Tests and Checks

- CI builds the benchmark generated-code tool and checks generated descriptions for drift (`.github/workflows/ci.yml:117-191`).
- `make benchmarks`, `make bench-all`, `make bench-deduplicate`, `make bench-preview`, and `make bench-generate-code` wrap common benchmark workflows (`Makefile:389-437`).
- `scripts/coverage-benchmarks.sh` runs selected e2e test binaries as small native-backend benchmark coverage (`scripts/coverage-benchmarks.sh:155-193`).

## Missing Tests

- Schema tests for one canonical result fixture, aggregate fixture, web fixture, and CSV output.
- Unit tests for `filter_valid_results`, CSV escaping, filename generation, deduplication keys, and empty-glob behavior.
- Browser/dashboard tests that load representative multi-device result JSON from each workload.
- Tests that all workloads honor shared CLI flags and produce benchmark names that exist in web configs.
- Memory-size guard tests for very large/default workload sizes.

## Concrete Improvement Candidates

- Centralize all workloads on `Benchmark_runner.run_benchmark` and remove local runner variants.
- Introduce a `throughput_unit` field instead of hard-coding `throughput_gflops`.
- Make `deduplicate_results.ml` key each device result separately or include a stable full-device set hash.
- Add a `--strict` or default failure mode that writes failed device result records with error messages instead of filtering them out silently.
- Add a `benchmarks/schema/` fixture suite exercised by Dune tests and the web viewer.
