# Sarek Negative Tests

## Component Inventory

Negative tests live in `sarek/tests/negative/` and are configured by `sarek/tests/negative/dune`.

## Per-File Purpose

- `sarek/tests/negative/dune`: documents expected compiler errors and declares profile-gated libraries.
- `test_barrier_diverged.ml`: barrier in diverged control flow.
- `test_superstep_diverged.ml`: superstep/barrier convergence violation.
- `test_warp_diverged.ml`: warp collective in diverged control flow.
- `test_unbound_function.ml`: unbound function/variable.
- `test_reserved_keyword.ml`: reserved keyword rejection.
- `test_convention_kernel_fail.ml`: field lookup failure.
- `test_convention_kernel_fail2.ml`: type unification failure.
- `test_inline_node_exhaustion.ml`: pragma inlining node-budget failure.
- `test_tuple_param.ml`: tuple-typed kernel parameter rejection (**added 2026-07-02, merged PRs #211/#213, commit `fdf53ac3`** — see `kb/sarek/ppx/lowering.md`).
- `test_fun_param.ml`: function-typed kernel parameter rejection (**added 2026-07-02, merged PRs #211/#213, commit `fdf53ac3`** — see `kb/sarek/ppx/lowering.md`).

## Features And APIs

- Expected errors are documented in `sarek/tests/negative/dune:8-18`.
- Each case is declared as a Dune library with `sarek_ppx` preprocessing.
- All cases are disabled unless profile is `negative`, for example `sarek/tests/negative/dune:26-27`; the same pattern repeats for each of the 10 declared cases, including `test_tuple_param`/`test_fun_param` (`sarek/tests/negative/dune:108-131`).

## Invariants

- Each negative test should fail compilation.
- The failure reason should match the expected diagnostic, not just any compile error.
- Negative tests should be easy to run from CI or `make test_negative`.

## Potential Invariant Violations Or Bugs

- **Corrected 2026-07-02 (re-verified against HEAD `618768b7`) — supersedes the "6 of 8" figure and the "not invoked by any Makefile target" gap immediately below; both are now stale.** `make test_negative` (`Makefile:82-108`) builds each case under `--profile=negative`, captures stderr to a `mktemp` file, and greps for the exact expected message for **all 10** currently-declared cases, including the two previously-unchecked ones: `test_convention_kernel_fail2` ("Cannot unify types"), `test_barrier_diverged` ("Barrier called in diverged control flow"), `test_superstep_diverged` (checked with a documented non-blocking KNOWN-ISSUE fallback, see below), `test_unbound_function` ("Unbound"), `test_reserved_keyword` ("reserved C/CUDA/OpenCL keyword"), `test_inline_node_exhaustion` ("Inlining produced .* nodes (limit: 10000)"), `test_convention_kernel_fail` (now checked against `"Unbound record field.*Geometry_lib\.z"` at `Makefile:98` — this exact string differs from the `sarek/tests/negative/dune:13` comment's `"Field z not found"`, a dune-comment/Makefile-assertion wording drift, not a functional gap), `test_warp_diverged` ("Warp collective 'warp_shuffle' called in diverged control flow", `Makefile:100`), and the two cases added by commit `fdf53ac3`: `test_tuple_param` ("Tuple-typed kernel parameters are not supported", `Makefile:102`) and `test_fun_param` ("Function-typed kernel parameters are not supported", `Makefile:104`). The dune-file comments at `sarek/tests/negative/dune:8-18` are documentation for humans; the real assertions live in the Makefile target, not the dune file.
- **Corrected 2026-07-02 (was stale) — the negative suite is no longer absent from CI.** `.github/workflows/ci.yml:105-114` now has a "Run negative tests (expected compile errors)" step that runs `make test_negative` inside the CI docker image on every run. `dune runtest` (a separate alias) still does not touch `sarek/tests/negative/`, but the suite as a whole is now exercised by dedicated CI, not only by a human running it locally.
- Confirmed, still live: the default top-level test alias does not include the negative profile in `sarek/tests/dune:1-15`; it is only reached via the dedicated `make test_negative` CI step, not `dune runtest`.
- Confirmed, still live: `test_superstep_diverged` has a documented, non-blocking exception in the Makefile (`Makefile:60-67, 90`) — if it compiles *without* the expected barrier-diverged error, the target treats this as a "KNOWN-ISSUE (non-blocking)" rather than a failure, tracking the pre-existing convergence-checker gap where the implicit end-of-superstep barrier is not checked against divergence introduced by the superstep body itself (see `Sarek_convergence.ml`'s `TESuperstep` case — this convergence gap must stay open per the current audit scope; see `kb/sarek/ppx/convergence.md`). A regression that makes this case newly compile without error would not fail CI.
- Confirmed risk: if a negative test fails earlier for an unrelated parse/module/build reason, the profile build could still be considered a failure in the expected direction unless the outer harness checks text — this is mitigated but not eliminated by the `grep -q "<exact message>"` checks in `Makefile:86-104`, since a coincidental substring match remains possible.

## Performance Or Maintainability Risks

- Profile-gated compile failures are easy to omit from regular local runs.
- Error message changes can silently desynchronize comments and actual diagnostics.
- Adding new negative cases requires remembering the profile gate pattern.

## Related Tests

- Unit error string tests live in `sarek/ppx/test/test_sarek_error.ml`.
- Unit convergence tests are included by `sarek/tests/unit/dune:22`.
- E2E converged barrier/superstep positive cases are declared in `sarek/tests/e2e/dune:74-75`.
- The real assertion harness for this suite is `Makefile:83-105` (`test_negative` target), not the dune file — see Invariants above.

## Missing Tests

- Negative tests for indirect convergence false negatives.
- Negative tests for duplicate fields/constructors and memory-space mismatches.
- Negative tests for invalid `%sarek_extend` or intrinsic PPX declarations.
- (Resolved, no longer missing: `test_warp_diverged` and `test_convention_kernel_fail` now have exact stderr assertions in `Makefile:98-100`; a CI job running `make test_negative` now exists at `.github/workflows/ci.yml:105-114`; tuple/function-typed kernel-parameter rejection now has negative coverage via `test_tuple_param.ml`/`test_fun_param.ml`.)

## Concrete Improvement/Fix Candidates

- Reconcile the `test_convention_kernel_fail` expected-message wording between `sarek/tests/negative/dune:13` ("Field z not found") and the actual `Makefile:98` grep target ("Unbound record field.*Geometry_lib\.z") — cosmetic doc drift, not a functional gap.
- Convert each negative case to a cram test or Dune action that asserts expected stderr, replacing the Makefile-`grep` approach with something dune-native and less bypassable.
- Add a small helper script to reduce duplicated profile-gated stanza boilerplate.
- Consider adding `dune runtest` (not just the separate CI `make test_negative` step) as a second, redundant path to catch a negative-suite regression even if the dedicated CI step is ever accidentally removed.
