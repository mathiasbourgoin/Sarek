# Sarek Convergence Analysis

## Component Inventory

Convergence analysis is in `sarek/ppx/Sarek_convergence.ml`, with intrinsic convergence metadata in `sarek/ppx/Sarek_core_primitives.ml` and environment lookup support in `sarek/ppx/Sarek_env.ml`.

## Per-File Purpose

- `Sarek_convergence.ml`: tracks uniform/diverged control-flow contexts, rejects unsafe barriers/warp collectives, computes dimension usage, and selects native execution strategy.
- `Sarek_core_primitives.ml`: declares convergence kind for built-in intrinsics.
- `Sarek_env.ml`: preserves core primitive metadata through module opens and shadowing in `sarek/ppx/Sarek_env.ml:183-259`.

## Features And APIs

- Execution modes and analysis context are defined in `sarek/ppx/Sarek_convergence.ml:35-47`.
- Intrinsic variance/barrier classification helpers are in `sarek/ppx/Sarek_convergence.ml:52-65`.
- Thread-varying expression detection is in `sarek/ppx/Sarek_convergence.ml:79-121`.
- Barrier and warp collective checks run through `check_expr` in `sarek/ppx/Sarek_convergence.ml:135-244`.
- Diverging-control-flow detection is in `sarek/ppx/Sarek_convergence.ml:245-318`.
- Dimensionality and execution strategy selection are in `sarek/ppx/Sarek_convergence.ml:409-616`.

## Invariants

- Thread-varying branch conditions put child control flow in a diverged context.
- Barrier and warp collective intrinsics are illegal in diverged context.
- Shared memory, barriers, block/thread intrinsics, and multidimensional indices force execution strategies that can model them.
- Supersteps carry an implicit barrier and must obey the same convergence safety rules.

## Potential Invariant Violations Or Bugs

- **FIXED (2026-07-02 audit):** indirect thread-varying divergence is now caught. `ctx.varying_vars` (a `StringSet.t`, `sarek/ppx/Sarek_convergence.ml:43-45`) is threaded through `check_expr` and `is_thread_varying_env` records `let`/`let mut` bindings whose value is thread-varying (`sarek/ppx/Sarek_convergence.ml:89, 108-111`); condition checks consult this env (e.g. `:170, 187, 193, 224, 237`). The previously-cited missed case — `let tid = thread_idx_x in if tid > 0 then block_barrier ()` — is now detected. Landed in commits `5e0cd683` ("fix(convergence): correct F-01 + F-02 barrier-safety checker bugs") and `c5c39c71` ("fix(convergence): F-04 — flag barriers after thread-varying early returns in sequences").
  - Stale artifact: the module's own header comment at `sarek/ppx/Sarek_convergence.ml:27` still lists "Dataflow analysis: track thread-varying through variable assignments" under "Future extensions" — the comment was not updated when the feature landed. Code-level bug, flagged here for a follow-up doc fix in-repo; not resolved by this KB edit.
- Confirmed maintainability issue: `TESuperstep` uses `true || ...` in `sarek/ppx/Sarek_convergence.ml:342-343`, making the later expression dead even though the resulting truth value is intentionally true.
- Probable: dimensionality analysis treats `global_size_x/y/z` as simple-compatible in `sarek/ppx/Sarek_convergence.ml:452-460`, but native simple intrinsic generation rejects many of them in `sarek/ppx/Sarek_native_intrinsics.ml:191-218`.

## Performance Or Maintainability Risks

- The analysis is syntactic and cheap, but false negatives are safety-relevant.
- Convergence kind depends on primitive names and module-open rewriting; name aliases can be fragile.
- Execution strategy inference must remain synchronized with native generation capabilities.

## Related Tests

- `sarek/tests/unit/dune:22` includes `test_convergence`.
- Negative diverged barrier/warp tests are documented in `sarek/tests/negative/dune:9-11`.
- E2E converged barrier and superstep tests are declared in `sarek/tests/e2e/dune:74-75`.

## Missing Tests

- Aliased or module-qualified barrier/warp intrinsic calls.
- `global_size_*` native simple execution.
- Superstep divergence with nested expressions past the current syntactic cases.

## Concrete Improvement/Fix Candidates

- Update the stale "Future extensions: Dataflow analysis" line in the file header comment (`Sarek_convergence.ml:27`) now that `varying_vars` dataflow is implemented.
- Remove the dead `true || ...` expression and document the superstep rule directly.
- Share an intrinsic capability table between convergence and native generation.
