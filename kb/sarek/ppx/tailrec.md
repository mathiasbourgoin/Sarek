# Sarek Tail Recursion And Inlining

## Component Inventory

Tail recursion support spans `sarek/ppx/Sarek_tailrec.ml`, `sarek/ppx/Sarek_tailrec_analysis.ml`, `sarek/ppx/Sarek_tailrec_elim.ml`, `sarek/ppx/Sarek_tailrec_pragma.ml`, and `sarek/ppx/Sarek_tailrec_bounded.ml`.

## Per-File Purpose

- `Sarek_tailrec.ml`: orchestrates transformation of module items in a typed kernel.
- `Sarek_tailrec_analysis.ml`: detects self/recursive calls, counts recursive calls, and decides tail position.
- `Sarek_tailrec_elim.ml`: rewrites tail-recursive functions into loops with mutable temporaries and result/continue state.
- `Sarek_tailrec_pragma.ml`: parses `sarek.inline` pragmas and inlines approved non-tail recursion.
- `Sarek_tailrec_bounded.ml`: older/experimental bounded recursion inliner retained for tests.

## Features And APIs

- Tail-recursive functions are converted to loops in `sarek/ppx/Sarek_tailrec.ml:51-166`.
- Non-tail recursion requires an inline pragma and otherwise fails in `sarek/ppx/Sarek_tailrec.ml:51-166`.
- Recursive-call counting and tail-position checks are in `sarek/ppx/Sarek_tailrec_analysis.ml` (file is 237 lines total; line ranges below are re-verified against the current file, correcting stale ~150-line-file-era references).
- Bounded-depth analysis is currently disabled and returns `None` in `sarek/ppx/Sarek_tailrec_analysis.ml:216-220` (`detect_bounded_depth`, hardwired `None`).
- Inline pragma node limit is defined as 10,000 nodes (`max_inlined_nodes`) in `sarek/ppx/Sarek_tailrec_pragma.ml:20`.

## Invariants

- Tail-recursive argument updates must be simultaneous, not sequentially dependent.
- Non-tail recursion must be bounded by explicit developer intent.
- Generated loop variables and temporary names must not collide with source variables.
- Inlined recursion must not exceed the configured node budget.

## Potential Invariant Violations Or Bugs

- **Confirmed dead code (graph-verified 2026-07-02):** `max_inline_limit = 16` at `sarek/ppx/Sarek_tailrec.ml:36` has 0 callers. The live budget for pragma-based inlining is the separate `max_inlined_nodes = 10000` in `sarek/ppx/Sarek_tailrec_pragma.ml:20` (enforced at `:348-355`). `max_inline_limit` is dead weight left over from an earlier design and should be removed or wired up, not read as an active limit.
- Probable: `inline_with_pragma`/pragma inlining checks the node limit before substitution in `sarek/ppx/Sarek_tailrec_pragma.ml:335-377` (enforced against `max_inlined_nodes`, `:348-355`); the final substitution that crosses the limit may not be rejected until another iteration exists.
- Confirmed limitation: bounded recursion is disabled by returning `None` in `sarek/ppx/Sarek_tailrec_analysis.ml:216-220` (`detect_bounded_depth`), while `Sarek_tailrec_bounded.ml` remains in the tree and tests. **Graph-verified (2026-07-02):** `inline_bounded_recursion` (`sarek/ppx/Sarek_tailrec_bounded.ml:23`) has 0 resolved callers outside its own unit tests (`sarek/tests/unit/test_tailrec_bounded.ml`) — it is not reachable from the live tailrec pipeline (`Sarek_tailrec.ml`/`Sarek_tailrec_pragma.ml`).
- Confirmed: invalid pragma handling in the orchestrator can use `failwith` paths instead of structured Sarek errors in `sarek/ppx/Sarek_tailrec.ml:51-166`.

## Performance Or Maintainability Risks

- Inlining can grow typed AST size quickly; monomorphization before/after recursion transforms increases the blast radius of size bugs.
- Tailrec loop generation introduces mutable state and synthetic variables that native generation/lowering must both understand.
- Retained bounded-recursion code increases ambiguity about the supported recursion model.

## Related Tests

- `sarek/tests/unit/dune:9-13` includes `test_tailrec`, `test_tailrec_analysis`, `test_tailrec_elim`, `test_tailrec_bounded`, and `test_tailrec_pragma`.
- E2E recursion tests are declared in `sarek/tests/e2e/dune:91-92`.
- Inline node exhaustion is a negative test with expected error in `sarek/tests/negative/dune:15` (line ref corrected 2026-07-02 after `test_tuple_param`/`test_fun_param` were appended to the file; was previously `:16`).

## Missing Tests

- Final-substitution node budget crossing.
- Invalid pragma syntax and negative inline depths as structured PPX errors.
- Tailrec simultaneous assignment with argument aliasing and more than two parameters.
- Native and IR parity for tailrec-converted loops.

## Concrete Improvement/Fix Candidates

- Remove the dead `max_inline_limit` (`Sarek_tailrec.ml:36`, 0 callers) or wire it to the live `max_inlined_nodes` budget (`Sarek_tailrec_pragma.ml:20`) so there is one config value.
- Check node count after substitution as well as before substitution.
- Convert `failwith` errors to `Sarek_error` values.
- Either revive bounded recursion deliberately or move `Sarek_tailrec_bounded.ml` (0 resolved non-test callers) and its tests out of the active compiler surface.
