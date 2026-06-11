# ConvergenceSafety — Divergence Findings

Per `policy/DIVERGENCE_POLICY.md`. Each finding records a discrepancy
between the abstract model and the real `Sarek_convergence.ml` behaviour,
or a reachability gap in the abstract model.

## Active findings

### F-01 — TESuperstep hard-resets ctx.mode to Converged, discarding inherited Diverged context

| Field | Value |
|---|---|
| ID | F-01 |
| Status | OPEN |
| Sub-status | Confirmed reachable |
| Classification | `a'` — spec gap (abstract model missing ESuperstep constructor) |
| Source | `sarek/ppx/Sarek_convergence.ml` lines 231, 239 |
| Regression | (none yet — blocked on SPOC test suite integration) |

**Description**: In `check_expr`, the non-divergent `TESuperstep` branch calls
`check_expr {mode = Converged} step_body` (line 231) and
`check_expr {mode = Converged} cont` (line 239), both unconditionally constructing a
fresh `Converged` context and ignoring the caller's `ctx.mode`.

The continuation reset (line 239) is independently correct: the implicit end-of-superstep
barrier re-synchronises all threads, so post-superstep code is always entered converged.
The body reset (line 231) is the real bug: if `ctx.mode` is already `Diverged` when a
non-divergent `TESuperstep` is encountered, the checker silently resets rather than
flagging that the implicit end-of-superstep barrier is reached under diverged flow.

The divergent-flagged path at lines 222–223 (`check_expr ctx step_body`) correctly
threads `ctx`, making the missed-divergence specific to the non-divergent branch.

The typer imposes no restriction on where `TESuperstep` may appear: `TEIf` branches are
typed with plain `infer env`, and `TESuperstep` is typed as a uniform expression. A
program such as:

```ocaml
if thread_idx_x > 16 then
  (let%superstep s = body in cont)
```

compiles and type-checks without error but silently drops the diverged-mode context at
the superstep site. The `contains_diverging_control_flow` secondary check (lines 249–317)
only scans for diverging CF *inside* the superstep body, not for divergence *inherited*
from the outer context.

**Abstract model impact**: `TESuperstep` is listed as elided in `ASSUMPTIONS.md §Elided
constructors` with the note "implicit barrier + cdcf interaction; different semantics
layer." After the scan, this classification must be revised from elided-safe to
**elided-risky**. The abstract `check` function (ConvergenceSpec.v lines 84–103) has no
`ESuperstep` case, so the spec provides no theorem capturing the missed-divergence
property. The abstract model must be extended:

1. Add `ESuperstep` of `bool * expr * expr` to the `expr` inductive (the `bool` mirrors
   `TESuperstep`'s `divergent` flag).
2. Add a `check` case: for `ESuperstep false body cont`, emit `BarrierError` when
   `m = Diverged` (the implicit barrier is entered under diverged flow), then check
   `body` with mode `m`, then check `cont` with mode `Converged`; for
   `ESuperstep true body cont`, check `body` with `m` and `cont` with `Converged` (no
   extra error — divergent superstep is expected).
3. Extend `has_diverging_cf` with an `ESuperstep` case mirroring
   `contains_diverging_control_flow` lines 307–309 (recurse into body and cont).
4. Update `ExprListInd` with `IH_superstep`.
5. Re-prove `diverged_clean_iff_barrier_free`, `mode_monotone`, and
   `cdcf_check_agreement` for the new case.
6. Add a new theorem `superstep_outer_diverged_error` stating:
   `check Diverged (ESuperstep false body cont) <> []`.

**New UCs discovered (F-01)**:
- UC-NEW-C: Divergent-flagged superstep nested inside a non-divergent superstep body:
  `contains_diverging_control_flow` recurses into nested supersteps (lines 307–309), but
  the interaction between the outer superstep's `contains_diverging_control_flow` check
  and the inner superstep's divergent flag is non-obvious and untested.
- UC-NEW-D: A non-divergent superstep nested inside a for-loop with thread-varying bounds:
  the loop body is checked in `Diverged` mode (lines 172–176), so the `TESuperstep`
  inside inherits `Diverged` outer mode — same mode-reset bug, with the additional
  complication that the implicit barrier fires on every loop iteration.
- UC-NEW-E: The continuation reset to `Converged` (line 239) becomes unsound if the
  superstep body is unreachable for some threads (e.g. the superstep is inside a diverged
  branch and only some threads execute it): those threads never hit the implicit barrier,
  so the continuation is not actually entered converged for all threads.

---

### F-02 — `is_thread_varying` is binding-blind (let-aliased thread-varying values missed)

| Field | Value |
|---|---|
| ID | F-02 |
| Status | OPEN |
| Sub-status | Confirmed reachable |
| Classification | `b` — implementation bug in `Sarek_convergence.ml`; spec accurately mirrors the (incomplete) implementation |
| Source | `sarek/ppx/Sarek_convergence.ml` line 86; `sarek/ppx/Sarek_core_primitives.ml` lines 732–733; `Sarek_convergence.ml` lines 199–200 |
| Regression | (none yet — blocked on SPOC test suite integration) |

**Description**: `Sarek_core_primitives.is_thread_varying name` (lines 732–733) performs
a hashtable lookup restricted to statically-known intrinsic identifiers. Any name absent
from the table returns `false`. User-defined let-bound variables are always absent.

`check_expr` dispatches `TEVar (name, _)` directly to this lookup (line 86), with no
environment parameter to carry binding-site variance information. The `ctx` record
(lines 41–44) holds only `mode : exec_mode`; the comment on line 43 reads
"Future: varying_vars : StringSet.t for dataflow analysis," confirming the limitation
is known and deferred.

`check_expr` for `TELet` and `TELetMut` (lines 199–200) passes the same `ctx` to both
the value and the body with no binding-to-variance mapping:
`check_expr ctx value @ check_expr ctx body`.

Concrete confirmed false-negative:
```ocaml
let x = thread_idx_x in
if x > 0l then block_barrier () else ()
```
When `check_expr` processes the `TEIf`, it calls `is_thread_varying` on `TEVar("x",_)`
→ `false` (x not in primitive table) → `inner_ctx` stays `Converged` → barrier inside
the true-branch is never flagged. At runtime all threads evaluate `x` to their own
`thread_idx_x` value → divergence → barrier safety violation.

The file header (lines 21–30) explicitly states "This minimal implementation catches
~80% of real bugs" and lists "Dataflow analysis: track thread-varying through variable
assignments" as a future extension.

**Abstract model impact**: `ConvergenceSpec.v` faithfully mirrors the current (incomplete)
checker behaviour: `ELet v b => is_varying v || is_varying b` (line 47) does not
substitute — if `v` is `EVary` but the body only mentions the let-bound name (not a
syntactic `EVary`), `is_varying` reports false for the name. The real checker's
`TEVar` lookup has the same gap by construction. The spec does NOT need to be changed
to accurately describe current checker behaviour.

However, if Phase 1a aims to reflect intended soundness rather than current behaviour,
`ConvergenceSpec.v` would need: (1) an environment `Gamma : string -> bool` threaded
through `is_varying` and `check`; (2) the `ELet` case in `is_varying` changed to
propagate `Gamma[x := is_varying_env Gamma v]` into the body; (3) all six theorems
re-proved under the extended model.

A T2 spec task is the appropriate vehicle: the current six theorems hold for the current
abstract model and should not be disturbed.

**New UCs discovered (F-02)**:
- UC-NEW-A: `contains_diverging_control_flow` shares the F-02 let-alias blind spot:
  at lines 252 and 258–261 it calls `is_thread_varying`, which has the same name-table
  limitation. A let-bound alias to `thread_idx_x` used as a loop or if condition inside
  a non-divergent `TESuperstep` body is not recognised as diverging CF, so the
  implicit-barrier error (lines 234–235) is also silently missed on this secondary path.
  F-02 false-negatives propagate into the superstep implicit-barrier detection.
- UC-NEW-B: `TELetRec` function body checked with caller `ctx` (lines 242–243): if a
  `TELetRec` is encountered while `ctx.mode = Diverged`, the function body is checked
  in `Diverged` mode at definition time. If the body contains a barrier it will be
  flagged even if the function is only ever called from a converged context — a
  false-positive risk. The discrepancy between definition-site and call-site `ctx` is
  not documented and not covered by any spec theorem.
- UC-NEW-F: `TEApp` argument aliasing: if a user-defined function `f` is called with a
  thread-varying argument and its return value is used as a branch condition,
  `check_expr TEApp` recurses into args but the return value variability defaults to
  false because the function body is not re-analysed at the call site. Same root cause
  as F-02 but at the inter-procedural boundary.

---

## Resolved findings

_(none yet)_

---

## Finding template

```markdown
### F-NN — short title

| Field | Value |
|---|---|
| ID | F-NN |
| Status | OPEN / RESOLVED |
| Sub-status | Reachability conjecture / Confirmed reachable / Classified |
| Classification | (pending / a / a' / b / c) |
| Source | file:line |
| Regression | test/regressions/test_convergence_safety_FNN_short.ml |

**Description**: …

**Abstract model impact**: …

**Resolution** (if RESOLVED): …
```
