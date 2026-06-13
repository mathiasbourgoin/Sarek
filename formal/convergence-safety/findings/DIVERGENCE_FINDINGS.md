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

### F-04 — `EReturn` transparency is a kernel-granularity false negative (varying early return skipping a later barrier)

| Field | Value |
|---|---|
| ID | F-04 |
| Status | OPEN |
| Sub-status | Formal counterexample (constructive) |
| Classification | `a'` — spec/checker gap; the abstract model faithfully mirrors the real `TEReturn` transparency, so the false negative is inherited from the implementation by construction |
| Source | `sarek/ppx/Sarek_convergence.ml` TEReturn handling; `theories/ConvergenceSpec.v` line 860 (`EReturn e => check_env m env e`) |
| Regression | (none yet — blocked on SPOC test suite integration; reachability in real kernels pending) |

**Description**: The checker treats `EReturn` as a transparent wrapper:
`check_env m env (EReturn e) = check_env m env e` (ConvergenceSpec.v line 860),
mirroring the real `Sarek_convergence.ml` `TEReturn` handling and the
`return_barrier_skip_safe` theorem (`check m (EReturn e) = check m e`). The
checker therefore reasons about an early return purely through its inner
expression and never accounts for the control-flow effect of the return itself:
that the early-returning thread skips every subsequent statement, including any
later barrier.

This is sound at the granularity at which the checker reasons (an `EReturn`
inside an `EIf` branch under a varying condition does not introduce a barrier of
its own), but it is **unsound at kernel granularity**: when a thread-varying
early return is followed — in sequence — by a barrier, the threads that take the
early return never reach that barrier while the threads that fall through do.
The barrier traces diverge, yet the checker reports no error.

The minimal hazard term (T3-S5, `theories/ConvergenceSemantics.v`):

```coq
Definition hazard : expr :=
  ESeq [EIf EVary (EReturn ELit) ELit; EBarrier].
```

- `EIf EVary (EReturn ELit) ELit` branches on a thread-varying condition.
  When `EVary` evaluates to 0 the `e_else = ELit` branch is taken (fall through);
  when nonzero the `e_then = EReturn ELit` branch is taken (early return).
- The early-return branch yields `ORet`, which short-circuits the `ESeq`, so the
  subsequent `EBarrier` is never evaluated for those threads (trace `[]`).
- The fall-through branch yields `ONorm`, letting `ESeq` reach `EBarrier`
  (trace `[EvBarrier]`).
- Because the `EReturn` sits under a varying `EIf` condition (whose body is
  checked transparently) and `Converged`-mode `EBarrier` is clean, the checker
  reports `[]`.

**Formal results** (`theories/ConvergenceSemantics.v`, 0 admits, 0 axioms, coqchk passes):

- `Lemma hazard_checker_blind : check_env Converged [] hazard = []` — the checker
  is blind (by `reflexivity`).
- `Definition hazard_vary (n : tid) := match n with O => 1 | S _ => 0 end` —
  concrete thread-varying witness.
- `Lemma hazard_eval_thread0 : eval hazard_vary 6 0 [] hazard = Some (ORet 0, [])`
  — thread 0 takes the early return; barrier never reached; empty trace.
- `Lemma hazard_eval_thread1 : eval hazard_vary 6 1 [] hazard = Some (ONorm 0, [EvBarrier])`
  — thread 1 falls through to the barrier; one `EvBarrier` event.
- `Theorem hazard_not_barrier_safe : ~ barrier_safe hazard_vary [] hazard` —
  threads 0 and 1 trivially env-agree on the empty environment, yet their
  `erase_warp` barrier traces (`[]` vs `[EvBarrier]`) differ, so the hazard is
  not `barrier_safe`. Combined with `hazard_checker_blind`, this is a formal
  counterexample to the checker being sound on terms containing `EReturn`.

**Abstract model impact**: F-04 makes explicit the residual gap that
`check_env_sound_core` (T3-S4) leaves implicit in its precondition. That theorem
proves soundness only for the `core_frag` fragment, which by definition excludes
`EReturn` (and `ESuperstep`); see `core_frag` `EReturn _ => false`
(ConvergenceSemantics.v ~line 2302/2403). `hazard_not_barrier_safe` demonstrates
that this exclusion is necessary, not merely a proof convenience: the checker
genuinely accepts a non-barrier-safe term once `EReturn` appears between a
varying branch and a barrier.

Closing F-04 would require the checker to model early-return control flow at
sequence granularity — e.g. when an `EReturn` is reachable under a varying
condition, every barrier sequenced after it must be flagged. This is a strictly
stronger analysis than the current transparent `EReturn` handling and is left as
a follow-up (a `b`-class fix in `Sarek_convergence.ml` mirrored by a spec
extension). Whether the hazard pattern (a barrier sequenced after a varying
early return within the same superstep) is reachable in real Sarek kernels is
the open reachability question for this finding.

---

## Resolved findings

### F-03 — `WarpConvergence` error class not modeled (warp-collective calls in diverged flow)

| Field | Value |
|---|---|
| ID | F-03 |
| Status | RESOLVED |
| Sub-status | Formally proven |
| Classification | `a'` — spec gap (abstract model missing `EWarpPoint` constructor and `WarpError` error) |
| Source | `sarek/ppx/Sarek_convergence.ml` lines 144–147, 153–155 |
| Regression | `test/test_convergence_conformance.ml` — `test_warp_diverged_error` property; `test/test_convergence_extraction.ml` — `extr:check_warp_agrees` test |

**Description**: The real checker emits `Warp_collective_in_diverged_flow(name, loc)` for
primitives tagged `WarpConvergence` (`warp_shuffle`, `warp_vote_all/any`, `warp_ballot`)
at lines 144–147 and 153–155 of `Sarek_convergence.ml`. This is a distinct error class
from `Barrier_in_diverged_flow`. The original abstract model had only one error constructor
(`BarrierError`) and no constructor representing warp-collective call sites, so the spec
was incomplete: any theorem claiming `check` exhausts all convergence errors was missing
this second class.

**Abstract model impact**: Required extending the abstract model with:
1. `EWarpPoint` — a new leaf constructor of `expr` representing a warp-collective call site.
2. `WarpError` — a second constructor of `error` representing `Warp_collective_in_diverged_flow`.
3. `check_warp (m : exec_mode) (e : expr) : list error` — a checker that emits `WarpError`
   when `EWarpPoint` is encountered under `Diverged` mode, and is otherwise structurally
   identical to `check`.
4. `warp_diverged_error` theorem: `check_warp Diverged EWarpPoint ≠ []`.

**Resolution**: Implemented in T2-WARP (2026-06-12). `EWarpPoint` and `WarpError` added to
`theories/ConvergenceSpec.v`; `check_warp` function defined; `warp_diverged_error` proven
by `simpl + discriminate`. Extraction list updated; conformance and extraction tests added.
The `warp_diverged_error` theorem directly captures the safety invariant that warp-collective
intrinsics must not appear in diverged control flow.

---

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
