---
name: roster-spec
type: spec
status: live
feature: Barrier convergence semantics (supersteps, variance dataflow, refusals) and its formal model
brief: /mnt/ssd-external-2to/spoc-pipeline-logs/global-review/PRIORITIES.md (P1; compiler.md C-1..C-7, C-49; docs-tests-formal.md D-38, D-54..D-58)
date: 2026-10-11
version: 1.0.0
---

# Spec — Barrier convergence semantics

Sarek claims that kernels are "convergence-checked": a block barrier
(`block_barrier`, the implicit barrier closing a `let%superstep`) or a warp
collective is never reached by only part of a block. The global review of
2026-10-11 found the claim false in eight reproduced ways (C-1..C-7) and found
three different superstep rules: production `Sarek_convergence`,
`ConvergenceSpec.v` and `ConvergenceSemantics.v` (D-38, D-57). This spec
settles the rules once. Production, the Rocq specification and the Rocq
operational model all implement this document, and a differential test checks
that the production checker and the extracted model agree on the fragment the
model covers.

It is implemented in three pull requests:

| PR | Sections | Findings |
|---|---|---|
| superstep semantics | §2, §5 (superstep and warp rows), §6 | D-38, D-57, C-7, D-56 |
| variance dataflow | §3, §4, §5 (all rows), production link | C-1..C-5 |
| refusals | §7 | C-49, C-6 |

## 1. Vocabulary

- **Block-uniform** value: every thread of a block that evaluates it at a given
  program point obtains the same value. The analysis computes a sound
  over-approximation, **may-vary**; everything else is treated as uniform.
- **Mode** of a program point: `Converged` if every thread of the block that
  entered the enclosing kernel or helper body reaches it, `Diverged` otherwise.
  Control under a condition that may vary is `Diverged`. `Diverged` is
  absorbing.
- **Synchronisation point**: `block_barrier`, the end of a `let%superstep`
  body, a warp collective, or a call to a helper that contains one of these
  (transitively).

The analysis is two-valued (may-vary or uniform) at block scope until S1
(§8). The primitive table's four-level variance maps onto it as:
`Uniform`, `BlockVarying` → uniform; `WarpVarying`, `ThreadVarying` → may-vary.

## 2. Supersteps (D-38, D-57, C-7)

`let%superstep[attrs] name = body in cont` executes `body`, then a block
barrier, then `cont`. It lowers to `SBlock body; SBarrier; cont` for every
attribute set.

| Rule | Statement |
|---|---|
| SS-1 | A superstep reached in `Diverged` mode is an error (`Barrier_in_diverged_flow`, located at the superstep). This holds **with or without `[@divergent]`**. |
| SS-2 | `body` is checked in the outer mode. Structured control flow inside `body` (`if`, `match`, loops) reconverges at its own end, before the implicit barrier, so a thread-varying `if` whose branches contain no synchronisation point is accepted. A synchronisation point inside a varying branch of `body` is rejected by the ordinary rules. Soundness of SS-2 assumes no thread leaves the body early: it depends on the `return_unit` refusal (§7, C-6). Until that refusal lands, `return_unit ()` under varying control inside a superstep body is accepted and unsound (it reaches the checker as an ordinary intrinsic call, not `TEReturn`, and the early-return analysis does not descend into superstep bodies), so some threads skip the implicit barrier. |
| SS-3 | `cont` is checked in the outer mode. Production no longer resets it to `Converged`; resetting hid nothing reachable only because SS-1 already rejects a diverged entry, and it made the three semantics disagree. |
| SS-4 | `[@divergent]` is accepted for source compatibility and has **no effect** on checking or lowering. Its former effect, skipping SS-1, was unsound: it accepted `if thread_idx_x > 16l then let%superstep[@divergent] s = ... in ()`, which lowers to `SBarrier` inside `SIf` (C-7). Any other attribute stays refused (backlog-192). |
| SS-5 | Variance is not reset by a superstep: a variable that may vary before it may vary after it. |

Consequence for D-38: the case `test_superstep_diverged.ml` expected
(a thread-varying `if` inside a superstep body, no barrier in the branch) is
**legal** under SS-2 and moves to a positive test. The negative test is
replaced by the case SS-1 forbids (a superstep under a thread-varying `if`),
plus the `[@divergent]` variant (C-7). The `KNOWN-ISSUE (non-blocking)`
branch of `make test_negative` is removed.

### 2.1 Model alignment

| Construct | Production `Sarek_convergence` | `ConvergenceSpec` (`check`, `check_env`) | `ConvergenceSemantics` (`check_warp_env`, `eval`) |
|---|---|---|---|
| entry in `Diverged` | error, any flag | `BarrierError`, any flag | barrier checker: as Spec; warp checker: no warp error (the entry is a barrier error) |
| body | outer mode | outer mode | outer mode |
| cont | outer mode | outer mode | outer mode |
| `barrier_free` / trace | — | `ESuperstep` is never barrier-free | `eval` emits `EvBarrier` for every flag |
| flag | inert | inert: theorem `check m (ESuperstep true b c) = check m (ESuperstep false b c)` (and for `check_env`, `check_warp`, `check_warp_env`) | inert in `eval` |

Error classes: `ConvergenceSpec.check_warp` (binding-blind, one checker for
both classes) reports a diverged superstep entry as `BarrierError`, while
`ConvergenceSemantics.check_warp_env` reports no error for it, because the
entry is a barrier error and belongs to `check_env`. Production reports it as a
barrier error. The production link therefore compares barrier errors with
`check_env` and warp errors with `check_warp_env`, never with `check_warp`.

`core_frag_ss` no longer excludes `[@divergent]` supersteps: the trust
boundary ASSUMPTIONS.md recorded for `dv = true` disappears, and
`check_env_sound_superstep` covers both flags.

## 3. Variance (C-1..C-5)

A whole-kernel fixpoint computes, for every variable id, an abstract value in
`{Varying} ∪ Deps(P)` where `P` is a set of helper parameter ids
(`Deps ∅` is uniform). `Deps` lets a helper's result be uniform at one call and
may-vary at another. Rules, by construct (`v(e)` is the abstract value of `e`,
`ctl` the join of the conditions controlling the current point):

| Construct | Value |
|---|---|
| literals, `TEGlobalRef`, kernel parameters, `block_idx_*`, `block_dim_*`, `grid_dim_*` | uniform |
| `thread_idx_*`, `global_*` ids, any intrinsic constant or function declared `WarpVarying` or `ThreadVarying` (atomics: C-4; ballot/vote) | `Varying` |
| intrinsic function call | join of its arguments ⊔ its declared variance |
| `TEVar x` | `v(x)` from the fixpoint |
| `let x = e in b`, `let x = mut e in b` | `v(x) ⊒ v(e)`; value is `v(b)` |
| `x := e` (C-1) | `v(x) ⊒ v(e) ⊔ ctl`: a variable assigned a varying value, or assigned under varying control, may vary; loops iterate to the fixpoint |
| `r.f <- e` on a local record variable `r` | as `r := e` |
| `if c then a else b`, `match c with ...` (C-2) | `v(c) ⊔ v(a) ⊔ v(b)`; pattern variables get `v(c)` |
| `while`, `for` (value) | `v(cond)` ⊔ `v(body)` (bounds for `for`); the loop variable gets `v(lo) ⊔ v(hi) ⊔ ctl` |
| sequence | join of its elements |
| binop, unop, tuple, record, constructor, field read | join of the parts |
| vector read `v.(i)` | `v(v) ⊔ v(i)` |
| array read `a.(i)` (C-5) | `Varying` if `a` may denote a private array (`create_array _ Local`/`Global`, or a parameter/alias that receives one); otherwise `v(a) ⊔ v(i)` |
| helper call `f args` (C-3) | `f`'s result summary with each parameter id replaced by the argument's value |
| `TENative` | `Varying` |

A helper parameter **may vary** if some call site passes an argument that may
vary (context-insensitive, fixpoint over the call graph, recursion included).
A condition makes its branches `Diverged` when its value is `Varying` or is
`Deps P` with some parameter of `P` that may vary.

Assumptions, documented in ASSUMPTIONS.md:

- **Race freedom.** A read of global or shared memory at a uniform index is
  uniform. A program that reads a location another thread of the block writes
  without an intervening barrier already has undefined results.
- Variance is **flow-insensitive** for a variable: a variable that may vary
  anywhere may vary everywhere. This rejects some safe programs and never
  accepts an unsafe one.
- Reads of private arrays are always `Varying` (no contents tracking).
- Inline native code (`TENative`) is not inspected for synchronisation points.
- User intrinsics (`let%sarek_intrinsic`) cannot declare variance or
  convergence yet (C-50): their result is the join of their arguments and
  they are never synchronisation points.

**Implementation status.** `Sarek_convergence` implements this table in a
*strict* variance analysis that is applied to every synchronisation point
(warp collectives, block barriers and supersteps): helper
parameters made varying by a call (the helper body is re-checked at the call
with those parameters varying, H-1), helper results that read the thread (the
body is unfolded per call, recursion cut), the value of `if`/`match`, and
mutables assigned a varying value or under varying control (C-1; flow-insensitive
per variable). Barriers and supersteps use it since the barrier-convergence
change (a barrier reached through a varying helper parameter, a thread-dependent
helper result, a mutable assigned in divergence or the value of a varying `if`
is refused like a warp collective); the repository's kernels and a downstream engine's
compile unchanged under it. Private-array reads (C-5) and loop-carried fixpoints through
several mutables are not tracked by either analysis.

## 4. Helpers (C-3)

The check runs after monomorphisation and defunctionalisation, so every call
names its callee (a module function or a `let rec`). For each helper:

| Rule | Statement |
|---|---|
| H-1 | The helper body is checked from `Converged`, with the parameter variance of §3. A synchronisation point under a condition that depends on a parameter some caller makes varying is rejected. |
| H-2 | A call reached in `Diverged` mode to a helper that contains a synchronisation point (transitively) is rejected at the call site, with the error of the first synchronisation point it contains. |
| H-3 | Assignments inside a `let rec` body to captured mutable variables use the join of the control at its call sites as `ctl`. |

## 5. Checking rules

| Construct | Rule |
|---|---|
| `block_barrier` | error in `Diverged` (`Barrier_in_diverged_flow`) |
| warp collective | error in `Diverged` (`Warp_collective_in_diverged_flow`) |
| `if`, `match`, `while`, `for` | branches/body `Diverged` if the condition, scrutinee or a bound may vary |
| superstep | §2 |
| sequence | after an element with a varying early return, the rest is `Diverged` (F-04). Barrier and warp checks share this rule (D-56). |
| helper call | §4 |

Production returns all errors; the PPX reports the first. Only emptiness is a
compatibility contract: error order and count may differ between production
and the model.

## 6. Formal model obligations

- `ConvergenceSpec.v`: SS-1 for both flags in `check`, `check_warp`,
  `check_env`; `barrier_free (ESuperstep _ _ _) = false`; flag-irrelevance
  theorems. Re-prove every theorem touched; no `Admitted`; the axiom
  allow-list (`formal/axiom-allowlist.txt`) unchanged.
- `ConvergenceSemantics.v`: `check_warp_env` threads the F-04 varying-return
  flag through `ESeq` like `check_env` (D-56); `core_frag_ss` admits both
  flags; `check_env_sound_superstep` and `check_warp_sound_core` re-proven.
- Extraction: `ConvergenceModel.ml` regenerated; it exports `check_env`,
  `check_warp_env` and `is_varying_in_env` so the production link can call them.
- Production link (variance PR): a QCheck property generates model `expr`
  terms, translates them to `texpr` (fresh ids per binder, `EVary` ↦
  `thread_idx_x`, `EBarrier` ↦ `block_barrier`, `EWarpPoint` ↦ a warp
  collective, `EApp` ↦ a uniform pure intrinsic), and asserts that
  production rejects with a barrier error iff `check_env Converged [] e ≠ []`
  and with a warp error iff `check_warp_env Converged [] e ≠ []`.
  `production-link.json` then claims `ConvergenceSpec`.
- Not modeled, stated in ASSUMPTIONS.md: assignment, helpers, arrays and the
  intrinsic table (they reach the model only through `EVary`); `EReturn`
  (no production construct produces it; `eval` keeps emitting the superstep
  barrier after a return inside the body, which a GPU would skip, and this
  stays outside the verified fragment). The same hazard exists in production
  through `return_unit` (an intrinsic, not `TEReturn`): the soundness of SS-2
  depends on the C-6 refusal of §7.

## 7. Refusals (C-49, C-6)

| Name | Refusal | Reason |
|---|---|---|
| `warp_vote_all`, `warp_vote_any`, `warp_ballot`, `warp_active_mask` | located PPX error: not yet lowered by any backend; warp-level collectives arrive with S1 | every backend fails with "Unknown intrinsic" late (C-49) |
| `return_unit` | located PPX error: Sarek has no early return; restructure with `if`/`else` | CUDA emits `return` and then skips later barriers, while Native and the interpreter continue (C-6) |

`warp_shuffle`, `warp_shuffle_down`, `warp_shuffle_up`, `warp_shuffle_xor`,
`warp_size` and `lane_id` left this table when every backend gained their
lowering (docs/plans/warp-shuffle-intake-2026-10-17.md). The shuffles keep the
`WarpConvergence` class: a shuffle reached in `Diverged` mode, directly or
through a helper (H-2), is `Warp_collective_in_diverged_flow`. They are
`Convergent` in the purity table, which is not `Pure`: no pass may hoist, sink,
duplicate or move them across control flow, and a kernel that uses any warp
operation is never fused, windowed with another launch, rewritten to a matrix
unit or index-sliced. A kernel that shuffles is also a barrier kernel for the
CPU runtimes (the simulation exchanges through shared memory between two
barriers) and its launch block must be a whole number of warps. The convergence
classes of the votes stay in the table and in the checker, unit-tested on
hand-built typed ASTs, so S1 starts from a checked base.

## 8. What S1 (warp-scope supersteps) adds on top

S1 is not implemented here. The rules above are its block-scope instance.

- **Lattice.** Value variance becomes `Uniform ⊑ WarpUniform ⊑ Varying`
  (block scope). `WarpVarying` primitives (ballot, vote) map to `WarpUniform`;
  `thread_idx_x / warp_size` is `WarpUniform`. Modes become
  `Converged ⊑ WarpConverged ⊑ Diverged`: a `WarpUniform` condition yields
  `WarpConverged`.
- **Scope argument.** `let%superstep[@warp]` (the parser already refuses
  unknown attributes, so the hook is free) gives `TESuperstep` a scope
  `Block | Warp`, ending with a block or a warp barrier. SS-1 holds per scope:
  a block superstep needs `Converged`, a warp superstep `WarpConverged` or
  better. Warp supersteps nest inside block supersteps, never the reverse.
- **Typed collectives.** Shuffle, reduce, broadcast, ballot (an abstract mask
  type), `lane_id`, `warp_id`, as IR nodes with a scope rather than registry
  intrinsics, so analyses see them; legal only inside a warp superstep and in
  `WarpConverged` or better. The refusals of §7 are lifted per backend as
  lowerings land.
- **Warp size** is a launch-time device parameter, not a constant.
- **Re-prove:** the lattice laws; mode monotonicity per scope; the
  diverged-clean ⇔ synchronisation-free equivalence per scope; core,
  superstep and warp soundness extended to warp steps and early returns;
  SS-1 per scope; the production link per scope.
- **Prerequisites:** this spec; an IR-level convergence re-check after
  rewrites that treats coopmat operations as warp collectives (D-58).

## 9. Scenarios

| Id | Given | When | Then |
|---|---|---|---|
| G-1 | `if thread_idx_x > 16l then let%superstep s = () in ()` | compiled | rejected, `Barrier called in diverged control flow` (SS-1) |
| G-2 | same with `[@divergent]` | compiled | rejected (SS-1, SS-4, C-7) |
| G-3 | `let%superstep s = if thread_idx_x > 16l then tile.(thread_idx_x) <- x in ...` | compiled | accepted (SS-2) |
| G-4 | `let%superstep s = if thread_idx_x > 16l then block_barrier () in ()` | compiled | rejected (SS-2) |
| G-5 | `x := thread_idx_x; if x > 16l then block_barrier ()` | compiled | rejected (C-1) |
| G-6 | `if thread_idx_x > 16l then x := 1l; if x = 1l then block_barrier ()` | compiled | rejected (C-1, control dependence) |
| G-7 | `let x = if thread_idx_x > 16l then 1l else 0l in if x = 1l then block_barrier ()` | compiled | rejected (C-2) |
| G-8 | helper returning `thread_idx_x` used as a barrier condition | compiled | rejected (C-3) |
| G-9 | `let sync () = block_barrier () in if thread_idx_x > 16l then sync ()` | compiled | rejected (C-3, H-2) |
| G-10 | `let f n = if n > 0l then block_barrier ()`; `f thread_idx_x` | compiled | rejected (C-3, H-1) |
| G-11 | same helper called only as `f n` with a kernel parameter `n` | compiled | accepted |
| G-12 | an `atomic_add_int32` result conditioning a barrier | compiled | rejected (C-4) |
| G-13 | a `create_array 4l Local` read conditioning a barrier | compiled | rejected (C-5) |
| G-14 | `if tid = 0l then s.(0l) <- v; block_barrier (); if s.(0l) > 0l then block_barrier ()` with `s` shared | compiled | accepted (race freedom) |
| G-15 | `warp_vote_all`, `warp_vote_any`, `warp_ballot` or `warp_active_mask`, any context | compiled | rejected, not yet lowered (C-49) |
| G-17 | `if thread_idx_x > 16l then warp_shuffle_xor v 1l` | compiled | rejected, `Warp collective 'warp_shuffle_xor' called in diverged control flow` (H-2 for a helper) |
| G-18 | `warp_shuffle_xor v 1l` after a varying `if` that has ended, or under a block-uniform `if` | compiled | accepted |
| G-16 | `return_unit ()` anywhere | compiled | rejected (C-6) |
