# ConvergenceSafety — Uncovered Edge Cases

Catalogue of scenarios not exercised by the current test suite. Every item
here is a gap, not a bug. Use this file to drive coverage improvements.

## Tier 1 — High-value gaps (should cover before lock claim)

| ID | Scenario | Assessment | Notes |
|---|---|---|---|
| UC-01 | `TESuperstep` with diverged outer mode | CONFIRMED GAP (F-01) | `Sarek_convergence.ml` line 231 hard-resets to `Converged`; diverged outer context silently dropped. Implicit end-of-superstep barrier entered under diverged flow without error. |
| UC-02 | Nested `EIf` where inner cond is varying but outer is not | NOT A GAP | Lines 151–160: `check_expr` sets `inner_ctx` from `ctx`, not a hard-reset. If outer is already `Diverged` and inner cond is varying, `diverge ctx` returns `Diverged`. Handled correctly. |
| UC-03 | `EFor` with varying `lo` but uniform `hi` | NOT A GAP | Lines 168–176: `inner_ctx = if is_thread_varying lo || is_thread_varying hi then diverge ctx else ctx`. The `||` covers asymmetric bound variation; `diverge` is triggered whenever either bound is varying. |
| UC-04 | `ESeq` containing only barriers | NOT A GAP | Line 201: `TESeq es -> List.concat_map (check_expr ctx) es`. Each barrier in `Diverged` mode produces `BarrierError`; sequential barrier-only programs are fully handled. |
| UC-05 | `EApp` with 0 args (empty application) | NOT A GAP | Line 197: with 0 args `List.concat_map` returns `[]`; only `f` is checked. `List.exists is_thread_varying []` returns `false`. No crash, no missed barrier. |
| UC-NEW-C | Divergent superstep nested inside non-divergent superstep body | CONFIRMED GAP | `contains_diverging_control_flow` recurses into nested supersteps (lines 307–309), but the interaction between outer superstep's cdcf check and inner superstep's divergent flag is untested. Emit path non-obvious. |
| UC-NEW-D | Non-divergent superstep inside a for-loop with varying bounds | CONFIRMED GAP (F-01 variant) | Loop body checked in `Diverged` mode (lines 172–176); `TESuperstep` inside inherits `Diverged` outer mode — same mode-reset bug, with implicit barrier firing on every loop iteration. |
| UC-NEW-E | Continuation correctness when superstep is unreachable for some threads | CONFIRMED GAP (semantic) | Line 239 continuation reset to `Converged` is unsound when the superstep is inside a diverged branch: threads that never execute the superstep never hit the implicit barrier, so the continuation is not actually entered converged for all threads. |

## Tier 2 — Lower-priority gaps

| ID | Scenario | Assessment | Notes |
|---|---|---|---|
| UC-06 | `WarpConvergence` error class | OUT OF SCOPE (T2) | `Warp_collective_in_diverged_flow` is a second real error type (lines 144–147) with finer warp-level granularity. Entirely unmodeled in the abstract spec. No abstract `error` constructor corresponds to it. |
| UC-07 | `TELetShared` with barrier in body | NOT A GAP | `TELetShared` lowers to `SLet EArrayCreate(Shared)`; the body is still recursed into and checked. Elided-safe for barrier analysis: the declaration itself is not a synchronisation point. |
| UC-08 | `TELetRec` (recursive functions) | CONFIRMED GAP (over-approx) | `check_expr TELetRec` recurses into `fn_body` with the caller's `ctx` (lines 242–243). A recursive function defined inside a diverged branch has its body checked in `Diverged` mode at definition time, even if only ever called from converged contexts — potential false positives. Not modeled by the spec. |
| UC-09 | Out-of-bounds vector write with thread-varying index in diverged mode | OUT OF SCOPE | `TEVecSet` with thread-varying index (lines 190–192) is checked for subexpression barriers only. Out-of-bounds vector access is a memory safety concern, not in scope for barrier-convergence analysis. |
| UC-10 | Let-bound thread-varying variable as `if` condition | CONFIRMED GAP (F-02) | `let x = thread_idx_x in if x > 0 then block_barrier ()` — `is_thread_varying "x"` returns false (x not in primitive table); barrier not flagged. False-negative confirmed by code inspection of lines 86, 199–200. |
| UC-11 | `While` loop with varying condition | NOT A GAP | Lines 162–166: `inner_ctx = if is_thread_varying cond then diverge ctx else ctx`. A while with a thread-varying condition correctly sets `inner_ctx` to `Diverged`; a barrier in the body is flagged. |
| UC-12 | Barrier on only one branch of a diverged `if` | NOT A GAP | Lines 151–160: both `then_e` and `else_opt` are checked with `inner_ctx`. If cond is thread-varying, `inner_ctx = Diverged` for both branches. A barrier on only one branch is flagged because `check_expr Diverged` catches it at lines 142–143. |
| UC-NEW-A | `contains_diverging_control_flow` let-alias blind spot | CONFIRMED GAP (F-02 propagation) | `contains_diverging_control_flow` calls `is_thread_varying` at lines 252, 258–261, which has the same name-table limitation as the main checker. A let-bound alias to `thread_idx_x` used as a loop/if condition inside a non-divergent `TESuperstep` body is not recognised as diverging CF, so the implicit-barrier error (lines 234–235) is also silently missed. |
| UC-NEW-B | `TELetRec` definition-site vs call-site context mismatch | CONFIRMED GAP (over-approx) | Lines 242–243: `TELetRec` body checked with the caller's `ctx.mode`. Function body with barriers checked in `Diverged` at definition site even if never called diverged → false positive risk. No spec theorem covers this. |
| UC-NEW-F | `TEApp` return value variability unknown | CONFIRMED GAP (F-02 inter-procedural) | `check_expr TEApp` recurses into args (line 197) but does not analyse the function body. Return value variability defaults to false. A function called with varying arguments whose return value is used as a branch condition is not flagged. Same root cause as F-02 at the inter-procedural boundary. |

## Out of scope (T2/T3)

| Scenario | Tier | Blocker |
|---|---|---|
| Semantic soundness of `is_varying` | T2 | Needs `texpr` eval relation |
| Deadlock freedom | T3 | Needs lockstep execution model |
| Warp divergence sub-properties (`WarpConvergence`) | T2 | Second error class not modeled |
| `TESuperstep` implicit barrier safety (full semantic model) | T2 | Requires BSP superstep boundary semantics beyond T1 static analysis |
| `TENative` barrier opacity | T3 | Inline GPU strings with `__syncthreads` are fully opaque to the checker; no static analysis possible |
| `TEReturn` early-return barrier skip | T2 | Early return inside divergent branch may skip barrier for some threads; `Sarek_convergence.ml` handling unaudited |
| Dataflow / alias analysis (`is_varying` completeness) | T2 | Requires environment-threaded `is_varying`; full F-02 fix |
