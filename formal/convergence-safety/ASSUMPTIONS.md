# ConvergenceSafety — Assumptions and Trust Boundary

## Abstract model vs real implementation

The Rocq spec is for an abstract `expr` type with 11 constructors. The real implementation
operates on `texpr` — a large, typed AST with ~35 constructors. The correspondence is:

| Abstract | Real | Notes |
|---|---|---|
| `EVary` | `TEVar` (varying), `TEIntrinsicConst` (varying), `TEVecGet`/`TEArrGet` with varying index | `is_varying` recurses through all these |
| `EBarrier` | `TEIntrinsicFun _ (Some ConvergencePoint) _` | `check_expr` reports `Barrier_in_diverged_flow` when mode=Diverged |
| `ELit` | `TEUnit`, `TEBool`, `TEInt`, `TEFloat`, all literals, `TEIntrinsicConst` (uniform) | All return false for `is_thread_varying` |
| `EIf` | `TEIf`, `TEMatch` | Both: diverge if scrutinee/cond is thread-varying |
| `EFor` | `TEFor` | lo/hi bounds checked for thread-variation |
| `ESeq` | `TESeq` | Sequential composition |
| `ELet` | `TELet`, `TELetMut` | Conservative: no dataflow tracking |
| `EApp` | `TEApp`, `TETuple`, `TERecord` | Recursive check of all args |
| `EBinop` | `TEBinop`, `TEVecSet`, `TEArrSet`, `TEFieldSet`, `TEAssign` | All two-subexpr cases |
| `EUnop` | `TEUnop`, `TEFieldGet`, `TEReturn`, `TECreateArray` | Single-subexpr cases |

### Elided constructors (not modeled, OUT OF SCOPE)

| Constructor | Reason |
|---|---|
| `TESuperstep` | **MODELED as of Phase 1a** — `ESuperstep : bool -> expr -> expr -> expr` added to `ConvergenceSpec.v`; `superstep_outer_diverged_error` proves F-01. See §Updates below. |
| `TELetShared` | Shared memory allocation; barrier semantics not modeled |
| `TELetRec` | Recursive function definitions; requires interprocedural analysis |
| `TEPragma` | Compiler pragma; no barrier semantics |
| `TEOpen` | Module open; passthrough in implementation |
| `TENative` | Native C/CUDA expression; black box |
| `TEGlobalRef` | Global mutable reference; no barrier semantics |
| `WarpConvergence` errors | `Warp_collective_in_diverged_flow`; second error class not modeled |

## Proven properties

| Property | Status | Note |
|---|---|---|
| `merge_dim_comm/assoc/idem/empty_l/empty_r` | PROVEN | `dim_usage` is a join-semilattice under `||` per field |
| `check_seq_hom` | PROVEN | `check m (ESeq (es1 ++ es2)) = check m (ESeq es1) ++ check m (ESeq es2)` |
| `diverged_clean_iff_barrier_free` | PROVEN | `check Diverged e = [] ↔ barrier_free e = true`; Diverged mode is absorbing |
| `mode_monotone` | PROVEN | `incl (check Converged e) (check Diverged e)` — strengthening never removes errors |
| `not_varying_converged_clean` | PROVEN | Helper: `is_varying e = false → check Converged e = []` |
| `cdcf_check_agreement` | PROVEN | `has_diverging_cf e = false → check Converged e = []` |
| `varying_if_flags_barriers` | PROVEN | Varying cond + non-barrier-free body → always an error |
| `superstep_outer_diverged_error` | PROVEN (Phase 1a) | `check Diverged (ESuperstep false body cont) ≠ []` — F-01 formal statement |

## Assumed / unmodeled

| Component | Status | Rationale |
|---|---|---|
| Rocq kernel soundness | ASSUMED | Standard assumption throughout |
| OCaml extraction + compiler | ASSUMED | Standard TCB; not extracted here (abstract model tested by QCheck) |
| Abstract model faithfully represents the real implementation | ASSUMED | Verified by code inspection of Sarek_convergence.ml; the 11 constructors cover all barrier-relevant paths. Elided constructors are documented above. |
| `is_thread_varying` correctness for `TEVar`/`TEIntrinsicConst` | ASSUMED | Depends on `Sarek_core_primitives.is_thread_varying` being complete — outside this spec's scope |
| `WarpConvergence` error class | OUT OF SCOPE | Second error type not modeled; would require extending the `error` type and `check` |

## Out of scope (T2/T3)

| Property | Tier | What's needed |
|---|---|---|
| `is_varying_semantic_soundness` | T2 | Requires an execution semantics for `texpr` (eval relation); `EVary` must be axiomatized as "value differs across threads" |
| `deadlock_freedom` | T3 | Requires lockstep execution model, workgroup synchronization semantics, and a whole-kernel correctness theorem |
| Warp divergence sub-properties | T2 | `WarpConvergence` error class, warp size parameterization |
| `TESuperstep` implicit barrier safety (outer-mode F-01) | **PROVEN Phase 1a** | `superstep_outer_diverged_error` — entering ESuperstep false under Diverged always errors |
| `TEReturn` early-return barrier skip | T2 | Early return inside a divergent branch that precedes a barrier in the surrounding sequence will cause some threads to skip the barrier. The abstract model has no `EReturn` node; it is mapped to `EUnop` in the abstract correspondence table. Whether `Sarek_convergence.ml` handles this path correctly is unaudited. |

## Updates from ground-truth audit (2026-06-11)

### Corrections to "Elided constructors" table

#### `TESuperstep` — revised entry

The previous entry read:

| `TESuperstep` | Implicit barrier + cdcf interaction; different semantics layer |

Revised to:

| `TESuperstep` | **ELIDED-RISKY** — implicit end-of-superstep barrier is missed when `TESuperstep` is entered under `Diverged` outer mode. `Sarek_convergence.ml` line 231 hard-resets body context to `Converged`, silently dropping inherited `ctx.mode`. The continuation reset (line 239) is correct (re-sync), but the body-entry miss is a real false-negative (F-01). The abstract model provides no `ESuperstep` constructor or theorem; this gap is open until Phase 1a adds `ESuperstep` to `ConvergenceSpec.v`. |

#### `TENative` — revised entry

The previous entry read:

| `TENative` | Native C/CUDA expression; black box |

Revised to:

| `TENative` | **ELIDED-RISKY** — inline GPU strings can contain `__syncthreads` or warp-level barrier calls that are completely invisible to `check_expr`. `Sarek_lower_ir.ml` emits `SNative` opaquely; the convergence checker cannot see inside the string. Any kernel using `[%native]` with barriers inside divergent control flow is unsound with respect to the spec. No static analysis is possible without GPU string parsing; remains OUT OF SCOPE (T3). |

### New row — `contains_diverging_control_flow` let-alias blind spot

Added to "Assumed / unmodeled":

| `contains_diverging_control_flow` let-alias propagation | KNOWN LIMITATION | `contains_diverging_control_flow` (lines 249–317) calls `is_thread_varying`, which has the same name-table limitation as the main checker (F-02). A let-bound alias to a thread-varying intrinsic used as a loop or if condition inside a non-divergent `TESuperstep` body is not recognised as diverging CF. The implicit-barrier error (lines 234–235) is silently missed on this secondary path. F-02 false-negatives propagate into superstep implicit-barrier detection. |

### Revised row — `is_thread_varying` correctness

The previous entry read:

| `is_thread_varying` correctness for `TEVar`/`TEIntrinsicConst` | ASSUMED | Depends on `Sarek_core_primitives.is_thread_varying` being complete — outside this spec's scope |

Revised to:

| `is_thread_varying` correctness for `TEVar`/`TEIntrinsicConst` | **KNOWN FALSE-NEGATIVE (F-02)** | `Sarek_core_primitives.is_thread_varying name` (lines 732–733) returns `false` for any name absent from the static intrinsic table, including all user-defined let-bound variables. The abstract `ELet v b => is_varying v || is_varying b` (ConvergenceSpec.v line 47) mirrors this by structural union without substitution — so the spec accurately describes current (incomplete) checker behaviour. A full fix requires an environment-threaded `is_varying`; deferred to T2. |

### Revised row — `WarpConvergence` error class

The previous entry read:

| `WarpConvergence` error class | OUT OF SCOPE | Second error type not modeled; would require extending the `error` type and `check` |

Additional note: The real checker emits `Warp_collective_in_diverged_flow(name, loc)` for primitives tagged `WarpConvergence` (warp_shuffle, warp_vote_all/any, warp_ballot) at lines 144–147 of `Sarek_convergence.ml`. This is a finer-grained correctness requirement than `Barrier_in_diverged_flow`. Any theorem claiming `check` exhausts all convergence errors is incomplete without this second class. Remains OUT OF SCOPE for Phase 1a.
