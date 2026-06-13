# ConvergenceSafety — Assumptions and Trust Boundary

## Abstract model vs real implementation

The Rocq spec is for an abstract `expr` type with 15 constructors. The real implementation
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
| `EUnop` | `TEUnop`, `TEFieldGet`, `TECreateArray` | Single-subexpr cases |
| `EReturn` | `TEReturn` | `check m (EReturn e) = check m e` — transparent wrapper per Sarek_convergence.ml:230 |

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
| `WarpConvergence` errors | **MODELED as of T2-WARP** — `EWarpPoint` constructor added to `expr`; `WarpError` added to `error`; `check_warp` function and `warp_diverged_error` theorem (Phase 2). |

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
| OCaml extraction + compiler | ASSUMED | Standard TCB. Both the static checkers (`check`, `check_warp`, ...) and the operational evaluator (`eval_concrete`, T3-S8) are extracted to `extraction/ConvergenceModel.ml` via `coqc` and tested by QCheck against inline OCaml references. The extraction mechanism (`Extraction` plugin) and the OCaml compiler are trusted; their fidelity to the Rocq source is corroborated — not proven — by the differential conformance suites (`test_convergence_extraction.ml`, `test_convergence_semantics.ml`). |
| Abstract model faithfully represents the real implementation | ASSUMED | Verified by code inspection of Sarek_convergence.ml; the 15 constructors cover all barrier-relevant paths. Elided constructors are documented above. |
| `is_thread_varying` correctness for `TEVar`/`TEIntrinsicConst` | ASSUMED | Depends on `Sarek_core_primitives.is_thread_varying` being complete — outside this spec's scope |
| `WarpConvergence` error class | IN SCOPE (T2-WARP) | `EWarpPoint` constructor, `WarpError` error, `check_warp` function, and `warp_diverged_error` theorem added. See §T2-WARP update below. |

## Out of scope (T2/T3)

| Property | Tier | What's needed |
|---|---|---|
| `is_varying_semantic_soundness` | T2 | Requires an execution semantics for `texpr` (eval relation); `EVary` must be axiomatized as "value differs across threads" |
| `deadlock_freedom` | T3 | Requires lockstep execution model, workgroup synchronization semantics, and a whole-kernel correctness theorem |
| Warp divergence sub-properties | **PROVEN (T3-S7)** | `WarpConvergence` error class: `EWarpPoint`/`WarpError`/`check_warp`/`warp_diverged_error` PROVEN (T2-WARP); semantic soundness `check_warp_sound_core` PROVEN (T3-S7). Warp size parameterization CLOSED: `warp_of : tid -> nat` is an abstract Section Variable, so soundness holds for any warp partition (no fixed warp size baked in). |
| `TESuperstep` implicit barrier safety (outer-mode F-01) | **PROVEN Phase 1a** | `superstep_outer_diverged_error` — entering ESuperstep false under Diverged always errors |
| `TEReturn` early-return barrier skip | **MODELED/conformant (T2-RETURN)** | `EReturn` constructor added; `check m (EReturn e) = check m e` (transparent wrapper, mirroring `Sarek_convergence.ml:230`). The `return_barrier_skip_safe` theorem proves compositionality. **Open audit item (residual):** the hazard of a conditional early return causing residual divergence at a later barrier — e.g. `ESeq [EIf EVary (EReturn ELit) ELit; EBarrier]` — is conformant with the current host checker (both model and `Sarek_convergence.ml` treat EReturn transparently and report the barrier), but the question of whether that checker behaviour is *correct* for all possible call-site continuations remains an open audit item. It is NOT recorded as resolved. |

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

### T2-WARP update (2026-06-12)

`WarpConvergence` error class is now **IN SCOPE**. `EWarpPoint` constructor added to `expr`; `WarpError` added to `error` inductive; `check_warp` function models the warp-collective divergence check; `warp_diverged_error` theorem (Tier T2) proves `check_warp Diverged EWarpPoint ≠ []`. The entry in "Assumed / unmodeled" table has been updated to IN SCOPE. The entry in "Elided constructors" has been updated to MODELED.

### T3-S7 update (2026-06-13) — warp semantic soundness, warp-size parameterization CLOSED

`WarpConvergence` soundness is now **PROVEN** at the semantic level (T3-S7), the warp dual of T3-S4. New in `theories/ConvergenceSemantics.v`:

- `erase_barrier` — trace projection keeping only `EvWarp` events (dual of `erase_warp`).
- `warp_free` / `warp_free_no_warps` — trace silence: a `warp_free` + `superstep_free` expression emits no `EvWarp` event (dual of `barrier_free_no_barriers`).
- `check_warp_env` — env-threaded warp checker mirroring `check_env`, flagging `EWarpPoint` (not `EBarrier`) under `Diverged` mode.
- `check_warp_env_diverged_clean_warp_free` — Diverged-clean bridge lemma.
- `eval_check_warp_uniform` — combined warp-trace + outcome uniformity (mechanical substitution of the T3-S4 induction; parametrization of the single induction was attempted and flagged infeasible within budget — see design note 1 in §12 of the source).
- `Section WarpModel` with `Variable warp_of : tid -> nat`, `warp_safe`, and the main theorem **`check_warp_sound_core`**: `core_frag e = true -> check_warp_env Converged env e = [] -> warp_safe env e`.

**Warp-size parameterization is CLOSED.** `warp_of` is an abstract `Section Variable`; the soundness theorem is universally quantified over it. No fixed warp size (32, 64, ...) is baked into the proof — `warp_safe` is the same-warp restriction `warp_of t1 = warp_of t2` for any partition. 0 admits, 0 axioms (`Closed under the global context`), coqchk clean.

### T3-S8 update (2026-06-13) — extraction + differential conformance for the operational semantics

The operational evaluator is now **extracted and differentially tested** (CMBT closure). New in `theories/ConvergenceSemantics.v`:

- **`eval_concrete`** — `eval` with the abstract per-thread varying value `vary_val` instantiated as the **identity** `fun th => th`. This is the concrete witness extracted to `extraction/ConvergenceModel.ml` (`eval_concrete : nat -> tid -> venv -> expr -> option (outcome * trace)`). `extraction/ConvergenceSafetyExtraction.v` now also `Require`s `ConvergenceSemantics` and emits `eval_concrete` plus the operational types (`outcome`, `trace`, `event`, `venv`).
- **`eval_concrete_fuel_monotone`** / **`eval_concrete_barrier_free_silent`** — the headline sanity + silence properties specialized to the extracted instantiation (direct applications of the section theorems `eval_fuel_monotone` and `barrier_free_no_barriers`).
- `test/test_convergence_semantics.ml` — 4 QCheck properties exercising the extracted `eval_concrete` against inline OCaml references: `sem:eval_fuel_monotone`, `sem:barrier_free_silent`, `sem:differential_barrier_safe` (CMBT instance of `check_env_sound_core`), `sem:f04_hazard_counterexample` (F-04 regression).

**TRUST BOUNDARY — choice of `vary_val := identity`.** The section theorems quantify over *every* `vary_val : tid -> value`, so the soundness results hold for any instantiation; the extracted `eval_concrete` therefore inherits them a fortiori. Identity is chosen for the executable witness because it is *maximally discriminating*: distinct thread ids yield distinct `EVary` values (`t1 <> t2 ⇒ vary_val t1 <> vary_val t2`), so any thread-dependent control-flow divergence becomes observable in the extracted traces. The differential tests run `eval_concrete` only — they do NOT (and cannot, by extraction-of-a-single-instance) corroborate the universal quantification over `vary_val`; that universality is established solely by the Rocq proofs. The extraction plugin and OCaml compiler remain in the TCB (see the "OCaml extraction + compiler" row above).

### T2-RETURN update (2026-06-12)

`TEReturn` early-return is now **MODELED/conformant**. `EReturn` constructor added to `expr`; `check m (EReturn e) = check m e` (transparent wrapper, mirroring `Sarek_convergence.ml:230` TEReturn handling); `return_barrier_skip_safe` theorem (Tier T2) proves compositionality. The correspondence table has been updated: `TEReturn` removed from the `EUnop` row and given its own `EReturn` row. The "11 constructors" counts (abstract model description and ASSUMED row) updated to 15.

**Residual open audit item:** The tick proves `EReturn` is transparent to the barrier checker (conformant with `Sarek_convergence.ml:230`). It does NOT resolve the hazard described in the original row 65: a conditional early return causing residual divergence at a later barrier (e.g. `ESeq [EIf EVary (EReturn ELit) ELit; EBarrier]`). Both the abstract model and the host checker treat this transparently — whether that behaviour is correct for all call-site continuations remains an open audit question and is NOT recorded as resolved.

### T3-S6 update (2026-06-13) — ESuperstep semantic grounding (semantic F-01)

`ESuperstep` now has an operational-soundness grounding in `ConvergenceSemantics.v`:

- **`core_frag_ss`** — the core fragment ENLARGED to admit `ESuperstep`. It is `core_frag` plus the one constructor `ESuperstep false b c => core_frag_ss b && core_frag_ss c` (recursively). `EReturn` remains excluded (early-exit bypasses later barriers — the T3-S5 / F-04 hazard).
- **`check_env_sound_superstep`** (0 admits, 0 axioms) — over `core_frag_ss`, a `check_env Converged`-clean expression is `barrier_safe`. Uniform-reachability supersteps emit the implicit boundary `[EvBarrier]` on **every** thread, so two env-agreeing threads keep equal barrier traces. Thread-varying entry into a `dv=false` superstep is excluded because `check_env` raises `[BarrierError]` for it in Diverged mode (the `Diverged, false => [BarrierError]` arm). This grounds the static `superstep_outer_diverged_error` at runtime.
- **`semantic_f01_corollary`** (0 admits, 0 axioms) — the witness `susp_hazard := EIf EVary (ESuperstep false EBarrier ELit) ELit` is simultaneously (a) flagged by `check_env Converged []` and (b) genuinely NOT `barrier_safe` (thread 0 emits `[EvBarrier; EvBarrier]`, thread 1 emits `[]`). The checker's `BarrierError` is therefore a SOUND rejection of a real barrier-divergence hazard.
- **T3-S3 side condition resolved.** The strengthened silence lemma `check_env_diverged_no_barriers_ss` discharges the `superstep_free` side condition over `core_frag_ss`: a Diverged-clean `core_frag_ss` expression is provably barrier-silent because any reachable `dv=false` superstep would itself flag `[BarrierError]`, so a clean check forces superstep-freedom (`core_frag_ss_barrier_free_superstep_free`).

#### TRUST BOUNDARY — `dv=true` supersteps

`core_frag_ss` admits **only** `dv=false` (uniform-reachability) supersteps; the clause is literally `negb dv && core_frag_ss b && core_frag_ss c`. A `dv=true` superstep (one whose entry is statically known to be under divergent control flow) is OUT of the verified fragment and is a documented trust boundary:

- The operational semantics emits the boundary `[EvBarrier]` for `dv=true` exactly as for `dv=false` (the runtime barrier always fires). But the static `check_env` does **not** raise `[BarrierError]` for a `dv=true` superstep entered under `Diverged` mode (the `Diverged, false` arm matches only `dv=false`). That is intentional: `dv=true` is the checker's signal that the programmer/front-end has asserted the boundary is reached uniformly despite the divergent context.
- We therefore TRUST that a `dv=true` annotation is correct — i.e. that the implicit boundary barrier really is reached by all threads. The Rocq development does not (and cannot, without a richer thread-scheduling model) verify that assertion; `check_env_diverged_no_barriers_ss` would be FALSE if `dv=true` supersteps were admitted into `core_frag_ss`, because such a superstep emits a barrier even inside a Diverged-mode branch.
- Consequence: `check_env_sound_superstep` is a guarantee about the `dv=false` fragment only. Kernels relying on `dv=true` supersteps inherit the soundness of the front-end's divergence-uniformity annotation, which is ASSUMED, not PROVEN.
