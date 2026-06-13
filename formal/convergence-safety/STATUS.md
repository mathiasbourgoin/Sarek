# ConvergenceSafety — Project Status

**Grade**: A (apparatus-native)
**Apparatus version**: 1.2.1
**Host profile**: SPOC/sarek
**Architecture**: 3-layer
**Built at**: 2026-06-11
**Last updated**: 2026-06-13 (T3-S7: warp-collective semantic soundness — `erase_barrier`/`warp_free`/`check_warp_env`, `check_warp_sound_core` parametric in `warp_of : tid -> nat`; warp-size parameterization CLOSED; T3-S4 induction parametrization attempted + flagged infeasible, duplicated as `eval_check_warp_uniform` — 0 admits, 0 axioms, coqchk passes)

## Project

ConvergenceSafety is a formal Rocq specification of the GPU barrier safety analysis for the Sarek frontend of SPOC. It proves correctness properties of `Sarek_convergence.check_expr`, which statically detects barriers placed inside diverged control flow on the abstract `expr` AST. The project covers 20 theorems at Tiers 0–2 (lattice laws through control-flow monotonicity, warp-collective safety, and early-return compositionality) — 16 Theorems + 2 named Lemmas (`env_lookup_extend_same`, `not_varying_converged_clean`) + 2 strengthened warp theorems (`warp_mode_monotone`, `warp_varying_if_flags`) — including the F-01 safety property for `ESuperstep` (Phase 1a), 3 env-threaded F-02 theorems (T2-F02), 3 warp theorems for the `WarpConvergence` error class (T2-WARP), and the `return_barrier_skip_safe` compositionality theorem for `EReturn` (T2-RETURN). Validated by 14 QCheck conformance properties against an abstract OCaml model and 7 extraction tests exercising the extracted `ConvergenceModel` module (including `check_warp` CMBT link).

## Trust root

Assumptions documented in `ASSUMPTIONS.md`:
- Rocq kernel soundness (standard)
- OCaml extraction + compiler (standard TCB)
- Abstract `expr` model faithfully represents the barrier-relevant paths in `Sarek_convergence.ml` (verified by code inspection; elided constructors documented; ESuperstep added Phase 1a)
- `is_thread_varying` correctness for `TEVar`/`TEIntrinsicConst` (depends on `Sarek_core_primitives.is_thread_varying` — outside scope)
- `WarpConvergence` error class: **PROVEN (T3-S7)** — `EWarpPoint`/`WarpError`/`check_warp`/`warp_diverged_error` modeled (T2-WARP); semantic soundness `check_warp_sound_core` proven, warp-size parameterization closed via abstract `warp_of`

## Proof status

| Theorem | Tier | Admits | Notes |
|---|---|---|---|
| `merge_dim_comm` | T0 | 0 | `f_equal; apply Bool.orb_comm` |
| `merge_dim_assoc` | T0 | 0 | `f_equal; apply Bool.orb_assoc` |
| `merge_dim_idempotent` | T0 | 0 | `f_equal; apply Bool.orb_diag` |
| `merge_dim_empty_r` | T0 | 0 | `f_equal; apply Bool.orb_false_r` |
| `merge_dim_empty_l` | T0 | 0 | `f_equal; apply Bool.orb_false_l` |
| `check_seq_hom` | T0 | 0 | `map_app + concat_app + reflexivity` |
| `diverged_clean_iff_barrier_free` | T1 | 0 | `diverged_absorbing` key lemma; `expr_list_rect`; ESuperstep+EVar cases added T2-F02 |
| `mode_monotone` | T1 | 0 | `incl_app_mono` + `diverged_absorbing` case split; ESuperstep+EVar cases added T2-F02 |
| `not_varying_converged_clean` | T1 | 0 | helper for `cdcf_check_agreement`; ESuperstep+EVar cases added T2-F02 |
| `cdcf_check_agreement` | T1 | 0 | uses `not_varying_converged_clean` for cond sub-expr; EVar+ESuperstep cases added T2-F02 |
| `varying_if_flags_barriers` | T1 | 0 | uses `diverged_clean_iff_barrier_free` as oracle |
| `superstep_outer_diverged_error` | T1 | 0 | F-01: `check Diverged (ESuperstep false body cont) ≠ []` |
| `warp_diverged_error` | T2 | 0 | **T2-WARP** — F-03 atomic: `check_warp Diverged EWarpPoint ≠ []`; models `Warp_collective_in_diverged_flow` |
| `warp_mode_monotone` | T2 | 0 | **T2-WARP** — quantified: `forall e, incl (check_warp Converged e) (check_warp Diverged e)` |
| `warp_varying_if_flags` | T2 | 0 | **T2-WARP** — context: `forall el, check_warp Converged (EIf EVary EWarpPoint el) ≠ []` |
| `return_barrier_skip_safe` | T2 | 0 | **NEW T2-RETURN** — compositionality: `forall m e, check m (EReturn e) = check m e`; EReturn transparent for barrier analysis |
| `env_lookup_extend_same` | T1 | 0 | **T2-F02** — lemma: `env_lookup (env_extend env x v) x = v` |
| `env_let_alias_varying` | T1 | 0 | **T2-F02** — F-02 core: let-alias variability propagated through ELet |
| `env_var_diverged_clean` | T1 | 0 | **T2-F02** — EVar carries no barrier under Diverged mode |
| `env_check_let_alias_catches` | T1 | 0 | **T2-F02** — F-02 soundness: `check_env` catches barrier behind let-alias |

**Total**: 20 theorems (ConvergenceSpec.v) + 7 theorems/corollaries (ConvergenceSemantics.v T3-S1+T3-S2) + 6 items (T3-S3: trace silence) + 4 items (T3-S4: core soundness — `core_frag` def + `eval_check_uniform` + `check_env_nonvarying_uniform` + `check_env_sound_core`) + 5 items (T3-S5: F-04 counterexample — `hazard` def + `hazard_vary` witness + `hazard_checker_blind` + `hazard_eval_thread0/1` + `hazard_not_barrier_safe`) + 11 items (T3-S6: ESuperstep grounding — `core_frag_ss` def + `core_frag_impl_ss` + `core_frag_ss_no_ret` + `eval_while_exits_immediately_ss` + `core_frag_ss_barrier_free_superstep_free` + `check_env_diverged_no_barriers_ss` + `eval_check_uniform_ss` + `check_env_sound_superstep` + `susp_hazard`/`susp_vary`/`susp_eval_thread0/1` + `semantic_f01_flagged` + `semantic_f01_not_barrier_safe` + `semantic_f01_corollary`) + 9 items (T3-S7: warp soundness — `erase_barrier` def + `warp_free` def + `warp_free_no_warps` + `check_warp_env` def + `check_warp_env_diverged_clean_warp_free` + `check_warp_env_diverged_no_warps` + `eval_check_warp_uniform` + `warp_safe` def + `check_warp_sound_core`), 0 admits, 0 axioms — `coqchk` passes (T3-S7)

## T3-S1 semantic layer (ConvergenceSemantics.v — new file)

| Item | Status | Notes |
|---|---|---|
| Type definitions (tid, value, venv, event, trace, outcome) | done | New file; ConvergenceSpec.v frozen |
| `Section Evaluator` / `Variable vary_val` | done | 0-axiom invariant preserved |
| `Fixpoint eval` (15 constructors) | done | fuel-indexed, total, option-valued |
| `eval_fuel_monotone` | done | 0 admits; uses for_loop_mono, eval_seq_mono, eval_args_mono helpers |
| `eval_app_seq_compose` | done | 0 admits; uses eval_seq_concat_acc helper |
| `coqchk` | passes | 0 new axioms |

## T3-S2 uniformity soundness (ConvergenceSemantics.v)

| Item | Status | Notes |
|---|---|---|
| `Definition env_agrees` | done | uniform-variable agreement predicate between two venvs |
| `Lemma env_agrees_extend` | done | lifts env_agrees through env_extend/venv_extend |
| `Fixpoint is_strongly_uniform` | done | stricter than is_varying_in_env: ELet binding must also be non-varying |
| `Lemma is_strongly_uniform_impl_is_not_varying` | done | is_strongly_uniform true → is_varying_in_env false |
| `Theorem not_varying_uniform` | done | 0 admits; forall vary_val fuel env e t1 t2 rho1 rho2, env_agrees → is_strongly_uniform env e = true → eval vary_val fuel t1 rho1 e = eval vary_val fuel t2 rho2 e |
| `Fixpoint is_var_free` | done | structural EVar-free check |
| `Lemma var_free_is_strongly_uniform_empty` | done | is_var_free ∧ is_varying=false → is_strongly_uniform [] e = true |
| `Lemma var_free_env_irrelevant` | done | EVar-free → is_strongly_uniform env-independent |
| `Corollary closed_uniform` | done | is_var_free ∧ is_varying=false → eval uniform across all tids and rhos |
| `coqchk` | passes | 0 new axioms |

**Design note**: `is_varying_in_env` has a soundness gap for ELet — it does not require the binding expression to be non-varying. A counterexample exists: `ELet x (EWhile EVary ELit) ELit` diverges for one thread and terminates for another when EVary values differ. `is_varying_strict` corrects this by additionally requiring the binding expression to be non-varying in ELet. `not_varying_uniform` uses `is_varying_strict`; a bridge lemma `is_varying_strict_impl_env` shows the strict predicate is stronger.

## T3-S4 core semantic soundness (ConvergenceSemantics.v — 2026-06-13)

| Item | Status | Notes |
|---|---|---|
| `Fixpoint core_frag` | done | Boolean predicate: true iff e contains no ESuperstep and no EReturn |
| `Definition erase_warp` + `Lemma erase_warp_app` + `Lemma erase_warp_no_barrier` | done | Project trace to barrier events only; EvWarp events are per-thread warp-collectives |
| `Definition barrier_safe` | done | All env-agreeing threads produce identical barrier-event sequences (erase_warp equality) |
| `Lemma core_frag_impl_superstep_free` | done | core_frag → superstep_free (needed to invoke T3-S3 lemmas) |
| `Lemma check_env_nonvarying_uniform_seq` / `_args` | done | Non-varying uniform inner loops for ESeq/EApp |
| `Lemma core_frag_no_ret` | done | core_frag e = true → eval never returns ORet |
| `Lemma eval_check_uniform` | done | Combined Part A (barrier-trace equality) + Part B (outcome equality for non-varying) by simultaneous fuel induction |
| `Theorem check_env_sound_core` | done | Main T3-S4 theorem: one-liner via proj1 (eval_check_uniform vary_val fuel) |
| `coqchk` | passes | 0 new axioms |

**Design note**: `barrier_safe` uses `erase_warp` equality rather than full trace equality, so `EWarpPoint` (which emits `EvWarp` per thread) does not create false negatives. The `check_env_sound_core` proof is a one-liner because all the weight is in `eval_check_uniform`, a combined induction on fuel establishing trace equality (Part A) and outcome equality for non-varying expressions (Part B) simultaneously — Part B feeds back into Part A for the condition-uniformity steps in `EIf`/`EWhile`/`EFor`.

## T3-S5 EReturn residual-divergence verdict (ConvergenceSemantics.v — 2026-06-13)

| Item | Status | Notes |
|---|---|---|
| `Definition hazard` | done | `ESeq [EIf EVary (EReturn ELit) ELit; EBarrier]` — varying early-return guarding a barrier |
| `Lemma hazard_checker_blind` | done | `check_env Converged [] hazard = []` by reflexivity — checker reports no error |
| `Definition hazard_vary` + `Lemma hazard_eval_thread0/1` | done | Concrete witness `vary_val 0 = 1, vary_val (S _) = 0`; thread 0 → `(ORet 0, [])`, thread 1 → `(ONorm 0, [EvBarrier])` by reflexivity |
| `Theorem hazard_not_barrier_safe` | done | `~ barrier_safe hazard_vary [] hazard` — constructive counterexample: erase_warp traces `[]` vs `[EvBarrier]` differ |
| `coqchk` | passes | 0 new axioms; `Print Assumptions` closed under global context |

**Design note (F-04)**: This is the boundary case of `check_env_sound_core` (T3-S4). That theorem requires `core_frag e = true`, which excludes `EReturn` (and `ESuperstep`) precisely because the checker's EReturn transparency (`check_env m env (EReturn e) = check_env m env e`, mirroring `Sarek_convergence.ml` TEReturn) is unsound at kernel granularity: a thread-varying early return that skips a later barrier passes the static check yet diverges the barrier trace across threads. `hazard_not_barrier_safe` makes the residual gap explicit and constructive rather than leaving it implicit in the `core_frag` precondition. The false negative is real in the abstract model; whether it is reachable in real Sarek kernels depends on whether a barrier can follow an early return within the same superstep — see F-04.

## T3-S6 ESuperstep semantic grounding (ConvergenceSemantics.v — 2026-06-13)

| Item | Status | Notes |
|---|---|---|
| `Fixpoint core_frag_ss` | done | `core_frag` ENLARGED to admit `ESuperstep false b c => core_frag_ss b && core_frag_ss c` (clause is `negb dv && ...`). `EReturn` still excluded. |
| `Lemma core_frag_impl_ss` | done | `core_frag e = true -> core_frag_ss e = true` — the enlarged fragment subsumes the core. |
| `Lemma core_frag_ss_no_ret` | done | No `core_frag_ss` expression returns `ORet` (superstep body cannot return; cont outcome is ORet-free by induction). |
| `Lemma eval_while_exits_immediately_ss` | done | `core_frag_ss` analogue of `eval_while_exits_immediately`. |
| `Lemma core_frag_ss_barrier_free_superstep_free` | done | `core_frag_ss e && barrier_free e -> superstep_free e` — a barrier-free `core_frag_ss` expression has no superstep at all. |
| `Lemma check_env_diverged_no_barriers_ss` | done | **Resolves the T3-S3 `superstep_free` side condition** over `core_frag_ss`: Diverged-clean `core_frag_ss` expressions are barrier-silent. |
| `Lemma eval_check_uniform_ss` | done | Combined Part A (barrier-trace) + Part B (outcome) uniformity by fuel induction over `core_frag_ss`; new ESuperstep case: implicit boundary `[EvBarrier]` emitted uniformly on all threads. |
| `Theorem check_env_sound_superstep` | done | `core_frag_ss e = true -> check_env Converged env e = [] -> barrier_safe vary_val env e`. Runtime grounding of `superstep_outer_diverged_error`. |
| `Definition susp_hazard` + `susp_vary` + `susp_eval_thread0/1` | done | Witness `EIf EVary (ESuperstep false EBarrier ELit) ELit`; thread 0 → `(ONorm 0, [EvBarrier; EvBarrier])`, thread 1 → `(ONorm 0, [])`. |
| `Lemma semantic_f01_flagged` + `Theorem semantic_f01_not_barrier_safe` | done | Checker flags `susp_hazard` AND it is genuinely not `barrier_safe`. |
| `Theorem semantic_f01_corollary` | done | Conjunction: flagged ∧ not barrier_safe — the F-01 `BarrierError` is a SOUND rejection. |
| `coqchk` | passes | 0 new axioms; `Print Assumptions` of all three theorems closed under global context. |

**Design note (semantic F-01)**: `check_env_sound_superstep` enlarges the verified fragment of `check_env_sound_core` from `core_frag` to `core_frag_ss`, admitting uniform-reachability (`dv=false`) supersteps. The implicit boundary barrier is emitted on every thread, so two env-agreeing threads keep equal barrier traces; thread-varying entry into a `dv=false` superstep is excluded because `check_env` raises `[BarrierError]` for it (the `Diverged, false` arm). `semantic_f01_corollary` couples the static F-01 verdict to a concrete runtime counterexample, giving `superstep_outer_diverged_error` operational meaning.

**TRUST BOUNDARY (`dv=true`)**: `core_frag_ss` admits only `dv=false` supersteps (`negb dv && ...`). A `dv=true` superstep emits the boundary barrier at runtime but is NOT flagged by `check_env` in Diverged mode — it is the front-end's assertion that the boundary is reached uniformly. `check_env_diverged_no_barriers_ss` would be FALSE if `dv=true` supersteps were admitted. The soundness guarantee therefore covers the `dv=false` fragment only; `dv=true` kernels inherit the (ASSUMED, not PROVEN) correctness of the divergence-uniformity annotation. Documented in `ASSUMPTIONS.md` §T3-S6.

## T3-S7 Warp-collective semantic soundness (ConvergenceSemantics.v — 2026-06-13)

| Item | Status | Notes |
|---|---|---|
| `Definition erase_barrier` | done | Trace projection keeping only `EvWarp` events (dual of `erase_warp`). |
| `Fixpoint warp_free` + `Lemma warp_free_no_warps` | done | `warp_free` = no `EWarpPoint`; a `warp_free` + `superstep_free` expression emits no `EvWarp` event (dual of `barrier_free_no_barriers`). |
| `Fixpoint check_warp_env` | done | Env-threaded warp checker mirroring `check_env` + `ConvergenceSpec.check_warp` EWarpPoint case; flags `EWarpPoint` (not `EBarrier`) under Diverged. |
| `Lemma check_warp_env_diverged_clean_warp_free` | done | Diverged-clean bridge lemma (ESuperstep carries no warp hazard, so it does not flag). |
| `Lemma check_warp_env_diverged_no_warps` | done | Diverged-mode warp silence; feeds the varying-branch cases of the uniformity induction. |
| `Lemma eval_check_warp_uniform` | done | Warp dual of `eval_check_uniform`: Part A `erase_barrier` equality + Part B outcome equality over `core_frag` with `check_warp_env Converged` clean. |
| `Section WarpModel` + `Variable warp_of : tid -> nat` + `Definition warp_safe` | done | `warp_safe` = `erase_barrier`-trace agreement restricted to thread pairs with `warp_of t1 = warp_of t2`. |
| `Theorem check_warp_sound_core` | done | `core_frag e = true -> check_warp_env Converged env e = [] -> warp_safe env e`, parametric in `warp_of`. |
| `coqchk` | passes | 0 new axioms; `Print Assumptions` of `check_warp_sound_core` / `eval_check_warp_uniform` / `warp_free_no_warps` all closed under global context. |

**Design note (warp soundness)**: `check_warp_sound_core` is the warp dual of `check_env_sound_core` (T3-S4). The same-warp restriction `warp_of t1 = warp_of t2` is a WEAKENING of the conclusion — `eval` is independent of `warp_of`, so the underlying trace equality holds for ALL thread pairs and `warp_safe` (same-warp pairs only) follows a fortiori. The checker catches a varying-EIf with `EWarpPoint` in one branch by checking both branches in Diverged mode, where `EWarpPoint` flags `WarpError`.

**PARAMETRIZATION (flagged)**: The plan asked whether the T3-S4 induction could be made parametric over (event class, agreement domain, checker) so T3-S7 becomes an instantiation. This was ATTEMPTED and found INFEASIBLE within budget: every case reduces the CONCRETE checker Fixpoint via `simpl; apply app_eq_nil` and the leaf inversions depend on concrete event constructors; a parametric induction would need ~15 checker-algebra lemmas threaded as Section hypotheses, itself a ~900-line abstraction with no working template. The induction was therefore DUPLICATED (`eval_check_warp_uniform`) via mechanical `EBarrier`<->`EWarpPoint` / `erase_warp`<->`erase_barrier` / `check_env`<->`check_warp_env` substitution. The two for-loop accumulator helpers and the trace-silence theorem are restated for `erase_barrier` (short proofs). See design note 1 in `theories/ConvergenceSemantics.v` §12.

**WARP-SIZE PARAMETERIZATION CLOSED**: `warp_of` is an abstract Section Variable; `check_warp_sound_core` is universally quantified over it. No fixed warp size (32, 64, ...) is baked into the proof. The `ASSUMPTIONS.md` warp-size parameterization item is now closed (§T3-S7).

## Test intensity

- **Conformance**: `test/test_convergence_conformance.ml` — 17 properties (`test_convergence_conformance`), 1000–2000 tests each — **17/17 GREEN** (2 new F-02 env-threaded properties added T2-F02; 1 new randomized warp property added T2-WARP+; 1 new return_barrier_skip_safe property added T2-RETURN; 3 new dedicated ESuperstep properties added T1A-CONF: superstep_outer_diverged_error, superstep_no_entry_error_converged, superstep_body_errors_propagate)
- **Extraction**: `test/test_convergence_extraction.ml` — 7 tests (`test_convergence_extraction`) — **7/7 GREEN** (extr:check_warp_agrees added T2-WARP+)
- **Live CMBT**: `formal/convergence-safety/test/test_convergence_live.ml` — 10 tests including F-01 + F-02 regressions — **10/10 GREEN**

## Known gates

None.

## Open findings

| ID | Title | Status | Classification |
|---|---|---|---|
| F-01 | TESuperstep hard-resets ctx.mode, discarding inherited Diverged context | **RESOLVED** | OCaml fix in PR #181 (merged); Rocq theorem `superstep_outer_diverged_error` in PR #182 |
| F-02 | is_thread_varying is binding-blind — let-aliased thread-varying values not propagated | **RESOLVED (formal)** | OCaml fix in PR #181 (merged); Rocq env-threaded model in T2-F02: `Env`, `is_varying_in_env`, `check_env`, 3 new theorems |
| F-04 | EReturn transparency is a kernel-granularity false negative — varying early return skipping a later barrier passes `check_env` but is not `barrier_safe` | **OPEN (formal counterexample)** | Classification `a'` (spec/checker models the real TEReturn transparency); Rocq counterexample `hazard_not_barrier_safe` in T3-S5; reachability in real kernels pending |

See `findings/DIVERGENCE_FINDINGS.md` for full descriptions.

## CMBT completeness chain

| Link | Item | Status |
|---|---|---|
| 1 | Spec source (`theories/ConvergenceSpec.v`) | checked |
| 2 | Abstract model in conformance test (`test_convergence_conformance.ml`) | checked |
| 3 | Integration target (`Sarek_convergence.check_expr`) | checked |
| 4 | Conformance tests GREEN | checked |
| 5 | Extraction tests GREEN (incl. check_warp CMBT link) | checked |
| 6 | `coqchk` passes (0 axioms) | checked |
| 7 | Open findings documented in `findings/DIVERGENCE_FINDINGS.md` | checked |

## Last bake-off

None yet.

## Next session prompt

```
Resume ConvergenceSafety (apparatus v1.2.1, grade A).
State: 20/20 theorems proven in ConvergenceSpec.v + 38 theorems/defs in ConvergenceSemantics.v (T3-S1..S7), 0 admits, 0 axioms, coqchk passes. T3-S7 complete.
Conformance: 17/17 green. Extraction: 7/7 green. Live CMBT: 10/10 green.
F-01 RESOLVED (OCaml + Rocq). F-02 RESOLVED (OCaml + Rocq env-threaded model).
F-03 (WarpConvergence) RESOLVED (Rocq: EWarpPoint/WarpError/check_warp/warp_diverged_error/warp_mode_monotone/warp_varying_if_flags; documented in findings/DIVERGENCE_FINDINGS.md).
T2-RETURN RESOLVED (Rocq: EReturn/return_barrier_skip_safe/return_converged_clean; TEReturn exits without crossing any barrier).
T1A-CONF RESOLVED (3 dedicated ESuperstep QCheck properties; PR #182 merged).
T3-S1 RESOLVED (ConvergenceSemantics.v: semantic domain + fuel-indexed big-step evaluator + eval_fuel_monotone + eval_app_seq_compose; 0 admits, 0 axioms, coqchk passes).
T3-S2 RESOLVED (ConvergenceSemantics.v: env_agrees, is_strongly_uniform, not_varying_uniform, is_var_free, var_free_is_strongly_uniform_empty, var_free_env_irrelevant, closed_uniform; 0 admits, 0 axioms, coqchk passes).
  KEY FINDING: is_varying_in_env has a soundness gap for ELet — is_strongly_uniform corrects it by requiring ELet bindings to also be non-varying.
T3-S3 RESOLVED (ConvergenceSemantics.v: no_barrier_event, superstep_free, barrier_free_no_barriers, diverged_clean_no_barriers; 0 admits, 0 axioms, coqchk passes).
T3-S4 RESOLVED (ConvergenceSemantics.v: core_frag, erase_warp, barrier_safe, eval_check_uniform, check_env_sound_core; 0 admits, 0 axioms, coqchk passes).
  KEY DESIGN: check_env_sound_core proved via combined eval_check_uniform lemma (simultaneous Part A barrier-trace + Part B outcome uniformity by fuel induction).
T3-S5 RESOLVED (ConvergenceSemantics.v: hazard, hazard_vary, hazard_checker_blind, hazard_eval_thread0/1, hazard_not_barrier_safe; new finding F-04; 0 admits, 0 axioms, coqchk passes).
  KEY FINDING (F-04): EReturn transparency is a kernel-granularity false negative — hazard ESeq[EIf EVary (EReturn ELit) ELit; EBarrier] passes check_env Converged [] = [] but is NOT barrier_safe (thread 0 returns early trace [], thread 1 reaches barrier trace [EvBarrier]). This is the boundary case excluded by the core_frag precondition of check_env_sound_core.
T3-S6 RESOLVED (ConvergenceSemantics.v: core_frag_ss, core_frag_impl_ss, core_frag_ss_no_ret, eval_while_exits_immediately_ss, core_frag_ss_barrier_free_superstep_free, check_env_diverged_no_barriers_ss, eval_check_uniform_ss, check_env_sound_superstep, susp_hazard/susp_vary/susp_eval_thread0/1, semantic_f01_flagged, semantic_f01_not_barrier_safe, semantic_f01_corollary; 0 admits, 0 axioms, coqchk passes).
  KEY DESIGN: check_env_sound_superstep enlarges check_env_sound_core's fragment from core_frag to core_frag_ss (admits dv=false supersteps). semantic_f01_corollary grounds superstep_outer_diverged_error at runtime: susp_hazard EIf EVary (ESuperstep false EBarrier ELit) ELit is flagged AND not barrier_safe (thread 0 [EvBarrier;EvBarrier], thread 1 []).
  TRUST BOUNDARY: core_frag_ss admits only dv=false supersteps; dv=true is a documented trust boundary (ASSUMPTIONS.md §T3-S6) — emits the barrier at runtime but is not flagged in Diverged mode. The T3-S3 superstep_free side condition is resolved over core_frag_ss via check_env_diverged_no_barriers_ss.
T3-S7 RESOLVED (ConvergenceSemantics.v: erase_barrier, warp_free, warp_free_no_warps, check_warp_env, check_warp_env_diverged_clean_warp_free, check_warp_env_diverged_no_warps, eval_check_warp_uniform, warp_safe, check_warp_sound_core in Section WarpModel with Variable warp_of; 0 admits, 0 axioms, coqchk passes).
  KEY DESIGN: check_warp_sound_core is the warp dual of check_env_sound_core, parametric in warp_of (warp-size parameterization CLOSED, ASSUMPTIONS §T3-S7). warp_safe = erase_barrier-trace agreement restricted to same-warp pairs (warp_of t1 = warp_of t2), which holds a fortiori since eval is independent of warp_of.
  PARAMETRIZATION: attempted making the T3-S4 induction parametric over (event class, agreement domain, checker); flagged INFEASIBLE within budget (concrete checker reductions + per-constructor leaf inversions) and DUPLICATED as eval_check_warp_uniform via mechanical substitution.
Next: continue T3-SEMANTIC breakdown (T3-S8).
Run /formal-check before any lock or milestone.
```
