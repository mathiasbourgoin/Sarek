# ConvergenceSafety — Project Status

**Grade**: A (apparatus-native)
**Apparatus version**: 1.2.1
**Host profile**: SPOC/sarek
**Architecture**: 3-layer
**Built at**: 2026-06-11
**Last updated**: 2026-06-13 (T3-S2: uniformity soundness via eval semantics — `env_agrees`, `is_strongly_uniform`, `not_varying_uniform`, `closed_uniform` — `ConvergenceSemantics.v`, 0 admits, 0 axioms)

## Project

ConvergenceSafety is a formal Rocq specification of the GPU barrier safety analysis for the Sarek frontend of SPOC. It proves correctness properties of `Sarek_convergence.check_expr`, which statically detects barriers placed inside diverged control flow on the abstract `expr` AST. The project covers 20 theorems at Tiers 0–2 (lattice laws through control-flow monotonicity, warp-collective safety, and early-return compositionality) — 16 Theorems + 2 named Lemmas (`env_lookup_extend_same`, `not_varying_converged_clean`) + 2 strengthened warp theorems (`warp_mode_monotone`, `warp_varying_if_flags`) — including the F-01 safety property for `ESuperstep` (Phase 1a), 3 env-threaded F-02 theorems (T2-F02), 3 warp theorems for the `WarpConvergence` error class (T2-WARP), and the `return_barrier_skip_safe` compositionality theorem for `EReturn` (T2-RETURN). Validated by 14 QCheck conformance properties against an abstract OCaml model and 7 extraction tests exercising the extracted `ConvergenceModel` module (including `check_warp` CMBT link).

## Trust root

Assumptions documented in `ASSUMPTIONS.md`:
- Rocq kernel soundness (standard)
- OCaml extraction + compiler (standard TCB)
- Abstract `expr` model faithfully represents the barrier-relevant paths in `Sarek_convergence.ml` (verified by code inspection; elided constructors documented; ESuperstep added Phase 1a)
- `is_thread_varying` correctness for `TEVar`/`TEIntrinsicConst` (depends on `Sarek_core_primitives.is_thread_varying` — outside scope)
- `WarpConvergence` error class: **IN SCOPE (T2-WARP)** — `EWarpPoint`/`WarpError`/`check_warp`/`warp_diverged_error` modeled and proven

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

**Total**: 20 theorems (ConvergenceSpec.v) + 7 theorems/corollaries (ConvergenceSemantics.v T3-S1+T3-S2), 0 admits, 0 axioms — `coqchk` passes (T3-S2)

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
State: 20/20 theorems proven in ConvergenceSpec.v + 7 theorems/corollaries in ConvergenceSemantics.v, 0 admits, 0 axioms, coqchk passes. T3-S2 complete.
Conformance: 17/17 green. Extraction: 7/7 green. Live CMBT: 10/10 green.
F-01 RESOLVED (OCaml + Rocq). F-02 RESOLVED (OCaml + Rocq env-threaded model).
F-03 (WarpConvergence) RESOLVED (Rocq: EWarpPoint/WarpError/check_warp/warp_diverged_error/warp_mode_monotone/warp_varying_if_flags; documented in findings/DIVERGENCE_FINDINGS.md).
T2-RETURN RESOLVED (Rocq: EReturn/return_barrier_skip_safe/return_converged_clean; TEReturn exits without crossing any barrier).
T1A-CONF RESOLVED (3 dedicated ESuperstep QCheck properties; PR #182 merged).
T3-S1 RESOLVED (ConvergenceSemantics.v: semantic domain + fuel-indexed big-step evaluator + eval_fuel_monotone + eval_app_seq_compose; 0 admits, 0 axioms, coqchk passes).
T3-S2 RESOLVED (ConvergenceSemantics.v: env_agrees, is_strongly_uniform, not_varying_uniform, is_var_free, var_free_is_strongly_uniform_empty, var_free_env_irrelevant, closed_uniform; 0 admits, 0 axioms, coqchk passes).
  KEY FINDING: is_varying_in_env has a soundness gap for ELet — is_strongly_uniform corrects it by requiring ELet bindings to also be non-varying.
Next: T3-S3 (next T3-SEMANTIC subtask per PLAN.md).
Run /formal-check before any lock or milestone.
```
