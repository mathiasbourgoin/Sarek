# ConvergenceSafety — Project Status

**Grade**: A (apparatus-native)
**Apparatus version**: 1.1.0
**Host profile**: SPOC/sarek
**Architecture**: 3-layer
**Built at**: 2026-06-11
**Last updated**: 2026-06-12 (T2-RETURN: EReturn early-return barrier-skip — 20 theorems, 14 conformance properties, 7 extraction tests)

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

**Total**: 20 theorems, 0 admits, 0 axioms — `coqchk` passes (T2-RETURN)

## Test intensity

- **Conformance**: `test/test_convergence_conformance.ml` — 14 properties (`test_convergence_conformance`), 1000–2000 tests each — **14/14 GREEN** (2 new F-02 env-threaded properties added T2-F02; 1 new randomized warp property added T2-WARP+; 1 new return_barrier_skip_safe property added T2-RETURN)
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
Resume ConvergenceSafety (apparatus v1.1.0, grade A).
State: 20/20 theorems proven, 0 admits, 0 axioms, coqchk passes. T2-RETURN complete.
Conformance: 14/14 green. Extraction: 7/7 green. Live CMBT: 10/10 green.
F-01 RESOLVED (OCaml + Rocq). F-02 RESOLVED (OCaml + Rocq env-threaded model).
F-03 (WarpConvergence) RESOLVED (Rocq: EWarpPoint/WarpError/check_warp/warp_diverged_error/warp_mode_monotone/warp_varying_if_flags; documented in findings/DIVERGENCE_FINDINGS.md).
T2-RETURN RESOLVED (Rocq: EReturn/return_barrier_skip_safe/return_converged_clean; TEReturn exits without crossing any barrier).
Next: T3-GATE (human decision on T3-SEMANTIC).
Run /formal-check before any lock or milestone.
```
