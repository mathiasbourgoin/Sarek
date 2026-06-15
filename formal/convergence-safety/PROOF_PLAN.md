# PROOF_PLAN — convergence-safety

Layer map + proof-obligation triage. Written at apparatus v1.2.1 retrofit
(2026-06-13) covering T0–T3 obligations. All T0–T2 obligations and T3-S1
are discharged; T3-S2..S8 are UNKNOWN (pending).

## Layer map

3-layer architecture (no rate-aware / issuance / compound-fee logic).

| Boundary | Obligation | Carrier file |
|---|---|---|
| Spec (L1) → Model (L2) | extraction fidelity (OCaml/Zarith) | conformance tests (TESTED — see ASSUMPTIONS.md §Universal TCB) |
| Model (L2) → target | per-expression projection equality | `test_helpers/coverage_probe.ml` + `test/test_convergence_conformance.ml` |
| Spec → semantics | barrier-trace uniformity (T3 semantic soundness) | `theories/ConvergenceSemantics.v` (T3-S1 through T3-S8) |

## Obligation triage — T0–T3

| # | Property | Tier | Verdict | Notes |
|---|---|---|---|---|
| 1 | `merge_dim_comm` — `merge_dim_usage` commutative | T0 | PROVEN | `f_equal; apply Bool.orb_comm` |
| 2 | `merge_dim_assoc` — `merge_dim_usage` associative | T0 | PROVEN | `f_equal; apply Bool.orb_assoc` |
| 3 | `merge_dim_idempotent` — `merge_dim_usage` idempotent | T0 | PROVEN | `f_equal; apply Bool.orb_diag` |
| 4 | `merge_dim_empty_r` — right identity | T0 | PROVEN | `f_equal; apply Bool.orb_false_r` |
| 5 | `merge_dim_empty_l` — left identity | T0 | PROVEN | `f_equal; apply Bool.orb_false_l` |
| 6 | `check_seq_hom` — sequential composition homomorphism | T0 | PROVEN | `map_app + concat_app + reflexivity` |
| 7 | `diverged_clean_iff_barrier_free` — Diverged mode barrier equivalence | T1 | PROVEN | `diverged_absorbing` key lemma; `expr_list_rect` |
| 8 | `mode_monotone` — strengthening never removes errors | T1 | PROVEN | `incl_app_mono + diverged_absorbing` case split |
| 9 | `not_varying_converged_clean` — non-varying → Converged is clean | T1 | PROVEN | Helper for `cdcf_check_agreement` |
| 10 | `cdcf_check_agreement` — no diverging CF → Converged is clean | T1 | PROVEN | Uses `not_varying_converged_clean` |
| 11 | `varying_if_flags_barriers` — varying cond + non-empty body → error | T1 | PROVEN | Uses `diverged_clean_iff_barrier_free` as oracle |
| 12 | `superstep_outer_diverged_error` — F-01: ESuperstep Diverged entry errors | T1 | PROVEN | `simpl + discriminate` |
| 13 | `env_lookup_extend_same` — env_extend same key returns stored value | T1 | PROVEN | F-02 helper |
| 14 | `env_let_alias_varying` — ELet propagates variability | T1 | PROVEN | F-02 core |
| 15 | `env_var_diverged_clean` — EVar under Diverged is barrier-free | T1 | PROVEN | F-02 |
| 16 | `env_check_let_alias_catches` — `check_env` catches barrier behind let-alias | T1 | PROVEN | F-02 soundness |
| 17 | `warp_diverged_error` — F-03: EWarpPoint under Diverged errors | T2 | PROVEN | T2-WARP atomic |
| 18 | `warp_mode_monotone` — warp check monotone in mode | T2 | PROVEN | `expr_list_rect + diverged_absorbing` |
| 19 | `warp_varying_if_flags` — EWarpPoint under varying EIf always caught | T2 | PROVEN | T2-WARP context |
| 20 | `return_barrier_skip_safe` — EReturn transparent for barrier analysis | T2 | PROVEN | `simpl + reflexivity` |
| 21 | `eval_fuel_monotone` — fuel-indexed evaluator monotone in fuel | T2 | PROVEN | T3-S1; `induction + expr case split` |
| 22 | `eval_app_seq_compose` — trace homomorphism over ESeq | T2 | PROVEN | T3-S1; semantic mirror of `check_seq_hom` |
| 23 | `is_varying_in_env` soundness — varying iff ∃ thread s.t. values differ | T2 | UNKNOWN | T3-S2 (blocked by T3-S1, unblocked) |
| 24 | Trace silence of barrier-free expressions | T2 | UNKNOWN | T3-S3 |
| 25 | Core semantic soundness of `check_env` | T2 | UNKNOWN | T3-S4 (blocked by T3-S2, T3-S3); main L effort |
| 26 | EReturn residual-divergence verdict (expected F-04) | T2 | UNKNOWN | T3-S5 (blocked by T3-S4) |
| 27 | ESuperstep semantic grounding | T2 | UNKNOWN | T3-S6 (blocked by T3-S4) |
| 28 | Warp-collective semantic soundness | T2 | UNKNOWN | T3-S7 (blocked by T3-S4) |
| 29 | Extraction + differential conformance for semantics (CMBT closure) | T2 | UNKNOWN | T3-S8 (blocked by T3-S5) |
| 30 | Deadlock freedom (whole-kernel lockstep execution model) | T3 | OUT OF SCOPE | Requires lockstep execution semantics; recorded in ASSUMPTIONS.md |
| 31 | Warp size parameterization correctness | T3 | OUT OF SCOPE | T3; see ASSUMPTIONS.md |
| 32 | Extraction mechanism soundness | T3 | ASSUMED | Universal TCB row; see ASSUMPTIONS.md §Universal TCB |
