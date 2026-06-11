# ConvergenceSafety — Project Status

**Grade**: A (apparatus-native)
**Apparatus version**: 1.1.0
**Host profile**: SPOC/sarek
**Architecture**: 3-layer
**Built at**: 2026-06-11
**Last updated**: 2026-06-11 (Phase 1a)

## Project

ConvergenceSafety is a formal Rocq specification of the GPU barrier safety analysis for the Sarek frontend of SPOC. It proves correctness properties of `Sarek_convergence.check_expr`, which statically detects barriers placed inside diverged control flow on the abstract `expr` AST. The project covers 12 theorems at Tiers 0–1 (lattice laws through control-flow monotonicity), including the F-01 safety property for `ESuperstep` (Phase 1a). Validated by 10 QCheck conformance properties against an abstract OCaml model and 6 extraction tests exercising the extracted `ConvergenceModel` module.

## Trust root

Assumptions documented in `ASSUMPTIONS.md`:
- Rocq kernel soundness (standard)
- OCaml extraction + compiler (standard TCB)
- Abstract `expr` model faithfully represents the barrier-relevant paths in `Sarek_convergence.ml` (verified by code inspection; elided constructors documented; ESuperstep added Phase 1a)
- `is_thread_varying` correctness for `TEVar`/`TEIntrinsicConst` (depends on `Sarek_core_primitives.is_thread_varying` — outside scope)
- `WarpConvergence` error class: out of scope

## Proof status

| Theorem | Tier | Admits | Notes |
|---|---|---|---|
| `merge_dim_comm` | T0 | 0 | `f_equal; apply Bool.orb_comm` |
| `merge_dim_assoc` | T0 | 0 | `f_equal; apply Bool.orb_assoc` |
| `merge_dim_idempotent` | T0 | 0 | `f_equal; apply Bool.orb_diag` |
| `merge_dim_empty_r` | T0 | 0 | `f_equal; apply Bool.orb_false_r` |
| `merge_dim_empty_l` | T0 | 0 | `f_equal; apply Bool.orb_false_l` |
| `check_seq_hom` | T0 | 0 | `map_app + concat_app + reflexivity` |
| `diverged_clean_iff_barrier_free` | T1 | 0 | `diverged_absorbing` key lemma; `expr_list_rect`; ESuperstep case added Phase 1a |
| `mode_monotone` | T1 | 0 | `incl_app_mono` + `diverged_absorbing` case split; ESuperstep case added Phase 1a |
| `not_varying_converged_clean` | T1 | 0 | helper for `cdcf_check_agreement`; ESuperstep case added Phase 1a |
| `cdcf_check_agreement` | T1 | 0 | uses `not_varying_converged_clean` for cond sub-expr; ESuperstep case added Phase 1a |
| `varying_if_flags_barriers` | T1 | 0 | uses `diverged_clean_iff_barrier_free` as oracle |
| `superstep_outer_diverged_error` | T1 | 0 | **NEW Phase 1a** — F-01: `check Diverged (ESuperstep false body cont) ≠ []` |

**Total**: 12 theorems, 0 admits, 0 axioms — `coqchk` passes (Phase 1a, PR #182)

## Test intensity

- **Conformance**: `test/test_convergence_conformance.ml` — 10 properties (`test_convergence_conformance`), 1000–2000 tests each — **10/10 GREEN**
- **Extraction**: `test/test_convergence_extraction.ml` — 6 tests (`test_convergence_extraction`) — **6/6 GREEN**
- **Live CMBT**: `formal/convergence-safety/test/test_convergence_live.ml` — 10 tests including F-01 + F-02 regressions — **10/10 GREEN**

## Known gates

None.

## Open findings

| ID | Title | Status | Classification |
|---|---|---|---|
| F-01 | TESuperstep hard-resets ctx.mode, discarding inherited Diverged context | **RESOLVED** | OCaml fix in PR #181 (merged); Rocq theorem `superstep_outer_diverged_error` in PR #182 |
| F-02 | is_thread_varying is binding-blind — let-aliased thread-varying values not propagated | OPEN | `b` — OCaml fix in PR #181 (merged, `varying_vars` context); spec mirrors old checker; T2 formal tracking deferred |

See `findings/DIVERGENCE_FINDINGS.md` for full descriptions.

## CMBT completeness chain

| Link | Item | Status |
|---|---|---|
| 1 | Spec source (`theories/ConvergenceSpec.v`) | checked |
| 2 | Abstract model in conformance test (`test_convergence_conformance.ml`) | checked |
| 3 | Integration target (`Sarek_convergence.check_expr`) | checked |
| 4 | Conformance tests GREEN | checked |
| 5 | Extraction tests GREEN | checked |
| 6 | `coqchk` passes (0 axioms) | checked |
| 7 | Open findings documented in `findings/DIVERGENCE_FINDINGS.md` | checked |

## Last bake-off

None yet.

## Next session prompt

```
Resume ConvergenceSafety (apparatus v1.1.0, grade A).
State: 12/12 theorems proven, 0 admits, 0 axioms, coqchk passes. Phase 1a complete (PR #182).
Conformance: 10/10 green. Extraction: 6/6 green. Live CMBT: 10/10 green.
F-01 RESOLVED (OCaml + Rocq). F-02 OPEN (OCaml fixed, formal T2 deferred).
Next candidates: (a) extend conformance tests to cover ESuperstep in the abstract model;
(b) T2 environment-threaded is_varying for F-02 formal tracking;
(c) WarpConvergence error class.
Run /formal-check before any lock or milestone.
```
