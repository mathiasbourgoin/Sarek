# ConvergenceSafety — Project Status

**Grade**: A (apparatus-native)
**Apparatus version**: 1.1.0
**Host profile**: SPOC/sarek
**Architecture**: 3-layer
**Built at**: 2026-06-11

## Project

ConvergenceSafety is a formal Rocq specification of the GPU barrier safety analysis for the Sarek frontend of SPOC. It proves correctness properties of `Sarek_convergence.check_expr`, which statically detects barriers placed inside diverged control flow on the abstract `expr` AST. The project covers 11 theorems at Tiers 0–1 (lattice laws through control-flow monotonicity), validated by 10 QCheck conformance properties against an abstract OCaml model and 6 extraction tests exercising the extracted `ConvergenceModel` module.

## Trust root

Assumptions documented in `ASSUMPTIONS.md`:
- Rocq kernel soundness (standard)
- OCaml extraction + compiler (standard TCB)
- Abstract `expr` model faithfully represents the 11 barrier-relevant paths in `Sarek_convergence.ml` (verified by code inspection; elided constructors documented)
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
| `diverged_clean_iff_barrier_free` | T1 | 0 | `diverged_absorbing` key lemma; `expr_list_rect` |
| `mode_monotone` | T1 | 0 | `incl_app_mono` + `diverged_absorbing` case split |
| `not_varying_converged_clean` | T1 | 0 | helper for `cdcf_check_agreement` |
| `cdcf_check_agreement` | T1 | 0 | uses `not_varying_converged_clean` for cond sub-expr |
| `varying_if_flags_barriers` | T1 | 0 | uses `diverged_clean_iff_barrier_free` as oracle |

**Total**: 11 theorems, 0 admits, 0 axioms — `coqchk` passes

## Test intensity

- **Conformance**: `test/test_convergence_conformance.ml` — 10 properties (`test_convergence_conformance`), 1000–2000 tests each — **10/10 GREEN**
- **Extraction**: `test/test_convergence_extraction.ml` — 6 tests (`test_convergence_extraction`) — **6/6 GREEN**

## Known gates

None.

## Open findings

| ID | Title | Status | Classification |
|---|---|---|---|
| F-01 | TESuperstep hard-resets ctx.mode to Converged, discarding inherited Diverged context | OPEN | `a'` — spec gap; ESuperstep constructor missing from ConvergenceSpec.v; Phase 1a plan in report/PHASE1A_PLAN.md |
| F-02 | is_thread_varying is binding-blind — let-aliased thread-varying values not propagated | OPEN | `b` — implementation bug in Sarek_convergence.ml; spec accurately mirrors current (incomplete) checker; T2 dataflow fix deferred |

See `findings/DIVERGENCE_FINDINGS.md §F-01` for full description and reproduction sketch.

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
State: 11/11 theorems proven, 0 admits, 0 axioms, coqchk passes.
Conformance: 10/10 green. Extraction: 6/6 green.
Open finding: F-01 (TESuperstep ignores outer exec mode) — pending user classification.
To proceed: classify F-01 (a/a'/b/c), then either add regression test (if a'/b)
or close as out-of-scope (if a/c). Next theorem candidates: WarpConvergence error
class or T2 is_varying_semantic_soundness (needs execution semantics for texpr).
Run /formal-check before any lock or milestone.
```
