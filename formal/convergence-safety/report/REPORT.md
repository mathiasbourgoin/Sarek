# ConvergenceSafety — Report

## Summary

Formal specification and CMBT certification of `Sarek_convergence.ml`,
the GPU barrier-safety analysis pass for the Sarek OCaml GPU DSL.

**Grade**: A (apparatus-native)
**Lock state**: LOCKED — 11/11 theorems, 0 admits, 0 axioms, coqchk PASS

## What was proved

11 Rocq theorems in two tiers:

- **T0 (5)**: `dim_usage` forms a join-semilattice under field-wise `||`
  (`merge_dim_{comm, assoc, idem, empty_r, empty_l}`); sequential
  composition of checks is homomorphic (`check_seq_hom`).
- **T1 (6)**: Core barrier-safety invariants — `diverged_clean_iff_barrier_free`
  (the keystone), `mode_monotone`, `not_varying_converged_clean`,
  `cdcf_check_agreement`, `varying_if_flags_barriers`.

## What was tested

- `test_convergence_conformance`: 10 QCheck2 properties × 1000–2000 samples
  each, 10/10 green. Abstract inline model tested.
- `test_convergence_extraction`: 6 QCheck2 properties × 1500–2000 samples
  each, 6/6 green. Extracted OCaml model vs inline model — confirms
  extraction fidelity.

## Known divergences

- **F-01**: `TESuperstep` ignores outer execution mode (candidate bug).
  Not yet classified by user. See `findings/DIVERGENCE_FINDINGS.md §F-01`.

## Assumptions

See `ASSUMPTIONS.md` for the full abstract↔real correspondence table and the
list of elided constructors.

## Apparatus version

1.1.0
