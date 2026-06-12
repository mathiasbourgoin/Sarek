(******************************************************************************)
(* ConvergenceSafetyExtraction.v
 *
 * Rocq → OCaml extraction configuration for ConvergenceSpec.
 * Extracts the abstract model to ConvergenceModel.ml; committed to repo
 * so test/test_convergence_extraction.ml can use it as the oracle.
 *
 * Run to refresh after spec changes:
 *   cd formal/convergence-safety
 *   eval $(opam env --switch=~/dev/SPOC)
 *   coqc -R theories ConvergenceSpec \
 *         -output-directory extraction \
 *         extraction/ConvergenceSafetyExtraction.v
 ******************************************************************************)

From Stdlib Require Import Extraction ExtrOcamlBasic ExtrOcamlNatInt.
From ConvergenceSpec Require Import ConvergenceSpec.

(* Let exec_mode and error extract as regular OCaml variants —
 * do NOT use Extract Inductive here so the type definitions are emitted. *)

Extraction "ConvergenceModel.ml"
  is_varying
  barrier_free
  has_diverging_cf
  check
  check_warp
  merge_dim_usage
  empty_dim_usage.
