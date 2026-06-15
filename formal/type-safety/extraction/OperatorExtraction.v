(******************************************************************************)
(* OperatorExtraction.v
 *
 * Rocq -> OCaml extraction for OperatorSpec.
 * Extracts the operator type inference model to OperatorModel.ml;
 * committed to repo so test/test_type_safety_conformance.ml can use it for
 * T3-S2 smoke tests.
 *
 * Run to refresh after spec changes:
 *   cd formal/type-safety
 *   eval $(opam env --switch=~/dev/SPOC)
 *   rocq compile -R theories TypeSafety \
 *         -output-directory extraction \
 *         extraction/OperatorExtraction.v
 ******************************************************************************)

From Stdlib Require Import Extraction ExtrOcamlBasic ExtrOcamlNatInt.
From TypeSafety Require Import TypeSafetySpec.
From TypeSafety Require Import VecSpec.
From TypeSafety Require Import RegistrySpec.
From TypeSafety Require Import ControlFlowSpec.
From TypeSafety Require Import OperatorSpec.

Extraction "OperatorModel.ml"
  infer_op_type.
