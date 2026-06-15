(******************************************************************************)
(* ControlFlowExtraction.v
 *
 * Rocq -> OCaml extraction configuration for ControlFlowSpec.
 * Extracts the control-flow type inference model to ControlFlowModel.ml;
 * committed to repo so test/test_type_safety_conformance.ml can use it for
 * T3-S1 smoke tests.
 *
 * Run to refresh after spec changes:
 *   cd formal/type-safety
 *   eval $(opam env --switch=~/dev/SPOC)
 *   rocq compile -R theories TypeSafety \
 *         -output-directory extraction \
 *         extraction/ControlFlowExtraction.v
 ******************************************************************************)

From Stdlib Require Import Extraction ExtrOcamlBasic ExtrOcamlNatInt.
From TypeSafety Require Import TypeSafetySpec.
From TypeSafety Require Import VecSpec.
From TypeSafety Require Import RegistrySpec.
From TypeSafety Require Import ControlFlowSpec.

Extraction "ControlFlowModel.ml"
  infer_cf_type.
