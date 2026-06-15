(******************************************************************************)
(* MutExtraction.v
 *
 * Rocq -> OCaml extraction for MutSpec.
 * Extracts the mutable-binding type inference model to MutModel.ml;
 * committed to repo so test/test_type_safety_conformance.ml can use it for
 * T3-S4 smoke tests.
 *
 * Run to refresh after spec changes:
 *   cd formal/type-safety
 *   eval $(opam env --switch=~/dev/SPOC)
 *   rocq compile -R theories TypeSafety \
 *         -output-directory extraction \
 *         extraction/MutExtraction.v
 ******************************************************************************)

From Stdlib Require Import Extraction ExtrOcamlBasic ExtrOcamlNatInt.
From TypeSafety Require Import TypeSafetySpec.
From TypeSafety Require Import VecSpec.
From TypeSafety Require Import RegistrySpec.
From TypeSafety Require Import ControlFlowSpec.
From TypeSafety Require Import OperatorSpec.
From TypeSafety Require Import FunSpec.
From TypeSafety Require Import MutSpec.

Extraction "MutModel.ml"
  infer_mut_type.
