(******************************************************************************)
(* PatExtraction.v
 *
 * Rocq -> OCaml extraction for PatternSpec.
 * Extracts the pattern-match type inference model to PatternModel.ml;
 * committed to repo so test/test_type_safety_conformance.ml can use it for
 * T3-S5 smoke tests.
 *
 * Run to refresh after spec changes:
 *   cd formal/type-safety
 *   eval $(opam env --switch=~/dev/SPOC)
 *   rocq compile -R theories TypeSafety \
 *         -output-directory extraction \
 *         extraction/PatExtraction.v
 ******************************************************************************)

From Stdlib Require Import Extraction ExtrOcamlBasic ExtrOcamlNatInt.
From TypeSafety Require Import TypeSafetySpec.
From TypeSafety Require Import VecSpec.
From TypeSafety Require Import RegistrySpec.
From TypeSafety Require Import ControlFlowSpec.
From TypeSafety Require Import OperatorSpec.
From TypeSafety Require Import FunSpec.
From TypeSafety Require Import MutSpec.
From TypeSafety Require Import PatternSpec.

Extraction "PatternModel.ml"
  infer_pat_type.
