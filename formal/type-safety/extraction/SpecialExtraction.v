(******************************************************************************)
(* SpecialExtraction.v
 *
 * Rocq -> OCaml extraction for SpecialSpec.
 * Extracts the special-form (EReturn/ECreateArray/ETyped) type inference model
 * to SpecialModel.ml; committed to repo so
 * test/test_type_safety_conformance.ml can use it for T3-S7 smoke tests.
 *
 * Run to refresh after spec changes:
 *   cd formal/type-safety
 *   eval $(opam env --switch=<your-opam-switch>)   (* e.g. the repo root *)
 *   rocq compile -R theories TypeSafety \
 *         -output-directory extraction \
 *         extraction/SpecialExtraction.v
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
From TypeSafety Require Import ConstrSpec.
From TypeSafety Require Import SpecialSpec.

Set Extraction Output Directory "extraction".

Extraction "SpecialModel.ml"
  infer_special_type.
