(******************************************************************************)
(* ConstrExtraction.v
 *
 * Rocq -> OCaml extraction for ConstrSpec.
 * Extracts the algebraic-construction (ERecord/EConstr) type inference model
 * to ConstrModel.ml; committed to repo so
 * test/test_type_safety_conformance.ml can use it for T3-S6 smoke tests.
 *
 * Run to refresh after spec changes:
 *   cd formal/type-safety
 *   eval $(opam env --switch=~/dev/SPOC)
 *   rocq compile -R theories TypeSafety \
 *         -output-directory extraction \
 *         extraction/ConstrExtraction.v
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

Set Extraction Output Directory "extraction".

Extraction "ConstrModel.ml"
  infer_constr_type.
