(******************************************************************************)
(* FunExtraction.v
 *
 * Rocq -> OCaml extraction for FunSpec.
 * Extracts the single-parameter function type inference model to FunModel.ml;
 * committed to repo so test/test_type_safety_conformance.ml can use it for
 * T3-S3 smoke tests.
 *
 * Run to refresh after spec changes:
 *   cd formal/type-safety
 *   eval $(opam env --switch=~/dev/SPOC)
 *   rocq compile -R theories TypeSafety \
 *         -output-directory extraction \
 *         extraction/FunExtraction.v
 ******************************************************************************)

From Stdlib Require Import Extraction ExtrOcamlBasic ExtrOcamlNatInt.
From TypeSafety Require Import TypeSafetySpec.
From TypeSafety Require Import VecSpec.
From TypeSafety Require Import RegistrySpec.
From TypeSafety Require Import ControlFlowSpec.
From TypeSafety Require Import OperatorSpec.
From TypeSafety Require Import FunSpec.

Extraction "FunModel.ml"
  infer_fun_type.
