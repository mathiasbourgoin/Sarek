(******************************************************************************)
(* VecExtraction.v
 *
 * Rocq -> OCaml extraction configuration for VecSpec.
 * Extracts the memory-access type inference model to VecModel.ml; committed
 * to repo so test/test_type_safety_conformance.ml can use it for T2-VEC tests.
 *
 * Run to refresh after spec changes:
 *   cd formal/type-safety
 *   eval $(opam env --switch=~/dev/SPOC)
 *   rocq compile -R theories TypeSafety \
 *         -output-directory extraction \
 *         extraction/VecExtraction.v
 ******************************************************************************)

From Stdlib Require Import Extraction ExtrOcamlBasic ExtrOcamlNatInt.
From TypeSafety Require Import TypeSafetySpec.
From TypeSafety Require Import VecSpec.

(* Extract infer_mem_type (the algorithmic type checker for mem_expr) and
 * sarek_type_eq_dec (decidable equality used in EVecSet/EArrSet checks). *)
Extraction "VecModel.ml"
  infer_mem_type
  sarek_type_eq_dec.
