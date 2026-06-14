(******************************************************************************)
(* RegistryExtraction.v
 *
 * Rocq -> OCaml extraction configuration for RegistrySpec.
 * Extracts the record field-access type inference model to RegistryModel.ml;
 * committed to repo so test/test_type_safety_conformance.ml can use it for
 * T2-REGISTRY tests.
 *
 * Run to refresh after spec changes:
 *   cd formal/type-safety
 *   eval $(opam env --switch=~/dev/SPOC)
 *   rocq compile -R theories TypeSafety \
 *         -output-directory extraction \
 *         extraction/RegistryExtraction.v
 ******************************************************************************)

From Stdlib Require Import Extraction ExtrOcamlBasic ExtrOcamlNatInt.
From TypeSafety Require Import TypeSafetySpec.
From TypeSafety Require Import VecSpec.
From TypeSafety Require Import RegistrySpec.

(* Extract infer_rec_type (the algorithmic type checker for rec_expr),
 * field_lookup (field name lookup in a record type's field list), and
 * sarek_type_eq_dec (decidable equality, also extracted from VecSpec). *)
Extraction "RegistryModel.ml"
  infer_rec_type
  field_lookup
  sarek_type_eq_dec.
