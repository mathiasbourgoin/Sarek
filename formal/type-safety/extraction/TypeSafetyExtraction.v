(******************************************************************************)
(* TypeSafetyExtraction.v
 *
 * Rocq → OCaml extraction configuration for TypeSafetySpec.
 * Extracts the abstract type-checker model to TypeSafetyModel.ml; committed
 * to repo so test/test_type_safety_conformance.ml can use it as the oracle.
 *
 * Run to refresh after spec changes:
 *   cd formal/type-safety
 *   eval $(opam env --switch=~/dev/SPOC)
 *   coqc -R theories TypeSafety \
 *         -output-directory extraction \
 *         extraction/TypeSafetyExtraction.v
 ******************************************************************************)

From Stdlib Require Import Extraction ExtrOcamlBasic ExtrOcamlNatInt.
From TypeSafety Require Import TypeSafetySpec.

(* Let prim_type / reg_type / mem_space / sarek_type / lit / expr / type_error
 * extract as regular OCaml variants — do NOT use Extract Inductive here so
 * the type definitions are emitted in the generated .ml.
 *
 * infer_type and lookup_env are the two algorithmic functions under test;
 * they are the oracle against which Sarek_typer.infer is compared in the
 * differential conformance harness. *)

Extraction "TypeSafetyModel.ml"
  infer_type
  lookup_env.
