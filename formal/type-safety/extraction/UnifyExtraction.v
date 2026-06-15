(******************************************************************************)
(* UnifyExtraction.v
 *
 * Rocq -> OCaml extraction configuration for UnifySpec.
 * Extracts the pure unifier model to UnifyModel.ml; committed to repo so
 * test/test_type_safety_conformance.ml can use it as the oracle for T2-UNIFY.
 *
 * Run to refresh after spec changes:
 *   cd formal/type-safety
 *   eval $(opam env --switch=~/dev/SPOC)
 *   coqc -R theories TypeSafety \
 *         -output-directory extraction \
 *         extraction/UnifyExtraction.v
 ******************************************************************************)

From Stdlib Require Import Extraction ExtrOcamlBasic ExtrOcamlNatInt.
From TypeSafety Require Import TypeSafetySpec.
From TypeSafety Require Import UnifySpec.

(* Extract the core unification functions.
 * unify_fun: pure unifier (returns option pre_subst).
 * apply_subst: apply substitution to a pre_type.
 * follow_pvar: follow PVar chains through a substitution. *)

Extraction "UnifyModel.ml"
  unify_fun
  apply_subst
  follow_pvar.
