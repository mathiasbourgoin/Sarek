(******************************************************************************)
(* GPUExtraction.v
 *
 * Rocq -> OCaml extraction for GPUSpec.
 * Extracts the GPU / BSP-form (ELetShared/ESuperstep) type inference model to
 * GPUModel.ml; committed to repo so test/test_type_safety_conformance.ml can
 * use it for T3-S8 smoke tests.
 *
 * Run to refresh after spec changes:
 *   cd formal/type-safety
 *   eval $(opam env --switch=<your-opam-switch>)   (* e.g. the repo root *)
 *   rocq compile -R theories TypeSafety \
 *         -output-directory extraction \
 *         extraction/GPUExtraction.v
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
From TypeSafety Require Import GPUSpec.

Set Extraction Output Directory "extraction".

Extraction "GPUModel.ml"
  infer_gpu_type.
