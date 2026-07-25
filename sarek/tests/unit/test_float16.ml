(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for the f16 DSL element type (#57 slice 1).
 *
 * Three groups:
 *   1. Host storage — an f16 vector really is binary16-backed.
 *   2. Rounding      — Sarek_float16.to_float16 is the shared narrowing used by
 *                      the interpreter, the native path and the Bigarray store,
 *                      so it must agree with the store exactly.
 *   3. Type system   — `float16` resolves as an annotation AND is excluded from
 *                      the numeric/float predicates, which is what forces
 *                      "compute in f32" at the type level.
 ******************************************************************************)

module Vector = Spoc_core.Vector
module F16 = Sarek_interp.Sarek_float16

(* ------------------------------------------------------------------ *)
(* 1. Host storage                                                    *)
(* ------------------------------------------------------------------ *)

let test_host_roundtrip () =
  let v = Vector.create Vector.float16 4 in
  Vector.set v 0 3.14159 ;
  (* 3.14159 is not representable in binary16; the nearest value is 3.140625.
     Reading back 3.14159 would prove the vector is NOT f16-backed. *)
  Alcotest.(check (float 1e-9))
    "3.14159 stores as binary16 3.140625"
    3.140625
    (Vector.get v 0) ;
  Vector.set v 1 1.0 ;
  Vector.set v 2 0.5 ;
  Vector.set v 3 (-2.5) ;
  (* Exactly representable values must survive untouched. *)
  Alcotest.(check (float 0.)) "1.0 exact" 1.0 (Vector.get v 1) ;
  Alcotest.(check (float 0.)) "0.5 exact" 0.5 (Vector.get v 2) ;
  Alcotest.(check (float 0.)) "-2.5 exact" (-2.5) (Vector.get v 3)

let test_host_elem_size () =
  let v = Vector.create Vector.float16 1 in
  Alcotest.(check int)
    "f16 element is 2 bytes"
    2
    (Vector.elem_size (Vector.kind v)) ;
  (* Contrast with f32, so this is a discrimination test and not a tautology. *)
  let f32 = Vector.create Vector.float32 1 in
  Alcotest.(check int)
    "f32 element is 4 bytes"
    4
    (Vector.elem_size (Vector.kind f32))

let test_host_range_edges () =
  let v = Vector.create Vector.float16 4 in
  (* 65504 is the largest finite binary16; above it, binary16 saturates to
     infinity rather than wrapping. *)
  Vector.set v 0 65504.0 ;
  Alcotest.(check (float 0.)) "max finite binary16" 65504.0 (Vector.get v 0) ;
  Vector.set v 1 70000.0 ;
  Alcotest.(check bool) "overflow -> +inf" true (Vector.get v 1 = infinity) ;
  Vector.set v 2 (-70000.0) ;
  Alcotest.(check bool) "overflow -> -inf" true (Vector.get v 2 = neg_infinity) ;
  (* Below the smallest subnormal, binary16 flushes to zero. *)
  Vector.set v 3 1e-10 ;
  Alcotest.(check (float 0.)) "underflow -> 0" 0.0 (Vector.get v 3)

(* ------------------------------------------------------------------ *)
(* 2. Rounding helper agrees with the storage path                    *)
(* ------------------------------------------------------------------ *)

let test_round_matches_store () =
  (* This is the load-bearing invariant of the whole slice: the narrowing the
     interpreter and native paths apply at an ECast MUST be the same narrowing
     the Bigarray.Float16 store applies. If these ever diverge, the interpreter
     stops being a faithful oracle for GPU f16 kernels. *)
  let v = Vector.create Vector.float16 1 in
  let samples =
    [
      3.14159;
      0.1;
      1.0 /. 3.0;
      -0.7;
      1e-5;
      6.0e-8;
      65504.0;
      70000.0;
      -70000.0;
      0.0;
      -0.0;
      1e-10;
      2.7182818284;
      1023.5;
      1024.5;
    ]
  in
  List.iter
    (fun x ->
      Vector.set v 0 x ;
      let stored = Vector.get v 0 in
      let rounded = F16.to_float16 x in
      if not (stored = rounded || (stored <> stored && rounded <> rounded)) then
        Alcotest.failf
          "narrowing disagrees for %.17g: store gave %.17g, to_float16 gave \
           %.17g"
          x
          stored
          rounded)
    samples

let test_round_is_idempotent () =
  (* Rounding an already-binary16 value must be a no-op — otherwise repeated
     store/load cycles would drift. *)
  List.iter
    (fun x ->
      let once = F16.to_float16 x in
      let twice = F16.to_float16 once in
      if not (once = twice || (once <> once && twice <> twice)) then
        Alcotest.failf
          "to_float16 not idempotent at %.17g: %.17g then %.17g"
          x
          once
          twice)
    [3.14159; 0.1; -0.7; 1e-5; 65504.0; 1024.5]

let test_round_is_lossy_where_expected () =
  (* Guard against a to_float16 that silently became the identity function. *)
  Alcotest.(check bool)
    "3.14159 is changed by narrowing"
    true
    (F16.to_float16 3.14159 <> 3.14159) ;
  Alcotest.(check (float 1e-9))
    "and lands on the binary16 neighbour"
    3.140625
    (F16.to_float16 3.14159) ;
  Alcotest.(check (float 0.))
    "an exact value is untouched"
    0.5
    (F16.to_float16 0.5)

(* ------------------------------------------------------------------ *)
(* 2b. Interpreter ECast narrowing                                     *)
(* ------------------------------------------------------------------ *)

(* This is asserted DIRECTLY on eval_expr rather than through a kernel, and
   deliberately so. End-to-end, an [ECast (TFloat16, _)] whose result is stored
   straight into an f16 vector is indistinguishable from no cast at all, because
   the Bigarray.Float16 store narrows anyway. Amplifying the difference with
   catastrophic cancellation is not a usable alternative either: it would equally
   amplify the interpreter's f64-vs-GPU-f32 intermediate difference and produce
   spurious divergence unrelated to f16.

   So the arm is pinned here, at the one place where it is observable in
   isolation: an f16-typed IR value must BE a binary16 value the moment the cast
   is evaluated, not merely by the time it is stored. *)

module Interp = Sarek.Sarek_ir_interp

let interp_env () =
  {
    Interp.vars = Hashtbl.create 4;
    vars_by_name = Hashtbl.create 4;
    arrays = Hashtbl.create 4;
    shared = Hashtbl.create 4;
    funcs = Hashtbl.create 4;
  }

let interp_state () =
  {
    Interp.thread_idx = (0, 0, 0);
    block_idx = (0, 0, 0);
    block_dim = (1, 1, 1);
    grid_dim = (1, 1, 1);
  }

let eval_f16_cast x =
  let e =
    Sarek_ir_types.ECast
      ( Sarek_ir_types.TFloat16,
        Sarek_ir_types.EConst (Sarek_ir_types.CFloat32 x) )
  in
  match Interp.eval_expr (interp_state ()) (interp_env ()) e with
  | Interp.VFloat32 f -> f
  | _ -> Alcotest.fail "ECast (TFloat16, _) did not evaluate to a float value"

let test_interp_cast_narrows () =
  Alcotest.(check (float 1e-9))
    "ECast to f16 narrows 3.14159 to 3.140625"
    3.140625
    (eval_f16_cast 3.14159) ;
  (* Agreement with the shared narrowing helper, for several magnitudes. *)
  List.iter
    (fun x ->
      let got = eval_f16_cast x in
      let want = F16.to_float16 x in
      if not (got = want || (got <> got && want <> want)) then
        Alcotest.failf
          "ECast (TFloat16) at %.17g gave %.17g, expected %.17g"
          x
          got
          want)
    [0.1; 1.0 /. 3.0; -0.7; 1e-5; 1024.5; 70000.0] ;
  (* And it must NOT be the identity. *)
  Alcotest.(check bool)
    "ECast to f16 is not the identity"
    true
    (eval_f16_cast 3.14159 <> 3.14159)

let test_interp_cast_f32_does_not_narrow () =
  (* Discrimination: an f32 cast must leave the value alone, so the arm above is
     specific to TFloat16 and not a blanket rounding of every cast. *)
  let e =
    Sarek_ir_types.ECast
      ( Sarek_ir_types.TFloat32,
        Sarek_ir_types.EConst (Sarek_ir_types.CFloat32 3.14159) )
  in
  match Interp.eval_expr (interp_state ()) (interp_env ()) e with
  | Interp.VFloat32 f ->
      Alcotest.(check (float 0.)) "f32 cast is value-preserving" 3.14159 f
  | _ -> Alcotest.fail "ECast (TFloat32, _) did not evaluate to a float value"

(* ------------------------------------------------------------------ *)
(* 3. Type system                                                     *)
(* ------------------------------------------------------------------ *)

open Sarek_types

let test_annotation_resolves () =
  (* `float16` as written in a [%kernel] parameter annotation. *)
  let t = type_of_type_expr (Sarek_ast.TEConstr ("float16", [])) in
  Alcotest.(check bool)
    "float16 resolves to TReg Float16"
    true
    (match repr t with TReg Float16 -> true | _ -> false) ;
  (* `float16 vector`, the surface slice 1 actually delivers. *)
  let tv =
    type_of_type_expr
      (Sarek_ast.TEConstr ("vector", [Sarek_ast.TEConstr ("float16", [])]))
  in
  Alcotest.(check bool)
    "float16 vector resolves to TVec (TReg Float16)"
    true
    (match repr tv with
    | TVec e -> ( match repr e with TReg Float16 -> true | _ -> false)
    | _ -> false)

let test_half_is_not_an_alias () =
  (* Deliberate decision: only `float16` is accepted. `half` stays reserved so
     it can be added later without breaking anything, but it must NOT silently
     resolve to something else today. *)
  let t = type_of_type_expr (Sarek_ast.TEConstr ("half", [])) in
  Alcotest.(check bool)
    "half does not resolve to float16"
    false
    (match repr t with TReg Float16 -> true | _ -> false)

let test_f16_is_not_numeric () =
  (* The enforcement mechanism for "storage type, compute in f32": because f16
     is outside these predicates, f16 values cannot be added or fed to math
     intrinsics — a conversion is mandatory. *)
  Alcotest.(check bool) "f16 is not numeric" false (is_numeric t_float16) ;
  Alcotest.(check bool) "f16 is not float" false (is_float t_float16) ;
  Alcotest.(check bool) "f16 is not integer" false (is_integer t_float16) ;
  (* Sanity: the predicates still hold for the widths that DO compute. *)
  Alcotest.(check bool) "f32 is numeric" true (is_numeric t_float32) ;
  Alcotest.(check bool) "f64 is numeric" true (is_numeric t_float64)

let test_bare_float_literal_cannot_be_f16 () =
  (* A bare float literal defaults into the f32/f64 lattice only. Allowing it to
     link to f16 would reintroduce implicit narrowing through the back door. *)
  Alcotest.(check bool)
    "float literal cannot link to f16"
    false
    (float_literal_can_link t_float16) ;
  Alcotest.(check bool)
    "float literal can link to f32"
    true
    (float_literal_can_link t_float32)

let test_f16_unifies_only_with_itself () =
  let ok = function Ok () -> true | Error _ -> false in
  Alcotest.(check bool) "f16 ~ f16" true (ok (unify t_float16 (TReg Float16))) ;
  Alcotest.(check bool)
    "f16 does not unify with f32"
    false
    (ok (unify t_float16 t_float32)) ;
  Alcotest.(check bool)
    "f16 does not unify with f64"
    false
    (ok (unify t_float16 t_float64))

let test_pretty_printer () =
  Alcotest.(check string)
    "float16 prints as float16"
    "float16"
    (Format.asprintf "%a" pp_registered Float16)

let () =
  Alcotest.run
    "float16"
    [
      ( "host_storage",
        [
          Alcotest.test_case "binary16 round-trip" `Quick test_host_roundtrip;
          Alcotest.test_case "element size" `Quick test_host_elem_size;
          Alcotest.test_case
            "overflow/underflow edges"
            `Quick
            test_host_range_edges;
        ] );
      ( "rounding",
        [
          Alcotest.test_case
            "to_float16 agrees with the store"
            `Quick
            test_round_matches_store;
          Alcotest.test_case "idempotent" `Quick test_round_is_idempotent;
          Alcotest.test_case
            "lossy where expected"
            `Quick
            test_round_is_lossy_where_expected;
        ] );
      ( "interpreter_cast",
        [
          Alcotest.test_case
            "ECast to f16 narrows immediately"
            `Quick
            test_interp_cast_narrows;
          Alcotest.test_case
            "ECast to f32 does not narrow"
            `Quick
            test_interp_cast_f32_does_not_narrow;
        ] );
      ( "type_system",
        [
          Alcotest.test_case
            "float16 annotation resolves"
            `Quick
            test_annotation_resolves;
          Alcotest.test_case
            "half is not an alias"
            `Quick
            test_half_is_not_an_alias;
          Alcotest.test_case
            "f16 is excluded from numeric predicates"
            `Quick
            test_f16_is_not_numeric;
          Alcotest.test_case
            "bare float literal cannot be f16"
            `Quick
            test_bare_float_literal_cannot_be_f16;
          Alcotest.test_case
            "f16 unifies only with itself"
            `Quick
            test_f16_unifies_only_with_itself;
          Alcotest.test_case "pretty printer" `Quick test_pretty_printer;
        ] );
    ]
