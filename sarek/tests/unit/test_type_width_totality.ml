(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * CLASS VALIDATOR for the "wrong-width / silently-wrong-type" family.
 *
 * Every member of that family so far (kernel-local tuple, vector-of-tuple,
 * helper return type, variant payload, `char` stride) has the same shape: a
 * SOURCE type is mapped to a DEVICE type by a function that has a placeholder
 * or wildcard arm collapsing the unknown case to `int` / [Ir.TInt32]. The
 * result compiles, runs, and produces wrong bytes with no diagnostic.
 *
 * The invariant this test enforces is the one those arms break:
 *
 *   for every scalar source type T,
 *     either [elttype_of_typ T] raises a located PPX error,
 *     or     byte_width (elttype_of_typ T) = host_byte_width T.
 *
 * A silent width change is never admissible; a REJECTION always is. That is
 * what makes this a validator for the class rather than three point checks —
 * a new element type added to [Sarek_types.registered_type] must either be
 * mapped at the right width or explicitly rejected, and it cannot reach the
 * backends through a wildcard.
 *
 * WHY THIS GATE CANNOT SILENTLY PASS (see the "gates that cannot fail" list):
 *
 *  - the source enumeration is not a hand-written list that can go stale: it
 *    is generated from a total successor chain ([next_reg] / [next_prim])
 *    whose match has no wildcard, so adding a constructor to
 *    [Sarek_types.registered_type] or [prim_type] fails to compile here;
 *  - the IR width function [ir_width] is likewise a total match on
 *    [Sarek_ir_ppx.elttype] with no wildcard, so a new IR element type fails
 *    to compile here too;
 *  - the host width table [host_width] is a total match on the source type;
 *  - the enumeration is asserted non-empty and of the expected size, so a
 *    chain accidentally cut short cannot make the sweep vacuous.
 ******************************************************************************)

module T = Sarek_types
module Ir = Sarek_ir_ppx

(* ------------------------------------------------------------------ *)
(* Compiler-enforced enumeration of the source scalar types.           *)
(*                                                                     *)
(* [next_reg] is a total match: it has one arm per constructor and NO   *)
(* wildcard, so a new [registered_type] constructor is a compile error  *)
(* here (warning 8 is an error in this test's flags). The list is then  *)
(* UNFOLDED from the chain, so it cannot drift from the type.          *)
(* ------------------------------------------------------------------ *)

let next_reg : T.registered_type -> T.registered_type option = function
  | T.Int -> Some T.Int64
  | T.Int64 -> Some T.Float16
  | T.Float16 -> Some T.Float32
  | T.Float32 -> Some T.Float64
  | T.Float64 -> Some T.Char
  | T.Char -> Some (T.Custom "some_registered_type")
  | T.Custom _ -> None

let next_prim : T.prim_type -> T.prim_type option = function
  | T.TInt32 -> Some T.TBool
  | T.TBool -> Some T.TUnit
  | T.TUnit -> None

let unfold next first =
  let rec go acc x =
    match next x with Some y -> go (y :: acc) y | None -> List.rev (x :: acc)
  in
  go [] first

let all_reg = unfold next_reg T.Int

let all_prim = unfold next_prim T.TInt32

let all_scalar_typs : T.typ list =
  List.map (fun r -> T.TReg r) all_reg @ List.map (fun p -> T.TPrim p) all_prim

(* ------------------------------------------------------------------ *)
(* Widths                                                              *)
(* ------------------------------------------------------------------ *)

(** Byte width the HOST uses for a value of this source type. This is the width
    the runtime actually moves: it is [Spoc_core.Vector.elem_size] of the
    corresponding vector kind, restated here on the source type so the test does
    not need a device or a vector.

    [Custom] is a user-registered aggregate whose width is only known from its
    registered layout, so it has no scalar width — a mapper is required to
    reject it, not to guess. *)
let host_width : T.typ -> int option = function
  | T.TReg T.Int -> Some 4 (* OCaml int, carried in a 32-bit slot *)
  | T.TReg T.Int64 -> Some 8
  | T.TReg T.Float16 -> Some 2 (* Vector.float16: IEEE binary16 *)
  | T.TReg T.Float32 -> Some 4
  | T.TReg T.Float64 -> Some 8
  | T.TReg T.Char -> Some 1 (* Vector.char: Bigarray char, ONE byte *)
  | T.TReg (T.Custom _) -> None
  | T.TPrim T.TInt32 -> Some 4
  | T.TPrim T.TBool -> Some 4
  | T.TPrim T.TUnit -> Some 4
  | T.TVar _ | T.TVec _ | T.TArr _ | T.TFun _ | T.TRecord _ | T.TVariant _
  | T.TTuple _ ->
      None

(** Byte width of an IR element type. Total match, no wildcard: a new IR element
    type is a compile error here. *)
let ir_width : Ir.elttype -> int option = function
  | Ir.TInt32 -> Some 4
  | Ir.TInt64 -> Some 8
  | Ir.TFloat16 -> Some 2
  | Ir.TFloat32 -> Some 4
  | Ir.TFloat64 -> Some 8
  | Ir.TBool -> Some 4
  | Ir.TUnit -> Some 4
  | Ir.TRecord _ | Ir.TVariant _ | Ir.TArray _ | Ir.TVec _ -> None

let string_of_typ t = Format.asprintf "%a" T.pp_typ t

(* ------------------------------------------------------------------ *)
(* The sweep                                                           *)
(* ------------------------------------------------------------------ *)

(** The enumeration must not be vacuous or truncated. If the successor chain is
    ever cut short, the sweep below would pass by testing nothing — this is the
    check that makes that impossible. *)
let test_enumeration_is_complete () =
  Alcotest.(check int) "registered_type constructors" 7 (List.length all_reg) ;
  Alcotest.(check int) "prim_type constructors" 3 (List.length all_prim) ;
  Alcotest.(check int)
    "scalar source types swept"
    10
    (List.length all_scalar_typs)

(** THE INVARIANT. For every scalar source type, [elttype_of_typ] must either
    reject it or preserve its byte width. *)
let test_elttype_of_typ_preserves_width () =
  List.iter
    (fun t ->
      let label = string_of_typ t in
      match Sarek_lower_ir.elttype_of_typ t with
      | exception Ppxlib.Location.Error _ ->
          (* Explicit rejection is always admissible: it is loud. *)
          ()
      | ir -> (
          match (host_width t, ir_width ir) with
          | Some hw, Some iw ->
              Alcotest.(check int)
                (Printf.sprintf
                   "%s: device element width must equal the host's %d byte(s)"
                   label
                   hw)
                hw
                iw
          | None, _ ->
              (* No scalar host width (Custom aggregate): the mapper must not
                 have produced a scalar element type silently. *)
              Alcotest.failf
                "%s has no known scalar host width, so mapping it to a scalar \
                 device element type is a guess: it must be rejected with a \
                 diagnostic instead"
                label
          | Some hw, None ->
              Alcotest.failf
                "%s is a %d-byte scalar on the host but lowered to a \
                 non-scalar device element type"
                label
                hw))
    all_scalar_typs

(** The same invariant for the data-slot mapper (vector elements and
    kernel-local bindings). It delegates to [elttype_of_typ] for non-tuples
    today; this pins that it cannot start diverging on widths. *)
let test_slot_elttype_of_typ_preserves_width () =
  List.iter
    (fun t ->
      let label = string_of_typ t in
      match Sarek_lower_ir.slot_elttype_of_typ t with
      | exception Ppxlib.Location.Error _ -> ()
      | ir -> (
          match (host_width t, ir_width ir) with
          | Some hw, Some iw ->
              Alcotest.(check int)
                (Printf.sprintf "%s (data slot): width" label)
                hw
                iw
          | None, _ ->
              Alcotest.failf
                "%s (data slot) has no known scalar host width but was mapped \
                 to a scalar device element type"
                label
          | Some hw, None ->
              Alcotest.failf
                "%s (data slot) is a %d-byte scalar but lowered to a \
                 non-scalar element type"
                label
                hw))
    all_scalar_typs

(** A vector of a scalar must carry that scalar's width through to the element
    type recorded on the parameter — this is the exact path a `char vector` took
    to reach the backends as `int*`. *)
let test_vector_element_width () =
  List.iter
    (fun t ->
      let label = string_of_typ t ^ " vector" in
      match Sarek_lower_ir.elttype_of_typ (T.TVec t) with
      | exception Ppxlib.Location.Error _ -> ()
      | Ir.TVec elem -> (
          match (host_width t, ir_width elem) with
          | Some hw, Some iw ->
              Alcotest.(check int)
                (Printf.sprintf
                   "%s: buffer stride must be the host's %d byte(s)"
                   label
                   hw)
                hw
                iw
          | None, _ ->
              Alcotest.failf
                "%s: element type has no known scalar host width but was \
                 mapped to a scalar device element type"
                label
          | Some hw, None ->
              Alcotest.failf
                "%s: %d-byte host element lowered to a non-scalar element type"
                label
                hw)
      | other ->
          Alcotest.failf
            "%s did not lower to a vector element type (%s)"
            label
            (match other with
            | Ir.TInt32 -> "TInt32"
            | Ir.TInt64 -> "TInt64"
            | Ir.TFloat16 -> "TFloat16"
            | Ir.TFloat32 -> "TFloat32"
            | Ir.TFloat64 -> "TFloat64"
            | Ir.TBool -> "TBool"
            | Ir.TUnit -> "TUnit"
            | Ir.TRecord _ -> "TRecord"
            | Ir.TVariant _ -> "TVariant"
            | Ir.TArray _ -> "TArray"
            | Ir.TVec _ -> "TVec"))
    all_scalar_typs

(** The C type-name mapper feeding generated struct/builder source is held to
    the same standard: no arm may silently answer "int" for a type it does not
    know. A type it cannot name must raise.

    [Sarek_ctype_gen] is the surviving emitter of that kind (the
    [Sarek_lower_ir] copy was write-only and has been removed). *)
let test_c_type_of_typ_has_no_silent_int () =
  List.iter
    (fun t ->
      let label = string_of_typ t in
      match Sarek_ctype_gen.c_type_of_typ t with
      | exception Ppxlib.Location.Error _ -> ()
      | c ->
          (* "int" is only correct for the 4-byte integer-ish source types. Any
             other type answering "int" is the wildcard defect. *)
          if String.equal c "int" then
            Alcotest.(check bool)
              (Printf.sprintf
                 "%s emitted C type \"int\": only 4-byte integer source types \
                  may do so"
                 label)
              true
              (match t with
              | T.TReg T.Int | T.TPrim T.TInt32 | T.TPrim T.TBool -> true
              | _ -> false))
    all_scalar_typs

let () =
  Alcotest.run
    "type_width_totality"
    [
      ( "wrong-width class validator",
        [
          Alcotest.test_case
            "enumeration is complete (gate is not vacuous)"
            `Quick
            test_enumeration_is_complete;
          Alcotest.test_case
            "elttype_of_typ preserves byte width or rejects"
            `Quick
            test_elttype_of_typ_preserves_width;
          Alcotest.test_case
            "slot_elttype_of_typ preserves byte width or rejects"
            `Quick
            test_slot_elttype_of_typ_preserves_width;
          Alcotest.test_case
            "vector element stride matches the host element width"
            `Quick
            test_vector_element_width;
          Alcotest.test_case
            "c_type_of_typ has no silent int wildcard"
            `Quick
            test_c_type_of_typ_has_no_silent_int;
        ] );
    ]
