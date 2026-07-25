(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * [Sarek_ctype_gen] turns a registered record/variant declaration into the C
 * struct / union / builder source the backends splice into generated kernels.
 *
 * Two members of the "wrong-width / silently-wrong-type" family live here:
 *
 *  1. VARIANT PAYLOAD TYPE COMPUTED AND THEN DISCARDED.
 *     [constructor_strings_of_core_type_decl] evaluated the payload's Sarek
 *     type and dropped it on the floor:
 *
 *         | Pcstr_tuple [ct] ->
 *             let _ = typ_of_core_type ~loc ct in     (* computed... *)
 *             (cd.pcd_name.txt, None)                 (* ...and discarded *)
 *
 *     [None] then means "no payload" downstream, so [Shade of float32] emitted
 *     a union member declared `int Shade_t;` and a builder taking no argument
 *     at all. A 4-byte integer slot holding an f32's bits, and no way to set
 *     it — wrong type, wrong builder, no diagnostic.
 *
 *  2. SILENT `int` WILDCARD in [c_type_of_typ] (`| _ -> "int"`), which answers
 *     "int" for every type it does not enumerate — including 2-byte float16
 *     and 8-byte int64-carrying registered types.
 *
 * These tests pin both against the emitted C source.
 ******************************************************************************)

open Ppxlib

let loc = Location.none

let contains hay needle =
  let nh = String.length needle and h = String.length hay in
  let rec go i =
    if i + nh > h then false
    else if String.sub hay i nh = needle then true
    else go (i + 1)
  in
  nh = 0 || go 0

(* [type col = Red | Shade of float32] as a ppxlib type declaration. *)
let variant_decl =
  let ctor name args =
    {
      pcd_name = {txt = name; loc};
      pcd_vars = [];
      pcd_args = Pcstr_tuple args;
      pcd_res = None;
      pcd_loc = loc;
      pcd_attributes = [];
    }
  in
  let f32 = [%type: float32] in
  {
    ptype_name = {txt = "col"; loc};
    ptype_params = [];
    ptype_cstrs = [];
    ptype_kind = Ptype_variant [ctor "Red" []; ctor "Shade" [f32]];
    ptype_private = Public;
    ptype_manifest = None;
    ptype_attributes = [];
    ptype_loc = loc;
  }

(* [constructor_strings_of_core_type_decl] returns an OCaml list expression of
   string literals; collect them back into the C source they denote. *)
let emitted_variant_source () =
  let strings = ref [] in
  let it =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match e.pexp_desc with
        | Pexp_constant (Pconst_string (s, _, _)) -> strings := s :: !strings
        | _ -> ()) ;
        super#expression e
    end
  in
  it#expression
    (Sarek_ctype_gen.constructor_strings_of_core_type_decl ~loc variant_decl) ;
  let src = String.concat "\n" (List.rev !strings) in
  if String.equal src "" then
    Alcotest.fail "no C source strings were emitted — the test would be vacuous" ;
  src

(** The payload's type must reach the emitted C. A [float32] payload must be
    declared `float`, never `int`. *)
let test_variant_payload_keeps_its_type () =
  let src = emitted_variant_source () in
  Alcotest.(check bool)
    ("the float32 payload member must be declared `float`, not `int`.\n\
      Emitted source was:\n" ^ src)
    true
    (contains src "float col_sarek_Shade_t;") ;
  Alcotest.(check bool)
    ("the payload member must NOT be declared `int` (the discarded-type defect).\n\
      Emitted source was:\n" ^ src)
    false
    (contains src "int col_sarek_Shade_t;")

(** The builder for a constructor WITH a payload must take that payload as a
    parameter and assign it. A dropped payload type also drops the parameter, so
    the generated `build_col_Shade()` could not set the union member at all. *)
let test_variant_builder_takes_the_payload () =
  let src = emitted_variant_source () in
  Alcotest.(check bool)
    ("build_col_Shade must take the float payload.\nEmitted source was:\n" ^ src)
    true
    (contains src "build_col_Shade(float v)") ;
  Alcotest.(check bool)
    ("build_col_Shade must not be a no-payload builder.\nEmitted source was:\n"
   ^ src)
    false
    (contains src "build_col_Shade()")

(** The nullary constructor keeps its placeholder integer member — this is the
    control showing the assertions above are about the payload, not about the
    shape of the emitted struct. *)
let test_nullary_constructor_unchanged () =
  let src = emitted_variant_source () in
  Alcotest.(check bool)
    "nullary constructor keeps its placeholder member"
    true
    (contains src "int col_sarek_Red_t;") ;
  Alcotest.(check bool)
    "nullary builder takes no argument"
    true
    (contains src "build_col_Red()")

(** No silent `int` for a type the mapper does not enumerate. float16 is 2
    bytes; answering "int" declares a 4-byte member for it. *)
let test_c_type_has_no_silent_int_wildcard () =
  let check_not_int label t =
    match Sarek_ctype_gen.c_type_of_typ t with
    | exception Location.Error _ -> () (* explicit rejection is fine *)
    | "int" ->
        Alcotest.failf
          "%s was mapped to C `int` by the wildcard arm — it is not a 4-byte \
           integer type"
          label
    | _ -> ()
  in
  check_not_int "float16" (Sarek_types.TReg Sarek_types.Float16) ;
  check_not_int "char" (Sarek_types.TReg Sarek_types.Char) ;
  check_not_int
    "a registered custom type"
    (Sarek_types.TReg (Sarek_types.Custom "df64"))

let () =
  Alcotest.run
    "ctype_gen"
    [
      ( "variant payload types reach the emitted C",
        [
          Alcotest.test_case
            "payload member keeps its type"
            `Quick
            test_variant_payload_keeps_its_type;
          Alcotest.test_case
            "builder takes the payload"
            `Quick
            test_variant_builder_takes_the_payload;
          Alcotest.test_case
            "nullary constructor unchanged (control)"
            `Quick
            test_nullary_constructor_unchanged;
          Alcotest.test_case
            "c_type_of_typ has no silent int wildcard"
            `Quick
            test_c_type_has_no_silent_int_wildcard;
        ] );
    ]
