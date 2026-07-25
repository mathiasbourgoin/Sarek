(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * C-type / struct-string generation for [%%sarek.type] top-level type
 * registrations. Maps Sarek types (and ppxlib core_types) to the C struct,
 * union and builder-function source strings.
 *
 * REACHABILITY, stated because it is not obvious: the only caller is
 * [Sarek_ppx.expand_sarek_type], and the [%%sarek.type] EXTENSION it
 * implements is NOT registered in the driver's rule list
 * ([Sarek_ppx.sarek_type_extension] is bound to [()], and the rules are
 * [sarek_type_rule; sarek_type_private_rule; kernel; kernel.real64;
 * sarek_include]). Writing [%%sarek.type type t = ...] therefore fails with
 * ppxlib's "Uninterpreted extension" — loudly, so no user code depends on it.
 * The live registration path is the [@@sarek.type] ATTRIBUTE, which goes
 * through [Sarek_ppx.register_sarek_type_decl] and the typed AST; the emitted
 * device definitions for records and variants come from
 * [Sarek_ir_codegen.gen_variant_def] and its record counterpart, not from
 * here. Note also that this module's naming convention (`build_X_sarek`,
 * `X_sarek_tag`, `union X_sarek_union`) has diverged from what those emitters
 * produce (`make_<mangled>_<ctor>`, `int tag`, `union {...} data`), so wiring
 * the extension up would need that reconciled first.
 *
 * It is kept and CORRECTED rather than deleted because it is the reference
 * implementation behind a documented (if presently unwired) entry point, and
 * because both of its defects — a variant payload type computed and then
 * discarded, and a wildcard mapping every unknown type to C `int` — are
 * members of the wrong-width family and would be re-introduced verbatim by
 * anyone reviving it. sarek/tests/unit/test_ctype_gen.ml pins both.
 ******************************************************************************)

open Ppxlib
open Sarek_types

let mangle_type_name name = String.map (function '.' -> '_' | c -> c) name

(** C type name for a Sarek type.

    TOTAL — deliberately no [| _ -> "int"] arm. That wildcard answered "int" (4
    bytes, integer) for every type it did not enumerate, including 2-byte
    [float16], 1-byte [char] and registered aggregates of arbitrary size, and it
    did so silently. A type this function cannot name must raise. *)
let rec c_type_of_typ ty : string =
  match repr ty with
  | TPrim TInt32 -> "int"
  | TPrim TBool -> "int"
  | TPrim TUnit -> "void"
  | TReg Int -> "int"
  | TReg Int64 -> "long"
  | TReg Float32 -> "float"
  | TReg Float64 -> "double"
  | TRecord (name, _) -> "struct " ^ mangle_type_name name ^ "_sarek"
  | TVariant (name, _) -> "struct " ^ mangle_type_name name ^ "_sarek"
  | TVec t -> c_type_of_typ t ^ " *"
  | TArr (t, _) -> c_type_of_typ t ^ " *"
  | TReg Float16 ->
      Location.raise_errorf
        ~loc:Location.none
        "float16 has no C type in a generated struct/builder: it is a 2-byte \
         storage-only element type and the previous wildcard declared it as a \
         4-byte `int`. Use float32 in aggregate fields."
  | TReg Char ->
      Location.raise_errorf
        ~loc:Location.none
        "`char` is not a supported Sarek element type: it is 1 byte on the \
         host and has no 1-byte device counterpart. Use int32."
  | TReg (Custom name) ->
      Location.raise_errorf
        ~loc:Location.none
        "Type %S is not a registered Sarek type, so its C representation is \
         unknown. Declare it with [@@sarek.type]."
        name
  | TVar _ ->
      Location.raise_errorf
        ~loc:Location.none
        "A type variable has no C representation; annotate with a concrete \
         type."
  | TFun _ ->
      Location.raise_errorf
        ~loc:Location.none
        "A function type has no C representation."
  | TTuple _ ->
      Location.raise_errorf
        ~loc:Location.none
        "A tuple type has no C representation; declare a record type with \
         [@@sarek.type] instead."

let record_constructor_strings name (fields : (string * typ * bool) list) =
  let name = mangle_type_name name in
  let struct_name = name ^ "_sarek" in
  let struct_fields =
    List.map
      (fun (fname, fty, _) -> "  " ^ c_type_of_typ fty ^ " " ^ fname ^ ";")
      fields
  in
  let struct_def =
    "struct " ^ struct_name ^ " {\n" ^ String.concat "\n" struct_fields ^ "\n};"
  in
  let params =
    String.concat
      ", "
      (List.map (fun (fname, fty, _) -> c_type_of_typ fty ^ " " ^ fname) fields)
  in
  let assigns =
    String.concat
      "\n"
      (List.map
         (fun (fname, _, _) -> "  res." ^ fname ^ " = " ^ fname ^ ";")
         fields)
  in
  let builder =
    "struct " ^ struct_name ^ " build_" ^ struct_name ^ "(" ^ params ^ ") {\n"
    ^ "  struct " ^ struct_name ^ " res;\n" ^ assigns ^ "\n  return res;\n}"
  in
  (* Emit struct definition first so OpenCL can see the type in builder
     signature. *)
  [struct_def; builder]

let variant_constructor_strings name constrs : string list =
  let name = mangle_type_name name in
  let struct_name = name ^ "_sarek" in
  let constr_structs =
    List.map
      (fun (cname, carg) ->
        let field =
          match carg with
          | None -> "  int " ^ name ^ "_sarek_" ^ cname ^ "_t;"
          | Some ty ->
              "  " ^ c_type_of_typ ty ^ " " ^ name ^ "_sarek_" ^ cname ^ "_t;"
        in
        "struct " ^ name ^ "_sarek_" ^ cname ^ " {\n" ^ field ^ "\n};")
      constrs
  in
  let union_fields =
    List.map
      (fun (cname, _carg) ->
        "  struct " ^ name ^ "_sarek_" ^ cname ^ " " ^ name ^ "_sarek_" ^ cname
        ^ ";")
      constrs
  in
  let union_def =
    "union " ^ name ^ "_sarek_union {\n"
    ^ String.concat "\n" union_fields
    ^ "\n};"
  in
  let main_struct =
    "struct " ^ struct_name ^ " {\n" ^ "  int " ^ name ^ "_sarek_tag;\n"
    ^ "  union " ^ name ^ "_sarek_union " ^ name ^ "_sarek_union;\n" ^ "};"
  in
  let builders =
    List.mapi
      (fun idx (cname, carg) ->
        let params, assign =
          match carg with
          | None -> ("", "  /* no payload */")
          | Some ty ->
              let pname = "v" in
              ( c_type_of_typ ty ^ " " ^ pname,
                "  res." ^ name ^ "_sarek_union." ^ name ^ "_sarek_" ^ cname
                ^ "." ^ name ^ "_sarek_" ^ cname ^ "_t = " ^ pname ^ ";" )
        in
        "struct " ^ struct_name ^ " build_" ^ name ^ "_" ^ cname ^ "(" ^ params
        ^ ") {\n" ^ "  struct " ^ struct_name ^ " res;\n" ^ "  res." ^ name
        ^ "_sarek_tag = " ^ string_of_int idx ^ ";\n" ^ assign ^ "\n"
        ^ "  return res;\n}")
      constrs
  in
  constr_structs @ (union_def :: main_struct :: builders)

let typ_of_core_type ~loc (ct : core_type) =
  match ct.ptyp_desc with
  | Ptyp_constr ({txt = Lident "float32"; _}, _) -> TReg Float32
  | Ptyp_constr ({txt = Lident "float64"; _}, _) -> TReg Float64
  | Ptyp_constr ({txt = Lident "float"; _}, _) -> TReg Float32
  | Ptyp_constr ({txt = Lident "int32"; _}, _) -> TPrim TInt32
  | Ptyp_constr ({txt = Lident "int"; _}, _) -> TPrim TInt32
  | Ptyp_constr ({txt = Lident "int64"; _}, _) -> TReg Int64
  | _ ->
      Location.raise_errorf
        ~loc
        "Unsupported type in Sarek top-level registration"

let constructor_strings_of_core_type_decl ~loc (tdecl : type_declaration) =
  match tdecl.ptype_kind with
  | Ptype_record labels ->
      let fields =
        List.map
          (fun ld ->
            ( ld.pld_name.txt,
              typ_of_core_type ~loc ld.pld_type,
              ld.pld_mutable = Mutable ))
          labels
      in
      let strs = record_constructor_strings tdecl.ptype_name.txt fields in
      Ast_builder.Default.elist
        ~loc
        (List.map (Ast_builder.Default.estring ~loc) strs)
  | Ptype_variant constrs ->
      let constrs =
        List.map
          (fun cd ->
            match cd.pcd_args with
            | Pcstr_tuple [] -> (cd.pcd_name.txt, None)
            | Pcstr_tuple [ct] ->
                (* WRONG-WIDTH #4. This arm used to read
                     let _ = typ_of_core_type ~loc ct in (cd.pcd_name.txt, None)
                   — the payload's type was computed (so an unsupported payload
                   type was still rejected) and then DISCARDED. [None] means
                   "nullary constructor" to [variant_constructor_strings], so
                   `Shade of float32` emitted a union member declared
                   `int col_sarek_Shade_t;` and a builder
                   `build_col_Shade()` with no parameter at all: the payload
                   was given the wrong C type AND no way to be set. *)
                (cd.pcd_name.txt, Some (typ_of_core_type ~loc ct))
            | Pcstr_tuple _ | Pcstr_record _ ->
                Location.raise_errorf
                  ~loc
                  "Only zero or single-argument constructors supported")
          constrs
      in
      let strs = variant_constructor_strings tdecl.ptype_name.txt constrs in
      Ast_builder.Default.elist
        ~loc
        (List.map (Ast_builder.Default.estring ~loc) strs)
  | _ ->
      Location.raise_errorf ~loc "Only record/variant types can be registered"
