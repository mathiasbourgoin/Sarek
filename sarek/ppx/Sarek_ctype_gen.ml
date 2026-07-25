(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * C-type / struct-string generation for [%%sarek.type] top-level type
 * registrations. Maps Sarek types (and ppxlib core_types) to the C struct,
 * union and builder-function source strings that the GPU backends splice into
 * generated kernels.
 ******************************************************************************)

open Ppxlib
open Sarek_types

let mangle_type_name name = String.map (function '.' -> '_' | c -> c) name

let rec c_type_of_typ ty : string =
  match repr ty with
  | TPrim TInt32 -> "int"
  | TPrim TBool -> "int"
  | TPrim TUnit -> "void"
  | TReg Int64 -> "long"
  | TReg Float32 -> "float"
  | TReg Float64 -> "double"
  | TRecord (name, _) -> "struct " ^ mangle_type_name name ^ "_sarek"
  | TVariant (name, _) -> "struct " ^ mangle_type_name name ^ "_sarek"
  | TVec t -> c_type_of_typ t ^ " *"
  | TArr (t, _) -> c_type_of_typ t ^ " *"
  | _ -> "int"

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
                let _ = typ_of_core_type ~loc ct in
                (cd.pcd_name.txt, None)
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
