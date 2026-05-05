(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This is the PPX entry point. It registers the [%kernel ...] extension
 * and orchestrates parsing, type checking, lowering, and code generation.
 *
 * Type and intrinsic information comes from:
 * 1. Sarek_env.with_stdlib - hardcoded core types and intrinsics
 * 2. [@@sarek.type] attributes - user-defined record/variant types
 * 3. Runtime Sarek_registry - populated by %sarek_intrinsic at load time
 ******************************************************************************)

open Ppxlib
open Sarek_ppx_lib

(* Force stdlib module initialization to populate the PPX registry.
   This must happen before any kernel expansion uses Sarek_env.with_stdlib(). *)
let () = Sarek_stdlib.force_init ()

(* Force Float64 module initialization to populate the PPX registry.
   Float64 is in a separate library (sarek.float64) for devices that support it.
   The PPX knows about Float64 intrinsics, but runtime code only links if needed. *)
let () = ignore Sarek_float64.Float64.sin

(* Registry of types declared in the current compilation unit via [@@sarek.type] *)
let registered_types : Sarek_ast.type_decl list ref = ref []

(* Registry of module items declared in the current compilation unit *)
let registered_mods : Sarek_ast.module_item list ref = ref []

let tdecl_key = function
  | Sarek_ast.Type_record {tdecl_name; tdecl_module; _}
  | Sarek_ast.Type_variant {tdecl_name; tdecl_module; _} -> (
      match tdecl_module with
      | Some m -> m ^ "." ^ tdecl_name
      | None -> tdecl_name)

let dedup_tdecls decls =
  let module S = Set.Make (String) in
  let _, revs =
    List.fold_left
      (fun (seen, acc) d ->
        let key = tdecl_key d in
        if S.mem key seen then (seen, acc) else (S.add key seen, d :: acc))
      (S.empty, [])
      decls
  in
  List.rev revs

let dedup_mods mods =
  let module S = Set.Make (String) in
  let _, revs =
    List.fold_left
      (fun (seen, acc) m ->
        let key =
          match m with
          | Sarek_ast.MConst (n, _, _) | Sarek_ast.MFun (n, _, _, _) -> n
        in
        if S.mem key seen then (seen, acc) else (S.add key seen, m :: acc))
      (S.empty, [])
      mods
  in
  List.rev revs

let rec flatten_longident = function
  | Lident s -> [s]
  | Ldot (li, s) -> flatten_longident li @ [s]
  | Lapply _ -> []

let custom_descriptor_expr_of_lid ~loc lid =
  match List.rev (flatten_longident lid) with
  | name :: modules ->
      let modules = List.rev modules in
      let ident =
        match modules with
        | [] -> Lident (name ^ "_custom")
        | first :: rest ->
            List.fold_left
              (fun acc m -> Ldot (acc, m))
              (Lident first)
              (rest @ [name ^ "_custom"])
      in
      Ast_builder.Default.pexp_ident ~loc {txt = ident; loc}
  | [] -> Ast_builder.Default.evar ~loc "unknown_custom"

let rec core_type_to_sarek_type_expr ~loc (ct : core_type) =
  match ct.ptyp_desc with
  | Ptyp_constr ({txt; _}, args) ->
      let name = String.concat "." (flatten_longident txt) in
      let args = List.map (core_type_to_sarek_type_expr ~loc) args in
      Sarek_ast.TEConstr (name, args)
  | Ptyp_var v -> Sarek_ast.TEVar v
  | Ptyp_arrow (_, a, b) ->
      Sarek_ast.TEArrow
        ( core_type_to_sarek_type_expr ~loc a,
          core_type_to_sarek_type_expr ~loc b )
  | Ptyp_tuple l ->
      Sarek_ast.TETuple (List.map (core_type_to_sarek_type_expr ~loc) l)
  | _ ->
      Location.raise_errorf ~loc "Unsupported type expression in [@@sarek.type]"

let module_name_of_loc loc =
  let file = loc.loc_start.pos_fname in
  let base = Filename.(remove_extension (basename file)) in
  String.capitalize_ascii base

(* Registry of known type sizes, populated during PPX processing.
   This allows nested types to look up sizes of previously processed types. *)
let type_size_registry : (string, int) Hashtbl.t = Hashtbl.create 16

(* Get size of a type, checking registry for custom types *)
let get_type_size_from_core_type (ct : core_type) : int =
  match ct.ptyp_desc with
  | Ptyp_constr ({txt = Lident "int32"; _}, _) -> 4
  | Ptyp_constr ({txt = Lident "int64"; _}, _) -> 8
  | Ptyp_constr ({txt = Lident "float32"; _}, _) -> 4
  | Ptyp_constr ({txt = Lident "float"; _}, _) -> 4 (* GPU float32 *)
  | Ptyp_constr ({txt = Lident "int"; _}, _) -> 4
  | Ptyp_constr ({txt = Lident type_name; _}, _) -> (
      (* Check if it's a known custom type *)
      try Hashtbl.find type_size_registry type_name with Not_found -> 4)
  | _ -> 4

let calc_type_size_early (labels : label_declaration list) : int =
  List.fold_left
    (fun acc ld -> acc + get_type_size_from_core_type ld.pld_type)
    0
    labels

(* Helper to get payload size in bytes for a variant constructor *)
let variant_payload_byte_size (cd : constructor_declaration) : int =
  match cd.pcd_args with
  | Pcstr_tuple [] -> 0
  | Pcstr_tuple [ct] -> get_type_size_from_core_type ct
  | Pcstr_tuple cts ->
      List.fold_left ( + ) 0 (List.map get_type_size_from_core_type cts)
  | Pcstr_record _ -> 0 (* TODO: support inline records *)

(* Compute type size and register it in the registry for nested type lookup. *)
let register_type_size (td : type_declaration) =
  let type_name = td.ptype_name.txt in
  let size =
    match td.ptype_kind with
    | Ptype_record labels -> calc_type_size_early labels
    | Ptype_variant constrs ->
        (* 4 bytes for tag + max payload size *)
        let max_payload =
          List.fold_left max 0 (List.map variant_payload_byte_size constrs)
        in
        4 + max_payload
    | _ -> 4
  in
  Hashtbl.replace type_size_registry type_name size

let register_sarek_type_decl ~loc (td : type_declaration) =
  (* Register type size first so nested types can look it up *)
  register_type_size td ;
  let tdecl =
    match td.ptype_kind with
    | Ptype_record labels ->
        let fields =
          List.map
            (fun ld ->
              ( ld.pld_name.txt,
                ld.pld_mutable = Mutable,
                core_type_to_sarek_type_expr ~loc ld.pld_type ))
            labels
        in
        Sarek_ast.Type_record
          {
            tdecl_name = td.ptype_name.txt;
            tdecl_module = Some (module_name_of_loc loc);
            tdecl_fields = fields;
            tdecl_loc = Sarek_ast.loc_of_ppxlib loc;
          }
    | Ptype_variant constrs ->
        let cons =
          List.map
            (fun cd ->
              match cd.pcd_args with
              | Pcstr_tuple [] -> (cd.pcd_name.txt, None)
              | Pcstr_tuple [ct] ->
                  (cd.pcd_name.txt, Some (core_type_to_sarek_type_expr ~loc ct))
              | Pcstr_tuple cts ->
                  let args = List.map (core_type_to_sarek_type_expr ~loc) cts in
                  (cd.pcd_name.txt, Some (Sarek_ast.TETuple args))
              | Pcstr_record _ ->
                  Location.raise_errorf
                    ~loc
                    "Inline records not supported in [@@sarek.type]")
            constrs
        in
        Sarek_ast.Type_variant
          {
            tdecl_name = td.ptype_name.txt;
            tdecl_module = Some (module_name_of_loc loc);
            tdecl_constructors = cons;
            tdecl_loc = Sarek_ast.loc_of_ppxlib loc;
          }
    | _ ->
        Location.raise_errorf
          ~loc
          "Only record or variant types can be used with [@@sarek.type]"
  in
  registered_types := tdecl :: !registered_types ;
  ()

let register_sarek_module_item ~loc:_ item =
  (if Sarek_debug.enabled then
     match item with
     | Sarek_ast.MFun (name, _, _, _) ->
         Format.eprintf "Sarek PPX: register module fun %s@." name
     | Sarek_ast.MConst (name, _, _) ->
         Format.eprintf "Sarek PPX: register module const %s@." name) ;
  registered_mods := item :: !registered_mods

(** Scan a single .ml file for [@@sarek.type] and [@sarek.module] declarations
*)
let scan_file_for_sarek_types path =
  try
    let ic = open_in path in
    let lexbuf = Lexing.from_channel ic in
    lexbuf.lex_curr_p <-
      {lexbuf.lex_curr_p with pos_fname = path; pos_bol = 0; pos_lnum = 1} ;
    let st = Parse.implementation lexbuf in
    close_in ic ;
    List.iter
      (fun item ->
        match item.pstr_desc with
        | Pstr_type (_rf, decls) ->
            List.iter
              (fun d ->
                let has_attr name a = String.equal a.attr_name.txt name in
                if
                  List.exists (has_attr "sarek.type") d.ptype_attributes
                  || List.exists
                       (has_attr "sarek.type_private")
                       d.ptype_attributes
                then register_sarek_type_decl ~loc:d.ptype_loc d)
              decls
        | Pstr_value (rec_flag, vbs) ->
            let is_rec = rec_flag = Recursive in
            List.iter
              (fun vb ->
                let has_attr name a = String.equal a.attr_name.txt name in
                if
                  List.exists (has_attr "sarek.module") vb.pvb_attributes
                  || List.exists
                       (has_attr "sarek.module_private")
                       vb.pvb_attributes
                then (
                  if Sarek_debug.enabled then
                    Format.eprintf
                      "Sarek PPX: sarek.module binding %s@."
                      (Option.value
                         (Sarek_parse.extract_name_from_pattern vb.pvb_pat)
                         ~default:"<anon>") ;
                  let name =
                    match Sarek_parse.extract_name_from_pattern vb.pvb_pat with
                    | Some n -> n
                    | None ->
                        Location.raise_errorf
                          ~loc:vb.pvb_pat.ppat_loc
                          "Expected variable name"
                  in
                  let ty = Sarek_parse.extract_type_from_pattern vb.pvb_pat in
                  let item =
                    match Sarek_parse.collect_fun_params vb.pvb_expr with
                    | params, Some (Fun_body body_expr) when params <> [] ->
                        let params =
                          List.map
                            (fun p ->
                              Sarek_parse.extract_param_from_pattern
                                (Sarek_parse.pattern_of_param p))
                            params
                        in
                        let body = Sarek_parse.parse_expression body_expr in
                        Sarek_ast.MFun (name, is_rec, params, body)
                    | _, Some (Fun_cases _) ->
                        Location.raise_errorf
                          ~loc:vb.pvb_expr.pexp_loc
                          "Pattern-matching functions are not supported for \
                           [@sarek.module]"
                    | _ ->
                        let value = Sarek_parse.parse_expression vb.pvb_expr in
                        let ty =
                          match ty with
                          | Some t -> t
                          | None ->
                              Location.raise_errorf
                                ~loc:vb.pvb_pat.ppat_loc
                                "[@sarek.module] constants require a type \
                                 annotation"
                        in
                        Sarek_ast.MConst (name, ty, value)
                  in
                  register_sarek_module_item ~loc:vb.pvb_loc item ;
                  let module_name =
                    String.capitalize_ascii
                      (Filename.chop_extension (Filename.basename path))
                  in
                  let item_name =
                    match item with
                    | Sarek_ast.MFun (n, _, _, _) -> n
                    | Sarek_ast.MConst (n, _, _) -> n
                  in
                  Sarek_ppx_registry.register_module_item
                    (Sarek_ppx_registry.make_module_item_info
                       ~name:item_name
                       ~module_name
                       ~item)))
              vbs
        | _ -> ())
      st
  with _ -> ()

(** Scan a directory for .ml files with Sarek declarations, or scan a single
    file *)
let scan_dir_for_sarek_types ?single_file directory =
  match single_file with
  | Some path -> scan_file_for_sarek_types path
  | None ->
      Array.iter
        (fun fname ->
          if Filename.check_suffix fname ".ml" then
            let path = Filename.concat directory fname in
            scan_file_for_sarek_types path)
        (try Sys.readdir directory with Sys_error _ -> [||])

(* Attribute used to mark Sarek-visible type declarations *)
let sarek_type_attr =
  Attribute.declare
    "sarek.type"
    Attribute.Context.type_declaration
    Ast_pattern.(pstr nil)
    ()

let sarek_type_private_attr =
  Attribute.declare
    "sarek.type_private"
    Attribute.Context.type_declaration
    Ast_pattern.(pstr nil)
    ()

(** Generate field accessor functions for a record type. Example: for type point
    with fields x and y, generates: let sarek_get_point_x (p : point) : float32
    = p.x let sarek_get_point_y (p : point) : float32 = p.y *)
let generate_field_accessors ~loc (td : type_declaration) : structure_item list
    =
  match td.ptype_kind with
  | Ptype_record labels ->
      let type_name = td.ptype_name.txt in
      let type_lid = {txt = Lident type_name; loc} in
      let type_ct = Ast_builder.Default.ptyp_constr ~loc type_lid [] in
      List.map
        (fun (ld : label_declaration) ->
          let field_name = ld.pld_name.txt in
          let field_type = ld.pld_type in
          let fn_name = Printf.sprintf "sarek_get_%s_%s" type_name field_name in
          let param_pat =
            Ast_builder.Default.ppat_constraint
              ~loc
              (Ast_builder.Default.pvar ~loc "p")
              type_ct
          in
          let field_access =
            Ast_builder.Default.pexp_field
              ~loc
              (Ast_builder.Default.evar ~loc "p")
              {txt = Lident field_name; loc}
          in
          let body =
            Ast_builder.Default.pexp_constraint ~loc field_access field_type
          in
          let fn_expr =
            Ast_builder.Default.pexp_fun ~loc Nolabel None param_pat body
          in
          let fn_pat = Ast_builder.Default.pvar ~loc fn_name in
          [%stri let [%p fn_pat] = [%e fn_expr]])
        labels
  | _ -> []

(** Extract type name from a core_type for registry registration *)
let type_name_of_core_type (ct : core_type) : string =
  match ct.ptyp_desc with
  | Ptyp_constr ({txt = Lident name; _}, _) -> name
  | Ptyp_constr ({txt = Ldot (_, name); _}, _) -> name
  | _ -> "unknown"

(** Calculate the size in bytes of a sarek type based on its fields. Uses 4
    bytes for int32/float32, 8 bytes for int64/float64. *)
let calc_type_size (labels : label_declaration list) : int =
  calc_type_size_early labels

(** Get the accessor function for a field type (legacy SPOC path removed) *)
let get_accessor_for_type ~loc (_ct : core_type) : expression =
  let _ = loc in
  [%expr failwith "SPOC custom accessor removed"]

(** Get the setter function for a field type (legacy SPOC path removed) *)
let set_accessor_for_type ~loc (_ct : core_type) : expression =
  let _ = loc in
  [%expr failwith "SPOC custom accessor removed"]

(** Get the field count (number of primitive fields, counting nested as 1 for
    now) *)
let field_element_count (ct : core_type) : int =
  match ct.ptyp_desc with
  | Ptyp_constr ({txt = Lident "float32"; _}, _) -> 1
  | Ptyp_constr ({txt = Lident "float"; _}, _) -> 1
  | Ptyp_constr ({txt = Lident "int32"; _}, _) -> 1
  | Ptyp_constr ({txt = Lident "int"; _}, _) -> 1
  | _ -> 1

(** Get field size in bytes for V2 custom types *)
let field_byte_size (ct : core_type) : int = get_type_size_from_core_type ct

(** Generate a <name>_custom value for Vector.Custom. For a record type like
    float4 with float32 fields, generates get/set functions using
    Spoc.Tools.float32get/set *)
let[@warning "-32"] generate_custom_value ~loc:_ (_td : type_declaration) :
    structure_item list =
  []

(** Helper to generate a V2 field read expression based on field type.
    Dispatches to the correct Custom_helpers function. For nested custom types,
    generates a call using the nested type's _custom accessor. *)
let gen_field_read ~loc (ftype : core_type) (byte_off_expr : expression) :
    expression =
  match ftype.ptyp_desc with
  | Ptyp_constr ({txt = Lident "float32"; _}, _) ->
      [%expr
        Spoc_core.Vector.Custom_helpers.read_float32
          raw_ptr
          (base_off + [%e byte_off_expr])]
  | Ptyp_constr ({txt = Lident "float"; _}, _) ->
      (* OCaml float is 8 bytes (float64) in GPU land - but for compatibility,
         check if this is meant to be float32 or float64 based on context.
         For now treat bare 'float' as float32 for GPU compatibility. *)
      [%expr
        Spoc_core.Vector.Custom_helpers.read_float32
          raw_ptr
          (base_off + [%e byte_off_expr])]
  | Ptyp_constr ({txt = Lident "int32"; _}, _) ->
      [%expr
        Spoc_core.Vector.Custom_helpers.read_int32
          raw_ptr
          (base_off + [%e byte_off_expr])]
  | Ptyp_constr ({txt = Lident "int64"; _}, _) ->
      [%expr
        Spoc_core.Vector.Custom_helpers.read_int64
          raw_ptr
          (base_off + [%e byte_off_expr])]
  | Ptyp_constr ({txt = Lident "int"; _}, _) ->
      [%expr
        Spoc_core.Vector.Custom_helpers.read_int
          raw_ptr
          (base_off + [%e byte_off_expr])]
  | Ptyp_constr ({txt; _}, _) ->
      (* Nested custom type - call its _custom.get directly with adjusted pointer *)
      let custom_get =
        Ast_builder.Default.pexp_field
          ~loc
          (custom_descriptor_expr_of_lid ~loc txt)
          {txt = Lident "get"; loc}
      in
      (* Adjust pointer to field offset, then read at index 0 *)
      [%expr
        let field_ptr =
          let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t raw_ptr in
          Ctypes.to_voidp Ctypes.(byte_ptr +@ (base_off + [%e byte_off_expr]))
        in
        [%e custom_get] field_ptr 0]
  | _ ->
      (* Fallback to float32 for unknown types *)
      [%expr
        Spoc_core.Vector.Custom_helpers.read_float32
          raw_ptr
          (base_off + [%e byte_off_expr])]

(** Helper to generate a V2 field write expression based on field type. *)
let gen_field_write ~loc (ftype : core_type) (byte_off_expr : expression)
    (value_expr : expression) : expression =
  match ftype.ptyp_desc with
  | Ptyp_constr ({txt = Lident "float32"; _}, _) ->
      [%expr
        Spoc_core.Vector.Custom_helpers.write_float32
          raw_ptr
          (base_off + [%e byte_off_expr])
          [%e value_expr]]
  | Ptyp_constr ({txt = Lident "float"; _}, _) ->
      [%expr
        Spoc_core.Vector.Custom_helpers.write_float32
          raw_ptr
          (base_off + [%e byte_off_expr])
          [%e value_expr]]
  | Ptyp_constr ({txt = Lident "int32"; _}, _) ->
      [%expr
        Spoc_core.Vector.Custom_helpers.write_int32
          raw_ptr
          (base_off + [%e byte_off_expr])
          [%e value_expr]]
  | Ptyp_constr ({txt = Lident "int64"; _}, _) ->
      [%expr
        Spoc_core.Vector.Custom_helpers.write_int64
          raw_ptr
          (base_off + [%e byte_off_expr])
          [%e value_expr]]
  | Ptyp_constr ({txt = Lident "int"; _}, _) ->
      [%expr
        Spoc_core.Vector.Custom_helpers.write_int
          raw_ptr
          (base_off + [%e byte_off_expr])
          [%e value_expr]]
  | Ptyp_constr ({txt; _}, _) ->
      (* Nested custom type - call its _custom.set directly with adjusted pointer *)
      let custom_set =
        Ast_builder.Default.pexp_field
          ~loc
          (custom_descriptor_expr_of_lid ~loc txt)
          {txt = Lident "set"; loc}
      in
      [%expr
        let field_ptr =
          let byte_ptr = Ctypes.from_voidp Ctypes.uint8_t raw_ptr in
          Ctypes.to_voidp Ctypes.(byte_ptr +@ (base_off + [%e byte_off_expr]))
        in
        [%e custom_set] field_ptr 0 [%e value_expr]]
  | _ ->
      [%expr
        Spoc_core.Vector.Custom_helpers.write_float32
          raw_ptr
          (base_off + [%e byte_off_expr])
          [%e value_expr]]

(** Generate a <name>_custom value for Spoc_core.Vector.custom_type. For a
    record type (e.g., point with float32 fields), generates get/set functions
    using Ctypes pointer arithmetic. Supports nested custom types via their
    _custom accessor. *)
let generate_custom_value ~loc (td : type_declaration) : structure_item list =
  match td.ptype_kind with
  | Ptype_record labels ->
      let type_name = td.ptype_name.txt in
      let custom_name = type_name ^ "_custom" in
      let custom_pat = Ast_builder.Default.pvar ~loc custom_name in
      let size = calc_type_size labels in
      let size_expr = Ast_builder.Default.eint ~loc size in
      let name_expr = Ast_builder.Default.estring ~loc type_name in
      let type_annot =
        Ast_builder.Default.ptyp_constr
          ~loc
          {txt = Ldot (Ldot (Lident "Spoc_core", "Vector"), "custom_type"); loc}
          [
            Ast_builder.Default.ptyp_constr ~loc {txt = Lident type_name; loc} [];
          ]
      in

      (* Calculate byte offsets for each field *)
      let field_infos =
        let rec calc_offsets byte_off = function
          | [] -> []
          | ld :: rest ->
              let fsize = field_byte_size ld.pld_type in
              (ld.pld_name.txt, byte_off, ld.pld_type)
              :: calc_offsets (byte_off + fsize) rest
        in
        calc_offsets 0 labels
      in

      (* Generate V2 custom type - use a helper function to avoid value restriction issues *)
      let _ = type_annot in
      (* suppress unused warning *)

      (* Generate a function that creates the custom_type when called.
         This avoids the value restriction since it's a function definition. *)
      let make_fn_name = type_name ^ "_make_custom" in
      let make_fn_pat = Ast_builder.Default.pvar ~loc make_fn_name in
      let type_id_name = type_name ^ "_type_id" in
      let type_id_pat = Ast_builder.Default.pvar ~loc type_id_name in
      let type_id_expr = Ast_builder.Default.evar ~loc type_id_name in
      let vector_type_id_name = type_name ^ "_vector_type_id" in
      let vector_type_id_pat =
        Ast_builder.Default.pvar ~loc vector_type_id_name
      in
      let vector_type_id_expr =
        Ast_builder.Default.evar ~loc vector_type_id_name
      in
      let make_fn_call =
        Ast_builder.Default.eapply
          ~loc
          (Ast_builder.Default.evar ~loc make_fn_name)
          [[%expr ()]]
      in

      (* Generate get function that reads each field and constructs record *)
      let get_field_exprs =
        List.map
          (fun (name, byte_off, ftype) ->
            let byte_off_expr = Ast_builder.Default.eint ~loc byte_off in
            let field_expr = gen_field_read ~loc ftype byte_off_expr in
            ({txt = Lident name; loc}, field_expr))
          field_infos
      in
      let get_record =
        Ast_builder.Default.pexp_record ~loc get_field_exprs None
      in
      let get_fn =
        [%expr
          fun raw_ptr idx ->
            let base_off = idx * [%e size_expr] in
            [%e get_record]]
      in

      (* Generate set function that writes each field *)
      let set_stmts =
        List.map
          (fun (name, byte_off, ftype) ->
            let byte_off_expr = Ast_builder.Default.eint ~loc byte_off in
            let field_access =
              Ast_builder.Default.pexp_field
                ~loc
                [%expr v]
                {txt = Lident name; loc}
            in
            gen_field_write ~loc ftype byte_off_expr field_access)
          field_infos
      in
      let set_body =
        match set_stmts with
        | [] -> [%expr ()]
        | [stmt] -> stmt
        | stmt :: rest ->
            List.fold_left
              (fun acc s ->
                [%expr
                  [%e acc] ;
                  [%e s]])
              stmt
              rest
      in
      let set_fn =
        [%expr
          fun raw_ptr idx v ->
            let base_off = idx * [%e size_expr] in
            [%e set_body]]
      in

      [
        [%stri let [%p type_id_pat] = Sarek_ir_types.Type_id.create ()];
        [%stri let [%p vector_type_id_pat] = Sarek_ir_types.Type_id.create ()];
        (* Generate: let point_make_custom = fun () -> { ... } *)
        [%stri
          let [%p make_fn_pat] =
           fun () ->
            ({
               Spoc_core.Vector.elem_size = [%e size_expr];
               type_id = [%e type_id_expr];
               vector_type_id = [%e vector_type_id_expr];
               name = [%e name_expr];
               get = [%e get_fn];
               set = [%e set_fn];
             }
              : [%t type_annot])];
        (* Generate: let point_custom = point_make_custom () *)
        [%stri let [%p custom_pat] = [%e make_fn_call]];
      ]
  | Ptype_variant constrs ->
      (* Generate V2 custom type for variants (with or without payloads).
         Layout: [tag:4 bytes (int32)][payload:max_payload_bytes]
         For simple enums (no args), elem_size=4.
         For variants with payloads, elem_size=4+max_payload_bytes. *)
      let type_name = td.ptype_name.txt in
      let custom_name = type_name ^ "_custom" in
      let custom_pat = Ast_builder.Default.pvar ~loc custom_name in
      let name_expr = Ast_builder.Default.estring ~loc type_name in
      let type_annot =
        Ast_builder.Default.ptyp_constr
          ~loc
          {txt = Ldot (Ldot (Lident "Spoc_core", "Vector"), "custom_type"); loc}
          [
            Ast_builder.Default.ptyp_constr ~loc {txt = Lident type_name; loc} [];
          ]
      in

      (* Helper to get payload size in bytes for a constructor *)
      let payload_byte_size cd =
        match cd.pcd_args with
        | Pcstr_tuple [] -> 0
        | Pcstr_tuple [ct] -> get_type_size_from_core_type ct
        | Pcstr_tuple cts ->
            List.fold_left ( + ) 0 (List.map get_type_size_from_core_type cts)
        | Pcstr_record _ -> 0 (* TODO: support inline records *)
      in
      let max_payload_bytes =
        List.fold_left max 0 (List.map payload_byte_size constrs)
      in
      let total_size = 4 + max_payload_bytes in
      (* 4 for int32 tag *)
      let size_expr = Ast_builder.Default.eint ~loc total_size in

      (* Build match cases for get: int32 tag -> constructor with payload *)
      let get_cases =
        List.mapi
          (fun i cd ->
            let tag_pat = Ast_builder.Default.pint ~loc i in
            let ctor_expr =
              match cd.pcd_args with
              | Pcstr_tuple [] ->
                  Ast_builder.Default.pexp_construct
                    ~loc
                    {txt = Lident cd.pcd_name.txt; loc}
                    None
              | Pcstr_tuple [ct] ->
                  (* Read payload at base_off+4 *)
                  let payload_off = Ast_builder.Default.eint ~loc 4 in
                  let payload_expr = gen_field_read ~loc ct payload_off in
                  Ast_builder.Default.pexp_construct
                    ~loc
                    {txt = Lident cd.pcd_name.txt; loc}
                    (Some payload_expr)
              | Pcstr_tuple cts ->
                  (* Multiple args: read each at successive offsets *)
                  let _, args =
                    List.fold_left
                      (fun (off, acc) ct ->
                        let off_expr = Ast_builder.Default.eint ~loc off in
                        let arg_expr = gen_field_read ~loc ct off_expr in
                        let size = get_type_size_from_core_type ct in
                        (off + size, acc @ [arg_expr]))
                      (4, [])
                      cts
                  in
                  Ast_builder.Default.pexp_construct
                    ~loc
                    {txt = Lident cd.pcd_name.txt; loc}
                    (Some (Ast_builder.Default.pexp_tuple ~loc args))
              | Pcstr_record _ ->
                  Ast_builder.Default.pexp_construct
                    ~loc
                    {txt = Lident cd.pcd_name.txt; loc}
                    None
            in
            {pc_lhs = tag_pat; pc_guard = None; pc_rhs = ctor_expr})
          constrs
      in
      (* Add a fallback case *)
      let fallback_ctor = List.hd constrs in
      let fallback_expr =
        match fallback_ctor.pcd_args with
        | Pcstr_tuple [] ->
            Ast_builder.Default.pexp_construct
              ~loc
              {txt = Lident fallback_ctor.pcd_name.txt; loc}
              None
        | Pcstr_tuple [ct] ->
            let payload_off = Ast_builder.Default.eint ~loc 4 in
            let payload_expr = gen_field_read ~loc ct payload_off in
            Ast_builder.Default.pexp_construct
              ~loc
              {txt = Lident fallback_ctor.pcd_name.txt; loc}
              (Some payload_expr)
        | Pcstr_tuple cts ->
            (* For fallback case with multiple arguments, generate default values *)
            let args =
              List.map
                (fun ct ->
                  (* Generate default value based on type *)
                  match ct.ptyp_desc with
                  | Ptyp_constr ({txt = Lident "float32"; _}, _) -> [%expr 0.0]
                  | Ptyp_constr ({txt = Lident "float"; _}, _) -> [%expr 0.0]
                  | Ptyp_constr ({txt = Lident "float64"; _}, _) -> [%expr 0.0]
                  | Ptyp_constr ({txt = Lident "int32"; _}, _) -> [%expr 0l]
                  | Ptyp_constr ({txt = Lident "int64"; _}, _) -> [%expr 0L]
                  | Ptyp_constr ({txt = Lident "int"; _}, _) -> [%expr 0l]
                  | Ptyp_constr ({txt = Lident "bool"; _}, _) -> [%expr false]
                  | Ptyp_constr ({txt = Lident "unit"; _}, _) -> [%expr ()]
                  | Ptyp_constr ({txt = Lident "char"; _}, _) -> [%expr '\000']
                  | _ ->
                      (* For custom types, we'd need their default constructor - 
                         this is a rare fallback case so using unit is acceptable *)
                      [%expr ()])
                cts
            in
            Ast_builder.Default.pexp_construct
              ~loc
              {txt = Lident fallback_ctor.pcd_name.txt; loc}
              (Some (Ast_builder.Default.pexp_tuple ~loc args))
        | _ ->
            Ast_builder.Default.pexp_construct
              ~loc
              {txt = Lident fallback_ctor.pcd_name.txt; loc}
              None
      in
      let fallback_case =
        {
          pc_lhs = Ast_builder.Default.ppat_any ~loc;
          pc_guard = None;
          pc_rhs = fallback_expr;
        }
      in
      let total_size_expr = Ast_builder.Default.eint ~loc total_size in
      let get_match =
        Ast_builder.Default.pexp_match
          ~loc
          [%expr
            Int32.to_int
              (Spoc_core.Vector.Custom_helpers.read_int32 raw_ptr base_off)]
          (get_cases @ [fallback_case])
      in
      let get_body =
        [%expr
          let base_off = idx * [%e total_size_expr] in
          [%e get_match]]
      in
      let get_fn = [%expr fun raw_ptr idx -> [%e get_body]] in

      (* Build match cases for set: constructor -> write tag + payload *)
      let set_cases =
        List.mapi
          (fun i cd ->
            let tag_expr = Ast_builder.Default.eint ~loc i in
            match cd.pcd_args with
            | Pcstr_tuple [] ->
                let ctor_pat =
                  Ast_builder.Default.ppat_construct
                    ~loc
                    {txt = Lident cd.pcd_name.txt; loc}
                    None
                in
                {
                  pc_lhs = ctor_pat;
                  pc_guard = None;
                  pc_rhs =
                    [%expr
                      Spoc_core.Vector.Custom_helpers.write_int32
                        raw_ptr
                        base_off
                        (Int32.of_int [%e tag_expr])];
                }
            | Pcstr_tuple [ct] ->
                let payload_pat = Ast_builder.Default.pvar ~loc "payload" in
                let ctor_pat =
                  Ast_builder.Default.ppat_construct
                    ~loc
                    {txt = Lident cd.pcd_name.txt; loc}
                    (Some payload_pat)
                in
                let payload_off = Ast_builder.Default.eint ~loc 4 in
                let write_payload =
                  gen_field_write ~loc ct payload_off [%expr payload]
                in
                {
                  pc_lhs = ctor_pat;
                  pc_guard = None;
                  pc_rhs =
                    [%expr
                      Spoc_core.Vector.Custom_helpers.write_int32
                        raw_ptr
                        base_off
                        (Int32.of_int [%e tag_expr]) ;
                      [%e write_payload]];
                }
            | Pcstr_tuple cts ->
                (* Multiple args: match tuple, write each *)
                let arg_pats =
                  List.mapi
                    (fun j _ ->
                      Ast_builder.Default.pvar ~loc (Printf.sprintf "arg%d" j))
                    cts
                in
                let ctor_pat =
                  Ast_builder.Default.ppat_construct
                    ~loc
                    {txt = Lident cd.pcd_name.txt; loc}
                    (Some (Ast_builder.Default.ppat_tuple ~loc arg_pats))
                in
                let _, write_stmts =
                  List.fold_left
                    (fun (off, acc) (j, ct) ->
                      let off_expr = Ast_builder.Default.eint ~loc off in
                      let arg_var =
                        Ast_builder.Default.evar ~loc (Printf.sprintf "arg%d" j)
                      in
                      let write_stmt =
                        gen_field_write ~loc ct off_expr arg_var
                      in
                      let size = get_type_size_from_core_type ct in
                      (off + size, acc @ [write_stmt]))
                    (4, [])
                    (List.mapi (fun j ct -> (j, ct)) cts)
                in
                let body =
                  List.fold_left
                    (fun acc s ->
                      [%expr
                        [%e acc] ;
                        [%e s]])
                    [%expr
                      Spoc_core.Vector.Custom_helpers.write_int32
                        raw_ptr
                        base_off
                        (Int32.of_int [%e tag_expr])]
                    write_stmts
                in
                {pc_lhs = ctor_pat; pc_guard = None; pc_rhs = body}
            | Pcstr_record _ ->
                let ctor_pat =
                  Ast_builder.Default.ppat_construct
                    ~loc
                    {txt = Lident cd.pcd_name.txt; loc}
                    None
                in
                {
                  pc_lhs = ctor_pat;
                  pc_guard = None;
                  pc_rhs =
                    [%expr
                      Spoc_core.Vector.Custom_helpers.write_int32
                        raw_ptr
                        base_off
                        (Int32.of_int [%e tag_expr])];
                })
          constrs
      in
      let set_match = Ast_builder.Default.pexp_match ~loc [%expr v] set_cases in
      let set_body =
        [%expr
          let base_off = idx * [%e total_size_expr] in
          [%e set_match]]
      in
      let set_fn = [%expr fun raw_ptr idx v -> [%e set_body]] in

      let make_fn_name = type_name ^ "_make_custom" in
      let make_fn_pat = Ast_builder.Default.pvar ~loc make_fn_name in
      let type_id_name = type_name ^ "_type_id" in
      let type_id_pat = Ast_builder.Default.pvar ~loc type_id_name in
      let type_id_expr = Ast_builder.Default.evar ~loc type_id_name in
      let vector_type_id_name = type_name ^ "_vector_type_id" in
      let vector_type_id_pat =
        Ast_builder.Default.pvar ~loc vector_type_id_name
      in
      let vector_type_id_expr =
        Ast_builder.Default.evar ~loc vector_type_id_name
      in
      let make_fn_call =
        Ast_builder.Default.eapply
          ~loc
          (Ast_builder.Default.evar ~loc make_fn_name)
          [[%expr ()]]
      in
      let _ = type_annot in

      [
        [%stri let [%p type_id_pat] = Sarek_ir_types.Type_id.create ()];
        [%stri let [%p vector_type_id_pat] = Sarek_ir_types.Type_id.create ()];
        [%stri
          let [%p make_fn_pat] =
           fun () ->
            ({
               Spoc_core.Vector.elem_size = [%e size_expr];
               type_id = [%e type_id_expr];
               vector_type_id = [%e vector_type_id_expr];
               name = [%e name_expr];
               get = [%e get_fn];
               set = [%e set_fn];
             }
              : [%t type_annot])];
        [%stri let [%p custom_pat] = [%e make_fn_call]];
      ]
  | _ -> []

(** Generate interpreter helper module for type-safe value conversion. Provides
    typed constructors for custom type handling. *)
let generate_interp_helpers ~loc (td : type_declaration) : structure_item list =
  let type_name = td.ptype_name.txt in
  let module_name = module_name_of_loc loc in
  let full_name = module_name ^ "." ^ type_name in
  let full_name_expr = Ast_builder.Default.estring ~loc full_name in

  match td.ptype_kind with
  | Ptype_record labels ->
      (* Generate from_values function: value array -> t *)
      let from_values_cases =
        List.mapi
          (fun i (ld : label_declaration) ->
            let field_name = ld.pld_name.txt in
            let field_type = ld.pld_type in
            let array_access =
              [%expr arr.([%e Ast_builder.Default.eint ~loc i])]
            in
            let converter =
              match field_type.ptyp_desc with
              | Ptyp_constr ({txt = Lident "float32"; _}, _)
              | Ptyp_constr ({txt = Lident "float"; _}, _) ->
                  [%expr
                    match [%e array_access] with
                    | Sarek.Sarek_value.VFloat32 f -> f
                    | Sarek.Sarek_value.VFloat64 f -> f
                    | _ ->
                        failwith
                          [%e
                            Ast_builder.Default.estring
                              ~loc
                              ("Field '" ^ field_name
                             ^ "' expected float32, got wrong type")]]
              | Ptyp_constr ({txt = Lident "float64"; _}, _) ->
                  [%expr
                    match [%e array_access] with
                    | Sarek.Sarek_value.VFloat64 f -> f
                    | Sarek.Sarek_value.VFloat32 f -> f
                    | _ ->
                        failwith
                          [%e
                            Ast_builder.Default.estring
                              ~loc
                              ("Field '" ^ field_name
                             ^ "' expected float64, got wrong type")]]
              | Ptyp_constr ({txt = Lident "int32"; _}, _)
              | Ptyp_constr ({txt = Lident "int"; _}, _) ->
                  [%expr
                    match [%e array_access] with
                    | Sarek.Sarek_value.VInt32 n -> n
                    | _ ->
                        failwith
                          [%e
                            Ast_builder.Default.estring
                              ~loc
                              ("Field '" ^ field_name
                             ^ "' expected int32, got wrong type")]]
              | Ptyp_constr ({txt = Lident "int64"; _}, _) ->
                  [%expr
                    match [%e array_access] with
                    | Sarek.Sarek_value.VInt64 n -> n
                    | _ ->
                        failwith
                          [%e
                            Ast_builder.Default.estring
                              ~loc
                              ("Field '" ^ field_name
                             ^ "' expected int64, got wrong type")]]
              | Ptyp_constr ({txt = Lident "bool"; _}, _) ->
                  [%expr
                    match [%e array_access] with
                    | Sarek.Sarek_value.VBool b -> b
                    | _ ->
                        failwith
                          [%e
                            Ast_builder.Default.estring
                              ~loc
                              ("Field '" ^ field_name ^ "' expected bool")]]
              | Ptyp_constr ({txt; _}, _) ->
                  (* Nested custom type - recursively call its helper *)
                  let custom_type = String.concat "." (flatten_longident txt) in
                  let custom_expr = custom_descriptor_expr_of_lid ~loc txt in
                  [%expr
                    match [%e array_access] with
                    | Sarek.Sarek_value.VRecord _ as nested_vrec -> (
                        match
                          Sarek.Sarek_type_helpers.lookup_typed
                            [%e custom_expr].Spoc_core.Vector.type_id
                        with
                        | Some (module H) -> H.from_value nested_vrec
                        | None ->
                            failwith
                              [%e
                                Ast_builder.Default.estring
                                  ~loc
                                  ("No helper for nested type " ^ custom_type)])
                    | _ ->
                        failwith
                          [%e
                            Ast_builder.Default.estring
                              ~loc
                              ("Field '" ^ field_name ^ "' expected record")]]
              | _ ->
                  [%expr
                    failwith
                      [%e
                        Ast_builder.Default.estring
                          ~loc
                          ("Unsupported field type: " ^ field_name)]]
            in
            (Ast_builder.Default.Located.lident ~loc field_name, converter))
          labels
      in
      let record_expr =
        Ast_builder.Default.pexp_record ~loc from_values_cases None
      in
      let from_values_fn = [%expr fun arr -> [%e record_expr]] in

      (* Generate to_values function: t -> value array *)
      let to_values_body =
        let field_conversions =
          List.map
            (fun (ld : label_declaration) ->
              let field_name = ld.pld_name.txt in
              let field_lid =
                Ast_builder.Default.Located.lident ~loc field_name
              in
              let field_access =
                Ast_builder.Default.pexp_field ~loc [%expr record] field_lid
              in
              let converter =
                match ld.pld_type.ptyp_desc with
                | Ptyp_constr ({txt = Lident "float32"; _}, _)
                | Ptyp_constr ({txt = Lident "float"; _}, _) ->
                    [%expr Sarek.Sarek_value.VFloat32 [%e field_access]]
                | Ptyp_constr ({txt = Lident "float64"; _}, _) ->
                    [%expr Sarek.Sarek_value.VFloat64 [%e field_access]]
                | Ptyp_constr ({txt = Lident "int32"; _}, _)
                | Ptyp_constr ({txt = Lident "int"; _}, _) ->
                    [%expr Sarek.Sarek_value.VInt32 [%e field_access]]
                | Ptyp_constr ({txt = Lident "int64"; _}, _) ->
                    [%expr Sarek.Sarek_value.VInt64 [%e field_access]]
                | Ptyp_constr ({txt = Lident "bool"; _}, _) ->
                    [%expr Sarek.Sarek_value.VBool [%e field_access]]
                | Ptyp_constr ({txt; _}, _) ->
                    (* Nested custom type - use helper to convert *)
                    let custom_type =
                      String.concat "." (flatten_longident txt)
                    in
                    let custom_expr = custom_descriptor_expr_of_lid ~loc txt in
                    [%expr
                      match
                        Sarek.Sarek_type_helpers.lookup_typed
                          [%e custom_expr].Spoc_core.Vector.type_id
                      with
                      | Some (module H) -> H.to_value [%e field_access]
                      | None ->
                          failwith
                            [%e
                              Ast_builder.Default.estring
                                ~loc
                                ("No helper for type " ^ custom_type)]]
                | _ ->
                    [%expr
                      failwith
                        [%e
                          Ast_builder.Default.estring
                            ~loc
                            ("Unsupported field type: " ^ field_name)]]
              in
              converter)
            labels
        in
        let array_expr = Ast_builder.Default.elist ~loc field_conversions in
        [%expr Array.of_list [%e array_expr]]
      in
      let to_values_fn = [%expr fun record -> [%e to_values_body]] in

      (* Generate get_field function: t -> string -> value *)
      let get_field_cases =
        List.map
          (fun (ld : label_declaration) ->
            let field_name = ld.pld_name.txt in
            let field_lid =
              Ast_builder.Default.Located.lident ~loc field_name
            in
            let field_access =
              Ast_builder.Default.pexp_field ~loc [%expr record] field_lid
            in
            let converter =
              match ld.pld_type.ptyp_desc with
              | Ptyp_constr ({txt = Lident "float32"; _}, _)
              | Ptyp_constr ({txt = Lident "float"; _}, _) ->
                  [%expr Sarek.Sarek_value.VFloat32 [%e field_access]]
              | Ptyp_constr ({txt = Lident "float64"; _}, _) ->
                  [%expr Sarek.Sarek_value.VFloat64 [%e field_access]]
              | Ptyp_constr ({txt = Lident "int32"; _}, _)
              | Ptyp_constr ({txt = Lident "int"; _}, _) ->
                  [%expr Sarek.Sarek_value.VInt32 [%e field_access]]
              | Ptyp_constr ({txt = Lident "int64"; _}, _) ->
                  [%expr Sarek.Sarek_value.VInt64 [%e field_access]]
              | Ptyp_constr ({txt = Lident "bool"; _}, _) ->
                  [%expr Sarek.Sarek_value.VBool [%e field_access]]
              | Ptyp_constr ({txt; _}, _) ->
                  let custom_type = String.concat "." (flatten_longident txt) in
                  let custom_expr = custom_descriptor_expr_of_lid ~loc txt in
                  [%expr
                    match
                      Sarek.Sarek_type_helpers.lookup_typed
                        [%e custom_expr].Spoc_core.Vector.type_id
                    with
                    | Some (module H) -> H.to_value [%e field_access]
                    | None ->
                        failwith
                          [%e
                            Ast_builder.Default.estring
                              ~loc
                              ("No helper for type " ^ custom_type)]]
              | _ ->
                  [%expr
                    failwith
                      [%e
                        Ast_builder.Default.estring
                          ~loc
                          ("Unsupported field type: " ^ field_name)]]
            in
            Ast_builder.Default.case
              ~lhs:
                (Ast_builder.Default.ppat_constant
                   ~loc
                   (Pconst_string (field_name, loc, None)))
              ~guard:None
              ~rhs:converter)
          labels
      in
      let get_field_default =
        Ast_builder.Default.case
          ~lhs:[%pat? _]
          ~guard:None
          ~rhs:
            [%expr
              failwith
                ("Unknown field in " ^ [%e full_name_expr] ^ ": " ^ field_name)]
      in
      let get_field_match =
        Ast_builder.Default.pexp_match
          ~loc
          [%expr field_name]
          (get_field_cases @ [get_field_default])
      in
      let get_field_fn =
        [%expr fun record field_name -> [%e get_field_match]]
      in

      (* Generate module and registration *)
      let helper_module_name = type_name ^ "_interp_helpers" in
      let type_lid = Ast_builder.Default.Located.lident ~loc type_name in
      let helper_module_lid =
        Ast_builder.Default.Located.lident ~loc helper_module_name
      in
      [
        Ast_builder.Default.pstr_module
          ~loc
          (Ast_builder.Default.module_binding
             ~loc
             ~name:
               (Ast_builder.Default.Located.mk ~loc (Some helper_module_name))
             ~expr:
               (Ast_builder.Default.pmod_structure
                  ~loc
	                  [
	                    [%stri
	                      type t =
	                        [%t Ast_builder.Default.ptyp_constr ~loc type_lid []]];
	                    [%stri
	                      let type_id =
	                        [%e
	                          Ast_builder.Default.evar
	                            ~loc
	                            (type_name ^ "_custom")]
	                          .Spoc_core.Vector.type_id];
	                    [%stri let from_values = [%e from_values_fn]];
                    [%stri let to_values = [%e to_values_fn]];
                    [%stri let get_field = [%e get_field_fn]];
                    [%stri
                      let to_value record =
                        Sarek.Sarek_value.VRecord
                          ([%e full_name_expr], to_values record)];
                    [%stri
                      let from_value = function
                        | Sarek.Sarek_value.VRecord (_, fields) ->
                            from_values fields
                        | _ ->
                            failwith
                              [%e
                                Ast_builder.Default.estring
                                  ~loc
                                  ("Expected " ^ type_name ^ " VRecord")]];
                  ]));
        [%stri
          let () =
            let module H =
              [%m
              Ast_builder.Default.pmod_ident ~loc helper_module_lid]
            in
            Sarek.Sarek_type_helpers.register
              [%e full_name_expr]
              (Sarek.Sarek_type_helpers.AnyHelpers (module H))];
      ]
  | _ ->
      (* Only generate helpers for records *)
      []

(** Generate runtime registration code for a type. The PPX emits calls to
    Sarek_registry at module initialization time so type info is available to
    codegen (record fields, variants, sizes). *)
let generate_type_registration ~loc (td : type_declaration) :
    structure_item list =
  let type_name = td.ptype_name.txt in
  (* Get module name from file path for fully qualified name *)
  let module_name = module_name_of_loc loc in
  let full_name = module_name ^ "." ^ type_name in
  let full_name_expr = Ast_builder.Default.estring ~loc full_name in

  match td.ptype_kind with
  | Ptype_record labels ->
      (* Build field info list *)
      let field_exprs =
        List.map
          (fun (ld : label_declaration) ->
            let fname = Ast_builder.Default.estring ~loc ld.pld_name.txt in
            let ftype =
              Ast_builder.Default.estring
                ~loc
                (type_name_of_core_type ld.pld_type)
            in
            let fmut =
              Ast_builder.Default.ebool ~loc (ld.pld_mutable = Mutable)
            in
            [%expr
              {
                Sarek_registry.field_name = [%e fname];
                field_type = [%e ftype];
                field_mutable = [%e fmut];
              }])
          labels
      in
      let fields_list = Ast_builder.Default.elist ~loc field_exprs in
      let size = calc_type_size labels in
      let size_expr = Ast_builder.Default.eint ~loc size in
      [
        [%stri
          let () =
            Sarek_registry.register_record
              [%e full_name_expr]
              ~fields:[%e fields_list]
              ~size:[%e size_expr]];
      ]
  | Ptype_variant constrs ->
      (* Build constructor info list *)
      let ctor_exprs =
        List.map
          (fun (cd : constructor_declaration) ->
            let cname = Ast_builder.Default.estring ~loc cd.pcd_name.txt in
            let carg =
              match cd.pcd_args with
              | Pcstr_tuple [] -> [%expr None]
              | Pcstr_tuple [t] ->
                  let tname =
                    Ast_builder.Default.estring ~loc (type_name_of_core_type t)
                  in
                  [%expr Some [%e tname]]
              | Pcstr_tuple _ -> [%expr Some "tuple"] (* Simplified for now *)
              | Pcstr_record _ -> [%expr Some "record"]
              (* Inline record *)
            in
            [%expr
              {Sarek_registry.ctor_name = [%e cname]; ctor_arg_type = [%e carg]}])
          constrs
      in
      let ctors_list = Ast_builder.Default.elist ~loc ctor_exprs in
      [
        [%stri
          let () =
            Sarek_registry.register_variant
              [%e full_name_expr]
              ~constructors:[%e ctors_list]];
      ]
  | _ -> []

(* Context-free rule to capture [@@sarek.type] before kernel expansion *)
let sarek_type_rule =
  Context_free.Rule.attr_str_type_decl
    sarek_type_attr
    (fun ~ctxt _rec_flag decls payloads ->
      let loc = Expansion_context.Deriver.derived_item_loc ctxt in
      let generated =
        List.concat_map
          (fun (td, payload) ->
            match payload with
            | Some () ->
                register_sarek_type_decl ~loc:td.ptype_loc td ;
                (* Generate field accessor functions for compile-time validation *)
                let accessors = generate_field_accessors ~loc td in
                (* Generate <name>_custom value for SPOC Vector.Custom *)
                let custom_val = generate_custom_value ~loc td in
                (* Generate <name>_custom value for Spoc_core.Vector.custom_type *)
                let custom_ir_val = generate_custom_value ~loc td in
                (* Generate runtime registration code for cross-module composability.
                   This follows the ppx_deriving pattern: the PPX generates OCaml code
                   that registers the type at module initialization time. When another
                   module depends on this library, the registration runs before any
                   kernels are JIT-compiled, making the type info available. *)
                let registration = generate_type_registration ~loc td in
                (* Generate interpreter helpers for type-safe value conversion. *)
                let interp_helpers = generate_interp_helpers ~loc td in
                accessors @ custom_val @ custom_ir_val @ registration
                @ interp_helpers
            | None -> [])
          (List.combine decls payloads)
      in
      generated)

let sarek_type_private_rule =
  Context_free.Rule.attr_str_type_decl
    sarek_type_private_attr
    (fun ~ctxt:_ _rec_flag decls payloads ->
      List.iter2
        (fun td payload ->
          match payload with
          | Some () -> register_sarek_type_decl ~loc:td.ptype_loc td
          | None -> ())
        decls
        payloads ;
      [])

(* NOTE: %sarek_intrinsic and %sarek_extend are handled by sarek_ppx_intrinsic.
   This allows breaking the circular dependency between sarek_ppx and sarek_stdlib.
   sarek_stdlib uses sarek_ppx_intrinsic, and sarek_ppx links sarek_stdlib. *)

(** The main kernel expansion function *)
let expand_kernel ~ctxt payload : expression =
  let loc = Expansion_context.Extension.extension_point_loc ctxt in
  Sarek_debug.log_to_file
    (Printf.sprintf "=== expand_kernel: %s ===" loc.loc_start.pos_fname) ;
  Sarek_debug.log "expand_kernel: %s" loc.loc_start.pos_fname ;

  try
    (* Scan the current file for [@sarek.module] and [@@sarek.type] declarations.
       This is needed because the impl pass (process_structure_for_module_items)
       runs AFTER extensions are expanded, so we need to scan now. *)
    let current_file = loc.loc_start.pos_fname in
    if Sys.file_exists current_file then scan_file_for_sarek_types current_file ;
    (* Types and module items registered in the current compilation unit. *)
    let pre_types = dedup_tdecls !registered_types in
    let local_mods = dedup_mods !registered_mods in
    (* Also include module items from the registry (populated by linked libraries) *)
    let registry_mods =
      List.map
        (fun (info : Sarek_ppx_registry.module_item_info) -> info.mi_item)
        (Sarek_ppx_registry.all_module_items ())
    in
    let pre_mods = dedup_mods (local_mods @ registry_mods) in
    Sarek_debug.log
      "pre_mods count=%d (local=%d, registry=%d)"
      (List.length pre_mods)
      (List.length local_mods)
      (List.length registry_mods) ;
    if Sarek_debug.enabled then begin
      Sarek_debug.log "  local_mods:" ;
      List.iter
        (fun m ->
          match m with
          | Sarek_ast.MFun (name, _, _, _) -> Sarek_debug.log "    fun %s" name
          | Sarek_ast.MConst (name, _, _) -> Sarek_debug.log "    const %s" name)
        local_mods ;
      Sarek_debug.log "  registry_mods:" ;
      List.iter
        (fun m ->
          match m with
          | Sarek_ast.MFun (name, _, _, _) -> Sarek_debug.log "    fun %s" name
          | Sarek_ast.MConst (name, _, _) -> Sarek_debug.log "    const %s" name)
        registry_mods
    end ;
    (* 1. Parse the PPX payload to Sarek AST *)
    Sarek_debug.log_enter "parse_payload" ;
    let ast = Sarek_parse.parse_payload payload in
    Sarek_debug.log_exit "parse_payload" ;
    let ast =
      {
        ast with
        Sarek_ast.kern_types = pre_types @ ast.kern_types;
        (* Merge module items from registry and local scope into the kernel. *)
        kern_module_items = pre_mods @ ast.kern_module_items;
        (* External items are prepended first, then inline items from payload.
           Track how many are external so native code gen can skip them. *)
        kern_external_item_count = List.length pre_mods;
      }
    in
    if Sarek_debug.enabled then begin
      Sarek_debug.log
        "  merged module_items count=%d"
        (List.length ast.kern_module_items) ;
      List.iter
        (fun m ->
          match m with
          | Sarek_ast.MFun (name, _, _, _) -> Sarek_debug.log "    fun %s" name
          | Sarek_ast.MConst (name, _, _) -> Sarek_debug.log "    const %s" name)
        ast.kern_module_items
    end ;

    (* 2. Set up the typing environment with stdlib *)
    let env = Sarek_env.(empty |> with_stdlib) in

    (* 3. Type inference *)
    Sarek_debug.log_to_file "  step 3: type inference start" ;
    Sarek_debug.log_enter "infer_kernel" ;
    match Sarek_typer.infer_kernel env ast with
    | Error errors ->
        (* Report the first error with location - this raises an exception *)
        Sarek_error.report_errors errors ;
        (* Unreachable: report_errors raises Location.Error *)
        failwith "Unreachable: report_errors should have raised an exception"
    | Ok tkernel -> (
        Sarek_debug.log_to_file "  step 3: type inference done" ;
        Sarek_debug.log_exit "infer_kernel" ;
        (* 4. Convergence analysis - check barrier safety *)
        Sarek_debug.log_to_file "  step 4: convergence check start" ;
        match Sarek_convergence.check_kernel tkernel with
        | Error (err :: _) ->
            (* Raise as Sarek_error to be caught by the handler below *)
            raise (Sarek_error.Sarek_error err)
        | Error [] | Ok () ->
            Sarek_debug.log_to_file "  step 4: convergence check done" ;
            (* 5. Monomorphization pass - specialize polymorphic functions *)
            Sarek_debug.log_to_file "  step 5: monomorphization start" ;
            Sarek_debug.log_enter "monomorphize" ;
            let tkernel = Sarek_mono.monomorphize tkernel in
            Sarek_debug.log_to_file "  step 5: monomorphization done" ;
            Sarek_debug.log_exit "monomorphize" ;

            (* 6. Tail recursion elimination pass (for GPU code) *)
            (* Keep original kernel for native OCaml which handles recursion *)
            let native_kernel = tkernel in
            Sarek_debug.log_to_file "  step 6: tailrec transform start" ;
            Sarek_debug.log_enter "transform_kernel" ;
            let tkernel = Sarek_tailrec.transform_kernel tkernel in
            Sarek_debug.log_to_file "  step 6: tailrec transform done" ;
            Sarek_debug.log_exit "transform_kernel" ;

            (* 7. Lower to Sarek_ir *)
            let kern_name = Option.value ~default:"anon" tkernel.tkern_name in
            Sarek_debug.log_to_file
              (Printf.sprintf "[%s] step 7: IR lowering start" kern_name) ;
            Sarek_debug.log_enter "lower_kernel" ;
            let t0 = Unix.gettimeofday () in
            let kernel, constructors =
              try Sarek_lower_ir.lower_kernel tkernel
              with e ->
                Sarek_debug.log_to_file
                  (Printf.sprintf
                     "[%s] step 7: IR lowering failed: %s"
                     kern_name
                     (Printexc.to_string e)) ;
                Sarek_debug.log_exit "lower_kernel (failed)" ;
                Location.raise_errorf
                  ~loc
                  "Kernel lowering failed: %s"
                  (Printexc.to_string e)
            in
            let t1 = Unix.gettimeofday () in
            Sarek_debug.log_to_file
              (Printf.sprintf
                 "[%s] step 7: IR lowering done (%.3fs)"
                 kern_name
                 (t1 -. t0)) ;
            Sarek_debug.log_exit "lower_kernel" ;

            (* 8. Quote the IR back to OCaml *)
            Sarek_debug.log_to_file
              (Printf.sprintf "[%s] step 8: quote start" kern_name) ;
            let t0 = Unix.gettimeofday () in
            let result =
              Sarek_quote.quote_kernel
                ~loc
                ~native_kernel
                ~ir_opt:kernel
                ~constructors
                tkernel
            in
            let t1 = Unix.gettimeofday () in
            Sarek_debug.log_to_file
              (Printf.sprintf
                 "[%s] step 8: quote done (%.3fs)"
                 kern_name
                 (t1 -. t0)) ;
            result)
  with
  | Sarek_parse.Parse_error_exn (msg, ploc) ->
      Location.raise_errorf ~loc:ploc "Sarek parse error: %s" msg
  | Sarek_error.Sarek_error e ->
      let eloc = Sarek_ast.loc_to_ppxlib (Sarek_error.error_loc e) in
      Location.raise_errorf ~loc:eloc "%a" Sarek_error.pp_error e
  | e ->
      Location.raise_errorf
        ~loc
        "Sarek internal error: %s"
        (Printexc.to_string e)

(** The [%kernel ...] extension for expressions *)
let kernel_extension =
  Extension.V3.declare
    "kernel"
    Extension.Context.expression
    Ast_pattern.(single_expr_payload __)
    expand_kernel

(* Register top-level Sarek type declarations *)
let expand_sarek_type ~ctxt payload =
  let loc = Expansion_context.Extension.extension_point_loc ctxt in
  match payload with
  | PStr ([{pstr_desc = Pstr_type (_rf, [tdecl]); _}] as items) ->
      let ctors =
        Sarek_lower.constructor_strings_of_core_type_decl ~loc tdecl
      in
      let register =
        [%stri
          let () =
            List.iter Sarek.Kirc_types.register_constructor_string [%e ctors]]
      in
      items @ [register]
  | _ ->
      Location.raise_errorf
        ~loc
        "%%sarek.type expects a single type declaration"

let sarek_type_extension = ()

(** Register sarek.module bindings on any structure we process, so libraries can
    publish module items for use in kernels.

    This generates registration code that runs at module initialization time,
    registering the items in Sarek_ppx_registry. This allows cross-module
    references: a library can define [@sarek.module] items and link with the PPX
    so they become available to kernels in other compilation units. *)
let process_structure_for_module_items (str : structure) : structure =
  let process_vb ~is_rec vb =
    let has_attr name a = String.equal a.attr_name.txt name in
    let is_private =
      List.exists (has_attr "sarek.module_private") vb.pvb_attributes
    in
    if List.exists (has_attr "sarek.module") vb.pvb_attributes || is_private
    then (
      let loc = vb.pvb_loc in
      let name =
        match Sarek_parse.extract_name_from_pattern vb.pvb_pat with
        | Some n -> n
        | None ->
            Location.raise_errorf
              ~loc:vb.pvb_pat.ppat_loc
              "Expected variable name"
      in
      let ty = Sarek_parse.extract_type_from_pattern vb.pvb_pat in
      let item =
        match Sarek_parse.collect_fun_params vb.pvb_expr with
        | params, Some (Fun_body body_expr) when params <> [] ->
            let params =
              List.map
                (fun p ->
                  Sarek_parse.extract_param_from_pattern
                    (Sarek_parse.pattern_of_param p))
                params
            in
            let body = Sarek_parse.parse_expression body_expr in
            Sarek_ast.MFun (name, is_rec, params, body)
        | _, Some (Fun_cases _) ->
            Location.raise_errorf
              ~loc:vb.pvb_expr.pexp_loc
              "Pattern-matching functions are not supported for [@sarek.module]"
        | _ ->
            let value = Sarek_parse.parse_expression vb.pvb_expr in
            let ty =
              match ty with
              | Some t -> t
              | None ->
                  Location.raise_errorf
                    ~loc:vb.pvb_pat.ppat_loc
                    "[@sarek.module] constants require a type annotation"
            in
            Sarek_ast.MConst (name, ty, value)
      in
      (* Register locally for this compilation unit *)
      register_sarek_module_item ~loc item ;
      (* For non-private items, generate registration code for cross-module use *)
      if is_private then None
      else
        let module_name = module_name_of_loc loc in
        let name_str = Ast_builder.Default.estring ~loc name in
        let module_name_str = Ast_builder.Default.estring ~loc module_name in
        let item_expr = Sarek_quote.quote_sarek_module_item ~loc item in
        Some
          [%stri
            let () =
              Sarek_ppx_lib.Sarek_ppx_registry.register_module_item
                (Sarek_ppx_lib.Sarek_ppx_registry.make_module_item_info
                   ~name:[%e name_str]
                   ~module_name:[%e module_name_str]
                   ~item:[%e item_expr])])
    else None
  in
  let extra_items =
    List.filter_map
      (fun item ->
        match item.pstr_desc with
        | Pstr_value (rf, vbs) ->
            let is_rec = rf = Recursive in
            let registrations = List.filter_map (process_vb ~is_rec) vbs in
            if registrations = [] then None else Some registrations
        | _ -> None)
      str
    |> List.flatten
  in
  str @ extra_items

(** [%sarek_include "path/to/file.ml"] - Include types and module items from
    another file.

    This scans the specified file for [@@sarek.type] and [@sarek.module]
    declarations and registers them for use in kernels in the current file. The
    path is relative to the current file's directory.

    Usage: [%sarek_include "registered_defs.ml"]

    let kernel =
    [%kernel fun ... -> let open Registered_defs in ... use types and functions
     from registered_defs.ml ... ] *)
let expand_sarek_include ~ctxt payload =
  let loc = Expansion_context.Extension.extension_point_loc ctxt in
  (* Extract path string from various payload forms *)
  let extract_path () =
    match payload with
    (* [%sarek_include "path"] *)
    | PStr
        [
          {
            pstr_desc =
              Pstr_eval
                ({pexp_desc = Pexp_constant (Pconst_string (path, _, _)); _}, _);
            _;
          };
        ] ->
        Some path
    (* let%sarek_include _ = "path" *)
    | PStr
        [
          {
            pstr_desc =
              Pstr_value
                ( _,
                  [
                    {
                      pvb_expr =
                        {
                          pexp_desc = Pexp_constant (Pconst_string (path, _, _));
                          _;
                        };
                      _;
                    };
                  ] );
            _;
          };
        ] ->
        Some path
    | _ -> None
  in
  match extract_path () with
  | Some path ->
      (* Resolve path relative to current file *)
      let current_file = loc.loc_start.pos_fname in
      let dir0 = Filename.dirname current_file in
      let dir =
        if Filename.is_relative dir0 then Filename.concat (Sys.getcwd ()) dir0
        else dir0
      in
      let rec find_repo_root d =
        let marker = Filename.concat d "dune-project" in
        if Sys.file_exists marker then Some d
        else
          let parent = Filename.dirname d in
          if String.equal d parent then None else find_repo_root parent
      in
      let rec find_build_root d =
        let parent = Filename.dirname d in
        if String.equal d parent then None
        else if String.equal (Filename.basename d) "_build" then Some parent
        else find_build_root parent
      in
      let resolve_relative base_dir p =
        if Filename.is_relative p then Filename.concat base_dir p else p
      in
      let full_path = resolve_relative dir path in
      (* When dune runs the PPX from _build/... the source path can be rewritten.
         Collect plausible candidates and pick the first that exists. *)
      let parts = String.split_on_char '/' full_path in
      let rec drop_build = function
        | "_build" :: _ctx :: rest -> Some (String.concat "/" rest)
        | _ :: tl -> drop_build tl
        | [] -> None
      in
      let build_root = find_build_root dir in
      let repo_root = find_repo_root dir in
      let candidates =
        List.filter_map
          (fun p -> p)
          [
            Some full_path;
            (* Strip _build/<ctx>/ prefix if present *)
            (match drop_build parts with
            | Some rel -> (
                match build_root with
                | Some root -> Some (Filename.concat root rel)
                | None -> Some (Filename.concat (Sys.getcwd ()) rel))
            | None -> None);
            (* Same drop_build, but anchored at repository root if found *)
            (match drop_build parts with
            | Some rel -> (
                match repo_root with
                | Some root -> Some (Filename.concat root rel)
                | None -> None)
            | None -> None);
            (* If still not found, try resolving relative to the repo root *)
            (match build_root with
            | Some root when Filename.is_relative path ->
                Some (Filename.concat root path)
            | _ -> None);
            (* Fallback to cwd *)
            (if Filename.is_relative path then
               Some (Filename.concat (Sys.getcwd ()) path)
             else None);
          ]
      in
      let full_path =
        match List.find_opt Sys.file_exists candidates with
        | Some p -> p
        | None -> full_path
      in
      if Sarek_debug.enabled then
        Sarek_debug.log
          "sarek_include: scanning %s (exists=%b)"
          full_path
          (Sys.file_exists full_path) ;
      (* Scan the file for types and module items *)
      (try scan_file_for_sarek_types full_path
       with e ->
         Location.raise_errorf
           ~loc
           "%%sarek_include: failed to scan %s: %s"
           full_path
           (Printexc.to_string e)) ;
      (* Return empty structure item - the side effect is registration *)
      [%stri let () = ()]
  | None ->
      Location.raise_errorf
        ~loc
        "%%sarek_include expects a string path, e.g. [%%sarek_include \
         \"file.ml\"]"

let sarek_include_extension =
  Extension.V3.declare
    "sarek_include"
    Extension.Context.structure_item
    Ast_pattern.(pstr __)
    (fun ~ctxt payload -> expand_sarek_include ~ctxt (PStr payload))

(** Register the transformation *)
let () =
  let rules =
    [
      sarek_type_rule;
      sarek_type_private_rule;
      Context_free.Rule.extension kernel_extension;
      Context_free.Rule.extension sarek_include_extension;
      (* NOTE: %sarek_intrinsic and %sarek_extend are handled by sarek_ppx_intrinsic *)
    ]
  in
  Driver.register_transformation
    ~rules
    ~impl:process_structure_for_module_items
    "sarek_ppx"
