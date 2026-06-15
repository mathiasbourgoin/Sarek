(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX Registry - Extensible intrinsic/type registration for PPX
 *
 * This module provides a registry that gets populated at PPX load time.
 * When sarek_ppx links sarek_stdlib, the stdlib's module initialization
 * registers types and intrinsics here. The kernel PPX then queries this
 * registry instead of using hardcoded definitions.
 *
 * This enables ppx_deriving-style extensibility: any library can register
 * its own types and intrinsics by using %sarek_intrinsic and linking with
 * the PPX.
 ******************************************************************************)

open Sarek_types

(** Type information for registered types (like float32, int64, etc.) *)
type type_info = {
  ti_name : string;  (** Type name, e.g., "float32" *)
  ti_size : int;  (** Size in bytes *)
  ti_sarek_type : typ;  (** Sarek type representation *)
}

(** Intrinsic function information *)
type intrinsic_info = {
  ii_name : string;  (** Function name, e.g., "sin" *)
  ii_qualified_name : string;  (** Qualified name, e.g., "Float32.sin" *)
  ii_type : typ;  (** Type signature *)
  ii_module : string list;  (** Module path, e.g., ["Sarek_stdlib"; "Float32"] *)
}

(** Intrinsic constant information (like thread_idx_x) *)
type const_info = {
  ci_name : string;  (** Constant name *)
  ci_qualified_name : string;  (** Qualified name *)
  ci_type : typ;  (** Type *)
  ci_module : string list;  (** Module path *)
}

(** Module item information for [@sarek.module] functions and constants. Unlike
    intrinsics, these are user-defined functions that get inlined into the
    generated GPU code. *)
type module_item_info = {
  mi_name : string;  (** Function/constant name *)
  mi_qualified_name : string;
      (** Qualified name, e.g., "Registered_defs.add_vec" *)
  mi_module : string;  (** Module name, e.g., "Registered_defs" *)
  mi_item : Sarek_ast.module_item;  (** The parsed AST item (MFun or MConst) *)
}

(** Record type information for [@sarek.type] records *)
type record_type_info = {
  rti_name : string;  (** Type name *)
  rti_qualified_name : string;  (** Qualified name *)
  rti_module : string;  (** Module name *)
  rti_decl : Sarek_ast.type_decl;  (** The type declaration *)
}

(** The registries - populated at PPX load time *)
let types : (string, type_info) Hashtbl.t = Hashtbl.create 16

let intrinsics : (string, intrinsic_info) Hashtbl.t = Hashtbl.create 64

let consts : (string, const_info) Hashtbl.t = Hashtbl.create 32

let module_items : (string, module_item_info) Hashtbl.t = Hashtbl.create 32

let record_types : (string, record_type_info) Hashtbl.t = Hashtbl.create 32

(** Register a type *)
let register_type (info : type_info) : unit =
  Hashtbl.replace types info.ti_name info

(** Register an intrinsic function *)
let register_intrinsic (info : intrinsic_info) : unit =
  (* Register under both short name and qualified name *)
  Hashtbl.replace intrinsics info.ii_name info ;
  Hashtbl.replace intrinsics info.ii_qualified_name info

(** Register an intrinsic constant *)
let register_const (info : const_info) : unit =
  Hashtbl.replace consts info.ci_name info ;
  Hashtbl.replace consts info.ci_qualified_name info

(** Register a module item ([@sarek.module] function or constant) *)
let register_module_item (info : module_item_info) : unit =
  Hashtbl.replace module_items info.mi_name info ;
  Hashtbl.replace module_items info.mi_qualified_name info

(** Register a record type ([@sarek.type]) *)
let register_record_type (info : record_type_info) : unit =
  Hashtbl.replace record_types info.rti_name info ;
  Hashtbl.replace record_types info.rti_qualified_name info

(** Find a type by name *)
let find_type (name : string) : type_info option = Hashtbl.find_opt types name

(** Find an intrinsic by name (short or qualified) *)
let find_intrinsic (name : string) : intrinsic_info option =
  Hashtbl.find_opt intrinsics name

(** Find a constant by name (short or qualified) *)
let find_const (name : string) : const_info option =
  Hashtbl.find_opt consts name

(** Check if a name is a registered intrinsic *)
let is_intrinsic (name : string) : bool = Hashtbl.mem intrinsics name

(** Check if a name is a registered constant *)
let is_const (name : string) : bool = Hashtbl.mem consts name

(** Find a module item by name (short or qualified) *)
let find_module_item (name : string) : module_item_info option =
  Hashtbl.find_opt module_items name

(** Find a record type by name (short or qualified) *)
let find_record_type (name : string) : record_type_info option =
  Hashtbl.find_opt record_types name

(** Check if a name is a registered module item *)
let is_module_item (name : string) : bool = Hashtbl.mem module_items name

(** Check if a name is a registered record type *)
let is_record_type (name : string) : bool = Hashtbl.mem record_types name

(** Get all registered types *)
let all_types () : type_info list =
  Hashtbl.fold (fun _ v acc -> v :: acc) types []

(** Get all registered intrinsics *)
let all_intrinsics () : intrinsic_info list =
  (* Deduplicate since we register under multiple names *)
  let seen = Hashtbl.create 64 in
  Hashtbl.fold
    (fun _ v acc ->
      if Hashtbl.mem seen v.ii_qualified_name then acc
      else begin
        Hashtbl.add seen v.ii_qualified_name () ;
        v :: acc
      end)
    intrinsics
    []

(** Get all registered constants *)
let all_consts () : const_info list =
  let seen = Hashtbl.create 32 in
  Hashtbl.fold
    (fun _ v acc ->
      if Hashtbl.mem seen v.ci_qualified_name then acc
      else begin
        Hashtbl.add seen v.ci_qualified_name () ;
        v :: acc
      end)
    consts
    []

(** Get all registered module items *)
let all_module_items () : module_item_info list =
  let seen = Hashtbl.create 32 in
  Hashtbl.fold
    (fun _ v acc ->
      if Hashtbl.mem seen v.mi_qualified_name then acc
      else begin
        Hashtbl.add seen v.mi_qualified_name () ;
        v :: acc
      end)
    module_items
    []

(** Get all registered record types *)
let all_record_types () : record_type_info list =
  let seen = Hashtbl.create 32 in
  Hashtbl.fold
    (fun _ v acc ->
      if Hashtbl.mem seen v.rti_qualified_name then acc
      else begin
        Hashtbl.add seen v.rti_qualified_name () ;
        v :: acc
      end)
    record_types
    []

(** Helper: create type_info for a numeric type *)
let make_type_info ~name ~size ~sarek_type : type_info =
  {ti_name = name; ti_size = size; ti_sarek_type = sarek_type}

(** Helper: create intrinsic_info *)
let make_intrinsic_info ~name ~module_path ~typ : intrinsic_info =
  let qualified = String.concat "." (module_path @ [name]) in
  {
    ii_name = name;
    ii_qualified_name = qualified;
    ii_type = typ;
    ii_module = module_path;
  }

(** Helper: create const_info *)
let make_const_info ~name ~module_path ~typ : const_info =
  let qualified = String.concat "." (module_path @ [name]) in
  {
    ci_name = name;
    ci_qualified_name = qualified;
    ci_type = typ;
    ci_module = module_path;
  }

(** Helper: create module_item_info *)
let make_module_item_info ~name ~module_name ~item : module_item_info =
  let qualified = module_name ^ "." ^ name in
  {
    mi_name = name;
    mi_qualified_name = qualified;
    mi_module = module_name;
    mi_item = item;
  }

(** Helper: create record_type_info *)
let make_record_type_info ~name ~module_name ~decl : record_type_info =
  let qualified = module_name ^ "." ^ name in
  {
    rti_name = name;
    rti_qualified_name = qualified;
    rti_module = module_name;
    rti_decl = decl;
  }

(** Debug: print registered items *)
let debug_print () =
  Printf.eprintf "=== Sarek PPX Registry ===\n" ;
  Printf.eprintf "Types: %d\n" (Hashtbl.length types) ;
  Hashtbl.iter (fun k _ -> Printf.eprintf "  - %s\n" k) types ;
  Printf.eprintf "Intrinsics: %d\n" (List.length (all_intrinsics ())) ;
  List.iter
    (fun i ->
      Printf.eprintf
        "  - %s : %s\n"
        i.ii_qualified_name
        (typ_to_string i.ii_type))
    (all_intrinsics ()) ;
  Printf.eprintf "Constants: %d\n" (List.length (all_consts ())) ;
  List.iter
    (fun c -> Printf.eprintf "  - %s\n" c.ci_qualified_name)
    (all_consts ()) ;
  Printf.eprintf "Module items: %d\n" (List.length (all_module_items ())) ;
  List.iter
    (fun m -> Printf.eprintf "  - %s\n" m.mi_qualified_name)
    (all_module_items ()) ;
  Printf.eprintf "Record types: %d\n" (List.length (all_record_types ())) ;
  List.iter
    (fun r -> Printf.eprintf "  - %s\n" r.rti_qualified_name)
    (all_record_types ()) ;
  Printf.eprintf "==========================\n"
