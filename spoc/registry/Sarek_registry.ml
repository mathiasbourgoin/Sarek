(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Runtime Registry
 *
 * This module implements a runtime registry for Sarek types, intrinsics, and
 * user-defined functions. It enables cross-module composability following the
 * same pattern as ppx_deriving:
 *
 * DESIGN RATIONALE:
 * -----------------
 * Instead of using a file-based registry that the PPX reads at compile time,
 * we follow the ppx_deriving approach where:
 *
 * 1. The PPX generates OCaml code that registers types/functions at module
 *    initialization time (when the library is linked).
 *
 * 2. Cross-module references work because:
 *    - Dune ensures libraries are compiled in dependency order
 *    - When a library is linked, its registration code runs
 *    - By the time JIT compilation happens, all types are registered
 *
 * 3. Compile-time type checking uses Sarek_env.with_stdlib for core types.
 *    For user-defined types from other modules, the PPX trusts the OCaml
 *    type checker and defers detailed validation to runtime.
 *
 * USAGE:
 * ------
 * - [@@sarek.type] on a record/variant generates registration code
 * - [%sarek_intrinsic] generates intrinsic registration + device function
 * - [%sarek_extend] chains a new device function to an existing intrinsic
 *
 * At JIT time, the code generator queries this registry to get device code
 * for types and functions.
 *
 * See also: Sarek_ppx.ml for the PPX implementation.
 ******************************************************************************)

(** Information about a primitive/intrinsic type (float32, int64, etc.) *)
type type_info = {
  ti_name : string;
  ti_device : string -> string;
  ti_size : int; (* bytes *)
}

(** Information about a record field *)
type field_info = {
  field_name : string;
  field_type : string;
  field_mutable : bool;
}

(** Information about a record type (user-defined via [@@sarek.type]) *)
type record_info = {
  ri_name : string; (* Full name including module path *)
  ri_fields : field_info list;
  ri_size : int; (* Total size in bytes *)
}

(** Information about a variant constructor *)
type constructor_info = {ctor_name : string; ctor_arg_type : string option}

(** Information about a variant type *)
type variant_info = {vi_name : string; vi_constructors : constructor_info list}

(** Information about an intrinsic function *)
type fun_info = {
  fi_name : string;
  fi_arity : int;
  fi_device : string -> string;
  fi_arg_types : string list;
  fi_ret_type : string;
}

(** Type registry - maps type names to their info (primitives) *)
let type_registry : (string, type_info) Hashtbl.t = Hashtbl.create 32

(** Record registry - maps type names to their info (user-defined records) *)
let record_registry : (string, record_info) Hashtbl.t = Hashtbl.create 32

(** Variant registry - maps type names to their info (user-defined variants) *)
let variant_registry : (string, variant_info) Hashtbl.t = Hashtbl.create 32

(** Function registry - maps (module_path, name) to their info *)
let fun_registry : (string list * string, fun_info) Hashtbl.t =
  Hashtbl.create 64

(** Register a primitive type *)
let register_type name ~device ~size =
  Hashtbl.replace
    type_registry
    name
    {ti_name = name; ti_device = device; ti_size = size}

(** Register a record type (called by PPX-generated code for [@@sarek.type]) *)
let register_record name ~fields ~size =
  Hashtbl.replace
    record_registry
    name
    {ri_name = name; ri_fields = fields; ri_size = size}

(** Register a variant type (called by PPX-generated code for [@@sarek.type]) *)
let register_variant name ~constructors =
  Hashtbl.replace
    variant_registry
    name
    {vi_name = name; vi_constructors = constructors}

(** Register an intrinsic function *)
let register_fun ?(module_path = []) name ~arity ~device ~arg_types ~ret_type =
  Hashtbl.replace
    fun_registry
    (module_path, name)
    {
      fi_name = name;
      fi_arity = arity;
      fi_device = device;
      fi_arg_types = arg_types;
      fi_ret_type = ret_type;
    }

(** Find a primitive type by name *)
let find_type name = Hashtbl.find_opt type_registry name

(** Find a record type by name *)
let find_record name = Hashtbl.find_opt record_registry name

(** Find a variant type by name *)
let find_variant name = Hashtbl.find_opt variant_registry name

(** Find a function by name, optionally in a module *)
let find_fun ?(module_path = []) name =
  Hashtbl.find_opt fun_registry (module_path, name)

(** Check if a name is a registered primitive type *)
let is_type name = Hashtbl.mem type_registry name

(** Check if a name is a registered record type *)
let is_record name = Hashtbl.mem record_registry name

(** Check if a name is a registered variant type *)
let is_variant name = Hashtbl.mem variant_registry name

(** Check if a name is a registered function *)
let is_fun ?(module_path = []) name =
  Hashtbl.mem fun_registry (module_path, name)

(** Get device code for a type *)
let type_device_code name dev =
  match find_type name with
  | Some ti -> ti.ti_device dev
  | None -> failwith ("Unknown intrinsic type: " ^ name)

(** Get device code for a function *)
let fun_device_code ?(module_path = []) name dev =
  match find_fun ~module_path name with
  | Some fi -> fi.fi_device dev
  | None ->
      let path = String.concat "." (module_path @ [name]) in
      failwith ("Unknown intrinsic function: " ^ path)

(** Get device code template for a function, using a minimal device. This is for
    V2 IR codegens that don't have SPOC device objects. *)
let fun_device_template ?(module_path = []) name =
  match find_fun ~module_path name with
  | Some fi ->
      (* Pass "generic" as the framework string: cuda_or_opencl maps
         "generic" to the CUDA/default branch, preserving byte-identical output. *)
      Some (fi.fi_device "generic")
  | None -> None

(** Find a record by short name (last component after '.'). This handles cases
    where the registry uses qualified names like "Module.typename" but the
    custom_type uses just "typename". *)
let find_record_by_short_name short_name =
  Hashtbl.fold
    (fun full_name ri acc ->
      match acc with
      | Some _ -> acc
      | None ->
          (* Get last component of full_name *)
          let last =
            match String.rindex_opt full_name '.' with
            | Some i ->
                String.sub full_name (i + 1) (String.length full_name - i - 1)
            | None -> full_name
          in
          if last = short_name then Some ri else None)
    record_registry
    None

(** Get record field info - tries exact match first, then short name *)
let record_fields name =
  match find_record name with
  | Some ri -> ri.ri_fields
  | None -> (
      (* Try short name search *)
      match find_record_by_short_name name with
      | Some ri -> ri.ri_fields
      | None -> failwith ("Unknown record type: " ^ name))

(** Get variant constructors *)
let variant_constructors name =
  match find_variant name with
  | Some vi -> vi.vi_constructors
  | None -> failwith ("Unknown variant type: " ^ name)

(******************************************************************************
 * Register standard types
 *
 * NOTE: Primitive types like float32, int32, etc. are now registered by their
 * respective stdlib modules (Float32.ml, Int32.ml, etc.) using %sarek_intrinsic.
 * Only truly fundamental types (bool, unit) that have no stdlib module are
 * registered here.
 ******************************************************************************)

let () =
  register_type "bool" ~device:(fun _ -> "int") ~size:4 ;
  register_type "unit" ~device:(fun _ -> "void") ~size:0

(******************************************************************************
 * Helper function for device-specific code
 ******************************************************************************)

let cuda_or_opencl (framework : string) cuda_code opencl_code =
  match framework with
  | "OpenCL" -> opencl_code
  | "CUDA" | "Native" | "Interpreter" | _ -> cuda_code
(* Use CUDA syntax for CUDA, interpreter, and native *)

(* Note: All intrinsics (Float32, Float64, Int32, Int64, GPU) are defined in
   Sarek_stdlib modules and auto-register via %sarek_intrinsic when that
   library is loaded. No hardcoded registrations needed here. *)
