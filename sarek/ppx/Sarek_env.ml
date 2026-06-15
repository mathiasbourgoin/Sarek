(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module defines the immutable typing environment for name resolution
 * and type checking. Replaces the global mutable hashtables from the old
 * implementation.
 *
 * Supports let-polymorphism via type schemes for local function bindings.
 ******************************************************************************)

open Sarek_types
open Sarek_scheme
module StringMap = Map.Make (String)

(** Information about a variable *)
type var_info = {
  vi_type : typ;
  vi_mutable : bool;
  vi_is_param : bool;  (** Kernel parameter? *)
  vi_index : int;  (** For parameter ordering *)
  vi_is_vec : bool;  (** Is this a vector parameter? *)
}

(** Reference to an intrinsic function or constant.

    - IntrinsicRef: Reference to a stdlib module intrinsic. The module path
      allows the PPX to generate correct references for compile-time type
      checking. For example:
    - IntrinsicRef (["Sarek_stdlib"; "Float32"], "sin") ->
      Sarek_stdlib.Float32.sin
    - IntrinsicRef (["Sarek_stdlib"; "Int32"], "add_int32") ->
      Sarek_stdlib.Int32.add_int32

    - CorePrimitiveRef: Reference to a core GPU primitive defined in
      Sarek_core_primitives. These have compile-time semantic properties
      (variance, convergence, purity) that cannot be overridden by user code.
      Device implementations are still resolved via Sarek_registry at JIT time.

    This enables extensibility: user libraries can define their own intrinsics
    via %sarek_intrinsic and the PPX will reference them correctly. *)
type intrinsic_ref =
  | IntrinsicRef of string list * string  (** module_path, function_name *)
  | CorePrimitiveRef of string  (** Core primitive name *)

(** Get the qualified name of an intrinsic ref for debugging/printing *)
let intrinsic_ref_name = function
  | IntrinsicRef (path, name) -> String.concat "." (path @ [name])
  | CorePrimitiveRef name -> "@core:" ^ name

(** Get user-friendly display name for error messages (no core prefix) *)
let intrinsic_ref_display_name = function
  | IntrinsicRef (path, name) -> String.concat "." (path @ [name])
  | CorePrimitiveRef name -> name

(** Information about an intrinsic function. Device code is resolved at JIT time
    via Sarek_registry. *)
type intrinsic_fun_info = {
  intr_type : typ;
  intr_ref : intrinsic_ref;  (** Reference to stdlib module function *)
  intr_convergence : Sarek_core_primitives.convergence option;
      (** Convergence requirement. None means NoEffect. User libraries can
          specify ConvergencePoint or WarpConvergence. *)
}

(** Information about an intrinsic constant. Device code is resolved at JIT time
    via Sarek_registry. *)
type intrinsic_const_info = {
  const_type : typ;
  const_ref : intrinsic_ref;  (** Reference to intrinsic constant *)
}

(** Information about a custom type *)
type type_info =
  | TIRecord of {
      ti_name : string;
      ti_fields : (string * typ * bool) list;  (** name, type, mutable *)
    }
  | TIVariant of {
      ti_name : string;
      ti_constrs : (string * typ option) list;
          (** constructor name, optional arg type *)
    }

(** The typing environment - immutable *)
type t = {
  vars : var_info StringMap.t;
  types : type_info StringMap.t;
  intrinsic_funs : intrinsic_fun_info StringMap.t;
  intrinsic_consts : intrinsic_const_info StringMap.t;
  constructors : (string * typ option) StringMap.t;
      (** constr -> (type_name, arg_type) *)
  fields : (string * int * typ * bool) StringMap.t;
      (** field -> (type_name, index, type, mutable) *)
  current_level : int;  (** For let-polymorphism *)
  local_funs : (string * scheme) StringMap.t;
      (** Local function definitions with polymorphic schemes *)
}

(** Empty environment *)
let empty =
  {
    vars = StringMap.empty;
    types = StringMap.empty;
    intrinsic_funs = StringMap.empty;
    intrinsic_consts = StringMap.empty;
    constructors = StringMap.empty;
    fields = StringMap.empty;
    current_level = 0;
    local_funs = StringMap.empty;
  }

(** Environment operations - all return new environments *)

let add_var name info env = {env with vars = StringMap.add name info env.vars}

let add_type name info env =
  match info with
  | TIRecord {ti_fields; ti_name; _} ->
      let env = {env with types = StringMap.add name info env.types} in
      (* Also add field mappings *)
      let _, fields =
        List.fold_left
          (fun (idx, fields) (fname, ftyp, fmut) ->
            (idx + 1, StringMap.add fname (ti_name, idx, ftyp, fmut) fields))
          (0, env.fields)
          ti_fields
      in
      {env with fields}
  | TIVariant {ti_constrs; ti_name; _} ->
      let env = {env with types = StringMap.add name info env.types} in
      (* Also add constructor mappings *)
      let constrs =
        List.fold_left
          (fun constrs (cname, carg) ->
            StringMap.add cname (ti_name, carg) constrs)
          env.constructors
          ti_constrs
      in
      {env with constructors = constrs}

let add_intrinsic_fun name info env =
  {env with intrinsic_funs = StringMap.add name info env.intrinsic_funs}

let add_intrinsic_const name info env =
  {env with intrinsic_consts = StringMap.add name info env.intrinsic_consts}

let add_local_fun name scheme env =
  {env with local_funs = StringMap.add name (name, scheme) env.local_funs}

let find_var name env = StringMap.find_opt name env.vars

let find_type name env = StringMap.find_opt name env.types

let find_intrinsic_fun name env = StringMap.find_opt name env.intrinsic_funs

let find_intrinsic_const name env = StringMap.find_opt name env.intrinsic_consts

let find_constructor name env = StringMap.find_opt name env.constructors

let find_field name env = StringMap.find_opt name env.fields

let find_local_fun name env = StringMap.find_opt name env.local_funs

(** Scope management *)
let enter_level env = {env with current_level = env.current_level + 1}

let exit_level env = {env with current_level = max 0 (env.current_level - 1)}

(** Get the short module name from a module path. E.g.,
    ["Sarek_stdlib"; "Float32"] -> "Float32" *)
let short_module_name = function [] -> "" | path -> List.hd (List.rev path)

(** Open a module: bring its bindings into scope under short names. E.g.,
    open_module ["Float32"] brings Float32.sin -> sin

    Also handles legacy aliases:
    - Std -> Gpu (backward compatibility) *)
let open_module (path : string list) env =
  (* Handle legacy module aliases for backward compatibility *)
  let module_name =
    match short_module_name path with
    | "Std" -> "Gpu" (* Std was the old name for GPU intrinsics *)
    | "Math" -> "Float32" (* Math.Float32 -> just Float32 *)
    | name -> name
  in
  if module_name = "" then env
  else
    (* Find all intrinsics in this module and add under short names.
       IMPORTANT: Preserve convergence from existing core primitive entries
       when they exist - stdlib intrinsics provide device implementations
       but core primitives define the convergence requirements. *)
    let prefix = module_name ^ "." in
    let env =
      StringMap.fold
        (fun name info env ->
          if
            String.length name > String.length prefix
            && String.sub name 0 (String.length prefix) = prefix
          then
            let short_name =
              String.sub
                name
                (String.length prefix)
                (String.length name - String.length prefix)
            in
            (* Check if there's an existing core primitive entry.
               Core primitives define the canonical type and convergence.
               Stdlib intrinsics just provide device implementations.

               IMPORTANT: Only preserve type/convergence from CorePrimitiveRef.
               When shadowing another IntrinsicRef (e.g., Float64 shadowing Float32),
               we want the new module's type to take precedence. *)
            let info =
              match StringMap.find_opt short_name env.intrinsic_funs with
              | Some existing -> (
                  match existing.intr_ref with
                  | CorePrimitiveRef _ ->
                      (* Core primitive: preserve type and convergence,
                         only update reference to use stdlib implementation. *)
                      {
                        intr_type = existing.intr_type;
                        intr_ref = info.intr_ref;
                        intr_convergence =
                          (if existing.intr_convergence <> None then
                             existing.intr_convergence
                           else info.intr_convergence);
                      }
                  | IntrinsicRef _ ->
                      (* Another stdlib intrinsic: new module shadows old completely *)
                      info)
              | None -> info
            in
            add_intrinsic_fun short_name info env
          else env)
        env.intrinsic_funs
        env
    in
    StringMap.fold
      (fun name info env ->
        if
          String.length name > String.length prefix
          && String.sub name 0 (String.length prefix) = prefix
        then
          let short_name =
            String.sub
              name
              (String.length prefix)
              (String.length name - String.length prefix)
          in
          add_intrinsic_const short_name info env
        else env)
      env.intrinsic_consts
      env

(** Create standard library environment from core primitives and the PPX
    registry.

    Step 1: Add all core primitives from Sarek_core_primitives. These have
    compile-time semantic properties (variance, convergence, purity) that the
    PPX uses for analysis. Core primitives use CorePrimitiveRef.

    Step 2: Add library intrinsics from Sarek_ppx_registry. These may shadow
    core primitives if they have the same name (which provides device
    implementations). Library intrinsics use IntrinsicRef.

    Intrinsics are registered under module-qualified names (e.g., "Float32.sin")
    to avoid ambiguity between Float32.sqrt and Float64.sqrt. Use `open_module`
    to bring a module's bindings into scope under short names.

    IMPORTANT: The caller must ensure stdlib modules are initialized before
    calling this function (e.g., via Sarek_stdlib.force_init()). This is done in
    Sarek_ppx.ml to avoid circular dependencies. *)
let with_stdlib env =
  (* Step 1: Add all core primitives first.
     These provide semantic properties (variance, convergence, purity)
     that the PPX needs for compile-time analysis. *)
  let env =
    List.fold_left
      (fun env (prim : Sarek_core_primitives.primitive) ->
        let ref = CorePrimitiveRef prim.name in
        match prim.typ with
        | TFun _ ->
            let convergence =
              if prim.convergence = Sarek_core_primitives.NoEffect then None
              else Some prim.convergence
            in
            let info =
              {
                intr_type = prim.typ;
                intr_ref = ref;
                intr_convergence = convergence;
              }
            in
            add_intrinsic_fun prim.name info env
        | _ ->
            let info = {const_type = prim.typ; const_ref = ref} in
            add_intrinsic_const prim.name info env)
      env
      Sarek_core_primitives.primitives
  in

  (* Step 2: Import all intrinsics from the PPX registry.
     These provide device implementations and may shadow core primitives. *)
  let intrinsics = Sarek_ppx_registry.all_intrinsics () in

  (* Add each intrinsic to the environment under module-qualified name.
     E.g., "Float32.sin" not just "sin". This avoids Float32/Float64 conflicts.
     Intrinsics with function types (TFun) go into intrinsic_funs.
     Intrinsics with non-function types (constants like thread_idx_x) go into
     intrinsic_consts. *)
  let env =
    List.fold_left
      (fun env (info : Sarek_ppx_registry.intrinsic_info) ->
        let ref = IntrinsicRef (info.ii_module, info.ii_name) in
        (* Use module-qualified name: e.g., "Float32.sin" *)
        let module_name = short_module_name info.ii_module in
        let qualified_name =
          if module_name = "" then info.ii_name
          else module_name ^ "." ^ info.ii_name
        in
        match info.ii_type with
        | TFun _ ->
            (* It's a function - register under qualified name *)
            let fun_info =
              {
                intr_type = info.ii_type;
                intr_ref = ref;
                intr_convergence = None;
              }
            in
            add_intrinsic_fun qualified_name fun_info env
        | _ ->
            (* It's a constant - register under qualified name *)
            let const_info = {const_type = info.ii_type; const_ref = ref} in
            add_intrinsic_const qualified_name const_info env)
      env
      intrinsics
  in
  (* Auto-open core modules - their intrinsics are fundamental
     and should be available without explicit qualification *)
  let env = open_module ["Gpu"] env in
  (* Also auto-open Float32 for backward compatibility - math functions like
     sqrt, sin, etc. were available at top level in the old system *)
  open_module ["Float32"] env

(** Lookup that checks all namespaces for an identifier *)
type lookup_result =
  | LVar of var_info
  | LIntrinsicConst of intrinsic_const_info
  | LIntrinsicFun of intrinsic_fun_info
  | LConstructor of string * typ option  (** type_name, arg_type *)
  | LLocalFun of string * scheme  (** name and type scheme (for polymorphism) *)
  | LNotFound

let lookup name env =
  match find_var name env with
  | Some info -> LVar info
  | None -> (
      match find_intrinsic_const name env with
      | Some info -> LIntrinsicConst info
      | None -> (
          match find_intrinsic_fun name env with
          | Some info -> LIntrinsicFun info
          | None -> (
              match find_constructor name env with
              | Some (type_name, arg_type) -> LConstructor (type_name, arg_type)
              | None -> (
                  match find_local_fun name env with
                  | Some (_name, scheme) -> LLocalFun (name, scheme)
                  | None -> LNotFound))))

(** Debug: print environment contents *)
let pp_env fmt env =
  Format.fprintf fmt "Variables:@." ;
  StringMap.iter
    (fun name info ->
      Format.fprintf
        fmt
        "  %s : %a (param=%b, idx=%d)@."
        name
        pp_typ
        info.vi_type
        info.vi_is_param
        info.vi_index)
    env.vars ;
  Format.fprintf fmt "Intrinsic constants:@." ;
  StringMap.iter
    (fun name info ->
      Format.fprintf
        fmt
        "  %s : %a (ref=%s)@."
        name
        pp_typ
        info.const_type
        (intrinsic_ref_name info.const_ref))
    env.intrinsic_consts ;
  Format.fprintf fmt "Intrinsic functions:@." ;
  StringMap.iter
    (fun name info ->
      Format.fprintf
        fmt
        "  %s : %a (ref=%s)@."
        name
        pp_typ
        info.intr_type
        (intrinsic_ref_name info.intr_ref))
    env.intrinsic_funs
