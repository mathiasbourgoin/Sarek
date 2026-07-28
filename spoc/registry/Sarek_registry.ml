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

(** Get the device-code template for a function on a given backend. This is for
    V2 IR codegens that don't have SPOC device objects.

    [framework] is the caller's backend tag ("CUDA", "OpenCL", "Metal", …). It
    used to be hardcoded to "generic", which [cuda_or_opencl] resolves to the
    CUDA branch — so every backend reaching this fallback got the CUDA spelling,
    and an OpenCL or Metal kernel calling e.g. [Float32.abs_float] was emitted
    as [fabsf(...)]. Neither OpenCL C nor MSL declares [fabsf]; both spell it
    [fabs]. The stdlib already declares both spellings ([dev "fabsf" "fabs"]);
    only this lookup was discarding the caller's framework.

    [?framework] defaults to "generic" so existing non-backend callers keep the
    previous behaviour. *)
let fun_device_template ?(module_path = []) ?(framework = "generic") name =
  match find_fun ~module_path name with
  | Some fi -> Some (fi.fi_device framework)
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

(** Raised when a shading-language framework asks this two-way dispatch for a
    spelling it cannot supply. See {!cuda_or_opencl}. *)
exception No_device_spelling of {framework : string; cuda : string}

let cuda_or_opencl (framework : string) cuda_code opencl_code =
  match framework with
  (* Metal joins OpenCL, not CUDA: across the whole stdlib these two branches
     differ ONLY in the CUDA `f` suffix (sinf/fabsf/…) — the operator and cast
     templates are byte-identical in both — and MSL, like OpenCL C, declares the
     unsuffixed overloads and has no `fabsf`/`sinf`. This matches the spelling
     Sarek_pure_registry already emits for Metal via its `generic_name`. *)
  | "OpenCL" | "Metal" -> opencl_code
  (* GLSL and WGSL get NEITHER branch. This dispatch has exactly two, and both
     are C-family: the wildcard below sends anything unrecognised to CUDA, so a
     shading language asking here would be handed `sinf`, `fabsf`, `powf` —
     names no GLSL or WGSL compiler declares.

     Today nothing reaches this arm: Sarek_ir_glsl and Sarek_ir_wgsl set
     `post_hook = (fun _ _ _ _ -> false)`, so on fall-through they raise
     `unknown_intrinsic` instead of consulting this registry. That inert
     post_hook is the SAFE behaviour and must stay: the three C-family backends
     wire `Dispatch.emit_registry_template` there, and doing the same for
     GLSL/WGSL — which reads like an obvious symmetry, and which an earlier
     backlog entry of mine explicitly recommended — would replace a loud refusal
     with silently-invalid shader source, on all 133 stdlib intrinsics
     registered through this helper (Float32 35, Float64 42, Gpu 21, Int32 16,
     Int64 16, Math 3).

     So this arm is a landmine guard, not a live path: it makes that mistake
     fail immediately and by name rather than emit C into a shader. It raises
     rather than returning a value because there is no correct value to return —
     a spelling for these frameworks has to be REGISTERED, not derived. The
     model to copy is Sarek_pure_registry, which dispatches on framework and
     carries the GLSL exceptions explicitly (`glsl_override_name`: fabs→abs,
     rsqrt→inversesqrt, atan2→atan); the un-suffixed OpenCL branch happens to be
     right for most GLSL builtins, which is exactly why guessing here is
     dangerous — it would be right often enough to look correct.

     This is an OCaml exception rather than a located Codegen_error because this
     module sits below codegen and must not depend on it. Reaching it is a
     programming error in the dispatcher wiring, not a user error in a kernel. *)
  | ("GLSL" | "WGSL") as fw ->
      raise (No_device_spelling {framework = fw; cuda = cuda_code})
  | "CUDA" | "Native" | "Interpreter" | _ -> cuda_code
(* Use CUDA syntax for CUDA, interpreter, and native. The wildcard also covers
   the "generic" default of [fun_device_template] and is deliberately NOT a
   refusal: "generic" is a live value with C-family semantics. Only the two
   shading languages above are refused, because only they are MEASURED to be
   mis-served by both branches. *)

let () =
  Printexc.register_printer (function
    | No_device_spelling {framework; cuda} ->
        Some
          (Printf.sprintf
             "Sarek_registry.cuda_or_opencl: no %s spelling for an intrinsic \
              registered with only CUDA/OpenCL templates (CUDA form: %S). This \
              two-way dispatch cannot serve a shading language — register a %s \
              template instead of routing %s through the FFI registry."
             framework
             cuda
             framework
             framework)
    | _ -> None)

(* Note: All intrinsics (Float32, Float64, Int32, Int64, GPU) are defined in
   Sarek_stdlib modules and auto-register via %sarek_intrinsic when that
   library is loaded. No hardcoded registrations needed here. *)
