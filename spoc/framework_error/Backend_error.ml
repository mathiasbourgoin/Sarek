(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Backend Error Types - Shared Structured Error Handling for GPU Backends
 *
 * Provides generic structured error types for all GPU backend operations.
 * Used by: CUDA, OpenCL, Vulkan, Metal, and optionally Native/Interpreter.
 *
 * Error Categories:
 * - Code generation errors (IR to backend source translation)
 * - Runtime errors (device operations, compilation, memory)
 * - Plugin errors (unsupported operations, missing libraries)
 ******************************************************************************)

(** {1 Error Type Definitions} *)

(** Error types for backend code generation (IR → source translation) *)
type codegen_error =
  | Unknown_intrinsic of {name : string}
      (** Intrinsic function not recognized by this backend *)
  | Invalid_arg_count of {intrinsic : string; expected : int; got : int}
      (** Wrong number of arguments to intrinsic *)
  | Unsupported_construct of {construct : string; reason : string}
      (** IR construct not supported by this backend *)
  | Type_error of {expr : string; expected : string; got : string}
      (** Type mismatch in expression *)
  | Invalid_memory_space of {decl : string; space : string}
      (** Invalid memory space qualifier for declaration *)
  | Unsupported_type of {type_name : string; backend : string}
      (** Type not supported by backend (e.g., fp64 without cl_khr_fp64) *)

(** Error types for backend runtime operations *)
type runtime_error =
  | No_device_selected of {operation : string}
      (** Operation requires a device but none is set *)
  | Device_not_found of {device_id : int; max_devices : int}
      (** Device ID out of range *)
  | Compilation_failed of {source : string; log : string}
      (** Kernel compilation failed *)
  | Module_load_failed of {size : int; reason : string}
      (** Failed to load compiled module/program *)
  | Kernel_launch_failed of {kernel_name : string; reason : string}
      (** Failed to launch kernel on device *)
  | Memory_allocation_failed of {bytes : int64; reason : string}
      (** Device memory allocation failed *)
  | Memory_copy_failed of {direction : string; bytes : int; reason : string}
      (** Memory transfer between host and device failed *)
  | Context_error of {operation : string; reason : string}
      (** GPU context creation/management failed *)
  | Synchronization_failed of {reason : string}
      (** Device synchronization failed *)

(** Error types for backend plugin operations *)
type plugin_error =
  | Unsupported_source_lang of {lang : string; backend : string}
      (** Source language not supported by backend *)
  | Backend_unavailable of {reason : string}
      (** Backend not available (missing drivers, no devices, etc.) *)
  | Library_not_found of {library : string; paths : string list}
      (** Required backend library not found *)
  | Initialization_failed of {backend : string; reason : string}
      (** Backend initialization failed *)
  | Feature_not_supported of {feature : string; backend : string}
      (** Feature not supported by this backend *)

(** {1 Parameterized Error Type} *)

(** Union type for backend errors, parameterized by backend name *)
type t =
  | Codegen of {backend : string; error : codegen_error}
  | Runtime of {backend : string; error : runtime_error}
  | Plugin of {backend : string; error : plugin_error}

(** Exception wrapper for backend errors *)
exception Backend_error of t

(** {1 Error Construction Helpers} *)

(** Create codegen error for a specific backend *)
let codegen ~backend error = Codegen {backend; error}

(** Create runtime error for a specific backend *)
let runtime ~backend error = Runtime {backend; error}

(** Create plugin error for a specific backend *)
let plugin ~backend error = Plugin {backend; error}

(** {1 Codegen Error Constructors} *)

let unknown_intrinsic ~backend name =
  codegen ~backend (Unknown_intrinsic {name})

let invalid_arg_count ~backend intrinsic expected got =
  codegen ~backend (Invalid_arg_count {intrinsic; expected; got})

let unsupported_construct ~backend construct reason =
  codegen ~backend (Unsupported_construct {construct; reason})

let type_error ~backend expr expected got =
  codegen ~backend (Type_error {expr; expected; got})

let invalid_memory_space ~backend decl space =
  codegen ~backend (Invalid_memory_space {decl; space})

let unsupported_type ~backend type_name =
  codegen ~backend (Unsupported_type {type_name; backend})

(** {1 Runtime Error Constructors} *)

let no_device_selected ~backend operation =
  runtime ~backend (No_device_selected {operation})

let device_not_found ~backend device_id max_devices =
  runtime ~backend (Device_not_found {device_id; max_devices})

let compilation_failed ~backend source log =
  runtime ~backend (Compilation_failed {source; log})

let module_load_failed ~backend size reason =
  runtime ~backend (Module_load_failed {size; reason})

let kernel_launch_failed ~backend kernel_name reason =
  runtime ~backend (Kernel_launch_failed {kernel_name; reason})

let memory_allocation_failed ~backend bytes reason =
  runtime ~backend (Memory_allocation_failed {bytes; reason})

let memory_copy_failed ~backend direction bytes reason =
  runtime ~backend (Memory_copy_failed {direction; bytes; reason})

let context_error ~backend operation reason =
  runtime ~backend (Context_error {operation; reason})

let synchronization_failed ~backend reason =
  runtime ~backend (Synchronization_failed {reason})

(** {1 Plugin Error Constructors} *)

let unsupported_source_lang ~backend lang =
  plugin ~backend (Unsupported_source_lang {lang; backend})

let backend_unavailable ~backend reason =
  plugin ~backend (Backend_unavailable {reason})

let library_not_found ~backend library paths =
  plugin ~backend (Library_not_found {library; paths})

let initialization_failed ~backend reason =
  plugin ~backend (Initialization_failed {backend; reason})

let feature_not_supported ~backend feature =
  plugin ~backend (Feature_not_supported {feature; backend})

(** {1 Error Conversion and Display} *)

(** Convert error to human-readable string *)
let to_string = function
  | Codegen {backend; error} -> (
      let prefix = Printf.sprintf "[%s Codegen]" backend in
      match error with
      | Unknown_intrinsic {name} ->
          Printf.sprintf "%s Unknown intrinsic: %s" prefix name
      | Invalid_arg_count {intrinsic; expected; got} ->
          Printf.sprintf
            "%s Intrinsic '%s' expects %d argument%s but got %d"
            prefix
            intrinsic
            expected
            (if expected = 1 then "" else "s")
            got
      | Unsupported_construct {construct; reason} ->
          Printf.sprintf
            "%s Unsupported construct '%s': %s"
            prefix
            construct
            reason
      | Type_error {expr; expected; got} ->
          Printf.sprintf
            "%s Type error in '%s': expected %s but got %s"
            prefix
            expr
            expected
            got
      | Invalid_memory_space {decl; space} ->
          Printf.sprintf
            "%s Invalid memory space '%s' for: %s"
            prefix
            space
            decl
      | Unsupported_type {type_name; backend = _} ->
          Printf.sprintf "%s Type not supported: %s" prefix type_name)
  | Runtime {backend; error} -> (
      let prefix = Printf.sprintf "[%s Runtime]" backend in
      match error with
      | No_device_selected {operation} ->
          Printf.sprintf
            "%s Operation '%s' requires a device but none is selected"
            prefix
            operation
      | Device_not_found {device_id; max_devices} ->
          Printf.sprintf
            "%s Device ID %d not found (available: 0-%d)"
            prefix
            device_id
            (max_devices - 1)
      | Compilation_failed {source; log} ->
          let preview =
            if String.length source > 100 then String.sub source 0 100 ^ "..."
            else source
          in
          Printf.sprintf
            "%s Compilation failed for:\n%s\n\nCompiler log:\n%s"
            prefix
            preview
            log
      | Module_load_failed {size; reason} ->
          Printf.sprintf
            "%s Failed to load compiled module (%d bytes): %s"
            prefix
            size
            reason
      | Kernel_launch_failed {kernel_name; reason} ->
          Printf.sprintf
            "%s Failed to launch kernel '%s': %s"
            prefix
            kernel_name
            reason
      | Memory_allocation_failed {bytes; reason} ->
          Printf.sprintf
            "%s Memory allocation failed (%Ld bytes): %s"
            prefix
            bytes
            reason
      | Memory_copy_failed {direction; bytes; reason} ->
          Printf.sprintf
            "%s Memory copy failed (%s, %d bytes): %s"
            prefix
            direction
            bytes
            reason
      | Context_error {operation; reason} ->
          Printf.sprintf
            "%s Context error during %s: %s"
            prefix
            operation
            reason
      | Synchronization_failed {reason} ->
          Printf.sprintf "%s Synchronization failed: %s" prefix reason)
  | Plugin {backend; error} -> (
      let prefix = Printf.sprintf "[%s Plugin]" backend in
      match error with
      | Unsupported_source_lang {lang; backend = _} ->
          Printf.sprintf "%s Source language not supported: %s" prefix lang
      | Backend_unavailable {reason} ->
          Printf.sprintf "%s Backend unavailable: %s" prefix reason
      | Library_not_found {library; paths} ->
          Printf.sprintf
            "%s Library '%s' not found in: %s"
            prefix
            library
            (String.concat ", " paths)
      | Initialization_failed {backend = _; reason} ->
          Printf.sprintf "%s Initialization failed: %s" prefix reason
      | Feature_not_supported {feature; backend = _} ->
          Printf.sprintf "%s Feature not supported: %s" prefix feature)

(** Raise backend error as exception *)
let raise_error err = raise (Backend_error err)

(** Print error to stderr *)
let print_error err = Printf.eprintf "%s\n%!" (to_string err)

(** Execute function with default fallback on error *)
let with_default ~default f = try f () with Backend_error _ -> default

(** Convert error to Result type *)
let to_result f = try Ok (f ()) with Backend_error err -> Error err

(** Map Result error to string *)
let result_to_string = function
  | Ok v -> Ok v
  | Error err -> Error (to_string err)

(** {1 Backend-Specific Modules} *)

(** Helper module for creating backend-specific error interfaces. Each backend
    can instantiate this functor with their name. *)
module Make (B : sig
  val name : string
end) =
struct
  let backend = B.name

  (** {1 Codegen Errors} *)

  let unknown_intrinsic name = unknown_intrinsic ~backend name

  let invalid_arg_count intrinsic expected got =
    invalid_arg_count ~backend intrinsic expected got

  let unsupported_construct construct reason =
    unsupported_construct ~backend construct reason

  let type_error expr expected got = type_error ~backend expr expected got

  let invalid_memory_space decl space = invalid_memory_space ~backend decl space

  let unsupported_type type_name = unsupported_type ~backend type_name

  (** {1 Runtime Errors} *)

  let no_device_selected operation = no_device_selected ~backend operation

  let device_not_found device_id max_devices =
    device_not_found ~backend device_id max_devices

  let compilation_failed source log = compilation_failed ~backend source log

  let module_load_failed size reason = module_load_failed ~backend size reason

  let kernel_launch_failed kernel_name reason =
    kernel_launch_failed ~backend kernel_name reason

  let memory_allocation_failed bytes reason =
    memory_allocation_failed ~backend bytes reason

  let memory_copy_failed direction bytes reason =
    memory_copy_failed ~backend direction bytes reason

  let context_error operation reason = context_error ~backend operation reason

  let synchronization_failed reason = synchronization_failed ~backend reason

  (** {1 Plugin Errors} *)

  let unsupported_source_lang lang = unsupported_source_lang ~backend lang

  let backend_unavailable reason = backend_unavailable ~backend reason

  let library_not_found library paths = library_not_found ~backend library paths

  let initialization_failed reason = initialization_failed ~backend reason

  let feature_not_supported feature = feature_not_supported ~backend feature

  (** {1 Re-export common utilities} *)

  let raise_error = raise_error

  let print_error = print_error

  let with_default = with_default

  let to_result = to_result

  let to_string = to_string
end
