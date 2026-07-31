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

(** Render [Backend_error] through {!to_string} anywhere an exception is
    stringified via [Printexc.to_string] — notably generic funnels that catch an
    arbitrary [exn] and re-wrap it (e.g. Sarek_transpile's
    [Internal_error (Printexc.to_string exn)]). Without this printer such paths
    emit the opaque [Backend_error(_)] constructor and lose the located message
    (backend + intrinsic name). *)
let () =
  Printexc.register_printer (function
    | Backend_error err -> Some (to_string err)
    | _ -> None)

(** Raise backend error as exception *)
let raise_error err = raise (Backend_error err)

(** {1 Shared refusals} *)

(** Refuse a non-empty [~soa_params] on a backend whose emitter has no
    Structure-of-Arrays lowering (backlog-214).

    [Framework_sig.generate_source] offers [?soa_params] to every backend, but
    only an emitter that actually lowers a named vector parameter to N per-leaf
    bindings plus one shared length can honour it. A backend without that
    lowering used to bind the argument away as [?soa_params:_] and return its
    ordinary packed-AoS source — ONE binding per vector — while the launch side
    expands an SoA-dispatched vector into N [RSA_Buffer]s plus one
    [RSA_Vector_Length]. That mismatch is never a compile error, and how badly
    it fails is per backend rather than uniform, which is why the refusal is
    stated here in terms of the mismatch and not of its symptom:

    - CUDA/C and HIP bind positionally into an unchecked argument array
      ([Cuda_shared.bind_args], [Hip_shared.bind_args]) handed to
      [cuLaunchKernel]/[hipModuleLaunchKernel], so N pointers land in a packed
      parameter block — not a crash, silently wrong data;
    - Metal and OpenCL derive their expected argument count from the indices the
      caller bound ([Kernel_args.count]), so they have no independent count to
      disagree with either;
    - Vulkan does have a second, source-derived count
      ([Vulkan_api_kernel.validate_buffer_indices], whose [expected_count] is
      read from the GLSL [binding = N] declarations), so there the mismatch
      surfaces late as "expected 1 buffer argument but got N" — a named
      rejection that says nothing about SoA.

    As of backlog-214 no caller in this tree reaches any of that:
    [Execute.soa_dispatch] restricts SoA to the CUDA/PTX device and [Soa_launch]
    gates on [PTX] being in the backend's [supported_source_langs]. So this is a
    boundary, not a live bug fix. What it changes is where the guarantee lives:
    it was one caller-side predicate and nothing else, and a backend that cannot
    honour the request now says so instead of answering with the wrong ABI.

    Scope of what this raises, stated narrowly on purpose: it says what THIS
    backend does with vector parameters. It deliberately does not name which
    other backend does support SoA — that set is expected to grow (backlog-215),
    and a message enumerating it would go stale in a file that is not edited
    when it does. It is also not re-exported through {!Make}: [Make] closes over
    one backend string per error module, and [Cuda_error]'s is "CUDA" for both
    the CUDA/PTX backend that implements SoA and the CUDA/C backend that refuses
    it, so a [Make]-based version could not tell the caller which one answered.

    [[]] returns [()], so every existing caller and the caller-side fast path
    are byte-for-byte unaffected. *)
let reject_soa_params ~backend (soa_params : string list) : unit =
  match soa_params with
  | [] -> ()
  | names ->
      raise_error
        (feature_not_supported
           ~backend
           (Printf.sprintf
              "Structure-of-Arrays parameter lowering, requested for %s. This \
               backend's emitter gives every vector parameter the packed \
               Array-of-Structures form (one binding plus one length), so the \
               N-leaf-binding argument list an SoA launch produces would not \
               match the source it generates. Pass the vector through the \
               ordinary AoS launch path on this backend."
              (String.concat ", " (List.map (Printf.sprintf "'%s'") names))))

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

  (** {1 Shared FFI check funnel}

      Generic replacement for each backend's hand-rolled [check ctx result]
      function (e.g. [Cuda_api.check], [Opencl_api.check], [Metal_api.check],
      [Vulkan_api_base.check]). Backends supply their own success predicate and
      stringifier since the underlying result types differ (CUDA [cu_result],
      OpenCL [cl_error], Metal [mtl_error], Vulkan [vk_result]); this folds the
      raw code/message into a canonical {!Backend_error} via [context_error]
      instead of a backend-specific exception, so every backend's FFI funnel
      raises the same exception shape and callers only ever need to catch
      [Backend_error]. *)
  let check ~is_success ~to_string (ctx : string) result =
    if is_success result then ()
    else raise_error (context_error ctx (to_string result))

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
