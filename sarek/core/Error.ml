(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Runtime - Error Handling & Debugging
 *
 * Provides structured exceptions and debugging tools.
 * Delegates logging to Log.ml so there is a single logging pipeline.
 ******************************************************************************)

(** {1 Structured Exceptions} *)

(** No compute device available *)
exception No_device_available

(** Device with given name not found *)
exception Device_not_found of string

(** Requested block size exceeds device limit *)
exception Invalid_block_size of {requested : int; max : int}

(** Requested grid size exceeds device limit *)
exception Invalid_grid_size of {requested : int; max : int}

(** Not enough memory for allocation *)
exception Out_of_memory of {requested : int64; available : int64}

(** Kernel compilation failed *)
exception Compilation_error of {device : string; message : string}

(** Data transfer failed *)
exception
  Transfer_error of {direction : [`To_device | `To_host]; message : string}

(** Kernel execution failed *)
exception Kernel_error of {name : string; message : string}

(** Invalid argument to runtime function *)
exception Invalid_argument of {func : string; message : string}

(** Backend-specific error *)
exception Backend_error of {backend : string; code : int; message : string}

(** {1 Exception Helpers} *)

let string_of_direction = function
  | `To_device -> "host->device"
  | `To_host -> "device->host"

let pp_exn fmt = function
  | No_device_available -> Format.fprintf fmt "No compute device available"
  | Device_not_found name -> Format.fprintf fmt "Device not found: %s" name
  | Invalid_block_size {requested; max} ->
      Format.fprintf fmt "Invalid block size: %d (max %d)" requested max
  | Invalid_grid_size {requested; max} ->
      Format.fprintf fmt "Invalid grid size: %d (max %d)" requested max
  | Out_of_memory {requested; available} ->
      Format.fprintf
        fmt
        "Out of memory: requested %Ld bytes, available %Ld"
        requested
        available
  | Compilation_error {device; message} ->
      Format.fprintf fmt "Compilation error on %s: %s" device message
  | Transfer_error {direction; message} ->
      Format.fprintf
        fmt
        "Transfer error (%s): %s"
        (string_of_direction direction)
        message
  | Kernel_error {name; message} ->
      Format.fprintf fmt "Kernel error in %s: %s" name message
  | Invalid_argument {func; message} ->
      Format.fprintf fmt "Invalid argument to %s: %s" func message
  | Backend_error {backend; code; message} ->
      Format.fprintf fmt "Backend error (%s, code %d): %s" backend code message
  | exn -> Format.fprintf fmt "%s" (Printexc.to_string exn)

let to_string exn = Format.asprintf "%a" pp_exn exn

(** {1 Debug Mode (delegates to Log.ml)} *)

type log_level = Silent | Errors | Warnings | Info | Trace

let current_level = ref Errors

let to_log_level = function
  | Silent -> Log.Error (* filtered by should_log *)
  | Errors -> Log.Error
  | Warnings -> Log.Warn
  | Info -> Log.Info
  | Trace -> Log.Debug

let level_to_int = function
  | Silent -> 0
  | Errors -> 1
  | Warnings -> 2
  | Info -> 3
  | Trace -> 4

let should_log level = level_to_int level <= level_to_int !current_level

let set_level level =
  current_level := level ;
  Log.set_level (to_log_level level)

let get_level () = !current_level

(* Ensure Error logs route through the unified logger *)
let () = Log.enable Log.All

(** {1 Logging Functions} *)

let log level fmt =
  if should_log level then
    Format.kasprintf (fun msg -> Log.log (to_log_level level) Log.All msg) fmt
  else Format.ikfprintf ignore Format.err_formatter fmt

let error fmt = log Errors fmt

let warn fmt = log Warnings fmt

let info fmt = log Info fmt

let trace fmt = log Trace fmt

(** {1 Runtime Validation} *)

let validate_kernel_args = ref false

let validate_bounds = ref false

let trace_transfers = ref false

let trace_kernels = ref false

let enable_validation () =
  validate_kernel_args := true ;
  validate_bounds := true

let disable_validation () =
  validate_kernel_args := false ;
  validate_bounds := false

let enable_tracing () =
  trace_transfers := true ;
  trace_kernels := true

let disable_tracing () =
  trace_transfers := false ;
  trace_kernels := false

(** {1 Debug Helpers} *)

(** Check block size against device limits *)
let check_block_size ~requested ~max =
  if requested > max then raise (Invalid_block_size {requested; max})

(** Check grid size against device limits *)
let check_grid_size ~requested ~max =
  if requested > max then raise (Invalid_grid_size {requested; max})

(** Check memory availability *)
let check_memory ~requested ~available =
  if requested > available then raise (Out_of_memory {requested; available})

(** Wrap function with error context *)
let with_context ~(func : string) f =
  try f () with
  | Invalid_argument _ as e -> raise e
  | exn -> raise (Invalid_argument {func; message = Printexc.to_string exn})

(** {1 Result Type Helpers} *)

type 'a result = ('a, exn) Result.t

let ok x = Result.Ok x

let error_result e = Result.Error e

let try_with f = try ok (f ()) with exn -> error_result exn

let ( let* ) = Result.bind

let map_error f = function
  | Result.Ok x -> Result.Ok x
  | Result.Error e -> Result.Error (f e)

(** {1 Debug Output} *)

(** Print debug info about a vector *)
let debug_vector (vec : ('a, 'b) Vector.t) =
  if should_log Trace then
    trace
      "Vector#%d: len=%d kind=%s loc=%s"
      (Vector.id vec)
      (Vector.length vec)
      (Vector.kind_name (Vector.kind vec))
      (Vector.location_to_string (Vector.location vec))

(** Print debug info about a device *)
let debug_device (dev : Device.t) =
  if should_log Trace then
    trace
      "Device#%d: %s (%s) mem=%Ld"
      dev.id
      dev.name
      dev.framework
      dev.capabilities.total_global_mem

(** {1 Assertions} *)

(** Assert condition or raise Invalid_argument *)
let assert_arg ~(func : string) ~(cond : bool) ~(message : string) =
  if not cond then raise (Invalid_argument {func; message})

(** Assert vector is on CPU *)
let assert_on_cpu ~(func : string) (vec : ('a, 'b) Vector.t) =
  if not (Vector.is_on_cpu vec) then
    raise (Invalid_argument {func; message = "Vector must be on CPU"})

(** Assert vector is on GPU *)
let assert_on_gpu ~(func : string) (vec : ('a, 'b) Vector.t) =
  if not (Vector.is_on_gpu vec) then
    raise (Invalid_argument {func; message = "Vector must be on GPU"})
