(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Runtime - Debug Logging
 *
 * Configurable logging by component and level.
 * Components can be enabled/disabled at runtime.
 ******************************************************************************)

(** Log levels *)
type level = Debug | Info | Warn | Error

(** Components that can be logged *)
type component = Transfer | Kernel | Device | Memory | Execute | All

(** Current log level (messages at or above this level are shown) *)
let log_level = ref Warn

(** Enabled components *)
let enabled_components : (component, bool) Hashtbl.t = Hashtbl.create 8

(** Initialize with all components disabled *)
let () =
  Hashtbl.add enabled_components Transfer false ;
  Hashtbl.add enabled_components Kernel false ;
  Hashtbl.add enabled_components Device false ;
  Hashtbl.add enabled_components Memory false ;
  Hashtbl.add enabled_components Execute false ;
  Hashtbl.add enabled_components All false

(** Enable a component *)
let enable comp =
  if comp = All then begin
    Hashtbl.replace enabled_components Transfer true ;
    Hashtbl.replace enabled_components Kernel true ;
    Hashtbl.replace enabled_components Device true ;
    Hashtbl.replace enabled_components Memory true ;
    Hashtbl.replace enabled_components Execute true
  end ;
  Hashtbl.replace enabled_components comp true

(** Disable a component *)
let disable comp =
  if comp = All then begin
    Hashtbl.replace enabled_components Transfer false ;
    Hashtbl.replace enabled_components Kernel false ;
    Hashtbl.replace enabled_components Device false ;
    Hashtbl.replace enabled_components Memory false ;
    Hashtbl.replace enabled_components Execute false
  end ;
  Hashtbl.replace enabled_components comp false

(** Enable from environment variable SAREK_DEBUG=transfer,kernel,... Use "*" or
    "all" to enable all components. *)
let init_from_env () =
  match Sys.getenv_opt "SAREK_DEBUG" with
  | None -> ()
  | Some s ->
      let parts = String.split_on_char ',' s in
      List.iter
        (fun p ->
          match String.lowercase_ascii (String.trim p) with
          | "transfer" -> enable Transfer
          | "kernel" -> enable Kernel
          | "device" -> enable Device
          | "memory" -> enable Memory
          | "execute" -> enable Execute
          | "all" | "*" -> enable All
          | "debug" -> log_level := Debug
          | "info" -> log_level := Info
          | "warn" -> log_level := Warn
          | "error" -> log_level := Error
          | _ -> ())
        parts

(** Initialize on module load *)
let () = init_from_env ()

(** Set log level *)
let set_level lvl = log_level := lvl

(** Check if component is enabled *)
let is_enabled comp =
  Hashtbl.find_opt enabled_components All = Some true
  || Hashtbl.find_opt enabled_components comp = Some true

(** Level to int for comparison *)
let level_to_int = function Debug -> 0 | Info -> 1 | Warn -> 2 | Error -> 3

(** Level to string *)
let level_to_string = function
  | Debug -> "DEBUG"
  | Info -> "INFO"
  | Warn -> "WARN"
  | Error -> "ERROR"

(** Component to string *)
let component_to_string = function
  | Transfer -> "Transfer"
  | Kernel -> "Kernel"
  | Device -> "Device"
  | Memory -> "Memory"
  | Execute -> "Execute"
  | All -> "All"

(** Core logging function - printf style *)
let logf level comp fmt =
  if is_enabled comp && level_to_int level >= level_to_int !log_level then begin
    Printf.printf "[%s][%s] " (level_to_string level) (component_to_string comp) ;
    Printf.kfprintf (fun oc -> Printf.fprintf oc "\n%!") stdout fmt
  end
  else Printf.ifprintf stdout fmt

(** String-based logging (simpler, no format issues) *)
let log level comp msg =
  if is_enabled comp && level_to_int level >= level_to_int !log_level then
    Printf.printf
      "[%s][%s] %s\n%!"
      (level_to_string level)
      (component_to_string comp)
      msg

(** Convenience functions - string based *)
let debug comp msg = log Debug comp msg

let info comp msg = log Info comp msg

let warn comp msg = log Warn comp msg

let error comp msg = log Error comp msg

(** Convenience functions - printf style *)
let debugf comp fmt = logf Debug comp fmt

let infof comp fmt = logf Info comp fmt

let warnf comp fmt = logf Warn comp fmt

let errorf comp fmt = logf Error comp fmt
