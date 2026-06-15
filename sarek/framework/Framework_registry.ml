(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Framework - Plugin Registry
 *
 * Central registry for GPU backend plugins. Plugins register themselves
 * on module load and can be queried by name or capability.
 ******************************************************************************)

open Spoc_framework.Framework_sig

(** Enable debug logging via environment variable *)
let debug_enabled =
  try
    let dbg = Sys.getenv "SAREK_DEBUG" in
    dbg = "all" || String.split_on_char ',' dbg |> List.mem "device"
  with Not_found -> false

let debugf fmt =
  if debug_enabled then Printf.eprintf ("[Framework] " ^^ fmt ^^ "\n%!")
  else Printf.ifprintf stderr fmt

let warnf fmt = Printf.eprintf ("[Framework] WARNING: " ^^ fmt ^^ "\n%!")

(** Registered plugins - keyed by name *)
let plugins : (string, (module S)) Hashtbl.t = Hashtbl.create 8

(** Full backend plugins with complete functionality *)
let backends : (string, (module BACKEND)) Hashtbl.t = Hashtbl.create 8

(** Priority ordering for auto-selection (higher = preferred) *)
let priorities : (string, int) Hashtbl.t = Hashtbl.create 8

(** Register a minimal plugin (name, version, is_available) *)
let register ?(priority = 50) (module P : S) =
  debugf "Registering plugin: %s (priority=%d)" P.name priority ;
  Hashtbl.replace plugins P.name (module P : S) ;
  Hashtbl.replace priorities P.name priority

(** Register a full backend plugin *)
let register_backend ?(priority = 50) (module B : BACKEND) =
  debugf
    "Registering backend: %s (priority=%d, model=%s)"
    B.name
    priority
    (match B.execution_model with
    | JIT -> "JIT"
    | Direct -> "Direct"
    | Custom -> "Custom") ;
  Hashtbl.replace backends B.name (module B : BACKEND) ;
  Hashtbl.replace plugins B.name (module B : S) ;
  Hashtbl.replace priorities B.name priority

(** Find a plugin by name *)
let find name =
  let result = Hashtbl.find_opt plugins name in
  (match result with
  | Some _ -> debugf "Plugin found: %s" name
  | None -> debugf "Plugin not found: %s" name) ;
  result

(** Find a full backend by name *)
let find_backend name =
  let result = Hashtbl.find_opt backends name in
  (match result with
  | Some _ -> debugf "Backend found: %s" name
  | None -> debugf "Backend not found: %s" name) ;
  result

(** List all registered plugin names *)
let names () = Hashtbl.to_seq_keys plugins |> List.of_seq

(** List all available plugins (those where is_available returns true) *)
let available () =
  Hashtbl.to_seq_values plugins
  |> Seq.filter (fun (module P : S) -> try P.is_available () with _ -> false)
  |> List.of_seq

(** List all available full backends *)
let available_backends () =
  Hashtbl.to_seq_values backends
  |> Seq.filter (fun (module B : BACKEND) ->
      try B.is_available () with _ -> false)
  |> List.of_seq

(** Get the best available backend (highest priority) *)
let best_backend () =
  let available =
    Hashtbl.to_seq backends
    |> Seq.filter (fun (_name, (module B : BACKEND)) ->
        try B.is_available () with _ -> false)
    |> List.of_seq
  in
  match available with
  | [] ->
      warnf "No backends available" ;
      None
  | backends ->
      let sorted =
        List.sort
          (fun (name1, _) (name2, _) ->
            let p1 =
              Hashtbl.find_opt priorities name1 |> Option.value ~default:50
            in
            let p2 =
              Hashtbl.find_opt priorities name2 |> Option.value ~default:50
            in
            compare p2 p1) (* descending *)
          backends
      in
      let best_name, best_backend = List.hd sorted in
      debugf
        "Best backend: %s (priority=%d)"
        best_name
        (Hashtbl.find_opt priorities best_name |> Option.value ~default:50) ;
      Some best_backend

(** Get plugin priority *)
let priority name = Hashtbl.find_opt priorities name |> Option.value ~default:50

(** Get all registered backend names sorted by priority (highest first) *)
let all_backend_names () =
  Hashtbl.to_seq_keys backends
  |> List.of_seq
  |> List.sort (fun name1 name2 ->
      let p1 = Hashtbl.find_opt priorities name1 |> Option.value ~default:50 in
      let p2 = Hashtbl.find_opt priorities name2 |> Option.value ~default:50 in
      compare p2 p1)
(* descending by priority *)
