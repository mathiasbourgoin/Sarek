(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** System information collection for benchmark metadata *)

open Spoc_core

type cpu_info = {model : string; cores : int; threads : int}

type device_info = {
  id : int;
  name : string;
  framework : string;
  compute_capability : string option;
  memory_gb : float;
  driver_version : string option;
  runtime_version : string option;
}

type system_info = {
  hostname : string;
  os : string;
  kernel : string;
  cpu : cpu_info;
  memory_gb : float;
  devices : device_info list;
}

let get_hostname () =
  try
    let ic = Unix.open_process_in "hostname" in
    let hostname = input_line ic in
    let _ = Unix.close_process_in ic in
    String.trim hostname
  with _ -> "unknown"

let get_os_info () =
  try
    let ic = Unix.open_process_in "uname -s" in
    let os = input_line ic in
    let _ = Unix.close_process_in ic in
    String.trim os
  with _ -> "unknown"

let get_kernel_info () =
  try
    let ic = Unix.open_process_in "uname -r" in
    let kernel = input_line ic in
    let _ = Unix.close_process_in ic in
    String.trim kernel
  with _ -> "unknown"

let get_cpu_info () =
  try
    (* Try to get CPU model from /proc/cpuinfo *)
    let ic = open_in "/proc/cpuinfo" in
    let rec find_model () =
      try
        let line = input_line ic in
        if String.starts_with ~prefix:"model name" line then
          let parts = String.split_on_char ':' line in
          if List.length parts >= 2 then Some (String.trim (List.nth parts 1))
          else find_model ()
        else find_model ()
      with End_of_file -> None
    in
    let model = match find_model () with Some m -> m | None -> "unknown" in
    close_in ic ;

    (* Get core count *)
    let ic = Unix.open_process_in "nproc" in
    let threads =
      try int_of_string (String.trim (input_line ic)) with _ -> 1
    in
    let _ = Unix.close_process_in ic in

    {model; cores = threads; threads}
  with _ -> {model = "unknown"; cores = 1; threads = 1}

let get_memory_gb () =
  try
    let ic = Unix.open_process_in "free -b | grep Mem | awk '{print $2}'" in
    let bytes = float_of_string (String.trim (input_line ic)) in
    let _ = Unix.close_process_in ic in
    bytes /. (1024.0 *. 1024.0 *. 1024.0)
  with _ -> 0.0

let get_device_info (dev : Device.t) dev_id =
  let memory_gb =
    Int64.to_float dev.capabilities.total_global_mem
    /. (1024.0 *. 1024.0 *. 1024.0)
  in
  let compute_capability =
    let major, minor = dev.capabilities.compute_capability in
    if major = 0 && minor = 0 then None
    else Some (Printf.sprintf "%d.%d" major minor)
  in
  {
    id = dev_id;
    name = dev.name;
    framework = dev.framework;
    compute_capability;
    memory_gb;
    driver_version = None;
    (* Not available in capabilities *)
    runtime_version = None;
    (* Could be extended per framework *)
  }

let collect devices =
  let hostname = get_hostname () in
  let os = get_os_info () in
  let kernel = get_kernel_info () in
  let cpu = get_cpu_info () in
  let memory_gb = get_memory_gb () in
  let devices =
    Array.to_list devices |> List.mapi (fun i dev -> get_device_info dev i)
  in
  {hostname; os; kernel; cpu; memory_gb; devices}

let to_json (info : system_info) =
  `Assoc
    [
      ("hostname", `String info.hostname);
      ("os", `String info.os);
      ("kernel", `String info.kernel);
      ( "cpu",
        `Assoc
          [
            ("model", `String info.cpu.model);
            ("cores", `Int info.cpu.cores);
            ("threads", `Int info.cpu.threads);
          ] );
      ("memory_gb", `Float info.memory_gb);
      ( "devices",
        `List
          (List.map
             (fun d ->
               let fields =
                 [
                   ("id", `Int d.id);
                   ("name", `String d.name);
                   ("framework", `String d.framework);
                   ("memory_gb", `Float d.memory_gb);
                 ]
               in
               let fields =
                 match d.compute_capability with
                 | Some cc -> ("compute_capability", `String cc) :: fields
                 | None -> fields
               in
               let fields =
                 match d.driver_version with
                 | Some v -> ("driver_version", `String v) :: fields
                 | None -> fields
               in
               `Assoc fields)
             info.devices) );
    ]
