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
  (* [machine] is an OPAQUE grouping label, never a hostname -- see
     [get_machine_label]. Deliberately absent from this record: the hostname
     (an identity) and the kernel version (a patch level). Both used to be
     collected and both were published; backlog-168 removed them. Host-level
     memory_gb went with them: no chart read it, and it narrows a machine.
     Kept: os, cpu, devices -- a benchmark number is unreadable without the
     hardware that produced it, so hardware model is NOT the line. Identity
     and patch level are. *)
  machine : string;
  os : string;
  cpu : cpu_info;
  devices : device_info list;
}

(* PRIVATE. The only legitimate use is rejecting an override that IS the
   hostname (see [get_machine_label]). This value must never reach a payload,
   a filename, or a CSV row -- that was backlog-168. *)
let read_hostname_for_rejection_check () =
  try
    let ic = Unix.open_process_in "hostname" in
    let hostname = input_line ic in
    let _ = Unix.close_process_in ic in
    String.trim hostname
  with _ -> "unknown"

(* Vendor of the first device that is not just the CPU-as-OpenCL-device.
   Derived from names we publish anyway, so it discloses nothing new. *)
let gpu_vendor_of devices cpu_model =
  let lower s = String.lowercase_ascii s in
  let contains hay needle =
    let nh = String.length needle and hl = String.length hay in
    let rec go i =
      i + nh <= hl && (String.sub hay i nh = needle || go (i + 1))
    in
    nh > 0 && go 0
  in
  let vendor_of name =
    let n = lower name in
    let table =
      [
        ("nvidia", "nvidia");
        ("geforce", "nvidia");
        ("rtx", "nvidia");
        ("radeon", "amd");
        ("gfx", "amd");
        ("amd", "amd");
        ("arc", "intel");
        ("intel", "intel");
        ("apple", "apple");
        ("m1", "apple");
        ("m2", "apple");
        ("m3", "apple");
        ("m4", "apple");
      ]
    in
    List.find_map
      (fun (tok, v) -> if contains n tok then Some v else None)
      table
  in
  let candidates =
    List.filter (fun d -> lower d.name <> lower cpu_model) devices
  in
  match List.find_map (fun d -> vendor_of d.name) candidates with
  | Some v -> v
  | None -> "unknown"

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
  let os = get_os_info () in
  try
    let model =
      if os = "Darwin" then
        (* macOS: use sysctl *)
        try
          let ic = Unix.open_process_in "sysctl -n machdep.cpu.brand_string" in
          let m = String.trim (input_line ic) in
          let _ = Unix.close_process_in ic in
          m
        with _ -> "unknown"
      else
        (* Linux: read /proc/cpuinfo *)
        try
          let ic = open_in "/proc/cpuinfo" in
          let rec find_model () =
            try
              let line = input_line ic in
              if String.starts_with ~prefix:"model name" line then
                let parts = String.split_on_char ':' line in
                if List.length parts >= 2 then
                  Some (String.trim (List.nth parts 1))
                else find_model ()
              else find_model ()
            with End_of_file -> None
          in
          let m = match find_model () with Some m -> m | None -> "unknown" in
          close_in ic ;
          m
        with _ -> "unknown"
    in

    (* Get thread count *)
    let threads =
      if os = "Darwin" then
        (* macOS: use sysctl *)
        try
          let ic = Unix.open_process_in "sysctl -n hw.logicalcpu" in
          let t = int_of_string (String.trim (input_line ic)) in
          let _ = Unix.close_process_in ic in
          t
        with _ -> 1
      else
        (* Linux: use nproc *)
        try
          let ic = Unix.open_process_in "nproc" in
          let t = int_of_string (String.trim (input_line ic)) in
          let _ = Unix.close_process_in ic in
          t
        with _ -> 1
    in

    (* Get physical core count *)
    let cores =
      if os = "Darwin" then
        (* macOS: use sysctl *)
        try
          let ic = Unix.open_process_in "sysctl -n hw.physicalcpu" in
          let c = int_of_string (String.trim (input_line ic)) in
          let _ = Unix.close_process_in ic in
          c
        with _ -> max 1 (threads / 2)
      else
        (* Linux: try lscpu; fallback to threads/2 *)
        try
          let ic2 = Unix.open_process_in "lscpu 2>/dev/null" in
          let rec loop cps sockets =
            try
              let line = input_line ic2 in
              if String.starts_with ~prefix:"Core(s) per socket" line then
                let v =
                  int_of_string
                    (String.trim (List.nth (String.split_on_char ':' line) 1))
                in
                loop (Some v) sockets
              else if String.starts_with ~prefix:"Socket(s)" line then
                let v =
                  int_of_string
                    (String.trim (List.nth (String.split_on_char ':' line) 1))
                in
                loop cps (Some v)
              else loop cps sockets
            with End_of_file -> (
              match (cps, sockets) with
              | Some cps, Some s -> cps * s
              | _ -> max 1 (threads / 2))
          in
          let result = loop None None in
          let _ = Unix.close_process_in ic2 in
          result
        with _ -> max 1 (threads / 2)
    in

    {model; cores; threads}
  with _ -> {model = "unknown"; cores = 1; threads = 1}

let get_memory_gb () =
  let os = get_os_info () in
  try
    if os = "Darwin" then
      (* macOS: use sysctl hw.memsize *)
      let ic = Unix.open_process_in "sysctl -n hw.memsize" in
      let bytes = float_of_string (String.trim (input_line ic)) in
      let _ = Unix.close_process_in ic in
      bytes /. (1024.0 *. 1024.0 *. 1024.0)
    else
      (* Linux: use free *)
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

(* The label an operator may set to keep two same-hardware machines apart.
   Two boxes with the same OS and GPU vendor derive the SAME label and their
   runs would be merged by the dedup key -- that is why the override exists.
   It is REFUSED when it equals the hostname, which is the mistake it is here
   to prevent: an opt-in escape hatch that silently reintroduced the leak
   would be worse than no escape hatch. *)
let machine_label_env = "SAREK_BENCH_MACHINE"

let get_machine_label ~os ~cpu_model ~devices =
  let derived =
    Printf.sprintf
      "%s-%s"
      (String.lowercase_ascii os)
      (gpu_vendor_of devices cpu_model)
  in
  match Sys.getenv_opt machine_label_env with
  | None | Some "" -> derived
  | Some override ->
      let host = read_hostname_for_rejection_check () in
      let norm s = String.lowercase_ascii (String.trim s) in
      if norm override = norm host then
        failwith
          (Printf.sprintf
             "%s is set to the machine's hostname. That is the identifier \
              benchmark output must not carry (backlog-168). Choose an opaque \
              label such as %S."
             machine_label_env
             derived)
      else String.trim override

let collect devices =
  let os = get_os_info () in
  let cpu = get_cpu_info () in
  let devices =
    Array.to_list devices |> List.mapi (fun i dev -> get_device_info dev i)
  in
  let machine = get_machine_label ~os ~cpu_model:cpu.model ~devices in
  {machine; os; cpu; devices}

let to_json (info : system_info) =
  `Assoc
    [
      ("machine", `String info.machine);
      ("os", `String info.os);
      ( "cpu",
        `Assoc
          [
            ("model", `String info.cpu.model);
            ("cores", `Int info.cpu.cores);
            ("threads", `Int info.cpu.threads);
          ] );
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
