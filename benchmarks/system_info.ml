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

(* Vendor token for the machine label. Derived from names we publish anyway, so
   it discloses nothing new.

   This label is load-bearing beyond cosmetics: it is the dedup key AND the
   leading token of every result FILENAME (see [Output.make_filename]) -- the
   slot the hostname used to occupy (backlog-168). So it has to be STABLE. Two
   runs on one machine must derive the same label, or the dedup key splits and
   the same box shows up as several machines.

   Two things it must therefore not do, both fixed here:

   1. It must not let a CPU device speak for the machine. An OpenCL platform
      commonly exposes the CPU as a device, and its name is the CPU brand
      string -- which contains a vendor token ("AMD Ryzen ...", "Intel(R)
      Core(TM) ..."). Excluding it by [name = cpu_model] does not work: the
      OpenCL name and /proc/cpuinfo's "model name" differ by whitespace runs
      and trailing text often enough that the CPU survived the filter and
      outvoted a real discrete GPU. Excluded now by BACKEND (Native and
      Interpreter are CPU by construction) and by normalized containment in
      either direction (which absorbs the whitespace/suffix drift).

   2. It must not depend on device enumeration ORDER. It used to take the first
      vendor-matching device in [devices], so a discrete + integrated box (or a
      multi-GPU one) derived a different label whenever the driver enumerated
      differently between runs. The vendor is now chosen from the whole set of
      surviving matches by a FIXED priority, so the result is a function of the
      set and not of the traversal. Discrete vendors rank first, which is also
      the answer you want on discrete + integrated. *)
let gpu_vendor_of devices cpu_model =
  let lower s = String.lowercase_ascii s in
  let contains hay needle =
    let nh = String.length needle and hl = String.length hay in
    let rec go i =
      i + nh <= hl && (String.sub hay i nh = needle || go (i + 1))
    in
    nh > 0 && go 0
  in
  (* Lowercase, collapse whitespace runs, trim -- so "  Apple  M1 Max " and
     "Apple M1 Max" compare equal. *)
  let normalize s =
    let buf = Buffer.create (String.length s) in
    String.iter
      (fun c ->
        let c = if c = '\t' || c = '\n' || c = '\r' then ' ' else c in
        if c = ' ' then begin
          if
            Buffer.length buf > 0
            && Buffer.nth buf (Buffer.length buf - 1) <> ' '
          then Buffer.add_char buf c
        end
        else Buffer.add_char buf (Char.lowercase_ascii c))
      s ;
    String.trim (Buffer.contents buf)
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
  (* A device that is really the CPU.

     The name test is applied ONLY on backends that can expose a CPU as a
     device -- in practice OpenCL. That restriction is the whole point: on a
     unified-memory SoC the GPU's name IS the CPU's name ("Apple M4 Max"), so a
     name test applied to the Metal device excludes the only GPU on the machine
     and the label collapses to "unknown". That was the behaviour before this
     change, under the old [name = cpu_model] exclusion: every Apple Silicon box
     derived darwin-unknown rather than darwin-apple, and nothing complained
     because "unknown" is a legal label.

     A CUDA/Metal/Vulkan/HIP device is a GPU by construction, so it is never
     excluded by name. Native and Interpreter are the CPU by construction, so
     they are always excluded. Residual corner: a machine whose ONLY backend is
     OpenCL and whose GPU is named exactly like its CPU is indistinguishable
     here and still yields "unknown". *)
  let is_cpu_device d =
    let backend = lower d.framework in
    if backend = "native" || backend = "interpreter" then true
    else if backend <> "opencl" then false
    else
      let n = normalize d.name and c = normalize cpu_model in
      (* "unknown" is [get_cpu_info]'s failure value, not a CPU name; matching
         on it would exclude every device on a box where the CPU probe failed,
         turning one probe failure into a label change. *)
      c <> "" && c <> "unknown" && (contains n c || contains c n)
  in
  let vendors =
    devices
    |> List.filter (fun d -> not (is_cpu_device d))
    |> List.filter_map (fun d -> vendor_of d.name)
  in
  (* Order-independent: pick by this fixed ranking, never by list position. *)
  let priority = ["nvidia"; "amd"; "intel"; "apple"] in
  match List.find_opt (fun v -> List.mem v vendors) priority with
  | Some v -> v
  | None -> "unknown"

let get_os_info () =
  try
    let ic = Unix.open_process_in "uname -s" in
    let os = input_line ic in
    let _ = Unix.close_process_in ic in
    String.trim os
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

   The policy itself lives in [Machine_label.resolve], which is Stdlib-only so
   the red-path harness can execute it against the same case table as the
   commit gate's regex. Three things it enforces, in this order: the override is
   REFUSED (hard [failwith]) when it equals the hostname -- the mistake this
   override is here to prevent, since an escape hatch that silently
   reintroduced the leak would be worse than no escape hatch -- then refused
   unless it has the label shape the commit gate accepts, so an operator cannot
   end up with results that cannot be committed, and finally refused unless it
   is the DERIVED label or that label plus a suffix: the scrubber recomputes the
   label from the payload's own hardware, so an override claiming different
   hardware comes back relabelled while the filenames keep the operator's
   version.

   Only the OVERRIDE is shape-checked. The DERIVED label is not: on an OS
   outside the shape's enumeration (uname -s says e.g. FreeBSD) the derived
   label is still what the producer uses, exactly as before -- widening the
   accept-list to cover more platforms is a separate decision, and failing a
   benchmark run over it here would be a new refusal, not this change. *)
let machine_label_env = Machine_label.env_var

let get_machine_label ~os ~cpu_model ~devices =
  let derived =
    Printf.sprintf
      "%s-%s"
      (String.lowercase_ascii os)
      (gpu_vendor_of devices cpu_model)
  in
  Machine_label.resolve
    ~derived
    ~override:(Sys.getenv_opt machine_label_env)
    ~hostname:read_hostname_for_rejection_check

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
