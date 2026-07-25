(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Runtime - Unified Device Abstraction
 *
 * Provides a unified view of all available compute devices across backends.
 * Queries registered plugins and presents a single array of devices.
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

(** Device type from SDK *)
type t = Spoc_framework.Device_type.t = {
  id : int;
  backend_id : int;
  name : string;
  framework : string;
  capabilities : Framework_sig.capabilities;
}

(** Device initialization state *)
let devices : t array ref = ref [||]

let initialized = ref false

(** Resolve a requested framework name to registered backend names. An exact
    name matches itself; a family name ("CUDA") matches every backend registered
    as "<family>/<variant>" ("CUDA/PTX", "CUDA/C"), highest priority first.
    Keeps callers written against the historical family names working after the
    plugin split. *)
let resolve_framework fw_name =
  let all = Framework_registry.all_backend_names () in
  if List.mem fw_name all then [fw_name]
  else List.filter (String.starts_with ~prefix:(fw_name ^ "/")) all

(** Initialize all available backends and enumerate devices *)
let init
    ?(frameworks =
      ["CUDA"; "HIP"; "OpenCL"; "Vulkan"; "Metal"; "Native"; "Interpreter"]) ()
    =
  if !initialized then !devices
  else begin
    let all_devices = ref [] in
    let global_id = ref 0 in
    let seen = Hashtbl.create 8 in

    frameworks
    |> List.concat_map (fun fw_name ->
        match resolve_framework fw_name with
        | [] ->
            Log.debugf Log.Device "Framework %s: not registered" fw_name ;
            []
        | names -> names)
    |> List.filter (fun fw_name ->
        if Hashtbl.mem seen fw_name then false
        else begin
          Hashtbl.add seen fw_name () ;
          true
        end)
    |> List.iter (fun fw_name ->
        match Framework_registry.find_backend fw_name with
        | None -> Log.debugf Log.Device "Framework %s: not registered" fw_name
        | Some (module B : Framework_sig.BACKEND) ->
            if B.is_available () then begin
              try
                B.Device.init () ;
                let count = B.Device.count () in
                Log.debugf Log.Device "Framework %s: %d device(s)" fw_name count ;
                for i = 0 to count - 1 do
                  let dev = B.Device.get i in
                  let device =
                    {
                      id = !global_id;
                      backend_id = i;
                      name = B.Device.name dev;
                      framework = B.name;
                      capabilities = B.Device.capabilities dev;
                    }
                  in
                  Log.debugf
                    Log.Device
                    "  [%d] %s (%s)"
                    !global_id
                    device.name
                    fw_name ;
                  all_devices := device :: !all_devices ;
                  incr global_id
                done
              with e ->
                Log.debugf
                  Log.Device
                  "Framework %s: init failed (%s)"
                  fw_name
                  (Printexc.to_string e)
            end
            else Log.debugf Log.Device "Framework %s: not available" fw_name) ;

    devices := Array.of_list (List.rev !all_devices) ;
    initialized := true ;
    !devices
  end

(** Get all initialized devices *)
let all () = if not !initialized then init () else !devices

(** Get device count *)
let count () = Array.length (all ())

(** Get device by global ID *)
let get id =
  let devs = all () in
  if id >= 0 && id < Array.length devs then Some devs.(id) else None

(** Get first available device (if any) *)
let first () =
  let devs = all () in
  if Array.length devs > 0 then Some devs.(0) else None

(** Filter devices by framework *)
let by_framework framework =
  all () |> Array.to_list
  |> List.filter (fun d -> d.framework = framework)
  |> Array.of_list

(** Filter devices by capability *)
let with_fp64 () =
  all () |> Array.to_list
  |> List.filter (fun d -> d.capabilities.supports_fp64)
  |> Array.of_list

(** CUDA backends register as "CUDA/PTX" and "CUDA/C"; match the family name or
    its slash-separated variants (not arbitrary "CUDA*" names). *)
let is_cuda_framework framework =
  framework = "CUDA" || String.starts_with ~prefix:"CUDA/" framework

(** Get the best device (first CUDA, then OpenCL, then Native) *)
let best () =
  match Array.find_opt (fun d -> is_cuda_framework d.framework) (all ()) with
  | Some d -> Some d
  | None ->
      let opencl = by_framework "OpenCL" in
      if Array.length opencl > 0 then Some opencl.(0) else first ()

(** Reset initialization state (for testing).

    Retires the global id space, so every cache keyed on it must be dropped in
    the same breath. [init] restarts [global_id] at 0 over whichever frameworks
    it is handed, so the same id routinely denotes a DIFFERENT physical device
    after a reset: with [~frameworks:["OpenCL"]] then
    [~frameworks:["Native"; "OpenCL"]], the single Native device shifts OpenCL
    down one and id 1 moves from the second OpenCL device to the first. A
    surviving entry keyed on id 1 is then served to the wrong device, silently —
    a compiled kernel carries no device check — which is exactly the aliasing
    class this cache was keyed by device to prevent.

    Only [Sarek.Runtime]'s outer memo is affected: it keys on the global
    [Device.t.id], while every per-backend cache keys on its own backend-local
    device index, which a reset does not perturb. So the notification here is
    enough, and it releases nothing — the listeners drop memoization only (see
    {!Spoc_framework.Cache_hooks}), so unlike [Kernel.clear_cache] this cannot
    invalidate a handle a caller still holds. *)
let reset () =
  Spoc_framework.Cache_hooks.notify_clear_all () ;
  devices := [||] ;
  initialized := false

(** Pretty-print device info *)
let to_string d =
  Printf.sprintf
    "[%d] %s (%s) - %s, %.1f GB, %d MPs"
    d.id
    d.name
    d.framework
    (let major, minor = d.capabilities.compute_capability in
     if major > 0 then Printf.sprintf "SM %d.%d" major minor else d.framework)
    (Int64.to_float d.capabilities.total_global_mem
    /. (1024.0 *. 1024.0 *. 1024.0))
    d.capabilities.multiprocessor_count

(** Print all devices *)
let print_all () = all () |> Array.iter (fun d -> print_endline (to_string d))

(** {1 Phase 4: Extended Device Queries} *)

(** {2 Type Predicates} *)

let is_cuda d = is_cuda_framework d.framework

let is_opencl d = d.framework = "OpenCL"

let is_native d = d.framework = "Native"

let is_cpu d = d.capabilities.is_cpu || is_native d

let is_gpu d =
  (is_cuda d || is_opencl d || d.framework = "Vulkan" || d.framework = "HIP")
  && not (is_cpu d)

(** {2 Capability Queries} *)

let allows_fp64 d = d.capabilities.supports_fp64

let supports_atomics d = d.capabilities.supports_atomics

let compute_capability d = d.capabilities.compute_capability

let warp_size d = d.capabilities.warp_size

let max_threads_per_block d = d.capabilities.max_threads_per_block

let max_block_dims d = d.capabilities.max_block_dims

let max_grid_dims d = d.capabilities.max_grid_dims

let shared_mem_per_block d = d.capabilities.shared_mem_per_block

let total_memory d = d.capabilities.total_global_mem

let multiprocessor_count d = d.capabilities.multiprocessor_count

let clock_rate_khz d = d.capabilities.clock_rate_khz

let max_registers_per_block d = d.capabilities.max_registers_per_block

(** {2 Finders} *)

let find_cuda devices = Array.find_opt is_cuda devices

let find_opencl devices = Array.find_opt is_opencl devices

let find_native devices = Array.find_opt is_native devices

let find_by_name devices name = Array.find_opt (fun d -> d.name = name) devices

let find_by_id devices id = Array.find_opt (fun d -> d.id = id) devices

(** {2 Filters} *)

let filter_cuda () =
  all () |> Array.to_list |> List.filter is_cuda |> Array.of_list

let filter_opencl () = by_framework "OpenCL"

let filter_native () = by_framework "Native"

let with_atomics () =
  all () |> Array.to_list |> List.filter supports_atomics |> Array.of_list

let with_min_memory min_bytes =
  all () |> Array.to_list
  |> List.filter (fun d -> d.capabilities.total_global_mem >= min_bytes)
  |> Array.of_list

let with_compute_capability ~major ~minor =
  all () |> Array.to_list
  |> List.filter (fun d ->
      let maj, min = d.capabilities.compute_capability in
      maj > major || (maj = major && min >= minor))
  |> Array.of_list

(** {2 Runtime Memory Query} *)

(** Query current free memory on device (if supported by backend) *)
let free_memory d : int64 option =
  match Framework_registry.find_backend d.framework with
  | None -> None
  | Some (module B : Framework_sig.BACKEND) ->
      (* Note: Would need to extend BACKEND to support this *)
      (* For now, return None - backends can override *)
      Some d.capabilities.total_global_mem (* Placeholder *)

(** {2 Synchronization} *)

let synchronize d =
  match Framework_registry.find_backend d.framework with
  | None -> failwith ("Unknown framework: " ^ d.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let backend_dev = B.Device.get d.backend_id in
      B.Device.synchronize backend_dev

let set_current d =
  match Framework_registry.find_backend d.framework with
  | None -> failwith ("Unknown framework: " ^ d.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let backend_dev = B.Device.get d.backend_id in
      B.Device.set_current backend_dev
