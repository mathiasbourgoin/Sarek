(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Cache_hooks - cross-layer cache-teardown notifications.
 *
 * See Cache_hooks.mli for the rationale. The registries are plain lists behind
 * a Mutex; notification snapshots the list under the lock and runs the
 * callbacks outside it, so a callback may itself register or touch a cache.
 ******************************************************************************)

let mutex = Mutex.create ()

let device_destroy_hooks : (backend:string -> int -> unit) list ref = ref []

let clear_all_hooks : (unit -> unit) list ref = ref []

let on_device_destroy f =
  Mutex.protect mutex (fun () ->
      device_destroy_hooks := f :: !device_destroy_hooks)

let on_clear_all f =
  Mutex.protect mutex (fun () -> clear_all_hooks := f :: !clear_all_hooks)

(* Every hook runs even if an earlier one raises: hooks are independent cache
   layers, and skipping the rest would leave some of them holding handles the
   caller is about to release. The first exception is re-raised once all hooks
   have run, so a failure is still reported rather than swallowed. *)
let run_all hooks =
  let first_exn = ref None in
  List.iter
    (fun h ->
      try h () with e -> if Option.is_none !first_exn then first_exn := Some e)
    hooks ;
  match !first_exn with None -> () | Some e -> raise e

let notify_device_destroy ~backend device_id =
  let hooks = Mutex.protect mutex (fun () -> !device_destroy_hooks) in
  run_all (List.map (fun h () -> h ~backend device_id) hooks)

let notify_clear_all () =
  let hooks = Mutex.protect mutex (fun () -> !clear_all_hooks) in
  run_all hooks

(* Nesting depth of [around_clear] on THIS domain. Per-domain, not global: a
   concurrent clear on another domain is a genuinely separate teardown and must
   still notify. Within one domain the nesting is always
   [Sarek.Kernel.clear_cache] -> [B.Kernel.clear_cache], and both would
   otherwise notify, giving four notifications where the contract says two. *)
let clear_depth = Domain.DLS.new_key (fun () -> 0)

let around_clear f =
  let depth = Domain.DLS.get clear_depth in
  if depth > 0 then f ()
  else begin
    Domain.DLS.set clear_depth (depth + 1) ;
    let listener_exn = ref None in
    let notify () =
      try notify_clear_all ()
      with e -> if Option.is_none !listener_exn then listener_exn := Some e
    in
    notify () ;
    Fun.protect
      ~finally:(fun () ->
        notify () ;
        Domain.DLS.set clear_depth depth)
      f ;
    Option.iter raise !listener_exn
  end
