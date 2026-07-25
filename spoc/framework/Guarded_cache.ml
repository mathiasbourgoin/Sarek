(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Guarded_cache - concurrency-safe kernel-compilation cache.
 *
 * See Guarded_cache.mli for the rationale. One Mutex guards the main
 * key -> value table, the device-id -> keys index and the per-device eviction
 * epochs. The expensive JIT compile runs outside the lock (check under lock ->
 * miss -> release lock -> build -> re-acquire -> double-check -> insert), so
 * multiple domains never serialize on compilation, only on the (cheap) table
 * operations. Every critical section here uses Mutex.protect: both are
 * straight-line, and an exception escaping a raw lock/unlock pair (a Hashtbl
 * resize under Out_of_memory, Stack_overflow) would leave the mutex held and
 * deadlock every later cache operation on every domain.
 *
 * This module guards the TABLE, not the VALUES it hands out - see the
 * "Value lifetime" section of the .mli for the caller's obligation.
 ******************************************************************************)

type ('k, 'v) t = {
  mutex : Mutex.t;
  table : ('k, 'v) Hashtbl.t;
  keys_by_device : (int, 'k list ref) Hashtbl.t;
  (* Per-device eviction epoch, bumped by [evict_device]. A device teardown
     landing inside [find_or_build]'s unlocked build window is invisible to
     [evict_device] (the entry is not in [table] yet), so the epoch is read at
     miss time and re-checked on re-acquire: a change means the value was built
     for a device that no longer exists and must not be installed. *)
  epochs : (int, int) Hashtbl.t;
  destroy : 'v -> unit;
}

exception Device_destroyed_during_build of int

let () =
  Printexc.register_printer (function
    | Device_destroyed_during_build id ->
        Some
          (Printf.sprintf
             "Guarded_cache: device %d was destroyed while its kernel was \
              being compiled"
             id)
    | _ -> None)

let create ?(size = 16) ~destroy () =
  {
    mutex = Mutex.create ();
    table = Hashtbl.create size;
    keys_by_device = Hashtbl.create 16;
    epochs = Hashtbl.create 16;
    destroy;
  }

(* Caller must hold [t.mutex]. *)
let record_key_for_device t device_id key =
  match Hashtbl.find_opt t.keys_by_device device_id with
  | Some keys -> keys := key :: !keys
  | None -> Hashtbl.add t.keys_by_device device_id (ref [key])

(* Caller must hold [t.mutex]. *)
let epoch_of t device_id =
  Option.value ~default:0 (Hashtbl.find_opt t.epochs device_id)

(* Destroy every victim even if some of them raise. The victims have already
   been unlinked from the table, so abandoning the tail on the first failure
   leaks it outright — the caller has no handle left to retry with. The first
   exception is re-raised once every victim has been attempted, so a failing
   release is still reported. *)
let destroy_all t victims =
  let first_exn = ref None in
  List.iter
    (fun v ->
      try t.destroy v
      with e -> if Option.is_none !first_exn then first_exn := Some e)
    victims ;
  match !first_exn with None -> () | Some e -> raise e

(* Destroy a value we are about to discard, on a path where a failing release
   must not become the caller's result. Used for the lost-race loser: a valid
   [winner] exists and turning its return into an intermittent,
   multi-domain-only compile failure would be strictly worse than dropping the
   release error. *)
let destroy_discarded t v = try t.destroy v with _ -> ()

let find_or_build t ~key ?device_id build =
  (* Fast path: hit under the lock. On a miss, snapshot the target device's
     eviction epoch so the post-build re-check can detect a teardown that
     happened while the lock was released. *)
  let hit_or_epoch =
    Mutex.protect t.mutex (fun () ->
        match Hashtbl.find_opt t.table key with
        | Some v -> Either.Left v
        | None ->
            Either.Right (Option.map (fun id -> (id, epoch_of t id)) device_id))
  in
  match hit_or_epoch with
  | Either.Left v -> v
  | Either.Right at_miss -> (
      (* Miss: the JIT compile runs with the lock released so it does not
         serialize other domains. *)
      let built = build () in
      let outcome =
        Mutex.protect t.mutex (fun () ->
            match Hashtbl.find_opt t.table key with
            | Some winner -> `Lost winner
            | None -> (
                match at_miss with
                | Some (id, epoch) when epoch_of t id <> epoch ->
                    (* [id] was destroyed (or fully evicted) during the build:
                       installing now would record the value against a dead
                       device id, so nothing would ever unload it and a later
                       lookup — possibly after the id is recreated — would be
                       served a value from the old context. *)
                    `Stale id
                | _ ->
                    Hashtbl.replace t.table key built ;
                    Option.iter
                      (fun id -> record_key_for_device t id key)
                      device_id ;
                    `Installed))
      in
      (* [destroy] always runs outside the lock. *)
      match outcome with
      | `Installed -> built
      | `Lost winner ->
          destroy_discarded t built ;
          winner
      | `Stale id ->
          destroy_discarded t built ;
          raise (Device_destroyed_during_build id))

let evict_device t device_id =
  (* Snapshot + unlink under the lock; destroy released outside it. The epoch
     bump happens under the same lock, so a build that is in flight for this
     device observes it on re-acquire and discards its result. *)
  let victims =
    Mutex.protect t.mutex (fun () ->
        Hashtbl.replace t.epochs device_id (epoch_of t device_id + 1) ;
        match Hashtbl.find_opt t.keys_by_device device_id with
        | None -> []
        | Some keys ->
            let vs =
              List.filter_map
                (fun key ->
                  match Hashtbl.find_opt t.table key with
                  | None -> None
                  | Some v ->
                      Hashtbl.remove t.table key ;
                      Some v)
                !keys
            in
            Hashtbl.remove t.keys_by_device device_id ;
            vs)
  in
  destroy_all t victims

let clear t =
  (* [epochs] is deliberately NOT reset: the counters are the only record that
     an in-flight build's device was retired, and clearing them could lose a
     bump. [clear] itself does not invalidate an in-flight build — the device is
     still alive, so the value being built is still valid and may be installed. *)
  let victims =
    Mutex.protect t.mutex (fun () ->
        let vs = Hashtbl.fold (fun _ v acc -> v :: acc) t.table [] in
        Hashtbl.clear t.table ;
        Hashtbl.clear t.keys_by_device ;
        vs)
  in
  destroy_all t victims

let length t = Mutex.protect t.mutex (fun () -> Hashtbl.length t.table)
