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
  (* Cache-wide generation, bumped by [clear]. Same window as [epochs], but for
     the whole cache rather than one device. Only consulted when
     [invalidated_by_clear] is set - see below. *)
  mutable generation : int;
  (* Does [clear] invalidate a build that is already in flight?
     - OWNING caches (every backend's): NO. [clear] releases the handles this
       cache owns; it does not touch the DEVICE, so a value built during the
       window is still perfectly valid and installing it is correct.
     - BORROWING caches (Runtime's outer memo, whose values are closures over
       another cache's handles): YES. For them [clear] is the announcement that
       the borrowed handles are about to be released, so a value built during
       the window closes over a handle that is dead by the time it is installed,
       and it would then be served for the rest of the process - surviving the
       very teardown the clear was performing.
     The distinction is per-cache, not global, which is why it is a create-time
     flag rather than a blanket rule. *)
  invalidated_by_clear : bool;
  destroy : 'v -> unit;
}

exception Device_destroyed_during_build of int

exception Cache_cleared_during_build

let () =
  Printexc.register_printer (function
    | Device_destroyed_during_build id ->
        Some
          (Printf.sprintf
             "Guarded_cache: device %d was destroyed while its kernel was \
              being compiled"
             id)
    | Cache_cleared_during_build ->
        Some
          "Guarded_cache: the cache was cleared while this kernel was being \
           compiled, and its value borrows handles that the clear released"
    | _ -> None)

let create ?(size = 16) ?(invalidated_by_clear = false) ~destroy () =
  {
    mutex = Mutex.create ();
    table = Hashtbl.create size;
    keys_by_device = Hashtbl.create 16;
    epochs = Hashtbl.create 16;
    generation = 0;
    invalidated_by_clear;
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
     eviction epoch and the cache generation so the post-build re-check can
     detect a teardown that happened while the lock was released. *)
  let hit_or_epoch =
    Mutex.protect t.mutex (fun () ->
        match Hashtbl.find_opt t.table key with
        | Some v -> Either.Left v
        | None ->
            Either.Right
              ( Option.map (fun id -> (id, epoch_of t id)) device_id,
                t.generation ))
  in
  match hit_or_epoch with
  | Either.Left v -> v
  | Either.Right (at_miss, generation_at_miss) -> (
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
                | _
                  when t.invalidated_by_clear
                       && t.generation <> generation_at_miss ->
                    (* Borrowing cache, cleared during the build: the handles
                       this value closes over were released by that clear, and
                       the clear has already passed over this table without
                       seeing the entry. Installing now would poison the cache
                       permanently — the value survives the very teardown that
                       was in progress. *)
                    `Cleared
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
          raise (Device_destroyed_during_build id)
      | `Cleared ->
          destroy_discarded t built ;
          raise Cache_cleared_during_build)

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
  (* [epochs] and [generation] are deliberately NOT reset: they are the only
     record that an in-flight build has been invalidated, and resetting them
     would lose the bump.

     Whether the generation bump actually invalidates an in-flight build depends
     on [invalidated_by_clear] (see the type declaration): for an OWNING cache
     the device is still alive and the value being built is still valid, so it
     is installed; for a BORROWING cache the clear is releasing the very handles
     that value closes over, so it must not be. *)
  let victims =
    Mutex.protect t.mutex (fun () ->
        t.generation <- t.generation + 1 ;
        let vs = Hashtbl.fold (fun _ v acc -> v :: acc) t.table [] in
        Hashtbl.clear t.table ;
        Hashtbl.clear t.keys_by_device ;
        vs)
  in
  destroy_all t victims

let length t = Mutex.protect t.mutex (fun () -> Hashtbl.length t.table)
