(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Guarded_cache - concurrency-safe kernel-compilation cache.
 *
 * See Guarded_cache.mli for the rationale. One Mutex guards both the main
 * key -> value table and the device-id -> keys index. The expensive JIT
 * compile runs outside the lock (check under lock -> miss -> release lock ->
 * build -> re-acquire -> double-check -> insert), so multiple domains never
 * serialize on compilation, only on the (cheap) table operations.
 ******************************************************************************)

type ('k, 'v) t = {
  mutex : Mutex.t;
  table : ('k, 'v) Hashtbl.t;
  keys_by_device : (int, 'k list ref) Hashtbl.t;
  destroy : 'v -> unit;
}

let create ?(size = 16) ~destroy () =
  {
    mutex = Mutex.create ();
    table = Hashtbl.create size;
    keys_by_device = Hashtbl.create 16;
    destroy;
  }

(* Caller must hold [t.mutex]. *)
let record_key_for_device t device_id key =
  match Hashtbl.find_opt t.keys_by_device device_id with
  | Some keys -> keys := key :: !keys
  | None -> Hashtbl.add t.keys_by_device device_id (ref [key])

let find_or_build t ~key ?device_id build =
  (* Fast path: hit under the lock. *)
  Mutex.lock t.mutex ;
  match Hashtbl.find_opt t.table key with
  | Some v ->
      Mutex.unlock t.mutex ;
      v
  | None -> (
      (* Miss: release the lock so the JIT compile does not serialize other
         domains, then re-acquire to install the result. *)
      Mutex.unlock t.mutex ;
      let built = build () in
      Mutex.lock t.mutex ;
      match Hashtbl.find_opt t.table key with
      | Some winner ->
          (* Lost a concurrent compile of the same key: keep the value already
             installed and discard ours. Destroy the loser outside the lock so
             no backend resource leaks. *)
          Mutex.unlock t.mutex ;
          t.destroy built ;
          winner
      | None ->
          Hashtbl.replace t.table key built ;
          Option.iter (fun id -> record_key_for_device t id key) device_id ;
          Mutex.unlock t.mutex ;
          built)

let evict_device t device_id =
  (* Snapshot + unlink under the lock; destroy released outside it. *)
  let victims =
    Mutex.protect t.mutex (fun () ->
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
  List.iter t.destroy victims

let clear t =
  let victims =
    Mutex.protect t.mutex (fun () ->
        let vs = Hashtbl.fold (fun _ v acc -> v :: acc) t.table [] in
        Hashtbl.clear t.table ;
        Hashtbl.clear t.keys_by_device ;
        vs)
  in
  List.iter t.destroy victims

let length t = Mutex.protect t.mutex (fun () -> Hashtbl.length t.table)
