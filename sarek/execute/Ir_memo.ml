(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Ir_memo - one record of launch-path facts per IR, by physical identity
 *
 * Every per-IR table of the launch path (capability verdicts and sources in
 * Execute, the window's ir_info, the matrix plan) is a typed slot of one
 * record, so a launch hashes its IR once to find all of them, and not at all
 * when it launches the same IR as the previous launch on its domain.
 *
 * LIFETIME. The table is weak in the IR (an ephemeron) and so is the
 * per-domain cache of the last lookup, so a record dies with its kernel. A
 * slot's value must not refer to the IR. Owners invalidate their slots
 * themselves (a generation number in the value); the record is never cleared.
 *
 * CONCURRENCY. One mutex guards the table. A record's slots are an immutable
 * list behind an atomic: reads take no lock, and a write replaces the list by
 * compare-and-set. The value in a slot is guarded by its owner.
 ******************************************************************************)

open Sarek_ir_types

let hashes = Atomic.make 0

(** Every hash of an IR by the launch-path tables; a test hook. *)
let hash_count () = Atomic.get hashes

(** Physical identity of an IR, for [Ephemeron.K1.Make]. *)
module Physical = struct
  type t = kernel

  let equal = ( == )

  (* Any function of the contents is consistent with [==]. [Hashtbl.hash]
     reads a bounded prefix, so this is cheap; a downstream consumer names every kernel
     "sarek_kern", so the parameter list is what spreads them. *)
  let hash (k : t) =
    Atomic.incr hashes ;
    Hashtbl.hash (k.kern_name, k.kern_params)
end

module Table = Ephemeron.K1.Make (Physical)

(** A typed slot. Each owner makes its keys once ([new_key]). *)
type 'a key = 'a Type.Id.t

let new_key () : 'a key = Type.Id.make ()

type binding = Binding : 'a key * 'a -> binding

(** The facts recorded for one IR. *)
type facts = {slots : binding list Atomic.t}

let table : facts Table.t = Table.create 64

let lock = Mutex.create ()

let lookup (ir : kernel) : facts =
  Mutex.protect lock (fun () ->
      match Table.find_opt table ir with
      | Some f -> f
      | None ->
          let f = {slots = Atomic.make []} in
          Table.replace table ir f ;
          f)

(* The last lookup of this domain. *)
let last : (kernel, facts) Ephemeron.K1.t option Domain.DLS.key =
  Domain.DLS.new_key (fun () -> None)

(** The record of [ir], created empty on first use. *)
let facts (ir : kernel) : facts =
  let cached =
    match Domain.DLS.get last with
    | Some eph -> Ephemeron.K1.query eph ir
    | None -> None
  in
  match cached with
  | Some f -> f
  | None ->
      let f = lookup ir in
      Domain.DLS.set last (Some (Ephemeron.K1.make ir f)) ;
      f

(** The value in slot [key] of [facts], if any. *)
let find (type a) (facts : facts) (key : a key) : a option =
  let rec go : binding list -> a option = function
    | [] -> None
    | Binding (k, v) :: rest -> (
        match Type.Id.provably_equal key k with
        | Some Type.Equal -> Some v
        | None -> go rest)
  in
  go (Atomic.get facts.slots)

(** Put [v] in slot [key] of [facts], replacing any value there. *)
let set (type a) (facts : facts) (key : a key) (v : a) : unit =
  let id = Type.Id.uid key in
  let rec loop () =
    let old = Atomic.get facts.slots in
    let others =
      List.filter (fun (Binding (k, _)) -> Type.Id.uid k <> id) old
    in
    if not (Atomic.compare_and_set facts.slots old (Binding (key, v) :: others))
    then loop ()
  in
  loop ()
