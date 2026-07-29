(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Regression test for audit finding L1 — the process-global tuple-shape
 * registry in Sarek_tuple_vec must resolve a concurrent first-time build of one
 * shape to a SINGLE winner.
 *
 * WHY THIS FILE EXISTS
 *   The 2026-07-29 audit records L1 as "fixed more thoroughly than asked — no
 *   test". Three commits are involved:
 *     ac896b73  mutex around the build-and-insert in [memoize]
 *     3aa4c1b9  drop the unlocked [find_opt] fast path in front of it
 *     339c8e2c  take the same mutex for [lookup_shape] reads
 *
 * WHAT THIS TEST DECIDES, AND WHAT IT CANNOT
 *   Only the FIRST of the three has an observable functional consequence, and
 *   that is what is pinned here. [memoize] hands out a [custom_type] carrying a
 *   freshly generated [Type_id]; the Native path later resolves the same shape
 *   through [descriptor_by_name] and compares with [Type_id.equal]. With
 *   build-and-insert unguarded, N domains racing on a shape that does not exist
 *   yet each build their own descriptor and only one reaches the table, so
 *   every other caller holds a descriptor that will fail that equality check —
 *   an observable, assertable divergence, which is what [single_winner_*] below
 *   asserts.
 *
 *   The other two are pure MEMORY-MODEL properties: an unlocked [Hashtbl]
 *   read concurrent with a [replace] is a data race under OCaml 5 whether or
 *   not it produces a wrong answer on any given run. There is no deterministic
 *   OCaml-level assertion for "this access was correctly synchronised" — the
 *   sound checker is a race detector (a TSan-instrumented runtime), which the
 *   OCaml 5 toolchain does not ship. The stress loop below does hammer
 *   [lookup_shape] from every domain while registrations are in flight, so a
 *   corrupted table or a torn read shows up as a wrong value or an exception,
 *   but that is a probabilistic bonus and NOT the gate: reverting 3aa4c1b9 or
 *   the [lookup_shape] lock alone is expected to leave this test green, and
 *   that limit is stated here rather than papered over.
 *
 * NON-VACUITY
 *   The registry is process-global and each shape can only be raced ONCE per
 *   executable (afterwards it is memoized and there is nothing left to race).
 *   [fresh_before_the_race] therefore asserts, for every shape, that it was
 *   genuinely unregistered before the domains started — without it this whole
 *   file would degrade into a memoization test the moment a shape got built
 *   during startup.
 ******************************************************************************)

module TV = Sarek_tuple_vec
module Vector = Spoc_core.Vector
module Type_id = Sarek_ir_types.Type_id

let domains = max 4 (min 8 (Domain.recommended_domain_count ()))

(** Run [f] on [domains] domains that all leave a spin barrier together, so the
    first-time build really is concurrent rather than serialized by startup. *)
let race (f : unit -> 'r) : 'r list =
  let arrived = Atomic.make 0 in
  let go = Atomic.make false in
  let spawn () =
    Domain.spawn (fun () ->
        Atomic.incr arrived ;
        while not (Atomic.get go) do
          Domain.cpu_relax ()
        done ;
        f ())
  in
  let ds = List.init domains (fun _ -> spawn ()) in
  while Atomic.get arrived < domains do
    Domain.cpu_relax ()
  done ;
  Atomic.set go true ;
  List.map Domain.join ds

(** Every domain's descriptor must be the one canonical descriptor: same
    [Type_id], same [vector_type_id], same identity as the one
    [descriptor_by_name] resolves for generated Native code. *)
let check_single_winner (type a) ~name (results : a Vector.custom_type list) =
  let canonical = TV.descriptor_by_name name in
  let same_type_id d =
    match Type_id.equal d.Vector.type_id canonical.Vector.type_id with
    | Some Type_id.Refl -> true
    | None -> false
  in
  let same_vector_type_id d =
    match
      Type_id.equal d.Vector.vector_type_id canonical.Vector.vector_type_id
    with
    | Some Type_id.Refl -> true
    | None -> false
  in
  Alcotest.(check int)
    (name ^ ": every domain returned a descriptor")
    domains
    (List.length results) ;
  Alcotest.(check int)
    (name ^ ": no domain holds a descriptor with a divergent element Type_id")
    0
    (List.length (List.filter (fun d -> not (same_type_id d)) results)) ;
  Alcotest.(check int)
    (name ^ ": no domain holds a descriptor with a divergent vector Type_id")
    0
    (List.length (List.filter (fun d -> not (same_vector_type_id d)) results)) ;
  (* Physical identity is the stronger statement and the one the memo table is
     actually meant to provide. *)
  Alcotest.(check int)
    (name ^ ": every domain got the physically same descriptor")
    0
    (List.length (List.filter (fun d -> not (d == canonical)) results)) ;
  (* The byte layout was registered once, not N times with divergent content. *)
  match TV.lookup_shape name with
  | None -> Alcotest.fail (name ^ ": shape layout was never registered")
  | Some sh ->
      Alcotest.(check string) (name ^ ": shape name") name sh.TV.sh_name ;
      Alcotest.(check int)
        (name ^ ": shape size agrees with the descriptor")
        canonical.Vector.elem_size
        sh.TV.sh_size

(** Guards against the whole race being a no-op: the shape must not exist yet.
*)
let fresh_before_the_race name =
  Alcotest.(check bool)
    (name ^ ": unregistered before the race (else nothing is being raced)")
    true
    (TV.lookup_shape name = None)

(* [lookup_shape] hammering, run inside every racing domain. Not a gate (see the
   header) - it exists so that a table corrupted by an unsynchronised read has
   somewhere to surface. *)
let hammer_lookups name =
  for _ = 1 to 200 do
    match TV.lookup_shape name with
    | None -> ()
    | Some sh -> if sh.TV.sh_name <> name then failwith "torn shape read"
  done

let race_pair : type a b. string -> a TV.component -> b TV.component -> unit =
 fun name ca cb ->
  fresh_before_the_race name ;
  let results =
    race (fun () ->
        let d = TV.pair ca cb in
        hammer_lookups name ;
        d)
  in
  check_single_winner ~name results

let race_triple : type a b c.
    string -> a TV.component -> b TV.component -> c TV.component -> unit =
 fun name ca cb cc ->
  fresh_before_the_race name ;
  let results =
    race (fun () ->
        let d = TV.triple ca cb cc in
        hammer_lookups name ;
        d)
  in
  check_single_winner ~name results

(* Sixteen pair shapes and four triple shapes. One shape can only be raced once
   per process, so the count is the number of independent attempts this test
   gets - a single attempt would make the outcome a coin flip. *)
let single_winner_pairs () =
  let comps =
    [("float32", `F32); ("float64", `F64); ("int32", `I32); ("int64", `I64)]
  in
  List.iter
    (fun (ta, ca) ->
      List.iter
        (fun (tb, cb) ->
          let name = Printf.sprintf "_tup_%s_%s" ta tb in
          match (ca, cb) with
          | `F32, `F32 -> race_pair name TV.float32 TV.float32
          | `F32, `F64 -> race_pair name TV.float32 TV.float64
          | `F32, `I32 -> race_pair name TV.float32 TV.int32
          | `F32, `I64 -> race_pair name TV.float32 TV.int64
          | `F64, `F32 -> race_pair name TV.float64 TV.float32
          | `F64, `F64 -> race_pair name TV.float64 TV.float64
          | `F64, `I32 -> race_pair name TV.float64 TV.int32
          | `F64, `I64 -> race_pair name TV.float64 TV.int64
          | `I32, `F32 -> race_pair name TV.int32 TV.float32
          | `I32, `F64 -> race_pair name TV.int32 TV.float64
          | `I32, `I32 -> race_pair name TV.int32 TV.int32
          | `I32, `I64 -> race_pair name TV.int32 TV.int64
          | `I64, `F32 -> race_pair name TV.int64 TV.float32
          | `I64, `F64 -> race_pair name TV.int64 TV.float64
          | `I64, `I32 -> race_pair name TV.int64 TV.int32
          | `I64, `I64 -> race_pair name TV.int64 TV.int64)
        comps)
    comps

let single_winner_triples () =
  race_triple "_tup_float32_int32_int64" TV.float32 TV.int32 TV.int64 ;
  race_triple "_tup_float64_float32_int32" TV.float64 TV.float32 TV.int32 ;
  race_triple "_tup_int32_int32_int32" TV.int32 TV.int32 TV.int32 ;
  race_triple "_tup_int64_float64_float64" TV.int64 TV.float64 TV.float64

(** Once the shape exists, every later caller keeps getting the same descriptor
    - the memoization half of the contract, single-threaded and deterministic.
*)
let memoized_after_the_race () =
  let name = "_tup_float32_int32" in
  let d1 = TV.pair TV.float32 TV.int32 in
  let d2 = TV.pair TV.float32 TV.int32 in
  Alcotest.(check bool) "repeat build is the same descriptor" true (d1 == d2) ;
  Alcotest.(check bool)
    "descriptor_by_name agrees"
    true
    (d1 == TV.descriptor_by_name name)

(** An unbuilt shape must fail loudly rather than hand back a wrong descriptor.
*)
let unbuilt_shape_refuses () =
  Alcotest.(check bool)
    "unbuilt shape has no layout"
    true
    (TV.lookup_shape "_tup_never_built" = None) ;
  match TV.descriptor_by_name "_tup_never_built" with
  | _ ->
      Alcotest.fail
        "descriptor_by_name returned a descriptor for an unbuilt shape"
  | exception Failure msg ->
      Alcotest.(check bool)
        "the failure names the shape"
        true
        (String.length msg > 0
        &&
        let re = "_tup_never_built" in
        let rec find i =
          i + String.length re <= String.length msg
          && (String.sub msg i (String.length re) = re || find (i + 1))
        in
        find 0)

let () =
  Alcotest.run
    "Sarek_tuple_vec registry (audit L1)"
    [
      ( "single winner under concurrent first-time build",
        [
          Alcotest.test_case "16 pair shapes" `Quick single_winner_pairs;
          Alcotest.test_case "4 triple shapes" `Quick single_winner_triples;
        ] );
      ( "deterministic registry contract",
        [
          Alcotest.test_case
            "memoized after the race"
            `Quick
            memoized_after_the_race;
          Alcotest.test_case
            "unbuilt shape refuses"
            `Quick
            unbuilt_shape_refuses;
        ] );
    ]
