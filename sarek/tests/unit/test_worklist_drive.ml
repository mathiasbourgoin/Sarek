(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Regression test for audit finding M8 — Sarek_worklist.Host.drive must stop
 * at the first OVERFLOW (Sarek_worklist.ml:208, the `|| overflow t > 0`
 * disjunct added by 473478c6).
 *
 * WHY THIS FILE EXISTS
 *   The external audit of 2026-07-29 confirmed the M8 fix landed on main and
 *   flagged it as pinned by nothing: reverting that line left the whole suite
 *   green. The one existing worklist test (sarek/tests/e2e/test_worklist.ml)
 *   does exercise the OVERFLOW flag, but only through the single-launch
 *   `push_guarded` scenario — it never calls `drive` on an overflowing queue,
 *   which is the only place the guard lives. It also needs a device, so it does
 *   not run at all on a CPU-only machine.
 *
 * WHAT IS UNDER TEST, AND WHY NO DEVICE IS NEEDED
 *   `drive` is pure host logic: it reads the device counters and calls a
 *   caller-supplied [launch]. So the device is replaced here by an OCaml
 *   function that reproduces exactly the counter arithmetic of the documented
 *   level-synchronous frontier kernel (the one in the library's own header):
 *
 *     t = TAIL++ ; if t - HEAD >= cap then OVERFLOW++ else slots[t mod cap] <- child
 *
 *   HEAD is never advanced by the level-sync pattern, so a push overflows
 *   exactly when its ticket reaches [cap].
 *
 * THE HARM BEING PINNED (both polarities)
 *   Positive — the guard fires: once OVERFLOW is set, TAIL has counted tickets
 *   whose ring slot was never written. The window [base, TAIL) handed to the
 *   next level therefore contains ring positions holding something other than
 *   the item that ticket logically stands for: with HEAD at 0 those positions
 *   lap onto already-processed slots, so the next level re-processes items it
 *   has already consumed (and TAIL keeps growing, so it does so for as many
 *   levels as `max_levels` allows). `bad_tickets` and `visits` below assert
 *   exactly that: no launch is ever handed an unwritten ticket, and no item is
 *   processed twice.
 *
 *   (The fix's own comment says the poisoned positions still hold the -1 seed.
 *   That is the shape for a caller whose kernel advances HEAD; in the pure
 *   level-sync flow the guard is [t - 0 >= cap], so slots 0..cap-1 are all
 *   written by the time OVERFLOW can be set and the garbage is a stale
 *   duplicate rather than -1. The assertions here are written on "was this
 *   ticket ever written", which covers both.)
 *
 *   Negative — the guard must not fire otherwise: `no_overflow_runs_to_fixpoint`
 *   drives a queue that never overflows and pins the exact launch sequence and
 *   full node coverage. A wrong-direction correction (`overflow t >= 0`, or
 *   hoisting the check so it also cuts the first level) makes that case red.
 *   `max_levels_still_respected` pins the pre-existing bound the new disjunct
 *   sits next to.
 ******************************************************************************)

module WL = Sarek_worklist
module Host = Sarek_worklist.Host
module Ctrl = Sarek_worklist.Ctrl
module Vector = Spoc_core.Vector

(** A stand-in for the device: runs the documented level-synchronous frontier
    push/pop arithmetic on the host, and records everything the assertions need.
*)
type sim = {
  q : Host.t;
  cap : int;
  fanout : int -> int;
  next_id : int ref;
  written : (int, unit) Hashtbl.t;
      (** tickets whose ring slot actually received a value *)
  launches : (int * int) list ref;  (** windows handed to [launch], in order *)
  bad_tickets : (int * int * int) list ref;
      (** (base, snap, ticket) for every window entry that was never written *)
  visits : int list ref;  (** items processed, in order *)
}

let make_sim ~cap ~fanout ~seed =
  let q = Host.create ~capacity:cap in
  Host.seed q (Array.map Int32.of_int seed) ;
  let written = Hashtbl.create 64 in
  (* Seeding writes the first |seed| ring slots, i.e. tickets 0..n-1. *)
  Array.iteri (fun i _ -> Hashtbl.replace written i ()) seed ;
  {
    q;
    cap;
    fanout;
    next_id = ref (Array.length seed);
    written;
    launches = ref [];
    bad_tickets = ref [];
    visits = ref [];
  }

(** One frontier "kernel launch" over the half-open window from [level_base] up
    to [snapshot_tail]. *)
let sim_launch s ~level_base ~snapshot_tail =
  s.launches := (level_base, snapshot_tail) :: !(s.launches) ;
  for i = level_base to snapshot_tail - 1 do
    if not (Hashtbl.mem s.written i) then
      s.bad_tickets := (level_base, snapshot_tail, i) :: !(s.bad_tickets)
  done ;
  for i = level_base to snapshot_tail - 1 do
    let u = Int32.to_int (Vector.get s.q.Host.slots (i mod s.cap)) in
    s.visits := u :: !(s.visits) ;
    for _ = 1 to s.fanout u do
      (* t = atomic_add(TAIL, 1); head = TAIL-unrelated read of HEAD *)
      let t = Host.tail s.q in
      Host.set_ctrl s.q Ctrl.tail (t + 1) ;
      let head = Host.head s.q in
      if t - head >= s.cap then
        Host.set_ctrl s.q Ctrl.overflow (Host.overflow s.q + 1)
      else begin
        let child = !(s.next_id) in
        incr s.next_id ;
        Vector.set s.q.Host.slots (t mod s.cap) (Int32.of_int child) ;
        Hashtbl.replace s.written t ()
      end
    done
  done

let drive_sim s ~max_levels =
  Host.drive
    s.q
    ~launch:(fun ~level_base ~snapshot_tail ->
      sim_launch s ~level_base ~snapshot_tail)
    ~max_levels

(* Both are accumulated head-first (appending with [@] is quadratic, and the
   reverted-fix run below launches levels whose windows grow by a factor of the
   fanout - it must fail on an assertion, not by taking forever). *)
let launches s = List.rev !(s.launches)

let visits s = List.rev !(s.visits)

(* ------------------------------------------------------------------ *)

let windows = Alcotest.(list (pair int int))

let has_duplicate l =
  let seen = Hashtbl.create 64 in
  List.exists
    (fun x ->
      if Hashtbl.mem seen x then true
      else (
        Hashtbl.replace seen x () ;
        false))
    l

(** POSITIVE POLARITY: a ring too small for the frontier. Level 1 overflows, so
    level 2 must never be launched. *)
let overflow_stops_the_driver () =
  (* cap=8, fanout 3, seed {0}:
     L0 window (0,1): 3 pushes, tickets 1..3   -> TAIL=4
     L1 window (1,4): 9 pushes, tickets 4..12; 8..12 overflow -> TAIL=13
     L2 would be window (4,13), whose tickets 8..12 were never written. *)
  let s = make_sim ~cap:8 ~fanout:(fun _ -> 3) ~seed:[|0|] in
  let levels = drive_sim s ~max_levels:4 in
  (* max_levels is 4, deliberately above the expected 2: the OVERFLOW guard has
     to be what stops the driver, not the bound. *)
  Alcotest.(check int) "levels launched before the overflow" 2 levels ;
  Alcotest.check
    windows
    "exactly the two pre-overflow windows"
    [(0, 1); (1, 4)]
    (launches s) ;
  (* The overflow really happened - without this the case above could pass by
     the frontier simply having drained. *)
  Alcotest.(check bool)
    "overflow flag is set for the caller to observe"
    true
    (Host.overflow s.q > 0) ;
  (* BAD BEHAVIOUR ABSENT: no launch was handed a ring position that no push
     ever wrote, and nothing was processed twice. *)
  Alcotest.(check (list (triple int int int)))
    "no launch window contains an unwritten ticket"
    []
    !(s.bad_tickets) ;
  Alcotest.(check bool)
    "no item processed twice"
    false
    (has_duplicate (visits s))

(** POSITIVE POLARITY, minimal form: OVERFLOW already set on entry. Not one
    launch may happen. *)
let overflow_set_on_entry_launches_nothing () =
  let s = make_sim ~cap:8 ~fanout:(fun _ -> 3) ~seed:[|0|] in
  Host.set_ctrl s.q Ctrl.overflow 1 ;
  let levels = drive_sim s ~max_levels:4 in
  Alcotest.(check int) "no level launched" 0 levels ;
  Alcotest.check windows "no window handed to launch" [] (launches s)

(** NEGATIVE POLARITY: the guard must not cut a healthy run short. A 15-node
    binary tree in a ring of 64 - no overflow is possible - must run to its
    fixpoint and cover every node exactly once. *)
let no_overflow_runs_to_fixpoint () =
  let s =
    make_sim ~cap:64 ~fanout:(fun u -> if u < 7 then 2 else 0) ~seed:[|0|]
  in
  let levels = drive_sim s ~max_levels:20 in
  Alcotest.(check int) "all four levels launched" 4 levels ;
  Alcotest.check
    windows
    "the full level-synchronous window sequence"
    [(0, 1); (1, 3); (3, 7); (7, 15)]
    (launches s) ;
  Alcotest.(check int) "overflow never set" 0 (Host.overflow s.q) ;
  Alcotest.(check (list int))
    "every node visited exactly once, in BFS order"
    (List.init 15 (fun i -> i))
    (visits s) ;
  Alcotest.(check (list (triple int int int)))
    "no unwritten ticket"
    []
    !(s.bad_tickets)

(** The pre-existing bound the overflow disjunct sits beside. *)
let max_levels_still_respected () =
  let s =
    make_sim ~cap:64 ~fanout:(fun u -> if u < 7 then 2 else 0) ~seed:[|0|]
  in
  let levels = drive_sim s ~max_levels:2 in
  Alcotest.(check int) "stopped at max_levels" 2 levels ;
  Alcotest.check
    windows
    "only the first two windows"
    [(0, 1); (1, 3)]
    (launches s)

let () =
  Alcotest.run
    "Sarek_worklist.Host.drive"
    [
      ( "M8 overflow guard",
        [
          Alcotest.test_case
            "overflow stops the driver"
            `Quick
            overflow_stops_the_driver;
          Alcotest.test_case
            "overflow set on entry launches nothing"
            `Quick
            overflow_set_on_entry_launches_nothing;
        ] );
      ( "guard is not over-broad",
        [
          Alcotest.test_case
            "no overflow runs to fixpoint"
            `Quick
            no_overflow_runs_to_fixpoint;
          Alcotest.test_case
            "max_levels still respected"
            `Quick
            max_levels_still_respected;
        ] );
    ]
