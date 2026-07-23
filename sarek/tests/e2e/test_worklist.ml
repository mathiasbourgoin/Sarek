(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek_worklist - portable dynamic parallelism.
 *
 * The library source is compiled to device code by the Sarek PPX and pulled in
 * with %sarek_include, so the SAME source runs on every backend - including
 * CUDA/PTX under ZLUDA. Scenarios (each verified against a CPU reference on
 * every available device):
 *   A. BFS variable-fanout tree frontier sum (level-synchronous, Host.drive).
 *      With far fewer threads than nodes it also proves the claim-loop re-pops.
 *   B. Deterministic tiny tree (order-independent sum, hand-checked).
 *   C. Push with a small ring -> OVERFLOW flag set; non-dropped items correct.
 *   D. Ring wrap-around: persistent single-thread linked list, capacity << N.
 ******************************************************************************)

open Sarek
module Std = Sarek_stdlib.Std
module Gpu = Sarek_stdlib.Gpu
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module WL = Sarek_worklist

let%sarek_include _ = "../../Sarek_worklist/Sarek_worklist.ml"

(* ============================ kernels ============================ *)

(* One level of a variable-fanout BFS frontier. Each thread claims tickets from
   [0, snapshot_tail) via the shared HEAD counter, adds the node value to a
   global accumulator, and pushes the node's children to TAIL for the next
   level. Uses the callable pure helper wl_ring_index. *)
let frontier_kernel =
  [%kernel
    fun (ctrl : int32 vector)
        (slots : int32 vector)
        (values : int32 vector)
        (child_off : int32 vector)
        (child_idx : int32 vector)
        (acc : int32 vector)
        (level_base : int32)
        (snapshot_tail : int32)
        (cap : int32) ->
      let open Std in
      let open Gpu in
      let open Sarek_worklist in
      let stride = block_dim_x * grid_dim_x in
      let i = mut (level_base + thread_idx_x + (block_idx_x * block_dim_x)) in
      while i < snapshot_tail do
        let u = slots.(wl_ring_index cap i) in
        let _ = atomic_add_global_int32 acc 0l values.(u) in
        let s = mut child_off.(u) in
        let e = child_off.(u + 1l) in
        while s < e do
          let t = atomic_add_global_int32 ctrl 1l 1l in
          slots.(wl_ring_index cap t) <- child_idx.(s) ;
          s := s + 1l
        done ;
        i := i + stride
      done]

(* Each thread pushes one item into a ring that may be too small; a bounds-check
   against HEAD sets the OVERFLOW flag instead of clobbering a live slot. *)
let push_guarded_kernel =
  [%kernel
    fun (ctrl : int32 vector)
        (slots : int32 vector)
        (input : int32 vector)
        (n : int32)
        (cap : int32) ->
      let open Std in
      let open Gpu in
      let open Sarek_worklist in
      let tid = thread_idx_x + (block_idx_x * block_dim_x) in
      if tid < n then begin
        let t = atomic_add_global_int32 ctrl 1l 1l in
        let head = atomic_add_global_int32 ctrl 0l 0l in
        if t - head >= cap then
          let _ = atomic_add_global_int32 ctrl 3l 1l in
          ()
        else slots.(wl_ring_index cap t) <- input.(tid)
      end]

(* Persistent single-launch traversal of a linked list by ONE thread: pop, sum,
   push the (single) successor, until the queue drains. A tiny ring wraps
   many times; live set is always 1 so no live slot is ever overwritten. *)
let persistent_wrap_kernel =
  [%kernel
    fun (ctrl : int32 vector)
        (slots : int32 vector)
        (values : int32 vector)
        (nxt : int32 vector)
        (acc : int32 vector)
        (cap : int32) ->
      let open Std in
      let open Gpu in
      let open Sarek_worklist in
      let cont = mut 1l in
      while cont = 1l do
        let tl = ctrl.(1) in
        let h = atomic_add_global_int32 ctrl 0l 1l in
        if h >= tl then cont := 0l
        else begin
          let u = slots.(wl_ring_index cap h) in
          let _ = atomic_add_global_int32 acc 0l values.(u) in
          let c = nxt.(u) in
          if c >= 0l then begin
            let t = atomic_add_global_int32 ctrl 1l 1l in
            slots.(wl_ring_index cap t) <- c
          end
        end
      done]

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

(* ============================ host helpers ============================ *)

type tree = {
  n : int;
  values : int32 array;
  child_off : int32 array; (* length n+1 *)
  child_idx : int32 array;
}

(* Build a variable-fanout tree over nodes [0, n): node i gets [i mod max_fan]
   children taken from the not-yet-assigned nodes, so every node is reachable
   from root 0 exactly once (a genuine tree). *)
let build_tree n ~max_fan =
  let fan = Array.make n 0 in
  let next = ref 1 in
  for i = 0 to n - 1 do
    (* At least one child (varying 1..max_fan) until all nodes are placed, so
       the root really expands and every node is reachable exactly once. *)
    let want = 1 + (i mod max_fan) in
    let f = if !next >= n then 0 else min want (n - !next) in
    fan.(i) <- f ;
    next := !next + f
  done ;
  let child_off = Array.make (n + 1) 0l in
  let child_idx = Array.make (n - 1) 0l in
  let cursor = ref 1 in
  let pos = ref 0 in
  for i = 0 to n - 1 do
    child_off.(i) <- Int32.of_int !pos ;
    for _ = 1 to fan.(i) do
      child_idx.(!pos) <- Int32.of_int !cursor ;
      incr cursor ;
      incr pos
    done
  done ;
  child_off.(n) <- Int32.of_int !pos ;
  let values = Array.init n (fun i -> Int32.of_int (i + 1)) in
  {n; values; child_off; child_idx}

(* CPU reference: BFS from root 0, summing visited node values. Mirrors the
   kernel's queue semantics exactly (order-independent result). *)
let cpu_frontier_sum tree =
  let q = Queue.create () in
  Queue.push 0 q ;
  let sum = ref 0 in
  while not (Queue.is_empty q) do
    let u = Queue.pop q in
    sum := !sum + Int32.to_int tree.values.(u) ;
    let s = Int32.to_int tree.child_off.(u) in
    let e = Int32.to_int tree.child_off.(u + 1) in
    for k = s to e - 1 do
      Queue.push (Int32.to_int tree.child_idx.(k)) q
    done
  done ;
  !sum

let vec_of_int32_array a =
  let v = Vector.create Vector.int32 (Array.length a) in
  Array.iteri (fun i x -> Vector.set v i x) a ;
  v

(* ============================ scenarios ============================ *)

(* A/B: run the level-synchronous frontier on [tree]; return the summed value.
   [threads] is deliberately small to force claim-loop re-pops. *)
let run_frontier dev tree ~threads =
  let cap = (2 * tree.n) + 8 in
  let q = WL.Host.create ~capacity:cap in
  WL.Host.seed q [|0l|] ;
  let values = vec_of_int32_array tree.values in
  let child_off = vec_of_int32_array tree.child_off in
  let child_idx =
    if Array.length tree.child_idx = 0 then Vector.create Vector.int32 1
    else vec_of_int32_array tree.child_idx
  in
  let acc = Vector.create Vector.int32 1 in
  Vector.set acc 0 0l ;
  let ir = ir_of frontier_kernel in
  let launch ~level_base ~snapshot_tail =
    Execute.run_vectors
      ~device:dev
      ~ir
      ~args:
        [
          Execute.Vec q.WL.Host.ctrl;
          Execute.Vec q.WL.Host.slots;
          Execute.Vec values;
          Execute.Vec child_off;
          Execute.Vec child_idx;
          Execute.Vec acc;
          Execute.Int level_base;
          Execute.Int snapshot_tail;
          Execute.Int cap;
        ]
      ~block:(Execute.dims1d threads)
      ~grid:(Execute.dims1d 1)
      () ;
    Transfer.flush dev
  in
  let _levels = WL.Host.drive q ~launch ~max_levels:(tree.n + 2) in
  Transfer.flush dev ;
  let got = Int32.to_int (Vector.get acc 0) in
  (got, WL.Host.overflow q)

(* C: push [n] items into a ring of [cap] < n; expect the overflow flag set and
   every non-dropped slot to hold a distinct input value. *)
let run_overflow dev =
  let n = 256 in
  let cap = 64 in
  let input =
    vec_of_int32_array (Array.init n (fun i -> Int32.of_int (i + 1)))
  in
  let q = WL.Host.create ~capacity:cap in
  WL.Host.seed q [||] ;
  Execute.run_vectors
    ~device:dev
    ~ir:(ir_of push_guarded_kernel)
    ~args:
      [
        Execute.Vec q.WL.Host.ctrl;
        Execute.Vec q.WL.Host.slots;
        Execute.Vec input;
        Execute.Int n;
        Execute.Int cap;
      ]
    ~block:(Execute.dims1d 64)
    ~grid:(Execute.dims1d 4)
    () ;
  Transfer.flush dev ;
  let overflow = WL.Host.overflow q in
  (* Distinct, in-range values in the ring; count them. *)
  let seen = Array.make (n + 1) false in
  let distinct = ref true in
  for i = 0 to cap - 1 do
    let v = Int32.to_int (Vector.get q.WL.Host.slots i) in
    if v >= 1 && v <= n then (
      if seen.(v) then distinct := false ;
      seen.(v) <- true)
  done ;
  overflow > 0 && !distinct

(* D: persistent single-thread linked-list traversal with a tiny wrapping ring. *)
let run_wrap dev =
  let n = 50 in
  let cap = 4 in
  let values =
    vec_of_int32_array (Array.init n (fun i -> Int32.of_int (i + 1)))
  in
  let nxt =
    vec_of_int32_array
      (Array.init n (fun i -> if i = n - 1 then -1l else Int32.of_int (i + 1)))
  in
  let acc = Vector.create Vector.int32 1 in
  Vector.set acc 0 0l ;
  let q = WL.Host.create ~capacity:cap in
  WL.Host.seed q [|0l|] ;
  Execute.run_vectors
    ~device:dev
    ~ir:(ir_of persistent_wrap_kernel)
    ~args:
      [
        Execute.Vec q.WL.Host.ctrl;
        Execute.Vec q.WL.Host.slots;
        Execute.Vec values;
        Execute.Vec nxt;
        Execute.Vec acc;
        Execute.Int cap;
      ]
    ~block:(Execute.dims1d 1)
    ~grid:(Execute.dims1d 1)
    () ;
  Transfer.flush dev ;
  let got = Int32.to_int (Vector.get acc 0) in
  let expected = n * (n + 1) / 2 in
  got = expected && WL.Host.overflow q = 0

(* ============================ driver ============================ *)

let () =
  Test_helpers.Benchmarks.init () ;
  let devices = Device.init () in
  (* deterministic tiny tree (B): root + explicit small fanout *)
  let big = build_tree 500 ~max_fan:4 in
  let tiny = build_tree 13 ~max_fan:3 in
  let exp_big = cpu_frontier_sum big in
  let exp_tiny = cpu_frontier_sum tiny in
  (* Sanity: the tree must fully expand (every node reachable), else the sums
     would be vacuous. Full sum = n(n+1)/2. *)
  assert (exp_big = big.n * (big.n + 1) / 2) ;
  assert (exp_tiny = tiny.n * (tiny.n + 1) / 2) ;
  let all_ok = ref true in
  Array.iter
    (fun dev ->
      let label =
        Printf.sprintf "%s (%s)" dev.Device.name dev.Device.framework
      in
      let check name ok =
        if not ok then all_ok := false ;
        Printf.sprintf "%s=%b" name ok
      in
      let results =
        try
          let sb, ob = run_frontier dev big ~threads:32 in
          let st, ot = run_frontier dev tiny ~threads:8 in
          [
            check "frontier500" (sb = exp_big && ob = 0);
            check "tiny13" (st = exp_tiny && ot = 0);
            check "overflow" (run_overflow dev);
            check "wrap" (run_wrap dev);
          ]
        with e ->
          all_ok := false ;
          [Printf.sprintf "EXN=%s" (Printexc.to_string e)]
      in
      Printf.printf "  %-54s : %s\n" label (String.concat " " results))
    devices ;
  Printf.printf "  (reference: frontier500=%d tiny13=%d)\n" exp_big exp_tiny ;
  if !all_ok then print_endline "test_worklist PASSED"
  else (
    print_endline "test_worklist FAILED" ;
    exit 1)
