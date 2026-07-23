(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_worklist - portable dynamic parallelism via a global work queue.
 *
 * Serves the irregular-work / frontier / tree-recursion use cases that CUDA
 * dynamic parallelism (CDP) targets - on every Sarek backend (CUDA/PTX incl.
 * ZLUDA, OpenCL, Vulkan, Metal, WGSL, Native, Interpreter) - using only atomics
 * Sarek already ships. No device-side kernel-launch mechanism is needed: the
 * launch happens once, host-side, like any other kernel; all "dynamic" behavior
 * flows through the queue. See roster/ptx-limits-campaign/L16-dynamic-
 * parallelism.md for the CDP-vs-worklist rationale (CDP is frequently slower
 * than a good worklist for its own headline workload and exists cleanly on only
 * 1 of 6 backends).
 *
 * QUEUE LAYOUT
 *   A control vector [ctrl] (int32, length {!Ctrl.size}) holds the counters:
 *     ctrl.(0) = HEAD         next pop  ticket (monotonic)
 *     ctrl.(1) = TAIL         next push ticket (monotonic)
 *     ctrl.(2) = OUTSTANDING  pushed - fully-processed (persistent variant)
 *     ctrl.(3) = OVERFLOW     nonzero if a push found the ring full
 *   plus a [slots] ring vector of int32 items (typically node/index
 *   descriptors). Tickets index the ring modulo capacity.
 *
 * WHY ATOMICS ARE INLINE, NOT HELPERS
 *   The push/pop atomics operate on the queue *vectors*. A [@sarek.module]
 *   helper cannot take a vector and call [atomic_add_global_int32] on it: the
 *   OCaml shim pins the parameter to [int32 array], which does not unify with
 *   the kernel's vector object type (the PPX only rewrites vector params through
 *   [.()]/[.()<-], not through intrinsic calls). So the queue operations are a
 *   documented *pattern* you paste into your own [%kernel] (see the PATTERNS
 *   section below), and this library provides the pure, callable helpers that
 *   do not touch vectors ({!wl_ring_index}, {!wl_has_work}) plus all the
 *   host-side orchestration ({!Host}). This split is the one the design doc
 *   explicitly anticipates.
 *
 * TWO PATTERNS (both run on every backend)
 *   1. LEVEL-SYNCHRONOUS FRONTIER (recommended; works on the sequential
 *      interpreter too). Each launch, every thread claims tickets from a fixed
 *      window [head, snapshot_tail) via the shared atomic HEAD counter - no
 *      thread ever waits on another - processes the item, and pushes any newly
 *      discovered items to TAIL for the *next* level. The host relaunches until
 *      the frontier drains ({!Host.drive}). Termination is host-driven and
 *      spec-safe (no GPU forward-progress assumption); the host sync between
 *      levels is the memory barrier.
 *
 *      The window [level_base, snapshot_tail) is distributed by grid-stride (no
 *      shared claim counter, so nothing needs resetting between launches); only
 *      TAIL is an atomic device counter, which the host reads but never writes:
 *
 *        [%kernel fun ctrl slots values off idx acc level_base snapshot_tail cap ->
 *          let open Std in let open Gpu in
 *          let stride = block_dim_x * grid_dim_x in
 *          let i = mut (level_base + thread_idx_x + (block_idx_x * block_dim_x)) in
 *          while i < snapshot_tail do
 *            let u = slots.(wl_ring_index cap i) in
 *            let _ = atomic_add_global_int32 acc 0l values.(u) in
 *            let s = mut off.(u) in
 *            while s < off.(u + 1l) do
 *              let t = atomic_add_global_int32 ctrl 1l 1l in
 *              let head = atomic_add_global_int32 ctrl 0l 0l in
 *              if t - head >= cap then
 *                (let _ = atomic_add_global_int32 ctrl 3l 1l in ())  (* OVERFLOW *)
 *              else slots.(wl_ring_index cap t) <- idx.(s) ;
 *              s := s + 1l
 *            done ;
 *            i := i + stride
 *          done]
 *
 *   2. PERSISTENT SINGLE-LAUNCH (one launch, each thread loops pop/work/push
 *      until [head >= tail]). Safe for a single thread on every backend; safe
 *      multi-thread only on real GPUs where all launched blocks are co-resident
 *      (persistent-threads occupancy) - it spin-waits across threads, which
 *      DEADLOCKS on the sequential interpreter and on any block-sequential or
 *      pool-bounded executor. Do not run it multi-thread on the interpreter.
 *
 * CAPACITY / TERMINATION CONTRACT (honest limits)
 *   - Level-sync main use: size capacity >= total items ever enqueued (the ring
 *     never reuses a slot, so ordering can never corrupt). Every push is guarded
 *     ([t - head >= cap] -> set OVERFLOW instead of writing to a live slot), so
 *     an under-sized ring is FLAGGED via {!Host.overflow} rather than silently
 *     clobbering live data. Once OVERFLOW is set the run's result is undefined
 *     (items were dropped) — treat a nonzero {!Host.overflow} as "grow capacity
 *     and re-run", exactly like the {!push_guarded} pattern.
 *   - Ring reuse (wrap) is correct only when capacity >= peak simultaneously-
 *     live items, so a slot is always consumed before its ticket laps it.
 *   - This is NOT a drop-in for arbitrary CDP: workers are homogeneous (same
 *     kernel body). A sub-problem needing a genuinely different launch
 *     configuration is Route B's one real gap (see the doc S5).
 *
 * USAGE (from another compilation unit)
 *   dune:   add [sarek.worklist] to (libraries); add this file to
 *           (preprocessor_deps) of the kernel's stanza.
 *   source: [let%sarek_include _ = "path/to/Sarek_worklist.ml"] then call the
 *           pure helpers from [%kernel]; use {!Host} from host code.
 ******************************************************************************)

[@@@warning "-32-33-34"]

(* Alias so the pure [@sarek.module] helpers type-check as OCaml; the PPX maps
   [int32] arithmetic to the int32 intrinsics. *)
type 'a vector = 'a array

(* Module-level int32 operator shim (same idea as Sarek_df64 before
   df64_of_int32): OCaml uses Int32.rem, the PPX emits the int32 mod intrinsic.
   Scoped to the helpers below; {!Host} rebinds Stdlib operators. *)
let ( mod ) = Int32.rem

(** Ring index of a monotonic ticket: [ticket mod capacity]. Callable from a
    [%kernel] as an array index. *)
let[@sarek.module] wl_ring_index (capacity : int32) (ticket : int32) : int32 =
  ticket mod capacity

(** Whether the queue has an unclaimed item: [head < tail]. *)
let[@sarek.module] wl_has_work (head : int32) (tail : int32) : bool =
  head < tail

(** Control-vector slot indices (host-side and documentation). *)
module Ctrl = struct
  let head = 0

  let tail = 1

  let outstanding = 2

  let overflow = 3

  (** Length of the control vector. *)
  let size = 4
end

(** Host-side queue management and the level-synchronous driver. *)
module Host = struct
  let ( mod ) = Stdlib.( mod )

  type int32_vector = (int32, Bigarray.int32_elt) Spoc_core.Vector.t

  type t = {ctrl : int32_vector; slots : int32_vector; capacity : int}

  (** Allocate a queue with a ring of [capacity] int32 slots. Counters start at
      zero; call {!seed} before running. *)
  let create ~capacity =
    let ctrl = Spoc_core.Vector.create Spoc_core.Vector.int32 Ctrl.size in
    let slots = Spoc_core.Vector.create Spoc_core.Vector.int32 capacity in
    for i = 0 to Ctrl.size - 1 do
      Spoc_core.Vector.set ctrl i 0l
    done ;
    for i = 0 to capacity - 1 do
      Spoc_core.Vector.set slots i (-1l)
    done ;
    {ctrl; slots; capacity}

  let get_ctrl t i = Int32.to_int (Spoc_core.Vector.get t.ctrl i)

  let set_ctrl t i v = Spoc_core.Vector.set t.ctrl i (Int32.of_int v)

  (** Seed the initial frontier with [items] (int32 node/index descriptors):
      writes them to the front of the ring and sets HEAD=0, TAIL=OUTSTANDING=
      |items|, OVERFLOW=0. *)
  let seed t (items : int32 array) =
    let n = Array.length items in
    if n > t.capacity then
      invalid_arg "Sarek_worklist.Host.seed: too many items" ;
    Array.iteri (fun i v -> Spoc_core.Vector.set t.slots i v) items ;
    set_ctrl t Ctrl.head 0 ;
    set_ctrl t Ctrl.tail n ;
    set_ctrl t Ctrl.outstanding n ;
    set_ctrl t Ctrl.overflow 0

  let head t = get_ctrl t Ctrl.head

  let tail t = get_ctrl t Ctrl.tail

  let outstanding t = get_ctrl t Ctrl.outstanding

  (** Nonzero if any push found the ring full during a run. *)
  let overflow t = get_ctrl t Ctrl.overflow

  (** Item currently stored at ring position [i mod capacity]. *)
  let slot t i = Int32.to_int (Spoc_core.Vector.get t.slots (i mod t.capacity))

  (** Level-synchronous driver. Each level processes the frontier window from
      [level_base] up to [snapshot_tail] (the current TAIL) and any children the
      workers push extend TAIL for the next level. Calls
      [launch ~level_base ~snapshot_tail] (which must run one frontier kernel
      over that window AND sync the device so TAIL reads back), until a level
      adds no work (TAIL did not grow) or [max_levels] is reached. Returns the
      number of levels launched.

      The driver passes the window as SCALAR arguments and only ever READS the
      device counters (never writes them back): the frontier kernel distributes
      the window by grid-stride, so there is no shared claim counter to reset
      and nothing to upload between launches. This is what makes the multi-
      launch loop coherent under Sarek's CPU/GPU/Both vector residency (a host
      write to a Both-resident vector would not re-upload). *)
  let drive t ~(launch : level_base:int -> snapshot_tail:int -> unit)
      ~max_levels =
    let rec loop level base =
      let snap = tail t in
      (* Stop at the first overflow (audit finding M8): once OVERFLOW is
         set, TAIL counts tickets whose ring slot was never written - still
         the -1 seed - and the next level would feed -1 to the kernel as a
         node index (out-of-bounds device reads on poisoned data). The
         caller observes {!overflow} > 0 and re-runs with a larger ring. *)
      if snap <= base || level >= max_levels || overflow t > 0 then level
      else begin
        launch ~level_base:base ~snapshot_tail:snap ;
        loop (level + 1) snap
      end
    in
    loop 0 0
end
