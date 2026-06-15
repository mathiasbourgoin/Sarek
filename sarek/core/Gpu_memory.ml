(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Runtime - GPU Memory Management
 *
 * Provides automatic GC integration for GPU memory:
 * - Finalizers on vectors to free device buffers when GC'd
 * - Retry-with-GC pattern on allocation failure
 * - GPU memory usage tracking
 * - Configurable auto-GC behavior
 ******************************************************************************)

(** {1 Configuration} *)

(** Whether to automatically retry allocations after GC on failure *)
let auto_gc = ref true

let set_auto_gc enabled = auto_gc := enabled

let get_auto_gc () = !auto_gc

(** {1 Memory Tracking} *)

(** Total GPU memory currently allocated (bytes), across all devices *)
let allocated_bytes = Atomic.make 0

(** Statistics counters *)
let alloc_count = Atomic.make 0

let free_count = Atomic.make 0

let gc_free_count = Atomic.make 0

let gc_freed_bytes = Atomic.make 0

let retry_count = Atomic.make 0

let peak_bytes = Atomic.make 0

let track_alloc bytes =
  Atomic.fetch_and_add alloc_count 1 |> ignore ;
  let current = Atomic.fetch_and_add allocated_bytes bytes + bytes in
  (* Update peak *)
  let rec update_peak () =
    let old_peak = Atomic.get peak_bytes in
    if current > old_peak then
      if not (Atomic.compare_and_set peak_bytes old_peak current) then
        update_peak ()
  in
  update_peak ()

let track_free bytes =
  Atomic.fetch_and_add free_count 1 |> ignore ;
  Atomic.fetch_and_add allocated_bytes (-bytes) |> ignore

(** Decrement allocated bytes without counting as an explicit free (used by GC
    finalizers) *)
let track_gc_free bytes =
  Atomic.fetch_and_add allocated_bytes (-bytes) |> ignore

(** Query current GPU memory usage in bytes *)
let usage () = Atomic.get allocated_bytes

(** Query peak GPU memory usage in bytes *)
let peak_usage () = Atomic.get peak_bytes

type stats = {
  current_bytes : int;
  peak_bytes : int;
  alloc_count : int;
  free_count : int;
  gc_free_count : int;
  gc_freed_bytes : int;
  retry_count : int;
}

(** Get all statistics *)
let stats () =
  {
    current_bytes = Atomic.get allocated_bytes;
    peak_bytes = Atomic.get peak_bytes;
    alloc_count = Atomic.get alloc_count;
    free_count = Atomic.get free_count;
    gc_free_count = Atomic.get gc_free_count;
    gc_freed_bytes = Atomic.get gc_freed_bytes;
    retry_count = Atomic.get retry_count;
  }

(** Print statistics to stderr *)
let print_stats () =
  let s = stats () in
  Printf.eprintf
    "[GPU Memory] current: %d bytes, peak: %d bytes\n"
    s.current_bytes
    s.peak_bytes ;
  Printf.eprintf
    "[GPU Memory] allocations: %d, manual frees: %d\n"
    s.alloc_count
    s.free_count ;
  Printf.eprintf
    "[GPU Memory] GC-triggered frees: %d (%d bytes)\n"
    s.gc_free_count
    s.gc_freed_bytes ;
  if s.retry_count > 0 then
    Printf.eprintf
      "[GPU Memory] allocation retries (after GC): %d\n"
      s.retry_count

(** Reset all statistics (useful for per-benchmark tracking) *)
let reset_stats () =
  Atomic.set allocated_bytes 0 ;
  Atomic.set peak_bytes 0 ;
  Atomic.set alloc_count 0 ;
  Atomic.set free_count 0 ;
  Atomic.set gc_free_count 0 ;
  Atomic.set gc_freed_bytes 0 ;
  Atomic.set retry_count 0

(** {1 GC Integration} *)

(** Trigger a full major GC to free unreachable GPU buffers *)
let trigger_gc () = Gc.full_major ()

(** Attempt an allocation, retrying after GC if it fails. On first failure: run
    Gc.full_major to trigger finalizers. On second failure: run Gc.compact for
    maximum cleanup. On third failure: propagate the exception. *)
let with_retry f =
  if not !auto_gc then f ()
  else
    try f ()
    with exn1 -> (
      (* First retry: trigger finalizers *)
      Atomic.fetch_and_add retry_count 1 |> ignore ;
      Gc.full_major () ;
      try f ()
      with _exn2 -> (
        (* Second retry: compact heap *)
        Atomic.fetch_and_add retry_count 1 |> ignore ;
        Gc.compact () ;
        try f () with _exn3 -> raise exn1 (* Raise original exception *)))

(** {1 Finalizer Registration} *)

(* Register at_exit handler to print stats if SPOC_GPU_STATS is set *)

(** Register a finalizer on a vector that frees all its device buffers when the
    vector is garbage collected. This is idempotent - calling free on an
    already-freed buffer is safe (backends handle double-free).

    The finalizer is registered only once per vector (on first device buffer
    allocation). *)
let () =
  if Sys.getenv_opt "SPOC_GPU_STATS" <> None then
    at_exit (fun () ->
        (* Final GC to flush pending finalizers before printing *)
        Gc.full_major () ;
        print_stats ())

(** Register a finalizer on a vector to free device buffers when GC'd. *)
let register_finalizer (vec : (_, _) Vector_types.t) =
  Gc.finalise
    (fun (v : (_, _) Vector_types.t) ->
      Hashtbl.iter
        (fun _dev_id buf ->
          let (module B : Vector_types.DEVICE_BUFFER) = buf in
          let bytes = B.size * B.elem_size in
          try
            B.free () ;
            Atomic.fetch_and_add gc_free_count 1 |> ignore ;
            Atomic.fetch_and_add gc_freed_bytes bytes |> ignore ;
            track_gc_free bytes
          with _ -> ())
        v.device_buffers ;
      Hashtbl.clear v.device_buffers)
    vec
