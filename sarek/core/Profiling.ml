(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Runtime - Profiling & Timing
 *
 * Provides first-class profiling support with GPU event-based timing.
 * Phase 5 of runtime V2 feature parity roadmap.
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

(** {1 Profiling State} *)

let enabled = ref false

let enable () = enabled := true

let disable () = enabled := false

let is_enabled () = !enabled

(** {1 Event-Based Timing} *)

(** Event handle - packages backend event with operations *)
module type EVENT = sig
  val device : Device.t

  val record : unit -> unit

  val synchronize : unit -> unit

  val destroy : unit -> unit
end

type event = (module EVENT)

(** Create a timing event on a device *)
let create_event (dev : Device.t) : event =
  match Framework_registry.find_backend dev.framework with
  | None -> failwith ("Unknown framework: " ^ dev.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let e = B.Event.create () in
      let default_stream = B.Stream.default (B.Device.get dev.backend_id) in
      (module struct
        let device = dev

        let record () = B.Event.record e default_stream

        let synchronize () = B.Event.synchronize e

        let destroy () = B.Event.destroy e
      end : EVENT)

(** Record event (mark current point in stream) *)
let record_event (e : event) =
  let (module E : EVENT) = e in
  E.record ()

(** Wait for event to complete *)
let synchronize_event (e : event) =
  let (module E : EVENT) = e in
  E.synchronize ()

(** Destroy event *)
let destroy_event (e : event) =
  let (module E : EVENT) = e in
  E.destroy ()

(** {1 Elapsed Time Measurement} *)

(** Event pair for timing - packages start/stop with backend *)
module type EVENT_PAIR = sig
  val device : Device.t

  val start : unit -> unit

  val stop : unit -> unit

  val elapsed_ms : unit -> float

  val destroy : unit -> unit
end

type timer = (module EVENT_PAIR)

(** Create a timer for measuring GPU operations *)
let create_timer (dev : Device.t) : timer =
  match Framework_registry.find_backend dev.framework with
  | None -> failwith ("Unknown framework: " ^ dev.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let start_event = B.Event.create () in
      let stop_event = B.Event.create () in
      let default_stream = B.Stream.default (B.Device.get dev.backend_id) in
      (module struct
        let device = dev

        let start () = B.Event.record start_event default_stream

        let stop () = B.Event.record stop_event default_stream

        let elapsed_ms () =
          B.Event.synchronize stop_event ;
          B.Event.elapsed ~start:start_event ~stop:stop_event

        let destroy () =
          B.Event.destroy start_event ;
          B.Event.destroy stop_event
      end : EVENT_PAIR)

let timer_start (t : timer) =
  let (module T : EVENT_PAIR) = t in
  T.start ()

let timer_stop (t : timer) =
  let (module T : EVENT_PAIR) = t in
  T.stop ()

let timer_elapsed_ms (t : timer) =
  let (module T : EVENT_PAIR) = t in
  T.elapsed_ms ()

let timer_destroy (t : timer) =
  let (module T : EVENT_PAIR) = t in
  T.destroy ()

(** {1 Convenience Timing Functions} *)

(** Time a GPU operation and return result with elapsed time in ms *)
let timed (dev : Device.t) (f : unit -> 'a) : 'a * float =
  let t = create_timer dev in
  timer_start t ;
  let result = f () in
  timer_stop t ;
  let elapsed = timer_elapsed_ms t in
  timer_destroy t ;
  (result, elapsed)

(** Time a GPU operation and print result *)
let time (name : string) (dev : Device.t) (f : unit -> 'a) : 'a =
  let result, elapsed = timed dev f in
  Printf.printf "[%s] %.3f ms\n%!" name elapsed ;
  result

(** {1 CPU-Side Timing} *)

(** Injectable wall-clock source. Defaults to [Unix.gettimeofday]. Replace in
    environments without Unix (e.g. jsoo) before calling [cpu_timed] or
    [cpu_time]. *)
let clock : (unit -> float) ref = ref Unix.gettimeofday

(** Simple wall-clock timing for CPU operations *)
let cpu_timed (f : unit -> 'a) : 'a * float =
  let start = !clock () in
  let result = f () in
  let stop = !clock () in
  (result, (stop -. start) *. 1000.0)

let cpu_time (name : string) (f : unit -> 'a) : 'a =
  let result, elapsed = cpu_timed f in
  Printf.printf "[%s] %.3f ms (CPU)\n%!" name elapsed ;
  result

(** {1 Kernel Statistics} *)

type kernel_stats = {
  name : string;
  mutable invocations : int;
  mutable total_time_ms : float;
}

let kernel_stats_table : (string, kernel_stats) Hashtbl.t = Hashtbl.create 32

(** Record a kernel invocation *)
let record_kernel_invocation ~(name : string) ~(time_ms : float) =
  match Hashtbl.find_opt kernel_stats_table name with
  | Some stats ->
      stats.invocations <- stats.invocations + 1 ;
      stats.total_time_ms <- stats.total_time_ms +. time_ms
  | None ->
      Hashtbl.replace
        kernel_stats_table
        name
        {name; invocations = 1; total_time_ms = time_ms}

(** Get all kernel statistics *)
let kernel_stats () : kernel_stats list =
  Hashtbl.to_seq_values kernel_stats_table |> List.of_seq

(** Get stats for a specific kernel *)
let get_kernel_stats (name : string) : kernel_stats option =
  Hashtbl.find_opt kernel_stats_table name

(** Reset all kernel statistics *)
let reset_stats () = Hashtbl.clear kernel_stats_table

(** Average time per invocation *)
let avg_time_ms (stats : kernel_stats) : float =
  if stats.invocations > 0 then
    stats.total_time_ms /. float_of_int stats.invocations
  else 0.0

(** Print kernel statistics summary *)
let print_stats () =
  let stats = kernel_stats () in
  if List.length stats = 0 then print_endline "No kernel statistics recorded."
  else begin
    print_endline "Kernel Statistics:" ;
    print_endline "----------------------------------------" ;
    Printf.printf
      "%-30s %8s %12s %12s\n"
      "Kernel"
      "Calls"
      "Total (ms)"
      "Avg (ms)" ;
    print_endline "----------------------------------------" ;
    List.iter
      (fun s ->
        Printf.printf
          "%-30s %8d %12.3f %12.3f\n"
          s.name
          s.invocations
          s.total_time_ms
          (avg_time_ms s))
      (List.sort (fun a b -> compare b.total_time_ms a.total_time_ms) stats) ;
    print_endline "----------------------------------------"
  end

(** {1 Scoped Profiling} *)

(** Run function with profiling enabled, then restore previous state *)
let with_profiling (f : unit -> 'a) : 'a =
  let was_enabled = !enabled in
  enabled := true ;
  let result = f () in
  enabled := was_enabled ;
  result
