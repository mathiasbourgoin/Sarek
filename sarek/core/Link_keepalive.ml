(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Link_keepalive - keep the host-device link out of power-saving states during
 * long submissions. Contract in [Link_keepalive.mli].
 *
 * The submissions come from their own domain, which has its own runtime lock:
 * a main domain that spends tens of milliseconds in OCaml code cannot delay
 * them (a systhread would wait for the main domain's lock). The loop does not
 * allocate itself; the backend's launch path allocates a few small blocks per
 * launch, so this domain takes a minor collection (a short stop-the-world)
 * about once a second at the default period.
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

type status = Running | Stopped | Unsupported | Failed of string

let min_period_us = 50

let max_period_us = 10_000

let default_period_us = 500

(* Launches between two synchronisations of the side stream. A synchronisation
   lets the backend release the argument blocks it retains per launch, so the
   loop runs in constant memory. *)
let launches_per_sync = 64

let kernel_name = "sarek_link_keepalive"

let kernel_source = "extern \"C\" __global__ void sarek_link_keepalive() {}"

type t = {
  device : Device.t;
  period_us : int;
  supported : bool;
  stop_requested : bool Atomic.t;
  launches : int Atomic.t;
  failure : string option Atomic.t;
  ready : bool Atomic.t;  (** Set by the worker once set-up has finished. *)
  mutable worker : unit Domain.t option;  (** Guarded by [lock]. *)
  lock : Mutex.t;
}

let supported (dev : Device.t) : bool =
  match dev.framework with
  | "HIP" | "CUDA" | "CUDA/PTX" | "CUDA/C" -> true
  | _ -> false

let clamp_period period_us = max min_period_us (min max_period_us period_us)

(* Active keep-alives, one per device. *)
let registry : (int, t) Hashtbl.t = Hashtbl.create 4

let registry_lock = Mutex.create ()

let status (t : t) : status =
  if not t.supported then Unsupported
  else
    match Atomic.get t.failure with
    | Some reason -> Failed reason
    | None ->
        let started = Mutex.protect t.lock (fun () -> t.worker <> None) in
        if started && not (Atomic.get t.stop_requested) then Running
        else Stopped

let device t = t.device

let period_us t = t.period_us

let launches t = Atomic.get t.launches

let fail t where exn =
  let reason = Printf.sprintf "%s: %s" where (Printexc.to_string exn) in
  Log.debugf Log.Device "Link_keepalive: %s" reason ;
  Atomic.set t.failure (Some reason)

(* The periodic loop, over closures built once so that it allocates nothing. *)
let run_loop (t : t) ~(launch : unit -> unit) ~(sync : unit -> unit) : unit =
  let delay = float_of_int t.period_us *. 1e-6 in
  let rec loop since_sync =
    if not (Atomic.get t.stop_requested) then begin
      launch () ;
      Atomic.incr t.launches ;
      if since_sync >= launches_per_sync then begin
        sync () ;
        loop 0
      end
      else begin
        Unix.sleepf delay ;
        loop (since_sync + 1)
      end
    end
  in
  loop 0

(* Worker body, run in its own domain. Everything that binds to the device
   (current device, module, stream) is created here, so the calling domain's
   current device is never touched. *)
let worker_body (t : t) (module B : Framework_sig.BACKEND) : unit =
  match
    let dev = B.Device.get t.device.backend_id in
    B.Device.set_current dev ;
    let kernel =
      B.Kernel.compile_cached dev ~name:kernel_name ~source:kernel_source
    in
    (kernel, B.Stream.create_non_blocking dev)
  with
  | exception exn -> fail t "set-up" exn
  | kernel, stream -> (
      let args = B.Kernel.create_args () in
      let grid = Framework_sig.dims_1d 1 in
      let block = Framework_sig.dims_1d 32 in
      let some_stream = Some stream in
      Atomic.set t.ready true ;
      (try
         run_loop
           t
           ~launch:(fun () ->
             B.Kernel.launch
               kernel
               ~args
               ~grid
               ~block
               ~shared_mem:0
               ~stream:some_stream)
           ~sync:(fun () -> B.Stream.synchronize stream)
       with exn -> fail t "launch" exn) ;
      (* Draining the side stream also releases what the backend retained for
         its launches. *)
      try
        B.Stream.synchronize stream ;
        B.Stream.destroy stream
      with exn -> fail t "clean-up" exn)

let worker (t : t) () : unit =
  (match Framework_registry.find_backend t.device.framework with
  | None -> Atomic.set t.failure (Some "backend not registered")
  | Some backend -> worker_body t backend) ;
  Atomic.set t.ready true

let make dev ~period_us =
  {
    device = dev;
    period_us = clamp_period period_us;
    supported = supported dev;
    stop_requested = Atomic.make false;
    launches = Atomic.make 0;
    failure = Atomic.make None;
    ready = Atomic.make false;
    worker = None;
    lock = Mutex.create ();
  }

(* Wait for the worker's set-up (module compilation can take tens of
   milliseconds the first time; it is cached afterwards). *)
let await_ready t =
  while not (Atomic.get t.ready) do
    Unix.sleepf 0.0005
  done

let spawn t =
  match Domain.spawn (worker t) with
  | exception exn -> fail t "Domain.spawn" exn
  | domain ->
      Mutex.protect t.lock (fun () -> t.worker <- Some domain) ;
      await_ready t

let stop (t : t) : unit =
  Atomic.set t.stop_requested true ;
  Mutex.protect t.lock (fun () ->
      match t.worker with
      | None -> ()
      | Some domain ->
          Domain.join domain ;
          t.worker <- None) ;
  Mutex.protect registry_lock (fun () ->
      match Hashtbl.find_opt registry t.device.id with
      | Some current when current == t -> Hashtbl.remove registry t.device.id
      | _ -> ())

let stop_all () =
  let all =
    Mutex.protect registry_lock (fun () ->
        Hashtbl.fold (fun _ t acc -> t :: acc) registry [])
  in
  List.iter stop all

let at_exit_installed = ref false

let start ?(period_us = default_period_us) (dev : Device.t) : t =
  Mutex.protect registry_lock (fun () ->
      match Hashtbl.find_opt registry dev.id with
      | Some existing when status existing = Running -> existing
      | _ ->
          let t = make dev ~period_us in
          if t.supported then begin
            Hashtbl.replace registry dev.id t ;
            if not !at_exit_installed then begin
              at_exit_installed := true ;
              at_exit stop_all
            end ;
            spawn t
          end ;
          t)
