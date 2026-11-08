(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Link_keepalive - keep the host-device link out of power-saving states during
 * long submissions. Contract in [Link_keepalive.mli].
 *
 * The launches come from a native thread (link_keepalive_stubs.c) that calls
 * the driver directly and never enters the OCaml runtime: it takes part in no
 * stop-the-world collection, holds no runtime lock and allocates nothing in
 * the OCaml heap. The OCaml side creates the stream and the kernel (through
 * the backend's registered provider), hands their native handles and the
 * period to the thread, and reads back its state and launch count. Set-up and
 * release run in a short-lived domain of their own, so the calling domain's
 * current device is never touched.
 ******************************************************************************)

type status = Running | Stopped | Unsupported | Failed of string

type native = {
  launch_fn : nativeint;
  prep_fn : nativeint;
  prep_arg : nativeint;
  kernel : nativeint;
  stream : nativeint;
  release : unit -> unit;
}

external stub_start : nativeint array -> int -> nativeint
  = "sarek_link_keepalive_start"

external stub_state : nativeint -> int = "sarek_link_keepalive_state"
[@@noalloc]

external stub_rc : nativeint -> int = "sarek_link_keepalive_rc" [@@noalloc]

external stub_launches : nativeint -> int = "sarek_link_keepalive_launches"
[@@noalloc]

external stub_stop : nativeint -> unit = "sarek_link_keepalive_stop"

external stub_free : nativeint -> unit = "sarek_link_keepalive_free" [@@noalloc]

let state_starting = 0

let state_failed = 3

let min_period_us = 50

let max_period_us = 10_000

let default_period_us = 500

let kernel_name = "sarek_link_keepalive"

let kernel_source = "extern \"C\" __global__ void sarek_link_keepalive() {}"

type t = {
  device : Device.t;
  period_us : int;
  supported : bool;
  failure : string option Atomic.t;
  mutable handle : nativeint;
      (** the thread; 0 when none. Guarded by [lock]. *)
  mutable final_launches : int;
  mutable release : (unit -> unit) option;
  lock : Mutex.t;
}

(* Backends that have a keep-alive, by framework family ("HIP", "CUDA"). *)
let providers : (string * (Device.t -> native)) list ref = ref []

let providers_lock = Mutex.create ()

let register_native ~family provider =
  Mutex.protect providers_lock (fun () ->
      providers := (family, provider) :: List.remove_assoc family !providers)

let provider_for (dev : Device.t) =
  Mutex.protect providers_lock (fun () ->
      List.find_map
        (fun (family, p) ->
          if
            dev.framework = family
            || String.starts_with ~prefix:(family ^ "/") dev.framework
          then Some p
          else None)
        !providers)

let supported (dev : Device.t) : bool = provider_for dev <> None

let clamp_period period_us = max min_period_us (min max_period_us period_us)

(* Active keep-alives, one per device. *)
let registry : (int, t) Hashtbl.t = Hashtbl.create 4

let registry_lock = Mutex.create ()

let fail t where reason =
  let reason = Printf.sprintf "%s: %s" where reason in
  Log.debugf Log.Device "Link_keepalive: %s" reason ;
  Atomic.set t.failure (Some reason)

let status (t : t) : status =
  if not t.supported then Unsupported
  else
    Mutex.protect t.lock (fun () ->
        match Atomic.get t.failure with
        | Some reason -> Failed reason
        | None ->
            if t.handle = 0n then Stopped
            else if stub_state t.handle = state_failed then
              Failed (Printf.sprintf "launch returned %d" (stub_rc t.handle))
            else Running)

let device t = t.device

let period_us t = t.period_us

let launches t =
  Mutex.protect t.lock (fun () ->
      if t.handle = 0n then t.final_launches else stub_launches t.handle)

(* Run [f] in a domain of its own and wait for it: everything that binds to the
   device (current device, module, stream) then happens on a thread that is
   not the caller's, so the caller's current device stays as it was. *)
let in_fresh_domain (f : unit -> 'a) : ('a, string) result =
  match Domain.spawn (fun () -> try Ok (f ()) with e -> Error e) with
  | exception e -> Error (Printexc.to_string e)
  | d -> (
      match Domain.join d with
      | Ok v -> Ok v
      | Error e -> Error (Printexc.to_string e))

let make dev ~period_us =
  {
    device = dev;
    period_us = clamp_period period_us;
    supported = supported dev;
    failure = Atomic.make None;
    handle = 0n;
    final_launches = 0;
    release = None;
    lock = Mutex.create ();
  }

(* Release the stream (in a domain of its own, as set-up). *)
let release t =
  match t.release with
  | None -> ()
  | Some f -> (
      t.release <- None ;
      match in_fresh_domain f with
      | Ok () -> ()
      | Error why -> fail t "clean-up" why)

(* Join the thread, keep its count, free it. *)
let stop_thread t =
  if t.handle <> 0n then begin
    stub_stop t.handle ;
    t.final_launches <- stub_launches t.handle ;
    if Atomic.get t.failure = None && stub_state t.handle = state_failed then
      fail t "launch" (Printf.sprintf "returned %d" (stub_rc t.handle)) ;
    stub_free t.handle ;
    t.handle <- 0n
  end

let stop (t : t) : unit =
  Mutex.protect t.lock (fun () ->
      stop_thread t ;
      release t) ;
  Mutex.protect registry_lock (fun () ->
      match Hashtbl.find_opt registry t.device.id with
      | Some current when current == t -> Hashtbl.remove registry t.device.id
      | _ -> ())

(* The thread reports a failure of its first call (the device selection)
   within microseconds: wait for it to leave the starting state. *)
let await_started t h =
  let rec go tries =
    if tries > 0 && stub_state h = state_starting then begin
      Unix.sleepf 0.0005 ;
      go (tries - 1)
    end
  in
  ignore t ;
  go 4000

let launch_thread t provider =
  match in_fresh_domain (fun () -> provider t.device) with
  | Error why -> fail t "set-up" why
  | Ok (n : native) -> (
      t.release <- Some n.release ;
      let args = [|n.launch_fn; n.prep_fn; n.prep_arg; n.kernel; n.stream|] in
      match stub_start args (t.period_us * 1000) with
      | 0n ->
          fail t "set-up" "could not create the thread" ;
          release t
      | h ->
          Mutex.protect t.lock (fun () -> t.handle <- h) ;
          await_started t h ;
          if stub_state h = state_failed then begin
            fail t "set-up" (Printf.sprintf "returned %d" (stub_rc h)) ;
            Mutex.protect t.lock (fun () ->
                stop_thread t ;
                release t)
          end)

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
          (match provider_for dev with
          | None -> ()
          | Some provider ->
              Hashtbl.replace registry dev.id t ;
              if not !at_exit_installed then begin
                at_exit_installed := true ;
                at_exit stop_all
              end ;
              launch_thread t provider) ;
          t)
