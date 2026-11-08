(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Link_keepalive: lifecycle contract. One body, run per device: by
 * test_link_keepalive.exe on the CPU backends (a documented no-op there) and
 * by sarek-hip/test/test_hip_link_keepalive.exe on HIP hardware (where it
 * launches for real). Sizes and durations are small: the GPU is shared.
 ******************************************************************************)

module Device = Spoc_core.Device
module Keepalive = Spoc_core.Link_keepalive

let failures = ref 0

let check name cond =
  if cond then Printf.printf "  [OK]   %s\n%!" name
  else begin
    incr failures ;
    Printf.printf "  [FAIL] %s\n%!" name
  end

let status_string = function
  | Keepalive.Running -> "Running"
  | Suspended -> "Suspended"
  | Stopped -> "Stopped"
  | Unsupported -> "Unsupported"
  | Failed reason -> "Failed " ^ reason

let is_running t = Keepalive.status t = Keepalive.Running

(* Wait until [cond ()] holds, for at most [seconds]. *)
let wait_for ~seconds cond =
  let deadline = Unix.gettimeofday () +. seconds in
  while (not (cond ())) && Unix.gettimeofday () < deadline do
    Unix.sleepf 0.001
  done ;
  cond ()

let run_unsupported (dev : Device.t) =
  check "supported is false" (not (Keepalive.supported dev)) ;
  let t = Keepalive.start dev in
  check "start returns an Unsupported handle" (Keepalive.status t = Unsupported) ;
  Unix.sleepf 0.02 ;
  check "no launch is ever made" (Keepalive.launches t = 0) ;
  Keepalive.stop t ;
  Keepalive.stop t ;
  check "stop is idempotent and harmless" (Keepalive.status t = Unsupported) ;
  let t' = Keepalive.start ~period_us:100 dev in
  check "restart is again a no-op" (Keepalive.status t' = Unsupported) ;
  Keepalive.stop t'

let run_supported (dev : Device.t) =
  check "supported is true" (Keepalive.supported dev) ;
  let t = Keepalive.start ~period_us:200 dev in
  check
    ("status is Running (" ^ status_string (Keepalive.status t) ^ ")")
    (is_running t) ;
  check "period is kept" (Keepalive.period_us t = 200) ;
  check
    "launches advance"
    (wait_for ~seconds:5.0 (fun () -> Keepalive.launches t >= 20)) ;
  (* Absolute deadlines: at a period of 200 us, 300 ms is about 1500
     launches; a loose floor keeps a loaded machine from failing it. *)
  let before = Keepalive.launches t in
  Unix.sleepf 0.3 ;
  check "launch rate follows the period" (Keepalive.launches t - before >= 150) ;
  Keepalive.suspend t ;
  check "suspend parks it" (Keepalive.status t = Suspended) ;
  let parked = Keepalive.launches t in
  Unix.sleepf 0.05 ;
  check "no launch while suspended" (Keepalive.launches t = parked) ;
  Keepalive.suspend t ;
  check "suspend is idempotent" (Keepalive.status t = Suspended) ;
  let resumed = Keepalive.start ~period_us:900 dev in
  check "start resumes the same handle" (resumed == t && is_running t) ;
  check
    "launches advance after resume"
    (wait_for ~seconds:5.0 (fun () -> Keepalive.launches t > parked + 5)) ;
  let cycles_ok = ref true in
  for _ = 1 to 500 do
    Keepalive.suspend t ;
    if Keepalive.status t <> Suspended then cycles_ok := false ;
    ignore (Keepalive.start dev) ;
    if not (is_running t) then cycles_ok := false
  done ;
  check "500 suspend/resume cycles" !cycles_ok ;
  let t2 = Keepalive.start ~period_us:900 dev in
  check "start while Running returns the same handle" (t2 == t) ;
  Unix.sleepf 0.05 ;
  check "still running after 50 ms" (is_running t) ;
  Keepalive.suspend t ;
  Keepalive.stop t ;
  check "stopped (from suspended)" (Keepalive.status t = Stopped) ;
  let after = Keepalive.launches t in
  Unix.sleepf 0.05 ;
  check "no launch after stop" (Keepalive.launches t = after) ;
  Keepalive.stop t ;
  check "stop is idempotent" (Keepalive.status t = Stopped) ;
  (* Repeated cycles leak neither domains, streams nor modules (OCaml allows
     128 domains at once, so a leaked domain per cycle would make set-up
     fail). *)
  let all_ok = ref true in
  for _ = 1 to 150 do
    let c = Keepalive.start dev in
    if not (is_running c) then all_ok := false ;
    Keepalive.stop c
  done ;
  check "150 start/stop cycles" !all_ok ;
  let low = Keepalive.start ~period_us:0 dev in
  check
    "period 0 is clamped up"
    (Keepalive.period_us low = Keepalive.min_period_us) ;
  Keepalive.stop low ;
  let high = Keepalive.start ~period_us:max_int dev in
  check
    "huge period is clamped down"
    (Keepalive.period_us high = Keepalive.max_period_us) ;
  Keepalive.stop high ;
  let again = Keepalive.start dev in
  check "restart after stop is Running" (is_running again) ;
  check
    "default period"
    (Keepalive.period_us again = Keepalive.default_period_us) ;
  Keepalive.stop again

let run_on (dev : Device.t) =
  Printf.printf "Link_keepalive on %s (%s)\n%!" dev.name dev.framework ;
  if Keepalive.supported dev then run_supported dev else run_unsupported dev

(* A backend of libc calls, so the failure and race paths run without a GPU:
   [isatty] on an invalid descriptor returns 0 (success) for any further
   arguments and [getpid] returns a positive number (an error for a launch or a
   device selection). *)
let fake_provider ~launch_ok ~prep_ok ~releases : Device.t -> Keepalive.native =
 fun _ ->
  let sym symbol = Dl.dlsym ?handle:None ~symbol in
  {
    Keepalive.launch_fn = sym (if launch_ok then "isatty" else "getpid");
    prep_fn = sym (if prep_ok then "isatty" else "getpid");
    prep_arg = 99999n;
    kernel = 99999n;
    stream = 0n;
    release = (fun () -> incr releases);
  }

let run_fake (template : Device.t) =
  Printf.printf "Link_keepalive on a fake backend (libc calls)\n%!" ;
  let releases = ref 0 in
  let device id framework = {template with Device.id; framework} in
  (* a launch that fails: Failed, and a new start replaces it without leaking *)
  Keepalive.register_native
    ~family:"FakeKA"
    (fake_provider ~launch_ok:false ~prep_ok:true ~releases) ;
  let dev = device 9001 "FakeKA" in
  check "fake backend is supported" (Keepalive.supported dev) ;
  let t = Keepalive.start dev in
  check
    "a failing launch is reported"
    (wait_for ~seconds:5.0 (fun () ->
         match Keepalive.status t with Keepalive.Failed _ -> true | _ -> false)) ;
  let t' = Keepalive.start dev in
  check "start after a failure makes a fresh one" (t' != t) ;
  check "the failed one was torn down (stream released)" (!releases >= 1) ;
  Keepalive.stop t ;
  Keepalive.stop t' ;
  Keepalive.stop t' ;
  check "every stream released exactly once" (!releases = 2) ;
  (* a device selection that fails is reported by start itself *)
  releases := 0 ;
  Keepalive.register_native
    ~family:"FakeKA"
    (fake_provider ~launch_ok:true ~prep_ok:false ~releases) ;
  let bad = Keepalive.start dev in
  check
    "a failing device selection is reported"
    (match Keepalive.status bad with Keepalive.Failed _ -> true | _ -> false) ;
  check "its stream was released" (!releases = 1) ;
  Keepalive.stop bad ;
  (* start, suspend and stop from two domains at once: a resume must not touch
     a handle that stop has freed *)
  Keepalive.register_native
    ~family:"FakeKA"
    (fake_provider ~launch_ok:true ~prep_ok:true ~releases) ;
  let hammer ~with_stop () =
    for _ = 1 to 300 do
      let k = Keepalive.start ~period_us:50 dev in
      Keepalive.suspend k ;
      if with_stop then Keepalive.stop k
    done
  in
  let other = Domain.spawn (hammer ~with_stop:true) in
  hammer ~with_stop:false () ;
  Domain.join other ;
  let last = Keepalive.start dev in
  check "still usable after concurrent start/suspend/stop" (is_running last) ;
  Keepalive.stop last ;
  check "stopped" (Keepalive.status last = Stopped)

let exit_code () = if !failures = 0 then 0 else 1
