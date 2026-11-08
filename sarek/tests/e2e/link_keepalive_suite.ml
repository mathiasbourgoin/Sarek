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
  let t2 = Keepalive.start ~period_us:900 dev in
  check "start while Running returns the same handle" (t2 == t) ;
  Unix.sleepf 0.05 ;
  check "still running after 50 ms" (is_running t) ;
  Keepalive.stop t ;
  check "stopped" (Keepalive.status t = Stopped) ;
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

let exit_code () = if !failures = 0 then 0 else 1
