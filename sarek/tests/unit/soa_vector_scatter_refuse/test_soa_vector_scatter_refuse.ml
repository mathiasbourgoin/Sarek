(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** backlog-220: [Soa_vector.scatter] used to silently transpose stale host
    bytes into the leaves whenever auto-sync was off and the AoS vector's host
    copy was behind a device's ([Stale_CPU]). It returned [unit], not an
    error, so the caller had no way to tell a correct scatter from a corrupt
    one. This pins that it now refuses instead — and, just as importantly,
    that it does NOT refuse in the other states auto-sync-off can be in
    ([CPU] and [Stale_GPU], where the host copy is already the fresh data, so
    there is nothing stale to guard against). *)

module Soa = Spoc_core.Soa
module Soa_vector = Spoc_core.Soa_vector
module Vector = Spoc_core.Vector

type float32 = float

type point3d = {mutable x : float32; mutable y : float32; mutable z : float32}
[@@sarek.type]

let make_caps () : Spoc_framework.Framework_sig.capabilities =
  {
    max_threads_per_block = 256;
    max_block_dims = (256, 256, 64);
    max_grid_dims = (65535, 65535, 65535);
    shared_mem_per_block = 16384;
    total_global_mem = 1073741824L;
    compute_capability = (0, 0);
    device_features = [Sarek_ir_analysis.Float64; Sarek_ir_analysis.Int64];
    coopmat = None;
    supports_atomics = true;
    warp_size = 32;
    max_registers_per_block = 16384;
    clock_rate_khz = 1000000;
    multiprocessor_count = 4;
    is_cpu = false;
  }

let make_device id =
  {
    Spoc_core.Device.id;
    backend_id = id;
    name = Printf.sprintf "Fake Device %d" id;
    framework = "Fake";
    capabilities = make_caps ();
  }

let refusal_message =
  "Soa_vector.scatter: this vector's host data is out of date and auto-sync \
   is off, so scattering now would copy stale values into this vector's \
   per-leaf host buffers with no error. Before scattering, either call \
   Transfer.to_cpu on its AoS vector (Soa_vector.aos_vector) to refresh the \
   host copy, or call Vector.set_auto_sync on it to turn auto-sync back on."

let make_soa_vector () =
  let sv = Soa_vector.create point3d_custom 4 in
  for i = 0 to 3 do
    Soa_vector.set
      sv
      i
      {x = float_of_int i; y = float_of_int (i + 1); z = float_of_int (i + 2)}
  done ;
  sv

(* The bug: auto-sync off AND the host copy behind a device ([Stale_CPU]) used
   to scatter the stale host bytes silently. Must now refuse, with a message
   naming the problem and the two remedies. *)
let test_refuses_when_stale_and_auto_sync_off () =
  let sv = make_soa_vector () in
  let aos = Soa_vector.aos_vector sv in
  Vector.set_auto_sync aos false ;
  aos.Spoc_core.Vector_types.location <-
    Spoc_core.Vector_types.Stale_CPU (make_device 0) ;
  Alcotest.check_raises
    "scatter refuses on Stale_CPU + auto-sync off"
    (Soa.Unsupported refusal_message)
    (fun () -> Soa_vector.scatter sv)

(* Both polarities of the predicate matter (house defect class: a claim wider
   than its code). Auto-sync off alone must NOT be enough to refuse — only
   when the host copy is ALSO stale relative to a device. [CPU] is the
   ordinary case (never touched a device); the host copy is already the only
   copy, so there is nothing stale to guard against. *)
let test_does_not_refuse_on_cpu_with_auto_sync_off () =
  let sv = make_soa_vector () in
  let aos = Soa_vector.aos_vector sv in
  Vector.set_auto_sync aos false ;
  (match aos.Spoc_core.Vector_types.location with
  | Spoc_core.Vector_types.CPU -> ()
  | _ -> Alcotest.fail "expected a fresh vector to start at location CPU") ;
  (* The assertion IS the absence of a raise here: an uncaught exception from
     [scatter] fails this test case on its own; there is nothing further to
     check once control reaches the end of the function. *)
  Soa_vector.scatter sv

(* [Stale_GPU]: the HOST copy is the fresh one and the device is behind. That
   is the opposite of the hazard scatter guards against, so auto-sync off
   must not refuse here either. *)
let test_does_not_refuse_on_stale_gpu_with_auto_sync_off () =
  let sv = make_soa_vector () in
  let aos = Soa_vector.aos_vector sv in
  Vector.set_auto_sync aos false ;
  aos.Spoc_core.Vector_types.location <-
    Spoc_core.Vector_types.Stale_GPU (make_device 0) ;
  Soa_vector.scatter sv

(* [Both]: host and device agree, so there is nothing stale to refuse either —
   auto-sync off is irrelevant here for the same reason as [CPU]. *)
let test_does_not_refuse_on_both_with_auto_sync_off () =
  let sv = make_soa_vector () in
  let aos = Soa_vector.aos_vector sv in
  Vector.set_auto_sync aos false ;
  aos.Spoc_core.Vector_types.location <-
    Spoc_core.Vector_types.Both (make_device 0) ;
  Soa_vector.scatter sv

(* The other half of the guard: [Stale_CPU] with auto-sync back ON must NOT
   refuse. Pins that the predicate is conjunctive (stale AND auto-sync-off),
   not "stale" alone — the exact regression an over-eager tightening of this
   check would introduce.

   With auto-sync on, [ensure_cpu_sync] actually invokes the registered sync
   callback (unlike every other case above, where auto-sync off or a
   non-stale location means it never gets that far). Linking [sarek] pulls in
   the real callback, which drives a real device transfer this test's fake
   [Device.t] has no backing buffer for — so this test registers its own
   trivial callback first, to exercise scatter's OWN refusal-or-not decision
   in isolation from that unrelated machinery. *)
let test_does_not_refuse_on_stale_cpu_with_auto_sync_on () =
  Vector.register_sync_callback {Vector.sync = (fun _ -> true)} ;
  let sv = make_soa_vector () in
  let aos = Soa_vector.aos_vector sv in
  Vector.set_auto_sync aos true ;
  aos.Spoc_core.Vector_types.location <-
    Spoc_core.Vector_types.Stale_CPU (make_device 0) ;
  Soa_vector.scatter sv

let () =
  Alcotest.run
    "soa_vector_scatter_refuse"
    [
      ( "scatter",
        [
          Alcotest.test_case
            "refuses stale+auto-sync-off"
            `Quick
            test_refuses_when_stale_and_auto_sync_off;
          Alcotest.test_case
            "does not refuse on CPU"
            `Quick
            test_does_not_refuse_on_cpu_with_auto_sync_off;
          Alcotest.test_case
            "does not refuse on Stale_GPU"
            `Quick
            test_does_not_refuse_on_stale_gpu_with_auto_sync_off;
          Alcotest.test_case
            "does not refuse on Both"
            `Quick
            test_does_not_refuse_on_both_with_auto_sync_off;
          Alcotest.test_case
            "does not refuse on Stale_CPU + auto-sync on"
            `Quick
            test_does_not_refuse_on_stale_cpu_with_auto_sync_on;
        ] );
    ]
