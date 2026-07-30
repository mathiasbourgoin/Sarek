(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Cross-device MIGRATION of a transparent SoA vector (backlog-54).
 *
 * [Transfer.to_device vec dev] has an arm nothing else has: when [vec] is
 * resident on a DIFFERENT device it first drains that device into host storage,
 * then uploads the host copy to [dev]. That drain is a read-back, and it called
 * [copy_device_to_host] DIRECTLY — it did not go through
 * [Transfer.read_back_to_host], the function whose whole purpose is to decide
 * whether a read-back reads the packed AoS buffer or the N SoA leaves.
 *
 * So after a transparent SoA launch this arm reproduced BOTH halves of the
 * failure pair the SoA read-back work claims to have eliminated:
 *
 *   - with no packed buffer (the transparent path never allocates one), it
 *     raised [Failure "to_cpu: no device buffer to transfer from"] from inside
 *     what the caller asked to be a transfer;
 *   - with a packed buffer present from an earlier upload, it downloaded that
 *     buffer's PRE-LAUNCH bytes over the host storage and discarded the
 *     launch's output silently — no exception, no warning, wrong data.
 *
 * This was filed against PR #375 round 2 and REFUTED as already fixed. The
 * refutation was wrong: the fix covered [to_cpu] and the two cleanup paths, and
 * the migration arm was a fourth caller that the comment on
 * [read_back_to_host] even claimed to include. Round 3 routes it and this file
 * is the check that keeps it routed.
 *
 * A SEPARATE FILE, not a case in test_soa_emitter_equiv.ml, for two reasons:
 * it is the ratchet check for a finding that survived a review round, which the
 * pipeline requires to be self-contained; and half A's pre-fix red state is a
 * RAISE, so a shared binary would abort before the other half reported.
 *
 * Exit codes: 0 = pass (including an honest skip), 1 = an assertion fired,
 * 2 = setup could not produce the device pair this needs.
 *
 * Needs TWO devices, at least one of them CUDA/PTX (the SoA ABI dispatches
 * nowhere else). Locally that is ZLUDA on an AMD RX 7900 XTX — a CUDA/PTX
 * device, NOT NVIDIA hardware:
 *   LD_LIBRARY_PATH=$HOME/opt/zluda \
 *     dune exec sarek/tests/e2e/test_soa_cross_device_migration.exe
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Soa_vector = Spoc_core.Soa_vector
module Benchmarks = Test_helpers.Benchmarks

type ('a, 'b) vector = ('a, 'b) Vector.t

type float32 = float

(* Mutable so the kernel can write the y leaf in place. *)
type point3d = {mutable x : float32; mutable y : float32; mutable z : float32}
[@@sarek.type]

let scale_y_kernel =
  snd
    [%kernel
      fun (pts : point3d vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then pts.(tid).y <- pts.(tid).y *. 2.0]

let ir_of kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let is_ptx (dev : Device.t) = dev.Device.framework = "CUDA/PTX"

let n = 256

let threads = 128

(* One round of the sequence, parameterised on the ONE thing that distinguishes
   the two halves: whether a packed AoS buffer already exists on the source
   device when the transparent launch runs. That is what selects which pre-fix
   failure you get, and both must be pinned — a fix that only stopped the raise
   would leave the silent-wrong-data half live, and it is the worse of the two.

   No explicit [to_cpu] before the migration, deliberately: inserting one would
   drain the leaves by another route and the case would pass unfixed. *)
let check ~label ~(packed_first : bool) (src : Device.t) (dst : Device.t) =
  let sv = Soa_vector.create_transparent point3d_custom n in
  let y0 i = float_of_int (i + 1) in
  for i = 0 to n - 1 do
    Vector.set sv i {x = float_of_int i; y = y0 i; z = float_of_int (n - i)}
  done ;
  let outcome =
    match
      (* Give the vector a PACKED buffer on [src] first, in the [packed_first]
         half. This is what makes the buggy read-back SUCCEED instead of
         raising, which is what makes it silent. *)
      if packed_first then Transfer.to_device sv src ;
      (* Transparent SoA launch: run_vectors passes ~soa_abi:true, so on a
         CUDA/PTX device this binds the N-leaf ABI, doubles the y LEAF, and
         leaves the vector SoA-owned (location Stale_CPU src). *)
      Sarek.Execute.run_vectors
        ~device:src
        ~ir:(ir_of scale_y_kernel)
        ~args:[Vec sv; Int n]
        ~block:(Sarek.Execute.dims1d threads)
        ~grid:(Sarek.Execute.dims1d ((n + threads - 1) / threads))
        () ;
      Transfer.flush src ;
      (* THE CALL UNDER TEST. Resident on [src], asked for [dst] -> the
         migration arm drains [src] into host storage. *)
      Transfer.to_device sv dst
    with
    | () -> None
    | exception e -> Some (Printexc.to_string e)
  in
  let ok = ref true in
  (match outcome with
  | Some msg ->
      (* Reported rather than left to abort: a red state that is a crash is not
         an observation, and this half's pre-fix red IS a raise. *)
      Printf.printf "  migration raised: %s\n%!" msg ;
      ok := false
  | None ->
      for i = 0 to n - 1 do
        let got = (Vector.get sv i).y in
        let want = y0 i *. 2.0 in
        if Float.abs (got -. want) > 1e-3 then begin
          if !ok then
            Printf.printf
              "  migration data wrong @%d: got=%g want=%g (the stale packed \
               buffer holds %g)\n\
               %!"
              i
              got
              want
              (y0 i) ;
          ok := false
        end
      done) ;
  Printf.printf "  %-58s %s\n%!" label (if !ok then "OK" else "FAILED") ;
  !ok

let () =
  (* REGISTER the backends before asking for devices. [Device.init] only
     enumerates frameworks that have registered themselves, and registration is a
     link-time side effect that [Benchmarks.init] (Backend_loader) forces —
     without it [Device.all ()] came back EMPTY on a host with nine devices and
     this file reported "0 device(s) enumerated, none CUDA/PTX", a skip that was
     loud and wrong at the same time. Measured while writing this test. *)
  Benchmarks.init () ;
  ignore (Device.init ()) ;
  let devs = Device.all () in
  let ptx = Array.to_list devs |> List.filter is_ptx in
  match ptx with
  | [] ->
      (* A skip that names the device class it needs and why it was absent. Exit
         0: an absent device is not a regression. But it must never be
         SILENT — that shape is how four cases in test_soa_emitter_equiv.ml sat
         unexercised while runtest read green. *)
      Printf.printf
        "test_soa_cross_device_migration: SKIP - needs a CUDA/PTX device (the \
         SoA ABI dispatches on no other backend); %d device(s) enumerated, \
         none CUDA/PTX. Locally: LD_LIBRARY_PATH=$HOME/opt/zluda\n\
         %!"
        (Array.length devs) ;
      exit 0
  | src :: _ -> (
      (* [dst] only has to be a DIFFERENT device — it receives an ordinary packed
         upload, so any framework will do. Another CUDA/PTX device is preferred
         so the pair is homogeneous. *)
      let others =
        Array.to_list devs
        |> List.filter (fun (d : Device.t) -> d.Device.id <> src.Device.id)
      in
      let dst =
        match List.filter is_ptx others with
        | d :: _ -> Some d
        | [] -> ( match others with d :: _ -> Some d | [] -> None)
      in
      match dst with
      | None ->
          Printf.printf
            "test_soa_cross_device_migration: SKIP - needs TWO devices to \
             migrate BETWEEN; only 1 enumerated (%s). The migration arm is \
             unreachable with a single device.\n\
             %!"
            src.Device.name ;
          exit 0
      | Some dst ->
          Printf.printf
            "test_soa_cross_device_migration: %s -> %s\n%!"
            src.Device.name
            dst.Device.name ;
          let a =
            check
              ~label:"migration after a transparent SoA launch (no packed buf)"
              ~packed_first:false
              src
              dst
          in
          let b =
            check
              ~label:"migration after a transparent SoA launch (packed present)"
              ~packed_first:true
              src
              dst
          in
          if not (a && b) then exit 1)
