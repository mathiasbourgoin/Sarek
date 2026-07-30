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
 * Exit codes: 0 = pass, INCLUDING both honest skips (no CUDA/PTX device, or only
 * one device to migrate between) — an absent device is not a regression, and
 * every skip prints a line naming the class it needed; 1 = an assertion fired.
 * There is no third code: a previous version of this header documented
 * "2 = setup could not produce the device pair this needs" while both skip paths
 * exit 0, so the row described a state this file cannot reach.
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
module Gpu_memory = Spoc_core.Gpu_memory
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

(* [free_buffer] is PER-DEVICE; [soa_leaves_live] is WHOLE-VECTOR. Those two
   scopes only agree when the free covered every device, and [soa_free_leaves]
   used to assign [false] regardless — so releasing one device disowned the
   leaves still live on every other one.

   The consequence is data loss, not a leak, and this case is built to show that
   rather than to count bytes. [free_buffer sv other] is called on a device this
   vector has NOTHING on: no packed buffer, no leaves, and [other] is not what
   [location] names, so every arm of the function correctly declines to touch
   anything. It is as close to a no-op as the API offers. Pre-fix it still
   cleared the flag, and the very next read-back — of results sitting untouched
   in [src]'s leaves — followed the packed buffer instead. On this path there is
   no packed buffer, so it raises; give the vector one first and it returns
   pre-launch bytes silently, which is the same pair of failures the migration
   arm had.

   Deliberately NO [to_cpu] before the free: draining first would move the
   results to host storage by another route and the case would pass unfixed.

   The byte split the reviewer measured is asserted too, as a second and weaker
   observation: [free_buffer sv dst] after a migration must release [dst]'s
   packed buffer and leave [src]'s leaves alone — a per-device free that freed
   both would be a different bug in the opposite direction. *)
let check_per_device_leaf_free (src : Device.t) (other : Device.t) =
  let sv = Soa_vector.create_transparent point3d_custom n in
  let y0 i = float_of_int (i + 1) in
  for i = 0 to n - 1 do
    Vector.set sv i {x = float_of_int i; y = y0 i; z = float_of_int (n - i)}
  done ;
  Gc.full_major () ;
  let ok = ref true in
  let fail fmt =
    Printf.ksprintf
      (fun s ->
        Printf.printf "  %s\n%!" s ;
        ok := false)
      fmt
  in
  let base = Gpu_memory.usage () in
  Sarek.Execute.run_vectors
    ~device:src
    ~ir:(ir_of scale_y_kernel)
    ~args:[Vec sv; Int n]
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d ((n + threads - 1) / threads))
    () ;
  Transfer.flush src ;
  let leaves_bytes = Gpu_memory.usage () - base in
  if leaves_bytes <= 0 then
    fail
      "the launch allocated %d bytes on %s, so the release assertions below \
       would be vacuous"
      leaves_bytes
      src.Device.name ;
  (* Half 1: freeing a device the vector holds nothing on must not disown the
     device it does. *)
  Transfer.free_buffer sv other ;
  if not (Transfer.has_device_data sv src) then
    fail
      "after free_buffer on %s (which holds nothing), has_device_data says %s \
       holds nothing either — the whole-vector leaf flag was cleared by a \
       per-device free"
      other.Device.name
      src.Device.name ;
  if Gpu_memory.usage () - base <> leaves_bytes then
    fail
      "free_buffer on %s released %d bytes; it holds none of this vector's \
       memory and must release nothing"
      other.Device.name
      (leaves_bytes - (Gpu_memory.usage () - base)) ;
  (match Transfer.to_cpu ~force:true sv with
  | () ->
      for i = 0 to n - 1 do
        let got = (Vector.get sv i).y and want = y0 i *. 2.0 in
        if Float.abs (got -. want) > 1e-3 && !ok then
          fail
            "read-back after free_buffer on an unrelated device is wrong @%d: \
             got=%g want=%g (the pre-launch host value is %g)"
            i
            got
            want
            (y0 i)
      done
  | exception e ->
      fail
        "to_cpu after free_buffer on an unrelated device raised: %s"
        (Printexc.to_string e)) ;
  Transfer.free_all_buffers sv ;
  ignore (Sys.opaque_identity sv) ;
  (* Half 2: the byte split of a per-device free after a real migration. *)
  let sv2 = Soa_vector.create_transparent point3d_custom n in
  for i = 0 to n - 1 do
    Vector.set sv2 i {x = float_of_int i; y = y0 i; z = float_of_int (n - i)}
  done ;
  Gc.full_major () ;
  let base2 = Gpu_memory.usage () in
  Sarek.Execute.run_vectors
    ~device:src
    ~ir:(ir_of scale_y_kernel)
    ~args:[Vec sv2; Int n]
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d ((n + threads - 1) / threads))
    () ;
  Transfer.flush src ;
  let after_launch2 = Gpu_memory.usage () in
  Transfer.to_device sv2 other ;
  let after_migration = Gpu_memory.usage () in
  let packed_bytes = after_migration - after_launch2 in
  if packed_bytes <= 0 then
    fail
      "the migration to %s allocated %d bytes, so the split below is not \
       measurable"
      other.Device.name
      packed_bytes ;
  Transfer.free_buffer sv2 other ;
  let released = after_migration - Gpu_memory.usage () in
  if released <> packed_bytes then
    fail
      "free_buffer on %s released %d bytes, want exactly its own %d (%s's \
       leaves hold the other %d and are not this call's to free)"
      other.Device.name
      released
      packed_bytes
      src.Device.name
      (after_launch2 - base2) ;
  Transfer.free_all_buffers sv2 ;
  if Gpu_memory.usage () > base2 then
    fail
      "after free_buffer on %s and free_all_buffers, %d bytes are still held — \
       the per-device free orphaned %s's leaves"
      other.Device.name
      (Gpu_memory.usage () - base2)
      src.Device.name ;
  ignore (Sys.opaque_identity sv2) ;
  Printf.printf
    "  %-58s %s\n%!"
    "a per-device free keeps another device's leaves"
    (if !ok then "OK" else "FAILED") ;
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
         so the pair is homogeneous.

         "Any framework" includes the CPU backends, and that holds for the BYTE
         assertions in [check_per_device_leaf_free] too, which is the half that
         could plausibly depend on it: the accounting is not the backend's.
         [Transfer.ensure_buffer] calls [Gpu_memory.track_alloc (size *
         elem_size)] for every framework (Transfer.ml:232) on a wrapper whose
         [size]/[elem_size] come from the requested length and element size
         (Transfer.ml:150-152), and [free_buffer] tracks the same product back. So
         a Native or Interpreter [dst] still yields a non-zero packed-byte delta
         and the split is still measurable — no GPU-only restriction on [dst] is
         needed, and adding one would drop the heterogeneous pair, which is the
         more interesting migration. *)
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
          (* Same two-device fixture, a different question: not "does the
             migration drain correctly" but "does a PER-DEVICE free respect the
             device boundary". It lives here because this is the only test with
             two devices in hand. *)
          let c = check_per_device_leaf_free src dst in
          if not (a && b && c) then exit 1)
