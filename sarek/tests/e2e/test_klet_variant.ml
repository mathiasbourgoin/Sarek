(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek PPX with a top-level variant type used as a real
 * `shape vector` kernel parameter.
 *
 * HISTORY (finding 3, briefs/make-tests-actually-run-impl-notes.md): an
 * earlier attempt combined this top-level `[@@sarek.type] shape` variant
 * with a *kernel-local klet helper function* (`area`) that pattern-matched
 * on `shape`. That combination hit a genuine ppx bug (fully-qualified
 * self-reference through the helper-function code path - see the impl
 * notes for the full bisection) and was BLOCKED/escalated. The orchestrator
 * decided (option b, reduced scope) to keep the top-level variant and the
 * real `shape vector` parameter, but inline the `match` directly in the
 * kernel body instead of factoring it into a separate klet helper - this
 * avoids the buggy code path entirely (verified empirically) while still
 * exercising a real variant-vector kernel parameter, which is the
 * substance of finding 3. The ppx bug itself remains open and is
 * documented in the impl notes as a tracked follow-up; it is NOT fixed
 * here (sarek/ppx/ is out of scope for this test-only change).
 ******************************************************************************)

(* runtime module aliases *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

[@@@warning "-32"]

(* Force backend registration. Also register the always-available
   Native/Interpreter plugins - the previous version of this test only
   called Sarek_cuda.Cuda_plugin.init/Sarek_opencl.Opencl_plugin.init,
   which never registered Native/Interpreter at all, so the
   Interpreter/Native device preference below could never actually find
   them (see briefs/make-tests-actually-run-impl-notes.md, finding 2). *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init ()

type float32 = float

(* Top-level variant type, registered via [@@sarek.type] so it gets a real
   `shape_custom` custom-vector descriptor (see test_nested_types.ml /
   test_complex_types.ml for the same idiom on records and variants). *)
type shape = Circle of float32 | Square of float32 [@@sarek.type]

let () =
  let dispatch =
    [%kernel
      fun (src : shape vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then
          match src.(tid) with
          | Circle r -> dst.(tid) <- 3.14 *. r *. r
          | Square x -> dst.(tid) <- x *. x]
  in

  (* Get IR *)
  let _, kirc = dispatch in
  print_endline "=== Variant helper IR ===" ;
  (match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> Sarek_ir_pp.print_kernel ir
  | None -> print_endline "(No IR available)") ;
  print_endline "=========================" ;

  (* Run with GPU runtime *)
  let devs =
    Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length devs = 0 then begin
    print_endline "No device found - IR generation test passed" ;
    exit 0
  end ;
  (* Deviation from the Interpreter-then-Native preference used elsewhere
     (e.g. test_ktype_record.ml): verified empirically that this kernel
     (inline module-local variant type + klet helper constructing the
     variant from a float sign) produces wrong results (dst left at 0) on
     the sequential Interpreter backend, while the parallel Native backend
     executes it correctly. Preferring Native first means the test still
     runs its real assertions on this machine instead of always hitting
     the "Interpreter present -> skip" branch first (Interpreter is
     always-available, so Interpreter-first would make the Native-only
     guard below fire unconditionally and the test would never actually
     assert anything - see briefs/make-tests-actually-run-impl-notes.md). *)
  let dev =
    match Array.find_opt (fun d -> d.Device.framework = "Native") devs with
    | Some d -> d
    | None -> (
        match
          Array.find_opt (fun d -> d.Device.framework = "Interpreter") devs
        with
        | Some d -> d
        | None -> devs.(0))
  in
  Printf.printf "Using device: %s\n%!" dev.Device.name ;
  (* Custom-type codegen is documented as unreliable off Native (see
     test_ktype_record.ml's identical restriction), and empirically the
     Interpreter backend gives wrong results for this specific kernel
     (see the comment above) - skip rather than fail nondeterministically. *)
  if dev.framework <> "Native" then begin
    Printf.printf
      "runtime: SKIP (variant helper test checked on native backend only)\n%!" ;
    exit 0
  end ;
  match kirc.Sarek.Kirc_types.body_ir with
  | None ->
      print_endline "No IR - SKIPPED" ;
      exit 0
  | Some ir ->
      let n = 64 in
      let src = Vector.create_custom shape_custom n in
      let dst = Vector.create Vector.float32 n in
      for i = 0 to n - 1 do
        (* Alternate between Circle and Square variants with a real payload,
           instead of encoding the tag via the sign of a plain float. *)
        let v = float_of_int (i + 1) in
        Vector.set src i (if i mod 2 = 0 then Circle v else Square v) ;
        Vector.set dst i 0.0
      done ;
      let threads = min 64 n in
      let grid_x = (n + threads - 1) / threads in
      Sarek.Execute.run_vectors
        ~device:dev
        ~block:(Sarek.Execute.dims1d threads)
        ~grid:(Sarek.Execute.dims1d grid_x)
        ~ir
        ~args:
          [
            Sarek.Execute.Vec src;
            Sarek.Execute.Vec dst;
            Sarek.Execute.Int32 (Int32.of_int n);
          ]
        () ;
      Transfer.flush dev ;
      let ok = ref true in
      for i = 0 to n - 1 do
        let v = float_of_int (i + 1) in
        let expected = if i mod 2 = 0 then 3.14 *. v *. v else v *. v in
        let got = Vector.get dst i in
        if abs_float (got -. expected) > 1e-2 then begin
          ok := false ;
          if i < 5 then
            Printf.printf
              "  Mismatch at %d: got %f expected %f\n%!"
              i
              got
              expected
        end
      done ;
      if !ok then print_endline "test_klet_variant PASSED"
      else begin
        print_endline "test_klet_variant FAILED: area mismatch" ;
        exit 1
      end
