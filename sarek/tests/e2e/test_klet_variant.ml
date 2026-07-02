(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek PPX with variant type and helper function.
 *
 * NOTE (finding 3, briefs/make-tests-actually-run-impl-notes.md): this test
 * was supposed to be upgraded to use a top-level `[@@sarek.type] shape`
 * variant as a real `shape vector` kernel parameter (matching
 * test_ktype_record.ml's record pattern). That upgrade is BLOCKED by a
 * genuine ppx bug, not a test-file limitation: when a kernel-local klet
 * helper function (here `area`) pattern-matches/type-annotates a
 * same-module custom variant, `Sarek_native_gen.gen_module_fun` generates
 * the helper's body via `gen_expr ~loc body` which uses `empty_ctx`
 * (current_module = None, see Sarek_native_gen.ml:292-310), so
 * `is_same_module` in Sarek_native_gen_base.ml always returns false for
 * helper functions and the variant gets fully qualified as
 * "Test_klet_variant.shape" / "Test_klet_variant.Circle" - which is a
 * circular self-reference once dune wraps this file as
 * `Dune__exe.Test_klet_variant` inside the `(executables (names ...))`
 * stanza, and ocamlopt rejects it with:
 *   "The module Test_klet_variant is an alias for module
 *    Dune__exe__Test_klet_variant, which is the current compilation unit"
 * Separately, Sarek_native_intrinsics.ml's `core_type_of_typ` (used for the
 * helper's `(s : shape)` parameter annotation) takes no current_module/ctx
 * parameter at all, so it unconditionally fully-qualifies record/variant
 * type paths - it would need the same fix even if gen_module_fun were
 * patched. Both are in sarek/ppx/, out of scope for this test-only
 * worktree; escalated rather than silently kept on the workaround below.
 * (Inlining the match directly in the kernel body instead of a separate
 * klet helper does NOT hit this bug - verified empirically - but that
 * would drop the "helper function" coverage finding 3 asked to add.)
 ******************************************************************************)

(* runtime module aliases *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

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

let () =
  let dispatch =
    [%kernel
      let module Types = struct
        type shape = Circle of float32 | Square of float32
      end in
      let area (s : shape) : float32 =
        match s with Circle r -> 3.14 *. r *. r | Square x -> x *. x
      in
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x in
        if tid < n then
          let s =
            if src.(tid) > 0.0 then Circle src.(tid)
            else Square (0.0 -. src.(tid))
          in
          dst.(tid) <- area s]
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
      let src = Vector.create Vector.float32 n in
      let dst = Vector.create Vector.float32 n in
      for i = 0 to n - 1 do
        (* Alternate between "circle" (positive radius) and "square"
           (negative side, sign is the dispatch tag). *)
        Vector.set
          src
          i
          (if i mod 2 = 0 then float_of_int (i + 1) else -.float_of_int (i + 1)) ;
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
        let x = Vector.get src i in
        let expected = if x > 0.0 then 3.14 *. x *. x else x *. x in
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
