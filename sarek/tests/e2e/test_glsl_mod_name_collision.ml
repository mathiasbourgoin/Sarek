(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E regression: the GLSL integer-[mod] helper name must not collide with a
 * user identifier (CodeRabbit Major on PR #255).
 *
 * The GLSL backend lowers integer [Mod] to a call to a top-level helper
 * (default name [sarek_smod]). A scalar kernel parameter emits a push-constant
 * alias [#define <name> pc.<name>]; if a param is literally named [sarek_smod],
 * that macro would rewrite the helper declaration [int sarek_smod(...)] to
 * [int pc.sarek_smod(...)] - invalid GLSL, shader fails to compile. The fix
 * (Sarek_ir_glsl.compute_smod_name) renames the helper to a fresh, non-colliding
 * name ([sarek_smod_1], ...) used at both the declaration and every call site.
 *
 * This kernel does exactly that: a scalar divisor parameter NAMED [sarek_smod],
 * used as the right operand of [mod]. If the collision were unhandled the
 * Vulkan shader would not compile and the backend could not launch. Results are
 * compared bit-for-bit against OCaml [Int32.rem] on every backend that runs.
 *
 * Run with (surfaces the CUDA device too):
 *   LD_LIBRARY_PATH=$HOME/opt/zluda \
 *     dune exec sarek/tests/e2e/test_glsl_mod_name_collision.exe
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Test_helpers.Benchmarks.init_backends ()

(* The divisor parameter is deliberately named [sarek_smod] - the same name as
   the default remainder helper - to force the collision path. A negative
   divisor also re-checks the dividend-signed C remainder. *)
let collision_kernel =
  [%kernel
    fun (a : int32 vector)
        (r : int32 vector)
        (sarek_smod : int32)
        (n : int32) ->
      let open Std in
      let gid = global_thread_id in
      if gid < n then r.(gid) <- a.(gid) mod sarek_smod]

let dividends = [|-7l; 7l; -100l; 100l; -1l; 0l; 123l; -123l|]

let divisor = -3l

let n = Array.length dividends

let run_on_device (dev : Device.t) ir =
  let a = Vector.create Vector.int32 n in
  let r = Vector.create Vector.int32 n in
  for i = 0 to n - 1 do
    Vector.set a i dividends.(i) ;
    Vector.set r i 0l
  done ;
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Execute.Vec a; Execute.Vec r; Execute.Int32 divisor; Execute.Int n]
    ~block:(Execute.dims1d n)
    ~grid:(Execute.dims1d 1)
    () ;
  Transfer.flush dev ;
  Vector.to_array r

let verify got_r =
  let bad = ref 0 in
  for i = 0 to n - 1 do
    let expected = Int32.rem dividends.(i) divisor in
    if got_r.(i) <> expected then begin
      if !bad < 5 then
        Printf.printf
          "    mismatch @%d: %ld mod %ld got %ld exp %ld\n%!"
          i
          dividends.(i)
          divisor
          got_r.(i)
          expected ;
      incr bad
    end
  done ;
  !bad

let is_native (dev : Device.t) =
  dev.Device.framework = "Native" || dev.Device.framework = "Interpreter"

(* Gate native/interpreter (oracle) AND Vulkan: the collision this test pins is
   GLSL-specific, so a Vulkan device that is present must both LAUNCH (an
   unhandled collision makes the shader fail to compile) and match Int32.rem.
   Treating a Vulkan launch failure as a mere skip would mask the exact
   regression - so for a present Vulkan device, failure is hard. PTX/OpenCL do
   not use the GLSL helper; they stay report-only for infra robustness. *)
let is_gated (dev : Device.t) = is_native dev || dev.Device.framework = "Vulkan"

let () =
  let _, kirc = collision_kernel in
  let ir =
    match kirc.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "collision kernel has no IR"
  in
  let devs = Device.init () in
  print_endline
    "=== GLSL mod-helper name-collision E2E (param named sarek_smod) ===" ;
  let native_ran = ref false in
  let failed = ref false in
  Array.iter
    (fun (dev : Device.t) ->
      let native = is_native dev in
      let gated = is_gated dev in
      try
        let got_r = run_on_device dev ir in
        let bad = verify got_r in
        Printf.printf
          "  %-11s %-40s %s (%d/%d ok)\n%!"
          dev.Device.framework
          dev.Device.name
          (if bad = 0 then "PASS" else "FAIL")
          (n - bad)
          n ;
        if bad <> 0 then failed := true ;
        if native then native_ran := true
      with e ->
        Printf.printf
          "  %-11s %-40s %s (%s)\n%!"
          dev.Device.framework
          dev.Device.name
          (if gated then "ERROR" else "SKIP (backend could not launch)")
          (Printexc.to_string e) ;
        (* A GLSL compile failure from an unhandled name collision surfaces here
           as a Vulkan launch failure - gated (hard fail), since Vulkan is the
           backend this regression targets. *)
        if gated then failed := true)
    devs ;
  if not !native_ran then begin
    print_endline
      "test_glsl_mod_name_collision: FAILED - no native/interpreter device ran" ;
    exit 1
  end ;
  if !failed then begin
    print_endline "test_glsl_mod_name_collision: FAILED" ;
    exit 1
  end ;
  print_endline "test_glsl_mod_name_collision: PASSED"
