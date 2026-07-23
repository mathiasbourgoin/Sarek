(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E positive control: integer [Mod] evaluates each operand exactly ONCE.
 *
 * The GLSL/Vulkan backend lowers signed integer [Mod] to a C-truncated
 * remainder. The naive rewrite [a - b * (a / b)] re-emits both operands, so an
 * operand carrying a side effect - a value-returning atomic, an effectful
 * helper call - would fire TWICE (silent double mutation + wrong result).
 * Every other backend single-evaluates [%]; the fix routes GLSL [Mod] through
 * a [sarek_smod(a, b)] helper call, and a GLSL function evaluates each argument
 * exactly once, so the effect fires once.
 *
 * This test is the red-on-mutation control for that guarantee. The mod operand
 * is an INLINE global-atomic increment (returns the old counter value); the
 * remainder result is stored so the atomic cannot be dead-code eliminated. Each
 * of [n] threads runs the kernel once, so:
 *   - single evaluation  => counter ends at n   (PASS)
 *   - double evaluation  => counter ends at 2*n (FAIL - catches the naive form)
 * The final counter is order-independent (a sum of increments), so the witness
 * is deterministic regardless of thread scheduling.
 *
 * Run with (surfaces the CUDA device too):
 *   LD_LIBRARY_PATH=$HOME/opt/zluda \
 *     dune exec sarek/tests/e2e/test_glsl_mod_single_eval.exe
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Gpu = Sarek_stdlib.Gpu
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Test_helpers.Benchmarks.init_backends ()

(* r.(gid) = (atomic_add_global counter[0] 1) mod 1000. The atomic call is the
   left operand of [mod] directly (not let-bound), so a Mod lowering that
   re-emits its operands fires the atomic twice per thread. *)
let single_eval_kernel =
  [%kernel
    fun (counter : int32 vector) (r : int32 vector) (n : int32) ->
      let open Std in
      let open Gpu in
      let gid = global_thread_id in
      if gid < n then r.(gid) <- atomic_add_global_int32 counter 0 1l mod 1000l]

let block_size = 256

let n = 4096 (* multiple of block_size: 16 blocks x 256 threads *)

let run_on_device (dev : Device.t) ir =
  let counter = Vector.create Vector.int32 1 in
  let r = Vector.create Vector.int32 n in
  Vector.set counter 0 0l ;
  for i = 0 to n - 1 do
    Vector.set r i 0l
  done ;
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Execute.Vec counter; Execute.Vec r; Execute.Int n]
    ~block:(Execute.dims1d block_size)
    ~grid:(Execute.dims1d (n / block_size))
    () ;
  Transfer.flush dev ;
  Int32.to_int (Vector.get counter 0)

(* Gate the backends whose atomics we can execute and read back here: the
   GLSL/Vulkan backend this fix targets, plus the CUDA/PTX and OpenCL GPU
   backends and the native/interpreter oracles. A backend that runs but
   double-counts is a hard failure. A GPU backend that cannot launch for infra
   reasons is reported only. *)
let is_native (dev : Device.t) =
  dev.Device.framework = "Native" || dev.Device.framework = "Interpreter"

let () =
  let _, kirc = single_eval_kernel in
  let ir =
    match kirc.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "single_eval kernel has no IR"
  in
  let devs = Device.init () in
  print_endline "=== integer Mod single-evaluation E2E (atomic operand) ===" ;
  Printf.printf
    "  expect final counter = n = %d (2*n = %d means double-eval)\n%!"
    n
    (2 * n) ;
  let native_ran = ref false in
  let failed = ref false in
  Array.iter
    (fun (dev : Device.t) ->
      let native = is_native dev in
      try
        let got = run_on_device dev ir in
        let ok = got = n in
        Printf.printf
          "  %-11s %-40s %s (counter=%d)\n%!"
          dev.Device.framework
          dev.Device.name
          (if ok then "PASS" else "FAIL")
          got ;
        if not ok then failed := true ;
        if native then native_ran := true
      with e ->
        let msg =
          match e with
          | Sarek_backend_error.Backend_error.Backend_error t ->
              Sarek_backend_error.Backend_error.to_string t
          | e -> Printexc.to_string e
        in
        Printf.printf
          "  %-11s %-40s %s (%s)\n%!"
          dev.Device.framework
          dev.Device.name
          (if native then "ERROR" else "SKIP (backend could not launch)")
          msg ;
        if native then failed := true)
    devs ;
  if not !native_ran then begin
    print_endline
      "test_glsl_mod_single_eval: FAILED - no native/interpreter device ran" ;
    exit 1
  end ;
  if !failed then begin
    print_endline "test_glsl_mod_single_eval: FAILED" ;
    exit 1
  end ;
  print_endline "test_glsl_mod_single_eval: PASSED"
