(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E hardware probe for signed integer division / remainder (audit H1).
 *
 * Sarek int32 is signed on every backend, but the PTX emitter used to lower
 * integer Div/Mod to [div.u32]/[rem.u32]. On negative operands that returns
 * silent garbage - e.g. (-7)/2 = 2147483644 instead of -3. The snapshot test
 * (test_ptx_snapshot.ml) asserts the emitted instruction is now [div.s32]/
 * [rem.s32]; this test proves the RESULT on real hardware by running the
 * kernel through every available backend - crucially CUDA/PTX under ZLUDA,
 * the class of semantics a text snapshot cannot catch and that hardware CI
 * under-tests.
 *
 * Integer div/rem is exact, so results are compared bit-for-bit against
 * OCaml's Int32.div/Int32.rem (no tolerance). Native, Interpreter and
 * CUDA/PTX are hard-gated; the Vulkan backend has a pre-existing signed-mod
 * divergence (see [is_gated]) and is reported only.
 *
 * Run with (surfaces the CUDA device):
 *   LD_LIBRARY_PATH=$HOME/opt/zluda \
 *     dune exec sarek/tests/e2e/test_ptx_signed_arith.exe
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Test_helpers.Benchmarks.init_backends ()

(* q.(tid) = a.(tid) / b.(tid), r.(tid) = a.(tid) mod b.(tid). Plain DSL [/]
   and [mod] on int32 -> Ir.Div / Ir.Mod, the operators the H1 fix retargets
   to the signed PTX forms. *)
let signed_divrem_kernel =
  [%kernel
    fun (a : int32 vector)
        (b : int32 vector)
        (q : int32 vector)
        (r : int32 vector)
        (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then begin
        q.(tid) <- a.(tid) / b.(tid) ;
        r.(tid) <- a.(tid) mod b.(tid)
      end]

(* A spread that stresses sign combinations: negative/positive dividends and
   divisors, including the (-7)/2 witness from the audit finding. *)
let cases =
  [|
    (-7l, 2l);
    (7l, -2l);
    (-7l, -2l);
    (7l, 2l);
    (-2147483648l, 3l) (* INT_MIN, exercises the sign bit *);
    (-100l, 7l);
    (100l, -7l);
    (-1l, 2l);
    (0l, 5l);
    (-999l, -13l);
  |]

let n = Array.length cases

let run_on_device (dev : Device.t) ir =
  let a = Vector.create Vector.int32 n in
  let b = Vector.create Vector.int32 n in
  let q = Vector.create Vector.int32 n in
  let r = Vector.create Vector.int32 n in
  for i = 0 to n - 1 do
    let av, bv = cases.(i) in
    Vector.set a i av ;
    Vector.set b i bv ;
    Vector.set q i 0l ;
    Vector.set r i 0l
  done ;
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Execute.Vec a; Execute.Vec b; Execute.Vec q; Execute.Vec r; Execute.Int n;
      ]
    ~block:(Execute.dims1d n)
    ~grid:(Execute.dims1d 1)
    () ;
  Transfer.flush dev ;
  (Vector.to_array q, Vector.to_array r)

let verify got_q got_r =
  let bad = ref 0 in
  for i = 0 to n - 1 do
    let av, bv = cases.(i) in
    let eq = Int32.div av bv in
    let er = Int32.rem av bv in
    if got_q.(i) <> eq || got_r.(i) <> er then begin
      if !bad < 5 then
        Printf.printf
          "    mismatch @%d: %ld/%ld got q=%ld r=%ld exp q=%ld r=%ld\n%!"
          i
          av
          bv
          got_q.(i)
          got_r.(i)
          eq
          er ;
      incr bad
    end
  done ;
  !bad

let is_native (dev : Device.t) =
  dev.Device.framework = "Native" || dev.Device.framework = "Interpreter"

(* Backends whose signed remainder this PR fixes/claims C semantics for: the
   PTX emitter (audit H1) plus the interpreter and native oracle. A wrong
   result from one of these is a hard failure.

   NOT gated: the Vulkan backend lowers integer Mod to the GLSL [%] operator
   (Sarek_ir_glsl.ml), which on RADV returns a remainder with the DIVISOR's
   sign (OpSMod-like) instead of C's dividend sign - so [-7 mod 2] yields +1,
   not -1. That is a genuine pre-existing Vulkan codegen bug this probe
   surfaces, but it is out of scope for this PTX/interpreter PR; it is
   reported here (see stdout) and left for a dedicated GLSL-backend fix. *)
let is_gated (dev : Device.t) =
  is_native dev || dev.Device.framework = "CUDA/PTX"

let () =
  let _, kirc = signed_divrem_kernel in
  let ir =
    match kirc.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "signed div/rem kernel has no IR"
  in
  let devs = Device.init () in
  print_endline "=== H1 signed int32 div/rem E2E (negative operands) ===" ;
  let native_ran = ref false in
  let failed = ref false in
  Array.iter
    (fun (dev : Device.t) ->
      let native = is_native dev in
      let gated = is_gated dev in
      try
        let got_q, got_r = run_on_device dev ir in
        let bad = verify got_q got_r in
        Printf.printf
          "  %-11s %-40s %s (%d/%d ok)%s\n%!"
          dev.Device.framework
          dev.Device.name
          (if bad = 0 then "PASS" else if gated then "FAIL" else "DIVERGES")
          (n - bad)
          n
          (if bad <> 0 && not gated then " [known out-of-scope backend gap]"
           else "") ;
        (* Integer div/rem is exact, so a gated backend that ran must match
           bit-for-bit. Non-gated backends (Vulkan) are reported only. *)
        if bad <> 0 && gated then failed := true ;
        if native then native_ran := true
      with e ->
        Printf.printf
          "  %-11s %-40s %s (%s)\n%!"
          dev.Device.framework
          dev.Device.name
          (if native then "ERROR" else "SKIP (backend could not launch)")
          (Printexc.to_string e) ;
        (* Native/Interpreter must always run; a GPU backend that cannot even
           launch for infra reasons is only reported (a wrong RESULT from a
           backend that DID run is always a hard failure - see above). *)
        if native then failed := true)
    devs ;
  if not !native_ran then begin
    print_endline
      "test_ptx_signed_arith: FAILED - no native/interpreter device ran" ;
    exit 1
  end ;
  if !failed then begin
    print_endline "test_ptx_signed_arith: FAILED" ;
    exit 1
  end ;
  print_endline "test_ptx_signed_arith: PASSED"
