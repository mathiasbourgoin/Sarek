(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * float64 literals, scalar conversions, and the softmath name collision.
 *
 * One sub-expression per output slot, each compared BIT-FOR-BIT against the
 * same expression evaluated in OCaml binary64. Slot-per-expression matters:
 * the three defects below were found by reading which slot moved, and a single
 * fused expression would only have said "wrong somewhere".
 *
 * 1. LITERAL PRECISION. [Sarek_quote]/[Sarek_quote_ir]/[Sarek_native_gen_base]
 *    reconstructed every float constant with [string_of_float], which is
 *    "%.12g" - so [3.14159265358979312G] reached the backends as
 *    3.14159265359, losing about eleven bits. The whole point of the `G`
 *    suffix is a binary64 literal, and it was not delivering one.
 *
 *    test_float64_kernel_arith, the `G`-suffix regression guard, could not see
 *    this: its literals are 0.0G, 2.0G and 4.0G, all exact in twelve digits.
 *    Slots 1 and 3 below are red before the fix and exact after it.
 *
 * 2. MISSING CONVERSIONS. [Float64.of_int32] and its siblings are declared in
 *    Sarek_float64/Float64.ml and type-check in the DSL, but had no arm in the
 *    interpreter or in the GLSL backend - "Unknown intrinsic". int32 is the
 *    DSL's integer type (thread ids, loop counters, scalar parameters), so
 *    [of_int32] is the conversion user code reaches for first.
 *
 * 3. SOFTMATH NAME COLLISION. On GLSL a SCALAR kernel parameter is exposed as
 *    [#define <name> pc.<name>] - a textual macro. The f64 transcendental
 *    helpers declared locals called n, t, k, a, r, s, z, j..., and are emitted
 *    as their own top-level functions, so the GLSL/WGSL shadow-rename pre-pass
 *    (which walks the kernel BODY) never saw them. A kernel with a scalar
 *    parameter named [n] plus any f64 transcendental emitted
 *    [int pc.n = int(j);] and glslang rejected the whole shader.
 *
 *    This test is the regression guard, and it only guards while the parameter
 *    below is named [n] and slot 5 calls a transcendental. Renaming that
 *    parameter silently disarms it.
 *
 * Run with: dune exec sarek/tests/e2e/test_f64_scalar_conversions.exe
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Test_helpers.Benchmarks.init_backends ()

(* [n] is load-bearing: it is a softmath helper local name (see 3 above). *)
let probe_kernel =
  [%kernel
    fun (out : float64 vector) (n : int32) ->
      let open Std in
      let open Sarek_float64 in
      let t = global_thread_id in
      if t < 1l then begin
        out.(0l) <- Float64.of_int32 n ;
        out.(1l) <- 2.0G *. 3.14159265358979312G ;
        out.(2l) <- 1.0G /. Float64.of_int32 n ;
        out.(3l) <- 3.14159265358979312G /. 4.0G ;
        out.(4l) <- Float64.of_int32 (Float64.to_int32 2.75G) ;
        out.(5l) <- Float64.of_float32 1.5 ;
        out.(6l) <- Float64.cos (3.14159265358979312G /. 2.0G)
      end]

let n = 4

let pi = 3.14159265358979312

let labels =
  [|
    "of_int32 n";
    "2 * pi (literal precision)";
    "1 / of_int32 n";
    "pi / 4 (literal precision)";
    "of_int32 (to_int32 2.75) -> 2";
    "of_float32 1.5 (widening, exact)";
    "cos(pi/2) (softmath, param named n)";
  |]

let expected =
  [|
    float_of_int n;
    2.0 *. pi;
    1.0 /. float_of_int n;
    pi /. 4.0;
    2.0;
    1.5;
    Stdlib.cos (pi /. 2.0);
  |]

let slots = Array.length labels

let describe = function
  | Sarek_interp.Interp_error.Interpreter_error e ->
      Sarek_interp.Interp_error.error_to_string e
  | e -> Printexc.to_string e

(* Slots 0-5 are exact arithmetic on values binary64 represents exactly, so they
   are compared BIT-FOR-BIT on every device; a tolerance there could only hide a
   real regression.

   The transcendental in slot 6 is different, and an earlier version of this
   test got it wrong: it justified a zero tolerance with "a shared softmath
   polynomial evaluated identically everywhere". That premise holds only for the
   backends that route Float64 transcendentals through Sarek_ir_softmath - GLSL
   and PTX - plus Native/Interpreter, which literally call [Stdlib.cos]. CUDA
   and OpenCL map [cos] straight to the device math library, specified to ~2 and
   4 ulp respectively and not bit-identical to glibc. The exact comparison
   happened to pass here only because no CUDA/OpenCL device is present locally;
   the first CI run on such a box would have failed it for a non-regression
   reason. Slot 6 therefore gets an ulp-scaled tolerance on those backends. Its
   PURPOSE is unaffected: it exists to force a softmath helper into the shader
   so the push-constant collision can fire, and that works at any tolerance. *)
(* Vulkan is NOT host-exact: it runs the generated softmath polynomial while
   [expected] is [Stdlib.cos]. The two agree bit-for-bit on this input today,
   measured — but nothing makes that a contract, so requiring it would be a test
   that passes for a reason it does not state. Native and Interpreter DO call
   Stdlib.cos literally, so they are the only exact rows. *)
let transcendental_tolerance framework want =
  match framework with
  | "Native" | "Interpreter" -> 0.0
  | "Vulkan" -> 1e-12
  | _ -> 4.0 *. epsilon_float *. Stdlib.abs_float want

let transcendental_slot = 6

let check framework name got =
  let bad = ref 0 in
  let tol i =
    if i <> transcendental_slot then 0.0
    else transcendental_tolerance framework expected.(i)
  in
  Array.iteri
    (fun i l ->
      if Stdlib.abs_float (got.(i) -. expected.(i)) > tol i then begin
        Printf.printf
          "    %-38s got=%.17g want=%.17g <-- WRONG\n%!"
          l
          got.(i)
          expected.(i) ;
        incr bad
      end)
    labels ;
  Printf.printf
    "  %-11s %-40s %s\n%!"
    framework
    name
    (if !bad = 0 then "PASS" else "FAIL") ;
  !bad = 0

let run_on_device (dev : Device.t) ir =
  let out = Vector.create Vector.float64 slots in
  for i = 0 to slots - 1 do
    Vector.set out i 0.0
  done ;
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Execute.Vec out; Execute.Int n]
    ~block:(Execute.dims1d 1)
    ~grid:(Execute.dims1d 1)
    () ;
  Transfer.flush dev ;
  Vector.to_array out

let () =
  let _, kirc = probe_kernel in
  let ir =
    match kirc.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "f64 conversion probe kernel has no IR"
  in
  print_endline "=== float64 literals, scalar conversions, softmath naming ===" ;
  let devs = Device.init () in
  let ran = ref 0 in
  let failures = ref 0 in
  Array.iter
    (fun (dev : Device.t) ->
      let framework = dev.Device.framework in
      let native = framework = "Native" || framework = "Interpreter" in
      (* The kernel is float64 throughout, so a device without fp64 cannot run
         it - skipping it is correct, not a silent pass. *)
      if native || Device.allows_fp64 dev then begin
        incr ran ;
        try
          if not (check framework dev.Device.name (run_on_device dev ir)) then
            incr failures
        with e ->
          Printf.printf
            "  %-11s %-40s ERROR (%s)\n%!"
            framework
            dev.Device.name
            (describe e) ;
          incr failures
      end)
    devs ;
  (* A run that reached no device proves nothing. *)
  if !ran = 0 then begin
    print_endline
      "test_f64_scalar_conversions: FAILED - no fp64-capable device available" ;
    exit 1
  end ;
  Printf.printf "  %d device(s), %d failure(s)\n%!" !ran !failures ;
  if !failures > 0 then exit 1
