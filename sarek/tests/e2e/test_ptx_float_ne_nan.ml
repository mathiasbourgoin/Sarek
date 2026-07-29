(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E hardware probe for float [<>] against NaN (H2 comparison family).
 *
 * The PTX emitter used to lower a float [<>] to [setp.ne.f32]/[setp.ne.f64].
 * Those are PTX's ORDERED not-equal, i.e. false as soon as either operand is
 * NaN. Every other backend disagrees: the C-family emitters (CUDA, OpenCL,
 * Metal, GLSL, WGSL) emit [!=], which is true for NaN, and both the native
 * path and the interpreter use OCaml's structural [<>], also true. So
 * [nan <> x] returned 0 on PTX and 1 everywhere else - a silent wrong answer,
 * not a crash, on one backend only. The fix emits the UNORDERED [setp.neu].
 *
 * test_ptx_snapshot.ml asserts the emitted instruction (both polarities:
 * [setp.neu.*] present AND [setp.ne.f32]/[setp.ne.f64] absent). This test
 * proves the RESULT on real hardware, running the kernel on every available
 * device against a pure-OCaml reference.
 *
 * NaN reaches the kernel through the input VECTOR, written host-side - the DSL
 * needs no NaN literal for this path.
 *
 * The reference is OCaml's [<>] on the same float pair, which is the semantics
 * every non-PTX backend already implements. Note the deliberate asymmetry that
 * is also pinned here: [=] on NaN is FALSE in OCaml, in C and in ordered PTX
 * [setp.eq], so the eq column must stay 0 for every NaN row. A backend that
 * "fixed" Eq to the unordered [setp.equ] would fail this test.
 *
 * Run with (surfaces the CUDA/PTX device):
 *   LD_LIBRARY_PATH=$HOME/opt/zluda \
 *     dune exec sarek/tests/e2e/test_ptx_float_ne_nan.exe
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Test_helpers.Benchmarks.init_backends ()

(* ne.(tid) = (a.(tid) <> b.(tid)), eq.(tid) = (a.(tid) = b.(tid)), as 0/1.
   Plain DSL [<>] and [=] on float32 -> Ir.Ne / Ir.Eq, the operators the
   comparison-family lowering retargets. *)
let float_ne_kernel_f32 =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (ne : int32 vector)
        (eq : int32 vector)
        (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then begin
        if a.(tid) <> b.(tid) then ne.(tid) <- 1l else ne.(tid) <- 0l ;
        if a.(tid) = b.(tid) then eq.(tid) <- 1l else eq.(tid) <- 0l
      end]

let float_ne_kernel_f64 =
  [%kernel
    fun (a : float64 vector)
        (b : float64 vector)
        (ne : int32 vector)
        (eq : int32 vector)
        (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then begin
        if a.(tid) <> b.(tid) then ne.(tid) <- 1l else ne.(tid) <- 0l ;
        if a.(tid) = b.(tid) then eq.(tid) <- 1l else eq.(tid) <- 0l
      end]

let nan = Float.nan

(* Rows 0-4 carry a NaN and are the ones the ordered/unordered distinction
   decides; rows 5-8 are NaN-free controls so a backend that returned a
   constant could not pass. *)
let cases =
  [|
    (nan, 1.0);
    (1.0, nan);
    (nan, nan);
    (nan, 0.0);
    (nan, Float.infinity);
    (1.0, 1.0);
    (1.0, 2.0);
    (0.0, 0.0);
    (Float.infinity, Float.infinity);
  |]

let n = Array.length cases

let run_f32 (dev : Device.t) ir =
  let a = Vector.create Vector.float32 n in
  let b = Vector.create Vector.float32 n in
  let ne = Vector.create Vector.int32 n in
  let eq = Vector.create Vector.int32 n in
  for i = 0 to n - 1 do
    let av, bv = cases.(i) in
    Vector.set a i av ;
    Vector.set b i bv ;
    (* Seed the outputs with the WRONG answer for the NaN rows, so a kernel
       that never wrote them cannot read as a pass. *)
    Vector.set ne i 7l ;
    Vector.set eq i 7l
  done ;
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Execute.Vec a;
        Execute.Vec b;
        Execute.Vec ne;
        Execute.Vec eq;
        Execute.Int n;
      ]
    ~block:(Execute.dims1d n)
    ~grid:(Execute.dims1d 1)
    () ;
  Transfer.flush dev ;
  (Vector.to_array ne, Vector.to_array eq)

let run_f64 (dev : Device.t) ir =
  let a = Vector.create Vector.float64 n in
  let b = Vector.create Vector.float64 n in
  let ne = Vector.create Vector.int32 n in
  let eq = Vector.create Vector.int32 n in
  for i = 0 to n - 1 do
    let av, bv = cases.(i) in
    Vector.set a i av ;
    Vector.set b i bv ;
    Vector.set ne i 7l ;
    Vector.set eq i 7l
  done ;
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Execute.Vec a;
        Execute.Vec b;
        Execute.Vec ne;
        Execute.Vec eq;
        Execute.Int n;
      ]
    ~block:(Execute.dims1d n)
    ~grid:(Execute.dims1d 1)
    () ;
  Transfer.flush dev ;
  (Vector.to_array ne, Vector.to_array eq)

(* Pure-OCaml reference: [<>] / [=] on the same pair. This is the semantics
   the C-family backends, the native path and the interpreter all share. *)
let expected_ne i =
  let av, bv = cases.(i) in
  if av <> bv then 1l else 0l

let expected_eq i =
  let av, bv = cases.(i) in
  if av = bv then 1l else 0l

let verify label got_ne got_eq =
  let bad = ref 0 in
  for i = 0 to n - 1 do
    let av, bv = cases.(i) in
    let ene = expected_ne i and eeq = expected_eq i in
    if got_ne.(i) <> ene || got_eq.(i) <> eeq then begin
      if !bad < 6 then
        Printf.printf
          "    %s mismatch @%d: (%h <> %h) got ne=%ld eq=%ld exp ne=%ld eq=%ld\n\
           %!"
          label
          i
          av
          bv
          got_ne.(i)
          got_eq.(i)
          ene
          eeq ;
      incr bad
    end
  done ;
  !bad

let is_native (dev : Device.t) =
  dev.Device.framework = "Native" || dev.Device.framework = "Interpreter"

(* Backends whose float [<>] is claimed to be C's [!=]: the PTX emitter (the
   fix under test), the interpreter and the native oracle. A wrong result from
   any of these is a hard failure. Other GPU backends emit [!=] in source
   form, but the driver/compiler is free to apply fast-math style NaN
   assumptions we do not control here, so they are reported, not gated. *)
let is_gated (dev : Device.t) =
  is_native dev || dev.Device.framework = "CUDA/PTX"

let ir_of kern name =
  let _, kirc = kern in
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith (name ^ " kernel has no IR")

let () =
  let ir32 = ir_of float_ne_kernel_f32 "float32 <>" in
  let ir64 = ir_of float_ne_kernel_f64 "float64 <>" in
  let devs = Device.init () in
  print_endline "=== float <> against NaN, E2E (H2 comparison family) ===" ;
  if Array.length devs = 0 then begin
    print_endline "test_ptx_float_ne_nan: FAILED - no devices found" ;
    exit 1
  end ;
  let native_ran = ref false in
  let failed = ref false in
  Array.iter
    (fun (dev : Device.t) ->
      let native = is_native dev in
      let gated = is_gated dev in
      List.iter
        (fun (label, runner, ir) ->
          try
            let got_ne, got_eq = runner dev ir in
            let bad = verify label got_ne got_eq in
            Printf.printf
              "  %-11s %-34s %-4s %s (%d/%d ok)%s\n%!"
              dev.Device.framework
              dev.Device.name
              label
              (if bad = 0 then "PASS" else if gated then "FAIL" else "DIVERGES")
              (n - bad)
              n
              (if bad <> 0 && not gated then " [ungated backend, reported only]"
               else "") ;
            if bad <> 0 && gated then failed := true ;
            if native then native_ran := true
          with e ->
            Printf.printf
              "  %-11s %-34s %-4s %s (%s)\n%!"
              dev.Device.framework
              dev.Device.name
              label
              (if native then "ERROR" else "SKIP (backend could not launch)")
              (Printexc.to_string e) ;
            (* Native/Interpreter must always run. A GPU backend that cannot
               even launch is only reported; a wrong RESULT from a gated
               backend that DID run is always a hard failure. *)
            if native then failed := true)
        [("f32", run_f32, ir32); ("f64", run_f64, ir64)])
    devs ;
  if not !native_ran then begin
    print_endline
      "test_ptx_float_ne_nan: FAILED - no native/interpreter device ran" ;
    exit 1
  end ;
  if !failed then begin
    print_endline "test_ptx_float_ne_nan: FAILED" ;
    exit 1
  end ;
  print_endline "test_ptx_float_ne_nan: PASSED"
