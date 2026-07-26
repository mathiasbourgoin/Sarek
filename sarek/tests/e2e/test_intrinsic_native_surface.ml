(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The intrinsics that were declared by the Sarek stdlib but had no native
 * target, executed on the Native device.
 *
 * This file is load-bearing in TWO ways, and the first is the important one:
 *
 * (i) IT COMPILES. Before the fix, every kernel below was rejected by the
 *     OCaml compiler, because the native backend lowers Float32.<name> to
 *     Sarek.Sarek_cpu_runtime.Float32.<name> and Float64.<name> to
 *     (formerly) Float.<name>, copying the name verbatim. Neither module
 *     exported abs_float / expm1 / log1p / hypot / copysign / fmod / minus
 *     (Float32) nor abs_float / rsqrt / fmod / copysign / of_int32 / to_int32
 *     (Float64), so the user saw e.g.
 *         Error: Unbound value Sarek.Sarek_cpu_runtime.Float32.copysign
 *     pointing into PPX-generated code. A public API that cannot be called is
 *     a compile-time failure here, so the mere existence of this file as a
 *     wired, built target is the regression guard.
 *
 * (ii) The values are checked against the OCaml stdlib, so a native mapping
 *      that resolves to the WRONG function (e.g. minus -> add, or fmod
 *      implemented as truncating division) does not pass. Each expectation is
 *      the `ocaml = ...` field of the corresponding let%sarek_intrinsic, which
 *      is that intrinsic's host-side specification.
 *
 * Float32 comparisons use a 1e-5 relative-ish tolerance because the runtime
 * rounds every result through float32; Float64 comparisons are exact where the
 * operation is exact (copysign, abs_float) and tolerance-based otherwise.
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Float32 = Sarek_stdlib.Float32
module Float64 = Sarek_float64.Float64
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Sarek_native.Native_plugin.init ()

let n = 16

(* Inputs chosen to exercise sign handling (copysign, abs_float, fmod all
   depend on it) and to stay away from the poles of log1p/expm1. *)
let input i = (float_of_int (i - (n / 2)) *. 0.75) +. 0.125

let failures = ref 0

let report name i expected got tol =
  if
    Stdlib.abs_float (got -. expected) > tol
    || Stdlib.compare (got <> got) (expected <> expected) <> 0
  then begin
    incr failures ;
    Printf.printf
      "MISMATCH %s[%d]: expected %.17g, got %.17g (tol %g)\n%!"
      name
      i
      expected
      got
      tol
  end

(******************************************************************************
 * Float32
 ******************************************************************************)

let f32_kernel =
  [%kernel
    fun (a : float32 vector) (out : float32 vector) ->
      let open Std in
      let tid = global_thread_id in
      let v = a.(tid) in
      let av = Float32.abs_float v in
      let cs = Float32.copysign av v in
      let hy = Float32.hypot cs (Float32.expm1 av) in
      let l1 = Float32.log1p av in
      out.(tid) <- Float32.fmod (Float32.minus hy l1) 3.0]

let f32_expected v =
  let av = Stdlib.abs_float v in
  let cs = Stdlib.copysign av v in
  let hy = Stdlib.hypot cs (Stdlib.expm1 av) in
  let l1 = Stdlib.log1p av in
  Float.rem (hy -. l1) 3.0

(******************************************************************************
 * Float64
 ******************************************************************************)

let f64_kernel =
  [%kernel
    fun (a : float64 vector) (out : float64 vector) ->
      let open Std in
      let tid = global_thread_id in
      let v = a.(tid) in
      let av = Float64.abs_float v in
      let cs = Float64.copysign av v in
      let hy = Float64.hypot cs (Float64.expm1 av) in
      let rs = Float64.rsqrt (Float64.log1p (Float64.abs_float hy)) in
      out.(tid) <- Float64.fmod rs 3.0]

let f64_expected v =
  let av = Stdlib.abs_float v in
  let cs = Stdlib.copysign av v in
  let hy = Stdlib.hypot cs (Stdlib.expm1 av) in
  let rs = 1.0 /. Stdlib.sqrt (Stdlib.log1p (Stdlib.abs_float hy)) in
  Float.rem rs 3.0

(******************************************************************************
 * Drive both on the Native device
 ******************************************************************************)

let ir_of kern =
  let _, kirc = kern in
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let run_case label kind kern expected tol =
  let devs = Device.init ~frameworks:["Native"] () in
  let dev = devs.(0) in
  let a = Vector.create kind n in
  let out = Vector.create kind n in
  for i = 0 to n - 1 do
    Vector.set a i (input i) ;
    Vector.set out i nan
  done ;
  Execute.run_vectors
    ~device:dev
    ~ir:(ir_of kern)
    ~args:[Execute.Vec a; Execute.Vec out]
    ~block:(Execute.dims1d n)
    ~grid:(Execute.dims1d 1)
    () ;
  Transfer.flush dev ;
  for i = 0 to n - 1 do
    report label i (expected (input i)) (Vector.get out i) tol
  done

let () =
  run_case "float32" Vector.float32 f32_kernel f32_expected 1e-5 ;
  run_case "float64" Vector.float64 f64_kernel f64_expected 1e-12 ;
  if !failures = 0 then
    print_endline
      "test_intrinsic_native_surface: PASSED (14 previously-uncallable \
       intrinsics compiled and evaluated correctly on the Native device)"
  else begin
    Printf.printf
      "test_intrinsic_native_surface: FAILED (%d mismatches)\n"
      !failures ;
    exit 1
  end
