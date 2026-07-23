(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for L17b: polymorphic bare float literals.
 *
 * This kernel mixes bare, UNSUFFIXED float literals (`2.0`, `1.0`, `0.0`) into
 * a float64 computation. Under the old rule a bare literal was hard-typed
 * float32 and `f64_value *. 2.0` failed to unify (see the note in
 * test_bare_float_kernel_arith.ml / test_float64_kernel_arith.ml). With L17b a
 * bare literal is a fresh tvar that unifies with its context, so here every
 * literal infers float64 from the surrounding f64 arithmetic - no `G` suffix
 * anywhere in the body.
 *
 * Two load-bearing halves:
 * (i)  the lowered IR uses float64 (kernel_uses_float64 = true) - proves the
 *      bare literals were inferred as f64 and did NOT default to f32;
 * (ii) the kernel executes and matches a pure-OCaml binary64 reference on
 *      every available device (native/interpreter gate the test; GPU devices
 *      with fp64 support are compared within tolerance and reported).
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Benchmarks = Test_helpers.Benchmarks

let () = Benchmarks.init ()

(* Bare literals only (0.0, 2.0, 1.0) - each infers float64 from the f64 vector
   context. `x` starts life as a bare-literal `mut 0.0` whose tvar is pinned to
   float64 by the `inp.(tid) *. 2.0` assignment. *)
let bare_f64_kernel =
  [%kernel
    fun (out : float64 vector) (inp : float64 vector) (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then begin
        let x = mut 0.0 in
        x := inp.(tid) *. 2.0 ;
        out.(tid) <- x +. 1.0
      end]

let n = 64

let inp_of i = (float_of_int i *. 0.5) -. 8.0

let ocaml_reference v = (v *. 2.0) +. 1.0

let run_on_device (dev : Device.t) ir =
  let out = Vector.create Vector.float64 n in
  let inp = Vector.create Vector.float64 n in
  for i = 0 to n - 1 do
    Vector.set inp i (inp_of i) ;
    Vector.set out i 0.0
  done ;
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Execute.Vec out; Execute.Vec inp; Execute.Int n]
    ~block:(Execute.dims1d n)
    ~grid:(Execute.dims1d 1)
    () ;
  Transfer.flush dev ;
  Vector.to_array out

let verify ~exact got =
  let tol = if exact then 0.0 else 1e-9 in
  let bad = ref 0 in
  for i = 0 to n - 1 do
    let expected = ocaml_reference (inp_of i) in
    if Stdlib.abs_float (got.(i) -. expected) > tol then begin
      if !bad < 5 then
        Printf.printf
          "    mismatch @%d: got=%.15g ref=%.15g\n%!"
          i
          got.(i)
          expected ;
      incr bad
    end
  done ;
  !bad

let is_native (dev : Device.t) =
  dev.Device.framework = "Native" || dev.Device.framework = "Interpreter"

let () =
  let _, kirc = bare_f64_kernel in
  let ir =
    match kirc.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "bare-f64 kernel has no IR"
  in
  (* (i) The bare literals must have been inferred float64. If they had defaulted
     to float32 the whole kernel would collapse to the f32 path. *)
  if not (Sarek_ir_analysis.kernel_uses_float64 ir) then begin
    print_endline
      "test_bare_float_infers_f64: FAILED - kernel_uses_float64 = false (bare \
       literals did not infer float64 from context)" ;
    exit 1
  end ;
  let devs = Device.init () in
  Printf.printf "=== bare float literals inferred as float64 ===\n%!" ;
  let native_ok = ref false in
  let native_failed = ref false in
  Array.iter
    (fun (dev : Device.t) ->
      let native = is_native dev in
      if native || Device.allows_fp64 dev then begin
        try
          let got = run_on_device dev ir in
          let bad = verify ~exact:native got in
          Printf.printf
            "  %-10s %-24s %s (%d/%d ok)\n%!"
            dev.Device.framework
            dev.Device.name
            (if bad = 0 then "PASS" else "FAIL")
            (n - bad)
            n ;
          if native then begin
            native_ok := true ;
            if bad <> 0 then native_failed := true
          end
        with e ->
          Printf.printf
            "  %-10s %-24s ERROR (%s)\n%!"
            dev.Device.framework
            dev.Device.name
            (Printexc.to_string e) ;
          if native then native_failed := true
      end)
    devs ;
  if not !native_ok then begin
    print_endline
      "test_bare_float_infers_f64: FAILED - no native/interpreter device ran \
       the kernel" ;
    exit 1
  end ;
  if !native_failed then begin
    print_endline "test_bare_float_infers_f64: FAILED" ;
    exit 1
  end ;
  print_endline "test_bare_float_infers_f64: PASSED"
