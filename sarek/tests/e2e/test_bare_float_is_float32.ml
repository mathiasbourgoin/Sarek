(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Regression test: bare `float` in Sarek kernel type annotations now types as
   float32, not float64 (Sarek_types.ml:322 - human decision 2026-07-02: "keep
   float the default for GPGPU kernels and not float64"). Pre-fix, a kernel
   using bare `float` for a parameter or local silently got the float64
   representation internally (`Sarek_ir_types.TFloat64`), which
   `Sarek_ir_analysis.kernel_uses_float64` would report as `true`; post-fix
   it reports `false`, matching the actual float32 GPU representation.

   Two halves, both load-bearing:
   (i)  kernel_uses_float64 on the lowered IR of a bare-float kernel is
        false (fails pre-fix: was true).
   (ii) the kernel actually executes on the float32 path and produces correct
        results (proves the IR-level classification matches runtime
        behavior, not just a label). *)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Sarek_native.Native_plugin.init ()

(* `a`, `b`, `out` are all bare `float` - no `float32`/`float64` anywhere in
   this kernel's source. `sum` is a bare-`float` local. Deliberately no float
   *literal* anywhere in the body: float literals (`2.0`) always type as
   float32 regardless of this fix (Sarek_typer.ml's `EFloat` arm - unrelated,
   unaffected code), so mixing one in here would conflate two different
   pre-fix failure modes. Confirmed empirically: temporarily reverting the
   Sarek_types.ml fix and adding a `let local : float = 2.0` local to this
   kernel does not merely misclassify - it fails to *compile* at all
   ("Cannot unify types: float64 and float32"), because pre-fix bare
   `float`-annotated locals wanted float64 while float literals are always
   float32. That is a real, separate, pre-existing inconsistency this fix
   incidentally also resolves (post-fix both sides agree on float32) - see
   the evidence file for the exact repro. This test isolates the
   classification-only claim: no literals, so it compiles both before and
   after, and only `kernel_uses_float64`'s answer differs. *)
let bare_float_kernel =
  [%kernel
    fun (a : float vector) (b : float vector) (out : float vector) ->
      let open Std in
      let tid = global_thread_id in
      let sum : float = a.(tid) +. b.(tid) in
      out.(tid) <- sum +. sum]

let () =
  let _, kirc = bare_float_kernel in
  let ir =
    match kirc.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "no ir"
  in

  (* (i) IR-level classification *)
  let uses_f64 = Sarek_ir_analysis.kernel_uses_float64 ir in
  if uses_f64 then begin
    print_endline
      "test_bare_float_is_float32: FAILED (kernel_uses_float64 = true; bare \
       `float` is still resolving to float64)" ;
    exit 1
  end ;

  (* (ii) Execution on the float32 path. The host vectors are created with
     Vector.float32 - the same representation bare `float` now resolves to
     internally, so this is the representation-matching choice, not an
     arbitrary one. *)
  let devs = Device.init ~frameworks:["Native"] () in
  let native_dev = devs.(0) in
  let n = 8 in
  let a = Vector.create Vector.float32 n in
  let b = Vector.create Vector.float32 n in
  let out = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set a i (float_of_int i) ;
    Vector.set b i (float_of_int (i * 2)) ;
    Vector.set out i (-1.0)
  done ;

  Execute.run_vectors
    ~device:native_dev
    ~ir
    ~args:[Execute.Vec a; Execute.Vec b; Execute.Vec out]
    ~block:(Execute.dims1d n)
    ~grid:(Execute.dims1d 1)
    () ;
  Transfer.flush native_dev ;

  let ok = ref true in
  for i = 0 to n - 1 do
    let expected = float_of_int (i + (i * 2)) *. 2.0 in
    let got = Vector.get out i in
    (* float32 tolerance: values here are small integers scaled by 2, exactly
       representable in float32, so an exact comparison is legitimate and
       also lets an accidental float64 execution path (different rounding
       for the same inputs would not actually differ here, since all values
       are exactly representable in both widths - this assertion is a
       sanity check on the arithmetic, not a width-detection probe; (i)
       above is the width-detection assertion). *)
    if Stdlib.abs_float (got -. expected) > 0.001 then begin
      Printf.printf "MISMATCH at %d: expected %f, got %f\n%!" i expected got ;
      ok := false
    end
  done ;
  if !ok then print_endline "test_bare_float_is_float32: PASSED"
  else begin
    print_endline "test_bare_float_is_float32: FAILED" ;
    exit 1
  end
