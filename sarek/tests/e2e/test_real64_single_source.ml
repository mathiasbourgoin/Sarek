(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for SINGLE-SOURCE Sarek_real64 (palier B).
 *
 * Palier A authored a real64 kernel as a hand-written PAIR: an f64 body over
 * `float64 vector` and a df64 body over `Sarek_df64.df64 vector`, kept in sync
 * by the author. This test proves palier B: the compute is written ONCE
 * against an abstract `real64 vector` element type with the intersection op
 * set (+. -. *. /. and sqrt), and [%kernel.real64] expands it to BOTH lowered
 * variants automatically. Real64.kernel_ir then picks the IR matching the
 * device substrate - exactly the plumbing palier A used.
 *
 * Coverage mirrors test_real64.ml op-for-op: on EVERY available device it runs
 * two passes -
 *   1. the device's DEFAULT substrate (native f64 where supported, else df64);
 *   2. the df64 fallback FORCED, so the emulation lowering runs even on fp64
 *      hardware -
 * and checks elementwise add / sub / mul / div / sqrt against an OCaml binary64
 * reference computed from the exact values the device saw. Tolerances follow
 * the substrate's honest contract (identical to test_real64's [tol_for]).
 ******************************************************************************)

[@@@warning "-32-33-34-69"]

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Execute = Sarek.Execute
module Real64 = Sarek_real64

type float32 = float

type ('a, 'b) vector = ('a, 'b) Vector.t

let () = Test_helpers.Benchmarks.init ()

(* df64 [@sarek.module] ops for the fallback lowering of the single source. *)
let%sarek_include _ = "../../Sarek_df64/Sarek_df64.ml"

(* ========== ONE kernel body, authored against abstract real64 ============== *)

(* Written once over `real64 vector` with the intersection op set. The PPX
   expands this SAME AST twice: once as native IEEE-754 float64 (operators
   unchanged, sqrt -> Float64.sqrt) and once as the df64 double-float fallback
   (+. -> df64_add, sqrt -> df64_sqrt, element type -> Sarek_df64.df64). *)
let real64_ops_kernel =
  [%kernel.real64
    fun (a : real64 vector)
        (b : real64 vector)
        (c : real64 vector)
        (add_o : real64 vector)
        (sub_o : real64 vector)
        (mul_o : real64 vector)
        (div_o : real64 vector)
        (sqrt_o : real64 vector)
        (nn : int32) ->
      let tid = thread_idx_x + (block_idx_x * block_dim_x) in
      if tid < nn then begin
        let x = a.(tid) in
        let y = b.(tid) in
        add_o.(tid) <- x +. y ;
        sub_o.(tid) <- x -. y ;
        mul_o.(tid) <- x *. y ;
        div_o.(tid) <- x /. y ;
        sqrt_o.(tid) <- sqrt c.(tid)
      end]

(* ========== Inputs and reference (identical to test_real64.ml) ============= *)

let n = 4096

let inputs =
  Random.init 0x5ea1 ;
  let logu emin emax =
    let e = emin +. Random.float (emax -. emin) in
    let m = 1.0 +. Random.float 9.0 in
    let s = if Random.bool () then 1.0 else -1.0 in
    s *. m *. (10.0 ** e)
  in
  let a = Array.init n (fun _ -> logu (-6.0) 6.0) in
  let b = Array.init n (fun _ -> logu (-3.0) 3.0) in
  (a, b)

let rel_error got expected =
  if expected = 0.0 then Stdlib.abs_float got
  else Stdlib.abs_float ((got -. expected) /. expected)

let f32_tol = 0x1p-22

let tol_for ~(substrate : Real64.substrate) ~framework ~op =
  match substrate with
  | Real64.Native_f64 -> 1e-12
  | Real64.Fallback_df64 -> (
      let base = match op with "add" | "sub" -> 0x1p-47 | _ -> 0x1p-46 in
      match (framework, op) with
      | "Native", _ -> f32_tol
      | "Vulkan", ("mul" | "div") -> f32_tol
      | _ -> base)

(* ========== One pass on one device with one substrate ===================== *)

let failures = ref 0

let run_pass (dev : Device.t) ~(substrate : Real64.substrate) =
  let a_data, b_data = inputs in
  let mk () = Real64.create_vector substrate n in
  let a = mk () and b = mk () and c = mk () in
  let add_o = mk () and sub_o = mk () and mul_o = mk () in
  let div_o = mk () and sqrt_o = mk () in
  for i = 0 to n - 1 do
    Real64.vset a i a_data.(i) ;
    Real64.vset b i b_data.(i) ;
    (* Dedicated nonnegative input for sqrt (avoids an abs intrinsic, which
       does not lower on GLSL). *)
    Real64.vset c i (Stdlib.abs_float a_data.(i))
  done ;
  (* The whole point of palier B: ONE authored kernel, IR picked per device. *)
  let ir = Real64.kernel_ir substrate real64_ops_kernel in
  let block = Execute.dims1d 256 in
  let grid = Execute.dims1d ((n + 255) / 256) in
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Real64.arg_of a;
        Real64.arg_of b;
        Real64.arg_of c;
        Real64.arg_of add_o;
        Real64.arg_of sub_o;
        Real64.arg_of mul_o;
        Real64.arg_of div_o;
        Real64.arg_of sqrt_o;
        Execute.Int n;
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let framework = dev.Device.framework in
  let worst = Hashtbl.create 8 in
  let check op got expected =
    let err =
      if Float.is_nan got then Float.infinity else rel_error got expected
    in
    let prev = try Hashtbl.find worst op with Not_found -> 0.0 in
    if err > prev then Hashtbl.replace worst op err
  in
  for i = 0 to n - 1 do
    let x = Real64.vget a i and y = Real64.vget b i in
    let cx = Real64.vget c i in
    check "add" (Real64.vget add_o i) (x +. y) ;
    check "sub" (Real64.vget sub_o i) (x -. y) ;
    check "mul" (Real64.vget mul_o i) (x *. y) ;
    check "div" (Real64.vget div_o i) (x /. y) ;
    check "sqrt" (Real64.vget sqrt_o i) (Stdlib.sqrt cx)
  done ;
  Printf.printf
    "  [%s / %s]\n%!"
    framework
    (Real64.string_of_substrate substrate) ;
  List.iter
    (fun op ->
      let err = try Hashtbl.find worst op with Not_found -> 0.0 in
      let tol = tol_for ~substrate ~framework ~op in
      let ok = err <= tol in
      if not ok then incr failures ;
      Printf.printf
        "    %-5s max rel err %.3g (tol %.3g) %s\n%!"
        op
        err
        tol
        (if ok then "PASS" else "FAIL"))
    ["add"; "sub"; "mul"; "div"; "sqrt"]

(* ========== Main ========== *)

let () =
  let devs = Device.init () in
  if Array.length devs = 0 then begin
    print_endline "No devices found - SKIPPED" ;
    exit 0
  end ;
  Array.iter
    (fun (dev : Device.t) ->
      Printf.printf
        "Device: %s [%s]  fp64=%b\n%!"
        dev.Device.name
        dev.Device.framework
        (Device.allows_fp64 dev) ;
      run_pass dev ~substrate:(Real64.substrate_for dev) ;
      run_pass
        dev
        ~substrate:(Real64.substrate_for ~force:Real64.Fallback_df64 dev))
    devs ;
  if !failures = 0 then print_endline "test_real64_single_source PASSED"
  else begin
    Printf.printf "test_real64_single_source FAILED (%d failures)\n" !failures ;
    exit 1
  end
