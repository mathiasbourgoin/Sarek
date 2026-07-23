(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek_real64 - portable "~double precision on every device".
 *
 * real64 selects, per device, between two concrete lowerings:
 *   - native IEEE-754 binary64 (Sarek_float64) on fp64-capable devices, and
 *   - the df64 double-float fallback (Sarek_df64) on devices without fp64.
 *
 * Because a single [%kernel] lowers to ONE concrete element type, the compute
 * is authored as a PAIR - a float64 body and a df64 body - and Sarek_real64
 * picks the matching IR at launch (Real64.select) and materialises the
 * device-appropriate vector storage (Real64.create_vector). To the test
 * driver both paths look identical: fill/read plain doubles, launch, compare.
 *
 * On EVERY available device this runs TWO passes:
 *   1. the device's DEFAULT substrate (native f64 where supported, else df64);
 *   2. the df64 fallback FORCED (Real64.substrate_for ~force:Fallback_df64),
 *      so the emulation lowering path is exercised even on fp64 hardware.
 *
 * Each pass checks elementwise add / sub / mul / div / sqrt against an OCaml
 * binary64 reference computed from the exact values the device saw (the df64
 * path stores inputs at ~48-bit precision, so the oracle reads them back).
 * Tolerances follow the substrate's honest contract (see [tol_for]).
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

(* df64 [@sarek.module] ops for the fallback kernel body. *)
let%sarek_include _ = "../../Sarek_df64/Sarek_df64.ml"

(* ========== The two lowered kernel bodies (one algorithm, two substrates) == *)

(* Native f64 body: real IEEE-754 binary64. Float64 ops + `G`-suffix literals. *)
let f64_ops_kernel =
  [%kernel
    fun (a : float64 vector)
        (b : float64 vector)
        (c : float64 vector)
        (add_o : float64 vector)
        (sub_o : float64 vector)
        (mul_o : float64 vector)
        (div_o : float64 vector)
        (sqrt_o : float64 vector)
        (nn : int32) ->
      let open Sarek_float64 in
      let tid = thread_idx_x + (block_idx_x * block_dim_x) in
      if tid < nn then begin
        let x = a.(tid) in
        let y = b.(tid) in
        add_o.(tid) <- x +. y ;
        sub_o.(tid) <- x -. y ;
        mul_o.(tid) <- x *. y ;
        div_o.(tid) <- x /. y ;
        sqrt_o.(tid) <- Float64.sqrt c.(tid)
      end]

(* df64 fallback body: double-float over pairs of float32. Same algorithm. *)
let df64_ops_kernel =
  [%kernel
    fun (a : Sarek_df64.df64 vector)
        (b : Sarek_df64.df64 vector)
        (c : Sarek_df64.df64 vector)
        (add_o : Sarek_df64.df64 vector)
        (sub_o : Sarek_df64.df64 vector)
        (mul_o : Sarek_df64.df64 vector)
        (div_o : Sarek_df64.df64 vector)
        (sqrt_o : Sarek_df64.df64 vector)
        (nn : int32) ->
      let open Sarek_df64 in
      let tid = thread_idx_x + (block_idx_x * block_dim_x) in
      if tid < nn then begin
        let x = a.(tid) in
        let y = b.(tid) in
        add_o.(tid) <- df64_add x y ;
        sub_o.(tid) <- df64_sub x y ;
        mul_o.(tid) <- df64_mul x y ;
        div_o.(tid) <- df64_div x y ;
        sqrt_o.(tid) <- df64_sqrt c.(tid)
      end]

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

(* ========== Inputs and reference ========== *)

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

(* Per-op relative tolerance for a (substrate, framework). Native f64 gets the
   full binary64 contract; df64 follows Sarek_df64's per-backend table (Native
   and Vulkan mul/div collapse to f32 storage precision - a documented,
   inherited deviation, not a silent widening). *)
let f32_tol = 0x1p-22

let tol_for ~(substrate : Real64.substrate) ~framework ~op =
  match substrate with
  | Real64.Native_f64 -> 1e-12
  | Real64.Fallback_df64 -> (
      let base =
        match op with
        | "add" | "sub" -> 0x1p-47
        | _ -> 0x1p-46 (* mul / div / sqrt *)
      in
      match (framework, op) with
      | "Native", _ -> f32_tol
      | "Vulkan", ("mul" | "div") -> f32_tol
      | _ -> base)

(* ========== One pass on one device with one substrate ========== *)

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
  let ir =
    Real64.select
      substrate
      ~native:(ir_of f64_ops_kernel)
      ~fallback:(ir_of df64_ops_kernel)
  in
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
    (* Reference from the values the device actually saw (df64 stores inputs
       at ~48-bit precision; vget decodes them exactly). *)
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
      (* Pass 1: the device's default substrate. *)
      run_pass dev ~substrate:(Real64.substrate_for dev) ;
      (* Pass 2: force the df64 fallback, so the emulation lowering path runs
         on EVERY device including fp64-capable ones. *)
      run_pass
        dev
        ~substrate:(Real64.substrate_for ~force:Real64.Fallback_df64 dev))
    devs ;
  if !failures = 0 then print_endline "test_real64 PASSED"
  else begin
    Printf.printf "test_real64 FAILED (%d failures)\n" !failures ;
    exit 1
  end
