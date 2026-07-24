(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek Float64 math intrinsics, on every device that supports
 * fp64.
 *
 * Builds Sarek IR kernels directly (no PPX) using
 * EIntrinsic (["Float64"], name, ...) for every unary/binary function exposed
 * by Sarek_float64.Float64, and runs each one on every device that reports
 * fp64 support (Device.allows_fp64). This is the regression coverage for the
 * pure-registry GLSL/Metal renaming fix: rsqrt, abs_float, atan2, hypot,
 * expm1, and log1p previously reached codegen with names those backends
 * don't define (see spoc/ir/Sarek_pure_registry.ml).
 *
 * Per the request that drove this test: a single (device, function) mismatch
 * must NOT fail the whole suite or block CI. Every combination is run in
 * isolation, failures and errors are collected into a report, and the
 * process always exits 0. The report is the deliverable, not a pass/fail
 * gate — read it to see which backend/function pairs are broken.
 *
 * Run with: dune exec sarek/tests/e2e/test_float64_math_intrinsics.exe
 ******************************************************************************)

open Sarek_ir_types
open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* Force backend registration *)
let () = Test_helpers.Benchmarks.init_backends ()

let n = 256

(** {1 IR builders}

    Path-qualified Float64 intrinsics resolve through [Sarek_pure_registry],
    bypassing PPX entirely — the same shape as [test_float32_sin_pure.ml]'s
    Float32.sin kernel, generalized to arity 1 and 2. *)

let make_unary_ir name : kernel =
  let a =
    {var_name = "a"; var_id = 0; var_type = TVec TFloat64; var_mutable = false}
  in
  let b =
    {var_name = "b"; var_id = 1; var_type = TVec TFloat64; var_mutable = false}
  in
  let idx =
    {var_name = "idx"; var_id = 2; var_type = TInt32; var_mutable = false}
  in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("b", EVar idx),
            EIntrinsic (["Float64"], name, [EArrayRead ("a", EVar idx)]) ) )
  in
  {
    kern_name = "float64_" ^ name ^ "_unary";
    kern_params =
      [
        DParam (a, Some {arr_elttype = TFloat64; arr_memspace = Global});
        DParam (b, Some {arr_elttype = TFloat64; arr_memspace = Global});
      ];
    kern_locals = [];
    kern_body = body;
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

let make_binary_ir name : kernel =
  let a =
    {var_name = "a"; var_id = 0; var_type = TVec TFloat64; var_mutable = false}
  in
  let b =
    {var_name = "b"; var_id = 1; var_type = TVec TFloat64; var_mutable = false}
  in
  let c =
    {var_name = "c"; var_id = 2; var_type = TVec TFloat64; var_mutable = false}
  in
  let idx =
    {var_name = "idx"; var_id = 3; var_type = TInt32; var_mutable = false}
  in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("c", EVar idx),
            EIntrinsic
              ( ["Float64"],
                name,
                [EArrayRead ("a", EVar idx); EArrayRead ("b", EVar idx)] ) ) )
  in
  {
    kern_name = "float64_" ^ name ^ "_binary";
    kern_params =
      [
        DParam (a, Some {arr_elttype = TFloat64; arr_memspace = Global});
        DParam (b, Some {arr_elttype = TFloat64; arr_memspace = Global});
        DParam (c, Some {arr_elttype = TFloat64; arr_memspace = Global});
      ];
    kern_locals = [];
    kern_body = body;
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

(** {1 Function specs}

    [gen] produces domain-safe inputs (e.g. positive for [log], bounded for
    [asin]/[acos]) so a mismatch reflects a codegen bug, not a math-domain error
    shared with the OCaml reference. *)

type unary_spec = {
  u_name : string;
  u_ocaml : float -> float;
  u_gen : int -> float;
  u_tol : float;
}

type binary_spec = {
  b_name : string;
  b_ocaml : float -> float -> float;
  b_gen : int -> float * float;
  b_tol : float;
}

let bounded lo hi i = lo +. ((hi -. lo) *. (float_of_int (i mod 100) /. 100.0))

let unary_specs =
  [
    {
      u_name = "sin";
      u_ocaml = Stdlib.sin;
      u_gen = bounded (-3.0) 3.0;
      u_tol = 1e-9;
    };
    {
      u_name = "cos";
      u_ocaml = Stdlib.cos;
      u_gen = bounded (-3.0) 3.0;
      u_tol = 1e-9;
    };
    {
      u_name = "tan";
      u_ocaml = Stdlib.tan;
      u_gen = bounded (-1.0) 1.0;
      u_tol = 1e-9;
    };
    {
      u_name = "asin";
      u_ocaml = Stdlib.asin;
      u_gen = bounded (-0.95) 0.95;
      u_tol = 1e-9;
    };
    {
      u_name = "acos";
      u_ocaml = Stdlib.acos;
      u_gen = bounded (-0.95) 0.95;
      u_tol = 1e-9;
    };
    {
      u_name = "atan";
      u_ocaml = Stdlib.atan;
      u_gen = bounded (-3.0) 3.0;
      u_tol = 1e-9;
    };
    {
      u_name = "sinh";
      u_ocaml = Stdlib.sinh;
      u_gen = bounded (-2.0) 2.0;
      u_tol = 1e-8;
    };
    {
      u_name = "cosh";
      u_ocaml = Stdlib.cosh;
      u_gen = bounded (-2.0) 2.0;
      u_tol = 1e-8;
    };
    {
      u_name = "tanh";
      u_ocaml = Stdlib.tanh;
      u_gen = bounded (-2.0) 2.0;
      u_tol = 1e-9;
    };
    {
      u_name = "exp";
      u_ocaml = Stdlib.exp;
      u_gen = bounded (-2.0) 2.0;
      u_tol = 1e-8;
    };
    {
      u_name = "expm1";
      u_ocaml = Stdlib.expm1;
      u_gen = bounded (-2.0) 2.0;
      u_tol = 1e-8;
    };
    {
      u_name = "log";
      u_ocaml = Stdlib.log;
      u_gen = bounded 0.01 5.0;
      u_tol = 1e-9;
    };
    {
      u_name = "log10";
      u_ocaml = Stdlib.log10;
      u_gen = bounded 0.01 5.0;
      u_tol = 1e-9;
    };
    {
      u_name = "log1p";
      u_ocaml = Stdlib.log1p;
      u_gen = bounded (-0.9) 5.0;
      u_tol = 1e-9;
    };
    {
      u_name = "sqrt";
      u_ocaml = Stdlib.sqrt;
      u_gen = bounded 0.01 10.0;
      u_tol = 1e-9;
    };
    {
      u_name = "rsqrt";
      u_ocaml = (fun x -> 1.0 /. Stdlib.sqrt x);
      u_gen = bounded 0.1 10.0;
      u_tol = 1e-6;
    };
    {
      u_name = "ceil";
      u_ocaml = Stdlib.ceil;
      u_gen = bounded (-5.0) 5.0;
      u_tol = 0.0;
    };
    {
      u_name = "floor";
      u_ocaml = Stdlib.floor;
      u_gen = bounded (-5.0) 5.0;
      u_tol = 0.0;
    };
    {
      u_name = "abs_float";
      u_ocaml = Stdlib.abs_float;
      u_gen = bounded (-5.0) 5.0;
      u_tol = 0.0;
    };
    (* exp2/log2/cbrt have no dedicated Float64 device builtin and no software
       helper of their own: the GLSL backend composes them over exp/log/pow
       (2^x = exp(x·ln2), log2 x = log x·log2e, cbrt x = sign x·pow(|x|,1/3)).
       These rows give the composition numeric coverage on real fp64 devices.
       The CPU interpreter has no eval arm for them, so it reports SKIP there —
       expected, and harmless to this report-only test. *)
    {
      u_name = "exp2";
      u_ocaml = (fun x -> Float.pow 2.0 x);
      u_gen = bounded (-2.0) 2.0;
      u_tol = 1e-8;
    };
    {
      u_name = "log2";
      u_ocaml = Float.log2;
      u_gen = bounded 0.01 5.0;
      u_tol = 1e-9;
    };
    {
      u_name = "cbrt";
      u_ocaml = Float.cbrt;
      u_gen = bounded (-5.0) 5.0;
      u_tol = 1e-9;
    };
  ]

let binary_specs =
  [
    {
      b_name = "pow";
      b_ocaml = Float.pow;
      b_gen = (fun i -> (bounded 0.1 3.0 i, bounded (-2.0) 2.0 (i + 7)));
      b_tol = 1e-6;
    };
    {
      b_name = "atan2";
      b_ocaml = Stdlib.atan2;
      b_gen = (fun i -> (bounded (-3.0) 3.0 i, bounded (-3.0) 3.0 (i + 13)));
      b_tol = 1e-9;
    };
    {
      b_name = "hypot";
      b_ocaml = Stdlib.hypot;
      b_gen = (fun i -> (bounded (-3.0) 3.0 i, bounded (-3.0) 3.0 (i + 5)));
      b_tol = 1e-8;
    };
    {
      b_name = "copysign";
      b_ocaml = Stdlib.copysign;
      (* Edge-case coverage vs the OCaml [Stdlib.copysign] reference, tol 0.0
         (copysign is exact — a pure sign-bit transfer). [x] is forced nonzero
         so the sign-transfer result has detectable magnitude, then [y] cycles
         through both signed zeros and both nonzero signs. The [y = ±0.0] rows
         are the discriminating ones: the naive [abs(x)*sign(y)] lowering gives
         [|x|*0 = 0] there (GLSL [sign(0)=0]), so it would report [0.0] where C
         copysign requires [±|x|] — caught here as a magnitude mismatch. The
         bit-level helper transfers [y]'s sign bit exactly, including for
         [-0.0]. *)
      b_gen =
        (fun i ->
          let x =
            let v = bounded (-3.0) 3.0 i in
            if v = 0.0 then 1.5 else v
          in
          let y =
            match i mod 6 with
            | 0 -> 0.0 (* +0.0 -> +|x| *)
            | 1 -> -0.0 (* -0.0 -> -|x| (naive abs*sign would give 0) *)
            | 2 -> 2.5
            | 3 -> -2.5
            | 4 ->
                float_of_int (i - 128) (* spans both signs across the range *)
            | _ -> bounded (-3.0) 3.0 (i + 11)
          in
          (x, y));
      b_tol = 0.0;
    };
  ]

(** {1 Runner} *)

type outcome = Pass | Mismatch of int | Skipped of string | Errored of string

let run_unary (dev : Device.t) spec =
  let ir = make_unary_ir spec.u_name in
  let a_vec = Vector.create Vector.float64 n in
  let b_vec = Vector.create Vector.float64 n in
  for i = 0 to n - 1 do
    Vector.set a_vec i (spec.u_gen i) ;
    Vector.set b_vec i 0.0
  done ;
  let block = Execute.dims1d 256 in
  let grid = Execute.dims1d 1 in
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Execute.Vec a_vec; Execute.Vec b_vec]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let result = Vector.to_array b_vec in
  let errors = ref 0 in
  for i = 0 to n - 1 do
    let expected = spec.u_ocaml (spec.u_gen i) in
    if abs_float (result.(i) -. expected) > spec.u_tol then incr errors
  done ;
  if !errors = 0 then Pass else Mismatch !errors

let run_binary (dev : Device.t) spec =
  let ir = make_binary_ir spec.b_name in
  let a_vec = Vector.create Vector.float64 n in
  let b_vec = Vector.create Vector.float64 n in
  let c_vec = Vector.create Vector.float64 n in
  for i = 0 to n - 1 do
    let x, y = spec.b_gen i in
    Vector.set a_vec i x ;
    Vector.set b_vec i y ;
    Vector.set c_vec i 0.0
  done ;
  let block = Execute.dims1d 256 in
  let grid = Execute.dims1d 1 in
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Execute.Vec a_vec; Execute.Vec b_vec; Execute.Vec c_vec]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let result = Vector.to_array c_vec in
  let errors = ref 0 in
  for i = 0 to n - 1 do
    let x, y = spec.b_gen i in
    let expected = spec.b_ocaml x y in
    if abs_float (result.(i) -. expected) > spec.b_tol then incr errors
  done ;
  if !errors = 0 then Pass else Mismatch !errors

let outcome_of_exn f =
  try f () with
  | Spoc_framework.Backend_error.Backend_error err ->
      Skipped (Spoc_framework.Backend_error.to_string err)
  | e -> Errored (Printexc.to_string e)

let string_of_outcome = function
  | Pass -> "PASS"
  | Mismatch n -> Printf.sprintf "FAIL (%d/%d mismatched)" n n
  | Skipped reason -> Printf.sprintf "SKIP (%s)" reason
  | Errored reason -> Printf.sprintf "ERROR (%s)" reason

let () =
  let cfg = Test_helpers.parse_args "test_float64_math_intrinsics" in
  let devs = Test_helpers.init_devices cfg in
  let fp64_devs = Array.to_list devs |> List.filter Device.allows_fp64 in
  print_endline
    "=== Float64 Math Intrinsics Report (all fp64-capable devices) ===" ;
  Printf.printf
    "fp64-capable devices: %d / %d total\n\n"
    (List.length fp64_devs)
    (Array.length devs) ;
  if fp64_devs = [] then
    print_endline
      "No fp64-capable device found - nothing to report (not a failure)."
  else begin
    let results = ref [] in
    List.iter
      (fun (dev : Device.t) ->
        Printf.printf "-- %s (%s) --\n" dev.Device.name dev.Device.framework ;
        List.iter
          (fun spec ->
            let outcome = outcome_of_exn (fun () -> run_unary dev spec) in
            Printf.printf
              "  %-12s %s\n%!"
              spec.u_name
              (string_of_outcome outcome) ;
            results := (dev, spec.u_name, outcome) :: !results)
          unary_specs ;
        List.iter
          (fun spec ->
            let outcome = outcome_of_exn (fun () -> run_binary dev spec) in
            Printf.printf
              "  %-12s %s\n%!"
              spec.b_name
              (string_of_outcome outcome) ;
            results := (dev, spec.b_name, outcome) :: !results)
          binary_specs)
      fp64_devs ;
    let total = List.length !results in
    let passed =
      List.length (List.filter (fun (_, _, o) -> o = Pass) !results)
    in
    Printf.printf
      "\n=== Summary: %d/%d (device, function) combinations passed ===\n"
      passed
      total ;
    Printf.printf
      "This is a report, not a gate: individual failures above do not fail \
       this test or block CI.\n"
  end ;
  exit 0
