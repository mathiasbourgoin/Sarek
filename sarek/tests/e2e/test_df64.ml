(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek_df64 - double-float (float-float) extended precision.
 *
 * The df64 arithmetic is written in pure Sarek ([@sarek.module] functions in
 * sarek/Sarek_df64/Sarek_df64.ml) and pulled into this compilation unit with
 * %sarek_include, so the SAME source runs on every backend - including
 * devices without native float64.
 *
 * Checks, on ALL available devices:
 *   - elementwise add/sub/mul/div/sqrt/lt/of_int vs an OCaml binary64
 *     reference, within the documented contract (~2^-47 relative);
 *   - a df64 dot product over 2^20 elements where plain float32
 *     accumulation visibly fails.
 *
 * With SAREK_DF64_BENCH=1, also micro-benchmarks a compute-bound df64
 * polynomial iteration against native float64 on fp64-capable devices.
 ******************************************************************************)

[@@@warning "-32"]

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Host = Sarek_df64.Host

let () = Test_helpers.Benchmarks.init ()

type float32 = float

type ('a, 'b) vector = ('a, 'b) Vector.t

let%sarek_include _ = "../../Sarek_df64/Sarek_df64.ml"

(* ========== Kernels ========== *)

let ops_kernel =
  [%kernel
    fun (a : Sarek_df64.df64 vector)
        (b : Sarek_df64.df64 vector)
        (add_out : Sarek_df64.df64 vector)
        (sub_out : Sarek_df64.df64 vector)
        (mul_out : Sarek_df64.df64 vector)
        (div_out : Sarek_df64.df64 vector)
        (sqrt_out : Sarek_df64.df64 vector)
        (lt_out : int32 vector)
        (conv_out : float32 vector)
        (n : int32) ->
      let open Sarek_df64 in
      let tid = thread_idx_x + (block_idx_x * block_dim_x) in
      if tid < n then begin
        let x = a.(tid) in
        let y = b.(tid) in
        add_out.(tid) <- df64_add x y ;
        sub_out.(tid) <- df64_sub x y ;
        mul_out.(tid) <- df64_mul x y ;
        div_out.(tid) <- df64_div x y ;
        sqrt_out.(tid) <- df64_sqrt (df64_abs x) ;
        lt_out.(tid) <- (if df64_lt x y then 1l else 0l) ;
        conv_out.(tid) <- df64_to_float32 (df64_of_int32 tid)
      end]

(* Grid-stride df64 dot product: one df64 partial sum per thread. *)
let dot_kernel =
  [%kernel
    fun (a : Sarek_df64.df64 vector)
        (b : Sarek_df64.df64 vector)
        (partial : Sarek_df64.df64 vector)
        (n : int32)
        (nthreads : int32) ->
      let open Sarek_df64 in
      let tid = thread_idx_x + (block_idx_x * block_dim_x) in
      if tid < nthreads then begin
        let h = mut 0.0 in
        let l = mut 0.0 in
        let i = mut tid in
        while i < n do
          let acc = df64_add {hi = h; lo = l} (df64_mul a.(i) b.(i)) in
          h := acc.hi ;
          l := acc.lo ;
          i := i + nthreads
        done ;
        partial.(tid) <- {hi = h; lo = l}
      end]

(* Compute-bound bench kernels: iterated polynomial acc <- acc*x + x. *)
let df64_poly_kernel =
  [%kernel
    fun (x : Sarek_df64.df64 vector)
        (out : Sarek_df64.df64 vector)
        (n : int32)
        (iters : int32) ->
      let open Sarek_df64 in
      let tid = thread_idx_x + (block_idx_x * block_dim_x) in
      if tid < n then begin
        let v = x.(tid) in
        let h = mut 0.0 in
        let l = mut 0.0 in
        let j = mut 0l in
        while j < iters do
          let acc = df64_add (df64_mul {hi = h; lo = l} v) v in
          h := acc.hi ;
          l := acc.lo ;
          j := j + 1l
        done ;
        out.(tid) <- {hi = h; lo = l}
      end]

let f64_poly_kernel =
  [%kernel
    fun (x : float64 vector)
        (out : float64 vector)
        (n : int32)
        (iters : int32) ->
      let tid = thread_idx_x + (block_idx_x * block_dim_x) in
      if tid < n then begin
        let v = x.(tid) in
        let acc = mut (float64_of_int 0l) in
        let j = mut 0l in
        while j < iters do
          acc := (acc *. v) +. v ;
          j := j + 1l
        done ;
        out.(tid) <- acc
      end]

(* ========== Helpers ========== *)

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let fill_df64 vec data =
  Array.iteri (fun i x -> Vector.set vec i (Host.encode x)) data

(** Exact binary64 value held by a df64 vector element. *)
let read_df64 vec i = Host.decode (Vector.get vec i)

let rel_error got expected =
  if expected = 0.0 then abs_float got
  else abs_float ((got -. expected) /. expected)

(* Log-uniform random value with decimal exponent in [emin, emax]. *)
let random_logu emin emax =
  let e = emin +. Random.float (emax -. emin) in
  let m = 1.0 +. Random.float 9.0 in
  let s = if Random.bool () then 1.0 else -1.0 in
  s *. m *. (10.0 ** e)

let failures = ref 0

(* Per-op max relative error accumulator: op -> (max_err, tol). *)
let op_stats : (string, float * float) Hashtbl.t = Hashtbl.create 8

let check ~dev:_ ~op ~tol _i got expected =
  let err =
    if Float.is_nan got then Float.infinity else rel_error got expected
  in
  match Hashtbl.find_opt op_stats op with
  | Some (prev, _) -> if err > prev then Hashtbl.replace op_stats op (err, tol)
  | None -> Hashtbl.replace op_stats op (err, tol)

(* [of_i32] is not a df64 op and keeps its own literal tolerance; every df64
   arithmetic op goes through the shared classifier in Test_helpers. *)
let report_op_stats dev =
  let framework = dev.Device.framework and device = dev.Device.name in
  Hashtbl.iter
    (fun op (err, tol) ->
      let status, tol =
        match op with
        | "add" | "sub" | "mul" | "div" | "sqrt" ->
            ( Test_helpers.classify_df64_result ~framework ~device ~op ~err,
              Test_helpers.df64_tol_for_op op )
        | _ -> ((if err <= tol then `Pass else `Fail), tol)
      in
      (* [`Xpass] counts as a failure: a stale allowlist entry that only printed
         a warning would leave the exit code at 0 and rot unnoticed. See the
         "WHY [`Xpass] IS HARD" note on Test_helpers.classify_df64_result. *)
      (match status with `Fail | `Xpass _ -> incr failures | _ -> ()) ;
      Printf.printf
        "  %-6s max rel err %.3g (tol %.3g) %s [%s]\n%!"
        op
        err
        tol
        (match status with
        | `Pass -> "PASS"
        | `Xpass msg -> msg
        | `Known_deviation label -> label
        | `Fail -> "FAIL")
        framework)
    op_stats ;
  Hashtbl.reset op_stats

(* ========== Elementwise ops test ========== *)

let n_ops = 4096

let ops_inputs =
  Random.init 0x5a5e ;
  let a = Array.init n_ops (fun _ -> random_logu (-8.0) 8.0) in
  let b = Array.init n_ops (fun _ -> random_logu (-4.0) 4.0) in
  (a, b)

let run_ops_test dev =
  let a_data, b_data = ops_inputs in
  let mk () = Vector.create_custom Sarek_df64.df64_custom n_ops in
  let a = mk () and b = mk () in
  let add_o = mk () and sub_o = mk () and mul_o = mk () in
  let div_o = mk () and sqrt_o = mk () in
  let lt_o = Vector.create Vector.int32 n_ops in
  let conv_o = Vector.create Vector.float32 n_ops in
  fill_df64 a a_data ;
  fill_df64 b b_data ;
  let block = Sarek.Execute.dims1d 256 in
  let grid = Sarek.Execute.dims1d ((n_ops + 255) / 256) in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir:(ir_of ops_kernel)
    ~args:
      Sarek.Execute.
        [
          Vec a;
          Vec b;
          Vec add_o;
          Vec sub_o;
          Vec mul_o;
          Vec div_o;
          Vec sqrt_o;
          Vec lt_o;
          Vec conv_o;
          Int n_ops;
        ]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  (* Tolerances and the per-backend deviation allowlist both live in
     [Test_helpers] now (task #118), so test_df64, test_real64 and
     test_real64_single_source cannot drift apart. The bounds are derived there
     from the algorithms' error analysis; the deviating backends are held to a
     two-sided BAND and reported as KNOWN-DEVIATION rather than being handed a
     widened ceiling.

     What was wrong before: the deviating backends were checked against
     [0x1p-22] = 2.38e-07, four times the float32 unit roundoff. A fully
     collapsed df64 (5.84e-08 on RADV) and a working one (9.07e-15) both read
     PASS, so the gate could not distinguish the bug from the fix. The old
     widening also keyed on the bare "Vulkan" framework tag, which swept in
     NVIDIA Vulkan — a backend that meets the full contract — and would have
     hidden a contraction-class regression there. The allowlist in
     [Test_helpers.df64_known_deviation] is keyed on driver identity (RADV,
     ANV) instead. *)
  let tol_add = Test_helpers.df64_tol_add_sub in
  let tol_mul = Test_helpers.df64_tol_mul_div_sqrt in
  let tol_dv = tol_mul in
  let tol_sqrt = tol_mul in
  for i = 0 to n_ops - 1 do
    let x = read_df64 a i and y = read_df64 b i in
    check ~dev ~op:"add" ~tol:tol_add i (read_df64 add_o i) (x +. y) ;
    check ~dev ~op:"sub" ~tol:tol_add i (read_df64 sub_o i) (x -. y) ;
    check ~dev ~op:"mul" ~tol:tol_mul i (read_df64 mul_o i) (x *. y) ;
    check ~dev ~op:"div" ~tol:tol_dv i (read_df64 div_o i) (x /. y) ;
    check
      ~dev
      ~op:"sqrt"
      ~tol:tol_sqrt
      i
      (read_df64 sqrt_o i)
      (sqrt (abs_float x)) ;
    let lt_ref = if x < y then 1l else 0l in
    if Vector.get lt_o i <> lt_ref then begin
      incr failures ;
      Printf.printf "  FAIL [%s] lt[%d]\n%!" dev.Device.framework i
    end ;
    (* Exact, not approximate: every index here is below [n_ops] = 4096 < 2^24,
       so it is representable in binary32 with no rounding, [df64_of_int32]
       stores it as (hi = i, lo = 0) and [df64_to_float32] returns hi. The
       tolerance is therefore 0 — a literal 1e-7 here was another f32-grade
       gate on a df64 test. Measured 0 on all seven devices enumerated on the
       campaign workstation. *)
    check ~dev ~op:"of_i32" ~tol:0.0 i (Vector.get conv_o i) (float_of_int i)
  done ;
  report_op_stats dev

(* ========== Dot product / accumulation test ========== *)

let n_dot = 1 lsl 20

let nthreads_dot = 8192

let dot_inputs =
  let a =
    Array.init n_dot (fun i -> 1.0 +. (float_of_int (i mod 97) *. 1e-4))
  in
  let b =
    Array.init n_dot (fun i -> 1.0 -. (float_of_int (i mod 89) *. 1e-4))
  in
  (a, b)

let run_dot_test dev =
  let a_data, b_data = dot_inputs in
  let a = Vector.create_custom Sarek_df64.df64_custom n_dot in
  let b = Vector.create_custom Sarek_df64.df64_custom n_dot in
  let partial = Vector.create_custom Sarek_df64.df64_custom nthreads_dot in
  fill_df64 a a_data ;
  fill_df64 b b_data ;
  for i = 0 to nthreads_dot - 1 do
    Vector.set partial i {Sarek_df64.hi = 0.0; lo = 0.0}
  done ;
  let block = Sarek.Execute.dims1d 256 in
  let grid = Sarek.Execute.dims1d (nthreads_dot / 256) in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir:(ir_of dot_kernel)
    ~args:Sarek.Execute.[Vec a; Vec b; Vec partial; Int n_dot; Int nthreads_dot]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  (* Combine partials on host in df64, then compare to binary64 reference. *)
  let sum = ref {Sarek_df64.hi = 0.0; lo = 0.0} in
  for i = 0 to nthreads_dot - 1 do
    sum := Host.add !sum (Vector.get partial i)
  done ;
  let reference = ref 0.0 in
  let f32_sum = ref 0.0 in
  for i = 0 to n_dot - 1 do
    let x = read_df64 a i and y = read_df64 b i in
    reference := !reference +. (x *. y) ;
    f32_sum := Host.(!f32_sum +% (Host.round_f32 x *% Host.round_f32 y))
  done ;
  let df64_err = rel_error (Host.decode !sum) !reference in
  let f32_err = rel_error !f32_sum !reference in
  Printf.printf
    "  dot(2^20): df64 rel err %.3g (f32: %.3g)\n%!"
    df64_err
    f32_err ;
  (* Worst-case error of n df64 adds is ~n*2^-48 = 2^20 * 2^-48 = 3.7e-09, so
     1e-9 is already tighter than the proven bound and is met only because the
     rounding errors random-walk rather than accumulate.

     THIS GATE DOES NOT DISCRIMINATE a collapsed df64: measured 1.41e-10 on
     RADV (where mul/div ARE collapsed) versus 2.89e-13 on OpenCL/rusticl and
     the interpreter — both under 1e-9. It is a smoke test that df64
     accumulation beats float32 accumulation (3.07e-04 here), nothing more.
     The discriminating gate is the elementwise ops test above; do not read a
     green line here as evidence that df64 is intact. Tightening this number
     to separate 1.41e-10 from 2.89e-13 would be curve-fitting to one dataset,
     not a derived bound, so it is left alone. *)
  if df64_err > 1e-9 then begin
    incr failures ;
    Printf.printf
      "  FAIL [%s] dot: df64 error too large\n%!"
      dev.Device.framework
  end ;
  if f32_err < 1e-6 then
    Printf.printf "  note: f32 accumulation unexpectedly accurate here\n%!"

(* ========== Perf micro-bench (compute-bound) ========== *)

let time_runs dev ~ir ~args ~block ~grid =
  let run () =
    Sarek.Execute.run_vectors ~device:dev ~ir ~args ~block ~grid () ;
    Transfer.flush dev
  in
  run () (* warmup + JIT *) ;
  let reps = 5 in
  let t0 = Unix.gettimeofday () in
  for _ = 1 to reps do
    run ()
  done ;
  let t1 = Unix.gettimeofday () in
  (t1 -. t0) /. float_of_int reps *. 1000.0

let run_bench dev =
  let n = 1 lsl 20 and iters = 512 in
  let x64 = Vector.create Vector.float64 n in
  let out64 = Vector.create Vector.float64 n in
  let xdf = Vector.create_custom Sarek_df64.df64_custom n in
  let outdf = Vector.create_custom Sarek_df64.df64_custom n in
  for i = 0 to n - 1 do
    let v = 0.5 +. (float_of_int (i mod 1000) *. 1e-6) in
    Vector.set x64 i v ;
    Vector.set xdf i (Host.encode v) ;
    Vector.set out64 i 0.0 ;
    Vector.set outdf i {Sarek_df64.hi = 0.0; lo = 0.0}
  done ;
  let block = Sarek.Execute.dims1d 256 in
  let grid = Sarek.Execute.dims1d (n / 256) in
  let df64_ms =
    time_runs
      dev
      ~ir:(ir_of df64_poly_kernel)
      ~args:Sarek.Execute.[Vec xdf; Vec outdf; Int n; Int iters]
      ~block
      ~grid
  in
  Printf.printf
    "  bench df64 poly (2^20 x %d iters): %8.3f ms\n%!"
    iters
    df64_ms ;
  if Device.allows_fp64 dev then begin
    let f64_ms =
      time_runs
        dev
        ~ir:(ir_of f64_poly_kernel)
        ~args:Sarek.Execute.[Vec x64; Vec out64; Int n; Int iters]
        ~block
        ~grid
    in
    Printf.printf
      "  bench f64  poly (2^20 x %d iters): %8.3f ms -> df64/f64 ratio %.2fx\n\
       %!"
      iters
      f64_ms
      (df64_ms /. f64_ms)
  end
  else Printf.printf "  bench f64: device has no fp64 support, skipped\n%!"

(* ========== Main ========== *)

let () =
  let devs = Device.init () in
  if Array.length devs = 0 then begin
    print_endline "No devices found - SKIPPED" ;
    exit 0
  end ;
  let bench = Sys.getenv_opt "SAREK_DF64_BENCH" = Some "1" in
  Array.iter
    (fun dev ->
      Printf.printf "Device: %s [%s]\n%!" dev.Device.name dev.Device.framework ;
      run_ops_test dev ;
      run_dot_test dev ;
      if bench && Test_helpers.Benchmarks.gpu_only dev then run_bench dev)
    devs ;
  if !failures = 0 then print_endline "test_df64 PASSED"
  else begin
    Printf.printf "test_df64 FAILED (%d failures)\n" !failures ;
    exit 1
  end
