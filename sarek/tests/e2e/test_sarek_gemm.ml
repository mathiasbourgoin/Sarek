(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek_gemm - shared-memory tiled SGEMM (pure Sarek, FMA).
 *
 * The ready kernel value Sarek_gemm.sgemm_tiled_kernel is exercised on every
 * available shared-memory backend (Native + CUDA/PTX incl. ZLUDA + OpenCL +
 * Vulkan; the Interpreter has no shared-memory model and is skipped, like the
 * matmul/reduce e2e tests), and cross-checked against a CPU reference GEMM and
 * against a naive one-element-per-thread kernel.
 *
 * Cases (all verified vs CPU reference within a float32 relative epsilon):
 *   - tiny hand-checkable 2x2 (known product) and 3x3 (identity);
 *   - EXACT multiple of the tile (32x32x32);
 *   - NON-multiple of the tile (boundary: 30x30x30, 17x17x17);
 *   - rectangular M<>N<>K (37x50x23);
 *   - large / many blocks (128x96x160);
 *   - alpha/beta: C := alpha*A*B + beta*C with alpha=2, beta=3.
 * Plus tiled-vs-naive agreement and a (non-gating) tiled-vs-naive perf note.
 ******************************************************************************)

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module GH = Sarek_gemm.Host

(* ========================== CPU reference ========================== *)

(* c := alpha * a * b + beta * c, row-major, a: MxK, b: KxN, c: MxN. *)
let cpu_gemm a b c ~m ~n ~k ~alpha ~beta =
  let out = Array.make (m * n) 0.0 in
  for row = 0 to m - 1 do
    for col = 0 to n - 1 do
      let sum = ref 0.0 in
      for i = 0 to k - 1 do
        sum := !sum +. (a.((row * k) + i) *. b.((i * n) + col))
      done ;
      out.((row * n) + col) <- (alpha *. !sum) +. (beta *. c.((row * n) + col))
    done
  done ;
  out

(* ========================== naive GPU kernel ========================== *)

(* One thread per output element; same C := alpha*A*B + beta*C contract, for
   cross-checking the tiled kernel and the perf-ratio note. *)
let sgemm_naive_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32)
        (alpha : float32)
        (beta : float32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      let row = tid / n in
      let col = tid mod n in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
        done ;
        c.((row * n) + col) <- (alpha *. sum) +. (beta *. c.((row * n) + col))
      end]

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

let tiled_ir = ir_of Sarek_gemm.sgemm_tiled_kernel

let naive_ir = ir_of sgemm_naive_kernel

(* ========================== host runners ========================== *)

let vec_of a =
  let v = Vector.create Vector.float32 (Array.length a) in
  Array.iteri (fun i x -> Vector.set v i x) a ;
  v

let args a b c ~m ~n ~k ~alpha ~beta =
  [
    Execute.Vec a;
    Execute.Vec b;
    Execute.Vec c;
    Execute.Int32 (Int32.of_int m);
    Execute.Int32 (Int32.of_int n);
    Execute.Int32 (Int32.of_int k);
    Execute.Float32 alpha;
    Execute.Float32 beta;
  ]

(* Run the tiled kernel; returns (result array, elapsed_ms). c_init seeds C
   (for the beta term). *)
let run_tiled dev a_arr b_arr c_init ~m ~n ~k ~alpha ~beta =
  let a = vec_of a_arr and b = vec_of b_arr and c = vec_of c_init in
  let t0 = Unix.gettimeofday () in
  Execute.run_vectors
    ~device:dev
    ~ir:tiled_ir
    ~args:(args a b c ~m ~n ~k ~alpha ~beta)
    ~block:(GH.block ())
    ~grid:(GH.grid ~m ~n)
    ~shared_mem:GH.shared_mem_bytes
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  (Vector.to_array c, (t1 -. t0) *. 1000.0)

let run_naive dev a_arr b_arr c_init ~m ~n ~k ~alpha ~beta =
  let a = vec_of a_arr and b = vec_of b_arr and c = vec_of c_init in
  let block_sz = 256 in
  let grid_sz = ((m * n) + block_sz - 1) / block_sz in
  let t0 = Unix.gettimeofday () in
  Execute.run_vectors
    ~device:dev
    ~ir:naive_ir
    ~args:(args a b c ~m ~n ~k ~alpha ~beta)
    ~block:(Execute.dims1d block_sz)
    ~grid:(Execute.dims1d grid_sz)
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  (Vector.to_array c, (t1 -. t0) *. 1000.0)

(* ========================== verification ========================== *)

(* Tiled accumulation order differs from naive, so compare within a relative
   epsilon scaled by K (accumulated float32 rounding), not bit-exact. *)
let close_enough ~k result expected =
  let n = Array.length expected in
  let eps = 1e-4 +. (float_of_int k *. 1e-6) in
  let bad = ref 0 in
  for i = 0 to n - 1 do
    let e = expected.(i) and g = result.(i) in
    let tol = eps *. (1.0 +. abs_float e) in
    if abs_float (e -. g) > tol then begin
      if !bad < 4 then
        Printf.printf "    mismatch @%d: expected %.6f got %.6f\n" i e g ;
      incr bad
    end
  done ;
  !bad = 0

(* Deterministic pseudo-random-ish fill in [-1,1], reproducible per shape. *)
let fill rows cols seed =
  Array.init (rows * cols) (fun i ->
      let x =
        float_of_int ((((i * 1103515245) + seed + 12345) mod 1000) - 500)
      in
      x /. 500.0)

(* ========================== named cases ========================== *)

type case = {
  name : string;
  m : int;
  n : int;
  k : int;
  alpha : float;
  beta : float;
}

let cases =
  [
    {
      name = "exact-multiple 32x32x32";
      m = 32;
      n = 32;
      k = 32;
      alpha = 1.0;
      beta = 0.0;
    };
    {
      name = "boundary 30x30x30";
      m = 30;
      n = 30;
      k = 30;
      alpha = 1.0;
      beta = 0.0;
    };
    {
      name = "boundary 17x17x17";
      m = 17;
      n = 17;
      k = 17;
      alpha = 1.0;
      beta = 0.0;
    };
    {
      name = "rectangular 37x50x23";
      m = 37;
      n = 50;
      k = 23;
      alpha = 1.0;
      beta = 0.0;
    };
    {
      name = "large 128x96x160";
      m = 128;
      n = 96;
      k = 160;
      alpha = 1.0;
      beta = 0.0;
    };
    {
      name = "alpha/beta 40x24x33";
      m = 40;
      n = 24;
      k = 33;
      alpha = 2.0;
      beta = 3.0;
    };
  ]

let run_case dev {name; m; n; k; alpha; beta} =
  let a = fill m k 1 and b = fill k n 2 in
  let c_init = fill m n 3 in
  let expected = cpu_gemm a b c_init ~m ~n ~k ~alpha ~beta in
  let tiled, _ = run_tiled dev a b c_init ~m ~n ~k ~alpha ~beta in
  let ref_ok = close_enough ~k tiled expected in
  (* Cross-check tiled vs naive on the same inputs. *)
  let naive, _ = run_naive dev a b c_init ~m ~n ~k ~alpha ~beta in
  let cross_ok = close_enough ~k tiled naive in
  if not (ref_ok && cross_ok) then
    Printf.printf "    [%s] ref_ok=%b cross_ok=%b\n" name ref_ok cross_ok ;
  ref_ok && cross_ok

(* Tiny hand-checkable products (also boundary: smaller than a tile). *)
let run_hand_checked dev =
  (* 2x2: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]] *)
  let a = [|1.; 2.; 3.; 4.|] and b = [|5.; 6.; 7.; 8.|] in
  let c0 = Array.make 4 0.0 in
  let got2, _ = run_tiled dev a b c0 ~m:2 ~n:2 ~k:2 ~alpha:1.0 ~beta:0.0 in
  let exp2 = [|19.; 22.; 43.; 50.|] in
  let ok2 = close_enough ~k:2 got2 exp2 in
  (* 3x3: A * I = A *)
  let a3 = Array.init 9 (fun i -> float_of_int (i + 1)) in
  let id3 = [|1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1.|] in
  let c0' = Array.make 9 0.0 in
  let got3, _ = run_tiled dev a3 id3 c0' ~m:3 ~n:3 ~k:3 ~alpha:1.0 ~beta:0.0 in
  let ok3 = close_enough ~k:3 got3 a3 in
  if not (ok2 && ok3) then
    Printf.printf "    hand-checked ok2=%b ok3=%b\n" ok2 ok3 ;
  ok2 && ok3

(* ========================== perf note (non-gating) ========================== *)

let median xs =
  let a = Array.of_list xs in
  Array.sort compare a ;
  a.(Array.length a / 2)

let best_of ~iters run =
  let ts = ref [] in
  for _ = 1 to iters do
    let _, t = run () in
    ts := t :: !ts
  done ;
  median !ts

let perf_note dev =
  (* Bigger, compute-bound square on GPUs where tiling actually pays; a modest
     size on CPU-class devices to keep the note fast. Non-gating. *)
  let dim = match dev.Device.framework with "Native" -> 384 | _ -> 1024 in
  let m, n, k = (dim, dim, dim) in
  let a = fill m k 7 and b = fill k n 9 in
  let c0 = Array.make (m * n) 0.0 in
  let _ = run_tiled dev a b c0 ~m ~n ~k ~alpha:1.0 ~beta:0.0 in
  (* warm-up *)
  let _ = run_naive dev a b c0 ~m ~n ~k ~alpha:1.0 ~beta:0.0 in
  let tt =
    best_of ~iters:5 (fun () ->
        run_tiled dev a b c0 ~m ~n ~k ~alpha:1.0 ~beta:0.0)
  in
  let tn =
    best_of ~iters:5 (fun () ->
        run_naive dev a b c0 ~m ~n ~k ~alpha:1.0 ~beta:0.0)
  in
  let ratio = if tt > 0.0 then tn /. tt else 0.0 in
  Printf.printf
    "    perf %d^3 (median of 5): tiled=%.3fms naive=%.3fms (tiled speedup \
     %.2fx)\n"
    dim
    tt
    tn
    ratio

(* ========================== driver ========================== *)

let () =
  Test_helpers.Benchmarks.init () ;
  let devices = Device.init () in
  let all_ok = ref true in
  Array.iter
    (fun dev ->
      let fw = dev.Device.framework in
      let label = Printf.sprintf "%s (%s)" dev.Device.name fw in
      if fw = "Interpreter" then
        Printf.printf "  %-48s : SKIP (no shared memory)\n" label
      else begin
        Printf.printf "  %s\n" label ;
        let ok =
          try
            let hand = run_hand_checked dev in
            let cs = List.map (fun c -> (c.name, run_case dev c)) cases in
            List.iter
              (fun (nm, ok) ->
                Printf.printf
                  "    %-24s : %s\n"
                  nm
                  (if ok then "OK" else "FAIL"))
              (("hand-checked 2x2/3x3", hand) :: cs) ;
            (try perf_note dev with _ -> ()) ;
            hand && List.for_all snd cs
          with e ->
            Printf.printf "    EXN: %s\n" (Printexc.to_string e) ;
            false
        in
        if not ok then all_ok := false
      end)
    devices ;
  if !all_ok then print_endline "test_sarek_gemm PASSED"
  else (
    print_endline "test_sarek_gemm FAILED" ;
    exit 1)
