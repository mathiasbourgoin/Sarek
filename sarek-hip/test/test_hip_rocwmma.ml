(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Etape-B gate: rocWMMA tensor-core GEMM through the native HIP backend.
 *
 * A hand-written rocWMMA 16x16x16 tiled SGEMM (f16 inputs, f32 accumulate) is
 * JIT-compiled via the HIP backend's own compile path (Hip_api.Kernel.
 * compile_with_options -> hiprtc with -I/opt/rocm/include -> hipModuleLoadData)
 * and launched with hipModuleLaunchKernel. Correctness is checked against a CPU
 * reference computed on the SAME half-rounded inputs; a speedup number vs the
 * pure-Sarek shared-memory tiled SGEMM (Sarek_gemm.sgemm_tiled_kernel) on the
 * same device is reported (honest, non-gating on the ratio).
 *
 * Requires an RDNA3+ device (WMMA; compute-capability major >= 11). Skip-clean
 * (exit 0) if none present. This is the DSL-level f16 element type's stand-in:
 * the rocWMMA kernel runs through the real backend plumbing; wiring an f16
 * vector element type into the high-level Sarek DSL is a documented follow-up.
 ******************************************************************************)

module SDevice = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module HA = Sarek_hip.Hip_api
open Sarek
module GH = Sarek_gemm.Host

let () = Sarek_hip.Hip_plugin.register ()

(* ---- IEEE-754 binary16 (half) encode/decode, round-toward-zero mantissa ---- *)
let half_bits_of_float x =
  let f = Int32.bits_of_float x in
  let fi = Int32.to_int (Int32.logand f 0xffffffffl) in
  let sign = (fi lsr 16) land 0x8000 in
  let exp = ((fi lsr 23) land 0xff) - 127 + 15 in
  let mant = (fi lsr 13) land 0x3ff in
  if x = 0.0 then sign
  else if exp <= 0 then sign
  else if exp >= 31 then sign lor 0x7c00
  else sign lor (exp lsl 10) lor mant

(* rocWMMA tiled SGEMM: block = one wavefront computes a 16x16 C tile; K marched
   in steps of 16 = one 16x16x16 fragment MAC per step (exactly the L15a tile).
   alpha=1, beta=0. M,N,K multiples of 16. *)
let wmma_src =
  {|
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;
extern "C" __global__ void wmma_sgemm(const float16_t* A, const float16_t* B,
                                      float* C, int M, int N, int K) {
  int tileM = blockIdx.y;
  int tileN = blockIdx.x;
  fragment<matrix_a, 16, 16, 16, float16_t, row_major> fa;
  fragment<matrix_b, 16, 16, 16, float16_t, row_major> fb;
  fragment<accumulator, 16, 16, 16, float> acc;
  fill_fragment(acc, 0.0f);
  for (int k = 0; k < K; k += 16) {
    load_matrix_sync(fa, A + (tileM * 16 * K) + k, K);
    load_matrix_sync(fb, B + (k * N) + (tileN * 16), N);
    mma_sync(acc, fa, fb, acc);
  }
  store_matrix_sync(C + (tileM * 16 * N) + (tileN * 16), acc, N, mem_row_major);
}
|}

let fill rows cols seed =
  Array.init (rows * cols) (fun i ->
      float_of_int ((((i * 2654435761) + seed) mod 17) - 8) /. 8.0)

(* CPU reference on the TRUE (un-rounded) inputs, accumulated in OCaml float
   (IEEE-754 f64). This is the ideal GEMM the device result is graded against:
   the device rounds its inputs to f16, so the difference device-vs-reference is
   dominated by that f16 input rounding (see [close_enough] for the tolerance
   derivation). Grading against the ideal — rather than a CPU copy that also
   rounds to f16 — is what makes this an actual correctness gate. *)
let cpu_gemm_half ah bh ~m ~n ~k =
  let out = Array.make (m * n) 0.0 in
  for row = 0 to m - 1 do
    for col = 0 to n - 1 do
      let s = ref 0.0 in
      for i = 0 to k - 1 do
        s := !s +. (ah.((row * k) + i) *. bh.((i * n) + col))
      done ;
      out.((row * n) + col) <- !s
    done
  done ;
  out

let run_rocwmma hdev ~m ~n ~k =
  let a = fill m k 1 and b = fill k n 2 in
  (* half-encode inputs into int16 device buffers *)
  let ah = Bigarray.(Array1.create int16_unsigned c_layout (m * k)) in
  let bh = Bigarray.(Array1.create int16_unsigned c_layout (k * n)) in
  Array.iteri (fun i x -> ah.{i} <- half_bits_of_float x) a ;
  Array.iteri (fun i x -> bh.{i} <- half_bits_of_float x) b ;
  let da = HA.Memory.alloc hdev (m * k) Bigarray.int16_unsigned in
  let db = HA.Memory.alloc hdev (k * n) Bigarray.int16_unsigned in
  let dc = HA.Memory.alloc hdev (m * n) Bigarray.float32 in
  HA.Memory.host_to_device ~src:ah ~dst:da ;
  HA.Memory.host_to_device ~src:bh ~dst:db ;
  let kern =
    HA.Kernel.compile_with_options
      hdev
      ~name:"wmma_sgemm"
      ~source:wmma_src
      ~options:["-I/opt/rocm/include"]
  in
  let args =
    HA.Kernel.
      [
        ArgBuffer da;
        ArgBuffer db;
        ArgBuffer dc;
        ArgInt32 (Int32.of_int m);
        ArgInt32 (Int32.of_int n);
        ArgInt32 (Int32.of_int k);
      ]
  in
  let launch () =
    HA.Kernel.launch
      kern
      ~args
      ~grid:(n / 16, m / 16, 1)
      ~block:(32, 1, 1)
      ~shared_mem:0
      ~stream:None ;
    HA.Device.synchronize hdev
  in
  launch () (* warm-up *) ;
  let t0 = Unix.gettimeofday () in
  let iters = 20 in
  for _ = 1 to iters do
    launch ()
  done ;
  let t1 = Unix.gettimeofday () in
  let hc = Bigarray.(Array1.create float32 c_layout (m * n)) in
  HA.Memory.device_to_host ~src:dc ~dst:hc ;
  let result = Array.init (m * n) (fun i -> hc.{i}) in
  let expected = cpu_gemm_half a b ~m ~n ~k in
  HA.Memory.free da ;
  HA.Memory.free db ;
  HA.Memory.free dc ;
  (result, expected, (t1 -. t0) *. 1000.0 /. float_of_int iters)

(* Pure-Sarek shared-memory tiled SGEMM timing on the same SDK device. *)
let tiled_ir =
  match (snd Sarek_gemm.sgemm_tiled_kernel).Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no tiled IR"

let run_sarek_tiled sdev ~m ~n ~k =
  let a = fill m k 1 and b = fill k n 2 in
  let vec arr =
    let v = Vector.create Vector.float32 (Array.length arr) in
    Array.iteri (fun i x -> Vector.set v i x) arr ;
    v
  in
  let va = vec a and vb = vec b and vc = vec (Array.make (m * n) 0.0) in
  let run () =
    Execute.run_vectors
      ~device:sdev
      ~ir:tiled_ir
      ~args:
        [
          Vec va;
          Vec vb;
          Vec vc;
          Int32 (Int32.of_int m);
          Int32 (Int32.of_int n);
          Int32 (Int32.of_int k);
          Float32 1.0;
          Float32 0.0;
        ]
      ~block:(GH.block ())
      ~grid:(GH.grid ~m ~n)
      () ;
    Transfer.flush sdev
  in
  run () (* warm-up *) ;
  let t0 = Unix.gettimeofday () in
  let iters = 20 in
  for _ = 1 to iters do
    run ()
  done ;
  let t1 = Unix.gettimeofday () in
  (t1 -. t0) *. 1000.0 /. float_of_int iters

(* Correctness gate tolerance, DERIVED from the f16 input rounding (not a fixed
   magic constant). Inputs are f16 (IEEE-754 binary16): 10-bit mantissa, so the
   unit roundoff is u = 2^-11 ~ 4.88e-4 relative. The device rounds A and B to
   f16 then accumulates the K-length dot product in f32; the reference is the
   ideal GEMM in f64. The device-vs-reference error is therefore dominated by
   the f16 input rounding, which for a K-term dot product grows between
   sqrt(K)*u (statistical / random-walk, independent rounding errors) and K*u
   (worst-case linear, fully-correlated errors) in RELATIVE terms.

   We gate at the statistical scale with a small safety constant C:
       rel_tol = C * sqrt(K) * u
   C = 4 gives ~4 standard deviations of headroom over the random-walk estimate
   for the pseudo-random [-1,1] inputs produced by [fill], while remaining ~1-2
   orders of magnitude TIGHTER than the worst-case K*u bound (and ~4x-8x tighter
   than the previous 3e-3 + K*5e-4 gate, which at K=1024 admitted a 51% error).

   For reference entries near zero (cancellation) the relative test is
   meaningless, so we add an absolute floor. Each product term |a_i*b_i| <= 1
   because [fill] constrains |a_i|,|b_i| <= 1 (P = 1 below); the absolute error
   of the sum is then bounded on the same C*sqrt(K)*u scale, giving
       abs_floor = C * sqrt(K) * u * P.
   The per-entry test is  |got - ref| <= rel_tol*|ref| + abs_floor .
   [max_ratio] is the worst-case (error / per-entry-tolerance); <= 1 passes. *)
let close_enough result expected ~k =
  let n = Array.length expected in
  let u = 2.0 ** -11.0 in
  let c = 4.0 in
  let rel_tol = c *. sqrt (float_of_int k) *. u in
  let product_bound = 1.0 in
  let abs_floor = rel_tol *. product_bound in
  let bad = ref 0 and max_ratio = ref 0.0 in
  for i = 0 to n - 1 do
    let e = expected.(i) and g = result.(i) in
    let tol = (rel_tol *. abs_float e) +. abs_floor in
    let ratio = abs_float (e -. g) /. tol in
    if ratio > !max_ratio then max_ratio := ratio ;
    if ratio > 1.0 then incr bad
  done ;
  (!bad = 0, !max_ratio)

let () =
  let devices = SDevice.init () in
  let hip_rdna3 =
    Array.to_list devices
    |> List.filter (fun d ->
        d.SDevice.framework = "HIP"
        && fst d.SDevice.capabilities.compute_capability >= 11)
  in
  match hip_rdna3 with
  | [] ->
      print_endline
        "test_hip_rocwmma: [SKIP] no RDNA3+ (WMMA-capable) HIP device present" ;
      exit 0
  | sdev :: _ ->
      Printf.printf
        "  rocWMMA device: %s (cc %d.%d)\n"
        sdev.SDevice.name
        (fst sdev.SDevice.capabilities.compute_capability)
        (snd sdev.SDevice.capabilities.compute_capability) ;
      let hdev = HA.Device.get sdev.SDevice.backend_id in
      let all_ok = ref true in
      List.iter
        (fun (m, n, k) ->
          let ok, ms_wmma =
            try
              let result, expected, ms = run_rocwmma hdev ~m ~n ~k in
              let ok, maxrel = close_enough result expected ~k in
              Printf.printf
                "    rocWMMA %dx%dx%d : %s (max tol ratio %.4g, %.4f ms)\n"
                m
                n
                k
                (if ok then "OK" else "FAIL")
                maxrel
                ms ;
              (ok, ms)
            with e ->
              Printf.printf
                "    rocWMMA %dx%dx%d EXN: %s\n"
                m
                n
                k
                (Printexc.to_string e) ;
              (false, 0.0)
          in
          if not ok then all_ok := false ;
          (* speedup vs Sarek tiled (non-gating) *)
          try
            let ms_tiled = run_sarek_tiled sdev ~m ~n ~k in
            if ms_wmma > 0.0 then
              Printf.printf
                "      vs Sarek tiled: tiled=%.4f ms  rocWMMA=%.4f ms  speedup \
                 %.2fx\n"
                ms_tiled
                ms_wmma
                (ms_tiled /. ms_wmma)
          with e ->
            Printf.printf
              "      (tiled timing skipped: %s)\n"
              (Printexc.to_string e))
        [(256, 256, 256); (512, 512, 512); (1024, 1024, 1024)] ;
      if !all_ok then print_endline "test_hip_rocwmma PASSED"
      else (
        print_endline "test_hip_rocwmma FAILED" ;
        exit 1)
