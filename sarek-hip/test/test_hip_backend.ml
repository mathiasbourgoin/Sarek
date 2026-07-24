(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Etape-A hardware gate for the native ROCm/HIP backend.
 *
 * Runs a Sarek DSL vector-add AND the shared-memory tiled SGEMM
 * (Sarek_gemm.sgemm_tiled_kernel, UNCHANGED) on the HIP device via hiprtc JIT,
 * cross-checked against a CPU reference. This is a Sarek kernel executing on
 * AMD WITHOUT ZLUDA.
 *
 * Skip-clean: if no HIP device is enumerated (no ROCm hardware / disabled),
 * prints [SKIP] and exits 0, so the suite stays green off-ROCm machines.
 *
 * Linking sarek_hip auto-registers the backend; the module reference below
 * also forces the link unit in.
 ******************************************************************************)

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module GH = Sarek_gemm.Host

(* Force the sarek_hip link unit in (auto-registers the HIP backend). *)
let () = Sarek_hip.Hip_plugin.register ()

let vector_add =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then c.(tid) <- a.(tid) + b.(tid)]

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

let vadd_ir = ir_of vector_add

let tiled_ir = ir_of Sarek_gemm.sgemm_tiled_kernel

(* ---- vector add ---- *)
let run_vector_add dev =
  let n = 4096 in
  let a = Vector.create Vector.float32 n in
  let b = Vector.create Vector.float32 n in
  let c = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set a i (float_of_int i) ;
    Vector.set b i (float_of_int (2 * i)) ;
    Vector.set c i (-999.0)
  done ;
  let block_sz = 256 in
  Execute.run_vectors
    ~device:dev
    ~ir:vadd_ir
    ~args:[Vec a; Vec b; Vec c; Int n]
    ~block:(Execute.dims1d block_sz)
    ~grid:(Execute.dims1d ((n + block_sz - 1) / block_sz))
    () ;
  Transfer.flush dev ;
  let res = Vector.to_array c in
  let ok = ref true in
  for i = 0 to n - 1 do
    let exp = float_of_int i +. float_of_int (2 * i) in
    if abs_float (res.(i) -. exp) > 1e-3 then ok := false
  done ;
  !ok

(* ---- tiled GEMM (unchanged kernel) vs CPU reference ---- *)
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

let vec_of a =
  let v = Vector.create Vector.float32 (Array.length a) in
  Array.iteri (fun i x -> Vector.set v i x) a ;
  v

let fill rows cols seed =
  Array.init (rows * cols) (fun i ->
      float_of_int ((((i * 1103515245) + seed + 12345) mod 1000) - 500) /. 500.0)

let close_enough ~k result expected =
  let n = Array.length expected in
  let eps = 1e-4 +. (float_of_int k *. 1e-6) in
  let bad = ref 0 in
  for i = 0 to n - 1 do
    let e = expected.(i) and g = result.(i) in
    if abs_float (e -. g) > eps *. (1.0 +. abs_float e) then incr bad
  done ;
  !bad = 0

let run_gemm_case dev ~m ~n ~k ~alpha ~beta =
  let a = fill m k 1 and b = fill k n 2 and c0 = fill m n 3 in
  let expected = cpu_gemm a b c0 ~m ~n ~k ~alpha ~beta in
  let va = vec_of a and vb = vec_of b and vc = vec_of c0 in
  Execute.run_vectors
    ~device:dev
    ~ir:tiled_ir
    ~args:
      [
        Vec va;
        Vec vb;
        Vec vc;
        Int32 (Int32.of_int m);
        Int32 (Int32.of_int n);
        Int32 (Int32.of_int k);
        Float32 alpha;
        Float32 beta;
      ]
    ~block:(GH.block ())
    ~grid:(GH.grid ~m ~n)
    () ;
  Transfer.flush dev ;
  close_enough ~k (Vector.to_array vc) expected

let () =
  let devices = Device.init () in
  let hip =
    Array.to_list devices |> List.filter (fun d -> d.Device.framework = "HIP")
  in
  match hip with
  | [] ->
      print_endline "test_hip_backend: [SKIP] no HIP device present" ;
      exit 0
  | devs ->
      let all_ok = ref true in
      List.iter
        (fun dev ->
          Printf.printf
            "  HIP device: %s (%s)\n"
            dev.Device.name
            dev.Device.framework ;
          let vadd =
            try run_vector_add dev
            with e ->
              Printf.printf "    vector_add EXN: %s\n" (Printexc.to_string e) ;
              false
          in
          Printf.printf
            "    vector_add            : %s\n"
            (if vadd then "OK" else "FAIL") ;
          let cases =
            [
              ("exact 32x32x32", 32, 32, 32, 1.0, 0.0);
              ("boundary 30x30x30", 30, 30, 30, 1.0, 0.0);
              ("boundary 17x17x17", 17, 17, 17, 1.0, 0.0);
              ("rect 37x50x23", 37, 50, 23, 1.0, 0.0);
              ("large 128x96x160", 128, 96, 160, 1.0, 0.0);
              ("alpha/beta 40x24x33", 40, 24, 33, 2.0, 3.0);
            ]
          in
          let gemm_ok =
            List.for_all
              (fun (nm, m, n, k, alpha, beta) ->
                let ok =
                  try run_gemm_case dev ~m ~n ~k ~alpha ~beta
                  with e ->
                    Printf.printf
                      "    gemm %s EXN: %s\n"
                      nm
                      (Printexc.to_string e) ;
                    false
                in
                Printf.printf
                  "    gemm %-20s : %s\n"
                  nm
                  (if ok then "OK" else "FAIL") ;
                ok)
              cases
          in
          if not (vadd && gemm_ok) then all_ok := false)
        devs ;
      if !all_ok then print_endline "test_hip_backend PASSED"
      else (
        print_endline "test_hip_backend FAILED" ;
        exit 1)
