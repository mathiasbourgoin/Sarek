(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Cost of the per-launch checks.
 *
 * A downstream LLM inference engine decodes a token with 281 to 497 kernel launches, so anything a launch
 * computes from the whole IR is paid that many times per token. The warp
 * launch rule ([Execute.check_launch_warp]) and the matrix-rewrite decision
 * ([Matrix_specialise.choose]) both used to fold the entire IR at every launch;
 * a measured decode went from 359 to 168 tok/s. Both verdicts are pure
 * functions of the (immutable) IR and are now recorded once per IR.
 *
 * The assertions are relative to the cost of one whole-IR fold measured here,
 * so they hold on any machine: a launch must cost a small fraction of a fold,
 * where the old code cost a full one.
 ******************************************************************************)

open Spoc_framework
module Execute = Sarek_execute.Execute
module Matrix_specialise = Sarek_execute.Matrix_specialise
module Device = Spoc_core.Device
open Sarek_ir_types

let caps : Framework_sig.capabilities =
  {
    max_threads_per_block = 1024;
    max_block_dims = (1024, 1024, 64);
    max_grid_dims = (65535, 65535, 65535);
    shared_mem_per_block = 65536;
    total_global_mem = 1073741824L;
    compute_capability = (0, 0);
    device_features = [Sarek_ir_analysis.Warp_shuffle];
    coopmat = None;
    supports_atomics = true;
    warp_size = 32;
    max_registers_per_block = 0;
    clock_rate_khz = 0;
    multiprocessor_count = 1;
    is_cpu = false;
  }

let device : Device.t =
  {
    id = 0;
    backend_id = 0;
    name = "cost";
    framework = "HIP";
    capabilities = caps;
  }

let one = EConst (CInt32 1l)

let filler = SExpr (EBinop (Add, one, EBinop (Mul, one, one)))

let shuffle = SExpr (EIntrinsic ([], "warp_shuffle_xor_i32", [one; one]))

(* A kernel the size of a real one (tens of thousands of nodes). A fresh record
   per call: structurally equal calls give physically distinct IRs. *)
let big_kernel ~with_shuffle : kernel =
  let body = List.init 20000 (fun _ -> filler) in
  let body = if with_shuffle then body @ [shuffle] else body in
  {default_kernel with kern_name = "cost_k"; kern_body = SSeq body}

let block n = Framework_sig.{x = n; y = 1; z = 1}

let now = Unix.gettimeofday

let median l = List.nth (List.sort compare l) (List.length l / 2)

(* Seconds for one call of [f], median of [trials]. *)
let time_one ?(trials = 7) f =
  median
    (List.init trials (fun _ ->
         let t0 = now () in
         f () ;
         now () -. t0))

let launches = 1000

(* The cost of [launches] calls after a first, warming call. *)
let time_repeated f =
  f () ;
  let t0 = now () in
  for _ = 1 to launches do
    f ()
  done ;
  now () -. t0

let fold_cost ir =
  time_one (fun () -> ignore (Sarek_ir_analysis.kernel_uses_shuffle ir))

(* [launches] launches may cost at most this many folds. The old code cost
   [launches] of them; the memoised one costs a lookup per launch. *)
let allowed_folds = 50.

let assert_cheap label ~fold ~total =
  let ratio = total /. fold in
  if ratio > allowed_folds then
    Alcotest.failf
      "%s: %d launches cost %.1f whole-IR folds (one fold = %.1f us, total = \
       %.2f ms); the limit is %.0f"
      label
      launches
      ratio
      (fold *. 1e6)
      (total *. 1e3)
      allowed_folds

let test_warp_check_is_memoised () =
  let ir = big_kernel ~with_shuffle:true in
  let fold = fold_cost ir in
  let total =
    time_repeated (fun () ->
        Execute.check_launch_warp ~device ~kernel:"cost_k" ir ~block:(block 64))
  in
  assert_cheap "check_launch_warp (shuffle kernel)" ~fold ~total ;
  let plain = big_kernel ~with_shuffle:false in
  let total =
    time_repeated (fun () ->
        Execute.check_launch_warp
          ~device
          ~kernel:"cost_k"
          plain
          ~block:(block 33))
  in
  assert_cheap "check_launch_warp (plain kernel)" ~fold ~total

let test_matrix_decision_is_memoised () =
  let ir = big_kernel ~with_shuffle:true in
  let fold =
    time_one (fun () -> ignore (Sarek_ir_analysis.kernel_uses_warp_ops ir))
  in
  let total =
    time_repeated (fun () ->
        ignore
          (Matrix_specialise.choose ~device ~ir ~launch:(fun () -> None) ()))
  in
  assert_cheap "Matrix_specialise.choose (warp kernel)" ~fold ~total

let refused ir ~width =
  match
    Execute.check_launch_warp ~device ~kernel:"cost_k" ir ~block:(block width)
  with
  | () -> false
  | exception
      Sarek_execute.Execute_error.Execution_error
        (Sarek_execute.Execute_error.Invalid_dimensions _) ->
      true

(* The verdict belongs to the IR value: a kernel derived from a shuffle-free
   one that now shuffles (and the reverse) is analysed afresh, and a refusal is
   raised on every launch, not recorded as a success. *)
let test_verdict_follows_the_kernel () =
  let plain = big_kernel ~with_shuffle:false in
  Alcotest.(check bool) "plain, block 33" false (refused plain ~width:33) ;
  Alcotest.(check bool) "plain, block 33 again" false (refused plain ~width:33) ;
  let derived =
    match plain.kern_body with
    | SSeq l -> {plain with kern_body = SSeq (l @ [shuffle])}
    | _ -> Alcotest.fail "unexpected body"
  in
  Alcotest.(check bool) "derived, block 33" true (refused derived ~width:33) ;
  Alcotest.(check bool)
    "derived, block 33 again"
    true
    (refused derived ~width:33) ;
  Alcotest.(check bool) "derived, block 64" false (refused derived ~width:64) ;
  Alcotest.(check bool) "plain still unaffected" false (refused plain ~width:33) ;
  let stripped = {derived with kern_body = plain.kern_body} in
  Alcotest.(check bool) "stripped, block 33" false (refused stripped ~width:33)

let () =
  Alcotest.run
    "launch check cost"
    [
      ( "cost",
        [
          Alcotest.test_case
            "warp check memoised"
            `Quick
            test_warp_check_is_memoised;
          Alcotest.test_case
            "matrix decision memoised"
            `Quick
            test_matrix_decision_is_memoised;
        ] );
      ( "invalidation",
        [
          Alcotest.test_case
            "verdict follows the kernel"
            `Quick
            test_verdict_follows_the_kernel;
        ] );
    ]
