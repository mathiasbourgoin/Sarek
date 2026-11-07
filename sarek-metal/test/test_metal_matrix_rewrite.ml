(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The launch-time matrix rewrite on Metal (specs/matrix-pattern-rewrite.md,
 * "Metal"). No device: the Metal device is a record whose capabilities are the
 * pure rule Metal_coopmat.support, so the admission and the generated MSL are
 * what is under test; the executed counterpart is
 * sarek/tests/e2e/test_matrix_rewrite_metal.ml (alias e2e-metal).
 *
 * The float opt-ins are process-wide and one-way, so the cases run in order:
 *   1. integer is declined with the reason (no integer simdgroup_matrix);
 *   2. float before any opt-in is declined for the policy;
 *   3. relaxed accumulation granted: the explicit rewrite is admitted at
 *      fragment 8 (a 4x4 grid of 8x8 fragments per wave), the automatic one
 *      still needs accept_automatic_float_rewrite;
 *   4. a device whose probe failed, and a SIMD width the templates do not
 *      cover, are declined;
 *   5. the opt-in granted: the automatic path rewrites;
 *   6. the rewritten kernel's MSL.
 ******************************************************************************)

[@@@alert "-sarek_matrix"]

[@@@warning "-33"]

open Sarek
module Spec = Sarek_execute.Matrix_specialise
module Device = Spoc_core.Device
module Fw = Spoc_framework.Framework_sig
module C = Sarek_coopmat
module Rw = Sarek_matrix_rewrite

let contains ~needle s =
  let nl = String.length needle and hl = String.length s in
  let rec go i = i + nl <= hl && (String.sub s i nl = needle || go (i + 1)) in
  go 0

let ir_of (_, kirc) =
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> Alcotest.fail "no IR"

let k_f16 =
  ir_of
    [%kernel
      fun (a : float16 vector)
          (b : float16 vector)
          (c : float32 vector)
          (m : int32)
          (n : int32)
          (k : int32) ->
        let row = global_idx_y in
        let col = global_idx_x in
        if row < m && col < n then begin
          let sum = mut 0.0 in
          for i = 0 to k - 1l do
            sum :=
              sum
              +. float32_of_float16 a.((row * k) + i)
                 *. float32_of_float16 b.((i * n) + col)
          done ;
          c.((row * n) + col) <- sum
        end]

let k_int =
  ir_of
    [%kernel
      fun (a : int32 vector)
          (b : int32 vector)
          (c : int32 vector)
          (m : int32)
          (n : int32)
          (k : int32) ->
        let row = global_idx_y in
        let col = global_idx_x in
        if row < m && col < n then begin
          let sum = mut 0l in
          for i = 0 to k - 1l do
            sum := sum + (a.((row * k) + i) * b.((i * n) + col))
          done ;
          c.((row * n) + col) <- sum
        end]

let k_f32 =
  ir_of
    [%kernel
      fun (a : float32 vector)
          (b : float32 vector)
          (c : float32 vector)
          (m : int32)
          (n : int32)
          (k : int32) ->
        let row = global_idx_y in
        let col = global_idx_x in
        if row < m && col < n then begin
          let sum = mut 0.0 in
          for i = 0 to k - 1l do
            sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
          done ;
          c.((row * n) + col) <- sum
        end]

let metal ?(simd_width = 32) ?(probe = Ok ()) () : Device.t =
  {
    id = 0;
    backend_id = 0;
    name = "fake Apple GPU";
    framework = "Metal";
    capabilities =
      {
        max_threads_per_block = 1024;
        max_block_dims = (1024, 1024, 64);
        max_grid_dims = (2147483647, 65535, 65535);
        shared_mem_per_block = 32768;
        total_global_mem = 1073741824L;
        compute_capability = (0, 0);
        device_features = [];
        coopmat = Some (Sarek_metal.Metal_coopmat.support ~simd_width ~probe);
        supports_atomics = false;
        warp_size = simd_width;
        max_registers_per_block = 0;
        clock_rate_khz = 0;
        multiprocessor_count = 1;
        is_cpu = false;
      };
  }

let launch ~m ~n ~k : unit -> Sarek_ir_eval_int.launch option =
 fun () ->
  let scalar (v : Sarek_ir_types.var) =
    match v.var_name with
    | "m" -> Some (Int32.of_int m)
    | "n" -> Some (Int32.of_int n)
    | "k" -> Some (Int32.of_int k)
    | _ -> None
  in
  Some
    {
      Sarek_ir_eval_int.grid = ((n + 15) / 16, (m + 15) / 16, 1);
      block = (16, 16, 1);
      scalar;
    }

let choose ?(explicit = true) ?(device = metal ()) ?(dims = (100, 37, 129))
    ?precision ir =
  let m, n, k = dims in
  Spec.choose ?precision ~explicit ~device ~ir ~launch:(launch ~m ~n ~k) ()

(* A device of another framework whose probe advertises 8x8x8 fragments only:
   fragments other than 16x16x16 have been executed on Metal alone. *)
let eight_only framework ~subgroup : Device.t =
  let m = metal () in
  let shape = {C.m = 8; n = 8; k = 8} in
  let cfg a acc =
    {
      C.cfg_shape = shape;
      cfg_a = a;
      cfg_b = a;
      cfg_c = acc;
      cfg_result = acc;
      cfg_saturating = false;
      cfg_scope = C.Subgroup;
    }
  in
  {
    m with
    name = "fake " ^ framework;
    framework;
    capabilities =
      {
        m.capabilities with
        warp_size = subgroup;
        coopmat =
          Some
            {
              C.ds_configs = [cfg C.Uint8 C.Sint32; cfg C.Float16 C.Float32];
              ds_robust_buffer_access = false;
              ds_subgroup_size = subgroup;
              ds_advertised_count = 2;
              ds_scale_add = false;
              ds_evidence = "test: 8x8x8 only";
            };
      };
  }

let declined what ~needle = function
  | Spec.Specialised _ -> Alcotest.failf "%s: rewritten" what
  | Spec.Original why ->
      if not (contains ~needle why) then
        Alcotest.failf "%s: the reason does not name %S: %s" what needle why

let specialised what = function
  | Spec.Specialised s -> s
  | Spec.Original why -> Alcotest.failf "%s: not rewritten (%s)" what why

let test_integer_declined () =
  declined "int32 explicit" ~needle:"integer simdgroup_matrix" (choose k_int) ;
  declined
    "int32 automatic"
    ~needle:"integer simdgroup_matrix"
    (choose ~explicit:false k_int)

let test_float_needs_policy () =
  declined
    "f16 before accept_relaxed_accumulation"
    ~needle:"accept_relaxed_accumulation"
    (choose k_f16)

let shape_of (s : Spec.specialised) = s.config.C.cfg_shape

let test_explicit_admitted () =
  C.accept_relaxed_accumulation ~reason:"test_metal_matrix_rewrite" ;
  let s = specialised "f16 explicit" (choose k_f16) in
  Alcotest.(check bool) "f16 operands" true (s.lowering = Rw.F16_operands) ;
  Alcotest.(check bool)
    "fragment 8x8x8"
    true
    (shape_of s = {C.m = 8; n = 8; k = 8}) ;
  Alcotest.(check bool) "staged" true (s.variant = Spec.Staged) ;
  Alcotest.(check bool) "four waves of 32" true (s.block = (128, 1, 1)) ;
  Alcotest.(check bool) "grid" true (s.grid = (1, 2, 1)) ;
  let aligned = specialised "aligned" (choose ~dims:(256, 256, 256) k_f16) in
  Alcotest.(check bool)
    "the aligned size class stays staged (Direct is laid out for 16x16)"
    true
    (aligned.variant = Spec.Staged) ;
  declined
    "automatic needs its own opt-in"
    ~needle:"accept_automatic_float_rewrite"
    (choose ~explicit:false k_f16)

let test_bf16_declined () =
  C.accept_operand_rounding
    ~precision:C.Operands_bf16
    ~reason:"test_metal_matrix_rewrite" ;
  declined
    "bf16 operands"
    ~needle:"Metal lowers only the f16 operands"
    (choose ~precision:C.Operands_bf16 k_f32)

let test_other_frameworks_stay_16 () =
  let needle = "executed on Metal" in
  declined
    "Vulkan advertising 8x8x8 only"
    ~needle
    (choose ~device:(eight_only "Vulkan" ~subgroup:64) k_int) ;
  declined
    "CUDA/C advertising 8x8x8 only"
    ~needle
    (choose ~device:(eight_only "CUDA/C" ~subgroup:32) k_int) ;
  declined
    "HIP advertising 8x8x8 only"
    ~needle
    (choose ~device:(eight_only "HIP" ~subgroup:32) k_f16)

let test_device_limits () =
  declined
    "probe failed"
    ~needle:"simdgroup_matrix"
    (choose ~device:(metal ~probe:(Error "no simdgroup_matrix") ()) k_f16) ;
  declined
    "16-wide SIMD-groups"
    ~needle:"SIMD-group width is 16"
    (choose ~device:(metal ~simd_width:16 ()) k_f16)

let test_automatic_after_optin () =
  C.accept_automatic_float_rewrite ~reason:"test_metal_matrix_rewrite" ;
  let s = specialised "f16 automatic" (choose ~explicit:false k_f16) in
  Alcotest.(check bool)
    "fragment 8x8x8"
    true
    (shape_of s = {C.m = 8; n = 8; k = 8})

(* The rewritten kernel as MSL: only 8x8 simdgroup types, 4x4 fragments per
   wave (16 accumulators), two chained k-steps of 8 per staged step of 16. *)
let count ~needle s =
  let nl = String.length needle in
  let rec go i acc =
    if i + nl > String.length s then acc
    else go (i + 1) (if String.sub s i nl = needle then acc + 1 else acc)
  in
  go 0 0

let test_msl () =
  let s = specialised "f16 explicit" (choose k_f16) in
  let src =
    Sarek_metal.Sarek_ir_metal.generate_with_types ~block:s.block ~types:[] s.ir
  in
  let n needle = count ~needle src in
  Alcotest.(check int) "accumulators" 16 (n "simdgroup_float8x8 sarek_mm_acc") ;
  Alcotest.(check int)
    "multiply-accumulates: 16 fragments x 2 k-steps"
    32
    (n "simdgroup_multiply_accumulate(") ;
  Alcotest.(check int)
    "operand loads: (4 A + 4 B) x 2 k-steps"
    16
    (n "simdgroup_load(") ;
  Alcotest.(check int) "accumulator stores" 16 (n "simdgroup_store(") ;
  Alcotest.(check bool)
    "no 16-wide statement"
    false
    (contains ~needle:"16x16" src) ;
  Alcotest.(check string)
    "digest of the MSL"
    "f00d2c514416850a3aee7a234852f585"
    (Digest.to_hex (Digest.string src))

let () =
  Alcotest.run
    "metal matrix rewrite"
    [
      ( "admission",
        [
          Alcotest.test_case "integer declined" `Quick test_integer_declined;
          Alcotest.test_case
            "float needs the policy"
            `Quick
            test_float_needs_policy;
          Alcotest.test_case
            "explicit admitted at 8x8x8"
            `Quick
            test_explicit_admitted;
          Alcotest.test_case "bf16 declined" `Quick test_bf16_declined;
          Alcotest.test_case
            "8x8x8 off Metal declined"
            `Quick
            test_other_frameworks_stay_16;
          Alcotest.test_case "device limits" `Quick test_device_limits;
          Alcotest.test_case
            "automatic after the opt-in"
            `Quick
            test_automatic_after_optin;
        ] );
      ("msl", [Alcotest.test_case "generated source" `Quick test_msl]);
    ]
