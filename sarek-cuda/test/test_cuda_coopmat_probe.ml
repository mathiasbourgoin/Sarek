(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The CUDA/C cooperative-matrix probe rule (Cuda_coopmat), driven with the
   facts of devices this machine does not have. No device needed. *)

module C = Sarek_coopmat
module P = Sarek_cuda.Cuda_coopmat

let cfg a b acc =
  {
    C.cfg_shape = {C.m = 16; n = 16; k = 16};
    cfg_a = a;
    cfg_b = b;
    cfg_c = acc;
    cfg_result = acc;
    cfg_saturating = false;
    cfg_scope = C.Subgroup;
  }

let f16 = cfg C.Float16 C.Float16 C.Float32

let u8 = cfg C.Uint8 C.Uint8 C.Sint32

let permits s c = Sarek_capability.permits (C.verdict ~support:(Some s) c)

let support ?(device_name = "NVIDIA GeForce RTX 3090") ?(warp_size = 32) cc =
  P.support ~device_name ~compute_capability:cc ~warp_size

let test_sm80 () =
  let s = support (8, 0) in
  Alcotest.(check bool) "f16*f16+f32" true (permits s f16) ;
  Alcotest.(check bool) "u8*u8+s32" true (permits s u8) ;
  Alcotest.(check bool)
    "s8*s8+u32"
    true
    (permits s (cfg C.Sint8 C.Sint8 C.Uint32)) ;
  Alcotest.(check bool)
    "mixed-sign 8-bit is an #error in the CUDA arm"
    false
    (permits s (cfg C.Uint8 C.Sint8 C.Sint32)) ;
  Alcotest.(check bool)
    "bf16 from float32 is an #error in the CUDA arm"
    false
    (permits s (cfg C.BFloat16 C.BFloat16 C.Float32)) ;
  Alcotest.(check int) "subgroup = warp" 32 s.C.ds_subgroup_size ;
  Alcotest.(check int)
    "count = list"
    (List.length s.C.ds_configs)
    s.C.ds_advertised_count

let test_thresholds () =
  Alcotest.(check bool) "sm_70: f16" true (permits (support (7, 0)) f16) ;
  Alcotest.(check bool) "sm_70: no 8-bit" false (permits (support (7, 0)) u8) ;
  Alcotest.(check bool) "sm_72: 8-bit" true (permits (support (7, 2)) u8) ;
  Alcotest.(check bool) "sm_61: nothing" false (permits (support (6, 1)) f16)

let refused what s =
  Alcotest.(check int)
    (what ^ ": no configuration")
    0
    (List.length s.C.ds_configs) ;
  match C.verdict ~support:(Some s) f16 with
  | Sarek_capability.Unavailable cap ->
      Alcotest.(check bool)
        (what ^ ": the refusal names the CUDA/C rule")
        true
        (let why = cap.Sarek_capability.cap_why in
         let rec at i =
           i + 7 <= String.length why
           && (String.sub why i 7 = "CUDA/C:" || at (i + 1))
         in
         at 0)
  | _ -> Alcotest.failf "%s: not refused" what

let test_refused () =
  refused "wave64" (support ~warp_size:64 (8, 0)) ;
  refused
    "ZLUDA (rejects wmma.* and u8 mma)"
    (support ~device_name:"AMD Radeon RX 7900 XTX [ZLUDA]" (8, 8))

(* The launch gate Execute applies to a hand-written SCoopmat kernel is
   [Sarek_coopmat.verdict] over the backend's probe: it admits u8 on a
   non-ZLUDA sm_72 device and refuses it on ZLUDA (by name, or by its 8.8
   capability alone) and on sm_61. *)
let test_gate () =
  Alcotest.(check bool) "sm_72 admits u8" true (permits (support (7, 2)) u8) ;
  Alcotest.(check bool)
    "ZLUDA by name refuses u8"
    false
    (permits (support ~device_name:"gfx1100 [ZLUDA]" (8, 0)) u8) ;
  Alcotest.(check bool)
    "capability 8.8 alone refuses u8"
    false
    (permits (support (8, 8)) u8) ;
  Alcotest.(check bool) "sm_61 refuses u8" false (permits (support (6, 1)) u8)

(* CUDA/PTX reports the same configurations as CUDA/C. *)
let ptx_support ?(device_name = "NVIDIA GeForce RTX 2080 Super")
    ?(warp_size = 32) cc =
  P.support_for ~lowering:P.Ptx ~device_name ~compute_capability:cc ~warp_size

let test_ptx () =
  let s = ptx_support (7, 5) in
  Alcotest.(check bool) "sm_75: f16*f16+f32" true (permits s f16) ;
  Alcotest.(check bool) "sm_75: u8*u8+s32" true (permits s u8) ;
  Alcotest.(check bool)
    "sm_75: s8*s8+u32"
    true
    (permits s (cfg C.Sint8 C.Sint8 C.Uint32)) ;
  Alcotest.(check bool)
    "sm_75: mixed-sign 8-bit is refused"
    false
    (permits s (cfg C.Uint8 C.Sint8 C.Sint32)) ;
  Alcotest.(check bool)
    "sm_70: f16 but no 8-bit"
    false
    (permits (ptx_support (7, 0)) u8) ;
  Alcotest.(check bool) "sm_72: 8-bit" true (permits (ptx_support (7, 2)) u8) ;
  Alcotest.(check bool) "sm_80: 8-bit" true (permits (ptx_support (8, 0)) u8) ;
  Alcotest.(check bool) "sm_70: f16" true (permits (ptx_support (7, 0)) f16) ;
  Alcotest.(check bool)
    "sm_61: nothing"
    false
    (permits (ptx_support (6, 1)) f16) ;
  Alcotest.(check bool)
    "wave64: nothing"
    false
    (permits (ptx_support ~warp_size:64 (8, 0)) f16) ;
  Alcotest.(check bool)
    "ZLUDA: nothing (rejects wmma.*)"
    false
    (permits (ptx_support ~device_name:"gfx1100 [ZLUDA]" (8, 8)) f16) ;
  Alcotest.(check bool)
    "evidence names CUDA/PTX"
    true
    (String.length s.C.ds_evidence >= 9
    && String.sub s.C.ds_evidence 0 9 = "CUDA/PTX:") ;
  Alcotest.(check bool)
    "CUDA/C rule unchanged on sm_75 (still offers 8-bit)"
    true
    (permits (support (7, 5)) u8 && permits (support (7, 5)) f16)

let () =
  Alcotest.run
    "cuda_coopmat_probe"
    [
      ( "rule",
        [
          Alcotest.test_case "sm_80" `Quick test_sm80;
          Alcotest.test_case "thresholds" `Quick test_thresholds;
          Alcotest.test_case "refused" `Quick test_refused;
          Alcotest.test_case "launch gate" `Quick test_gate;
          Alcotest.test_case "CUDA/PTX" `Quick test_ptx;
        ] );
    ]
