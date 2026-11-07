(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The HIP cooperative-matrix probe rule (specs/coopmat-hip-wmma.md), driven
   with the facts of devices this machine has (gfx1100, gfx1036) and of devices
   it does not (gfx12, CDNA, a wave64 gfx11). No device needed. *)

module C = Sarek_coopmat
module P = Sarek_hip.Hip_coopmat

let f16 =
  {
    C.cfg_shape = {C.m = 16; n = 16; k = 16};
    cfg_a = C.Float16;
    cfg_b = C.Float16;
    cfg_c = C.Float32;
    cfg_result = C.Float32;
    cfg_saturating = false;
    cfg_scope = C.Subgroup;
  }

let permits s cfg = Sarek_capability.permits (C.verdict ~support:(Some s) cfg)

let test_gfx1100 () =
  let s = P.support ~compute_capability:(11, 0) ~warp_size:32 in
  Alcotest.(check int) "ten configurations" 10 (List.length s.C.ds_configs) ;
  Alcotest.(check int) "count = list" 10 s.C.ds_advertised_count ;
  Alcotest.(check bool) "f16*f16+f32" true (permits s f16) ;
  Alcotest.(check bool)
    "bf16*bf16+f32 (specs/matrix-pattern-rewrite.md)"
    true
    (permits s {f16 with C.cfg_a = C.BFloat16; cfg_b = C.BFloat16}) ;
  Alcotest.(check bool)
    "s8*u8+u32"
    true
    (permits
       s
       {
         f16 with
         C.cfg_a = C.Sint8;
         cfg_b = C.Uint8;
         cfg_c = C.Uint32;
         cfg_result = C.Uint32;
       }) ;
  Alcotest.(check bool)
    "no saturating configuration"
    false
    (permits
       s
       {
         f16 with
         C.cfg_a = C.Sint8;
         cfg_b = C.Sint8;
         cfg_c = C.Sint32;
         cfg_result = C.Sint32;
         cfg_saturating = true;
       }) ;
  Alcotest.(check bool)
    "no f16 accumulate"
    false
    (permits s {f16 with C.cfg_c = C.Float16; cfg_result = C.Float16}) ;
  Alcotest.(check int) "subgroup = warp" 32 s.C.ds_subgroup_size ;
  Alcotest.(check bool)
    "CM_scale_add is lowered (docs/design/coopmat-scaled-accumulate.md)"
    true
    s.C.ds_scale_add

let test_refused () =
  List.iter
    (fun (what, cc, warp) ->
      let s = P.support ~compute_capability:cc ~warp_size:warp in
      Alcotest.(check int)
        (what ^ ": no configuration")
        0
        (List.length s.C.ds_configs) ;
      Alcotest.(check bool) (what ^ ": no CM_scale_add") false s.C.ds_scale_add ;
      match C.verdict ~support:(Some s) f16 with
      | Sarek_capability.Unavailable cap ->
          Alcotest.(check bool)
            (what ^ ": refusal names the HIP rule")
            true
            (Astring.String.is_infix ~affix:"HIP:" cap.Sarek_capability.cap_why)
      | _ ->
          Alcotest.failf "%s must be Unavailable (probed, nothing offered)" what)
    [
      ("gfx1036 iGPU", (10, 3), 32);
      ("gfx1201", (12, 0), 32);
      ("gfx942 CDNA", (9, 4), 64);
      ("gfx11 in wave64", (11, 0), 64);
    ]

let () =
  Alcotest.run
    "hip_coopmat_probe"
    [
      ( "probe",
        [
          Alcotest.test_case "gfx1100 wave32" `Quick test_gfx1100;
          Alcotest.test_case "everything else" `Quick test_refused;
        ] );
    ]
