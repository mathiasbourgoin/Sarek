(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Host-only tests for {!Sarek_coopmat} (backlog-62, slices 2 and 4b).

    No device, no Vulkan, no GPU. The device-side observation lives in
    [sarek-vulkan/test/test_vulkan_coopmat_capability.ml], which can only run
    where there is a Vulkan device; these are the cases that must hold
    everywhere, and in particular they are where the REFUSAL branches are
    exercised deterministically rather than at the mercy of whichever hardware
    the runner happens to have. *)

open Sarek_coopmat

let shape16 = {m = 16; n = 16; k = 16}

let shape8 = {m = 8; n = 8; k = 8}

let f16_f32_config =
  {
    cfg_shape = shape16;
    cfg_a = Float16;
    cfg_b = Float16;
    cfg_c = Float32;
    cfg_result = Float32;
    cfg_saturating = false;
    cfg_scope = Subgroup;
  }

let f16_f16_config = {f16_f32_config with cfg_c = Float16; cfg_result = Float16}

let s8_s32_config =
  {
    cfg_shape = shape16;
    cfg_a = Sint8;
    cfg_b = Sint8;
    cfg_c = Sint32;
    cfg_result = Sint32;
    cfg_saturating = false;
    cfg_scope = Subgroup;
  }

let u8_u32_config =
  {
    s8_s32_config with
    cfg_a = Uint8;
    cfg_b = Uint8;
    cfg_c = Uint32;
    cfg_result = Uint32;
  }

let s8_s32_saturating = {s8_s32_config with cfg_saturating = true}

(* The fourteen configurations measured on the RX 7900 XTX (RADV NAVI31), radv
   / Mesa 26.1.4-arch3.1, in docs/design/f16-relaxed-accuracy.md §4. Written out
   rather than generated so that a change to the local hardware record is a
   visible diff here. *)
let navi31_configs =
  let int_cfg a b c saturating =
    {
      cfg_shape = shape16;
      cfg_a = a;
      cfg_b = b;
      cfg_c = c;
      cfg_result = c;
      cfg_saturating = saturating;
      cfg_scope = Subgroup;
    }
  in
  let operand_pairs =
    [(Uint8, Uint8); (Uint8, Sint8); (Sint8, Uint8); (Sint8, Sint8)]
  in
  List.concat_map
    (fun (a, b) ->
      [
        int_cfg a b Uint32 false;
        int_cfg a b Sint32 false;
        int_cfg a b Sint32 true;
      ])
    operand_pairs
  @ [f16_f16_config; f16_f32_config]

let navi31_support =
  {
    ds_configs = navi31_configs;
    ds_robust_buffer_access = true;
    ds_subgroup_size = 64;
    ds_advertised_count = List.length navi31_configs;
  }

(* The Raphael iGPU, same driver, same Mesa: extension not advertised,
   cooperativeMatrix feature false, therefore an empty PROBED list. *)
let raphael_support =
  {
    ds_configs = [];
    ds_robust_buffer_access = false;
    ds_subgroup_size = 64;
    ds_advertised_count = 0;
  }

(** {1 The local hardware record} *)

let test_navi31_record_shape () =
  Alcotest.(check int) "14 configurations" 14 (List.length navi31_configs) ;
  let integer_configs = List.filter accumulation_is_exact navi31_configs in
  (* §4 and §8: twelve of the fourteen are integer, and it is exactly those
     twelve that land under the existing strict contract. If this number ever
     moves, §8's whole argument — that a tensor-core path is reachable with no
     accuracy relaxation at all — has to be re-derived rather than assumed. *)
  Alcotest.(check int)
    "12 of 14 accumulate exactly"
    12
    (List.length integer_configs) ;
  Alcotest.(check int)
    "2 of 14 need the relaxed contract"
    2
    (List.length
       (List.filter (fun c -> regime c = Relaxed_bounded) navi31_configs)) ;
  List.iter
    (fun c ->
      Alcotest.(check string)
        ("all local configurations are 16x16x16: " ^ config_name c)
        "16x16x16"
        (shape_name c.cfg_shape) ;
      Alcotest.(check string)
        ("all local configurations are subgroup scope: " ^ config_name c)
        "subgroup"
        (scope_name c.cfg_scope))
    navi31_configs

(** {1 Exactness — the §8 discriminator} *)

let test_accumulation_exactness () =
  (* SPV_KHR_cooperative_matrix: integer accumulation is exact at the precision
     of the result type, so an integer configuration computes the same function
     as the interpreter and needs no relaxation. *)
  Alcotest.(check bool)
    "s8*s8+s32 is exact"
    true
    (accumulation_is_exact s8_s32_config) ;
  Alcotest.(check bool)
    "u8*u8+u32 is exact"
    true
    (accumulation_is_exact u8_u32_config) ;
  Alcotest.(check bool)
    "saturating s8*s8+s32 is still exact"
    true
    (accumulation_is_exact s8_s32_saturating) ;
  (* Both float configurations are inexact, and for the same reason: the
     specification leaves the ORDER of the k + 1 additions to the
     implementation (§5.1). The f32-accumulate one is inexact even though every
     f16 x f16 PRODUCT is exact in binary32 — reading the operand types would
     have got this wrong, which is why accumulation_is_exact reads cfg_c and
     cfg_result. *)
  Alcotest.(check bool)
    "f16*f16+f32 is NOT exact despite exact products"
    false
    (accumulation_is_exact f16_f32_config) ;
  Alcotest.(check bool)
    "f16*f16+f16 is not exact"
    false
    (accumulation_is_exact f16_f16_config) ;
  Alcotest.(check string)
    "integer regime is strict"
    "strict"
    (regime_name (regime s8_s32_config)) ;
  Alcotest.(check string)
    "float regime is relaxed-bounded"
    "relaxed-bounded"
    (regime_name (regime f16_f32_config))

let test_component_classification () =
  List.iter
    (fun (c, expect_int, name, bits) ->
      Alcotest.(check bool)
        (name ^ " integer?")
        expect_int
        (component_is_integer c) ;
      Alcotest.(check string) (name ^ " name") name (component_name c) ;
      Alcotest.(check int) (name ^ " bits") bits (component_bits c))
    [
      (Float16, false, "f16", 16);
      (Float32, false, "f32", 32);
      (Uint8, true, "u8", 8);
      (Sint8, true, "s8", 8);
      (Uint32, true, "u32", 32);
      (Sint32, true, "s32", 32);
    ]

(** {1 The verdict — and that it can refuse} *)

let non_permitting name v =
  Alcotest.(check bool)
    (name ^ ": does not permit")
    false
    (Sarek_capability.permits v)

let test_unprobed_device_refuses () =
  (* The safety property, inherited from Sarek_capability: an unprobed device
     is refused, not admitted. Positive control below. *)
  let v = verdict ~support:None f16_f32_config in
  non_permitting "unprobed device" v ;
  match v with
  | Sarek_capability.Unknown why ->
      Alcotest.(check bool)
        "the Unknown reason says it was not probed"
        true
        (let re = Str.regexp_string "not probed" in
         try
           ignore (Str.search_forward re why 0) ;
           true
         with Not_found -> false)
  | _ -> Alcotest.fail "an unprobed device must yield Unknown, not Unavailable"

let test_capable_device_permits () =
  (* The positive control for every refusal below: the SAME call on a device
     that does advertise the configuration must permit, or the refusals prove
     nothing about the gate and only that this function always says no. *)
  Alcotest.(check bool)
    "NAVI31 permits f16*f16+f32"
    true
    (Sarek_capability.permits
       (verdict ~support:(Some navi31_support) f16_f32_config)) ;
  Alcotest.(check bool)
    "NAVI31 permits s8*s8+s32"
    true
    (Sarek_capability.permits
       (verdict ~support:(Some navi31_support) s8_s32_config))

let test_device_without_support_refuses () =
  (* The Raphael iGPU shape: probed, and advertises nothing. *)
  let v = verdict ~support:(Some raphael_support) f16_f32_config in
  non_permitting "probed device with no configurations" v ;
  match v with
  | Sarek_capability.Unavailable cap ->
      Alcotest.(check string)
        "capability is named"
        "cooperative-matrix"
        cap.Sarek_capability.cap_name ;
      Alcotest.(check string)
        "and it is Device_optional, not a backend or policy refusal"
        "device-optional"
        (Sarek_capability.kind_name cap.Sarek_capability.cap_kind) ;
      let msg = Sarek_capability.explain ~target:"AMD Ryzen 9 7950X iGPU" cap in
      Alcotest.(check bool)
        "the diagnostic names the target"
        true
        (let re = Str.regexp_string "AMD Ryzen 9 7950X iGPU" in
         try
           ignore (Str.search_forward re msg 0) ;
           true
         with Not_found -> false) ;
      Alcotest.(check bool)
        "the diagnostic names the requested configuration"
        true
        (let re = Str.regexp_string "16x16x16" in
         try
           ignore (Str.search_forward re msg 0) ;
           true
         with Not_found -> false)
  | _ ->
      Alcotest.fail "a probed device advertising nothing must yield Unavailable"

let test_unadvertised_configuration_refuses () =
  (* A capable device still refuses a configuration it does not advertise. This
     is the case that separates "has tensor cores" from "has THIS operation",
     and it is the one a boolean capability model gets wrong. *)
  let f32_operands = {f16_f32_config with cfg_a = Float32; cfg_b = Float32} in
  non_permitting
    "f32*f32 operands on NAVI31"
    (verdict ~support:(Some navi31_support) f32_operands) ;
  non_permitting
    "8x8x8 shape on a 16x16x16-only device"
    (verdict
       ~support:(Some navi31_support)
       {f16_f32_config with cfg_shape = shape8}) ;
  non_permitting
    "workgroup scope on a subgroup-only device"
    (verdict
       ~support:(Some navi31_support)
       {f16_f32_config with cfg_scope = Workgroup}) ;
  (* Saturation is part of the operation, not decoration: NAVI31 advertises
     saturating s8 accumulate but not saturating u8 accumulate. *)
  non_permitting
    "saturating u8*u8+u32 is not advertised"
    (verdict
       ~support:(Some navi31_support)
       {u8_u32_config with cfg_saturating = true}) ;
  Alcotest.(check bool)
    "but saturating s8*s8+s32 is"
    true
    (Sarek_capability.permits
       (verdict ~support:(Some navi31_support) s8_s32_saturating))

let test_find_config () =
  Alcotest.(check bool)
    "unprobed device finds nothing"
    true
    (find_config
       ~support:None
       ~shape:shape16
       ~a:Float16
       ~b:Float16
       ~c:Float32
       ~result:Float32
    = None) ;
  (match
     find_config
       ~support:(Some navi31_support)
       ~shape:shape16
       ~a:Float16
       ~b:Float16
       ~c:Float32
       ~result:Float32
   with
  | Some c ->
      Alcotest.(check string)
        "found the f32-accumulate configuration"
        "16x16x16 f16*f16+f32->f32 subgroup"
        (config_name c)
  | None -> Alcotest.fail "NAVI31 advertises f16*f16+f32") ;
  (* Both a saturating and a non-saturating s8 configuration exist; the
     unqualified request must resolve to the non-saturating one, because that
     is the one that computes an ordinary accumulate. *)
  match
    find_config
      ~support:(Some navi31_support)
      ~shape:shape16
      ~a:Sint8
      ~b:Sint8
      ~c:Sint32
      ~result:Sint32
  with
  | Some c ->
      Alcotest.(check bool)
        "prefers the non-saturating variant"
        false
        c.cfg_saturating
  | None -> Alcotest.fail "NAVI31 advertises s8*s8+s32"

(** {1 The fragment type (slice 4b)} *)

let test_fragment_dims_are_derived () =
  let a, b, c, r = fragments_of_config f16_f32_config in
  Alcotest.(check (pair int int)) "A is m x k" (16, 16) (fragment_dims a) ;
  Alcotest.(check (pair int int)) "B is k x n" (16, 16) (fragment_dims b) ;
  Alcotest.(check (pair int int)) "C is m x n" (16, 16) (fragment_dims c) ;
  Alcotest.(check (pair int int)) "D is m x n" (16, 16) (fragment_dims r) ;
  (* A square shape cannot tell the three roles apart, so the discriminating
     case is a rectangular one. Nothing advertises it locally; that is the
     point — §7 slice 4b makes non-hard-coded dimensions binding precisely
     because the two backends in the plan measured different sizes. *)
  let rect = {f16_f32_config with cfg_shape = {m = 8; n = 32; k = 4}} in
  let a, b, c, _ = fragments_of_config rect in
  Alcotest.(check (pair int int)) "A is 8 x 4" (8, 4) (fragment_dims a) ;
  Alcotest.(check (pair int int)) "B is 4 x 32" (4, 32) (fragment_dims b) ;
  Alcotest.(check (pair int int)) "C is 8 x 32" (8, 32) (fragment_dims c) ;
  Alcotest.(check int) "A has 32 components" 32 (fragment_components a) ;
  Alcotest.(check int) "B has 128 components" 128 (fragment_components b)

let test_fragment_admits_integer_components () =
  (* Slice 4b's binding requirement: the fragment type must carry integer
     component types, not only f16/f32. This is the whole of §8's fallback —
     if the f16 accuracy story stalls, an integer path still lands under the
     strict contract, and only if the type was built for it. *)
  let a, b, c, r = fragments_of_config s8_s32_saturating in
  Alcotest.(check string)
    "A fragment is s8"
    "s8"
    (component_name a.frag_component) ;
  Alcotest.(check string)
    "B fragment is s8"
    "s8"
    (component_name b.frag_component) ;
  Alcotest.(check string)
    "C fragment is s32"
    "s32"
    (component_name c.frag_component) ;
  Alcotest.(check string)
    "D fragment is s32"
    "s32"
    (component_name r.frag_component) ;
  List.iter
    (fun f ->
      Alcotest.(check bool)
        "every fragment of an integer configuration has an integer component"
        true
        (component_is_integer f.frag_component))
    [a; b; c; r] ;
  (* And the mixed-signedness configurations really are distinct values, not
     one configuration seen twice. *)
  let mixed = {s8_s32_config with cfg_a = Uint8} in
  let ma, mb, _, _ = fragments_of_config mixed in
  Alcotest.(check string)
    "mixed A is u8"
    "u8"
    (component_name ma.frag_component) ;
  Alcotest.(check string)
    "mixed B is s8"
    "s8"
    (component_name mb.frag_component)

let test_metal_shape_is_expressible () =
  (* §7 slice 6: 8x8 is the ONLY size MSL offers, so a fragment type that could
     not spell it would already be wrong for a backend whose measurements are
     in the same document. No Metal code exists here; the claim under test is
     only that the TYPE does not exclude it. *)
  let metal_like =
    {
      f16_f32_config with
      cfg_shape = shape8;
      cfg_c = Float32;
      cfg_result = Float32;
    }
  in
  let a, _, _, _ = fragments_of_config metal_like in
  Alcotest.(check (pair int int)) "8x8 A fragment" (8, 8) (fragment_dims a) ;
  Alcotest.(check int) "64 components" 64 (fragment_components a)

(** {2 The subgroup calling convention} *)

let test_components_per_invocation () =
  let a, _, _, _ = fragments_of_config f16_f32_config in
  (* Measured subgroup size on the RX 7900 XTX is 64, not 32. A 16x16 fragment
     over 64 invocations is 4 components each. *)
  (match components_per_invocation ~subgroup_size:64 a with
  | Ok n -> Alcotest.(check int) "16x16 over a 64-wide subgroup" 4 n
  | Error e -> Alcotest.fail e) ;
  (match components_per_invocation ~subgroup_size:32 a with
  | Ok n -> Alcotest.(check int) "16x16 over a 32-wide subgroup" 8 n
  | Error e -> Alcotest.fail e) ;
  (* The failure case is real, not defensive: 256 components have no
     distribution over 24 invocations, and a codegen slice must find that out
     before emitting rather than after. *)
  (match components_per_invocation ~subgroup_size:24 a with
  | Ok n -> Alcotest.failf "24 must not divide 256, got %d per invocation" n
  | Error e ->
      Alcotest.(check bool)
        "the error names both numbers"
        true
        (let has s =
           let re = Str.regexp_string s in
           try
             ignore (Str.search_forward re e 0) ;
             true
           with Not_found -> false
         in
         has "24" && has "256")) ;
  match components_per_invocation ~subgroup_size:0 a with
  | Ok _ -> Alcotest.fail "a zero-wide subgroup must not divide anything"
  | Error _ -> ()

let test_config_fits_subgroup () =
  Alcotest.(check bool)
    "16x16x16 fits a 64-wide subgroup"
    true
    (config_fits_subgroup ~subgroup_size:64 f16_f32_config) ;
  Alcotest.(check bool)
    "8x8x8 fits a 64-wide subgroup exactly once per invocation"
    true
    (config_fits_subgroup
       ~subgroup_size:64
       {f16_f32_config with cfg_shape = shape8}) ;
  Alcotest.(check bool)
    "8x8x8 does NOT fit a 128-wide subgroup"
    false
    (config_fits_subgroup
       ~subgroup_size:128
       {f16_f32_config with cfg_shape = shape8}) ;
  (* A rectangular shape can fit some fragments and not others, which is why
     this checks all four rather than one. A = 8x4 = 32 components divides 32;
     B = 4x32 = 128 divides 32; C = 8x32 = 256 divides 32 — all fit. Widen the
     subgroup to 64 and A (32 components) no longer does. *)
  let rect = {f16_f32_config with cfg_shape = {m = 8; n = 32; k = 4}} in
  Alcotest.(check bool)
    "rectangular fits 32"
    true
    (config_fits_subgroup ~subgroup_size:32 rect) ;
  Alcotest.(check bool)
    "rectangular does not fit 64 — its A fragment is too small"
    false
    (config_fits_subgroup ~subgroup_size:64 rect)

let () =
  let open Alcotest in
  run
    "Sarek_coopmat"
    [
      ( "local hardware record",
        [
          test_case
            "NAVI31: 14 configurations, 12 exact"
            `Quick
            test_navi31_record_shape;
        ] );
      ( "accuracy regime",
        [
          test_case
            "integer accumulate is exact, float is not"
            `Quick
            test_accumulation_exactness;
          test_case
            "component classification"
            `Quick
            test_component_classification;
        ] );
      ( "device verdict",
        [
          test_case
            "unprobed device is refused"
            `Quick
            test_unprobed_device_refuses;
          test_case
            "capable device permits (positive control)"
            `Quick
            test_capable_device_permits;
          test_case
            "probed device with no support is refused"
            `Quick
            test_device_without_support_refuses;
          test_case
            "unadvertised configurations are refused"
            `Quick
            test_unadvertised_configuration_refuses;
          test_case "find_config" `Quick test_find_config;
        ] );
      ( "fragment type",
        [
          test_case
            "dimensions are derived from use and shape"
            `Quick
            test_fragment_dims_are_derived;
          test_case
            "integer component types are admitted"
            `Quick
            test_fragment_admits_integer_components;
          test_case
            "Metal's 8x8 is expressible"
            `Quick
            test_metal_shape_is_expressible;
        ] );
      ( "subgroup ABI",
        [
          test_case
            "components per invocation"
            `Quick
            test_components_per_invocation;
          test_case "config fits subgroup" `Quick test_config_fits_subgroup;
        ] );
    ]
