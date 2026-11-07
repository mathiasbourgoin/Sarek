(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The CUDA-C generator's cooperative-matrix lowering, shared by HIP
   (specs/coopmat-hip-wmma.md): what each arm emits, and every refusal. No
   toolchain needed; test_cuda_nvrtc_gate / test_cuda_coopmat_sass compile the
   CUDA arm and test_coopmat_hip_e2e executes the HIP arm. *)

open Sarek_ir_types
module C = Sarek_coopmat

let cfg ?(shape = {C.m = 16; n = 16; k = 16}) ?(sat = false)
    ?(scope = C.Subgroup) a b c r =
  {
    C.cfg_shape = shape;
    cfg_a = a;
    cfg_b = b;
    cfg_c = c;
    cfg_result = r;
    cfg_saturating = sat;
    cfg_scope = scope;
  }

let elt = function
  | C.Float16 -> TFloat16
  | C.BFloat16 | C.Float32 -> TFloat32
  | C.Uint8 | C.Sint8 -> TUint8
  | C.Uint32 | C.Sint32 -> TInt32

let param name e =
  DParam
    ( {var_name = name; var_id = 0; var_type = TVec e; var_mutable = false},
      Some {arr_elttype = e; arr_memspace = Global} )

(** One tile; [a_buf] overrides the element type of [a]'s buffer and [store_a]
    stores the A fragment instead of D. *)
let tile ?a_buf ?(store_a = false) ?(index = 0) ?(stride = 16) cfg =
  let fa, fb, fc, fd = C.fragments_of_config cfg in
  let i = EConst (CInt32 (Int32.of_int index))
  and s = EConst (CInt32 (Int32.of_int stride)) in
  let load dst frag src =
    SCoopmat
      (CM_load
         {
           dst;
           frag;
           src;
           index = i;
           stride = s;
           layout = Sarek_coopmat_types.Row_major;
         })
  in
  let a_elt = Option.value a_buf ~default:(elt cfg.C.cfg_a) in
  {
    default_kernel with
    kern_name = "cm";
    kern_params =
      [
        param "a" a_elt;
        param "b" (elt cfg.C.cfg_b);
        param "c" (elt cfg.C.cfg_c);
        param "d" (elt cfg.C.cfg_result);
      ];
    kern_body =
      SSeq
        [
          SCoopmat (CM_decl {name = "fa"; frag = fa});
          SCoopmat (CM_decl {name = "fb"; frag = fb});
          SCoopmat (CM_decl {name = "fc"; frag = fc});
          load "fa" fa "a";
          load "fb" fb "b";
          load "fc" fc "c";
          SCoopmat (CM_muladd {dst = "fc"; a = "fa"; b = "fb"; c = "fc"; cfg});
          (if store_a then
             SCoopmat
               (CM_store
                  {
                    src = "fa";
                    frag = fa;
                    dst = "d";
                    index = i;
                    stride = s;
                    layout = Sarek_coopmat_types.Row_major;
                  })
           else
             SCoopmat
               (CM_store
                  {
                    src = "fc";
                    frag = fd;
                    dst = "d";
                    index = i;
                    stride = s;
                    layout = Sarek_coopmat_types.Row_major;
                  }));
        ];
  }

let gen ?block k =
  Sarek_codegen.Sarek_ir_cuda.generate_with_types ?block ~types:[] k

let has src needle = Astring.String.is_infix ~affix:needle src

let check_has src needles =
  List.iter
    (fun n -> Alcotest.(check bool) ("emits " ^ n) true (has src n))
    needles

let refuses ?block what k =
  match gen ?block k with
  | _ -> Alcotest.failf "%s: generation succeeded, expected a refusal" what
  | exception Sarek_backend_error.Backend_error.Backend_error _ -> ()

let f16 = cfg C.Float16 C.Float16 C.Float32 C.Float32

let test_f16 () =
  let src = gen (tile f16) in
  check_has
    src
    [
      "#if defined(__HIP_PLATFORM_AMD__)";
      "sarek_cm_h16 fa = {};";
      "sarek_cm_load_a_f16(fa, a, (0), (16));";
      "sarek_cm_load_b_f16(fb, b, (0), (16));";
      "sarek_cm_load_acc_f32(fc, c, (0), (16));";
      "fc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fa, fb, fc);";
      "sarek_cm_store_acc_f32(fc, d, (0), (16));";
      "#include <mma.h>";
      "nvcuda::wmma::mma_sync(fc, fa, fb, fc);";
      "accept_relaxed_accumulation";
      "!defined(__GFX11__)";
    ]

(* bf16 operands loaded from float32 buffers (specs/matrix-pattern-rewrite.md):
   the HIP arm rounds on load and calls the bf16 builtin; the CUDA arm is a
   named #error (no faithful nvcuda::wmma spelling). *)
let test_bf16 () =
  let bf16 = cfg C.BFloat16 C.BFloat16 C.Float32 C.Float32 in
  let src = gen (tile bf16) in
  check_has
    src
    [
      "sarek_cm_s16 fa = {};";
      "sarek_cm_load_a_bf16(fa, a, (0), (16));";
      "fc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(fa, fb, fc);";
      "lowered on HIP gfx11 only";
      "accept_relaxed_accumulation";
    ] ;
  refuses "bf16 from a float16 buffer" (tile ~a_buf:TFloat16 bf16)

let test_int_signs () =
  List.iter
    (fun (a, b, flags) ->
      let src = gen (tile (cfg a b C.Sint32 C.Sint32)) in
      check_has
        src
        [
          Printf.sprintf
            "fc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(%s, fc, false);"
            flags;
          "sarek_cm_load_a_i8(fa, a, (0), (16));";
          "unsigned char* __restrict__ a";
        ] ;
      Alcotest.(check bool) "no relaxed header" false (has src "accept_relaxed"))
    [
      (C.Uint8, C.Uint8, "false, fa, false, fb");
      (C.Uint8, C.Sint8, "false, fa, true, fb");
      (C.Sint8, C.Uint8, "true, fa, false, fb");
      (C.Sint8, C.Sint8, "true, fa, true, fb");
    ] ;
  let mixed = gen (tile (cfg C.Uint8 C.Sint8 C.Sint32 C.Sint32)) in
  check_has
    mixed
    ["#error \"Sarek cooperative matrix: nvcuda::wmma has no mixed-sign"]

let test_refusals () =
  refuses
    "8x8x8"
    (tile
       (cfg
          ~shape:{C.m = 8; n = 8; k = 8}
          C.Float16
          C.Float16
          C.Float32
          C.Float32)) ;
  refuses "saturating" (tile (cfg ~sat:true C.Sint8 C.Sint8 C.Sint32 C.Sint32)) ;
  refuses "f16 accumulate" (tile (cfg C.Float16 C.Float16 C.Float16 C.Float16)) ;
  refuses
    "workgroup scope"
    (tile (cfg ~scope:C.Workgroup C.Float16 C.Float16 C.Float32 C.Float32)) ;
  refuses "A fragment store" (tile ~store_a:true f16) ;
  refuses "f16 fragment from a float32 buffer" (tile ~a_buf:TFloat32 f16)

let test_block () =
  refuses ~block:(16, 1, 1) "16-thread block" (tile f16) ;
  refuses ~block:(48, 1, 1) "48-thread block" (tile f16) ;
  ignore (gen ~block:(32, 1, 1) (tile f16)) ;
  ignore (gen ~block:(16, 2, 1) (tile f16)) ;
  let plain =
    {
      default_kernel with
      kern_name = "plain";
      kern_params = [param "x" TFloat32];
    }
  in
  let src = gen ~block:(16, 1, 1) plain in
  Alcotest.(check bool)
    "no preamble without coopmat"
    false
    (has src "sarek_cm_")

(* Alignment: constants 0 and 16 are proven (the bare wmma call); a start index
   or stride that is not a multiple of the element count is not, and with a known
   block it is staged through a per-warp shared buffer instead of trapping. *)
let lacks src needle =
  Alcotest.(check bool) ("no " ^ needle) false (has src needle)

let test_alignment () =
  let proven = gen ~block:(32, 1, 1) (tile f16) in
  lacks proven "sarek_cm_require(((" ;
  lacks proven "if (sarek_cm_misaligned(((" ;
  lacks proven "sarek_cm_tiles[" ;
  check_has
    proven
    ["nvcuda::wmma::load_matrix_sync(fa, ((const __half *)(a) + (0)), (16));"] ;
  let staged = gen ~block:(32, 2, 1) (tile ~index:1 ~stride:17 f16) in
  check_has
    staged
    [
      "if (sarek_cm_misaligned(((const __half *)(a) + (1)), (17), 2)) { __half \
       *sarek_cm_t = (__half *)sarek_cm_tile();";
      "sarek_cm_stage_in(sarek_cm_t, ((const __half *)(a) + (1)), (17)); \
       __syncwarp(); nvcuda::wmma::load_matrix_sync(fa, sarek_cm_t, 16);";
      "nvcuda::wmma::store_matrix_sync(sarek_cm_t, fc, 16, \
       nvcuda::wmma::mem_row_major); __syncwarp(); sarek_cm_stage_out(((float \
       *)(d) + (1)), sarek_cm_t, (17), 0); __syncwarp();";
      "__shared__ __align__(32) unsigned char sarek_cm_tiles[2048];";
      "#if !defined(__HIP_PLATFORM_AMD__)";
      "return sarek_cm_tiles + w * 1024;";
    ] ;
  lacks staged "sarek_cm_require(((" ;
  (* without a block no buffer can be sized: the trap, no buffer *)
  let trap = gen (tile ~index:1 ~stride:17 f16) in
  check_has trap ["sarek_cm_require(((const __half *)(a) + (1)), (17), 2);"] ;
  lacks trap "sarek_cm_tiles[" ;
  (* the 8-bit and column-major forms *)
  let u8 =
    gen
      ~block:(32, 1, 1)
      (tile ~index:1 ~stride:17 (cfg C.Uint8 C.Uint8 C.Sint32 C.Sint32))
  in
  check_has
    u8
    ["sarek_cm_stage_in(sarek_cm_t, ((const unsigned char *)(a) + (1)), (17));"] ;
  (* a kernel with no unproven access declares no buffer, whatever the block *)
  lacks (gen ~block:(1024, 1, 1) (tile f16)) "sarek_cm_tiles["

let () =
  Alcotest.run
    "cuda_coopmat_codegen"
    [
      ( "coopmat",
        [
          Alcotest.test_case "f16 both arms" `Quick test_f16;
          Alcotest.test_case "bf16 operands" `Quick test_bf16;
          Alcotest.test_case "integer sign flags" `Quick test_int_signs;
          Alcotest.test_case "refusals" `Quick test_refusals;
          Alcotest.test_case "launch block" `Quick test_block;
          Alcotest.test_case
            "alignment: proof, staging, trap"
            `Quick
            test_alignment;
        ] );
    ]
