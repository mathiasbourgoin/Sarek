(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The [layout] of [CM_load] / [CM_store], COMPILE-ONLY: the pure source
 * generators run, no device is created. The HIP arm is executed by
 * sarek/tests/e2e/test_coopmat_layout_e2e.ml, the GLSL one on RADV by the
 * same test; the CUDA arm has no hardware here.
 *
 *   1. CUDA/HIP: a column-major operand takes the other access-pattern helper
 *      (column-major B = the contiguous [load_a] helper), a column-major
 *      accumulator the [acc_cm] helpers; the CUDA arm puts [col_major] in the
 *      operand's fragment type and [mem_col_major] on accumulator loads and
 *      stores; an operand loaded in both layouts is a named #error on CUDA.
 *   2. Every helper takes a 64-bit stride.
 *   3. GLSL: [gl_CooperativeMatrixLayoutColumnMajor].
 *   4. OpenCL, Metal and WGSL refuse a column-major load.
 *   5. PTX lowers it: wmma.load / wmma.store take the layout.
 ******************************************************************************)

open Sarek_ir_types
open Sarek_codegen
module C = Sarek_coopmat_types

let contains ~needle haystack =
  let nl = String.length needle and hl = String.length haystack in
  let rec go i =
    i + nl <= hl && (String.sub haystack i nl = needle || go (i + 1))
  in
  nl = 0 || go 0

let check_contains name ~needle out =
  if not (contains ~needle out) then
    Alcotest.failf "%s: expected %S in the output:\n%s" name needle out

let check_absent name ~needle out =
  if contains ~needle out then
    Alcotest.failf "%s: unexpected %S in the output:\n%s" name needle out

let cfg a acc =
  {
    C.cfg_shape = {C.m = 16; n = 16; k = 16};
    cfg_a = a;
    cfg_b = a;
    cfg_c = acc;
    cfg_result = acc;
    cfg_saturating = false;
    cfg_scope = C.Subgroup;
  }

let f16 = cfg C.Float16 C.Float32

let u8 = cfg C.Uint8 C.Sint32

let elt = function
  | C.Float16 -> TFloat16
  | C.BFloat16 | C.Float32 -> TFloat32
  | C.Uint8 | C.Sint8 -> TUint8
  | C.Uint32 | C.Sint32 -> TInt32

let param name e =
  DParam
    ( {var_name = name; var_id = 0; var_type = TVec e; var_mutable = false},
      Some {arr_elttype = e; arr_memspace = Global} )

(** [d = a * b + c]; [lb2] loads B a second time in that layout. *)
let tile ?lb2 ~la ~lb ~lc ~ld cfg =
  let fa, fb, fc, fd = C.fragments_of_config cfg in
  let i = EConst (CInt32 0l) and s = EConst (CInt32 16l) in
  let load dst frag src layout =
    SCoopmat (CM_load {dst; frag; src; index = i; stride = s; layout})
  in
  {
    default_kernel with
    kern_name = "cm_layout";
    kern_params =
      [
        param "a" (elt cfg.C.cfg_a);
        param "b" (elt cfg.C.cfg_b);
        param "c" (elt cfg.C.cfg_c);
        param "d" (elt cfg.C.cfg_result);
      ];
    kern_body =
      SSeq
        ([
           SCoopmat (CM_decl {name = "fa"; frag = fa});
           SCoopmat (CM_decl {name = "fb"; frag = fb});
           SCoopmat (CM_decl {name = "fc"; frag = fc});
           load "fa" fa "a" la;
           load "fb" fb "b" lb;
         ]
        @ Option.to_list (Option.map (load "fb" fb "b") lb2)
        @ [
            load "fc" fc "c" lc;
            SCoopmat (CM_muladd {dst = "fc"; a = "fa"; b = "fb"; c = "fc"; cfg});
            SCoopmat
              (CM_store
                 {
                   src = "fc";
                   frag = fd;
                   dst = "d";
                   index = i;
                   stride = s;
                   layout = ld;
                 });
          ]);
  }

let r = C.Row_major

let c = C.Col_major

let cuda k = Sarek_ir_cuda.generate_with_types ~types:[] k

(* ------------------------------------------------------------------ *)
(* 1. CUDA / HIP                                                      *)
(* ------------------------------------------------------------------ *)

let test_cuda_col_b () =
  let src = cuda (tile ~la:r ~lb:c ~lc:r ~ld:r f16) in
  List.iter
    (fun needle -> check_contains "col-major B" ~needle src)
    [
      "sarek_cm_load_a_f16(fa, a, (0), (16));";
      "sarek_cm_load_a_f16(fb, b, (0), (16));";
      "nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, \
       nvcuda::wmma::col_major> fb;";
      "nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, \
       nvcuda::wmma::row_major> fa;";
    ] ;
  check_absent "col-major B" ~needle:"sarek_cm_load_b_f16(fb" src

let test_cuda_col_a () =
  let src = cuda (tile ~la:c ~lb:r ~lc:r ~ld:r u8) in
  List.iter
    (fun needle -> check_contains "col-major A" ~needle src)
    [
      "sarek_cm_load_b_i8(fa, a, (0), (16));";
      "sarek_cm_load_b_i8(fb, b, (0), (16));";
      "nvcuda::wmma::matrix_a, 16, 16, 16, unsigned char, \
       nvcuda::wmma::col_major> fa;";
    ]

let test_cuda_col_acc () =
  let src = cuda (tile ~la:r ~lb:r ~lc:c ~ld:c f16) in
  List.iter
    (fun needle -> check_contains "col-major accumulator" ~needle src)
    [
      "sarek_cm_load_acc_cm_f32(fc, c, (0), (16));";
      "sarek_cm_store_acc_cm_f32(fc, d, (0), (16));";
      "nvcuda::wmma::load_matrix_sync(fc, ((const float *)(c) + (0)), (16), \
       nvcuda::wmma::mem_col_major);";
      "(16), nvcuda::wmma::mem_col_major);";
    ] ;
  let src = cuda (tile ~la:r ~lb:r ~lc:r ~ld:c u8) in
  check_contains
    "int accumulator stored column-major"
    ~needle:"sarek_cm_store_acc_cm_i32(fc, d, (0), (16));"
    src ;
  check_contains
    "int accumulator loaded row-major"
    ~needle:"sarek_cm_load_acc_i32(fc, c, (0), (16));"
    src

let test_cuda_row_unchanged () =
  let src = cuda (tile ~la:r ~lb:r ~lc:r ~ld:r f16) in
  List.iter
    (fun needle -> check_contains "row-major" ~needle src)
    [
      "sarek_cm_load_a_f16(fa, a, (0), (16));";
      "sarek_cm_load_b_f16(fb, b, (0), (16));";
      "sarek_cm_load_acc_f32(fc, c, (0), (16));";
      "sarek_cm_store_acc_f32(fc, d, (0), (16));";
      "__half, nvcuda::wmma::row_major> fb;";
      "nvcuda::wmma::mem_row_major);";
    ] ;
  check_absent "row-major" ~needle:"col_major" src

let test_cuda_both_layouts () =
  let src = cuda (tile ~lb2:c ~la:r ~lb:r ~lc:r ~ld:r f16) in
  check_contains
    "B in both layouts: HIP arm"
    ~needle:"sarek_cm_load_a_f16(fb, b, (0), (16));"
    src ;
  check_contains
    "B in both layouts: CUDA arm"
    ~needle:"fb is loaded both row- and column-major"
    src

(* ------------------------------------------------------------------ *)
(* 2. 64-bit stride                                                   *)
(* ------------------------------------------------------------------ *)

let test_wide_stride () =
  let src = cuda (tile ~la:r ~lb:r ~lc:r ~ld:r f16) in
  check_absent "helpers" ~needle:"int i, int s)" src ;
  check_contains
    "helpers"
    ~needle:
      "sarek_cm_load_b_f16(sarek_cm_h16 &f, const __half *p, int i, long long \
       s)"
    src

(* ------------------------------------------------------------------ *)
(* 3. GLSL                                                            *)
(* ------------------------------------------------------------------ *)

let test_glsl () =
  let src =
    Sarek_ir_glsl.generate_with_types
      ~types:[]
      (tile ~la:r ~lb:c ~lc:r ~ld:c u8)
  in
  List.iter
    (fun needle -> check_contains "GLSL" ~needle src)
    [
      "coopMatLoad(fa, a, 0, 16, gl_CooperativeMatrixLayoutRowMajor);";
      "coopMatLoad(fb, b, 0, 16, gl_CooperativeMatrixLayoutColumnMajor);";
      "coopMatStore(fc, d, 0, 16, gl_CooperativeMatrixLayoutColumnMajor);";
    ]

(* ------------------------------------------------------------------ *)
(* 4. Refusals                                                        *)
(* ------------------------------------------------------------------ *)

let test_refusals () =
  let k = tile ~la:c ~lb:c ~lc:c ~ld:c u8 in
  let refuse name gen =
    match gen k with
    | (_ : string) -> Alcotest.failf "%s emitted a column-major load" name
    | exception Sarek_backend_error.Backend_error.Backend_error _ -> ()
    | exception Sarek_ir_ptx_types.Ptx_codegen_error _ -> ()
  in
  refuse "OpenCL" (Sarek_ir_opencl.generate_with_types ~types:[]) ;
  refuse "Metal" (Sarek_ir_metal.generate_with_types ~types:[]) ;
  refuse "WGSL" (Sarek_ir_wgsl.generate_with_types ~types:[])

(* PTX takes the layout as a [.row] / [.col] qualifier of every wmma access. *)
let test_ptx_layouts () =
  let k = tile ~la:c ~lb:c ~lc:c ~ld:c u8 in
  let src = Sarek_ir_ptx.generate_with_types ~types:[] k in
  check_contains
    "PTX"
    ~needle:"wmma.load.b.sync.aligned.col.m16n16k16.global.u8"
    src ;
  check_contains
    "PTX"
    ~needle:"wmma.store.d.sync.aligned.col.m16n16k16.global.s32"
    src

let () =
  Alcotest.run
    "coopmat_layout_codegen"
    [
      ( "cuda-hip",
        [
          Alcotest.test_case "column-major B" `Quick test_cuda_col_b;
          Alcotest.test_case "column-major A" `Quick test_cuda_col_a;
          Alcotest.test_case "column-major accumulator" `Quick test_cuda_col_acc;
          Alcotest.test_case
            "row-major unchanged"
            `Quick
            test_cuda_row_unchanged;
          Alcotest.test_case "both layouts" `Quick test_cuda_both_layouts;
          Alcotest.test_case "64-bit stride" `Quick test_wide_stride;
        ] );
      ("glsl", [Alcotest.test_case "column-major" `Quick test_glsl]);
      ("refusals", [Alcotest.test_case "no coopmat" `Quick test_refusals]);
      ("ptx", [Alcotest.test_case "column-major" `Quick test_ptx_layouts]);
    ]
