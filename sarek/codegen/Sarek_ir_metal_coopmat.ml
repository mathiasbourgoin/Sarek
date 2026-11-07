(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Cooperative matrix ([SCoopmat]) for the Metal generator, lowered to MSL
    [simdgroup_matrix] (specs/coopmat-metal-simdgroup.md).

    MSL offers exactly three instantiations, all 8x8 ([half], [float], [bfloat];
    docs/design/f16-relaxed-accuracy.md §7 slice 6, measured on an Apple M4),
    and no integer one. The lowered configuration is therefore the single one
    whose numerics were measured and are a closed form:
    [simdgroup_half8x8 x half8x8 + float8x8 -> float8x8], 8x8x8, bit-equal on
    65536/65536 elements to "accumulator = C, then add the eight products in
    index order, all in binary32". Everything else is refused here, at code
    generation, with the reason.

    Layout is the [transpose_matrix] flag of [simdgroup_load] /
    [simdgroup_store]: with [elements_per_row = stride] and no transpose,
    element (r, c) is at [ptr + r * stride + c]; with it, at
    [ptr + c * stride + r]. This is exactly {!Sarek_coopmat_types.layout}. *)

open Sarek_ir_types
module C = Sarek_coopmat_types

module Codegen_error = Sarek_backend_error.Backend_error.Make (struct
  let name = "Metal"
end)

let unsupported what why =
  Codegen_error.raise_error (Codegen_error.unsupported_construct what why)

(** {1 What the lowering accepts} *)

let shape_8 = {C.m = 8; n = 8; k = 8}

(** The reason integer fragments have no lowering. A plain 8-bit buffer is
    ordinary storage on Metal ([uchar]/[char]); only the integer fragment is
    refused. *)
let no_integer_reason =
  "Metal: there is no integer simdgroup_matrix (MSL's simdgroup_matrix element \
   types are half, float and bfloat), so integer cooperative matrices have no \
   Metal lowering"

(** Refuse, at code generation, a fragment this lowering cannot represent. Total
    over the fragment type. *)
let check_fragment (f : C.fragment) =
  if f.C.frag_shape <> shape_8 then
    unsupported
      "cooperative-matrix shape"
      (Printf.sprintf
         "Metal: only 8x8x8 is lowered (simdgroup_matrix is 8x8 and nothing \
          else); got %s"
         (C.shape_name f.C.frag_shape)) ;
  if f.C.frag_scope <> C.Subgroup then
    unsupported
      "cooperative-matrix scope"
      (Printf.sprintf
         "Metal: only subgroup (SIMD-group) scope exists; got %s"
         (C.scope_name f.C.frag_scope)) ;
  match (f.C.frag_use, f.C.frag_component) with
  | (C.Matrix_a | C.Matrix_b), C.Float16 | C.Accumulator, C.Float32 -> ()
  | _, (C.Uint8 | C.Sint8 | C.Uint32 | C.Sint32) ->
      unsupported "cooperative-matrix fragment" no_integer_reason
  | _, C.BFloat16 ->
      unsupported
        "cooperative-matrix fragment"
        "Metal: bf16 fragments are not lowered (bfloat 8x8 simdgroup_matrix \
         exists but its numerics are unmeasured, and the IR's bf16 fragment \
         loads a float32 buffer rounded to bfloat16)"
  | (C.Matrix_a | C.Matrix_b), C.Float32 ->
      unsupported
        "cooperative-matrix fragment"
        "Metal: f32 operand fragments are not lowered (float 8x8 \
         simdgroup_matrix exists, but its products are not exact in binary32, \
         so no error bound or oracle has been derived for it)"
  | C.Accumulator, C.Float16 ->
      unsupported
        "cooperative-matrix fragment"
        "Metal: an f16 accumulator is not lowered: simdgroup_half8x8 x half8x8 \
         -> half8x8 matches no closed-form model (65520/65536 against \
         sequential binary16 on an Apple M4), so it has no derived rounding \
         (docs/design/f16-relaxed-accuracy.md §5.4)"

(** The configurations the lowering emits a multiply-add for. *)
let check_config (cfg : C.config) =
  let a, b, c, r = C.fragments_of_config cfg in
  List.iter check_fragment [a; b; c; r] ;
  if cfg.C.cfg_saturating then
    unsupported
      "saturating cooperative-matrix accumulation"
      "Metal: simdgroup_multiply_accumulate has no saturating variant"

(** The kernel buffer element type a fragment loads from or stores to. *)
let buffer_elttype (f : C.fragment) =
  match f.C.frag_component with C.Float16 -> TFloat16 | _ -> TFloat32

let elttype_name = function
  | TFloat16 -> "float16"
  | TFloat32 -> "float32"
  | TUint8 -> "uint8"
  | TInt32 -> "int32"
  | TInt64 -> "int64"
  | TFloat64 -> "float64"
  | _ -> "a non-numeric type"

let check_buffer ~buffers ~buf (f : C.fragment) =
  match List.assoc_opt buf buffers with
  | None ->
      unsupported
        "cooperative-matrix buffer"
        (Printf.sprintf
           "%s is not declared as an array parameter or a shared array of the \
            kernel; simdgroup_load/store address device or threadgroup memory \
            only"
           buf)
  | Some elt ->
      let want = buffer_elttype f in
      if elt <> want then
        unsupported
          "cooperative-matrix buffer"
          (Printf.sprintf
             "a %s %s fragment needs a %s buffer, but %s holds %s"
             (C.component_name f.C.frag_component)
             (C.use_name f.C.frag_use)
             (elttype_name want)
             buf
             (elttype_name elt))

(** {!Sarek_ir_types.CM_scale_add} is not lowered on Metal
    (docs/design/coopmat-scaled-accumulate.md §3): an integer source for the
    same reason as every integer fragment, an f32 source because the
    [simdgroup_matrix] element map is not exposed, so the only route is a
    threadgroup round trip, not written yet. *)
let refuse_scale_add (src_frag : C.fragment) =
  if C.component_is_integer src_frag.C.frag_component then
    unsupported "CM_scale_add" no_integer_reason
  else
    unsupported
      "CM_scale_add"
      "Metal: CM_scale_add with an f32 source is not lowered: simdgroup_matrix \
       does not expose which lane holds which element, so it needs a \
       threadgroup-memory round trip (store, scalar scaling, load), which has \
       not been written or run"

let check_op ~buffers = function
  | CM_decl {frag; _} -> check_fragment frag
  | CM_load {frag; src; _} ->
      check_fragment frag ;
      check_buffer ~buffers ~buf:src frag
  | CM_store {frag; dst; _} ->
      check_fragment frag ;
      if frag.C.frag_use <> C.Accumulator then
        unsupported
          "cooperative-matrix store"
          "Metal: only an accumulator fragment is stored" ;
      check_buffer ~buffers ~buf:dst frag
  | CM_muladd {cfg; _} -> check_config cfg
  | CM_scale_add {src_frag; _} -> refuse_scale_add src_frag

(** Whole-kernel check, run before any source is emitted: every fragment,
    configuration and buffer of every cooperative-matrix statement. *)
let check_kernel (k : kernel) =
  let buffers = Sarek_ir_cuda_coopmat.kernel_buffers k in
  let ops = Sarek_ir_analysis.kernel_coopmat_ops k in
  List.iter (check_op ~buffers) ops

(** {1 Statements} *)

let msl_elt = function C.Float16 -> "half" | _ -> "float"

let msl_type (f : C.fragment) =
  Printf.sprintf "simdgroup_%s8x8" (msl_elt f.C.frag_component)

let transpose = function C.Row_major -> "false" | C.Col_major -> "true"

(** [gen_op ~indent ~expr_str op] is the MSL for one statement. [expr_str]
    renders an index or stride expression. *)
let gen_op ~indent ~expr_str op =
  match op with
  | CM_decl {name; frag} ->
      Printf.sprintf
        "%s%s %s = make_filled_simdgroup_matrix<%s, 8, 8>(0);\n"
        indent
        (msl_type frag)
        name
        (msl_elt frag.C.frag_component)
  | CM_load {dst; src; index; stride; layout; _} ->
      Printf.sprintf
        "%ssimdgroup_load(%s, (%s + (%s)), (ulong)(%s), ulong2(0, 0), %s);\n"
        indent
        dst
        src
        (expr_str index)
        (expr_str stride)
        (transpose layout)
  | CM_store {src; dst; index; stride; layout; _} ->
      Printf.sprintf
        "%ssimdgroup_store(%s, (%s + (%s)), (ulong)(%s), ulong2(0, 0), %s);\n"
        indent
        src
        dst
        (expr_str index)
        (expr_str stride)
        (transpose layout)
  | CM_muladd {dst; a; b; c; _} ->
      Printf.sprintf
        "%ssimdgroup_multiply_accumulate(%s, %s, %s, %s);\n"
        indent
        dst
        a
        b
        c
  | CM_scale_add {src_frag; _} -> refuse_scale_add src_frag

(** {1 Launch shape} *)

(** A [simdgroup_matrix] operation needs every lane of its SIMD-group active, so
    the block must be whole 32-wide SIMD-groups; refused before any compile. *)
let check_block (k : kernel) (block : int * int * int) =
  let bx, by, bz = block in
  let threads = bx * by * bz in
  if threads mod 32 <> 0 && Sarek_ir_analysis.kernel_has_coopmat_op k then
    unsupported
      "cooperative-matrix launch shape"
      (Printf.sprintf
         "Metal: a cooperative-matrix kernel must run whole 32-wide \
          SIMD-groups, but the block has %d threads (%dx%dx%d)"
         threads
         bx
         by
         bz)
