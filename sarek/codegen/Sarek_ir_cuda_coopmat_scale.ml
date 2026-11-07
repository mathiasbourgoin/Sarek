(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** [CM_scale_add] for the CUDA-C generator, which HIP shares
    (docs/design/coopmat-scaled-accumulate.md §3).

    {b HIP gfx11.} One call of the preamble's [sarek_cm_scale_add] template
    ({!Sarek_ir_cuda_coopmat_preamble}), in registers: accumulator element [e]
    of lane [l] is row [2e + l / 16], column [l mod 16], the layout the
    accumulator loads and stores already use. The template's flags are the
    statement's ([col], [offset], [accumulate]), so each instance has only the
    steps it needs. Non-contraction rests on hiprtc's [-ffp-contract=off]
    ([Hip_rtc.base_options], appended after any caller option). A
    [#pragma clang fp contract(off)] in the helper would add no guard: clang's
    [-ffp-contract=fast] disregards that pragma (measured with hipcc for
    gfx1100: the helper with the pragma, under [-ffp-contract=fast], has
    [v_fma]). Executed on an RX 7900 XTX, bit-identical to the interpreter
    ([test_coopmat_scale_add_e2e], which also checks the ISA: no fused
    multiply-add under [-ffp-contract=off], one under [-ffp-contract=fast], so
    the check can fail).

    {b CUDA.} Refused with {!cuda_reason}: a named [#error] in the CUDA arm of
    the shared source, as for the other CUDA-only gaps, and the CUDA probes
    report [ds_scale_add = false] so the launch gate refuses first. *)

open Sarek_ir_types
module C = Sarek_coopmat_types

module Codegen_error = Sarek_backend_error.Backend_error.Make (struct
  let name = "CUDA"
end)

let unsupported what why =
  Codegen_error.raise_error (Codegen_error.unsupported_construct what why)

(** Why neither CUDA lowering (the [nvcuda::wmma] arm here, and the [wmma.*]
    instructions of [Sarek_ir_ptx_coopmat]) emits [CM_scale_add]. *)
let cuda_reason =
  "CUDA: CM_scale_add is not lowered. The wmma accumulator fragment's \
   element-to-(row, column) map is not documented, so the per-row and \
   per-column scales cannot be applied in registers; a shared-memory round \
   trip needs a per-warp scratch tile whose size depends on the warps per \
   block, which code generation does not know, and moving the 8-bit lowering \
   to mma.sync (documented layouts) rewrites its loads, stores and products; \
   neither can be executed without an NVIDIA GPU"

(** The CUDA arm of a [CM_scale_add]. *)
let cuda_error =
  Printf.sprintf "\n#error \"Sarek cooperative matrix: %s\"\n" cuda_reason

let check_operands ~dst_frag ~src_frag =
  match Sarek_coopmat.scale_add_operands_verdict ~dst_frag ~src_frag with
  | Sarek_capability.Available -> ()
  | Sarek_capability.Unavailable cap ->
      unsupported "CM_scale_add" ("CUDA/HIP: " ^ cap.Sarek_capability.cap_why)
  | Sarek_capability.Unknown why ->
      unsupported "CM_scale_add" ("CUDA/HIP: " ^ why)

(* A scale buffer whose element type the generator knows must be float32; an
   unknown name is left to the C++ compiler, which rejects any other pointer
   type for the helper's [const float *]. *)
let check_scale_buffer ~buffers (s : coopmat_scale) =
  match List.assoc_opt s.scale_buf buffers with
  | None | Some TFloat32 -> ()
  | Some _ ->
      unsupported
        "CM_scale_add"
        (Printf.sprintf
           "CUDA/HIP: the scale buffer %s must be a float32 buffer"
           s.scale_buf)

(** The HIP call:
    [sarek_cm_scale_add<COL, OFF, ACC>(dst, src, r, (ri), c, (ci), p, (pi), q,
     (qi));], an absent scale passed as [nullptr, 0]. *)
let amd_call ~buffers ~expr_str ~dst ~dst_frag ~src ~src_frag ~row ~col ~offset
    ~accumulate =
  check_operands ~dst_frag ~src_frag ;
  let scales =
    row
    :: (Option.to_list col
       @ match offset with None -> [] | Some o -> [o.off_row; o.off_col])
  in
  List.iter (check_scale_buffer ~buffers) scales ;
  let arg = function
    | None -> "nullptr, 0"
    | Some s -> Printf.sprintf "%s, (%s)" s.scale_buf (expr_str s.scale_index)
  in
  let flag b = if b then "true" else "false" in
  Printf.sprintf
    "sarek_cm_scale_add<%s, %s, %s, %s>(%s, %s, %s, %s, %s, %s);"
    (flag (Option.is_some col))
    (flag (Option.is_some offset))
    (flag accumulate)
    (* The source type is spelled out from [src_frag], never deduced: a
       fragment declared as another type is then a compile error, as with the
       other helpers, instead of a silent reinterpretation of its lanes. *)
    (if src_frag.C.frag_component = C.Float32 then "sarek_cm_f8"
     else "sarek_cm_i8")
    dst
    src
    (arg (Some row))
    (arg col)
    (arg (Option.map (fun o -> o.off_row) offset))
    (arg (Option.map (fun o -> o.off_col) offset))
