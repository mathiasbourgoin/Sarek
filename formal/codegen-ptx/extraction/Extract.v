(** Extract.v — Rocq extraction for the codegen-ptx formal model.
 *
 * Extracts the key data types and translation functions to OCaml, producing
 * an interface that CMBT conformance tests can use to cross-check against
 * the production PTX emitter.
 *
 * Extraction configuration:
 * - [ExtrOcamlNatInt]: maps [nat] to [int].
 * - [ExtrOCamlFloats]: maps [PrimFloat.float] to [Float64.t] (Rocq runtime).
 * - Math [Parameter]s are mapped to Stdlib.Float wrappers.
 * - The extracted .ml files appear in the project root (default extraction
 *   output directory).
 *
 * Extracted modules (in order):
 *   AGpuSemantics.ml  — ptx_val, ir_const, ir_binop, ir_expr,
 *                       agpu_state, agpu_eval_ir, agpu_eval_binop
 *   PtxTypes.ml       — ptx_expr_ast, agpu_eval_ptx, agpu_eval_ptx_binop,
 *                       agpu_eval_ptx_cmp
 *   PtxExprSpec.ml    — emit_ast_expr
 *   PtxStmtSpec.ml    — ir_stmt, ptx_stmt_ast, emit_ast_stmt,
 *                       agpu_exec_ir, agpu_exec_ptx_stmt
 *   PtxKernelSpec.ml  — ir_kernel, ptx_kernel_ast, emit_ast_kernel
 *)

From CodegenPtx Require Import AGpuSemantics.
From CodegenPtx Require Import PtxTypes.
From CodegenPtx Require Import PtxExprSpec.
From CodegenPtx Require Import PtxStmtSpec.
From CodegenPtx Require Import PtxKernelSpec.

From Stdlib Require Import Extraction.
From Stdlib Require Import extraction.ExtrOcamlNatInt.
From Stdlib Require Import extraction.ExtrOCamlFloats.

(* ------------------------------------------------------------------ *)
(** * Math intrinsic [Parameter]s → OCaml Float wrappers              *)
(* ------------------------------------------------------------------ *)

Extract Constant sin_f32 => "Float.sin".
Extract Constant cos_f32 => "Float.cos".
Extract Constant fma_f32 => "Float.fma".
Extract Constant sin_f64 => "Float.sin".
Extract Constant cos_f64 => "Float.cos".
Extract Constant fma_f64 => "Float.fma".

(* ------------------------------------------------------------------ *)
(** * Extraction targets                                               *)
(* ------------------------------------------------------------------ *)

Extraction Library AGpuSemantics.
Extraction Library PtxTypes.
Extraction Library PtxExprSpec.
Extraction Library PtxStmtSpec.
Extraction Library PtxKernelSpec.
