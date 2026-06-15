open AGpuSemantics
open Datatypes
open Nat
open PrimFloat
open String

type ptx_type =
| PTX_U32
| PTX_U64
| PTX_F32
| PTX_F64
| PTX_Pred

val ptx_type_rect : 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> ptx_type -> 'a1

val ptx_type_rec : 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> ptx_type -> 'a1

type elttype =
| ET_Int32
| ET_Int64
| ET_Float32
| ET_Float64
| ET_Bool
| ET_Unit
| ET_Vec of elttype
| ET_Array of elttype

val elttype_rect :
  'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> (elttype -> 'a1 -> 'a1) ->
  (elttype -> 'a1 -> 'a1) -> elttype -> 'a1

val elttype_rec :
  'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> (elttype -> 'a1 -> 'a1) ->
  (elttype -> 'a1 -> 'a1) -> elttype -> 'a1

type ptx_binop_tag =
| PAdd
| PSub
| PMul
| PDiv
| PMod
| PAnd
| POr
| PShl
| PShr
| PBitAnd
| PBitOr
| PBitXor

val ptx_binop_tag_rect :
  'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1
  -> 'a1 -> ptx_binop_tag -> 'a1

val ptx_binop_tag_rec :
  'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1
  -> 'a1 -> ptx_binop_tag -> 'a1

type ptx_cmp_tag =
| PEq
| PNe
| PLt
| PLe
| PGt
| PGe

val ptx_cmp_tag_rect :
  'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> ptx_cmp_tag -> 'a1

val ptx_cmp_tag_rec :
  'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> ptx_cmp_tag -> 'a1

type ptx_intrinsic_tag =
| PISin32
| PICos32
| PISqrt32
| PIFabs32
| PISin64
| PICos64
| PISqrt64
| PIFabs64
| PIFma

val ptx_intrinsic_tag_rect :
  'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 ->
  ptx_intrinsic_tag -> 'a1

val ptx_intrinsic_tag_rec :
  'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 ->
  ptx_intrinsic_tag -> 'a1

type ptx_expr_ast =
| PtxLitU32 of int
| PtxLitU64 of int
| PtxLitF32 of Float64.t
| PtxLitF64 of Float64.t
| PtxReg of string
| PtxBinop of ptx_binop_tag * ptx_expr_ast * ptx_expr_ast
| PtxCmp of ptx_cmp_tag * ptx_expr_ast * ptx_expr_ast
| PtxGlobalRead of ptx_expr_ast
| PtxSharedRead of ptx_expr_ast
| PtxTidx
| PtxBidx
| PtxBdim
| PtxIntrinsic of ptx_intrinsic_tag * ptx_expr_ast
| PtxFma32 of ptx_expr_ast * ptx_expr_ast * ptx_expr_ast
| PtxFma64 of ptx_expr_ast * ptx_expr_ast * ptx_expr_ast

val ptx_expr_ast_rect :
  (int -> 'a1) -> (int -> 'a1) -> (Float64.t -> 'a1) -> (Float64.t -> 'a1) ->
  (string -> 'a1) -> (ptx_binop_tag -> ptx_expr_ast -> 'a1 -> ptx_expr_ast ->
  'a1 -> 'a1) -> (ptx_cmp_tag -> ptx_expr_ast -> 'a1 -> ptx_expr_ast -> 'a1
  -> 'a1) -> (ptx_expr_ast -> 'a1 -> 'a1) -> (ptx_expr_ast -> 'a1 -> 'a1) ->
  'a1 -> 'a1 -> 'a1 -> (ptx_intrinsic_tag -> ptx_expr_ast -> 'a1 -> 'a1) ->
  (ptx_expr_ast -> 'a1 -> ptx_expr_ast -> 'a1 -> ptx_expr_ast -> 'a1 -> 'a1)
  -> (ptx_expr_ast -> 'a1 -> ptx_expr_ast -> 'a1 -> ptx_expr_ast -> 'a1 ->
  'a1) -> ptx_expr_ast -> 'a1

val ptx_expr_ast_rec :
  (int -> 'a1) -> (int -> 'a1) -> (Float64.t -> 'a1) -> (Float64.t -> 'a1) ->
  (string -> 'a1) -> (ptx_binop_tag -> ptx_expr_ast -> 'a1 -> ptx_expr_ast ->
  'a1 -> 'a1) -> (ptx_cmp_tag -> ptx_expr_ast -> 'a1 -> ptx_expr_ast -> 'a1
  -> 'a1) -> (ptx_expr_ast -> 'a1 -> 'a1) -> (ptx_expr_ast -> 'a1 -> 'a1) ->
  'a1 -> 'a1 -> 'a1 -> (ptx_intrinsic_tag -> ptx_expr_ast -> 'a1 -> 'a1) ->
  (ptx_expr_ast -> 'a1 -> ptx_expr_ast -> 'a1 -> ptx_expr_ast -> 'a1 -> 'a1)
  -> (ptx_expr_ast -> 'a1 -> ptx_expr_ast -> 'a1 -> ptx_expr_ast -> 'a1 ->
  'a1) -> ptx_expr_ast -> 'a1

val ptx_reg_type_of : elttype -> ptx_type

val agpu_eval_ptx_binop :
  ptx_binop_tag -> ptx_val -> ptx_val -> ptx_val option

val agpu_eval_ptx_cmp : ptx_cmp_tag -> ptx_val -> ptx_val -> ptx_val option

val agpu_eval_ptx_intrinsic : ptx_intrinsic_tag -> ptx_val -> ptx_val option

val agpu_eval_ptx :
  agpu_state -> ptx_expr_ast -> (ptx_val * agpu_state) option
