open AGpuSemantics
open PtxTypes

(** val is_cmp_op : ir_binop -> bool **)

let is_cmp_op = function
  | Eq -> true
  | Ne -> true
  | Lt -> true
  | Le -> true
  | Gt -> true
  | Ge -> true
  | _ -> false

(** val ir_binop_to_ptx_binop : ir_binop -> ptx_binop_tag **)

let ir_binop_to_ptx_binop = function
  | Sub -> PSub
  | Mul -> PMul
  | Div -> PDiv
  | Mod -> PMod
  | And -> PAnd
  | Or -> POr
  | Shl -> PShl
  | Shr -> PShr
  | BitAnd -> PBitAnd
  | BitOr -> PBitOr
  | BitXor -> PBitXor
  | _ -> PAdd

(** val ir_binop_to_ptx_cmp : ir_binop -> ptx_cmp_tag **)

let ir_binop_to_ptx_cmp = function
  | Ne -> PNe
  | Lt -> PLt
  | Le -> PLe
  | Gt -> PGt
  | Ge -> PGe
  | _ -> PEq

(** val emit_ast_expr : ir_expr -> ptx_expr_ast **)

let rec emit_ast_expr = function
  | IEConst i -> (
      match i with
      | CInt32 n -> PtxLitU32 n
      | CInt64 n -> PtxLitU64 n
      | CFloat32 f -> PtxLitF32 f
      | CFloat64 f -> PtxLitF64 f
      | CBool b -> PtxLitU32 (bool_to_nat b)
      | CUnit -> PtxLitU32 0)
  | IEVar name -> PtxReg name
  | IEBinop (op, e1, e2) ->
      if is_cmp_op op then
        PtxCmp (ir_binop_to_ptx_cmp op, emit_ast_expr e1, emit_ast_expr e2)
      else
        PtxBinop (ir_binop_to_ptx_binop op, emit_ast_expr e1, emit_ast_expr e2)
  | IEArrayRead (i, base_e, idx_e) -> (
      match i with
      | MS_Global ->
          PtxGlobalRead
            (PtxBinop (PAdd, emit_ast_expr base_e, emit_ast_expr idx_e))
      | MS_Shared ->
          PtxSharedRead
            (PtxBinop (PAdd, emit_ast_expr base_e, emit_ast_expr idx_e)))
  | IEThreadIdxX -> PtxTidx
  | IEBlockIdxX -> PtxBidx
  | IEBlockDimX -> PtxBdim
  | IEGlobalIdx -> PtxBinop (PAdd, PtxBinop (PMul, PtxBidx, PtxBdim), PtxTidx)
  | IEBarrier -> PtxLitU32 0
  | IESin32 e1 -> PtxIntrinsic (PISin32, emit_ast_expr e1)
  | IECos32 e1 -> PtxIntrinsic (PICos32, emit_ast_expr e1)
  | IESqrt32 e1 -> PtxIntrinsic (PISqrt32, emit_ast_expr e1)
  | IEFabs32 e1 -> PtxIntrinsic (PIFabs32, emit_ast_expr e1)
  | IEFma32 (ea, eb, ec) ->
      PtxFma32 (emit_ast_expr ea, emit_ast_expr eb, emit_ast_expr ec)
  | IESin64 e1 -> PtxIntrinsic (PISin64, emit_ast_expr e1)
  | IECos64 e1 -> PtxIntrinsic (PICos64, emit_ast_expr e1)
  | IESqrt64 e1 -> PtxIntrinsic (PISqrt64, emit_ast_expr e1)
  | IEFabs64 e1 -> PtxIntrinsic (PIFabs64, emit_ast_expr e1)
  | IEFma64 (ea, eb, ec) ->
      PtxFma64 (emit_ast_expr ea, emit_ast_expr eb, emit_ast_expr ec)
