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

(** val ptx_type_rect : 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> ptx_type -> 'a1 **)

let ptx_type_rect f f0 f1 f2 f3 = function
| PTX_U32 -> f
| PTX_U64 -> f0
| PTX_F32 -> f1
| PTX_F64 -> f2
| PTX_Pred -> f3

(** val ptx_type_rec : 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> ptx_type -> 'a1 **)

let ptx_type_rec f f0 f1 f2 f3 = function
| PTX_U32 -> f
| PTX_U64 -> f0
| PTX_F32 -> f1
| PTX_F64 -> f2
| PTX_Pred -> f3

type elttype =
| ET_Int32
| ET_Int64
| ET_Float32
| ET_Float64
| ET_Bool
| ET_Unit
| ET_Vec of elttype
| ET_Array of elttype

(** val elttype_rect :
    'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> (elttype -> 'a1 -> 'a1) ->
    (elttype -> 'a1 -> 'a1) -> elttype -> 'a1 **)

let rec elttype_rect f f0 f1 f2 f3 f4 f5 f6 = function
| ET_Int32 -> f
| ET_Int64 -> f0
| ET_Float32 -> f1
| ET_Float64 -> f2
| ET_Bool -> f3
| ET_Unit -> f4
| ET_Vec e0 -> f5 e0 (elttype_rect f f0 f1 f2 f3 f4 f5 f6 e0)
| ET_Array e0 -> f6 e0 (elttype_rect f f0 f1 f2 f3 f4 f5 f6 e0)

(** val elttype_rec :
    'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> (elttype -> 'a1 -> 'a1) ->
    (elttype -> 'a1 -> 'a1) -> elttype -> 'a1 **)

let rec elttype_rec f f0 f1 f2 f3 f4 f5 f6 = function
| ET_Int32 -> f
| ET_Int64 -> f0
| ET_Float32 -> f1
| ET_Float64 -> f2
| ET_Bool -> f3
| ET_Unit -> f4
| ET_Vec e0 -> f5 e0 (elttype_rec f f0 f1 f2 f3 f4 f5 f6 e0)
| ET_Array e0 -> f6 e0 (elttype_rec f f0 f1 f2 f3 f4 f5 f6 e0)

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

(** val ptx_binop_tag_rect :
    'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1
    -> 'a1 -> ptx_binop_tag -> 'a1 **)

let ptx_binop_tag_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 = function
| PAdd -> f
| PSub -> f0
| PMul -> f1
| PDiv -> f2
| PMod -> f3
| PAnd -> f4
| POr -> f5
| PShl -> f6
| PShr -> f7
| PBitAnd -> f8
| PBitOr -> f9
| PBitXor -> f10

(** val ptx_binop_tag_rec :
    'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1
    -> 'a1 -> ptx_binop_tag -> 'a1 **)

let ptx_binop_tag_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 = function
| PAdd -> f
| PSub -> f0
| PMul -> f1
| PDiv -> f2
| PMod -> f3
| PAnd -> f4
| POr -> f5
| PShl -> f6
| PShr -> f7
| PBitAnd -> f8
| PBitOr -> f9
| PBitXor -> f10

type ptx_cmp_tag =
| PEq
| PNe
| PLt
| PLe
| PGt
| PGe

(** val ptx_cmp_tag_rect :
    'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> ptx_cmp_tag -> 'a1 **)

let ptx_cmp_tag_rect f f0 f1 f2 f3 f4 = function
| PEq -> f
| PNe -> f0
| PLt -> f1
| PLe -> f2
| PGt -> f3
| PGe -> f4

(** val ptx_cmp_tag_rec :
    'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> ptx_cmp_tag -> 'a1 **)

let ptx_cmp_tag_rec f f0 f1 f2 f3 f4 = function
| PEq -> f
| PNe -> f0
| PLt -> f1
| PLe -> f2
| PGt -> f3
| PGe -> f4

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

(** val ptx_intrinsic_tag_rect :
    'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 ->
    ptx_intrinsic_tag -> 'a1 **)

let ptx_intrinsic_tag_rect f f0 f1 f2 f3 f4 f5 f6 f7 = function
| PISin32 -> f
| PICos32 -> f0
| PISqrt32 -> f1
| PIFabs32 -> f2
| PISin64 -> f3
| PICos64 -> f4
| PISqrt64 -> f5
| PIFabs64 -> f6
| PIFma -> f7

(** val ptx_intrinsic_tag_rec :
    'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 ->
    ptx_intrinsic_tag -> 'a1 **)

let ptx_intrinsic_tag_rec f f0 f1 f2 f3 f4 f5 f6 f7 = function
| PISin32 -> f
| PICos32 -> f0
| PISqrt32 -> f1
| PIFabs32 -> f2
| PISin64 -> f3
| PICos64 -> f4
| PISqrt64 -> f5
| PIFabs64 -> f6
| PIFma -> f7

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

(** val ptx_expr_ast_rect :
    (int -> 'a1) -> (int -> 'a1) -> (Float64.t -> 'a1) -> (Float64.t -> 'a1)
    -> (string -> 'a1) -> (ptx_binop_tag -> ptx_expr_ast -> 'a1 ->
    ptx_expr_ast -> 'a1 -> 'a1) -> (ptx_cmp_tag -> ptx_expr_ast -> 'a1 ->
    ptx_expr_ast -> 'a1 -> 'a1) -> (ptx_expr_ast -> 'a1 -> 'a1) ->
    (ptx_expr_ast -> 'a1 -> 'a1) -> 'a1 -> 'a1 -> 'a1 -> (ptx_intrinsic_tag
    -> ptx_expr_ast -> 'a1 -> 'a1) -> (ptx_expr_ast -> 'a1 -> ptx_expr_ast ->
    'a1 -> ptx_expr_ast -> 'a1 -> 'a1) -> (ptx_expr_ast -> 'a1 ->
    ptx_expr_ast -> 'a1 -> ptx_expr_ast -> 'a1 -> 'a1) -> ptx_expr_ast -> 'a1 **)

let rec ptx_expr_ast_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 = function
| PtxLitU32 n -> f n
| PtxLitU64 n -> f0 n
| PtxLitF32 f14 -> f1 f14
| PtxLitF64 f14 -> f2 f14
| PtxReg s -> f3 s
| PtxBinop (p0, p1, p2) ->
  f4 p0 p1
    (ptx_expr_ast_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p1) p2
    (ptx_expr_ast_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p2)
| PtxCmp (p0, p1, p2) ->
  f5 p0 p1
    (ptx_expr_ast_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p1) p2
    (ptx_expr_ast_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p2)
| PtxGlobalRead p0 ->
  f6 p0 (ptx_expr_ast_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p0)
| PtxSharedRead p0 ->
  f7 p0 (ptx_expr_ast_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p0)
| PtxTidx -> f8
| PtxBidx -> f9
| PtxBdim -> f10
| PtxIntrinsic (p0, p1) ->
  f11 p0 p1
    (ptx_expr_ast_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p1)
| PtxFma32 (p0, p1, p2) ->
  f12 p0
    (ptx_expr_ast_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p0) p1
    (ptx_expr_ast_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p1) p2
    (ptx_expr_ast_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p2)
| PtxFma64 (p0, p1, p2) ->
  f13 p0
    (ptx_expr_ast_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p0) p1
    (ptx_expr_ast_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p1) p2
    (ptx_expr_ast_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p2)

(** val ptx_expr_ast_rec :
    (int -> 'a1) -> (int -> 'a1) -> (Float64.t -> 'a1) -> (Float64.t -> 'a1)
    -> (string -> 'a1) -> (ptx_binop_tag -> ptx_expr_ast -> 'a1 ->
    ptx_expr_ast -> 'a1 -> 'a1) -> (ptx_cmp_tag -> ptx_expr_ast -> 'a1 ->
    ptx_expr_ast -> 'a1 -> 'a1) -> (ptx_expr_ast -> 'a1 -> 'a1) ->
    (ptx_expr_ast -> 'a1 -> 'a1) -> 'a1 -> 'a1 -> 'a1 -> (ptx_intrinsic_tag
    -> ptx_expr_ast -> 'a1 -> 'a1) -> (ptx_expr_ast -> 'a1 -> ptx_expr_ast ->
    'a1 -> ptx_expr_ast -> 'a1 -> 'a1) -> (ptx_expr_ast -> 'a1 ->
    ptx_expr_ast -> 'a1 -> ptx_expr_ast -> 'a1 -> 'a1) -> ptx_expr_ast -> 'a1 **)

let rec ptx_expr_ast_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 = function
| PtxLitU32 n -> f n
| PtxLitU64 n -> f0 n
| PtxLitF32 f14 -> f1 f14
| PtxLitF64 f14 -> f2 f14
| PtxReg s -> f3 s
| PtxBinop (p0, p1, p2) ->
  f4 p0 p1
    (ptx_expr_ast_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p1) p2
    (ptx_expr_ast_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p2)
| PtxCmp (p0, p1, p2) ->
  f5 p0 p1
    (ptx_expr_ast_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p1) p2
    (ptx_expr_ast_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p2)
| PtxGlobalRead p0 ->
  f6 p0 (ptx_expr_ast_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p0)
| PtxSharedRead p0 ->
  f7 p0 (ptx_expr_ast_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p0)
| PtxTidx -> f8
| PtxBidx -> f9
| PtxBdim -> f10
| PtxIntrinsic (p0, p1) ->
  f11 p0 p1
    (ptx_expr_ast_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p1)
| PtxFma32 (p0, p1, p2) ->
  f12 p0
    (ptx_expr_ast_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p0) p1
    (ptx_expr_ast_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p1) p2
    (ptx_expr_ast_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p2)
| PtxFma64 (p0, p1, p2) ->
  f13 p0
    (ptx_expr_ast_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p0) p1
    (ptx_expr_ast_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p1) p2
    (ptx_expr_ast_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 p2)

(** val ptx_reg_type_of : elttype -> ptx_type **)

let ptx_reg_type_of = function
| ET_Int32 -> PTX_U32
| ET_Float32 -> PTX_F32
| ET_Float64 -> PTX_F64
| ET_Bool -> PTX_U32
| ET_Unit -> PTX_U32
| _ -> PTX_U64

(** val agpu_eval_ptx_binop :
    ptx_binop_tag -> ptx_val -> ptx_val -> ptx_val option **)

let agpu_eval_ptx_binop op v1 v2 =
  match op with
  | PAdd ->
    (match v1 with
     | U32 a -> (match v2 with
                 | U32 b -> Some (U32 (Nat.add a b))
                 | _ -> None)
     | U64 a -> (match v2 with
                 | U64 b -> Some (U64 (Nat.add a b))
                 | _ -> None)
     | F32 a -> (match v2 with
                 | F32 b -> Some (F32 (add a b))
                 | _ -> None)
     | F64 a -> (match v2 with
                 | F64 b -> Some (F64 (add a b))
                 | _ -> None)
     | Pred _ -> None)
  | PSub ->
    (match v1 with
     | U32 a -> (match v2 with
                 | U32 b -> Some (U32 (Nat.sub a b))
                 | _ -> None)
     | U64 a -> (match v2 with
                 | U64 b -> Some (U64 (Nat.sub a b))
                 | _ -> None)
     | F32 a -> (match v2 with
                 | F32 b -> Some (F32 (sub a b))
                 | _ -> None)
     | F64 a -> (match v2 with
                 | F64 b -> Some (F64 (sub a b))
                 | _ -> None)
     | Pred _ -> None)
  | PMul ->
    (match v1 with
     | U32 a -> (match v2 with
                 | U32 b -> Some (U32 (Nat.mul a b))
                 | _ -> None)
     | U64 a -> (match v2 with
                 | U64 b -> Some (U64 (Nat.mul a b))
                 | _ -> None)
     | F32 a -> (match v2 with
                 | F32 b -> Some (F32 (mul a b))
                 | _ -> None)
     | F64 a -> (match v2 with
                 | F64 b -> Some (F64 (mul a b))
                 | _ -> None)
     | Pred _ -> None)
  | PDiv ->
    (match v1 with
     | U32 a ->
       (match v2 with
        | U32 b -> Some (U32 (PeanoNat.Nat.div a b))
        | _ -> None)
     | U64 a ->
       (match v2 with
        | U64 b -> Some (U64 (PeanoNat.Nat.div a b))
        | _ -> None)
     | F32 a -> (match v2 with
                 | F32 b -> Some (F32 (div a b))
                 | _ -> None)
     | F64 a -> (match v2 with
                 | F64 b -> Some (F64 (div a b))
                 | _ -> None)
     | Pred _ -> None)
  | PMod ->
    (match v1 with
     | U32 a ->
       (match v2 with
        | U32 b -> Some (U32 (PeanoNat.Nat.modulo a b))
        | _ -> None)
     | U64 a ->
       (match v2 with
        | U64 b -> Some (U64 (PeanoNat.Nat.modulo a b))
        | _ -> None)
     | _ -> None)
  | PAnd ->
    (match v1 with
     | U32 a ->
       (match v2 with
        | U32 b -> Some (U32 (PeanoNat.Nat.coq_land a b))
        | _ -> None)
     | _ -> None)
  | PShl ->
    (match v1 with
     | U32 a ->
       (match v2 with
        | U32 b -> Some (U32 (PeanoNat.Nat.shiftl a b))
        | _ -> None)
     | _ -> None)
  | PShr ->
    (match v1 with
     | U32 a ->
       (match v2 with
        | U32 b -> Some (U32 (PeanoNat.Nat.shiftr a b))
        | _ -> None)
     | _ -> None)
  | PBitAnd ->
    (match v1 with
     | U32 a ->
       (match v2 with
        | U32 b -> Some (U32 (PeanoNat.Nat.coq_land a b))
        | _ -> None)
     | _ -> None)
  | PBitXor ->
    (match v1 with
     | U32 a ->
       (match v2 with
        | U32 b -> Some (U32 (PeanoNat.Nat.coq_lxor a b))
        | _ -> None)
     | _ -> None)
  | _ ->
    (match v1 with
     | U32 a ->
       (match v2 with
        | U32 b -> Some (U32 (PeanoNat.Nat.coq_lor a b))
        | _ -> None)
     | _ -> None)

(** val agpu_eval_ptx_cmp :
    ptx_cmp_tag -> ptx_val -> ptx_val -> ptx_val option **)

let agpu_eval_ptx_cmp op v1 v2 =
  match op with
  | PEq ->
    (match v1 with
     | U32 a ->
       (match v2 with
        | U32 b -> Some (nat_cmp_to_u32 ((=) a b))
        | _ -> None)
     | U64 a ->
       (match v2 with
        | U64 b -> Some (nat_cmp_to_u32 ((=) a b))
        | _ -> None)
     | F32 a ->
       (match v2 with
        | F32 b -> Some (nat_cmp_to_u32 (PrimFloat.eqb a b))
        | _ -> None)
     | F64 a ->
       (match v2 with
        | F64 b -> Some (nat_cmp_to_u32 (PrimFloat.eqb a b))
        | _ -> None)
     | Pred _ -> None)
  | PNe ->
    (match v1 with
     | U32 a ->
       (match v2 with
        | U32 b -> Some (nat_cmp_to_u32 (negb ((=) a b)))
        | _ -> None)
     | U64 a ->
       (match v2 with
        | U64 b -> Some (nat_cmp_to_u32 (negb ((=) a b)))
        | _ -> None)
     | F32 a ->
       (match v2 with
        | F32 b -> Some (nat_cmp_to_u32 (negb (PrimFloat.eqb a b)))
        | _ -> None)
     | F64 a ->
       (match v2 with
        | F64 b -> Some (nat_cmp_to_u32 (negb (PrimFloat.eqb a b)))
        | _ -> None)
     | Pred _ -> None)
  | PLt ->
    (match v1 with
     | U32 a ->
       (match v2 with
        | U32 b -> Some (nat_cmp_to_u32 (PeanoNat.Nat.ltb a b))
        | _ -> None)
     | U64 a ->
       (match v2 with
        | U64 b -> Some (nat_cmp_to_u32 (PeanoNat.Nat.ltb a b))
        | _ -> None)
     | F32 a ->
       (match v2 with
        | F32 b -> Some (nat_cmp_to_u32 (PrimFloat.ltb a b))
        | _ -> None)
     | F64 a ->
       (match v2 with
        | F64 b -> Some (nat_cmp_to_u32 (PrimFloat.ltb a b))
        | _ -> None)
     | Pred _ -> None)
  | PLe ->
    (match v1 with
     | U32 a ->
       (match v2 with
        | U32 b -> Some (nat_cmp_to_u32 ((<=) a b))
        | _ -> None)
     | U64 a ->
       (match v2 with
        | U64 b -> Some (nat_cmp_to_u32 ((<=) a b))
        | _ -> None)
     | F32 a ->
       (match v2 with
        | F32 b -> Some (nat_cmp_to_u32 (PrimFloat.leb a b))
        | _ -> None)
     | F64 a ->
       (match v2 with
        | F64 b -> Some (nat_cmp_to_u32 (PrimFloat.leb a b))
        | _ -> None)
     | Pred _ -> None)
  | PGt ->
    (match v1 with
     | U32 a ->
       (match v2 with
        | U32 b -> Some (nat_cmp_to_u32 (PeanoNat.Nat.ltb b a))
        | _ -> None)
     | U64 a ->
       (match v2 with
        | U64 b -> Some (nat_cmp_to_u32 (PeanoNat.Nat.ltb b a))
        | _ -> None)
     | F32 a ->
       (match v2 with
        | F32 b -> Some (nat_cmp_to_u32 (PrimFloat.ltb b a))
        | _ -> None)
     | F64 a ->
       (match v2 with
        | F64 b -> Some (nat_cmp_to_u32 (PrimFloat.ltb b a))
        | _ -> None)
     | Pred _ -> None)
  | PGe ->
    (match v1 with
     | U32 a ->
       (match v2 with
        | U32 b -> Some (nat_cmp_to_u32 ((<=) b a))
        | _ -> None)
     | U64 a ->
       (match v2 with
        | U64 b -> Some (nat_cmp_to_u32 ((<=) b a))
        | _ -> None)
     | F32 a ->
       (match v2 with
        | F32 b -> Some (nat_cmp_to_u32 (PrimFloat.leb b a))
        | _ -> None)
     | F64 a ->
       (match v2 with
        | F64 b -> Some (nat_cmp_to_u32 (PrimFloat.leb b a))
        | _ -> None)
     | Pred _ -> None)

(** val agpu_eval_ptx_intrinsic :
    ptx_intrinsic_tag -> ptx_val -> ptx_val option **)

let agpu_eval_ptx_intrinsic tag v =
  match tag with
  | PISin32 -> (match v with
                | F32 x -> Some (F32 (sin_f32 x))
                | _ -> None)
  | PICos32 -> (match v with
                | F32 x -> Some (F32 (cos_f32 x))
                | _ -> None)
  | PISqrt32 -> (match v with
                 | F32 x -> Some (F32 (sqrt x))
                 | _ -> None)
  | PIFabs32 -> (match v with
                 | F32 x -> Some (F32 (abs x))
                 | _ -> None)
  | PISin64 -> (match v with
                | F64 x -> Some (F64 (sin_f64 x))
                | _ -> None)
  | PICos64 -> (match v with
                | F64 x -> Some (F64 (cos_f64 x))
                | _ -> None)
  | PISqrt64 -> (match v with
                 | F64 x -> Some (F64 (sqrt x))
                 | _ -> None)
  | PIFabs64 -> (match v with
                 | F64 x -> Some (F64 (abs x))
                 | _ -> None)
  | PIFma -> None

(** val agpu_eval_ptx :
    agpu_state -> ptx_expr_ast -> (ptx_val * agpu_state) option **)

let rec agpu_eval_ptx st = function
| PtxLitU32 n -> Some ((U32 n), st)
| PtxLitU64 n -> Some ((U64 n), st)
| PtxLitF32 f -> Some ((F32 f), st)
| PtxLitF64 f -> Some ((F64 f), st)
| PtxReg name ->
  (match st.regs name with
   | Some v -> Some (v, st)
   | None -> None)
| PtxBinop (op, e1, e2) ->
  (match agpu_eval_ptx st e1 with
   | Some p ->
     let (v1, st1) = p in
     (match agpu_eval_ptx st1 e2 with
      | Some p0 ->
        let (v2, st2) = p0 in
        (match agpu_eval_ptx_binop op v1 v2 with
         | Some v -> Some (v, st2)
         | None -> None)
      | None -> None)
   | None -> None)
| PtxCmp (op, e1, e2) ->
  (match agpu_eval_ptx st e1 with
   | Some p ->
     let (v1, st1) = p in
     (match agpu_eval_ptx st1 e2 with
      | Some p0 ->
        let (v2, st2) = p0 in
        (match agpu_eval_ptx_cmp op v1 v2 with
         | Some v -> Some (v, st2)
         | None -> None)
      | None -> None)
   | None -> None)
| PtxGlobalRead addr_e ->
  (match agpu_eval_ptx st addr_e with
   | Some p ->
     let (p0, st1) = p in
     (match p0 with
      | U32 a -> Some ((st1.mem.global_mem a), st1)
      | U64 a -> Some ((st1.mem.global_mem a), st1)
      | _ -> None)
   | None -> None)
| PtxSharedRead addr_e ->
  (match agpu_eval_ptx st addr_e with
   | Some p ->
     let (p0, st1) = p in
     (match p0 with
      | U32 a -> Some ((st1.mem.shared_mem a), st1)
      | U64 a -> Some ((st1.mem.shared_mem a), st1)
      | _ -> None)
   | None -> None)
| PtxTidx -> Some ((U32 st.tc.tidx), st)
| PtxBidx -> Some ((U32 st.tc.bidx), st)
| PtxBdim -> Some ((U32 st.tc.bdim), st)
| PtxIntrinsic (tag, e1) ->
  (match agpu_eval_ptx st e1 with
   | Some p ->
     let (v, st1) = p in
     (match agpu_eval_ptx_intrinsic tag v with
      | Some r -> Some (r, st1)
      | None -> None)
   | None -> None)
| PtxFma32 (ea, eb, ec) ->
  (match agpu_eval_ptx st ea with
   | Some p ->
     let (p0, st1) = p in
     (match p0 with
      | F32 a ->
        (match agpu_eval_ptx st1 eb with
         | Some p1 ->
           let (p2, st2) = p1 in
           (match p2 with
            | F32 b ->
              (match agpu_eval_ptx st2 ec with
               | Some p3 ->
                 let (p4, st3) = p3 in
                 (match p4 with
                  | F32 c -> Some ((F32 (fma_f32 a b c)), st3)
                  | _ -> None)
               | None -> None)
            | _ -> None)
         | None -> None)
      | _ -> None)
   | None -> None)
| PtxFma64 (ea, eb, ec) ->
  (match agpu_eval_ptx st ea with
   | Some p ->
     let (p0, st1) = p in
     (match p0 with
      | F64 a ->
        (match agpu_eval_ptx st1 eb with
         | Some p1 ->
           let (p2, st2) = p1 in
           (match p2 with
            | F64 b ->
              (match agpu_eval_ptx st2 ec with
               | Some p3 ->
                 let (p4, st3) = p3 in
                 (match p4 with
                  | F64 c -> Some ((F64 (fma_f64 a b c)), st3)
                  | _ -> None)
               | None -> None)
            | _ -> None)
         | None -> None)
      | _ -> None)
   | None -> None)
