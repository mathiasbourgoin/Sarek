open Datatypes
open Nat
open PrimFloat
open String

type ptx_val =
| U32 of int
| U64 of int
| F32 of Float64.t
| F64 of Float64.t
| Pred of bool

(** val ptx_val_rect :
    (int -> 'a1) -> (int -> 'a1) -> (Float64.t -> 'a1) -> (Float64.t -> 'a1)
    -> (bool -> 'a1) -> ptx_val -> 'a1 **)

let ptx_val_rect f f0 f1 f2 f3 = function
| U32 n -> f n
| U64 n -> f0 n
| F32 f4 -> f1 f4
| F64 f4 -> f2 f4
| Pred b -> f3 b

(** val ptx_val_rec :
    (int -> 'a1) -> (int -> 'a1) -> (Float64.t -> 'a1) -> (Float64.t -> 'a1)
    -> (bool -> 'a1) -> ptx_val -> 'a1 **)

let ptx_val_rec f f0 f1 f2 f3 = function
| U32 n -> f n
| U64 n -> f0 n
| F32 f4 -> f1 f4
| F64 f4 -> f2 f4
| Pred b -> f3 b

type mem_space =
| Global
| Shared

(** val mem_space_rect : 'a1 -> 'a1 -> mem_space -> 'a1 **)

let mem_space_rect f f0 = function
| Global -> f
| Shared -> f0

(** val mem_space_rec : 'a1 -> 'a1 -> mem_space -> 'a1 **)

let mem_space_rec f f0 = function
| Global -> f
| Shared -> f0

type thread_const = { tidx : int; bidx : int; bdim : int }

(** val tidx : thread_const -> int **)

let tidx t =
  t.tidx

(** val bidx : thread_const -> int **)

let bidx t =
  t.bidx

(** val bdim : thread_const -> int **)

let bdim t =
  t.bdim

type agpu_mem = { global_mem : (int -> ptx_val); shared_mem : (int -> ptx_val) }

(** val global_mem : agpu_mem -> int -> ptx_val **)

let global_mem a =
  a.global_mem

(** val shared_mem : agpu_mem -> int -> ptx_val **)

let shared_mem a =
  a.shared_mem

type agpu_state = { regs : (string -> ptx_val option); tc : thread_const;
                    mem : agpu_mem }

(** val regs : agpu_state -> string -> ptx_val option **)

let regs a =
  a.regs

(** val tc : agpu_state -> thread_const **)

let tc a =
  a.tc

(** val mem : agpu_state -> agpu_mem **)

let mem a =
  a.mem

(** val sin_f32 : Float64.t -> Float64.t **)

let sin_f32 = Float.sin

(** val cos_f32 : Float64.t -> Float64.t **)

let cos_f32 = Float.cos

(** val fma_f32 : Float64.t -> Float64.t -> Float64.t -> Float64.t **)

let fma_f32 = Float.fma

(** val sin_f64 : Float64.t -> Float64.t **)

let sin_f64 = Float.sin

(** val cos_f64 : Float64.t -> Float64.t **)

let cos_f64 = Float.cos

(** val fma_f64 : Float64.t -> Float64.t -> Float64.t -> Float64.t **)

let fma_f64 = Float.fma

type ir_const =
| CInt32 of int
| CInt64 of int
| CFloat32 of Float64.t
| CFloat64 of Float64.t
| CBool of bool
| CUnit

(** val ir_const_rect :
    (int -> 'a1) -> (int -> 'a1) -> (Float64.t -> 'a1) -> (Float64.t -> 'a1)
    -> (bool -> 'a1) -> 'a1 -> ir_const -> 'a1 **)

let ir_const_rect f f0 f1 f2 f3 f4 = function
| CInt32 n -> f n
| CInt64 n -> f0 n
| CFloat32 f5 -> f1 f5
| CFloat64 f5 -> f2 f5
| CBool b -> f3 b
| CUnit -> f4

(** val ir_const_rec :
    (int -> 'a1) -> (int -> 'a1) -> (Float64.t -> 'a1) -> (Float64.t -> 'a1)
    -> (bool -> 'a1) -> 'a1 -> ir_const -> 'a1 **)

let ir_const_rec f f0 f1 f2 f3 f4 = function
| CInt32 n -> f n
| CInt64 n -> f0 n
| CFloat32 f5 -> f1 f5
| CFloat64 f5 -> f2 f5
| CBool b -> f3 b
| CUnit -> f4

type ir_binop =
| Add
| Sub
| Mul
| Div
| Mod
| Eq
| Ne
| Lt
| Le
| Gt
| Ge
| And
| Or
| Shl
| Shr
| BitAnd
| BitOr
| BitXor

(** val ir_binop_rect :
    'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1
    -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> ir_binop -> 'a1 **)

let ir_binop_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16 = function
| Add -> f
| Sub -> f0
| Mul -> f1
| Div -> f2
| Mod -> f3
| Eq -> f4
| Ne -> f5
| Lt -> f6
| Le -> f7
| Gt -> f8
| Ge -> f9
| And -> f10
| Or -> f11
| Shl -> f12
| Shr -> f13
| BitAnd -> f14
| BitOr -> f15
| BitXor -> f16

(** val ir_binop_rec :
    'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1
    -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> ir_binop -> 'a1 **)

let ir_binop_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16 = function
| Add -> f
| Sub -> f0
| Mul -> f1
| Div -> f2
| Mod -> f3
| Eq -> f4
| Ne -> f5
| Lt -> f6
| Le -> f7
| Gt -> f8
| Ge -> f9
| And -> f10
| Or -> f11
| Shl -> f12
| Shr -> f13
| BitAnd -> f14
| BitOr -> f15
| BitXor -> f16

type ir_elttype =
| TInt32
| TInt64
| TFloat32
| TFloat64
| TBool

(** val ir_elttype_rect :
    'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> ir_elttype -> 'a1 **)

let ir_elttype_rect f f0 f1 f2 f3 = function
| TInt32 -> f
| TInt64 -> f0
| TFloat32 -> f1
| TFloat64 -> f2
| TBool -> f3

(** val ir_elttype_rec :
    'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> ir_elttype -> 'a1 **)

let ir_elttype_rec f f0 f1 f2 f3 = function
| TInt32 -> f
| TInt64 -> f0
| TFloat32 -> f1
| TFloat64 -> f2
| TBool -> f3

type ir_memspace =
| MS_Global
| MS_Shared

(** val ir_memspace_rect : 'a1 -> 'a1 -> ir_memspace -> 'a1 **)

let ir_memspace_rect f f0 = function
| MS_Global -> f
| MS_Shared -> f0

(** val ir_memspace_rec : 'a1 -> 'a1 -> ir_memspace -> 'a1 **)

let ir_memspace_rec f f0 = function
| MS_Global -> f
| MS_Shared -> f0

type ir_expr =
| IEConst of ir_const
| IEVar of string
| IEBinop of ir_binop * ir_expr * ir_expr
| IEArrayRead of ir_memspace * ir_expr * ir_expr
| IEThreadIdxX
| IEBlockIdxX
| IEBlockDimX
| IEGlobalIdx
| IEBarrier
| IESin32 of ir_expr
| IECos32 of ir_expr
| IESqrt32 of ir_expr
| IEFabs32 of ir_expr
| IEFma32 of ir_expr * ir_expr * ir_expr
| IESin64 of ir_expr
| IECos64 of ir_expr
| IESqrt64 of ir_expr
| IEFabs64 of ir_expr
| IEFma64 of ir_expr * ir_expr * ir_expr

(** val ir_expr_rect :
    (ir_const -> 'a1) -> (string -> 'a1) -> (ir_binop -> ir_expr -> 'a1 ->
    ir_expr -> 'a1 -> 'a1) -> (ir_memspace -> ir_expr -> 'a1 -> ir_expr ->
    'a1 -> 'a1) -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> (ir_expr -> 'a1 -> 'a1)
    -> (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1
    -> 'a1) -> (ir_expr -> 'a1 -> ir_expr -> 'a1 -> ir_expr -> 'a1 -> 'a1) ->
    (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1 ->
    'a1) -> (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1 -> ir_expr -> 'a1 ->
    ir_expr -> 'a1 -> 'a1) -> ir_expr -> 'a1 **)

let rec ir_expr_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16 f17 = function
| IEConst i0 -> f i0
| IEVar s -> f0 s
| IEBinop (i0, i1, i2) ->
  f1 i0 i1
    (ir_expr_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i1)
    i2
    (ir_expr_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i2)
| IEArrayRead (i0, i1, i2) ->
  f2 i0 i1
    (ir_expr_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i1)
    i2
    (ir_expr_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i2)
| IEThreadIdxX -> f3
| IEBlockIdxX -> f4
| IEBlockDimX -> f5
| IEGlobalIdx -> f6
| IEBarrier -> f7
| IESin32 i0 ->
  f8 i0
    (ir_expr_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
| IECos32 i0 ->
  f9 i0
    (ir_expr_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
| IESqrt32 i0 ->
  f10 i0
    (ir_expr_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
| IEFabs32 i0 ->
  f11 i0
    (ir_expr_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
| IEFma32 (i0, i1, i2) ->
  f12 i0
    (ir_expr_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
    i1
    (ir_expr_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i1)
    i2
    (ir_expr_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i2)
| IESin64 i0 ->
  f13 i0
    (ir_expr_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
| IECos64 i0 ->
  f14 i0
    (ir_expr_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
| IESqrt64 i0 ->
  f15 i0
    (ir_expr_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
| IEFabs64 i0 ->
  f16 i0
    (ir_expr_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
| IEFma64 (i0, i1, i2) ->
  f17 i0
    (ir_expr_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
    i1
    (ir_expr_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i1)
    i2
    (ir_expr_rect f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i2)

(** val ir_expr_rec :
    (ir_const -> 'a1) -> (string -> 'a1) -> (ir_binop -> ir_expr -> 'a1 ->
    ir_expr -> 'a1 -> 'a1) -> (ir_memspace -> ir_expr -> 'a1 -> ir_expr ->
    'a1 -> 'a1) -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> (ir_expr -> 'a1 -> 'a1)
    -> (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1
    -> 'a1) -> (ir_expr -> 'a1 -> ir_expr -> 'a1 -> ir_expr -> 'a1 -> 'a1) ->
    (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1 ->
    'a1) -> (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1 -> ir_expr -> 'a1 ->
    ir_expr -> 'a1 -> 'a1) -> ir_expr -> 'a1 **)

let rec ir_expr_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16 f17 = function
| IEConst i0 -> f i0
| IEVar s -> f0 s
| IEBinop (i0, i1, i2) ->
  f1 i0 i1
    (ir_expr_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i1)
    i2
    (ir_expr_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i2)
| IEArrayRead (i0, i1, i2) ->
  f2 i0 i1
    (ir_expr_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i1)
    i2
    (ir_expr_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i2)
| IEThreadIdxX -> f3
| IEBlockIdxX -> f4
| IEBlockDimX -> f5
| IEGlobalIdx -> f6
| IEBarrier -> f7
| IESin32 i0 ->
  f8 i0
    (ir_expr_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
| IECos32 i0 ->
  f9 i0
    (ir_expr_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
| IESqrt32 i0 ->
  f10 i0
    (ir_expr_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
| IEFabs32 i0 ->
  f11 i0
    (ir_expr_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
| IEFma32 (i0, i1, i2) ->
  f12 i0
    (ir_expr_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
    i1
    (ir_expr_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i1)
    i2
    (ir_expr_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i2)
| IESin64 i0 ->
  f13 i0
    (ir_expr_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
| IECos64 i0 ->
  f14 i0
    (ir_expr_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
| IESqrt64 i0 ->
  f15 i0
    (ir_expr_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
| IEFabs64 i0 ->
  f16 i0
    (ir_expr_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
| IEFma64 (i0, i1, i2) ->
  f17 i0
    (ir_expr_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i0)
    i1
    (ir_expr_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i1)
    i2
    (ir_expr_rec f f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16
      f17 i2)

(** val bool_to_nat : bool -> int **)

let bool_to_nat = function
| true -> Stdlib.Int.succ 0
| false -> 0

(** val nat_cmp_to_u32 : bool -> ptx_val **)

let nat_cmp_to_u32 b =
  U32 (bool_to_nat b)

(** val agpu_eval_binop : ir_binop -> ptx_val -> ptx_val -> ptx_val option **)

let agpu_eval_binop op v1 v2 =
  match op with
  | Add ->
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
  | Sub ->
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
  | Mul ->
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
  | Div ->
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
  | Mod ->
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
  | Eq ->
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
  | Ne ->
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
  | Lt ->
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
  | Le ->
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
  | Gt ->
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
  | Ge ->
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
  | And ->
    (match v1 with
     | U32 a ->
       (match v2 with
        | U32 b -> Some (U32 (PeanoNat.Nat.coq_land a b))
        | _ -> None)
     | _ -> None)
  | Shl ->
    (match v1 with
     | U32 a ->
       (match v2 with
        | U32 b -> Some (U32 (PeanoNat.Nat.shiftl a b))
        | _ -> None)
     | _ -> None)
  | Shr ->
    (match v1 with
     | U32 a ->
       (match v2 with
        | U32 b -> Some (U32 (PeanoNat.Nat.shiftr a b))
        | _ -> None)
     | _ -> None)
  | BitAnd ->
    (match v1 with
     | U32 a ->
       (match v2 with
        | U32 b -> Some (U32 (PeanoNat.Nat.coq_land a b))
        | _ -> None)
     | _ -> None)
  | BitXor ->
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

(** val agpu_eval_ir :
    agpu_state -> ir_expr -> (ptx_val * agpu_state) option **)

let rec agpu_eval_ir st = function
| IEConst i ->
  (match i with
   | CInt32 n -> Some ((U32 n), st)
   | CInt64 n -> Some ((U64 n), st)
   | CFloat32 f -> Some ((F32 f), st)
   | CFloat64 f -> Some ((F64 f), st)
   | CBool b -> Some ((U32 (bool_to_nat b)), st)
   | CUnit -> Some ((U32 0), st))
| IEVar name ->
  (match st.regs name with
   | Some v -> Some (v, st)
   | None -> None)
| IEBinop (op, e1, e2) ->
  (match agpu_eval_ir st e1 with
   | Some p ->
     let (v1, st1) = p in
     (match agpu_eval_ir st1 e2 with
      | Some p0 ->
        let (v2, st2) = p0 in
        (match agpu_eval_binop op v1 v2 with
         | Some v -> Some (v, st2)
         | None -> None)
      | None -> None)
   | None -> None)
| IEArrayRead (i, base_e, idx_e) ->
  (match i with
   | MS_Global ->
     (match agpu_eval_ir st base_e with
      | Some p ->
        let (p0, st1) = p in
        (match p0 with
         | U32 base ->
           (match agpu_eval_ir st1 idx_e with
            | Some p1 ->
              let (p2, st2) = p1 in
              (match p2 with
               | U32 idx ->
                 Some ((st2.mem.global_mem (Nat.add base idx)), st2)
               | _ -> None)
            | None -> None)
         | U64 base ->
           (match agpu_eval_ir st1 idx_e with
            | Some p1 ->
              let (p2, st2) = p1 in
              (match p2 with
               | U64 idx ->
                 Some ((st2.mem.global_mem (Nat.add base idx)), st2)
               | _ -> None)
            | None -> None)
         | _ -> None)
      | None -> None)
   | MS_Shared ->
     (match agpu_eval_ir st base_e with
      | Some p ->
        let (p0, st1) = p in
        (match p0 with
         | U32 base ->
           (match agpu_eval_ir st1 idx_e with
            | Some p1 ->
              let (p2, st2) = p1 in
              (match p2 with
               | U32 idx ->
                 Some ((st2.mem.shared_mem (Nat.add base idx)), st2)
               | _ -> None)
            | None -> None)
         | U64 base ->
           (match agpu_eval_ir st1 idx_e with
            | Some p1 ->
              let (p2, st2) = p1 in
              (match p2 with
               | U64 idx ->
                 Some ((st2.mem.shared_mem (Nat.add base idx)), st2)
               | _ -> None)
            | None -> None)
         | _ -> None)
      | None -> None))
| IEThreadIdxX -> Some ((U32 st.tc.tidx), st)
| IEBlockIdxX -> Some ((U32 st.tc.bidx), st)
| IEBlockDimX -> Some ((U32 st.tc.bdim), st)
| IEGlobalIdx ->
  let gid = Nat.add (Nat.mul st.tc.bidx st.tc.bdim) st.tc.tidx in
  Some ((U32 gid), st)
| IEBarrier -> Some ((U32 0), st)
| IESin32 e1 ->
  (match agpu_eval_ir st e1 with
   | Some p ->
     let (p0, st1) = p in
     (match p0 with
      | F32 x -> Some ((F32 (sin_f32 x)), st1)
      | _ -> None)
   | None -> None)
| IECos32 e1 ->
  (match agpu_eval_ir st e1 with
   | Some p ->
     let (p0, st1) = p in
     (match p0 with
      | F32 x -> Some ((F32 (cos_f32 x)), st1)
      | _ -> None)
   | None -> None)
| IESqrt32 e1 ->
  (match agpu_eval_ir st e1 with
   | Some p ->
     let (p0, st1) = p in
     (match p0 with
      | F32 x -> Some ((F32 (sqrt x)), st1)
      | _ -> None)
   | None -> None)
| IEFabs32 e1 ->
  (match agpu_eval_ir st e1 with
   | Some p ->
     let (p0, st1) = p in
     (match p0 with
      | F32 x -> Some ((F32 (abs x)), st1)
      | _ -> None)
   | None -> None)
| IEFma32 (ea, eb, ec) ->
  (match agpu_eval_ir st ea with
   | Some p ->
     let (p0, st1) = p in
     (match p0 with
      | F32 a ->
        (match agpu_eval_ir st1 eb with
         | Some p1 ->
           let (p2, st2) = p1 in
           (match p2 with
            | F32 b ->
              (match agpu_eval_ir st2 ec with
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
| IESin64 e1 ->
  (match agpu_eval_ir st e1 with
   | Some p ->
     let (p0, st1) = p in
     (match p0 with
      | F64 x -> Some ((F64 (sin_f64 x)), st1)
      | _ -> None)
   | None -> None)
| IECos64 e1 ->
  (match agpu_eval_ir st e1 with
   | Some p ->
     let (p0, st1) = p in
     (match p0 with
      | F64 x -> Some ((F64 (cos_f64 x)), st1)
      | _ -> None)
   | None -> None)
| IESqrt64 e1 ->
  (match agpu_eval_ir st e1 with
   | Some p ->
     let (p0, st1) = p in
     (match p0 with
      | F64 x -> Some ((F64 (sqrt x)), st1)
      | _ -> None)
   | None -> None)
| IEFabs64 e1 ->
  (match agpu_eval_ir st e1 with
   | Some p ->
     let (p0, st1) = p in
     (match p0 with
      | F64 x -> Some ((F64 (abs x)), st1)
      | _ -> None)
   | None -> None)
| IEFma64 (ea, eb, ec) ->
  (match agpu_eval_ir st ea with
   | Some p ->
     let (p0, st1) = p in
     (match p0 with
      | F64 a ->
        (match agpu_eval_ir st1 eb with
         | Some p1 ->
           let (p2, st2) = p1 in
           (match p2 with
            | F64 b ->
              (match agpu_eval_ir st2 ec with
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
