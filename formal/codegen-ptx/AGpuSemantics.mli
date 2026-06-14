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

val ptx_val_rect :
  (int -> 'a1) -> (int -> 'a1) -> (Float64.t -> 'a1) -> (Float64.t -> 'a1) ->
  (bool -> 'a1) -> ptx_val -> 'a1

val ptx_val_rec :
  (int -> 'a1) -> (int -> 'a1) -> (Float64.t -> 'a1) -> (Float64.t -> 'a1) ->
  (bool -> 'a1) -> ptx_val -> 'a1

type mem_space =
| Global
| Shared

val mem_space_rect : 'a1 -> 'a1 -> mem_space -> 'a1

val mem_space_rec : 'a1 -> 'a1 -> mem_space -> 'a1

type thread_const = { tidx : int; bidx : int; bdim : int }

val tidx : thread_const -> int

val bidx : thread_const -> int

val bdim : thread_const -> int

type agpu_mem = { global_mem : (int -> ptx_val); shared_mem : (int -> ptx_val) }

val global_mem : agpu_mem -> int -> ptx_val

val shared_mem : agpu_mem -> int -> ptx_val

type agpu_state = { regs : (string -> ptx_val option); tc : thread_const;
                    mem : agpu_mem }

val regs : agpu_state -> string -> ptx_val option

val tc : agpu_state -> thread_const

val mem : agpu_state -> agpu_mem

val sin_f32 : Float64.t -> Float64.t

val cos_f32 : Float64.t -> Float64.t

val fma_f32 : Float64.t -> Float64.t -> Float64.t -> Float64.t

val sin_f64 : Float64.t -> Float64.t

val cos_f64 : Float64.t -> Float64.t

val fma_f64 : Float64.t -> Float64.t -> Float64.t -> Float64.t

type ir_const =
| CInt32 of int
| CInt64 of int
| CFloat32 of Float64.t
| CFloat64 of Float64.t
| CBool of bool
| CUnit

val ir_const_rect :
  (int -> 'a1) -> (int -> 'a1) -> (Float64.t -> 'a1) -> (Float64.t -> 'a1) ->
  (bool -> 'a1) -> 'a1 -> ir_const -> 'a1

val ir_const_rec :
  (int -> 'a1) -> (int -> 'a1) -> (Float64.t -> 'a1) -> (Float64.t -> 'a1) ->
  (bool -> 'a1) -> 'a1 -> ir_const -> 'a1

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

val ir_binop_rect :
  'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1
  -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> ir_binop -> 'a1

val ir_binop_rec :
  'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1
  -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> ir_binop -> 'a1

type ir_elttype =
| TInt32
| TInt64
| TFloat32
| TFloat64
| TBool

val ir_elttype_rect : 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> ir_elttype -> 'a1

val ir_elttype_rec : 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> ir_elttype -> 'a1

type ir_memspace =
| MS_Global
| MS_Shared

val ir_memspace_rect : 'a1 -> 'a1 -> ir_memspace -> 'a1

val ir_memspace_rec : 'a1 -> 'a1 -> ir_memspace -> 'a1

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

val ir_expr_rect :
  (ir_const -> 'a1) -> (string -> 'a1) -> (ir_binop -> ir_expr -> 'a1 ->
  ir_expr -> 'a1 -> 'a1) -> (ir_memspace -> ir_expr -> 'a1 -> ir_expr -> 'a1
  -> 'a1) -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> (ir_expr -> 'a1 -> 'a1) ->
  (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1 ->
  'a1) -> (ir_expr -> 'a1 -> ir_expr -> 'a1 -> ir_expr -> 'a1 -> 'a1) ->
  (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1 ->
  'a1) -> (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1 -> ir_expr -> 'a1 ->
  ir_expr -> 'a1 -> 'a1) -> ir_expr -> 'a1

val ir_expr_rec :
  (ir_const -> 'a1) -> (string -> 'a1) -> (ir_binop -> ir_expr -> 'a1 ->
  ir_expr -> 'a1 -> 'a1) -> (ir_memspace -> ir_expr -> 'a1 -> ir_expr -> 'a1
  -> 'a1) -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> 'a1 -> (ir_expr -> 'a1 -> 'a1) ->
  (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1 ->
  'a1) -> (ir_expr -> 'a1 -> ir_expr -> 'a1 -> ir_expr -> 'a1 -> 'a1) ->
  (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1 ->
  'a1) -> (ir_expr -> 'a1 -> 'a1) -> (ir_expr -> 'a1 -> ir_expr -> 'a1 ->
  ir_expr -> 'a1 -> 'a1) -> ir_expr -> 'a1

val bool_to_nat : bool -> int

val nat_cmp_to_u32 : bool -> ptx_val

val agpu_eval_binop : ir_binop -> ptx_val -> ptx_val -> ptx_val option

val agpu_eval_ir : agpu_state -> ir_expr -> (ptx_val * agpu_state) option
