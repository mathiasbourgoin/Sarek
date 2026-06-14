(******************************************************************************)
(* test_codegen_ptx_conformance.ml
 *
 * CMBT (certified model-based testing) conformance harness for the
 * Sarek PTX code generation formal model.
 *
 * Strategy: define a pure OCaml reference interpreter that mirrors the Rocq
 * definitions in AGpuSemantics.v, PtxTypes.v, and PtxExprSpec.v.  Run Alcotest
 * smoke tests that exercise the reference model on concrete inputs and verify:
 *
 *   1. Literals evaluate to the expected PTX value.
 *   2. Thread intrinsics (tidx, bidx, bdim, global_idx) evaluate correctly.
 *   3. Arithmetic binary-ops evaluate correctly on U32 operands.
 *   4. Comparison binary-ops return U32 0/1.
 *   5. Math intrinsics (sin32, cos32, sqrt32, fabs32, fma32) evaluate correctly.
 *   6. [emit_ast_expr] followed by [agpu_eval_ptx] matches [agpu_eval_ir]
 *      for each test case (the key correctness property modelled by
 *      emit_expr_correct / eval_ir_ptx_eq in the Rocq spec).
 *
 * This file has no dependency on extracted Rocq code (Float64, Datatypes).
 * It is a standalone OCaml model faithful to the Rocq spec.
 *)

(* ======================================================================= *)
(** * 1. PTX value domain — mirrors [ptx_val] in AGpuSemantics.v *)
(* ======================================================================= *)

type ptx_val =
  | U32 of int
  | U64 of int
  | F32 of float
  | F64 of float
  | Pred of bool

(* ======================================================================= *)
(** * 2. Abstract GPU state *)
(* ======================================================================= *)

type thread_const = {tidx : int; bidx : int; bdim : int}

type agpu_mem = {global_mem : int -> ptx_val; shared_mem : int -> ptx_val}

type agpu_state = {
  regs : string -> ptx_val option;
  tc : thread_const;
  mem : agpu_mem;
}

let make_state ?(tidx = 0) ?(bidx = 0) ?(bdim = 1) () =
  {
    regs = (fun _ -> None);
    tc = {tidx; bidx; bdim};
    mem = {global_mem = (fun i -> U32 i); shared_mem = (fun i -> U32 i)};
  }

let reg_write name v st =
  {st with regs = (fun n -> if n = name then Some v else st.regs n)}

(* ======================================================================= *)
(** * 3. IR expression subset — mirrors [ir_expr] *)
(* ======================================================================= *)

type ir_const =
  | CInt32 of int
  | CInt64 of int
  | CFloat32 of float
  | CFloat64 of float
  | CBool of bool
  | CUnit

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

type ir_memspace = MS_Global | MS_Shared

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

(* ======================================================================= *)
(** * 4. [agpu_eval_binop] — mirrors AGpuSemantics.agpu_eval_binop *)
(* ======================================================================= *)

let nat_cmp b = U32 (if b then 1 else 0)

let agpu_eval_binop op v1 v2 =
  match (op, v1, v2) with
  (* U32 arithmetic *)
  | Add, U32 a, U32 b -> Some (U32 (a + b))
  | Sub, U32 a, U32 b -> Some (U32 (max 0 (a - b)))
  | Mul, U32 a, U32 b -> Some (U32 (a * b))
  | Div, U32 a, U32 b -> Some (U32 (if b = 0 then 0 else a / b))
  | Mod, U32 a, U32 b -> Some (U32 (if b = 0 then 0 else a mod b))
  | Eq, U32 a, U32 b -> Some (nat_cmp (a = b))
  | Ne, U32 a, U32 b -> Some (nat_cmp (a <> b))
  | Lt, U32 a, U32 b -> Some (nat_cmp (a < b))
  | Le, U32 a, U32 b -> Some (nat_cmp (a <= b))
  | Gt, U32 a, U32 b -> Some (nat_cmp (a > b))
  | Ge, U32 a, U32 b -> Some (nat_cmp (a >= b))
  | And, U32 a, U32 b -> Some (U32 (a land b))
  | Or, U32 a, U32 b -> Some (U32 (a lor b))
  | BitAnd, U32 a, U32 b -> Some (U32 (a land b))
  | BitOr, U32 a, U32 b -> Some (U32 (a lor b))
  | BitXor, U32 a, U32 b -> Some (U32 (a lxor b))
  | Shl, U32 a, U32 b -> Some (U32 (a lsl b))
  | Shr, U32 a, U32 b -> Some (U32 (a lsr b))
  (* U64 arithmetic *)
  | Add, U64 a, U64 b -> Some (U64 (a + b))
  | Sub, U64 a, U64 b -> Some (U64 (max 0 (a - b)))
  | Mul, U64 a, U64 b -> Some (U64 (a * b))
  | Div, U64 a, U64 b -> Some (U64 (if b = 0 then 0 else a / b))
  | Mod, U64 a, U64 b -> Some (U64 (if b = 0 then 0 else a mod b))
  | Eq, U64 a, U64 b -> Some (nat_cmp (a = b))
  | Ne, U64 a, U64 b -> Some (nat_cmp (a <> b))
  | Lt, U64 a, U64 b -> Some (nat_cmp (a < b))
  | Le, U64 a, U64 b -> Some (nat_cmp (a <= b))
  | Gt, U64 a, U64 b -> Some (nat_cmp (a > b))
  | Ge, U64 a, U64 b -> Some (nat_cmp (a >= b))
  (* F32 arithmetic *)
  | Add, F32 a, F32 b -> Some (F32 (a +. b))
  | Sub, F32 a, F32 b -> Some (F32 (a -. b))
  | Mul, F32 a, F32 b -> Some (F32 (a *. b))
  | Div, F32 a, F32 b -> Some (F32 (a /. b))
  | Eq, F32 a, F32 b -> Some (nat_cmp (a = b))
  | Ne, F32 a, F32 b -> Some (nat_cmp (a <> b))
  | Lt, F32 a, F32 b -> Some (nat_cmp (a < b))
  | Le, F32 a, F32 b -> Some (nat_cmp (a <= b))
  | Gt, F32 a, F32 b -> Some (nat_cmp (a > b))
  | Ge, F32 a, F32 b -> Some (nat_cmp (a >= b))
  (* F64 arithmetic *)
  | Add, F64 a, F64 b -> Some (F64 (a +. b))
  | Sub, F64 a, F64 b -> Some (F64 (a -. b))
  | Mul, F64 a, F64 b -> Some (F64 (a *. b))
  | Div, F64 a, F64 b -> Some (F64 (a /. b))
  | Eq, F64 a, F64 b -> Some (nat_cmp (a = b))
  | Ne, F64 a, F64 b -> Some (nat_cmp (a <> b))
  | Lt, F64 a, F64 b -> Some (nat_cmp (a < b))
  | Le, F64 a, F64 b -> Some (nat_cmp (a <= b))
  | Gt, F64 a, F64 b -> Some (nat_cmp (a > b))
  | Ge, F64 a, F64 b -> Some (nat_cmp (a >= b))
  | _, _, _ -> None

(* ======================================================================= *)
(** * 5. [agpu_eval_ir] — mirrors AGpuSemantics.agpu_eval_ir *)
(* ======================================================================= *)

let rec agpu_eval_ir st = function
  | IEConst (CInt32 n) -> Some (U32 n, st)
  | IEConst (CInt64 n) -> Some (U64 n, st)
  | IEConst (CFloat32 f) -> Some (F32 f, st)
  | IEConst (CFloat64 f) -> Some (F64 f, st)
  | IEConst (CBool b) -> Some (U32 (if b then 1 else 0), st)
  | IEConst CUnit -> Some (U32 0, st)
  | IEVar name -> (
      match st.regs name with Some v -> Some (v, st) | None -> None)
  | IEBinop (op, e1, e2) -> (
      match agpu_eval_ir st e1 with
      | None -> None
      | Some (v1, st1) -> (
          match agpu_eval_ir st1 e2 with
          | None -> None
          | Some (v2, st2) -> (
              match agpu_eval_binop op v1 v2 with
              | None -> None
              | Some v -> Some (v, st2))))
  | IEArrayRead (MS_Global, base_e, idx_e) -> (
      match agpu_eval_ir st base_e with
      | Some (U32 base, st1) -> (
          match agpu_eval_ir st1 idx_e with
          | Some (U32 idx, st2) -> Some (st2.mem.global_mem (base + idx), st2)
          | _ -> None)
      | Some (U64 base, st1) -> (
          match agpu_eval_ir st1 idx_e with
          | Some (U64 idx, st2) -> Some (st2.mem.global_mem (base + idx), st2)
          | _ -> None)
      | _ -> None)
  | IEArrayRead (MS_Shared, base_e, idx_e) -> (
      match agpu_eval_ir st base_e with
      | Some (U32 base, st1) -> (
          match agpu_eval_ir st1 idx_e with
          | Some (U32 idx, st2) -> Some (st2.mem.shared_mem (base + idx), st2)
          | _ -> None)
      | Some (U64 base, st1) -> (
          match agpu_eval_ir st1 idx_e with
          | Some (U64 idx, st2) -> Some (st2.mem.shared_mem (base + idx), st2)
          | _ -> None)
      | _ -> None)
  | IEThreadIdxX -> Some (U32 st.tc.tidx, st)
  | IEBlockIdxX -> Some (U32 st.tc.bidx, st)
  | IEBlockDimX -> Some (U32 st.tc.bdim, st)
  | IEGlobalIdx ->
      let gid = (st.tc.bidx * st.tc.bdim) + st.tc.tidx in
      Some (U32 gid, st)
  | IEBarrier -> Some (U32 0, st)
  | IESin32 e -> (
      match agpu_eval_ir st e with
      | Some (F32 x, st1) -> Some (F32 (Float.sin x), st1)
      | _ -> None)
  | IECos32 e -> (
      match agpu_eval_ir st e with
      | Some (F32 x, st1) -> Some (F32 (Float.cos x), st1)
      | _ -> None)
  | IESqrt32 e -> (
      match agpu_eval_ir st e with
      | Some (F32 x, st1) -> Some (F32 (Float.sqrt x), st1)
      | _ -> None)
  | IEFabs32 e -> (
      match agpu_eval_ir st e with
      | Some (F32 x, st1) -> Some (F32 (Float.abs x), st1)
      | _ -> None)
  | IEFma32 (ea, eb, ec) -> (
      match agpu_eval_ir st ea with
      | Some (F32 a, st1) -> (
          match agpu_eval_ir st1 eb with
          | Some (F32 b, st2) -> (
              match agpu_eval_ir st2 ec with
              | Some (F32 c, st3) -> Some (F32 (Float.fma a b c), st3)
              | _ -> None)
          | _ -> None)
      | _ -> None)
  | IESin64 e -> (
      match agpu_eval_ir st e with
      | Some (F64 x, st1) -> Some (F64 (Float.sin x), st1)
      | _ -> None)
  | IECos64 e -> (
      match agpu_eval_ir st e with
      | Some (F64 x, st1) -> Some (F64 (Float.cos x), st1)
      | _ -> None)
  | IESqrt64 e -> (
      match agpu_eval_ir st e with
      | Some (F64 x, st1) -> Some (F64 (Float.sqrt x), st1)
      | _ -> None)
  | IEFabs64 e -> (
      match agpu_eval_ir st e with
      | Some (F64 x, st1) -> Some (F64 (Float.abs x), st1)
      | _ -> None)
  | IEFma64 (ea, eb, ec) -> (
      match agpu_eval_ir st ea with
      | Some (F64 a, st1) -> (
          match agpu_eval_ir st1 eb with
          | Some (F64 b, st2) -> (
              match agpu_eval_ir st2 ec with
              | Some (F64 c, st3) -> Some (F64 (Float.fma a b c), st3)
              | _ -> None)
          | _ -> None)
      | _ -> None)

(* ======================================================================= *)
(** * 6. PTX expression AST — mirrors PtxTypes.ptx_expr_ast *)
(* ======================================================================= *)

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

type ptx_cmp_tag = PEq | PNe | PLt | PLe | PGt | PGe

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

type ptx_expr_ast =
  | PtxLitU32 of int
  | PtxLitU64 of int
  | PtxLitF32 of float
  | PtxLitF64 of float
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

(* ======================================================================= *)
(** * 7. PTX evaluators *)
(* ======================================================================= *)

let agpu_eval_ptx_binop op v1 v2 =
  match (op, v1, v2) with
  | PAdd, U32 a, U32 b -> Some (U32 (a + b))
  | PSub, U32 a, U32 b -> Some (U32 (max 0 (a - b)))
  | PMul, U32 a, U32 b -> Some (U32 (a * b))
  | PDiv, U32 a, U32 b -> Some (U32 (if b = 0 then 0 else a / b))
  | PMod, U32 a, U32 b -> Some (U32 (if b = 0 then 0 else a mod b))
  | PAnd, U32 a, U32 b -> Some (U32 (a land b))
  | POr, U32 a, U32 b -> Some (U32 (a lor b))
  | PBitAnd, U32 a, U32 b -> Some (U32 (a land b))
  | PBitOr, U32 a, U32 b -> Some (U32 (a lor b))
  | PBitXor, U32 a, U32 b -> Some (U32 (a lxor b))
  | PShl, U32 a, U32 b -> Some (U32 (a lsl b))
  | PShr, U32 a, U32 b -> Some (U32 (a lsr b))
  | PAdd, U64 a, U64 b -> Some (U64 (a + b))
  | PSub, U64 a, U64 b -> Some (U64 (max 0 (a - b)))
  | PMul, U64 a, U64 b -> Some (U64 (a * b))
  | PDiv, U64 a, U64 b -> Some (U64 (if b = 0 then 0 else a / b))
  | PMod, U64 a, U64 b -> Some (U64 (if b = 0 then 0 else a mod b))
  | PAdd, F32 a, F32 b -> Some (F32 (a +. b))
  | PSub, F32 a, F32 b -> Some (F32 (a -. b))
  | PMul, F32 a, F32 b -> Some (F32 (a *. b))
  | PDiv, F32 a, F32 b -> Some (F32 (a /. b))
  | PAdd, F64 a, F64 b -> Some (F64 (a +. b))
  | PSub, F64 a, F64 b -> Some (F64 (a -. b))
  | PMul, F64 a, F64 b -> Some (F64 (a *. b))
  | PDiv, F64 a, F64 b -> Some (F64 (a /. b))
  | _, _, _ -> None

let agpu_eval_ptx_cmp op v1 v2 =
  match (op, v1, v2) with
  | PEq, U32 a, U32 b -> Some (nat_cmp (a = b))
  | PNe, U32 a, U32 b -> Some (nat_cmp (a <> b))
  | PLt, U32 a, U32 b -> Some (nat_cmp (a < b))
  | PLe, U32 a, U32 b -> Some (nat_cmp (a <= b))
  | PGt, U32 a, U32 b -> Some (nat_cmp (a > b))
  | PGe, U32 a, U32 b -> Some (nat_cmp (a >= b))
  | PEq, U64 a, U64 b -> Some (nat_cmp (a = b))
  | PNe, U64 a, U64 b -> Some (nat_cmp (a <> b))
  | PLt, U64 a, U64 b -> Some (nat_cmp (a < b))
  | PLe, U64 a, U64 b -> Some (nat_cmp (a <= b))
  | PGt, U64 a, U64 b -> Some (nat_cmp (a > b))
  | PGe, U64 a, U64 b -> Some (nat_cmp (a >= b))
  | PEq, F32 a, F32 b -> Some (nat_cmp (a = b))
  | PNe, F32 a, F32 b -> Some (nat_cmp (a <> b))
  | PLt, F32 a, F32 b -> Some (nat_cmp (a < b))
  | PLe, F32 a, F32 b -> Some (nat_cmp (a <= b))
  | PGt, F32 a, F32 b -> Some (nat_cmp (a > b))
  | PGe, F32 a, F32 b -> Some (nat_cmp (a >= b))
  | PEq, F64 a, F64 b -> Some (nat_cmp (a = b))
  | PNe, F64 a, F64 b -> Some (nat_cmp (a <> b))
  | PLt, F64 a, F64 b -> Some (nat_cmp (a < b))
  | PLe, F64 a, F64 b -> Some (nat_cmp (a <= b))
  | PGt, F64 a, F64 b -> Some (nat_cmp (a > b))
  | PGe, F64 a, F64 b -> Some (nat_cmp (a >= b))
  | _, _, _ -> None

let agpu_eval_ptx_intrinsic tag v =
  match (tag, v) with
  | PISin32, F32 x -> Some (F32 (Float.sin x))
  | PICos32, F32 x -> Some (F32 (Float.cos x))
  | PISqrt32, F32 x -> Some (F32 (Float.sqrt x))
  | PIFabs32, F32 x -> Some (F32 (Float.abs x))
  | PISin64, F64 x -> Some (F64 (Float.sin x))
  | PICos64, F64 x -> Some (F64 (Float.cos x))
  | PISqrt64, F64 x -> Some (F64 (Float.sqrt x))
  | PIFabs64, F64 x -> Some (F64 (Float.abs x))
  | _, _ -> None

let rec agpu_eval_ptx st = function
  | PtxLitU32 n -> Some (U32 n, st)
  | PtxLitU64 n -> Some (U64 n, st)
  | PtxLitF32 f -> Some (F32 f, st)
  | PtxLitF64 f -> Some (F64 f, st)
  | PtxReg name -> (
      match st.regs name with Some v -> Some (v, st) | None -> None)
  | PtxBinop (op, e1, e2) -> (
      match agpu_eval_ptx st e1 with
      | None -> None
      | Some (v1, st1) -> (
          match agpu_eval_ptx st1 e2 with
          | None -> None
          | Some (v2, st2) -> (
              match agpu_eval_ptx_binop op v1 v2 with
              | None -> None
              | Some v -> Some (v, st2))))
  | PtxCmp (op, e1, e2) -> (
      match agpu_eval_ptx st e1 with
      | None -> None
      | Some (v1, st1) -> (
          match agpu_eval_ptx st1 e2 with
          | None -> None
          | Some (v2, st2) -> (
              match agpu_eval_ptx_cmp op v1 v2 with
              | None -> None
              | Some v -> Some (v, st2))))
  | PtxGlobalRead addr_e -> (
      match agpu_eval_ptx st addr_e with
      | Some (U32 a, st1) -> Some (st1.mem.global_mem a, st1)
      | Some (U64 a, st1) -> Some (st1.mem.global_mem a, st1)
      | _ -> None)
  | PtxSharedRead addr_e -> (
      match agpu_eval_ptx st addr_e with
      | Some (U32 a, st1) -> Some (st1.mem.shared_mem a, st1)
      | Some (U64 a, st1) -> Some (st1.mem.shared_mem a, st1)
      | _ -> None)
  | PtxTidx -> Some (U32 st.tc.tidx, st)
  | PtxBidx -> Some (U32 st.tc.bidx, st)
  | PtxBdim -> Some (U32 st.tc.bdim, st)
  | PtxIntrinsic (tag, e) -> (
      match agpu_eval_ptx st e with
      | None -> None
      | Some (v, st1) -> (
          match agpu_eval_ptx_intrinsic tag v with
          | None -> None
          | Some r -> Some (r, st1)))
  | PtxFma32 (ea, eb, ec) -> (
      match agpu_eval_ptx st ea with
      | Some (F32 a, st1) -> (
          match agpu_eval_ptx st1 eb with
          | Some (F32 b, st2) -> (
              match agpu_eval_ptx st2 ec with
              | Some (F32 c, st3) -> Some (F32 (Float.fma a b c), st3)
              | _ -> None)
          | _ -> None)
      | _ -> None)
  | PtxFma64 (ea, eb, ec) -> (
      match agpu_eval_ptx st ea with
      | Some (F64 a, st1) -> (
          match agpu_eval_ptx st1 eb with
          | Some (F64 b, st2) -> (
              match agpu_eval_ptx st2 ec with
              | Some (F64 c, st3) -> Some (F64 (Float.fma a b c), st3)
              | _ -> None)
          | _ -> None)
      | _ -> None)

(* ======================================================================= *)
(** * 8. [emit_ast_expr] — mirrors PtxExprSpec.emit_ast_expr *)
(* ======================================================================= *)

let is_cmp_op = function Eq | Ne | Lt | Le | Gt | Ge -> true | _ -> false

let ir_binop_to_ptx_binop = function
  | Add -> PAdd
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
  | _ -> PAdd (* unreachable *)

let ir_binop_to_ptx_cmp = function
  | Eq -> PEq
  | Ne -> PNe
  | Lt -> PLt
  | Le -> PLe
  | Gt -> PGt
  | Ge -> PGe
  | _ -> PEq (* unreachable *)

let rec emit_ast_expr = function
  | IEConst (CInt32 n) -> PtxLitU32 n
  | IEConst (CInt64 n) -> PtxLitU64 n
  | IEConst (CFloat32 f) -> PtxLitF32 f
  | IEConst (CFloat64 f) -> PtxLitF64 f
  | IEConst (CBool b) -> PtxLitU32 (if b then 1 else 0)
  | IEConst CUnit -> PtxLitU32 0
  | IEVar name -> PtxReg name
  | IEBinop (op, e1, e2) ->
      if is_cmp_op op then
        PtxCmp (ir_binop_to_ptx_cmp op, emit_ast_expr e1, emit_ast_expr e2)
      else
        PtxBinop (ir_binop_to_ptx_binop op, emit_ast_expr e1, emit_ast_expr e2)
  | IEArrayRead (MS_Global, base_e, idx_e) ->
      PtxGlobalRead (PtxBinop (PAdd, emit_ast_expr base_e, emit_ast_expr idx_e))
  | IEArrayRead (MS_Shared, base_e, idx_e) ->
      PtxSharedRead (PtxBinop (PAdd, emit_ast_expr base_e, emit_ast_expr idx_e))
  | IEThreadIdxX -> PtxTidx
  | IEBlockIdxX -> PtxBidx
  | IEBlockDimX -> PtxBdim
  | IEGlobalIdx -> PtxBinop (PAdd, PtxBinop (PMul, PtxBidx, PtxBdim), PtxTidx)
  | IEBarrier -> PtxLitU32 0
  | IESin32 e -> PtxIntrinsic (PISin32, emit_ast_expr e)
  | IECos32 e -> PtxIntrinsic (PICos32, emit_ast_expr e)
  | IESqrt32 e -> PtxIntrinsic (PISqrt32, emit_ast_expr e)
  | IEFabs32 e -> PtxIntrinsic (PIFabs32, emit_ast_expr e)
  | IEFma32 (ea, eb, ec) ->
      PtxFma32 (emit_ast_expr ea, emit_ast_expr eb, emit_ast_expr ec)
  | IESin64 e -> PtxIntrinsic (PISin64, emit_ast_expr e)
  | IECos64 e -> PtxIntrinsic (PICos64, emit_ast_expr e)
  | IESqrt64 e -> PtxIntrinsic (PISqrt64, emit_ast_expr e)
  | IEFabs64 e -> PtxIntrinsic (PIFabs64, emit_ast_expr e)
  | IEFma64 (ea, eb, ec) ->
      PtxFma64 (emit_ast_expr ea, emit_ast_expr eb, emit_ast_expr ec)

(* ======================================================================= *)
(** * 9. Helpers for tests *)
(* ======================================================================= *)

(** [ptx_val_eq]: structural equality for test assertions. *)
let ptx_val_eq v1 v2 =
  match (v1, v2) with
  | U32 a, U32 b -> a = b
  | U64 a, U64 b -> a = b
  | F32 a, F32 b -> Float.equal a b
  | F64 a, F64 b -> Float.equal a b
  | Pred a, Pred b -> a = b
  | _, _ -> false

(** [eval_agrees e st]: the key correctness check. Returns [true] iff
    [agpu_eval_ir st e] and [agpu_eval_ptx st (emit_ast_expr e)] agree on the
    result value (both None, or both Some with equal ptx_val). *)
let eval_agrees e st =
  let ir_result = agpu_eval_ir st e in
  let ptx_result = agpu_eval_ptx st (emit_ast_expr e) in
  match (ir_result, ptx_result) with
  | None, None -> true
  | Some (v1, _), Some (v2, _) -> ptx_val_eq v1 v2
  | _, _ -> false

(* ======================================================================= *)
(** * 10. Alcotest tests *)
(* ======================================================================= *)

let st0 = make_state ()

let st_thread = make_state ~tidx:3 ~bidx:2 ~bdim:32 ()

(** Helper: assert that [agpu_eval_ir] returns [Some (expected, _)] *)
let expect_ir_val e st expected =
  match agpu_eval_ir st e with
  | None -> Alcotest.fail "expected Some but got None"
  | Some (v, _) ->
      if ptx_val_eq v expected then ()
      else
        Alcotest.failf
          "expected %s but got different value"
          (match expected with
          | U32 n -> Printf.sprintf "U32 %d" n
          | U64 n -> Printf.sprintf "U64 %d" n
          | F32 f -> Printf.sprintf "F32 %g" f
          | F64 f -> Printf.sprintf "F64 %g" f
          | Pred b -> Printf.sprintf "Pred %b" b)

(** [check_emit_agree e st msg]: assert [eval_agrees e st] *)
let check_emit_agree e st msg =
  if not (eval_agrees e st) then
    Alcotest.failf "emit_expr_correct violated: %s" msg

(* ---- Group 1: Literals ---- *)
let test_lit_u32 () =
  expect_ir_val (IEConst (CInt32 42)) st0 (U32 42) ;
  check_emit_agree (IEConst (CInt32 42)) st0 "CInt32 42"

let test_lit_u64 () =
  expect_ir_val (IEConst (CInt64 999)) st0 (U64 999) ;
  check_emit_agree (IEConst (CInt64 999)) st0 "CInt64 999"

let test_lit_f32 () =
  expect_ir_val (IEConst (CFloat32 3.14)) st0 (F32 3.14) ;
  check_emit_agree (IEConst (CFloat32 3.14)) st0 "CFloat32 3.14"

let test_lit_f64 () =
  expect_ir_val (IEConst (CFloat64 2.718)) st0 (F64 2.718) ;
  check_emit_agree (IEConst (CFloat64 2.718)) st0 "CFloat64 2.718"

let test_lit_bool_true () =
  expect_ir_val (IEConst (CBool true)) st0 (U32 1) ;
  check_emit_agree (IEConst (CBool true)) st0 "CBool true"

let test_lit_bool_false () =
  expect_ir_val (IEConst (CBool false)) st0 (U32 0) ;
  check_emit_agree (IEConst (CBool false)) st0 "CBool false"

let test_lit_unit () =
  expect_ir_val (IEConst CUnit) st0 (U32 0) ;
  check_emit_agree (IEConst CUnit) st0 "CUnit"

(* ---- Group 2: Thread intrinsics ---- *)
let test_thread_tidx () =
  expect_ir_val IEThreadIdxX st_thread (U32 3) ;
  check_emit_agree IEThreadIdxX st_thread "IEThreadIdxX"

let test_thread_bidx () =
  expect_ir_val IEBlockIdxX st_thread (U32 2) ;
  check_emit_agree IEBlockIdxX st_thread "IEBlockIdxX"

let test_thread_bdim () =
  expect_ir_val IEBlockDimX st_thread (U32 32) ;
  check_emit_agree IEBlockDimX st_thread "IEBlockDimX"

let test_global_idx () =
  (* global_idx = bidx * bdim + tidx = 2*32+3 = 67 *)
  expect_ir_val IEGlobalIdx st_thread (U32 67) ;
  check_emit_agree IEGlobalIdx st_thread "IEGlobalIdx"

(* ---- Group 3: Binary arithmetic ---- *)
let test_add_u32 () =
  let e = IEBinop (Add, IEConst (CInt32 10), IEConst (CInt32 20)) in
  expect_ir_val e st0 (U32 30) ;
  check_emit_agree e st0 "Add U32"

let test_sub_u32 () =
  let e = IEBinop (Sub, IEConst (CInt32 10), IEConst (CInt32 3)) in
  expect_ir_val e st0 (U32 7) ;
  check_emit_agree e st0 "Sub U32"

let test_mul_u32 () =
  let e = IEBinop (Mul, IEConst (CInt32 6), IEConst (CInt32 7)) in
  expect_ir_val e st0 (U32 42) ;
  check_emit_agree e st0 "Mul U32"

let test_add_f32 () =
  let e = IEBinop (Add, IEConst (CFloat32 1.5), IEConst (CFloat32 2.5)) in
  expect_ir_val e st0 (F32 4.0) ;
  check_emit_agree e st0 "Add F32"

(* ---- Group 4: Comparisons ---- *)
let test_eq_u32_true () =
  let e = IEBinop (Eq, IEConst (CInt32 5), IEConst (CInt32 5)) in
  expect_ir_val e st0 (U32 1) ;
  check_emit_agree e st0 "Eq U32 true"

let test_eq_u32_false () =
  let e = IEBinop (Eq, IEConst (CInt32 5), IEConst (CInt32 6)) in
  expect_ir_val e st0 (U32 0) ;
  check_emit_agree e st0 "Eq U32 false"

let test_lt_u32 () =
  let e = IEBinop (Lt, IEConst (CInt32 3), IEConst (CInt32 5)) in
  expect_ir_val e st0 (U32 1) ;
  check_emit_agree e st0 "Lt U32"

let test_le_u32 () =
  let e = IEBinop (Le, IEConst (CInt32 5), IEConst (CInt32 5)) in
  expect_ir_val e st0 (U32 1) ;
  check_emit_agree e st0 "Le U32"

(* ---- Group 5: Math intrinsics ---- *)
let test_sin32 () =
  let x = 0.5 in
  let e = IESin32 (IEConst (CFloat32 x)) in
  expect_ir_val e st0 (F32 (Float.sin x)) ;
  check_emit_agree e st0 "IESin32"

let test_cos32 () =
  let x = 0.0 in
  let e = IECos32 (IEConst (CFloat32 x)) in
  expect_ir_val e st0 (F32 (Float.cos x)) ;
  check_emit_agree e st0 "IECos32"

let test_sqrt32 () =
  let x = 4.0 in
  let e = IESqrt32 (IEConst (CFloat32 x)) in
  expect_ir_val e st0 (F32 (Float.sqrt x)) ;
  check_emit_agree e st0 "IESqrt32"

let test_fabs32 () =
  let x = -3.0 in
  let e = IEFabs32 (IEConst (CFloat32 x)) in
  expect_ir_val e st0 (F32 (Float.abs x)) ;
  check_emit_agree e st0 "IEFabs32"

let test_fma32 () =
  let a, b, c = (2.0, 3.0, 1.0) in
  let e =
    IEFma32 (IEConst (CFloat32 a), IEConst (CFloat32 b), IEConst (CFloat32 c))
  in
  expect_ir_val e st0 (F32 (Float.fma a b c)) ;
  check_emit_agree e st0 "IEFma32"

(* ---- Group 6: Type-mismatch gives None ---- *)
let test_sin32_wrong_type () =
  (* IESin32 applied to a U32 value must return None (wrong type) *)
  let e = IESin32 (IEConst (CInt32 1)) in
  match agpu_eval_ir st0 e with
  | None -> ()
  | Some _ -> Alcotest.fail "IESin32(U32) should give None"

let test_sin64_correct_type () =
  let x = 0.5 in
  let e = IESin64 (IEConst (CFloat64 x)) in
  expect_ir_val e st0 (F64 (Float.sin x)) ;
  check_emit_agree e st0 "IESin64"

let test_sin64_wrong_type () =
  (* IESin64 applied to a F32 value must return None (wrong type) *)
  let e = IESin64 (IEConst (CFloat32 0.5)) in
  match agpu_eval_ir st0 e with
  | None -> ()
  | Some _ -> Alcotest.fail "IESin64(F32) should give None"

(* ---- Group 7: Register reads ---- *)
let test_var_found () =
  let st =
    {st0 with regs = (fun n -> if n = "x" then Some (U32 7) else None)}
  in
  expect_ir_val (IEVar "x") st (U32 7) ;
  check_emit_agree (IEVar "x") st "IEVar found"

let test_var_not_found () =
  match agpu_eval_ir st0 (IEVar "y") with
  | None -> ()
  | Some _ -> Alcotest.fail "IEVar missing register should give None"

(* ---- Group 8: Barrier ---- *)
let test_barrier () =
  expect_ir_val IEBarrier st0 (U32 0) ;
  check_emit_agree IEBarrier st0 "IEBarrier"

(* ======================================================================= *)
(** * 11. Run all tests *)
(* ======================================================================= *)

let () =
  Alcotest.run
    "codegen-ptx"
    [
      ( "literals",
        [
          Alcotest.test_case "CInt32" `Quick test_lit_u32;
          Alcotest.test_case "CInt64" `Quick test_lit_u64;
          Alcotest.test_case "CFloat32" `Quick test_lit_f32;
          Alcotest.test_case "CFloat64" `Quick test_lit_f64;
          Alcotest.test_case "CBool-true" `Quick test_lit_bool_true;
          Alcotest.test_case "CBool-false" `Quick test_lit_bool_false;
          Alcotest.test_case "CUnit" `Quick test_lit_unit;
        ] );
      ( "thread-intrinsics",
        [
          Alcotest.test_case "tidx" `Quick test_thread_tidx;
          Alcotest.test_case "bidx" `Quick test_thread_bidx;
          Alcotest.test_case "bdim" `Quick test_thread_bdim;
          Alcotest.test_case "global-idx" `Quick test_global_idx;
        ] );
      ( "arithmetic",
        [
          Alcotest.test_case "add-u32" `Quick test_add_u32;
          Alcotest.test_case "sub-u32" `Quick test_sub_u32;
          Alcotest.test_case "mul-u32" `Quick test_mul_u32;
          Alcotest.test_case "add-f32" `Quick test_add_f32;
        ] );
      ( "comparisons",
        [
          Alcotest.test_case "eq-u32-true" `Quick test_eq_u32_true;
          Alcotest.test_case "eq-u32-false" `Quick test_eq_u32_false;
          Alcotest.test_case "lt-u32" `Quick test_lt_u32;
          Alcotest.test_case "le-u32" `Quick test_le_u32;
        ] );
      ( "math-intrinsics",
        [
          Alcotest.test_case "sin32" `Quick test_sin32;
          Alcotest.test_case "cos32" `Quick test_cos32;
          Alcotest.test_case "sqrt32" `Quick test_sqrt32;
          Alcotest.test_case "fabs32" `Quick test_fabs32;
          Alcotest.test_case "fma32" `Quick test_fma32;
          Alcotest.test_case "sin64" `Quick test_sin64_correct_type;
        ] );
      ( "type-safety",
        [
          Alcotest.test_case "sin32-wrong-type" `Quick test_sin32_wrong_type;
          Alcotest.test_case "sin64-wrong-type" `Quick test_sin64_wrong_type;
        ] );
      ( "registers",
        [
          Alcotest.test_case "var-found" `Quick test_var_found;
          Alcotest.test_case "var-missing" `Quick test_var_not_found;
        ] );
      ("barrier", [Alcotest.test_case "barrier" `Quick test_barrier]);
    ]
