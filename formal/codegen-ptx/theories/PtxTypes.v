(** PtxTypes.v — PTX register type model for Sarek PTX code generation.
 *
 * Defines the PTX register type enumeration [ptx_type], a structured PTX
 * expression AST [ptx_expr_ast], the total type-mapping function
 * [ptx_reg_type_of], and the AST evaluator [agpu_eval_ptx].
 *
 * Design notes:
 * - [ptx_reg_type_of] mirrors [Sarek_ir_ptx_types.ml:ptx_reg_type_of].
 *   The OCaml function raises [unsupported] for TRecord and TVariant; those
 *   are excluded from the "covered" domain.  The Rocq function is total on
 *   a [covered_elttype] predicate that precisely captures the covered subset.
 * - [ptx_expr_ast] is a Rocq-only structured AST; the OCaml emitter remains
 *   string-based.  Correctness of the emitter is proved by relating
 *   [agpu_eval_ptx] (this module) to [agpu_eval_ir] (AGpuSemantics) in
 *   PtxExprSpec.v.
 * - No [Admitted] is used anywhere in this file.
 *)

From CodegenPtx Require Import AGpuSemantics.
From Stdlib Require Import Strings.String.
From Stdlib Require Import Bool.Bool.
From Stdlib Require Import Arith.Arith.
From Stdlib Require Import Floats.

Open Scope string_scope.

(* ------------------------------------------------------------------ *)
(** * PTX register type enumeration
 *
 * Mirrors the PTX ISA register classes used by the Sarek PTX emitter:
 * - [PTX_U32]  — 32-bit unsigned integer  (.u32 in PTX)
 * - [PTX_U64]  — 64-bit unsigned integer  (.u64 in PTX)
 * - [PTX_F32]  — 32-bit floating point    (.f32 in PTX)
 * - [PTX_F64]  — 64-bit floating point    (.f64 in PTX)
 * - [PTX_Pred] — predicate register       (.pred in PTX)
 *)
(* ------------------------------------------------------------------ *)

Inductive ptx_type :=
  | PTX_U32
  | PTX_U64
  | PTX_F32
  | PTX_F64
  | PTX_Pred.

(* ------------------------------------------------------------------ *)
(** * Extended element type
 *
 * [AGpuSemantics.ir_elttype] covers the five scalar types that appear in
 * IR expressions.  The PTX emitter additionally handles [TUnit], [TVec],
 * and [TArray] at the statement/kernel level.  We define [elttype] here as
 * a superset so that [ptx_reg_type_of] can be stated over the full domain
 * that the OCaml function supports without raising [unsupported].
 *)
(* ------------------------------------------------------------------ *)

Inductive elttype :=
  | ET_Int32
  | ET_Int64
  | ET_Float32
  | ET_Float64
  | ET_Bool
  | ET_Unit
  | ET_Vec    : elttype -> elttype
  | ET_Array  : elttype -> elttype.

(* ------------------------------------------------------------------ *)
(** * PTX binary operator tags
 *
 * Used in [PtxBinop] and [PtxCmp] nodes of [ptx_expr_ast].
 *)
(* ------------------------------------------------------------------ *)

(** Arithmetic / bitwise operators that appear in PTX instructions. *)
Inductive ptx_binop_tag :=
  | PAdd | PSub | PMul | PDiv | PMod
  | PAnd | POr
  | PShl | PShr
  | PBitAnd | PBitOr | PBitXor.

(** Comparison operators for setp instructions. *)
Inductive ptx_cmp_tag :=
  | PEq | PNe | PLt | PLe | PGt | PGe.

(** Math intrinsic tags (unary, operating on a specific element type).
 *  f32 and f64 variants are kept separate so that the evaluation function
 *  is type-unambiguous and [eval_ir_ptx_eq] can be proved without a typing
 *  predicate.  Each tag operates on exactly one [ptx_val] constructor.
 *)
Inductive ptx_intrinsic_tag :=
  | PISin32  | PICos32  | PISqrt32  | PIFabs32
  | PISin64  | PICos64  | PISqrt64  | PIFabs64
  | PIFma.   (* FMA is ternary and handled by PtxFma32/PtxFma64 *)

(* ------------------------------------------------------------------ *)
(** * Structured PTX expression AST
 *
 * This is the Rocq-only representation that [agpu_eval_ptx] evaluates.
 * The OCaml emitter is string-based and is not extracted from this type.
 * Correctness is proved in PtxExprSpec.v by showing that evaluating the
 * AST produced by [emit_ast_expr] agrees with [agpu_eval_ir].
 *
 * Constructors:
 * - [PtxLitU32 n]           — U32 immediate
 * - [PtxLitF32 f]           — F32 immediate
 * - [PtxReg name]            — read a named register
 * - [PtxBinop op e1 e2]      — binary arithmetic / bitwise / logical
 * - [PtxCmp op e1 e2]        — comparison (setp), yields U32 0/1
 * - [PtxGlobalRead addr]     — load from global memory at address [addr]
 * - [PtxSharedRead addr]     — load from shared memory at address [addr]
 * - [PtxTidx]                — threadIdx.x
 * - [PtxBidx]                — blockIdx.x
 * - [PtxBdim]                — blockDim.x
 * - [PtxIntrinsic tag e]     — unary math intrinsic applied to [e]
 *)
(* ------------------------------------------------------------------ *)

Inductive ptx_expr_ast :=
  | PtxLitU32    : nat            -> ptx_expr_ast
  | PtxLitU64    : nat            -> ptx_expr_ast   (** .u64 immediate *)
  | PtxLitF32    : float          -> ptx_expr_ast
  | PtxLitF64    : float          -> ptx_expr_ast   (** .f64 immediate *)
  | PtxReg       : string         -> ptx_expr_ast
  | PtxBinop     : ptx_binop_tag  -> ptx_expr_ast -> ptx_expr_ast -> ptx_expr_ast
  | PtxCmp       : ptx_cmp_tag    -> ptx_expr_ast -> ptx_expr_ast -> ptx_expr_ast
  | PtxGlobalRead : ptx_expr_ast  -> ptx_expr_ast
  | PtxSharedRead : ptx_expr_ast  -> ptx_expr_ast
  | PtxTidx      : ptx_expr_ast
  | PtxBidx      : ptx_expr_ast
  | PtxBdim      : ptx_expr_ast
  | PtxIntrinsic  : ptx_intrinsic_tag -> ptx_expr_ast -> ptx_expr_ast
  (** Ternary fused multiply-add; separate from [PtxIntrinsic] because FMA
      needs three sub-expressions and [agpu_eval_ptx_intrinsic] is unary. *)
  | PtxFma32     : ptx_expr_ast -> ptx_expr_ast -> ptx_expr_ast -> ptx_expr_ast
  | PtxFma64     : ptx_expr_ast -> ptx_expr_ast -> ptx_expr_ast -> ptx_expr_ast.

(* ------------------------------------------------------------------ *)
(** * [ptx_reg_type_of] — element type → PTX register class
 *
 * Mirrors [Sarek_ir_ptx_types.ptx_reg_type_of]:
 *   TInt32 | TBool  → .u32   (PTX_U32)
 *   TUnit           → .u32   (PTX_U32)   [unit is represented as 0 : u32]
 *   TInt64          → .u64   (PTX_U64)
 *   TFloat32        → .f32   (PTX_F32)
 *   TFloat64        → .f64   (PTX_F64)
 *   TVec _          → .u64   (PTX_U64)   [pointer-sized]
 *   TArray _        → .u64   (PTX_U64)   [pointer-sized]
 *
 * TRecord and TVariant are excluded from [elttype] above — the OCaml
 * function raises [unsupported] for them, so they are not in the covered
 * domain.
 *)
(* ------------------------------------------------------------------ *)

Definition ptx_reg_type_of (t : elttype) : ptx_type :=
  match t with
  | ET_Int32   => PTX_U32
  | ET_Bool    => PTX_U32
  | ET_Unit    => PTX_U32
  | ET_Int64   => PTX_U64
  | ET_Float32 => PTX_F32
  | ET_Float64 => PTX_F64
  | ET_Vec _   => PTX_U64
  | ET_Array _ => PTX_U64
  end.

(* ------------------------------------------------------------------ *)
(** * Totality theorem
 *
 * [ptx_reg_type_of] returns a [ptx_type] for every [elttype], so it is
 * total by construction.  The theorem is proved by structural induction /
 * case analysis, which [auto] closes in one step because every branch of
 * the [Fixpoint] produces a [Some]-like witness (actually a plain value —
 * the function is already total, so the statement merely records the
 * design intent as a proof object).
 *)
(* ------------------------------------------------------------------ *)

(** For every covered element type [t], [ptx_reg_type_of t] is defined
    (i.e. it equals some [ptx_type]). *)
Theorem ptx_reg_type_of_total :
  forall (t : elttype), exists (p : ptx_type), ptx_reg_type_of t = p.
Proof.
  intro t.
  induction t as [| | | | | | t IHt | t IHt];
    (try (eexists; reflexivity));
    destruct IHt as [p _]; eexists; reflexivity.
Qed.

(** Concrete case equations — useful as rewrite lemmas in downstream proofs. *)
Lemma ptx_reg_type_of_int32  : ptx_reg_type_of ET_Int32   = PTX_U32. Proof. reflexivity. Qed.
Lemma ptx_reg_type_of_bool   : ptx_reg_type_of ET_Bool    = PTX_U32. Proof. reflexivity. Qed.
Lemma ptx_reg_type_of_unit   : ptx_reg_type_of ET_Unit    = PTX_U32. Proof. reflexivity. Qed.
Lemma ptx_reg_type_of_int64  : ptx_reg_type_of ET_Int64   = PTX_U64. Proof. reflexivity. Qed.
Lemma ptx_reg_type_of_float32: ptx_reg_type_of ET_Float32 = PTX_F32. Proof. reflexivity. Qed.
Lemma ptx_reg_type_of_float64: ptx_reg_type_of ET_Float64 = PTX_F64. Proof. reflexivity. Qed.
Lemma ptx_reg_type_of_vec    : forall t, ptx_reg_type_of (ET_Vec t) = PTX_U64. Proof. reflexivity. Qed.
Lemma ptx_reg_type_of_array  : forall t, ptx_reg_type_of (ET_Array t) = PTX_U64. Proof. reflexivity. Qed.

(* ------------------------------------------------------------------ *)
(** * [agpu_eval_ptx] — evaluate a [ptx_expr_ast] in an [agpu_state]
 *
 * Signature matches [agpu_eval_ir] in AGpuSemantics so that PtxExprSpec.v
 * can state the correctness theorem as a direct equality.
 *
 * State threading follows the same convention as [agpu_eval_ir]: expressions
 * are pure (they do not write registers or memory), but state is threaded
 * through so that sub-expression evaluation can propagate future stateful
 * extensions without breaking the interface.
 *)
(* ------------------------------------------------------------------ *)

(** Helper: evaluate a binop tag on two [ptx_val]s. *)
Definition agpu_eval_ptx_binop (op : ptx_binop_tag) (v1 v2 : ptx_val)
    : option ptx_val :=
  match op, v1, v2 with
  (* ---- U32 ---- *)
  | PAdd,    U32 a, U32 b => Some (U32 (a + b))
  | PSub,    U32 a, U32 b => Some (U32 (a - b))
  | PMul,    U32 a, U32 b => Some (U32 (a * b))
  | PDiv,    U32 a, U32 b => Some (U32 (Nat.div a b))
  | PMod,    U32 a, U32 b => Some (U32 (Nat.modulo a b))
  | PAnd,    U32 a, U32 b => Some (U32 (Nat.land a b))
  | POr,     U32 a, U32 b => Some (U32 (Nat.lor  a b))
  | PBitAnd, U32 a, U32 b => Some (U32 (Nat.land a b))
  | PBitOr,  U32 a, U32 b => Some (U32 (Nat.lor  a b))
  | PBitXor, U32 a, U32 b => Some (U32 (Nat.lxor a b))
  | PShl,    U32 a, U32 b => Some (U32 (Nat.shiftl a b))
  | PShr,    U32 a, U32 b => Some (U32 (Nat.shiftr a b))
  (* ---- U64 ---- *)
  | PAdd, U64 a, U64 b => Some (U64 (a + b))
  | PSub, U64 a, U64 b => Some (U64 (a - b))
  | PMul, U64 a, U64 b => Some (U64 (a * b))
  | PDiv, U64 a, U64 b => Some (U64 (Nat.div a b))
  | PMod, U64 a, U64 b => Some (U64 (Nat.modulo a b))
  (* ---- F32 ---- *)
  | PAdd, F32 a, F32 b => Some (F32 (add a b))
  | PSub, F32 a, F32 b => Some (F32 (sub a b))
  | PMul, F32 a, F32 b => Some (F32 (mul a b))
  | PDiv, F32 a, F32 b => Some (F32 (div a b))
  (* ---- F64 ---- *)
  | PAdd, F64 a, F64 b => Some (F64 (add a b))
  | PSub, F64 a, F64 b => Some (F64 (sub a b))
  | PMul, F64 a, F64 b => Some (F64 (mul a b))
  | PDiv, F64 a, F64 b => Some (F64 (div a b))
  (* ---- unsupported combinations ---- *)
  | _, _, _ => None
  end.

(** Helper: evaluate a comparison tag on two [ptx_val]s. *)
Definition agpu_eval_ptx_cmp (op : ptx_cmp_tag) (v1 v2 : ptx_val)
    : option ptx_val :=
  match op, v1, v2 with
  (* ---- U32 ---- *)
  | PEq, U32 a, U32 b => Some (nat_cmp_to_u32 (Nat.eqb a b))
  | PNe, U32 a, U32 b => Some (nat_cmp_to_u32 (negb (Nat.eqb a b)))
  | PLt, U32 a, U32 b => Some (nat_cmp_to_u32 (Nat.ltb a b))
  | PLe, U32 a, U32 b => Some (nat_cmp_to_u32 (Nat.leb a b))
  | PGt, U32 a, U32 b => Some (nat_cmp_to_u32 (Nat.ltb b a))
  | PGe, U32 a, U32 b => Some (nat_cmp_to_u32 (Nat.leb b a))
  (* ---- U64 ---- *)
  | PEq, U64 a, U64 b => Some (nat_cmp_to_u32 (Nat.eqb a b))
  | PNe, U64 a, U64 b => Some (nat_cmp_to_u32 (negb (Nat.eqb a b)))
  | PLt, U64 a, U64 b => Some (nat_cmp_to_u32 (Nat.ltb a b))
  | PLe, U64 a, U64 b => Some (nat_cmp_to_u32 (Nat.leb a b))
  | PGt, U64 a, U64 b => Some (nat_cmp_to_u32 (Nat.ltb b a))
  | PGe, U64 a, U64 b => Some (nat_cmp_to_u32 (Nat.leb b a))
  (* ---- F32 ---- *)
  | PEq, F32 a, F32 b => Some (nat_cmp_to_u32 (eqb a b))
  | PNe, F32 a, F32 b => Some (nat_cmp_to_u32 (negb (eqb a b)))
  | PLt, F32 a, F32 b => Some (nat_cmp_to_u32 (ltb a b))
  | PLe, F32 a, F32 b => Some (nat_cmp_to_u32 (leb a b))
  | PGt, F32 a, F32 b => Some (nat_cmp_to_u32 (ltb b a))
  | PGe, F32 a, F32 b => Some (nat_cmp_to_u32 (leb b a))
  (* ---- F64 ---- *)
  | PEq, F64 a, F64 b => Some (nat_cmp_to_u32 (eqb a b))
  | PNe, F64 a, F64 b => Some (nat_cmp_to_u32 (negb (eqb a b)))
  | PLt, F64 a, F64 b => Some (nat_cmp_to_u32 (ltb a b))
  | PLe, F64 a, F64 b => Some (nat_cmp_to_u32 (leb a b))
  | PGt, F64 a, F64 b => Some (nat_cmp_to_u32 (ltb b a))
  | PGe, F64 a, F64 b => Some (nat_cmp_to_u32 (leb b a))
  (* ---- unsupported ---- *)
  | _, _, _ => None
  end.

(** Helper: apply a unary math intrinsic to a [ptx_val].
 *
 * Each tag is type-specific: [PISin32] only accepts [F32], [PISin64] only
 * [F64], etc.  This ensures that [eval_ir_ptx_eq] holds without a typing
 * predicate — the IR's [IESin32] only returns [Some] for F32 inputs, and
 * [PtxIntrinsic PISin32] also only returns [Some] for F32 inputs.
 *)
Definition agpu_eval_ptx_intrinsic (tag : ptx_intrinsic_tag) (v : ptx_val)
    : option ptx_val :=
  match tag, v with
  | PISin32,  F32 x => Some (F32 (sin_f32 x))
  | PICos32,  F32 x => Some (F32 (cos_f32 x))
  | PISqrt32, F32 x => Some (F32 (sqrt x))
  | PIFabs32, F32 x => Some (F32 (abs  x))
  | PISin64,  F64 x => Some (F64 (sin_f64 x))
  | PICos64,  F64 x => Some (F64 (cos_f64 x))
  | PISqrt64, F64 x => Some (F64 (sqrt x))
  | PIFabs64, F64 x => Some (F64 (abs  x))
  (* PtxIntrinsic PIFma is ternary — handled separately in [agpu_eval_ptx] *)
  | _, _ => None
  end.

(** Main evaluator for [ptx_expr_ast].
 *
 * [PtxIntrinsic PIFma e] is intentionally mapped to [None] here: FMA
 * takes three operands but [PtxIntrinsic] is a unary node.  Ternary FMA
 * is instead represented by [PtxFma32] / [PtxFma64] which carry three
 * sub-expressions and evaluate directly to [fma_f32] / [fma_f64].
 *)
Fixpoint agpu_eval_ptx (st : agpu_state) (e : ptx_expr_ast)
    : option (ptx_val * agpu_state) :=
  match e with

  (* ---- Literals ---- *)
  | PtxLitU32 n => Some (U32 n, st)
  | PtxLitU64 n => Some (U64 n, st)
  | PtxLitF32 f => Some (F32 f, st)
  | PtxLitF64 f => Some (F64 f, st)

  (* ---- Register read ---- *)
  | PtxReg name =>
      match (st.(regs)) name with
      | Some v => Some (v, st)
      | None   => None
      end

  (* ---- Binary operation ---- *)
  | PtxBinop op e1 e2 =>
      match agpu_eval_ptx st e1 with
      | None => None
      | Some (v1, st1) =>
          match agpu_eval_ptx st1 e2 with
          | None => None
          | Some (v2, st2) =>
              match agpu_eval_ptx_binop op v1 v2 with
              | None   => None
              | Some v => Some (v, st2)
              end
          end
      end

  (* ---- Comparison ---- *)
  | PtxCmp op e1 e2 =>
      match agpu_eval_ptx st e1 with
      | None => None
      | Some (v1, st1) =>
          match agpu_eval_ptx st1 e2 with
          | None => None
          | Some (v2, st2) =>
              match agpu_eval_ptx_cmp op v1 v2 with
              | None   => None
              | Some v => Some (v, st2)
              end
          end
      end

  (* ---- Global memory read ---- *)
  | PtxGlobalRead addr_e =>
      match agpu_eval_ptx st addr_e with
      | Some (U32 a, st1) => Some (st1.(mem).(global_mem) a, st1)
      | Some (U64 a, st1) => Some (st1.(mem).(global_mem) a, st1)
      | _ => None
      end

  (* ---- Shared memory read ---- *)
  | PtxSharedRead addr_e =>
      match agpu_eval_ptx st addr_e with
      | Some (U32 a, st1) => Some (st1.(mem).(shared_mem) a, st1)
      | Some (U64 a, st1) => Some (st1.(mem).(shared_mem) a, st1)
      | _ => None
      end

  (* ---- Thread-block intrinsics ---- *)
  | PtxTidx => Some (U32 st.(tc).(tidx), st)
  | PtxBidx => Some (U32 st.(tc).(bidx), st)
  | PtxBdim => Some (U32 st.(tc).(bdim), st)

  (* ---- Math intrinsics (unary) ---- *)
  | PtxIntrinsic tag e1 =>
      match agpu_eval_ptx st e1 with
      | None => None
      | Some (v, st1) =>
          match agpu_eval_ptx_intrinsic tag v with
          | None   => None
          | Some r => Some (r, st1)
          end
      end

  (* ---- Ternary FMA — f32 ---- *)
  | PtxFma32 ea eb ec =>
      match agpu_eval_ptx st ea with
      | Some (F32 a, st1) =>
          match agpu_eval_ptx st1 eb with
          | Some (F32 b, st2) =>
              match agpu_eval_ptx st2 ec with
              | Some (F32 c, st3) => Some (F32 (fma_f32 a b c), st3)
              | _ => None
              end
          | _ => None
          end
      | _ => None
      end

  (* ---- Ternary FMA — f64 ---- *)
  | PtxFma64 ea eb ec =>
      match agpu_eval_ptx st ea with
      | Some (F64 a, st1) =>
          match agpu_eval_ptx st1 eb with
          | Some (F64 b, st2) =>
              match agpu_eval_ptx st2 ec with
              | Some (F64 c, st3) => Some (F64 (fma_f64 a b c), st3)
              | _ => None
              end
          | _ => None
          end
      | _ => None
      end

  end.

(* ------------------------------------------------------------------ *)
(** * Sanity lemmas (proved by [reflexivity]) *)
(* ------------------------------------------------------------------ *)

Lemma agpu_eval_ptx_litu32 :
  forall n st, agpu_eval_ptx st (PtxLitU32 n) = Some (U32 n, st).
Proof. intros. reflexivity. Qed.

Lemma agpu_eval_ptx_litu64 :
  forall n st, agpu_eval_ptx st (PtxLitU64 n) = Some (U64 n, st).
Proof. intros. reflexivity. Qed.

Lemma agpu_eval_ptx_litf32 :
  forall f st, agpu_eval_ptx st (PtxLitF32 f) = Some (F32 f, st).
Proof. intros. reflexivity. Qed.

Lemma agpu_eval_ptx_litf64 :
  forall f st, agpu_eval_ptx st (PtxLitF64 f) = Some (F64 f, st).
Proof. intros. reflexivity. Qed.

Lemma agpu_eval_ptx_tidx :
  forall st, agpu_eval_ptx st PtxTidx = Some (U32 st.(tc).(tidx), st).
Proof. intros. reflexivity. Qed.

Lemma agpu_eval_ptx_bidx :
  forall st, agpu_eval_ptx st PtxBidx = Some (U32 st.(tc).(bidx), st).
Proof. intros. reflexivity. Qed.

Lemma agpu_eval_ptx_bdim :
  forall st, agpu_eval_ptx st PtxBdim = Some (U32 st.(tc).(bdim), st).
Proof. intros. reflexivity. Qed.
