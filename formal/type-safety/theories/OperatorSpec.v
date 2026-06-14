(******************************************************************************)
(* Rocq 9 spec for GPU operator typing -- T3-S2 (OperatorSpec).
 *
 * Extends ControlFlowSpec.v with an op_expr language that adds:
 *
 *   OPBinop op lhs rhs  -- binary operation (Add/Sub/Mul/Div/Mod/Eq/Ne/...)
 *   OPUnop  op operand  -- unary operation (Neg/Not/Lnot)
 *
 * Rules mirror Sarek_typer.ml: infer_binop (lines 146-176), infer_unop
 * (lines 179-189). Post-unification model: both operands must already be the
 * same resolved type; TVar is not modelled.
 *
 * Type-class predicates (post-unification, no TVar):
 *   is_numeric : TPrim TInt32, TReg RInt/RInt64/RFloat32/RFloat64
 *   is_integer : TPrim TInt32, TReg RInt/RInt64
 *
 * Binop groups (mirroring the 5 match arms of infer_binop):
 *   NumericBinop   (Add/Sub/Mul/Div) : t1=t2, numeric(t)  -> t
 *   IntegerBinop   (Mod/Land/Lor/Lxor/Lsl/Lsr/Asr) : t1=t2, integer(t) -> t
 *   EqBinop        (Eq/Ne)           : t1=t2              -> TPrim TBool
 *   CmpBinop       (Lt/Le/Gt/Ge)     : t1=t2, numeric(t)  -> TPrim TBool
 *   BoolBinop      (And/Or)          : t1=t2=TBool         -> TPrim TBool
 *
 * Proven (all Qed, 0 admits):
 *   infer_op_type_sound    -- infer succeeds -> has_op_type
 *   infer_op_type_complete -- has_op_type -> infer succeeds
 *   has_op_type_det        -- uniqueness of the declarative judgement
 *   op_type_preservation   -- bi-directional iff
 ******************************************************************************)

From Stdlib Require Import Bool List String Arith Nat.
From TypeSafety Require Import TypeSafetySpec.
From TypeSafety Require Import VecSpec.
From TypeSafety Require Import RegistrySpec.
From TypeSafety Require Import ControlFlowSpec.
Import TypeSafetySpec VecSpec RegistrySpec ControlFlowSpec.
Import ListNotations.

Set Implicit Arguments.

(* ===== 1. Operator AST (mirrors Sarek_ast.ml binop / unop) ===== *)

Inductive binop : Type :=
  | Add | Sub | Mul | Div | Mod
  | And | Or
  | Eq | Ne | Lt | Le | Gt | Ge
  | Land | Lor | Lxor | Lsl | Lsr | Asr.

Inductive unop : Type :=
  | Neg | Not | Lnot.

(* ===== 2. Type-class predicates ===== *)

Definition is_numeric (t : sarek_type) : bool :=
  match t with
  | TPrim TInt32 => true
  | TReg RInt | TReg RInt64 | TReg RFloat32 | TReg RFloat64 => true
  | _ => false
  end.

Definition is_integer (t : sarek_type) : bool :=
  match t with
  | TPrim TInt32 => true
  | TReg RInt | TReg RInt64 => true
  | _ => false
  end.

(* ===== 3. Operator error kinds ===== *)

Inductive op_error : Type :=
  | OCF          : cf_error -> op_error
  | NotNumeric   : sarek_type -> op_error
  | NotInteger   : sarek_type -> op_error
  | NotBool      : sarek_type -> op_error
  | OperandMismatch : sarek_type -> sarek_type -> op_error.

(* ===== 4. Operator expression language ===== *)

Inductive op_expr : Type :=
  | OPCf    : cf_expr -> op_expr
  | OPBinop : binop -> op_expr -> op_expr -> op_expr
  | OPUnop  : unop -> op_expr -> op_expr.

(* ===== 5. Algorithmic type inference ===== *)

Fixpoint infer_op_type (env : type_env) (e : op_expr)
    : (sarek_type + op_error)%type :=
  match e with
  | OPCf ce =>
      match infer_cf_type env ce with
      | inl t   => inl t
      | inr err => inr (OCF err)
      end
  | OPBinop op lhs rhs =>
      match infer_op_type env lhs with
      | inr err => inr err
      | inl t1  =>
          match infer_op_type env rhs with
          | inr err => inr err
          | inl t2  =>
              match sarek_type_eq_dec t1 t2 with
              | right _ => inr (OperandMismatch t1 t2)
              | left  _ =>
                  match op with
                  | Add | Sub | Mul | Div =>
                      if is_numeric t1 then inl t1 else inr (NotNumeric t1)
                  | Mod | Land | Lor | Lxor | Lsl | Lsr | Asr =>
                      if is_integer t1 then inl t1 else inr (NotInteger t1)
                  | Eq | Ne => inl (TPrim TBool)
                  | Lt | Le | Gt | Ge =>
                      if is_numeric t1 then inl (TPrim TBool) else inr (NotNumeric t1)
                  | And | Or =>
                      match sarek_type_eq_dec t1 (TPrim TBool) with
                      | right _ => inr (NotBool t1)
                      | left  _ => inl (TPrim TBool)
                      end
                  end
              end
          end
      end
  | OPUnop op operand =>
      match infer_op_type env operand with
      | inr err => inr err
      | inl t   =>
          match op with
          | Neg  => if is_numeric t then inl t else inr (NotNumeric t)
          | Not  =>
              match sarek_type_eq_dec t (TPrim TBool) with
              | right _ => inr (NotBool t)
              | left  _ => inl (TPrim TBool)
              end
          | Lnot => if is_integer t then inl t else inr (NotInteger t)
          end
      end
  end.

(* ===== 6. Declarative well-typedness judgement =====
   Five grouped binop constructors mirror the five match arms of infer_binop.
   This keeps the inductive concise and the soundness proof uniform. *)

(* Shorthand predicates for op-group membership *)
Definition is_numeric_binop (op : binop) : bool :=
  match op with Add | Sub | Mul | Div => true | _ => false end.

Definition is_integer_binop (op : binop) : bool :=
  match op with Mod | Land | Lor | Lxor | Lsl | Lsr | Asr => true | _ => false end.

Definition is_eq_binop (op : binop) : bool :=
  match op with Eq | Ne => true | _ => false end.

Definition is_cmp_binop (op : binop) : bool :=
  match op with Lt | Le | Gt | Ge => true | _ => false end.

Definition is_bool_binop (op : binop) : bool :=
  match op with And | Or => true | _ => false end.

Inductive has_op_type : type_env -> op_expr -> sarek_type -> Prop :=
  | HOT_CF : forall env ce t,
      has_cf_type env ce t ->
      has_op_type env (OPCf ce) t
  | HOT_NumericBinop : forall env op lhs rhs t,
      is_numeric_binop op = true ->
      has_op_type env lhs t ->
      has_op_type env rhs t ->
      is_numeric t = true ->
      has_op_type env (OPBinop op lhs rhs) t
  | HOT_IntegerBinop : forall env op lhs rhs t,
      is_integer_binop op = true ->
      has_op_type env lhs t ->
      has_op_type env rhs t ->
      is_integer t = true ->
      has_op_type env (OPBinop op lhs rhs) t
  | HOT_EqBinop : forall env op lhs rhs t,
      is_eq_binop op = true ->
      has_op_type env lhs t ->
      has_op_type env rhs t ->
      has_op_type env (OPBinop op lhs rhs) (TPrim TBool)
  | HOT_CmpBinop : forall env op lhs rhs t,
      is_cmp_binop op = true ->
      has_op_type env lhs t ->
      has_op_type env rhs t ->
      is_numeric t = true ->
      has_op_type env (OPBinop op lhs rhs) (TPrim TBool)
  | HOT_BoolBinop : forall env op lhs rhs,
      is_bool_binop op = true ->
      has_op_type env lhs (TPrim TBool) ->
      has_op_type env rhs (TPrim TBool) ->
      has_op_type env (OPBinop op lhs rhs) (TPrim TBool)
  | HOT_Neg : forall env operand t,
      has_op_type env operand t ->
      is_numeric t = true ->
      has_op_type env (OPUnop Neg operand) t
  | HOT_Not : forall env operand,
      has_op_type env operand (TPrim TBool) ->
      has_op_type env (OPUnop Not operand) (TPrim TBool)
  | HOT_Lnot : forall env operand t,
      has_op_type env operand t ->
      is_integer t = true ->
      has_op_type env (OPUnop Lnot operand) t.

(* ===== 7. Soundness: infer succeeds -> has_op_type ===== *)

Theorem infer_op_type_sound :
  forall env e t,
    infer_op_type env e = inl t ->
    has_op_type env e t.
Proof.
  intros env e. revert env.
  induction e as [ce | op lhs IHl rhs IHr | op operand IHop];
  intros env t H; simpl in H.
  - (* OPCf *)
    destruct (infer_cf_type env ce) as [ct | err] eqn:Hce; [| discriminate].
    injection H as <-. apply HOT_CF. apply infer_cf_type_sound. exact Hce.
  - (* OPBinop *)
    destruct (infer_op_type env lhs) as [t1 | err] eqn:Hl; [| discriminate].
    destruct (infer_op_type env rhs) as [t2 | err] eqn:Hr; [| discriminate].
    destruct (sarek_type_eq_dec t1 t2) as [Heq | _]; [| discriminate]. subst t2.
    (* Goal order after destruct op:
       1=Add 2=Sub 3=Mul 4=Div 5=Mod 6=And 7=Or 8=Eq 9=Ne
       10=Lt 11=Le 12=Gt 13=Ge 14=Land 15=Lor 16=Lxor 17=Lsl 18=Lsr 19=Asr *)
    destruct op; simpl in H.
    (* 1-4: Add Sub Mul Div *)
    1-4: (destruct (is_numeric t1) eqn:Hk; [| discriminate]; injection H as <-;
          apply HOT_NumericBinop; [reflexivity | apply IHl; exact Hl |
                                   apply IHr; exact Hr | exact Hk]).
    (* 1=Mod (was 5) *)
    1: (destruct (is_integer t1) eqn:Hk; [| discriminate]; injection H as <-;
        apply HOT_IntegerBinop; [reflexivity | apply IHl; exact Hl |
                                 apply IHr; exact Hr | exact Hk]).
    (* 1-2=And Or (were 6-7) *)
    1-2: (destruct (sarek_type_eq_dec t1 (TPrim TBool)) as [Hb | Hne];
          [injection H as <-; rewrite Hb in Hl; rewrite Hb in Hr;
           apply HOT_BoolBinop; [reflexivity | apply IHl; exact Hl | apply IHr; exact Hr]
          | discriminate]).
    (* 1-2=Eq Ne (were 8-9) *)
    1-2: (injection H as <-;
          apply HOT_EqBinop with (t := t1);
          [reflexivity | apply IHl; exact Hl | apply IHr; exact Hr]).
    (* 1-4=Lt Le Gt Ge (were 10-13) *)
    1-4: (destruct (is_numeric t1) eqn:Hk; [| discriminate]; injection H as <-;
          apply HOT_CmpBinop with (t := t1);
          [reflexivity | apply IHl; exact Hl | apply IHr; exact Hr | exact Hk]).
    (* 1-6=Land Lor Lxor Lsl Lsr Asr (were 14-19) *)
    1-6: (destruct (is_integer t1) eqn:Hk; [| discriminate]; injection H as <-;
          apply HOT_IntegerBinop; [reflexivity | apply IHl; exact Hl |
                                   apply IHr; exact Hr | exact Hk]).
  - (* OPUnop *)
    destruct (infer_op_type env operand) as [t' | err] eqn:Hop; [| discriminate].
    destruct op; simpl in H.
    + destruct (is_numeric t') eqn:Hk; [| discriminate]. injection H as <-.
      apply HOT_Neg. apply IHop. exact Hop. exact Hk.
    + destruct (sarek_type_eq_dec t' (TPrim TBool)) as [Hb | Hne];
      [injection H as <-; rewrite Hb in Hop; apply HOT_Not; apply IHop; exact Hop
      | discriminate].
    + destruct (is_integer t') eqn:Hk; [| discriminate]. injection H as <-.
      apply HOT_Lnot. apply IHop. exact Hop. exact Hk.
Qed.

(* ===== 8. Completeness: has_op_type -> infer succeeds ===== *)

Theorem infer_op_type_complete :
  forall env e t,
    has_op_type env e t ->
    infer_op_type env e = inl t.
Proof.
  intros env e t H. induction H; simpl.
  - (* HOT_CF *)
    rewrite (infer_cf_type_complete H). reflexivity.
  - (* HOT_NumericBinop *)
    rewrite IHhas_op_type1. rewrite IHhas_op_type2.
    destruct (sarek_type_eq_dec t t) as [_ | Hne]; [| exfalso; apply Hne; reflexivity].
    destruct op; simpl in H; try discriminate.
    all: (destruct (is_numeric t) eqn:Hnum; [reflexivity | congruence]).
  - (* HOT_IntegerBinop *)
    rewrite IHhas_op_type1. rewrite IHhas_op_type2.
    destruct (sarek_type_eq_dec t t) as [_ | Hne]; [| exfalso; apply Hne; reflexivity].
    destruct op; simpl in H; try discriminate.
    all: (destruct (is_integer t) eqn:Hnum; [reflexivity | congruence]).
  - (* HOT_EqBinop *)
    rewrite IHhas_op_type1. rewrite IHhas_op_type2.
    destruct (sarek_type_eq_dec t t) as [_ | Hne]; [| exfalso; apply Hne; reflexivity].
    destruct op; simpl in H; try discriminate; reflexivity.
  - (* HOT_CmpBinop *)
    rewrite IHhas_op_type1. rewrite IHhas_op_type2.
    destruct (sarek_type_eq_dec t t) as [_ | Hne]; [| exfalso; apply Hne; reflexivity].
    destruct op; simpl in H; try discriminate.
    all: (destruct (is_numeric t) eqn:Hnum; [reflexivity | congruence]).
  - (* HOT_BoolBinop *)
    rewrite IHhas_op_type1. rewrite IHhas_op_type2.
    destruct (sarek_type_eq_dec (TPrim TBool) (TPrim TBool)) as [_ | Hne];
      [| exfalso; apply Hne; reflexivity].
    destruct op; simpl in H; try discriminate.
    all: (destruct (sarek_type_eq_dec (TPrim TBool) (TPrim TBool)) as [_ | Hne2];
          [reflexivity | exfalso; apply Hne2; reflexivity]).
  - (* HOT_Neg *)
    rewrite IHhas_op_type.
    destruct (is_numeric t) eqn:Hnum; [reflexivity | congruence].
  - (* HOT_Not *)
    rewrite IHhas_op_type.
    destruct (sarek_type_eq_dec (TPrim TBool) (TPrim TBool)) as [_ | Hne];
      [reflexivity | exfalso; apply Hne; reflexivity].
  - (* HOT_Lnot *)
    rewrite IHhas_op_type.
    destruct (is_integer t) eqn:Hnum; [reflexivity | congruence].
Qed.

(* ===== 9. Determinism: piggybacking on completeness ===== *)

Lemma has_op_type_det :
  forall env e t1 t2,
    has_op_type env e t1 ->
    has_op_type env e t2 ->
    t1 = t2.
Proof.
  intros env e t1 t2 H1 H2.
  pose proof (infer_op_type_complete H1) as Hc1.
  pose proof (infer_op_type_complete H2) as Hc2.
  rewrite Hc1 in Hc2. injection Hc2 as <-. reflexivity.
Qed.

(* ===== 10. Type preservation: algorithmic <-> declarative ===== *)

Theorem op_type_preservation :
  forall env e t,
    infer_op_type env e = inl t <-> has_op_type env e t.
Proof.
  intros env e t. split.
  - apply infer_op_type_sound.
  - apply infer_op_type_complete.
Qed.
