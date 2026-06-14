(******************************************************************************)
(* Rocq 9 spec for single-parameter function typing -- T3-S3 (FunSpec).
 *
 * Extends OperatorSpec.v with a fun_expr language that adds:
 *
 *   FEApp    fn arg               -- function application (single argument)
 *   FELetRec fn p body cont       -- recursive single-param function binding
 *
 * Rules mirror Sarek_typer.ml:
 *   - EApp  (lines 686-742): the regular-application branch builds an
 *     expected function type TFun [arg_ty] ret_ty and unifies; here, in the
 *     post-unification model, we require the inferred fn type to be exactly
 *     TFun [p_ty] ret_ty and the arg type to equal p_ty.
 *   - ELetRec (infer_let_binding): the function name is bound to its own
 *     declared TFun [p_ty] ret_ty in the env so the body may recurse; the
 *     body is inferred in env extended with the parameter p_name:p_ty and
 *     must equal ret_ty; the continuation is inferred in the fn-only env.
 *
 * Single-param model: TFun always carries a one-element param list [p_ty].
 *
 * Error kinds:
 *   FEOpErr      -- delegated operator-layer error
 *   NotAFunc     -- applied a non-function type
 *   ArgMismatch  -- argument type != parameter type
 *   BodyMismatch -- let-rec body type != declared return type
 *
 * Proven (all Qed, 0 admits):
 *   infer_fun_type_sound    -- infer succeeds -> has_fun_type
 *   infer_fun_type_complete -- has_fun_type -> infer succeeds
 *   has_fun_type_det        -- uniqueness of the declarative judgement
 *   fun_type_preservation   -- bi-directional iff
 ******************************************************************************)

From Stdlib Require Import Bool List String Arith Nat.
From TypeSafety Require Import TypeSafetySpec.
From TypeSafety Require Import VecSpec.
From TypeSafety Require Import RegistrySpec.
From TypeSafety Require Import ControlFlowSpec.
From TypeSafety Require Import OperatorSpec.
Import TypeSafetySpec VecSpec RegistrySpec ControlFlowSpec OperatorSpec.
Import ListNotations.

Set Implicit Arguments.

(* ===== 1. Function error kinds ===== *)

Inductive fun_error : Type :=
  | FEOpErr     : op_error -> fun_error
  | NotAFunc    : sarek_type -> fun_error
  | ArgMismatch : sarek_type -> sarek_type -> fun_error   (* param, got *)
  | BodyMismatch : sarek_type -> sarek_type -> fun_error.  (* declared, got *)

(* ===== 2. Function expression language ===== *)

Inductive fun_expr : Type :=
  | FEOp     : op_expr -> fun_expr
  | FEApp    : fun_expr -> fun_expr -> fun_expr
  (* FELetRec fn_name p_name p_ty ret_ty body cont *)
  | FELetRec : string -> string -> sarek_type -> sarek_type ->
               fun_expr -> fun_expr -> fun_expr.

(* ===== 3. Algorithmic type inference ===== *)

Fixpoint infer_fun_type (env : type_env) (e : fun_expr)
    : (sarek_type + fun_error)%type :=
  match e with
  | FEOp oe =>
      match infer_op_type env oe with
      | inl t   => inl t
      | inr err => inr (FEOpErr err)
      end
  | FEApp fn arg =>
      match infer_fun_type env fn with
      | inr err => inr err
      | inl tfn =>
          match tfn with
          | TFun (p_ty :: nil) ret_ty =>
              match infer_fun_type env arg with
              | inr err => inr err
              | inl targ =>
                  match sarek_type_eq_dec targ p_ty with
                  | left  _ => inl ret_ty
                  | right _ => inr (ArgMismatch p_ty targ)
                  end
              end
          | _ => inr (NotAFunc tfn)
          end
      end
  | FELetRec fn_name p_name p_ty ret_ty body cont =>
      let fn_ty := TFun (p_ty :: nil) ret_ty in
      let body_env := (p_name, p_ty) :: (fn_name, fn_ty) :: env in
      match infer_fun_type body_env body with
      | inr err => inr err
      | inl tbody =>
          match sarek_type_eq_dec tbody ret_ty with
          | right _ => inr (BodyMismatch ret_ty tbody)
          | left  _ => infer_fun_type ((fn_name, fn_ty) :: env) cont
          end
      end
  end.

(* ===== 4. Declarative well-typedness judgement ===== *)

Inductive has_fun_type : type_env -> fun_expr -> sarek_type -> Prop :=
  | HFT_Op : forall env oe t,
      has_op_type env oe t ->
      has_fun_type env (FEOp oe) t
  | HFT_App : forall env fn arg p_ty ret_ty,
      has_fun_type env fn (TFun (p_ty :: nil) ret_ty) ->
      has_fun_type env arg p_ty ->
      has_fun_type env (FEApp fn arg) ret_ty
  | HFT_LetRec : forall env fn_name p_name p_ty ret_ty body cont t,
      has_fun_type ((p_name, p_ty) :: (fn_name, TFun (p_ty :: nil) ret_ty) :: env)
                   body ret_ty ->
      has_fun_type ((fn_name, TFun (p_ty :: nil) ret_ty) :: env) cont t ->
      has_fun_type env (FELetRec fn_name p_name p_ty ret_ty body cont) t.

(* ===== 5. Soundness: infer succeeds -> has_fun_type ===== *)

Theorem infer_fun_type_sound :
  forall env e t,
    infer_fun_type env e = inl t ->
    has_fun_type env e t.
Proof.
  intros env e. revert env.
  induction e as [oe | fn IHfn arg IHarg
                  | fn_name p_name p_ty ret_ty body IHbody cont IHcont];
  intros env t H; simpl in H.
  - (* FEOp *)
    destruct (infer_op_type env oe) as [ot | err] eqn:Hoe; [| discriminate].
    injection H as <-. apply HFT_Op. apply infer_op_type_sound. exact Hoe.
  - (* FEApp *)
    destruct (infer_fun_type env fn) as [tfn | err] eqn:Hfn; [| discriminate].
    destruct tfn as [ | | | | params ret_ty | | | ]; try discriminate.
    destruct params as [ | p_ty rest]; try discriminate.
    destruct rest as [ | ]; try discriminate.
    destruct (infer_fun_type env arg) as [targ | err] eqn:Harg; [| discriminate].
    destruct (sarek_type_eq_dec targ p_ty) as [Heq | _]; [| discriminate].
    subst targ. injection H as <-.
    apply HFT_App with (p_ty := p_ty).
    + apply IHfn. exact Hfn.
    + apply IHarg. exact Harg.
  - (* FELetRec *)
    destruct (infer_fun_type
                ((p_name, p_ty) :: (fn_name, TFun (p_ty :: nil) ret_ty) :: env) body)
      as [tbody | err] eqn:Hbody; [| discriminate].
    destruct (sarek_type_eq_dec tbody ret_ty) as [Heq | _]; [| discriminate].
    subst tbody.
    apply HFT_LetRec.
    + apply IHbody. exact Hbody.
    + apply IHcont. exact H.
Qed.

(* ===== 6. Completeness: has_fun_type -> infer succeeds ===== *)

Theorem infer_fun_type_complete :
  forall env e t,
    has_fun_type env e t ->
    infer_fun_type env e = inl t.
Proof.
  intros env e t H. induction H; simpl.
  - (* HFT_Op *)
    rewrite (infer_op_type_complete H). reflexivity.
  - (* HFT_App *)
    rewrite IHhas_fun_type1. rewrite IHhas_fun_type2.
    destruct (sarek_type_eq_dec p_ty p_ty) as [_ | Hne];
      [reflexivity | exfalso; apply Hne; reflexivity].
  - (* HFT_LetRec *)
    rewrite IHhas_fun_type1.
    destruct (sarek_type_eq_dec ret_ty ret_ty) as [_ | Hne];
      [| exfalso; apply Hne; reflexivity].
    exact IHhas_fun_type2.
Qed.

(* ===== 7. Determinism: piggybacking on completeness ===== *)

Lemma has_fun_type_det :
  forall env e t1 t2,
    has_fun_type env e t1 ->
    has_fun_type env e t2 ->
    t1 = t2.
Proof.
  intros env e t1 t2 H1 H2.
  pose proof (infer_fun_type_complete H1) as Hc1.
  pose proof (infer_fun_type_complete H2) as Hc2.
  rewrite Hc1 in Hc2. injection Hc2 as <-. reflexivity.
Qed.

(* ===== 8. Type preservation: algorithmic <-> declarative ===== *)

Theorem fun_type_preservation :
  forall env e t,
    infer_fun_type env e = inl t <-> has_fun_type env e t.
Proof.
  intros env e t. split.
  - apply infer_fun_type_sound.
  - apply infer_fun_type_complete.
Qed.
