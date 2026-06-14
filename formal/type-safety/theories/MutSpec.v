(******************************************************************************)
(* Rocq 9 spec for mutable bindings -- T3-S4 (MutSpec).
 *
 * Extends FunSpec.v with a mut_expr language that adds:
 *
 *   MEFun    fe                -- delegation to the function layer
 *   MELetMut name init body    -- mutable let binding (let mut name = init in body)
 *   MEAssign name value        -- assignment to a mutable variable (name <- value)
 *
 * Rules mirror Sarek_typer.ml infer_let_binding (~line 563):
 *   - ELetMut: infer the init value type [t]; extend the environment with
 *     (name, t) marked *mutable*; infer the body in the extended environment;
 *     the whole expression has the body's type.
 *   - EAssign: look up [name]; it must be bound AND mutable; the value's type
 *     must match the variable's declared type; the result type is unit.
 *
 * Mutability is tracked by a second environment [mut_env] (the list of names
 * currently bound as mutable).  The ordinary [type_env] carries the types and
 * is shared with the delegated function layer so reads of a mutable variable
 * succeed through MEFun.  MELetMut adds (name,t) to *both* environments;
 * MEFun delegation reads only the type_env, exactly as the function layer
 * expects.
 *
 * Error kinds:
 *   MEFunErr        -- delegated function-layer error
 *   MEUnbound       -- assignment target is not bound at all
 *   MEImmutable     -- assignment target is bound but not mutable
 *   MEAssignMismatch -- assigned value type != variable's declared type
 *
 * Proven (all Qed, 0 admits):
 *   infer_mut_type_sound    -- infer succeeds -> has_mut_type
 *   infer_mut_type_complete -- has_mut_type -> infer succeeds
 *   has_mut_type_det        -- uniqueness of the declarative judgement
 *   mut_type_preservation   -- bi-directional iff
 ******************************************************************************)

From Stdlib Require Import Bool List String Arith Nat.
From TypeSafety Require Import TypeSafetySpec.
From TypeSafety Require Import VecSpec.
From TypeSafety Require Import RegistrySpec.
From TypeSafety Require Import ControlFlowSpec.
From TypeSafety Require Import OperatorSpec.
From TypeSafety Require Import FunSpec.
Import TypeSafetySpec VecSpec RegistrySpec ControlFlowSpec OperatorSpec FunSpec.
Import ListNotations.

Set Implicit Arguments.

(* ===== 1. Mutability environment =====
   The set of names currently bound as mutable, modelled as a list. *)

Definition mut_env := list string.

Fixpoint is_mutable (mu : mut_env) (x : string) : bool :=
  match mu with
  | [] => false
  | y :: rest => if String.eqb x y then true else is_mutable rest x
  end.

(* ===== 2. Mutable-binding error kinds ===== *)

Inductive mut_error : Type :=
  | MEFunErr        : fun_error -> mut_error
  | MEUnbound       : string -> mut_error
  | MEImmutable     : string -> mut_error
  | MEAssignMismatch : sarek_type -> sarek_type -> mut_error. (* declared, got *)

(* ===== 3. Mutable expression language ===== *)

Inductive mut_expr : Type :=
  | MEFun    : fun_expr -> mut_expr
  | MELetMut : string -> mut_expr -> mut_expr -> mut_expr
  | MEAssign : string -> mut_expr -> mut_expr.

(* ===== 4. Algorithmic type inference =====
   Threads both the ordinary type environment and the mutability environment. *)

Fixpoint infer_mut_type (env : type_env) (mu : mut_env) (e : mut_expr)
    : (sarek_type + mut_error)%type :=
  match e with
  | MEFun fe =>
      match infer_fun_type env fe with
      | inl t   => inl t
      | inr err => inr (MEFunErr err)
      end
  | MELetMut name init body =>
      match infer_mut_type env mu init with
      | inr err => inr err
      | inl t   =>
          infer_mut_type ((name, t) :: env) (name :: mu) body
      end
  | MEAssign name value =>
      match infer_mut_type env mu value with
      | inr err => inr err
      | inl tv  =>
          match lookup_env env name with
          | None => inr (MEUnbound name)
          | Some tdecl =>
              if is_mutable mu name then
                match sarek_type_eq_dec tv tdecl with
                | left  _ => inl (TPrim TUnit)
                | right _ => inr (MEAssignMismatch tdecl tv)
                end
              else inr (MEImmutable name)
          end
      end
  end.

(* ===== 5. Declarative well-typedness judgement ===== *)

Inductive has_mut_type : type_env -> mut_env -> mut_expr -> sarek_type -> Prop :=
  | HMT_Fun : forall env mu fe t,
      has_fun_type env fe t ->
      has_mut_type env mu (MEFun fe) t
  | HMT_LetMut : forall env mu name init body t tbody,
      has_mut_type env mu init t ->
      has_mut_type ((name, t) :: env) (name :: mu) body tbody ->
      has_mut_type env mu (MELetMut name init body) tbody
  | HMT_Assign : forall env mu name value tv tdecl,
      has_mut_type env mu value tv ->
      lookup_env env name = Some tdecl ->
      is_mutable mu name = true ->
      tv = tdecl ->
      has_mut_type env mu (MEAssign name value) (TPrim TUnit).

(* ===== 6. Soundness: infer succeeds -> has_mut_type ===== *)

Theorem infer_mut_type_sound :
  forall e env mu t,
    infer_mut_type env mu e = inl t ->
    has_mut_type env mu e t.
Proof.
  induction e as [fe | name init IHinit body IHbody | name value IHvalue];
  intros env mu t H; simpl in H.
  - (* MEFun *)
    destruct (infer_fun_type env fe) as [tf | err] eqn:Hfe; [| discriminate].
    injection H as <-. apply HMT_Fun. apply infer_fun_type_sound. exact Hfe.
  - (* MELetMut *)
    destruct (infer_mut_type env mu init) as [tinit | err] eqn:Hinit;
      [| discriminate].
    apply HMT_LetMut with (t := tinit).
    + apply IHinit. exact Hinit.
    + apply IHbody. exact H.
  - (* MEAssign *)
    destruct (infer_mut_type env mu value) as [tv | err] eqn:Hval;
      [| discriminate].
    destruct (lookup_env env name) as [tdecl | ] eqn:Hlk; [| discriminate].
    destruct (is_mutable mu name) eqn:Hmut; [| discriminate].
    destruct (sarek_type_eq_dec tv tdecl) as [Heq | _]; [| discriminate].
    injection H as <-.
    apply HMT_Assign with (tv := tv) (tdecl := tdecl).
    + apply IHvalue. exact Hval.
    + exact Hlk.
    + exact Hmut.
    + exact Heq.
Qed.

(* ===== 7. Completeness: has_mut_type -> infer succeeds ===== *)

Theorem infer_mut_type_complete :
  forall env mu e t,
    has_mut_type env mu e t ->
    infer_mut_type env mu e = inl t.
Proof.
  intros env mu e t H. induction H; simpl.
  - (* HMT_Fun *)
    rewrite (infer_fun_type_complete H). reflexivity.
  - (* HMT_LetMut *)
    rewrite IHhas_mut_type1. exact IHhas_mut_type2.
  - (* HMT_Assign *)
    rewrite IHhas_mut_type. rewrite H0. rewrite H1. subst tv.
    destruct (sarek_type_eq_dec tdecl tdecl) as [_ | Hne];
      [reflexivity | exfalso; apply Hne; reflexivity].
Qed.

(* ===== 8. Determinism: piggybacking on completeness ===== *)

Lemma has_mut_type_det :
  forall env mu e t1 t2,
    has_mut_type env mu e t1 ->
    has_mut_type env mu e t2 ->
    t1 = t2.
Proof.
  intros env mu e t1 t2 H1 H2.
  pose proof (infer_mut_type_complete H1) as Hc1.
  pose proof (infer_mut_type_complete H2) as Hc2.
  rewrite Hc1 in Hc2. injection Hc2 as <-. reflexivity.
Qed.

(* ===== 9. Type preservation: algorithmic <-> declarative ===== *)

Theorem mut_type_preservation :
  forall env mu e t,
    infer_mut_type env mu e = inl t <-> has_mut_type env mu e t.
Proof.
  intros env mu e t. split.
  - apply infer_mut_type_sound.
  - apply infer_mut_type_complete.
Qed.
