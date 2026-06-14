(******************************************************************************)
(* Rocq 9 spec for the Sarek PPX type checker — TYPE SAFETY.
 *
 * Targets:
 *   ~/dev/SPOC/sarek/ppx/Sarek_typer.ml      (1154 lines) — inference engine
 *   ~/dev/SPOC/sarek/ppx/Sarek_types.ml      ( 343 lines) — type repr + unify
 *   ~/dev/SPOC/sarek/ppx/Sarek_typed_ast.ml  ( 425 lines) — typed AST (texpr)
 *
 * Goal: prove that [infer] correctly infers and preserves types for Sarek
 * kernel expressions. The Sarek type system is GPU-aware Hindley-Milner with
 * mutable unification variables (TVar of tvar ref). Coq has no mutable
 * unification, so this spec models POST-UNIFICATION types: every type here is
 * already resolved (no TVar). This mirrors Sarek_typed_ast.texpr, whose [ty]
 * field is "Always resolved, never contains unbound TVar".
 *
 * Phase T1-SPEC: type universe + environment + a simplified [infer_type] that
 * models the checker's post-unification behaviour on literals, variables and
 * let-bindings. Soundness lemmas are fully proved (Qed).
 *
 * Elided vs. Sarek_types.ml (documented for ASSUMPTIONS.md):
 *   - TVar/unification: replaced by resolved types (see header above).
 *   - TRecord / TVariant: deferred to T2 (custom-type inference).
 *   - registered_type Custom: deferred (user-registered [@@sarek.type]).
 ******************************************************************************)

From Stdlib Require Import Bool List String Arith.
Import ListNotations.

Set Implicit Arguments.

(* ===== 1. Type universe (mirrors Sarek_types.ml) ===== *)

(* prim_type = TUnit | TBool | TInt32  (core language, Sarek_types.ml:29) *)
Inductive prim_type : Type := TUnit | TBool | TInt32.

(* registered_type, minus Custom (library-defined numerics; Sarek_types.ml:34) *)
Inductive reg_type : Type := RInt | RInt64 | RFloat32 | RFloat64 | RChar.

(* memspace = Local | Shared | Global  (Sarek_types.ml:43) *)
Inductive mem_space : Type := Local | Shared | Global.

(* typ, post-unification (Sarek_types.ml:49). TVar/TRecord/TVariant elided. *)
Inductive sarek_type : Type :=
  | TPrim  : prim_type -> sarek_type
  | TReg   : reg_type -> sarek_type
  | TVec   : sarek_type -> sarek_type
  | TArr   : sarek_type -> mem_space -> sarek_type
  | TFun   : list sarek_type -> sarek_type -> sarek_type
  | TTuple : list sarek_type -> sarek_type.

(* Decidable equality on types is needed for the soundness statements. *)
Scheme Equality for prim_type.
Scheme Equality for reg_type.
Scheme Equality for mem_space.

(* ===== 2. Source expressions (abstract; subset of Sarek_ast) ===== *)

Inductive lit : Type :=
  | LInt   : nat -> lit
  | LFloat : nat -> lit        (* float payload abstracted to nat *)
  | LBool  : bool -> lit
  | LUnit  : lit.

Inductive expr : Type :=
  | ELit : lit -> expr
  | EVar : string -> expr
  | ELet : string -> expr -> expr -> expr.   (* let x = e1 in e2 *)

(* ===== 3. Type environment (Sarek_env.ml, modelled as assoc list) ===== *)

Definition type_env := list (string * sarek_type).

Fixpoint lookup_env (env : type_env) (x : string) : option sarek_type :=
  match env with
  | [] => None
  | (y, t) :: rest =>
      if String.eqb x y then Some t else lookup_env rest x
  end.

(* ===== 4. Type errors + inference result ===== *)

Inductive type_error : Type :=
  | UnboundVar   : string -> type_error
  | TypeMismatch : sarek_type -> sarek_type -> type_error.

Definition infer_result := (sarek_type + type_error)%type.

(* ===== 5. Simplified infer_type (models the post-unification checker) =====
   Literal rules match Sarek_typer.ml:
     EInt   -> TPrim TInt32
     EFloat -> TReg  RFloat32
     EBool  -> TPrim TBool
     EUnit  -> TPrim TUnit
   EVar reads the environment. ELet threads the inferred binding type. *)

Fixpoint infer_type (env : type_env) (e : expr) : infer_result :=
  match e with
  | ELit (LInt _)   => inl (TPrim TInt32)
  | ELit (LFloat _) => inl (TReg RFloat32)
  | ELit (LBool _)  => inl (TPrim TBool)
  | ELit LUnit      => inl (TPrim TUnit)
  | EVar x =>
      match lookup_env env x with
      | Some t => inl t
      | None   => inr (UnboundVar x)
      end
  | ELet x e1 e2 =>
      match infer_type env e1 with
      | inl t1 => infer_type ((x, t1) :: env) e2
      | inr err => inr err
      end
  end.

(* ===== 6. Declarative well-typedness relation (the "spec" side) =====
   has_type env e t mirrors the intended typing judgement. infer_type is the
   algorithmic side; soundness/completeness connect the two. *)

Inductive has_type : type_env -> expr -> sarek_type -> Prop :=
  | HT_Int  : forall env n,  has_type env (ELit (LInt n))   (TPrim TInt32)
  | HT_Float: forall env n,  has_type env (ELit (LFloat n)) (TReg RFloat32)
  | HT_Bool : forall env b,  has_type env (ELit (LBool b))  (TPrim TBool)
  | HT_Unit : forall env,    has_type env (ELit LUnit)      (TPrim TUnit)
  | HT_Var  : forall env x t,
      lookup_env env x = Some t ->
      has_type env (EVar x) t
  | HT_Let  : forall env x e1 e2 t1 t2,
      has_type env e1 t1 ->
      has_type ((x, t1) :: env) e2 t2 ->
      has_type env (ELet x e1 e2) t2.

(* ===== 7. Soundness properties (all proved — Qed) ===== *)

(* infer_lit_int: integer literals always infer to TPrim TInt32. *)
Theorem infer_lit_int :
  forall (env : type_env) (n : nat),
    infer_type env (ELit (LInt n)) = inl (TPrim TInt32).
Proof.
  reflexivity.
Qed.

(* infer_lit_bool: boolean literals always infer to TPrim TBool. *)
Theorem infer_lit_bool :
  forall (env : type_env) (b : bool),
    infer_type env (ELit (LBool b)) = inl (TPrim TBool).
Proof.
  reflexivity.
Qed.

(* infer_var_bound: a variable present in the environment infers to its type.
   NB: requires the first binding for x in env to be (x, t) — stated here for
   the head binding; the general lookup form is proved via lookup_env. *)
Theorem infer_var_bound :
  forall (env : type_env) (x : string) (t : sarek_type),
    lookup_env env x = Some t ->
    infer_type env (EVar x) = inl t.
Proof.
  intros env x t H. simpl. rewrite H. reflexivity.
Qed.

(* lookup_env_sound: a successful lookup means the binding really is in env. *)
Theorem lookup_env_sound :
  forall (env : type_env) (x : string) (t : sarek_type),
    lookup_env env x = Some t ->
    In (x, t) env.
Proof.
  intros env x t. revert t.
  induction env as [| [y t'] rest IH]; intro t; intro H.
  - simpl in H. discriminate.
  - simpl in H. destruct (String.eqb x y) eqn:Heq.
    + apply String.eqb_eq in Heq. subst y. injection H as ->. left. reflexivity.
    + right. apply IH. exact H.
Qed.

(* infer_type_sound: if inference succeeds with type t, the expression is
   well-typed at t under the declarative judgement. This is the central
   soundness theorem connecting the algorithm to the spec. *)
Theorem infer_type_sound :
  forall (env : type_env) (e : expr) (t : sarek_type),
    infer_type env e = inl t ->
    has_type env e t.
Proof.
  intros env e. revert env.
  induction e as [l | x | x e1 IH1 e2 IH2]; intros env t H.
  - (* ELit *)
    destruct l; simpl in H; injection H as <-; constructor.
  - (* EVar *)
    simpl in H. destruct (lookup_env env x) eqn:Hlk.
    + injection H as <-. apply HT_Var. exact Hlk.
    + discriminate.
  - (* ELet *)
    simpl in H. destruct (infer_type env e1) as [t1 | err] eqn:He1.
    + apply HT_Let with (t1 := t1).
      * apply IH1. exact He1.
      * apply IH2. exact H.
    + discriminate.
Qed.

(* ===== 8. T1-SOUND: determinism, completeness, preservation ===== *)

(* lookup_env_correct: lookup_env is a (partial) function — at most one result.
   Trivial since lookup_env is a total function, but stated for use in
   has_type inversion (HT_Var) reasoning. *)
Lemma lookup_env_correct :
  forall (env : type_env) (x : string) (t : sarek_type),
    lookup_env env x = Some t ->
    forall t', lookup_env env x = Some t' -> t = t'.
Proof.
  intros env x t H t' H'. rewrite H in H'. injection H' as ->. reflexivity.
Qed.

(* has_type_det: the declarative judgement assigns at most one type to an
   expression in a given environment. Proved by induction on the first
   derivation, inverting the second. EVar uses lookup_env's functionality;
   ELet chains the two IHs (the e1 type must match before e2 can be compared). *)
Lemma has_type_det :
  forall (env : type_env) (e : expr) (t1 t2 : sarek_type),
    has_type env e t1 -> has_type env e t2 -> t1 = t2.
Proof.
  intros env e t1 t2 H1. revert t2.
  induction H1 as
    [ env n | env n | env b | env
    | env x t Hlk
    | env x e1 e2 ta tb He1 IH1 He2 IH2 ];
    intros tB H2.
  - inversion H2; subst. reflexivity.
  - inversion H2; subst. reflexivity.
  - inversion H2; subst. reflexivity.
  - inversion H2; subst. reflexivity.
  - (* HT_Var: both lookups succeed; lookup_env is a function *)
    inversion H2; subst. eapply lookup_env_correct; eauto.
  - (* HT_Let: e1 type agrees by IH1, so e2 environments match, then IH2 *)
    inversion H2 as [ | | | | | env' x' e1' e2' tc td He1' He2' Hx Henv ]; subst.
    assert (Heq : ta = tc) by (apply IH1; assumption). subst tc.
    apply IH2. assumption.
Qed.

(* infer_type_complete: the converse of infer_type_sound. Every derivable
   typing is found by the algorithm. Induction on the has_type derivation.
   HT_Var is immediate: the premise IS lookup_env x = Some t, which is exactly
   what the EVar branch of infer_type matches on. HT_Let rewrites with both
   IHs to thread the binding type. *)
Lemma infer_type_complete :
  forall (env : type_env) (e : expr) (t : sarek_type),
    has_type env e t ->
    infer_type env e = inl t.
Proof.
  intros env e t H.
  induction H.
  - reflexivity.            (* HT_Int *)
  - reflexivity.            (* HT_Float *)
  - reflexivity.            (* HT_Bool *)
  - reflexivity.            (* HT_Unit *)
  - simpl. rewrite H. reflexivity.   (* HT_Var *)
  - simpl. rewrite IHhas_type1. rewrite IHhas_type2. reflexivity. (* HT_Let *)
Qed.

(* type_preservation: the algorithmic checker and the declarative judgement
   coincide exactly. This is the central T1-SOUND theorem — infer_type is both
   sound (=> direction) and complete (<= direction) w.r.t. has_type. *)
Theorem type_preservation :
  forall (env : type_env) (e : expr) (t : sarek_type),
    infer_type env e = inl t <-> has_type env e t.
Proof.
  intros env e t. split.
  - apply infer_type_sound.
  - apply infer_type_complete.
Qed.
