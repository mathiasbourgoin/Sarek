(******************************************************************************)
(* Rocq 9 spec for GPU control flow typing -- T3-S1 (ControlFlowSpec).
 *
 * Extends RegistrySpec.v with a cf_expr language that adds:
 *
 *   CFIfThen cond then_e         -- if c then e (no else; then must be unit)
 *   CFIfElse cond then_e else_e  -- if c then e1 else e2 (branches must agree)
 *   CFFor var lo hi body         -- for i = lo to hi do body (body type ignored)
 *   CFWhile cond body            -- while c do body (body type ignored)
 *   CFSeq e1 e2                  -- e1 ; e2 (result is e2's type)
 *
 * Rules mirror Sarek_typer.ml: infer_control_flow (lines 345-386).
 *
 * Two if-constructors (no-else / with-else) avoid option inside the inductive,
 * so Rocq's default induction gives IHs for all sub-expressions directly.
 *
 * Simplifications vs. the real typer:
 *   - For-loop direction (Upto/Downto) does not affect types; it is elided.
 *   - var_id (fresh variable index) is internal to the typer; not modelled.
 *
 * Proven (all Qed, 0 admits):
 *   infer_cf_type_sound    -- infer succeeds -> has_cf_type
 *   infer_cf_type_complete -- has_cf_type -> infer succeeds with same type
 *   has_cf_type_det        -- uniqueness of the declarative judgement
 *   cf_type_preservation   -- bi-directional iff
 ******************************************************************************)

From Stdlib Require Import Bool List String Arith Nat.
From TypeSafety Require Import TypeSafetySpec.
From TypeSafety Require Import VecSpec.
From TypeSafety Require Import RegistrySpec.
Import TypeSafetySpec VecSpec RegistrySpec.
Import ListNotations.

Set Implicit Arguments.

(* ===== 1. Control-flow error kinds ===== *)

Inductive cf_error : Type :=
  | CRec           : rec_error -> cf_error
  | CondNotBool    : sarek_type -> cf_error
  | BranchMismatch : sarek_type -> sarek_type -> cf_error
  | BoundNotInt32  : sarek_type -> cf_error.

(* ===== 2. Control-flow expression language ===== *)

Inductive cf_expr : Type :=
  | CFRec    : rec_expr -> cf_expr
  | CFIfThen : cf_expr -> cf_expr -> cf_expr
  | CFIfElse : cf_expr -> cf_expr -> cf_expr -> cf_expr
  | CFFor    : string -> cf_expr -> cf_expr -> cf_expr -> cf_expr
  | CFWhile  : cf_expr -> cf_expr -> cf_expr
  | CFSeq    : cf_expr -> cf_expr -> cf_expr.

(* ===== 3. Algorithmic type inference ===== *)

Fixpoint infer_cf_type (env : type_env) (e : cf_expr)
    : (sarek_type + cf_error)%type :=
  match e with
  | CFRec re =>
      match infer_rec_type env re with
      | inl t   => inl t
      | inr err => inr (CRec err)
      end
  | CFIfThen cond then_e =>
      match infer_cf_type env cond with
      | inr err    => inr err
      | inl cond_t =>
          match sarek_type_eq_dec cond_t (TPrim TBool) with
          | right _ => inr (CondNotBool cond_t)
          | left  _ =>
              match infer_cf_type env then_e with
              | inr err    => inr err
              | inl then_t =>
                  match sarek_type_eq_dec then_t (TPrim TUnit) with
                  | right _ => inr (BranchMismatch then_t (TPrim TUnit))
                  | left  _ => inl (TPrim TUnit)
                  end
              end
          end
      end
  | CFIfElse cond then_e else_e =>
      match infer_cf_type env cond with
      | inr err    => inr err
      | inl cond_t =>
          match sarek_type_eq_dec cond_t (TPrim TBool) with
          | right _ => inr (CondNotBool cond_t)
          | left  _ =>
              match infer_cf_type env then_e with
              | inr err    => inr err
              | inl then_t =>
                  match infer_cf_type env else_e with
                  | inr err    => inr err
                  | inl else_t =>
                      match sarek_type_eq_dec then_t else_t with
                      | right _ => inr (BranchMismatch then_t else_t)
                      | left  _ => inl then_t
                      end
                  end
              end
          end
      end
  | CFFor var lo hi body =>
      match infer_cf_type env lo with
      | inr err  => inr err
      | inl lo_t =>
          match sarek_type_eq_dec lo_t (TPrim TInt32) with
          | right _ => inr (BoundNotInt32 lo_t)
          | left  _ =>
              match infer_cf_type env hi with
              | inr err  => inr err
              | inl hi_t =>
                  match sarek_type_eq_dec hi_t (TPrim TInt32) with
                  | right _ => inr (BoundNotInt32 hi_t)
                  | left  _ =>
                      match infer_cf_type ((var, TPrim TInt32) :: env) body with
                      | inl _ => inl (TPrim TUnit)
                      | inr err => inr err
                      end
                  end
              end
          end
      end
  | CFWhile cond body =>
      match infer_cf_type env cond with
      | inr err    => inr err
      | inl cond_t =>
          match sarek_type_eq_dec cond_t (TPrim TBool) with
          | right _ => inr (CondNotBool cond_t)
          | left  _ =>
              match infer_cf_type env body with
              | inl _ => inl (TPrim TUnit)
              | inr err => inr err
              end
          end
      end
  | CFSeq e1 e2 =>
      match infer_cf_type env e1 with
      | inr err => inr err
      | inl _   =>
          match infer_cf_type env e2 with
          | inl t2  => inl t2
          | inr err => inr err
          end
      end
  end.

(* ===== 4. Declarative well-typedness judgement ===== *)

Inductive has_cf_type : type_env -> cf_expr -> sarek_type -> Prop :=
  | HCF_Rec : forall env re t,
      has_rec_type env re t ->
      has_cf_type env (CFRec re) t
  | HCF_IfThen : forall env cond then_e,
      has_cf_type env cond (TPrim TBool) ->
      has_cf_type env then_e (TPrim TUnit) ->
      has_cf_type env (CFIfThen cond then_e) (TPrim TUnit)
  | HCF_IfElse : forall env cond then_e else_e t,
      has_cf_type env cond (TPrim TBool) ->
      has_cf_type env then_e t ->
      has_cf_type env else_e t ->
      has_cf_type env (CFIfElse cond then_e else_e) t
  | HCF_For : forall env var lo hi body t_body,
      has_cf_type env lo (TPrim TInt32) ->
      has_cf_type env hi (TPrim TInt32) ->
      has_cf_type ((var, TPrim TInt32) :: env) body t_body ->
      has_cf_type env (CFFor var lo hi body) (TPrim TUnit)
  | HCF_While : forall env cond body t_body,
      has_cf_type env cond (TPrim TBool) ->
      has_cf_type env body t_body ->
      has_cf_type env (CFWhile cond body) (TPrim TUnit)
  | HCF_Seq : forall env e1 e2 t1 t2,
      has_cf_type env e1 t1 ->
      has_cf_type env e2 t2 ->
      has_cf_type env (CFSeq e1 e2) t2.

(* ===== 5. Soundness: infer succeeds -> has_cf_type ===== *)

Theorem infer_cf_type_sound :
  forall env e t,
    infer_cf_type env e = inl t ->
    has_cf_type env e t.
Proof.
  intros env e t H. revert env t H.
  induction e as
    [ re
    | cond IHcond then_e IHthen
    | cond IHcond then_e IHthen else_e IHelse
    | var lo IHlo hi IHhi body IHbody
    | cond IHcond body IHbody
    | e1 IH1 e2 IH2 ];
  intros env t H; simpl in H.
  - (* CFRec *)
    destruct (infer_rec_type env re) as [rt | err] eqn:Hre; [| discriminate].
    injection H as <-. apply HCF_Rec. apply infer_rec_type_sound. exact Hre.
  - (* CFIfThen *)
    destruct (infer_cf_type env cond) as [ct | err] eqn:Hcond; [| discriminate].
    destruct (sarek_type_eq_dec ct (TPrim TBool)) as [Hb | _]; [| discriminate].
    subst ct.
    destruct (infer_cf_type env then_e) as [tt | err] eqn:Hthen; [| discriminate].
    destruct (sarek_type_eq_dec tt (TPrim TUnit)) as [Hu | _]; [| discriminate].
    subst tt. injection H as <-.
    apply HCF_IfThen.
    + apply IHcond. exact Hcond.
    + apply IHthen. exact Hthen.
  - (* CFIfElse *)
    destruct (infer_cf_type env cond) as [ct | err] eqn:Hcond; [| discriminate].
    destruct (sarek_type_eq_dec ct (TPrim TBool)) as [Hb | _]; [| discriminate].
    subst ct.
    destruct (infer_cf_type env then_e) as [tt | err] eqn:Hthen; [| discriminate].
    destruct (infer_cf_type env else_e) as [et | err] eqn:Helse; [| discriminate].
    destruct (sarek_type_eq_dec tt et) as [Heq | _]; [| discriminate].
    subst et. injection H as <-.
    apply HCF_IfElse.
    + apply IHcond. exact Hcond.
    + apply IHthen. exact Hthen.
    + apply IHelse. exact Helse.
  - (* CFFor *)
    destruct (infer_cf_type env lo) as [lt | err] eqn:Hlo; [| discriminate].
    destruct (sarek_type_eq_dec lt (TPrim TInt32)) as [Hli | _]; [| discriminate].
    subst lt.
    destruct (infer_cf_type env hi) as [ht | err] eqn:Hhi; [| discriminate].
    destruct (sarek_type_eq_dec ht (TPrim TInt32)) as [Hhi' | _]; [| discriminate].
    subst ht.
    destruct (infer_cf_type ((var, TPrim TInt32) :: env) body) as [bt | err] eqn:Hbody;
      [| discriminate].
    injection H as <-.
    apply HCF_For with (t_body := bt).
    + apply IHlo. exact Hlo.
    + apply IHhi. exact Hhi.
    + apply IHbody. exact Hbody.
  - (* CFWhile *)
    destruct (infer_cf_type env cond) as [ct | err] eqn:Hcond; [| discriminate].
    destruct (sarek_type_eq_dec ct (TPrim TBool)) as [Hb | _]; [| discriminate].
    subst ct.
    destruct (infer_cf_type env body) as [bt | err] eqn:Hbody; [| discriminate].
    injection H as <-.
    apply HCF_While with (t_body := bt).
    + apply IHcond. exact Hcond.
    + apply IHbody. exact Hbody.
  - (* CFSeq *)
    destruct (infer_cf_type env e1) as [t1 | err] eqn:H1; [| discriminate].
    destruct (infer_cf_type env e2) as [t2 | err] eqn:H2; [| discriminate].
    injection H as <-.
    apply HCF_Seq with (t1 := t1).
    + apply IH1. exact H1.
    + apply IH2. exact H2.
Qed.

(* ===== 6. Completeness: has_cf_type -> infer succeeds ===== *)

Theorem infer_cf_type_complete :
  forall env e t,
    has_cf_type env e t ->
    infer_cf_type env e = inl t.
Proof.
  intros env e t H.
  induction H as
    [ env re t Hrec
    | env cond then_e Hcond IHcond Hthen IHthen
    | env cond then_e else_e t Hcond IHcond Hthen IHthen Helse IHelse
    | env var lo hi body t_body Hlo IHlo Hhi IHhi Hbody IHbody
    | env cond body t_body Hcond IHcond Hbody IHbody
    | env e1 e2 t1 t2 H1 IH1 H2 IH2 ]; simpl.
  - (* HCF_Rec *)
    rewrite (infer_rec_type_complete Hrec). reflexivity.
  - (* HCF_IfThen *)
    rewrite IHcond.
    destruct (sarek_type_eq_dec (TPrim TBool) (TPrim TBool)) as [_ | Hne].
    + rewrite IHthen.
      destruct (sarek_type_eq_dec (TPrim TUnit) (TPrim TUnit)) as [_ | Hne2].
      * reflexivity.
      * exfalso. apply Hne2. reflexivity.
    + exfalso. apply Hne. reflexivity.
  - (* HCF_IfElse *)
    rewrite IHcond.
    destruct (sarek_type_eq_dec (TPrim TBool) (TPrim TBool)) as [_ | Hne].
    + rewrite IHthen. rewrite IHelse.
      destruct (sarek_type_eq_dec t t) as [_ | Hne2].
      * reflexivity.
      * exfalso. apply Hne2. reflexivity.
    + exfalso. apply Hne. reflexivity.
  - (* HCF_For *)
    rewrite IHlo.
    destruct (sarek_type_eq_dec (TPrim TInt32) (TPrim TInt32)) as [_ | Hne].
    + rewrite IHhi.
      destruct (sarek_type_eq_dec (TPrim TInt32) (TPrim TInt32)) as [_ | Hne2].
      * rewrite IHbody. reflexivity.
      * exfalso. apply Hne2. reflexivity.
    + exfalso. apply Hne. reflexivity.
  - (* HCF_While *)
    rewrite IHcond.
    destruct (sarek_type_eq_dec (TPrim TBool) (TPrim TBool)) as [_ | Hne].
    + rewrite IHbody. reflexivity.
    + exfalso. apply Hne. reflexivity.
  - (* HCF_Seq *)
    rewrite IH1. rewrite IH2. reflexivity.
Qed.

(* ===== 7. Determinism: piggybacking on completeness ===== *)

Lemma has_cf_type_det :
  forall env e t1 t2,
    has_cf_type env e t1 ->
    has_cf_type env e t2 ->
    t1 = t2.
Proof.
  intros env e t1 t2 H1 H2.
  pose proof (infer_cf_type_complete H1) as Hc1.
  pose proof (infer_cf_type_complete H2) as Hc2.
  rewrite Hc1 in Hc2. injection Hc2 as <-. reflexivity.
Qed.

(* ===== 8. Type preservation: algorithmic <-> declarative ===== *)

Theorem cf_type_preservation :
  forall env e t,
    infer_cf_type env e = inl t <-> has_cf_type env e t.
Proof.
  intros env e t. split.
  - apply infer_cf_type_sound.
  - apply infer_cf_type_complete.
Qed.
