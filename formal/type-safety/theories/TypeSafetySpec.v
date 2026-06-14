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
 * Phase T2-CUSTOM: ETuple extension — adds ETuple to expr, expr_ind_strong for
 * Forall-preserving IHs over list elements, infer_type as mutual fixpoint with
 * infer_list, HT_Tuple to has_type, and auxiliary lemmas for all five theorems.
 *
 * Elided vs. Sarek_types.ml (documented for ASSUMPTIONS.md):
 *   - TVar/unification: replaced by resolved types (see header above).
 *   - TRecord / TVariant: defined here; field access typing in RegistrySpec.v.
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

(* typ, post-unification (Sarek_types.ml:49). TVar elided (replaced by resolved types). *)
Inductive sarek_type : Type :=
  | TPrim    : prim_type -> sarek_type
  | TReg     : reg_type -> sarek_type
  | TVec     : sarek_type -> sarek_type
  | TArr     : sarek_type -> mem_space -> sarek_type
  | TFun     : list sarek_type -> sarek_type -> sarek_type
  | TTuple   : list sarek_type -> sarek_type
  | TRecord  : string -> list (string * sarek_type) -> sarek_type
  | TVariant : string -> list (string * option sarek_type) -> sarek_type.

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
  | ELit   : lit -> expr
  | EVar   : string -> expr
  | ELet   : string -> expr -> expr -> expr   (* let x = e1 in e2 *)
  | ETuple : list expr -> expr.

(* expr_ind_strong: Coq's default induction on expr does not yield IHs for
   list elements inside ETuple. This custom principle gives Forall P es for
   the ETuple case, enabling induction under the list. *)
Lemma expr_ind_strong :
  forall (P : expr -> Prop),
    (forall l, P (ELit l)) ->
    (forall x, P (EVar x)) ->
    (forall x e1 e2, P e1 -> P e2 -> P (ELet x e1 e2)) ->
    (forall es, Forall P es -> P (ETuple es)) ->
    forall e, P e.
Proof.
  intros P Hlit Hvar Hlet Htup.
  fix IH 1.
  intro e. destruct e as [l | x | x e1 e2 | es].
  - apply Hlit.
  - apply Hvar.
  - apply Hlet; apply IH.
  - apply Htup. induction es as [| e' rest IHrest].
    + apply Forall_nil.
    + apply Forall_cons; [apply IH | exact IHrest].
Defined.

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
   EVar reads the environment. ELet threads the inferred binding type.
   ETuple infers each element in order; first error short-circuits. *)

(* infer_list_with: helper taking infer_type as a parameter for termination.
   Used internally by infer_type to traverse ETuple element lists. *)
Fixpoint infer_list_with
    (infer_f : type_env -> expr -> infer_result)
    (env : type_env)
    (es : list expr)
    : (list sarek_type + type_error)%type :=
  match es with
  | [] => inl []
  | e :: rest =>
      match infer_f env e with
      | inl t =>
          match infer_list_with infer_f env rest with
          | inl ts => inl (t :: ts)
          | inr err => inr err
          end
      | inr err => inr err
      end
  end.

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
  | ETuple es =>
      match infer_list_with infer_type env es with
      | inl ts => inl (TTuple ts)
      | inr err => inr err
      end
  end.

(* infer_list: top-level list inference, specialised to infer_type.
   Definitionally equal to [infer_list_with infer_type]. *)
Definition infer_list (env : type_env) (es : list expr)
    : (list sarek_type + type_error)%type :=
  infer_list_with infer_type env es.

(* Unfolding lemma: makes ETuple case of infer_type transparent for proofs. *)
Lemma infer_type_etuple (env : type_env) (es : list expr) :
  infer_type env (ETuple es) =
  match infer_list env es with
  | inl ts => inl (TTuple ts)
  | inr err => inr err
  end.
Proof.
  reflexivity.
Qed.

(* ===== 6. Declarative well-typedness relation (the "spec" side) =====
   has_type env e t mirrors the intended typing judgement. infer_type is the
   algorithmic side; soundness/completeness connect the two.
   HT_Tuple: each element types to its corresponding component type. *)

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
      has_type env (ELet x e1 e2) t2
  | HT_Tuple : forall env es ts,
      Forall2 (has_type env) es ts ->
      has_type env (ETuple es) (TTuple ts).

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

(* infer_list_sound_helper: if infer_list succeeds, each element is well-typed.
   Auxiliary for infer_type_sound's ETuple case. *)
Lemma infer_list_sound_helper :
  forall (es : list expr)
    (IHes : Forall (fun e => forall env t, infer_type env e = inl t -> has_type env e t) es)
    (env : type_env) (ts : list sarek_type),
    infer_list env es = inl ts -> Forall2 (has_type env) es ts.
Proof.
  intros es IHes.
  induction IHes as [| e es' IHe IHes' IH'].
  - intros env ts H. unfold infer_list, infer_list_with in H. injection H as <-. constructor.
  - intros env ts H. unfold infer_list in H. simpl in H.
    destruct (infer_type env e) as [t | err] eqn:Ht.
    + destruct (infer_list_with infer_type env es') as [ts' | err'] eqn:Hlist.
      * injection H as <-. apply Forall2_cons.
        -- apply IHe. exact Ht.
        -- apply IH'. unfold infer_list. exact Hlist.
      * discriminate.
    + discriminate.
Qed.

(* infer_type_sound: if inference succeeds with type t, the expression is
   well-typed at t under the declarative judgement. This is the central
   soundness theorem connecting the algorithm to the spec. *)
(* Helper with e as the first argument for expr_ind_strong application. *)
Lemma infer_type_sound_inner :
  forall (ex : expr) (env : type_env) (ty : sarek_type),
    infer_type env ex = inl ty ->
    has_type env ex ty.
Proof.
  intro ex.
  induction ex using expr_ind_strong.
  - (* ELit *)
    intros env ty H. destruct l; simpl in H; injection H as <-; constructor.
  - (* EVar *)
    intros env ty H. simpl in H.
    destruct (lookup_env env x) eqn:Hlk.
    + injection H as <-. apply HT_Var. exact Hlk.
    + discriminate.
  - (* ELet: IHs from expr_ind_strong are first two H-named hypotheses *)
    intros env ty Hinfer. simpl in Hinfer.
    match goal with
    | |- has_type env (ELet _ ?sub1 _) _ =>
        destruct (infer_type env sub1) as [t1 | err] eqn:He1
    end.
    + apply HT_Let with (t1 := t1).
      * match goal with IH1 : forall _ _, infer_type _ ?sub1 = _ -> _ |- _ =>
          apply IH1; exact He1 end.
      * match goal with IH2 : forall _ _, infer_type _ ?sub2 = _ -> _ |- has_type _ ?sub2 _ =>
          apply IH2; exact Hinfer end.
    + discriminate.
  - (* ETuple: induction gives H/H0 for the Forall IH; rename to avoid clash *)
    rename H into Hforall.
    intros env ty Hinfer. rewrite infer_type_etuple in Hinfer.
    destruct (infer_list env es) as [ts | err] eqn:Hlist.
    + injection Hinfer as <-.
      apply HT_Tuple.
      apply (infer_list_sound_helper Hforall env Hlist).
    + discriminate.
Qed.

Theorem infer_type_sound :
  forall (env : type_env) (e : expr) (t : sarek_type),
    infer_type env e = inl t ->
    has_type env e t.
Proof.
  intros env e t H. exact (infer_type_sound_inner e env H).
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

(* has_type_list_det: the declarative judgement assigns at most one type list
   to a list of expressions. Auxiliary for has_type_det's ETuple case. *)
Lemma has_type_list_det :
  forall (env : type_env) (es : list expr) (ts1 ts2 : list sarek_type),
    (forall e t1 t2, In e es -> has_type env e t1 -> has_type env e t2 -> t1 = t2) ->
    Forall2 (has_type env) es ts1 ->
    Forall2 (has_type env) es ts2 ->
    ts1 = ts2.
Proof.
  intros env es ts1 ts2 IHel H1.
  revert ts2.
  induction H1 as [| e t1' es' ts1' He1 Hf1 IHf1].
  - intros ts2 H2. inversion H2. reflexivity.
  - intros ts2 H2. inversion H2 as [| e2 t2' es'' ts2' He2 Hf2 Heq1 Heq2]; subst.
    f_equal.
    + apply (IHel e t1' t2'). left. reflexivity. exact He1. exact He2.
    + apply IHf1.
      * intros e' s1 s2 Hin. apply (IHel e' s1 s2). right. exact Hin.
      * exact Hf2.
Qed.

(* has_type_det: the declarative judgement assigns at most one type to an
   expression in a given environment. Proved using expr_ind_strong so that
   the ETuple case has IHs for all list elements. *)
Lemma has_type_det_inner :
  forall (ex : expr) (env : type_env) (t1 t2 : sarek_type),
    has_type env ex t1 -> has_type env ex t2 -> t1 = t2.
Proof.
  intro ex. induction ex using expr_ind_strong.
  - (* ELit *)
    intros env t1 t2 H1 H2. inversion H1; inversion H2; subst; try discriminate; reflexivity.
  - (* EVar *)
    intros env t1 t2 H1 H2.
    inversion H1; inversion H2; subst.
    eapply lookup_env_correct; eauto.
  - (* ELet: IHe1 and IHe2 come from expr_ind_strong for the two sub-exprs *)
    intros env t1 t2 H1 H2.
    inversion H1 as [| | | | | env1 x1 f1 g1 ta1 tb1 Hf1 Hg1 |]; subst.
    inversion H2 as [| | | | | env2 x2 f2 g2 ta2 tb2 Hf2 Hg2 |]; subst.
    assert (Heq : ta1 = ta2) by
      (match goal with IH : forall _ _ _, has_type _ ?sub _ -> has_type _ ?sub _ -> _ = _,
                       Hh1 : has_type _ ?sub ta1, Hh2 : has_type _ ?sub ta2 |- _ =>
          apply (IH _ _ _ Hh1 Hh2) end). subst.
    match goal with IH : forall _ _ _, has_type _ ?sub _ -> has_type _ ?sub _ -> _ = _,
                    Hh1 : has_type _ ?sub t1, Hh2 : has_type _ ?sub t2 |- t1 = t2 =>
        apply (IH _ _ _ Hh1 Hh2) end.
  - (* ETuple *)
    rename H into Hforall.
    intros env t1 t2 H1 H2.
    inversion H1 as [| | | | | | env1 es1 ts1 Hf1]; subst.
    inversion H2 as [| | | | | | env2 es2 ts2 Hf2]; subst.
    f_equal.
    eapply has_type_list_det.
    + intros e' t1' t2' Hin Ht1 Ht2.
      rewrite Forall_forall in Hforall.
      eapply Hforall; eassumption.
    + exact Hf1.
    + exact Hf2.
Qed.

Lemma has_type_det :
  forall (env : type_env) (e : expr) (t1 t2 : sarek_type),
    has_type env e t1 -> has_type env e t2 -> t1 = t2.
Proof.
  intros env e t1 t2 H1 H2. exact (has_type_det_inner H1 H2).
Qed.

(* infer_list_complete_helper: if each element is well-typed, infer_list
   succeeds with the corresponding type list. Auxiliary for
   infer_type_complete's ETuple case. *)
Lemma infer_list_complete_helper :
  forall (es : list expr)
    (IHes : Forall (fun e => forall env t, has_type env e t -> infer_type env e = inl t) es),
  forall (env : type_env) (ts : list sarek_type),
    Forall2 (has_type env) es ts -> infer_list env es = inl ts.
Proof.
  intros es IHes.
  induction IHes as [| e es' IHe IHes' IH'].
  - intros env ts H. inversion H; subst. unfold infer_list, infer_list_with. reflexivity.
  - intros env ts H.
    inversion H as [| e' t' es'' ts'' He Hf2 Heq1 Heq2]; subst.
    unfold infer_list. simpl.
    rewrite (IHe env t' He).
    unfold infer_list in IH'.
    rewrite (IH' env ts'' Hf2).
    reflexivity.
Qed.

(* infer_type_complete: the converse of infer_type_sound. Every derivable
   typing is found by the algorithm. Uses expr_ind_strong so ETuple has
   element-wise IHs. *)
Lemma infer_type_complete_inner :
  forall (ex : expr) (env : type_env) (ty : sarek_type),
    has_type env ex ty ->
    infer_type env ex = inl ty.
Proof.
  intro ex. induction ex using expr_ind_strong.
  - (* ELit *)
    intros env ty H. inversion H; subst; reflexivity.
  - (* EVar *)
    intros env ty H. inversion H as [| | | | env0 x0 t0 Hlk | |]; subst.
    simpl. rewrite Hlk. reflexivity.
  - (* ELet *)
    intros env ty H.
    inversion H as [| | | | | env0 x0 f g ta tb Hf Hg |]; subst.
    simpl.
    match goal with IH : forall _ _, has_type _ ?sub _ -> infer_type _ ?sub = _,
                    Hx : has_type _ ?sub ta |- _ =>
        rewrite (IH env ta Hx) end.
    match goal with IH : forall _ _, has_type _ ?sub _ -> infer_type _ ?sub = _,
                    Hx : has_type ((_, ta) :: env) ?sub ty |- _ =>
        rewrite (IH _ ty Hx); reflexivity end.
  - (* ETuple *)
    rename H into Hforall.
    intros env ty H.
    inversion H as [| | | | | | env0 es0 ts Hf2]; subst.
    rewrite infer_type_etuple.
    pose proof (infer_list_complete_helper Hforall Hf2) as Hil.
    rewrite Hil. reflexivity.
Qed.

Lemma infer_type_complete :
  forall (env : type_env) (e : expr) (t : sarek_type),
    has_type env e t ->
    infer_type env e = inl t.
Proof.
  intros env e t H. exact (infer_type_complete_inner H).
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
