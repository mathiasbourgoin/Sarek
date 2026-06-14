(******************************************************************************)
(* Rocq 9 spec for GPU vector/array memory access typing -- T2-VEC.
 *
 * Extends TypeSafetySpec.v with a mem_expr language that adds four memory
 * access operations on top of the core [expr] fragment:
 *
 *   EVecGet  : vec[idx]          (indexed read  from a TVec)
 *   EVecSet  : vec[idx] <- val   (indexed write to  a TVec)
 *   EArrGet  : arr[idx]          (indexed read  from a TArr)
 *   EArrSet  : arr[idx] <- val   (indexed write to  a TArr)
 *
 * Architecture mirrors Sarek_typer.ml's [infer_memory_access] function, which
 * handles EVecGet/EVecSet/EArrGet/EArrSet as a separate pass after the core
 * Hindley-Milner inference has resolved all type variables.
 *
 * Simplification vs. the real typer: [EArrGet] in Sarek_typer.ml first tries
 * TVec unification, then TArr (for polymorphic arrays).  Post-unification,
 * all types are resolved, so VecSpec.v's [EArrGet] matches strictly on [TArr].
 *
 * Proven (all Qed, 0 admits):
 *   sarek_type_eq_dec        -- decidable equality on sarek_type
 *   infer_mem_type_sound     -- infer -> has_mem_type
 *   infer_mem_type_complete  -- has_mem_type -> infer
 *   has_mem_type_det         -- uniqueness of the declarative judgement
 *   mem_type_preservation    -- bi-directional biconditional (iff)
 ******************************************************************************)

From Stdlib Require Import Bool List String Arith Nat.
From TypeSafety Require Import TypeSafetySpec.
Import TypeSafetySpec.
Import ListNotations.

Set Implicit Arguments.

(* ===== 1. Decidable equality on sarek_type ===== *)

(* sarek_type_eq_dec: needed for EVecSet/EArrSet where we must check that the
   value type matches the element type of the container. *)
Lemma sarek_type_eq_dec : forall t1 t2 : sarek_type, {t1 = t2} + {t1 <> t2}.
Proof.
  fix IH 1.
  intros t1 t2.
  destruct t1 as [p | r | sv | sa ms | lf sf | lt];
  destruct t2 as [p0 | r0 | sv0 | sa0 ms0 | lf0 sf0 | lt0];
    try (right; discriminate).
  - (* TPrim/TPrim *)
    destruct (prim_type_eq_dec p p0).
    + left. subst. reflexivity.
    + right. intro H. apply n. injection H as <-. reflexivity.
  - (* TReg/TReg *)
    destruct (reg_type_eq_dec r r0).
    + left. subst. reflexivity.
    + right. intro H. apply n. injection H as <-. reflexivity.
  - (* TVec/TVec *)
    destruct (IH sv sv0).
    + left. subst. reflexivity.
    + right. intro H. apply n. injection H as <-. reflexivity.
  - (* TArr/TArr *)
    destruct (IH sa sa0).
    + destruct (mem_space_eq_dec ms ms0).
      * left. subst. reflexivity.
      * right. intro H. apply n. injection H as _ <-. reflexivity.
    + right. intro H. apply n. injection H as <- _. reflexivity.
  - (* TFun/TFun: need list eq *)
    assert (forall (ts1 ts2 : list sarek_type), {ts1 = ts2} + {ts1 <> ts2}) as IHL. {
      fix IHL 1.
      intros ts1 ts2.
      destruct ts1 as [| h1 rest1]; destruct ts2 as [| h2 rest2];
        try (right; discriminate); try (left; reflexivity).
      destruct (IH h1 h2).
      - destruct (IHL rest1 rest2).
        + left. subst. reflexivity.
        + right. intro H. apply n. injection H as _ ->. reflexivity.
      - right. intro H. apply n. injection H as <- _. reflexivity.
    }
    destruct (IHL lf lf0).
    + destruct (IH sf sf0).
      * left. subst. reflexivity.
      * right. intro H. apply n. injection H as _ <-. reflexivity.
    + right. intro H. apply n. injection H as <- _. reflexivity.
  - (* TTuple/TTuple *)
    assert (forall (ts1 ts2 : list sarek_type), {ts1 = ts2} + {ts1 <> ts2}) as IHL. {
      fix IHL 1.
      intros ts1 ts2.
      destruct ts1 as [| h1 rest1]; destruct ts2 as [| h2 rest2];
        try (right; discriminate); try (left; reflexivity).
      destruct (IH h1 h2).
      - destruct (IHL rest1 rest2).
        + left. subst. reflexivity.
        + right. intro H. apply n. injection H as _ ->. reflexivity.
      - right. intro H. apply n. injection H as <- _. reflexivity.
    }
    destruct (IHL lt lt0).
    + left. subst. reflexivity.
    + right. intro H. apply n. injection H as <-. reflexivity.
Qed.

(* ===== 2. Memory access error kinds ===== *)

Inductive vec_error : Type :=
  | VCoreError  : type_error -> vec_error   (* error from core inference *)
  | NotAVector  : sarek_type -> vec_error   (* expected TVec, got this *)
  | NotAnArray  : sarek_type -> vec_error   (* expected TArr, got this *)
  | IndexNotInt : sarek_type -> vec_error   (* index must be TPrim TInt32 *)
  | ElemMismatch : sarek_type -> sarek_type -> vec_error. (* value/elem type clash *)

Definition vec_result := (sarek_type + vec_error)%type.

(* ===== 3. Memory access expressions ===== *)

Inductive mem_expr : Type :=
  | MCore   : expr -> mem_expr
  | EVecGet : mem_expr -> mem_expr -> mem_expr
  | EVecSet : mem_expr -> mem_expr -> mem_expr -> mem_expr
  | EArrGet : mem_expr -> mem_expr -> mem_expr
  | EArrSet : mem_expr -> mem_expr -> mem_expr -> mem_expr.

(* ===== 4. Type inference for mem_expr ===== *)

Fixpoint infer_mem_type (env : type_env) (e : mem_expr) : vec_result :=
  match e with
  | MCore ce =>
      match infer_type env ce with
      | inl t   => inl t
      | inr err => inr (VCoreError err)
      end
  | EVecGet vec idx =>
      match infer_mem_type env vec with
      | inr err           => inr err
      | inl (TVec elem_t) =>
          match infer_mem_type env idx with
          | inr err            => inr err
          | inl (TPrim TInt32) => inl elem_t
          | inl ti             => inr (IndexNotInt ti)
          end
      | inl tv => inr (NotAVector tv)
      end
  | EVecSet vec idx value =>
      match infer_mem_type env vec with
      | inr err           => inr err
      | inl (TVec elem_t) =>
          match infer_mem_type env idx with
          | inr err            => inr err
          | inl (TPrim TInt32) =>
              match infer_mem_type env value with
              | inr err => inr err
              | inl vt  =>
                  match sarek_type_eq_dec vt elem_t with
                  | left _  => inl (TPrim TUnit)
                  | right _ => inr (ElemMismatch vt elem_t)
                  end
              end
          | inl ti => inr (IndexNotInt ti)
          end
      | inl tv => inr (NotAVector tv)
      end
  | EArrGet arr idx =>
      match infer_mem_type env arr with
      | inr err              => inr err
      | inl (TArr elem_t _m) =>
          match infer_mem_type env idx with
          | inr err            => inr err
          | inl (TPrim TInt32) => inl elem_t
          | inl ti             => inr (IndexNotInt ti)
          end
      | inl ta => inr (NotAnArray ta)
      end
  | EArrSet arr idx value =>
      match infer_mem_type env arr with
      | inr err              => inr err
      | inl (TArr elem_t _m) =>
          match infer_mem_type env idx with
          | inr err            => inr err
          | inl (TPrim TInt32) =>
              match infer_mem_type env value with
              | inr err => inr err
              | inl vt  =>
                  match sarek_type_eq_dec vt elem_t with
                  | left _  => inl (TPrim TUnit)
                  | right _ => inr (ElemMismatch vt elem_t)
                  end
              end
          | inl ti => inr (IndexNotInt ti)
          end
      | inl ta => inr (NotAnArray ta)
      end
  end.

(* ===== 5. Declarative typing for mem_expr ===== *)

Inductive has_mem_type : type_env -> mem_expr -> sarek_type -> Prop :=
  | HMT_Core   : forall env e t,
      has_type env e t ->
      has_mem_type env (MCore e) t
  | HMT_VecGet : forall env vec idx elem_t,
      has_mem_type env vec (TVec elem_t) ->
      has_mem_type env idx (TPrim TInt32) ->
      has_mem_type env (EVecGet vec idx) elem_t
  | HMT_VecSet : forall env vec idx value elem_t,
      has_mem_type env vec (TVec elem_t) ->
      has_mem_type env idx (TPrim TInt32) ->
      has_mem_type env value elem_t ->
      has_mem_type env (EVecSet vec idx value) (TPrim TUnit)
  | HMT_ArrGet : forall env arr idx elem_t m,
      has_mem_type env arr (TArr elem_t m) ->
      has_mem_type env idx (TPrim TInt32) ->
      has_mem_type env (EArrGet arr idx) elem_t
  | HMT_ArrSet : forall env arr idx value elem_t m,
      has_mem_type env arr (TArr elem_t m) ->
      has_mem_type env idx (TPrim TInt32) ->
      has_mem_type env value elem_t ->
      has_mem_type env (EArrSet arr idx value) (TPrim TUnit).

(* ===== 6. Soundness ===== *)

(* infer_mem_type_sound: if inference succeeds with type t, the expression is
   well-typed at t under the declarative judgement.
   Strategy: induction on e; for MCore use infer_type_sound; for EVecGet
   destruct on vec result (must be inl (TVec elem_t)), then idx result (must be
   inl (TPrim TInt32)), apply IH, apply HMT_VecGet; EVecSet additionally
   destructs on value result and sarek_type_eq_dec; EArrGet/EArrSet analogous
   matching TArr. *)
Theorem infer_mem_type_sound :
  forall env e t,
    infer_mem_type env e = inl t ->
    has_mem_type env e t.
Proof.
  intros env e t H. revert env t H.
  induction e; intros env t H; simpl in H.
  - (* MCore *)
    destruct (infer_type env e) as [t' | err] eqn:Hce.
    + injection H as <-. apply HMT_Core. apply infer_type_sound. exact Hce.
    + discriminate.
  - (* EVecGet *)
    destruct (infer_mem_type env e1) as [tv | err] eqn:Hvec; [| discriminate].
    destruct tv as [| | elem_t | | |]; try discriminate.
    (* tv = TVec elem_t *)
    destruct (infer_mem_type env e2) as [ti | err] eqn:Hidx; [| discriminate].
    destruct ti as [p | | | | |]; try discriminate.
    destruct p; try discriminate.
    (* ti = TPrim TInt32 *)
    injection H as <-.
    apply HMT_VecGet.
    + apply IHe1. exact Hvec.
    + apply IHe2. exact Hidx.
  - (* EVecSet *)
    destruct (infer_mem_type env e1) as [tv | err] eqn:Hvec; [| discriminate].
    destruct tv as [| | elem_t | | |]; try discriminate.
    (* tv = TVec elem_t *)
    destruct (infer_mem_type env e2) as [ti | err] eqn:Hidx; [| discriminate].
    destruct ti as [p | | | | |]; try discriminate.
    destruct p; try discriminate.
    (* ti = TPrim TInt32 *)
    destruct (infer_mem_type env e3) as [vt | err] eqn:Hval; [| discriminate].
    destruct (sarek_type_eq_dec vt elem_t) as [Heq | Hne]; [| discriminate].
    injection H as <-. subst vt.
    eapply HMT_VecSet.
    + apply IHe1. exact Hvec.
    + apply IHe2. exact Hidx.
    + apply IHe3. exact Hval.
  - (* EArrGet *)
    destruct (infer_mem_type env e1) as [ta | err] eqn:Harr; [| discriminate].
    destruct ta as [| | | elem_t ms | |]; try discriminate.
    (* ta = TArr elem_t ms *)
    destruct (infer_mem_type env e2) as [ti | err] eqn:Hidx; [| discriminate].
    destruct ti as [p | | | | |]; try discriminate.
    destruct p; try discriminate.
    (* ti = TPrim TInt32 *)
    injection H as <-.
    eapply HMT_ArrGet with (m := ms).
    + apply IHe1. exact Harr.
    + apply IHe2. exact Hidx.
  - (* EArrSet *)
    destruct (infer_mem_type env e1) as [ta | err] eqn:Harr; [| discriminate].
    destruct ta as [| | | elem_t ms | |]; try discriminate.
    (* ta = TArr elem_t ms *)
    destruct (infer_mem_type env e2) as [ti | err] eqn:Hidx; [| discriminate].
    destruct ti as [p | | | | |]; try discriminate.
    destruct p; try discriminate.
    (* ti = TPrim TInt32 *)
    destruct (infer_mem_type env e3) as [vt | err] eqn:Hval; [| discriminate].
    destruct (sarek_type_eq_dec vt elem_t) as [Heq | Hne]; [| discriminate].
    injection H as <-. subst vt.
    eapply HMT_ArrSet with (m := ms).
    + apply IHe1. exact Harr.
    + apply IHe2. exact Hidx.
    + apply IHe3. exact Hval.
Qed.

(* ===== 7. Completeness ===== *)

(* infer_mem_type_complete: the converse of infer_mem_type_sound.
   Strategy: induction on has_mem_type; for HMT_Core use infer_type_complete;
   for HMT_VecGet rewrite IH_vec (-> inl (TVec elem_t)), IH_idx (-> inl (TPrim
   TInt32)); for HMT_VecSet additionally rewrite IH_value then destruct
   (sarek_type_eq_dec elem_t elem_t) -- left branch, done; EArrGet/EArrSet
   analogous. *)
Theorem infer_mem_type_complete :
  forall env e t,
    has_mem_type env e t ->
    infer_mem_type env e = inl t.
Proof.
  intros env0 e0 t0 H.
  induction H; simpl.
  - (* HMT_Core *)
    match goal with Hht : has_type _ _ _ |- _ =>
      rewrite (infer_type_complete Hht); reflexivity end.
  - (* HMT_VecGet *)
    rewrite IHhas_mem_type1. rewrite IHhas_mem_type2. reflexivity.
  - (* HMT_VecSet *)
    rewrite IHhas_mem_type1. rewrite IHhas_mem_type2. rewrite IHhas_mem_type3.
    destruct (sarek_type_eq_dec elem_t elem_t) as [_ | Hne].
    + reflexivity.
    + exfalso. apply Hne. reflexivity.
  - (* HMT_ArrGet *)
    rewrite IHhas_mem_type1. rewrite IHhas_mem_type2. reflexivity.
  - (* HMT_ArrSet *)
    rewrite IHhas_mem_type1. rewrite IHhas_mem_type2. rewrite IHhas_mem_type3.
    destruct (sarek_type_eq_dec elem_t elem_t) as [_ | Hne].
    + reflexivity.
    + exfalso. apply Hne. reflexivity.
Qed.

(* ===== 8. Determinism ===== *)

(* has_mem_type_det: the declarative judgement assigns at most one type to a
   mem_expr.  Proof uses completeness: infer_mem_type is a function, so two
   successful results for the same (env, e) must agree. *)
Lemma has_mem_type_det :
  forall env e t1 t2,
    has_mem_type env e t1 ->
    has_mem_type env e t2 ->
    t1 = t2.
Proof.
  intros env e t1 t2 H1 H2.
  pose proof (infer_mem_type_complete H1) as Hc1.
  pose proof (infer_mem_type_complete H2) as Hc2.
  rewrite Hc1 in Hc2. injection Hc2 as <-. reflexivity.
Qed.

(* ===== 9. Type preservation ===== *)

(* mem_type_preservation: the algorithmic checker and the declarative judgement
   coincide exactly for mem_expr. *)
Theorem mem_type_preservation :
  forall env e t,
    infer_mem_type env e = inl t <-> has_mem_type env e t.
Proof.
  intros env e t. split.
  - apply infer_mem_type_sound.
  - apply infer_mem_type_complete.
Qed.
