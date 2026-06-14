(******************************************************************************)
(* Rocq 9 spec for GPU record/variant field access typing -- T2-REGISTRY.
 *
 * Extends VecSpec.v with a rec_expr language that adds record field access:
 *
 *   EFieldGet field rec      -- rec.field (read)
 *   EFieldSet field rec val  -- rec.field <- val (write)
 *
 * Architecture mirrors Sarek_typer.ml's infer_memory_access cases for
 * EFieldGet/EFieldSet. Simplifications vs. the real typer:
 *   - Only known record types (TRecord name fields with fields <> []) are
 *     handled; external/deferred record types are out of scope.
 *   - TVariant is in sarek_type but variant construction/matching are
 *     not in scope for this phase (deferred to T3-SEMANTIC).
 *
 * Proven (all Qed, 0 admits):
 *   field_lookup_sound    -- field_lookup f fields = Some t -> In (f,t) fields
 *   infer_rec_type_sound  -- infer -> has_rec_type
 *   infer_rec_type_complete -- has_rec_type -> infer
 *   has_rec_type_det      -- uniqueness of the declarative judgement
 *   rec_type_preservation -- bi-directional biconditional (iff)
 ******************************************************************************)

From Stdlib Require Import Bool List String Arith Nat.
From TypeSafety Require Import TypeSafetySpec.
From TypeSafety Require Import VecSpec.
Import TypeSafetySpec VecSpec.
Import ListNotations.

Set Implicit Arguments.

(* ===== 1. Field lookup ===== *)

(* field_lookup: find the type of a field in a record's field list.
   Returns the type of the first matching field (shadowing is first-wins). *)
Fixpoint field_lookup (field : string) (fields : list (string * sarek_type))
    : option sarek_type :=
  match fields with
  | [] => None
  | (f, t) :: rest =>
      if String.eqb f field then Some t else field_lookup field rest
  end.

(* field_lookup_sound: a successful lookup means the binding is in the list. *)
Lemma field_lookup_sound :
  forall (field : string) (fields : list (string * sarek_type)) (t : sarek_type),
    field_lookup field fields = Some t ->
    In (field, t) fields.
Proof.
  intros field. induction fields as [| [f ft] rest IH].
  - discriminate.
  - intros t H. simpl in H. destruct (String.eqb_spec f field) as [Hfeq | Hfne].
    + injection H as <-. left. subst. reflexivity.
    + right. apply IH. exact H.
Qed.

(* ===== 2. Record access error kinds ===== *)

Inductive rec_error : Type :=
  | RMemError     : vec_error -> rec_error
  | NotARecord    : sarek_type -> rec_error
  | FieldNotFound : string -> sarek_type -> rec_error
  | FieldMismatch : sarek_type -> sarek_type -> rec_error.

Definition rec_result := (sarek_type + rec_error)%type.

(* ===== 3. Record access expressions ===== *)

Inductive rec_expr : Type :=
  | RMem      : mem_expr -> rec_expr
  | EFieldGet : string -> rec_expr -> rec_expr
  | EFieldSet : string -> rec_expr -> rec_expr -> rec_expr.

(* ===== 4. Type inference for rec_expr ===== *)

Fixpoint infer_rec_type (env : type_env) (e : rec_expr) : rec_result :=
  match e with
  | RMem me =>
      match infer_mem_type env me with
      | inl t   => inl t
      | inr err => inr (RMemError err)
      end
  | EFieldGet field rec =>
      match infer_rec_type env rec with
      | inr err => inr err
      | inl (TRecord name fields) =>
          match field_lookup field fields with
          | Some t => inl t
          | None   => inr (FieldNotFound field (TRecord name fields))
          end
      | inl t => inr (NotARecord t)
      end
  | EFieldSet field rec value =>
      match infer_rec_type env rec with
      | inr err => inr err
      | inl (TRecord name fields) =>
          match field_lookup field fields with
          | None         => inr (FieldNotFound field (TRecord name fields))
          | Some field_t =>
              match infer_rec_type env value with
              | inr err => inr err
              | inl vt  =>
                  match sarek_type_eq_dec vt field_t with
                  | left _  => inl (TPrim TUnit)
                  | right _ => inr (FieldMismatch vt field_t)
                  end
              end
          end
      | inl t => inr (NotARecord t)
      end
  end.

(* ===== 5. Declarative typing for rec_expr ===== *)

Inductive has_rec_type : type_env -> rec_expr -> sarek_type -> Prop :=
  | HRT_Mem      : forall env me t,
      has_mem_type env me t ->
      has_rec_type env (RMem me) t
  | HRT_FieldGet : forall env field rec name fields field_t,
      has_rec_type env rec (TRecord name fields) ->
      field_lookup field fields = Some field_t ->
      has_rec_type env (EFieldGet field rec) field_t
  | HRT_FieldSet : forall env field rec value name fields field_t,
      has_rec_type env rec (TRecord name fields) ->
      field_lookup field fields = Some field_t ->
      has_rec_type env value field_t ->
      has_rec_type env (EFieldSet field rec value) (TPrim TUnit).

(* ===== 6. Soundness ===== *)

Theorem infer_rec_type_sound :
  forall env e t,
    infer_rec_type env e = inl t ->
    has_rec_type env e t.
Proof.
  intros env e t H. revert env t H.
  induction e; intros env t H; simpl in H.
  - (* RMem *)
    destruct (infer_mem_type env m) as [mt | err] eqn:Hme.
    + injection H as <-. apply HRT_Mem. apply infer_mem_type_sound. exact Hme.
    + discriminate.
  - (* EFieldGet *)
    destruct (infer_rec_type env e) as [tr | err] eqn:Hrec; [| discriminate].
    destruct tr as [| | | | | | rname rflds | ]; try discriminate.
    (* tr = TRecord rname rflds *)
    destruct (field_lookup s rflds) as [ft |] eqn:Hfl; [| discriminate].
    injection H as <-.
    apply HRT_FieldGet with (name := rname) (fields := rflds).
    + apply IHe. exact Hrec.
    + exact Hfl.
  - (* EFieldSet *)
    destruct (infer_rec_type env e1) as [tr | err] eqn:Hrec; [| discriminate].
    destruct tr as [| | | | | | rname rflds | ]; try discriminate.
    (* tr = TRecord rname rflds *)
    destruct (field_lookup s rflds) as [ft |] eqn:Hfl; [| discriminate].
    destruct (infer_rec_type env e2) as [vt | err] eqn:Hval; [| discriminate].
    destruct (sarek_type_eq_dec vt ft) as [Heq | Hne]; [| discriminate].
    injection H as <-. subst vt.
    apply HRT_FieldSet with (name := rname) (fields := rflds) (field_t := ft).
    + apply IHe1. exact Hrec.
    + exact Hfl.
    + apply IHe2. exact Hval.
Qed.

(* ===== 7. Completeness ===== *)

Theorem infer_rec_type_complete :
  forall env e t,
    has_rec_type env e t ->
    infer_rec_type env e = inl t.
Proof.
  intros env0 e0 t0 H.
  induction H; simpl.
  - (* HRT_Mem *)
    match goal with Hmem : has_mem_type _ _ _ |- _ =>
      rewrite (infer_mem_type_complete Hmem); reflexivity end.
  - (* HRT_FieldGet *)
    rewrite IHhas_rec_type. rewrite H0. reflexivity.
  - (* HRT_FieldSet *)
    rewrite IHhas_rec_type1. rewrite H0. rewrite IHhas_rec_type2.
    destruct (sarek_type_eq_dec field_t field_t) as [_ | Hne].
    + reflexivity.
    + exfalso. apply Hne. reflexivity.
Qed.

(* ===== 8. Determinism ===== *)

Lemma has_rec_type_det :
  forall env e t1 t2,
    has_rec_type env e t1 ->
    has_rec_type env e t2 ->
    t1 = t2.
Proof.
  intros env e t1 t2 H1 H2.
  pose proof (infer_rec_type_complete H1) as Hc1.
  pose proof (infer_rec_type_complete H2) as Hc2.
  rewrite Hc1 in Hc2. injection Hc2 as <-. reflexivity.
Qed.

(* ===== 9. Type preservation ===== *)

Theorem rec_type_preservation :
  forall env e t,
    infer_rec_type env e = inl t <-> has_rec_type env e t.
Proof.
  intros env e t. split.
  - apply infer_rec_type_sound.
  - apply infer_rec_type_complete.
Qed.
