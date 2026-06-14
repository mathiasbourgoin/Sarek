(******************************************************************************)
(* Rocq 9 spec for special forms (EReturn, ECreateArray, ETyped) -- T3-S7.
 *
 * Extends ConstrSpec.v with a special_expr language that adds the three
 * "special" / meta-level forms from Sarek_typer.ml (infer_data_structure and
 * infer_special):
 *
 *   SEConstr ce                 -- delegation to the construction layer
 *   SEReturn allowed body       -- (return body)            Sarek_typer.ml:467
 *   SECreateArray size elt mem  -- create_array size : elt[]  Sarek_typer.ml:470
 *   SETyped body annot          -- (body : annot)           Sarek_typer.ml:505
 *
 * Mirroring the Sarek typing rules:
 *
 *   - EReturn e (line 467): [infer e]; the result type is exactly [te.ty] --
 *     a pure pass-through.  Sarek itself never rejects a return, but a return
 *     is only meaningful in a tail position.  We model that side condition with
 *     an explicit [allowed : bool] flag carried by the AST node: when it is
 *     [false] the form is rejected with [EarlyReturnNotAllowed].  This keeps the
 *     error constructor genuinely reachable while leaving the [allowed = true]
 *     path a faithful pass-through of the Sarek rule.
 *
 *   - ECreateArray (size, elem_ty, mem) (line 470): [infer size]; its type must
 *     unify with [t_int32], i.e. in this post-unification model it must equal
 *     [TPrim TInt32] (ArraySizeNotInt otherwise).  The element type [elem_ty] is
 *     resolved directly from the annotation, and the result type is
 *     [TArr (elem_ty, mem)], exactly as Sarek returns [arr_ty].
 *
 *   - ETyped (e, ty_expr) (line 505): [infer e]; [te.ty] must unify with the
 *     annotation [ty], i.e. equal it here (TypeAnnotMismatch otherwise).  The
 *     result type is the annotation [ty] (Sarek returns [{te with ty = repr ty}]).
 *
 * Error kinds:
 *   SConstrErr            -- delegated construction-layer error
 *   EarlyReturnNotAllowed -- a (return ..) in a non-tail (disallowed) position
 *   ArraySizeNotInt       -- create_array size expression is not TPrim TInt32
 *   TypeAnnotMismatch     -- annotated type != inferred type  (declared, got)
 *
 * Proven (all Qed, 0 admits):
 *   infer_special_type_sound    -- infer succeeds -> has_special_type
 *   infer_special_type_complete -- has_special_type -> infer succeeds
 *   has_special_type_det        -- uniqueness of the declarative judgement
 *   special_type_preservation   -- bi-directional iff
 ******************************************************************************)

From Stdlib Require Import Bool List String Arith Nat.
From TypeSafety Require Import TypeSafetySpec.
From TypeSafety Require Import VecSpec.
From TypeSafety Require Import RegistrySpec.
From TypeSafety Require Import ControlFlowSpec.
From TypeSafety Require Import OperatorSpec.
From TypeSafety Require Import FunSpec.
From TypeSafety Require Import MutSpec.
From TypeSafety Require Import PatternSpec.
From TypeSafety Require Import ConstrSpec.
Import TypeSafetySpec VecSpec RegistrySpec ControlFlowSpec OperatorSpec
       FunSpec MutSpec PatternSpec ConstrSpec.
Import ListNotations.

Set Implicit Arguments.

(* ===== 1. Special-form error kinds ===== *)

Inductive special_error : Type :=
  | SConstrErr            : constr_error -> special_error
  | EarlyReturnNotAllowed : special_error
  | ArraySizeNotInt       : sarek_type -> special_error  (* got, expected int32 *)
  | TypeAnnotMismatch     : sarek_type -> sarek_type -> special_error.
                            (* declared annotation, inferred *)

(* ===== 2. Special-form expression language =====
   SEReturn carries an [allowed] flag (whether a return is permitted at this
   syntactic position) and its body.
   SECreateArray carries the size expression, the resolved element type and the
   memory space; the result type is [TArr elt mem].
   SETyped carries the body and the annotated type. *)

Inductive special_expr : Type :=
  | SEConstr      : constr_expr -> special_expr
  | SEReturn      : bool -> special_expr -> special_expr
  | SECreateArray : special_expr -> sarek_type -> mem_space -> special_expr
  | SETyped       : special_expr -> sarek_type -> special_expr.

(* ===== 3. Algorithmic type inference ===== *)

Fixpoint infer_special_type (env : type_env) (mu : mut_env) (e : special_expr)
    : (sarek_type + special_error)%type :=
  match e with
  | SEConstr ce =>
      match infer_constr_type env mu ce with
      | inl t   => inl t
      | inr err => inr (SConstrErr err)
      end
  | SEReturn allowed body =>
      if allowed then
        match infer_special_type env mu body with
        | inl t   => inl t
        | inr err => inr err
        end
      else inr EarlyReturnNotAllowed
  | SECreateArray size elt mem =>
      match infer_special_type env mu size with
      | inr err => inr err
      | inl sz_ty =>
          match sarek_type_eq_dec sz_ty (TPrim TInt32) with
          | right _ => inr (ArraySizeNotInt sz_ty)
          | left  _ => inl (TArr elt mem)
          end
      end
  | SETyped body annot =>
      match infer_special_type env mu body with
      | inr err => inr err
      | inl got =>
          match sarek_type_eq_dec got annot with
          | right _ => inr (TypeAnnotMismatch annot got)
          | left  _ => inl annot
          end
      end
  end.

(* ===== 4. Declarative well-typedness judgement ===== *)

Inductive has_special_type
    : type_env -> mut_env -> special_expr -> sarek_type -> Prop :=
  | HST_Constr : forall env mu ce t,
      has_constr_type env mu ce t ->
      has_special_type env mu (SEConstr ce) t
  | HST_Return : forall env mu body t,
      has_special_type env mu body t ->
      has_special_type env mu (SEReturn true body) t
  | HST_CreateArray : forall env mu size elt mem,
      has_special_type env mu size (TPrim TInt32) ->
      has_special_type env mu (SECreateArray size elt mem) (TArr elt mem)
  | HST_Typed : forall env mu body annot,
      has_special_type env mu body annot ->
      has_special_type env mu (SETyped body annot) annot.

(* ===== 5. Soundness: infer succeeds -> has_special_type ===== *)

Theorem infer_special_type_sound :
  forall e env mu t,
    infer_special_type env mu e = inl t ->
    has_special_type env mu e t.
Proof.
  induction e as [ce | allowed body IHbody | size IHsize elt mem | body IHbody annot];
  intros env mu t H; simpl in H.
  - (* SEConstr *)
    destruct (infer_constr_type env mu ce) as [tc | err] eqn:Hce; [| discriminate].
    injection H as <-. apply HST_Constr. apply infer_constr_type_sound. exact Hce.
  - (* SEReturn *)
    destruct allowed; [| discriminate].
    destruct (infer_special_type env mu body) as [tb | err] eqn:Hbody;
      [| discriminate].
    injection H as <-. apply HST_Return. apply IHbody. exact Hbody.
  - (* SECreateArray *)
    destruct (infer_special_type env mu size) as [sz | err] eqn:Hsize;
      [| discriminate].
    destruct (sarek_type_eq_dec sz (TPrim TInt32)) as [Heq | _]; [| discriminate].
    injection H as <-. subst sz. apply HST_CreateArray. apply IHsize. exact Hsize.
  - (* SETyped *)
    destruct (infer_special_type env mu body) as [gt | err] eqn:Hbody;
      [| discriminate].
    destruct (sarek_type_eq_dec gt annot) as [Heq | _]; [| discriminate].
    injection H as <-. subst gt. apply HST_Typed. apply IHbody. exact Hbody.
Qed.

(* ===== 6. Completeness: has_special_type -> infer succeeds ===== *)

Theorem infer_special_type_complete :
  forall env mu e t,
    has_special_type env mu e t ->
    infer_special_type env mu e = inl t.
Proof.
  intros env mu e t H.
  induction H as [env mu ce t Hce
                  | env mu body t Hbody IHbody
                  | env mu size elt mem Hsize IHsize
                  | env mu body annot Hbody IHbody].
  - (* HST_Constr *)
    simpl. rewrite (infer_constr_type_complete Hce). reflexivity.
  - (* HST_Return *)
    simpl. rewrite IHbody. reflexivity.
  - (* HST_CreateArray *)
    simpl. rewrite IHsize.
    destruct (sarek_type_eq_dec (TPrim TInt32) (TPrim TInt32)) as [_ | Hne];
      [reflexivity | exfalso; apply Hne; reflexivity].
  - (* HST_Typed *)
    simpl. rewrite IHbody.
    destruct (sarek_type_eq_dec annot annot) as [_ | Hne];
      [reflexivity | exfalso; apply Hne; reflexivity].
Qed.

(* ===== 7. Determinism: piggybacking on completeness ===== *)

Lemma has_special_type_det :
  forall env mu e t1 t2,
    has_special_type env mu e t1 ->
    has_special_type env mu e t2 ->
    t1 = t2.
Proof.
  intros env mu e t1 t2 H1 H2.
  pose proof (infer_special_type_complete H1) as Hc1.
  pose proof (infer_special_type_complete H2) as Hc2.
  rewrite Hc1 in Hc2. injection Hc2 as <-. reflexivity.
Qed.

(* ===== 8. Type preservation: algorithmic <-> declarative ===== *)

Theorem special_type_preservation :
  forall env mu e t,
    infer_special_type env mu e = inl t <-> has_special_type env mu e t.
Proof.
  intros env mu e t. split.
  - apply infer_special_type_sound.
  - apply infer_special_type_complete.
Qed.
