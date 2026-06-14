(******************************************************************************)
(* Rocq 9 spec for GPU / BSP forms (ELetShared, ESuperstep) -- T3-S8.
 *
 * Extends SpecialSpec.v with a gpu_expr language that adds the two BSP / GPU
 * constructs from Sarek_typer.ml:768-811 (the [let%shared] and [let%superstep]
 * forms in infer):
 *
 *   GESpecial se          -- delegation to the special-form layer
 *   GELetShared name ty b -- let%shared name : ty = .. in b   Sarek_typer.ml:769
 *   GESuperstep b         -- let%superstep .. = b             Sarek_typer.ml:800
 *
 * Mirroring the Sarek typing rules:
 *
 *   - ELetShared (name, elem_ty, size_opt, body) (line 769): Sarek resolves the
 *     element type and binds [name] to the shared-array type
 *     [TArr (elem_t, Shared)] (line 781), then infers the body; the form's type
 *     is [tbody.ty].  In this post-unification model the AST node carries the
 *     fully-resolved array type [ty] directly.  That [ty] must be a shared array
 *     -- i.e. of shape [TArr _ Shared] -- which we check explicitly:
 *       * a non-array [ty]                 -> SharedNotArray ty
 *       * an array in a non-Shared space   -> SharedNotShared ty
 *     The (size_opt : int32) side-condition of the source rule (line 776) is a
 *     unification on the optional size sub-expression; it is orthogonal to the
 *     body typing modelled here and is elided (documented for ASSUMPTIONS.md).
 *     On success [name] is bound to [ty] in the environment and the body is
 *     inferred; the result type is the body type.
 *
 *   - ESuperstep (name, divergent, step_body, cont) (line 800): Sarek infers the
 *     step body and unifies its type with [t_unit] (line 803).  We model the
 *     load-bearing well-typedness condition of a superstep: its body must have
 *     type [TPrim TUnit] (SuperstepBodyNotUnit otherwise).  A superstep is a
 *     barrier-delimited side-effecting block, so the construct itself yields
 *     [TPrim TUnit].  The source's separate continuation [cont] (whose type
 *     Sarek returns at line 809) is sequenced at the enclosing block level and
 *     is not part of the superstep node modelled here (documented for
 *     ASSUMPTIONS.md).
 *
 * Error kinds:
 *   GSpecialErr         -- delegated special-form-layer error
 *   SharedNotArray      -- let%shared annotated type is not an array       (got)
 *   SharedNotShared     -- let%shared array is not in the Shared space      (got)
 *   SuperstepBodyNotUnit -- superstep body type is not TPrim TUnit          (got)
 *
 * Proven (all Qed, 0 admits):
 *   infer_gpu_type_sound    -- infer succeeds -> has_gpu_type
 *   infer_gpu_type_complete -- has_gpu_type -> infer succeeds
 *   has_gpu_type_det        -- uniqueness of the declarative judgement
 *   gpu_type_preservation   -- bi-directional iff
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
From TypeSafety Require Import SpecialSpec.
Import TypeSafetySpec VecSpec RegistrySpec ControlFlowSpec OperatorSpec
       FunSpec MutSpec PatternSpec ConstrSpec SpecialSpec.
Import ListNotations.

Set Implicit Arguments.

(* ===== 1. GPU-form error kinds ===== *)

Inductive gpu_error : Type :=
  | GSpecialErr          : special_error -> gpu_error
  | SharedNotArray       : sarek_type -> gpu_error   (* got, expected TArr _ Shared *)
  | SharedNotShared      : sarek_type -> gpu_error   (* got an array, wrong memspace *)
  | SuperstepBodyNotUnit : sarek_type -> gpu_error.  (* got, expected TPrim TUnit *)

(* ===== 2. GPU-form expression language =====
   GELetShared carries the bound name, the (resolved) shared-array type and the
   body inferred under [name : ty].
   GESuperstep carries its (unit-typed) body. *)

Inductive gpu_expr : Type :=
  | GESpecial   : special_expr -> gpu_expr
  | GELetShared : string -> sarek_type -> gpu_expr -> gpu_expr
  | GESuperstep : gpu_expr -> gpu_expr.

(* ===== 3. Algorithmic type inference ===== *)

Fixpoint infer_gpu_type (env : type_env) (mu : mut_env) (e : gpu_expr)
    : (sarek_type + gpu_error)%type :=
  match e with
  | GESpecial se =>
      match infer_special_type env mu se with
      | inl t   => inl t
      | inr err => inr (GSpecialErr err)
      end
  | GELetShared name ty body =>
      match ty with
      | TArr elt Shared =>
          infer_gpu_type ((name, ty) :: env) mu body
      | TArr _ _ => inr (SharedNotShared ty)
      | _        => inr (SharedNotArray ty)
      end
  | GESuperstep body =>
      match infer_gpu_type env mu body with
      | inr err => inr err
      | inl bt  =>
          match sarek_type_eq_dec bt (TPrim TUnit) with
          | right _ => inr (SuperstepBodyNotUnit bt)
          | left  _ => inl (TPrim TUnit)
          end
      end
  end.

(* ===== 4. Declarative well-typedness judgement ===== *)

Inductive has_gpu_type
    : type_env -> mut_env -> gpu_expr -> sarek_type -> Prop :=
  | HGT_Special : forall env mu se t,
      has_special_type env mu se t ->
      has_gpu_type env mu (GESpecial se) t
  | HGT_LetShared : forall env mu name elt body t,
      has_gpu_type ((name, TArr elt Shared) :: env) mu body t ->
      has_gpu_type env mu (GELetShared name (TArr elt Shared) body) t
  | HGT_Superstep : forall env mu body,
      has_gpu_type env mu body (TPrim TUnit) ->
      has_gpu_type env mu (GESuperstep body) (TPrim TUnit).

(* ===== 5. Soundness: infer succeeds -> has_gpu_type ===== *)

Theorem infer_gpu_type_sound :
  forall e env mu t,
    infer_gpu_type env mu e = inl t ->
    has_gpu_type env mu e t.
Proof.
  induction e as [se | name ty body IHbody | body IHbody];
  intros env mu t H; simpl in H.
  - (* GESpecial *)
    destruct (infer_special_type env mu se) as [ts | err] eqn:Hse; [| discriminate].
    injection H as <-. apply HGT_Special. apply infer_special_type_sound. exact Hse.
  - (* GELetShared *)
    destruct ty as [pt | rt | vt | elt mem | args ret | tys | rn fs | vn cs];
      try discriminate.
    (* only TArr elt mem reaches here *)
    destruct mem; try discriminate.
    (* Shared: recurse under the extended env *)
    apply HGT_LetShared. apply IHbody. exact H.
  - (* GESuperstep *)
    destruct (infer_gpu_type env mu body) as [bt | err] eqn:Hbody; [| discriminate].
    destruct (sarek_type_eq_dec bt (TPrim TUnit)) as [Heq | _]; [| discriminate].
    injection H as <-. subst bt. apply HGT_Superstep. apply IHbody. exact Hbody.
Qed.

(* ===== 6. Completeness: has_gpu_type -> infer succeeds ===== *)

Theorem infer_gpu_type_complete :
  forall env mu e t,
    has_gpu_type env mu e t ->
    infer_gpu_type env mu e = inl t.
Proof.
  intros env mu e t H.
  induction H as [env mu se t Hse
                  | env mu name elt body t Hbody IHbody
                  | env mu body Hbody IHbody].
  - (* HGT_Special *)
    simpl. rewrite (infer_special_type_complete Hse). reflexivity.
  - (* HGT_LetShared *)
    simpl. exact IHbody.
  - (* HGT_Superstep *)
    simpl. rewrite IHbody.
    destruct (sarek_type_eq_dec (TPrim TUnit) (TPrim TUnit)) as [_ | Hne];
      [reflexivity | exfalso; apply Hne; reflexivity].
Qed.

(* ===== 7. Determinism: piggybacking on completeness ===== *)

Lemma has_gpu_type_det :
  forall env mu e t1 t2,
    has_gpu_type env mu e t1 ->
    has_gpu_type env mu e t2 ->
    t1 = t2.
Proof.
  intros env mu e t1 t2 H1 H2.
  pose proof (infer_gpu_type_complete H1) as Hc1.
  pose proof (infer_gpu_type_complete H2) as Hc2.
  rewrite Hc1 in Hc2. injection Hc2 as <-. reflexivity.
Qed.

(* ===== 8. Type preservation: algorithmic <-> declarative ===== *)

Theorem gpu_type_preservation :
  forall env mu e t,
    infer_gpu_type env mu e = inl t <-> has_gpu_type env mu e t.
Proof.
  intros env mu e t. split.
  - apply infer_gpu_type_sound.
  - apply infer_gpu_type_complete.
Qed.
