(******************************************************************************)
(* Rocq 9 spec for algebraic value construction (ERecord, EConstr) -- T3-S6.
 *
 * Extends PatternSpec.v with a constr_expr language that adds two ways of
 * *building* algebraic values (the pattern layer matches them; this layer
 * constructs them):
 *
 *   CEPat    pe                          -- delegation to the pattern layer
 *   CERecord rname declared provided     -- { f1 = e1; ...; fn = en } : rname
 *   CEConstr tyname constrs cname arg     -- Constr (arg?)  of variant tyname
 *
 * This mirrors Sarek_typer.ml's [ERecord]/[EConstr] cases (~line 393/434):
 *
 *   - ERecord: the record's declared field list is known (the [TRecord] the
 *     literal targets).  Every *provided* field expression is inferred and its
 *     type must equal the *declared* type of the field of the same name.  A
 *     provided field name that is not declared is an error
 *     (Sarek: silently re-derives an anon_record; here we are stricter and
 *     reject with UnknownField, matching the registered-record path where the
 *     declared layout is authoritative).  The result type is the declared
 *     [TRecord rname declared].
 *
 *   - EConstr: the variant's full constructor list [constrs] is known.  The
 *     chosen constructor [cname] is looked up (reusing PatternSpec.lookup_constr):
 *       * not found                       -> UnknownConstr        (Unbound_constructor)
 *       * found, no payload + no arg        -> ok, result TVariant tyname constrs
 *       * found, payload pty + arg e        -> infer e; type must equal pty
 *                                              (FieldTypeMismatch on mismatch)
 *       * arity disagreement (payload vs arg presence)
 *                                          -> ConstrArity            (Wrong_arity)
 *     The result type is always the full [TVariant tyname constrs], exactly as
 *     Sarek returns [full_variant_ty].
 *
 * Error kinds:
 *   CPatternErr        -- delegated pattern-layer error
 *   FieldTypeMismatch  -- a provided field / payload type != declared type
 *   UnknownField       -- a provided record field is not declared
 *   UnknownConstr      -- a constructor name is not in the variant
 *   ConstrArity        -- payload-presence disagreement for a constructor
 *
 * Proven (all Qed, 0 admits):
 *   infer_constr_type_sound    -- infer succeeds -> has_constr_type
 *   infer_constr_type_complete -- has_constr_type -> infer succeeds
 *   has_constr_type_det        -- uniqueness of the declarative judgement
 *   constr_type_preservation   -- bi-directional iff
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
Import TypeSafetySpec VecSpec RegistrySpec ControlFlowSpec OperatorSpec
       FunSpec MutSpec PatternSpec.
Import ListNotations.

Set Implicit Arguments.

(* ===== 1. Construction error kinds ===== *)

Inductive constr_error : Type :=
  | CPatternErr       : pat_error -> constr_error
  | FieldTypeMismatch : string -> sarek_type -> sarek_type -> constr_error
                        (* field/constructor name, declared, got *)
  | UnknownField      : string -> constr_error  (* provided field not declared *)
  | UnknownConstr     : string -> constr_error  (* constructor not in variant *)
  | ConstrArity       : string -> constr_error. (* payload-presence disagrees *)

(* ===== 2. Construction expression language =====
   CERecord carries the declared record name + the declared field layout
   (name -> type), and the list of provided (field_name, value_expr).
   CEConstr carries the full variant (type name + constructor list), the
   chosen constructor name, and the optional payload expression. *)

Inductive constr_expr : Type :=
  | CEPat    : pat_expr -> constr_expr
  | CERecord : string -> list (string * sarek_type)
               -> list (string * constr_expr) -> constr_expr
  | CEConstr : string -> list (string * option sarek_type)
               -> string -> option constr_expr -> constr_expr.

(* Strong induction principle: the default one gives no IH for the provided
   field bodies (inside the CERecord list) nor for the CEConstr payload. *)
Lemma constr_expr_ind_strong :
  forall (P : constr_expr -> Prop),
    (forall pe, P (CEPat pe)) ->
    (forall rname declared provided,
        Forall (fun fld => P (snd fld)) provided ->
        P (CERecord rname declared provided)) ->
    (forall tyname constrs cname arg,
        match arg with Some a => P a | None => True end ->
        P (CEConstr tyname constrs cname arg)) ->
    forall e, P e.
Proof.
  intros P Hpat Hrec Hconstr.
  fix IH 1.
  intro e. destruct e as [pe | rname declared provided | tyname constrs cname arg].
  - apply Hpat.
  - apply Hrec.
    induction provided as [| fld rest IHrest].
    + apply Forall_nil.
    + apply Forall_cons; [apply IH | exact IHrest].
  - apply Hconstr. destruct arg as [a |]; [apply IH | exact I].
Defined.

(* ===== 3. Algorithmic type inference ===== *)

(* check_fields: type each provided field against the declared layout.
   [infer_f] is the recursive [infer_constr_type], threaded for termination.
   Returns inl tt on success, or the first error encountered. *)
Fixpoint check_fields
    (infer_f : type_env -> mut_env -> constr_expr -> (sarek_type + constr_error)%type)
    (env : type_env) (mu : mut_env)
    (declared : list (string * sarek_type))
    (provided : list (string * constr_expr))
    : (unit + constr_error)%type :=
  match provided with
  | [] => inl tt
  | (fname, fexpr) :: rest =>
      match field_lookup fname declared with
      | None => inr (UnknownField fname)
      | Some declared_ty =>
          match infer_f env mu fexpr with
          | inr err => inr err
          | inl got_ty =>
              match sarek_type_eq_dec got_ty declared_ty with
              | right _ => inr (FieldTypeMismatch fname declared_ty got_ty)
              | left  _ => check_fields infer_f env mu declared rest
              end
          end
      end
  end.

Fixpoint infer_constr_type (env : type_env) (mu : mut_env) (e : constr_expr)
    : (sarek_type + constr_error)%type :=
  match e with
  | CEPat pe =>
      match infer_pat_type env mu pe with
      | inl t   => inl t
      | inr err => inr (CPatternErr err)
      end
  | CERecord rname declared provided =>
      match check_fields infer_constr_type env mu declared provided with
      | inr err => inr err
      | inl _   => inl (TRecord rname declared)
      end
  | CEConstr tyname constrs cname arg =>
      match lookup_constr constrs cname with
      | None => inr (UnknownConstr cname)
      | Some payload =>
          match payload, arg with
          | None, None => inl (TVariant tyname constrs)
          | Some pty, Some a =>
              match infer_constr_type env mu a with
              | inr err => inr err
              | inl got =>
                  match sarek_type_eq_dec got pty with
                  | right _ => inr (FieldTypeMismatch cname pty got)
                  | left  _ => inl (TVariant tyname constrs)
                  end
              end
          | _, _ => inr (ConstrArity cname)
          end
      end
  end.

(* ===== 4. Declarative well-typedness judgement =====
   fields_have_type lifts per-field checking over the provided list. *)

Inductive has_constr_type
    : type_env -> mut_env -> constr_expr -> sarek_type -> Prop :=
  | HCT_Pat : forall env mu pe t,
      has_pat_type env mu pe t ->
      has_constr_type env mu (CEPat pe) t
  | HCT_Record : forall env mu rname declared provided,
      fields_have_type env mu declared provided ->
      has_constr_type env mu (CERecord rname declared provided)
        (TRecord rname declared)
  | HCT_ConstrNone : forall env mu tyname constrs cname,
      lookup_constr constrs cname = Some None ->
      has_constr_type env mu (CEConstr tyname constrs cname None)
        (TVariant tyname constrs)
  | HCT_ConstrSome : forall env mu tyname constrs cname pty arg,
      lookup_constr constrs cname = Some (Some pty) ->
      has_constr_type env mu arg pty ->
      has_constr_type env mu (CEConstr tyname constrs cname (Some arg))
        (TVariant tyname constrs)

with fields_have_type
    : type_env -> mut_env -> list (string * sarek_type)
      -> list (string * constr_expr) -> Prop :=
  | FHT_nil : forall env mu declared,
      fields_have_type env mu declared []
  | FHT_cons : forall env mu declared fname fexpr declared_ty rest,
      field_lookup fname declared = Some declared_ty ->
      has_constr_type env mu fexpr declared_ty ->
      fields_have_type env mu declared rest ->
      fields_have_type env mu declared ((fname, fexpr) :: rest).

(* Combined induction scheme for the mutually-recursive judgements. *)
Scheme has_constr_type_mind := Induction for has_constr_type Sort Prop
  with fields_have_type_mind := Induction for fields_have_type Sort Prop.

(* ===== 5. Soundness: infer succeeds -> has_constr_type ===== *)

(* check_fields succeeds -> every provided field has its declared type.
   Proved by induction on the provided list, using per-field soundness carried
   as [Hfields] (a Forall from the strong induction on e). *)
Lemma check_fields_sound :
  forall provided env mu declared,
    Forall (fun fld => forall env' mu' t',
                infer_constr_type env' mu' (snd fld) = inl t' ->
                has_constr_type env' mu' (snd fld) t') provided ->
    check_fields infer_constr_type env mu declared provided = inl tt ->
    fields_have_type env mu declared provided.
Proof.
  induction provided as [| fld rest IHrest];
  intros env mu declared Hfields Hcheck.
  - apply FHT_nil.
  - destruct fld as [fname fexpr]. simpl in Hcheck.
    inversion Hfields as [| ? ? Hhead Htail]; subst.
    destruct (field_lookup fname declared) as [declared_ty | ] eqn:Hlk;
      [| discriminate].
    destruct (infer_constr_type env mu fexpr) as [got_ty | err] eqn:Hbody;
      [| discriminate Hcheck].
    destruct (sarek_type_eq_dec got_ty declared_ty) as [Heq | _];
      [| discriminate Hcheck].
    subst got_ty.
    apply FHT_cons with (declared_ty := declared_ty).
    + exact Hlk.
    + apply Hhead. exact Hbody.
    + apply IHrest; [exact Htail | exact Hcheck].
Qed.

Theorem infer_constr_type_sound :
  forall e env mu t,
    infer_constr_type env mu e = inl t ->
    has_constr_type env mu e t.
Proof.
  induction e as [pe | rname declared provided Hfields
                     | tyname constrs cname arg IHarg]
    using constr_expr_ind_strong;
  intros env mu t H; simpl in H.
  - (* CEPat *)
    destruct (infer_pat_type env mu pe) as [tp | err] eqn:Hpe; [| discriminate].
    injection H as <-. apply HCT_Pat. apply infer_pat_type_sound. exact Hpe.
  - (* CERecord *)
    destruct (check_fields infer_constr_type env mu declared provided)
      as [u | err] eqn:Hcheck; [| discriminate].
    destruct u. injection H as <-.
    apply HCT_Record. apply check_fields_sound; [exact Hfields | exact Hcheck].
  - (* CEConstr *)
    destruct (lookup_constr constrs cname) as [payload | ] eqn:Hlk;
      [| discriminate].
    destruct payload as [pty |]; destruct arg as [a |].
    + (* Some pty, Some a *)
      destruct (infer_constr_type env mu a) as [got | err] eqn:Harg;
        [| discriminate].
      destruct (sarek_type_eq_dec got pty) as [Heq | _]; [| discriminate].
      injection H as <-. subst got.
      apply HCT_ConstrSome with (pty := pty).
      * exact Hlk.
      * apply IHarg. exact Harg.
    + (* Some pty, None -> ConstrArity *)
      discriminate.
    + (* None, Some a -> ConstrArity *)
      discriminate.
    + (* None, None *)
      injection H as <-. apply HCT_ConstrNone. exact Hlk.
Qed.

(* ===== 6. Completeness: has_constr_type -> infer succeeds ===== *)

Theorem infer_constr_type_complete :
  forall env mu e t,
    has_constr_type env mu e t ->
    infer_constr_type env mu e = inl t.
Proof.
  intros env mu e t H.
  induction H using has_constr_type_mind
    with (P0 := fun env mu declared provided
                    (_ : fields_have_type env mu declared provided) =>
                  check_fields infer_constr_type env mu declared provided
                    = inl tt).
  - (* HCT_Pat *)
    simpl. rewrite (infer_pat_type_complete h). reflexivity.
  - (* HCT_Record *)
    simpl. rewrite IHhas_constr_type. reflexivity.
  - (* HCT_ConstrNone *)
    simpl. rewrite e. reflexivity.
  - (* HCT_ConstrSome *)
    simpl. rewrite e. rewrite IHhas_constr_type.
    destruct (sarek_type_eq_dec pty pty) as [_ | Hne];
      [reflexivity | exfalso; apply Hne; reflexivity].
  - (* FHT_nil *)
    reflexivity.
  - (* FHT_cons *)
    simpl. rewrite e. rewrite IHhas_constr_type.
    destruct (sarek_type_eq_dec declared_ty declared_ty) as [_ | Hne];
      [exact IHhas_constr_type0 | exfalso; apply Hne; reflexivity].
Qed.

(* ===== 7. Determinism: piggybacking on completeness ===== *)

Lemma has_constr_type_det :
  forall env mu e t1 t2,
    has_constr_type env mu e t1 ->
    has_constr_type env mu e t2 ->
    t1 = t2.
Proof.
  intros env mu e t1 t2 H1 H2.
  pose proof (infer_constr_type_complete H1) as Hc1.
  pose proof (infer_constr_type_complete H2) as Hc2.
  rewrite Hc1 in Hc2. injection Hc2 as <-. reflexivity.
Qed.

(* ===== 8. Type preservation: algorithmic <-> declarative ===== *)

Theorem constr_type_preservation :
  forall env mu e t,
    infer_constr_type env mu e = inl t <-> has_constr_type env mu e t.
Proof.
  intros env mu e t. split.
  - apply infer_constr_type_sound.
  - apply infer_constr_type_complete.
Qed.
