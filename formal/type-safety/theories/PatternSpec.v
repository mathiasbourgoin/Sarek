(******************************************************************************)
(* Rocq 9 spec for pattern matching (EMatch) -- T3-S5 (PatternSpec).
 *
 * Extends MutSpec.v with a pat_expr language that adds:
 *
 *   PEMut   me                     -- delegation to the mutable-binding layer
 *   PEMatch scrut branches         -- match scrut with | C1 x -> e1 | ...
 *
 * Branches are modelled as a list of (constructor_name, opt bound_var, body).
 * This mirrors Sarek_typer.ml infer_match_cases / infer_pattern (~line 846):
 *
 *   - EMatch (scrutinee, cases): infer the scrutinee type [ts]; it must be a
 *     TVariant.  Each branch pattern is a single-level variant constructor
 *     pattern PConstr (name, arg_pat).
 *   - infer_pattern PConstr: look up [name] among the variant's constructors.
 *     * not found            -> Unbound_constructor  (here: PatternMismatch)
 *     * found, no payload     -> bind nothing; pattern matches.
 *     * found, has payload    -> bind the payload type under the branch's
 *                                bound variable in the body's environment.
 *   - infer_match_cases: the FIRST branch body fixes [result_ty]; every
 *     subsequent branch body must unify to the same [result_ty]
 *     (here: equality, post-unification).  An empty case list is an error
 *     (Invalid_kernel "empty match"; here: PEEmpty).
 *
 * Payload binding scope: a bound variable is added to the type_env only for
 * the body of its own branch (and shadows any outer binding of the same name),
 * exactly as add_var in infer_pattern scopes it to that case.  The mutability
 * environment is threaded unchanged into delegated reads (PEMut) and is not
 * extended by pattern binders -- pattern-bound variables are immutable, which
 * matches vi_mutable = false in infer_pattern PVar / PConstr.
 *
 * Error kinds:
 *   PEMutErr      -- delegated mutable-layer error
 *   PENotVariant  -- scrutinee did not infer to a TVariant
 *   PEMismatch    -- branch constructor is not in the variant
 *   PEBranchType  -- a branch body's type != the first branch's result type
 *   PEEmpty       -- empty branch list
 *
 * Proven (all Qed, 0 admits):
 *   infer_pat_type_sound    -- infer succeeds -> has_pat_type
 *   infer_pat_type_complete -- has_pat_type -> infer succeeds
 *   has_pat_type_det        -- uniqueness of the declarative judgement
 *   pat_type_preservation   -- bi-directional iff
 ******************************************************************************)

From Stdlib Require Import Bool List String Arith Nat.
From TypeSafety Require Import TypeSafetySpec.
From TypeSafety Require Import VecSpec.
From TypeSafety Require Import RegistrySpec.
From TypeSafety Require Import ControlFlowSpec.
From TypeSafety Require Import OperatorSpec.
From TypeSafety Require Import FunSpec.
From TypeSafety Require Import MutSpec.
Import TypeSafetySpec VecSpec RegistrySpec ControlFlowSpec OperatorSpec FunSpec MutSpec.
Import ListNotations.

Set Implicit Arguments.

(* ===== 1. Constructor lookup over a variant's constructor list =====
   A variant carries [list (string * option sarek_type)]: each constructor
   name maps to an optional payload type.  Lookup returns:
     None              -- the name is not a constructor of this variant
     Some payload_opt  -- it is, with its (optional) payload type. *)

Fixpoint lookup_constr (constrs : list (string * option sarek_type))
    (name : string) : option (option sarek_type) :=
  match constrs with
  | [] => None
  | (cname, payload) :: rest =>
      if String.eqb name cname then Some payload
      else lookup_constr rest name
  end.

(* ===== 2. Pattern-match error kinds ===== *)

Inductive pat_error : Type :=
  | PEMutErr     : mut_error -> pat_error
  | PENotVariant : sarek_type -> pat_error      (* scrutinee's inferred type *)
  | PEMismatch   : string -> pat_error          (* unknown constructor name *)
  | PEBranchType : sarek_type -> sarek_type -> pat_error (* result, got *)
  | PEEmpty      : pat_error.

(* ===== 3. Pattern-match expression language =====
   A branch is (constructor_name, optional bound var, body).  The bound var is
   meaningful only when the constructor has a payload; when present it binds
   the payload type in the body's environment. *)

Inductive pat_expr : Type :=
  | PEMut   : mut_expr -> pat_expr
  | PEMatch : pat_expr -> list (string * option string * pat_expr) -> pat_expr.

(* Strong induction principle: the default one gives no IH for the branch
   bodies inside the list.  This yields [Forall] over the branch bodies. *)
Lemma pat_expr_ind_strong :
  forall (P : pat_expr -> Prop),
    (forall me, P (PEMut me)) ->
    (forall scrut,
        P scrut ->
        forall branches,
        Forall (fun b => P (snd b)) branches ->
        P (PEMatch scrut branches)) ->
    forall e, P e.
Proof.
  intros P Hmut Hmatch.
  fix IH 1.
  intro e. destruct e as [me | scrut branches].
  - apply Hmut.
  - apply Hmatch.
    + apply IH.
    + induction branches as [| b rest IHrest].
      * apply Forall_nil.
      * apply Forall_cons; [apply IH | exact IHrest].
Defined.

(* ===== 4. Algorithmic type inference =====
   Threads the type environment and the (delegated) mutability environment. *)

(* branch_body_env: the environment in which a branch body is typed.  When the
   constructor carries a payload [Some pty] and the branch binds it to [v], the
   body sees [(v, pty)] prepended (shadowing any outer binding); otherwise the
   environment is unchanged.  Shared by the inference fixpoint and the
   declarative judgement so the two stay syntactically aligned. *)
Definition branch_body_env
    (env : type_env)
    (payload : option sarek_type) (bvar : option string) : type_env :=
  match payload, bvar with
  | Some pty, Some v => (v, pty) :: env
  | _, _ => env
  end.

(* check_branches: type every branch against the scrutinee's variant
   constructor list, requiring all branch bodies to share [result_ty].
   [infer_f] is the recursive [infer_pat_type], passed in for termination.
   The result type is fixed by the *caller* (the first branch's body type).
   Returns inl tt on success, or inr err on the first failure. *)
Fixpoint check_branches
    (infer_f : type_env -> mut_env -> pat_expr -> (sarek_type + pat_error)%type)
    (env : type_env) (mu : mut_env)
    (constrs : list (string * option sarek_type))
    (result_ty : sarek_type)
    (branches : list (string * option string * pat_expr))
    : (unit + pat_error)%type :=
  match branches with
  | [] => inl tt
  | (cname, bvar, body) :: rest =>
      match lookup_constr constrs cname with
      | None => inr (PEMismatch cname)
      | Some payload =>
          match infer_f (branch_body_env env payload bvar) mu body with
          | inr err => inr err
          | inl tbody =>
              match sarek_type_eq_dec tbody result_ty with
              | right _ => inr (PEBranchType result_ty tbody)
              | left  _ =>
                  check_branches infer_f env mu constrs result_ty rest
              end
          end
      end
  end.

Fixpoint infer_pat_type (env : type_env) (mu : mut_env) (e : pat_expr)
    : (sarek_type + pat_error)%type :=
  match e with
  | PEMut me =>
      match infer_mut_type env mu me with
      | inl t   => inl t
      | inr err => inr (PEMutErr err)
      end
  | PEMatch scrut branches =>
      match infer_pat_type env mu scrut with
      | inr err => inr err
      | inl tscrut =>
          match tscrut with
          | TVariant _ constrs =>
              match branches with
              | [] => inr PEEmpty
              | (cname, bvar, body) :: rest =>
                  (* The first branch fixes the result type; the remaining
                     branches are checked to agree with it. *)
                  match lookup_constr constrs cname with
                  | None => inr (PEMismatch cname)
                  | Some payload =>
                      match infer_pat_type
                              (branch_body_env env payload bvar) mu body with
                      | inr err => inr err
                      | inl result_ty =>
                          match check_branches infer_pat_type env mu
                                  constrs result_ty rest with
                          | inr err => inr err
                          | inl _   => inl result_ty
                          end
                      end
                  end
              end
          | other => inr (PENotVariant other)
          end
      end
  end.

(* ===== 5. Declarative well-typedness judgement =====
   has_branch types a single branch body against the variant constructors,
   requiring it to have the fixed result type.  branches_have_type lifts that
   over a branch list. *)

Inductive has_pat_type : type_env -> mut_env -> pat_expr -> sarek_type -> Prop :=
  | HPT_Mut : forall env mu me t,
      has_mut_type env mu me t ->
      has_pat_type env mu (PEMut me) t
  | HPT_Match :
      forall env mu scrut tyname constrs
             cname bvar body payload rest result_ty,
      has_pat_type env mu scrut (TVariant tyname constrs) ->
      lookup_constr constrs cname = Some payload ->
      has_pat_type (branch_body_env env payload bvar) mu body result_ty ->
      branches_have_type env mu constrs result_ty rest ->
      has_pat_type env mu
        (PEMatch scrut ((cname, bvar, body) :: rest)) result_ty

with branches_have_type
    : type_env -> mut_env -> list (string * option sarek_type)
      -> sarek_type -> list (string * option string * pat_expr) -> Prop :=
  | BHT_nil : forall env mu constrs result_ty,
      branches_have_type env mu constrs result_ty []
  | BHT_cons : forall env mu constrs result_ty cname bvar body payload rest,
      lookup_constr constrs cname = Some payload ->
      has_pat_type (branch_body_env env payload bvar) mu body result_ty ->
      branches_have_type env mu constrs result_ty rest ->
      branches_have_type env mu constrs result_ty
        ((cname, bvar, body) :: rest).

(* Combined induction scheme for the mutually-recursive judgements. *)
Scheme has_pat_type_mind := Induction for has_pat_type Sort Prop
  with branches_have_type_mind := Induction for branches_have_type Sort Prop.

(* ===== 6. Soundness: infer succeeds -> has_pat_type ===== *)

(* check_branches succeeds -> every branch body has the fixed result type.
   Proved by induction on the branch list, using the per-body soundness fact
   carried as [Hbodies] (a Forall coming from the strong induction on e). *)
Lemma check_branches_sound :
  forall branches env mu constrs result_ty,
    Forall (fun b => forall env' mu' t',
                infer_pat_type env' mu' (snd b) = inl t' ->
                has_pat_type env' mu' (snd b) t') branches ->
    check_branches infer_pat_type env mu constrs result_ty branches = inl tt ->
    branches_have_type env mu constrs result_ty branches.
Proof.
  induction branches as [| b rest IHrest];
  intros env mu constrs result_ty Hbodies Hcheck.
  - apply BHT_nil.
  - destruct b as [[cname bvar] body]. simpl in Hcheck.
    inversion Hbodies as [| ? ? Hhead Htail]; subst.
    destruct (lookup_constr constrs cname) as [payload | ] eqn:Hlk;
      [| discriminate].
    destruct (infer_pat_type (branch_body_env env payload bvar) mu body)
      as [tbody | err] eqn:Hbody; [| discriminate Hcheck].
    destruct (sarek_type_eq_dec tbody result_ty) as [Heq | _];
      [| discriminate Hcheck].
    subst tbody.
    apply BHT_cons with (payload := payload).
    + exact Hlk.
    + apply Hhead. exact Hbody.
    + apply IHrest; [exact Htail | exact Hcheck].
Qed.

Theorem infer_pat_type_sound :
  forall e env mu t,
    infer_pat_type env mu e = inl t ->
    has_pat_type env mu e t.
Proof.
  induction e as [me | scrut IHscrut branches Hbranches]
    using pat_expr_ind_strong;
  intros env mu t H; simpl in H.
  - (* PEMut *)
    destruct (infer_mut_type env mu me) as [tm | err] eqn:Hme; [| discriminate].
    injection H as <-. apply HPT_Mut. apply infer_mut_type_sound. exact Hme.
  - (* PEMatch *)
    destruct (infer_pat_type env mu scrut) as [tscrut | err] eqn:Hscrut;
      [| discriminate].
    destruct tscrut as [ | | | | | | tyname fields | tyname constrs ];
      try discriminate.
    (* Case-split the branch list via the Forall hypothesis (destruct on the
       list itself fails to infer the elimination motive because the strong
       induction predicate quantifies over env/mu/t). *)
    inversion Hbranches as [| b rest Hhead Htail Heqbs]; subst.
    + (* empty branch list -> PEEmpty error, contradicts H *)
      simpl in H. discriminate.
    + (* non-empty: head fixes result_ty, whole list checked *)
      destruct b as [[cname bvar] body]. simpl in Hhead. simpl in H.
      destruct (lookup_constr constrs cname) as [payload | ] eqn:Hlk;
        [| discriminate].
      destruct (infer_pat_type (branch_body_env env payload bvar) mu body)
        as [result_ty | err] eqn:Hbody; [| discriminate].
      destruct (check_branches infer_pat_type env mu constrs result_ty rest)
        as [u | err] eqn:Hcheck; [| discriminate].
      destruct u.
      injection H as <-.
      apply HPT_Match with (tyname := tyname) (constrs := constrs)
                           (payload := payload).
      * apply IHscrut. exact Hscrut.
      * exact Hlk.
      * apply Hhead. exact Hbody.
      * apply check_branches_sound; [exact Htail | exact Hcheck].
Qed.

(* ===== 7. Completeness: has_pat_type -> infer succeeds ===== *)

Theorem infer_pat_type_complete :
  forall env mu e t,
    has_pat_type env mu e t ->
    infer_pat_type env mu e = inl t.
Proof.
  intros env mu e t H.
  induction H using has_pat_type_mind
    with (P0 := fun env mu constrs result_ty branches
                    (_ : branches_have_type env mu constrs result_ty branches) =>
                  check_branches infer_pat_type env mu constrs result_ty branches
                    = inl tt).
  - (* HPT_Mut *)
    simpl. rewrite (infer_mut_type_complete h). reflexivity.
  - (* HPT_Match *)
    simpl.
    rewrite IHhas_pat_type1. rewrite e. rewrite IHhas_pat_type2.
    rewrite IHhas_pat_type3. reflexivity.
  - (* BHT_nil *)
    reflexivity.
  - (* BHT_cons *)
    simpl. rewrite e. rewrite IHhas_pat_type.
    destruct (sarek_type_eq_dec result_ty result_ty) as [_ | Hne];
      [exact IHhas_pat_type0 | exfalso; apply Hne; reflexivity].
Qed.

(* ===== 8. Determinism: piggybacking on completeness ===== *)

Lemma has_pat_type_det :
  forall env mu e t1 t2,
    has_pat_type env mu e t1 ->
    has_pat_type env mu e t2 ->
    t1 = t2.
Proof.
  intros env mu e t1 t2 H1 H2.
  pose proof (infer_pat_type_complete H1) as Hc1.
  pose proof (infer_pat_type_complete H2) as Hc2.
  rewrite Hc1 in Hc2. injection Hc2 as <-. reflexivity.
Qed.

(* ===== 9. Type preservation: algorithmic <-> declarative ===== *)

Theorem pat_type_preservation :
  forall env mu e t,
    infer_pat_type env mu e = inl t <-> has_pat_type env mu e t.
Proof.
  intros env mu e t. split.
  - apply infer_pat_type_sound.
  - apply infer_pat_type_complete.
Qed.
