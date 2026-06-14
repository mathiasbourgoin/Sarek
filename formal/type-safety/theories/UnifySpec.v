(******************************************************************************)
(* Rocq 9 spec for the Sarek PPX unification algorithm — T2-UNIFY.
 *
 * Target: ~/dev/SPOC/sarek/ppx/Sarek_types.ml (unify, occurs, repr)
 *
 * Goal: model the FUNCTIONAL semantics of the HM unifier in Sarek_types.ml.
 * The production code uses mutable [tvar ref]. This spec replaces mutation
 * with a pure substitution (association list: list (nat * pre_type)).
 *
 * Relationship with TypeSafetySpec.v:
 *   - TypeSafetySpec models POST-unification types (sarek_type, no TVar).
 *   - UnifySpec models PRE-unification types (pre_type, with PVar n).
 *   - Reuses prim_type and reg_type from TypeSafetySpec via import.
 *
 * Termination strategy:
 *   - follow_pvar is a Fixpoint on nat (fuel). follow is a Definition that
 *     dispatches on pre_type: ground constructors are identity; PVar delegates
 *     to follow_pvar. Because follow is a Definition (not Fixpoint), Rocq's
 *     kernel reduces follow fuel s (PPrim p) = PPrim p immediately by iota,
 *     regardless of the fuel value.
 *   - unify_fun is a Fixpoint on fuel with {struct fuel} annotation. The
 *     PTuple branch calls unify_list_with (unify_fun n) where n is the
 *     predecessor — strictly smaller, satisfying the guard checker.
 *   - occurs_in = recurse_occurs (structural Fixpoint on pre_type) composed
 *     with follow. No mutual fixpoint is needed.
 *
 * Proof strategy:
 *   - Ground-vs-ground cases (PPrim/PPrim, PReg/PReg): unify_fun (S n) reduces
 *     by reflexivity after follow inlines. Exposed via unify_fun_prim_prim and
 *     unify_fun_reg_reg proved by reflexivity.
 *   - PVar-vs-ground cases: require follow n s (PVar id) = PVar id as a
 *     hypothesis (the PVar is unbound at depth n). Proved by simpl + rewrite.
 *
 * Proven properties (all Qed, 0 admits):
 *   pre_type_ind_strong              — custom induction for PTuple
 *   follow_ground_prim/reg/tuple     — ground types fixed by follow
 *   follow_pvar_none                 — unbound PVar follows to itself
 *   prim_type_beq_true/false         — prim_type_beq reflects =/≠
 *   reg_type_beq_true/false          — reg_type_beq reflects =/≠
 *   subst_lookup_head                — head binding lookup
 *   occurs_ground_false              — PPrim has no occurrences
 *   occurs_ground_reg_false          — PReg has no occurrences
 *   unify_fun_prim_prim              — unfolding lemma for PPrim/PPrim
 *   unify_fun_reg_reg                — unfolding lemma for PReg/PReg
 *   unify_fun_var_prim               — unfolding lemma for PVar/PPrim (free)
 *   unify_fun_var_reg                — unfolding lemma for PVar/PReg (free)
 *   unify_zero_none                  — zero fuel always returns None
 *   unify_prim_sound/complete        — PPrim unification sound and complete
 *   unify_reg_sound/complete         — PReg unification sound and complete
 *   unify_var_binds_prim             — free PVar/PPrim adds binding
 *   unify_var_binds_reg              — free PVar/PReg adds binding
 *   occurs_check_blocks_prim         — occurs_in (PPrim p) = false
 *   occurs_check_blocks_reg          — occurs_in (PReg r) = false
 ******************************************************************************)

From Stdlib Require Import Bool List String Arith Nat.
From TypeSafety Require Import TypeSafetySpec.
Import TypeSafetySpec.
Import ListNotations.

Set Implicit Arguments.

(* ===== 1. Pre-unification type universe ===== *)

(* pre_type adds PVar (unification variable, id = nat) to the resolved types.
   Mirrors Sarek_types.typ with TVar, minus TRecord/TVariant/TVec/TArr/TFun
   (those are elided for this phase — see ASSUMPTIONS.md). *)
Inductive pre_type : Type :=
  | PVar   : nat -> pre_type             (* unification variable (id) *)
  | PPrim  : prim_type -> pre_type       (* primitive type *)
  | PReg   : reg_type  -> pre_type       (* registered type *)
  | PTuple : list pre_type -> pre_type.  (* tuple *)

(* pre_type_ind_strong: custom induction principle that supplies Forall IHs
   for PTuple elements (Rocq's default does not descend into list fields). *)
Lemma pre_type_ind_strong : forall (P : pre_type -> Prop),
  (forall id, P (PVar id)) ->
  (forall p, P (PPrim p)) ->
  (forall r, P (PReg r)) ->
  (forall ts, Forall P ts -> P (PTuple ts)) ->
  forall t, P t.
Proof.
  intros P Hvar Hprim Hreg Htup.
  fix IH 1. intro t. destruct t as [id | p | r | ts].
  - apply Hvar.
  - apply Hprim.
  - apply Hreg.
  - apply Htup. induction ts as [| t' rest IHrest].
    + apply Forall_nil.
    + apply Forall_cons; [apply IH | exact IHrest].
Defined.

(* ===== 2. Pure substitution ===== *)

(* A substitution is an association list mapping nat ids to pre_type.
   This replaces the mutable [tvar ref] chains of Sarek_types.ml. *)
Definition pre_subst := list (nat * pre_type).

(* subst_lookup: first-match lookup. *)
Fixpoint subst_lookup (s : pre_subst) (id : nat) : option pre_type :=
  match s with
  | [] => None
  | (id', t) :: rest =>
      if Nat.eqb id id' then Some t else subst_lookup rest id
  end.

(* ===== 3. Follow (repr in Sarek_types.ml) ===== *)

(* follow_pvar: follow a PVar id through the substitution.
   Fixpoint on fuel. Returns the representative type (PVar if unbound). *)
Fixpoint follow_pvar (fuel : nat) (s : pre_subst) (id : nat) : pre_type :=
  match fuel with
  | 0 => PVar id
  | S n =>
      match subst_lookup s id with
      | None            => PVar id
      | Some (PVar id') => follow_pvar n s id'  (* follow PVar -> PVar chain *)
      | Some t          => t                     (* ground: stop *)
      end
  end.

(* follow: dispatch on t.
   Ground constructors are fixed points returned as-is.
   PVar id delegates to follow_pvar.
   Because follow is a DEFINITION (not a Fixpoint), the kernel reduces
   follow fuel s (PPrim p) = PPrim p by iota regardless of fuel. *)
Definition follow (fuel : nat) (s : pre_subst) (t : pre_type) : pre_type :=
  match t with
  | PPrim _  => t
  | PReg  _  => t
  | PTuple _ => t
  | PVar id  => follow_pvar fuel s id
  end.

Lemma follow_ground_prim : forall fuel s p,
  follow fuel s (PPrim p) = PPrim p.
Proof. reflexivity. Qed.

Lemma follow_ground_reg : forall fuel s r,
  follow fuel s (PReg r) = PReg r.
Proof. reflexivity. Qed.

Lemma follow_ground_tuple : forall fuel s ts,
  follow fuel s (PTuple ts) = PTuple ts.
Proof. reflexivity. Qed.

(* An unbound PVar (not in substitution) follows to itself. *)
Lemma follow_pvar_none : forall fuel s id,
  subst_lookup s id = None ->
  follow fuel s (PVar id) = PVar id.
Proof.
  intros fuel s id Hnone.
  unfold follow.
  destruct fuel as [| n].
  - reflexivity.
  - simpl. rewrite Hnone. reflexivity.
Qed.

(* ===== 4. Occurs check (Sarek_types.ml:occurs) ===== *)

(* recurse_occurs: structural recursion on pre_type. *)
Fixpoint recurse_occurs (id : nat) (t : pre_type) : bool :=
  match t with
  | PVar id'  => Nat.eqb id id'
  | PPrim _   => false
  | PReg  _   => false
  | PTuple ts => List.existsb (recurse_occurs id) ts
  end.

(* occurs_in: follow t to its representative, then check structurally. *)
Definition occurs_in (fuel : nat) (s : pre_subst) (id : nat) (t : pre_type) : bool :=
  recurse_occurs id (follow fuel s t).

Lemma occurs_ground_false : forall fuel s id p,
  occurs_in fuel s id (PPrim p) = false.
Proof. reflexivity. Qed.

Lemma occurs_ground_reg_false : forall fuel s id r,
  occurs_in fuel s id (PReg r) = false.
Proof. reflexivity. Qed.

(* ===== 5. Unify (Sarek_types.ml:unify) ===== *)

(* unify_list_with: higher-order helper; structurally recursive on the list.
   Mirrors infer_list_with from TypeSafetySpec. *)
Fixpoint unify_list_with
    (unify_f : pre_subst -> pre_type -> pre_type -> option pre_subst)
    (s : pre_subst) (ts1 ts2 : list pre_type) : option pre_subst :=
  match ts1, ts2 with
  | [], []               => Some s
  | t1 :: rest1, t2 :: rest2 =>
      match unify_f s t1 t2 with
      | None    => None
      | Some s' => unify_list_with unify_f s' rest1 rest2
      end
  | _, _ => None
  end.

(* unify_fun: pure functional unifier.
   {struct fuel} annotation is required — without it Rocq may infer the wrong
   structural argument (e.g., the pre_subst list s).
   PTuple branch uses unify_fun n (predecessor), satisfying the guard. *)
Fixpoint unify_fun (fuel : nat) (s : pre_subst) (t1 t2 : pre_type) {struct fuel}
    : option pre_subst :=
  match fuel with
  | 0    => None
  | S n  =>
      match follow n s t1, follow n s t2 with
      | PVar id1, PVar id2 =>
          if Nat.eqb id1 id2 then Some s
          else Some ((id1, PVar id2) :: s)
      | PVar id, PPrim p =>
          if occurs_in n s id (PPrim p) then None
          else Some ((id, PPrim p) :: s)
      | PVar id, PReg r =>
          if occurs_in n s id (PReg r) then None
          else Some ((id, PReg r) :: s)
      | PVar id, PTuple ts =>
          if occurs_in n s id (PTuple ts) then None
          else Some ((id, PTuple ts) :: s)
      | PPrim p, PVar id =>
          if occurs_in n s id (PPrim p) then None
          else Some ((id, PPrim p) :: s)
      | PReg r, PVar id =>
          if occurs_in n s id (PReg r) then None
          else Some ((id, PReg r) :: s)
      | PTuple ts, PVar id =>
          if occurs_in n s id (PTuple ts) then None
          else Some ((id, PTuple ts) :: s)
      | PPrim p1, PPrim p2 =>
          if prim_type_beq p1 p2 then Some s else None
      | PReg r1, PReg r2 =>
          if reg_type_beq r1 r2 then Some s else None
      | PTuple ts1, PTuple ts2 =>
          if negb (Nat.eqb (List.length ts1) (List.length ts2)) then None
          else unify_list_with (unify_fun n) s ts1 ts2
      | _, _ => None
      end
  end.

(* ===== 6. Boolean equality reflection lemmas ===== *)

Lemma prim_type_beq_true : forall p1 p2,
  prim_type_beq p1 p2 = true <-> p1 = p2.
Proof.
  intros p1 p2. split.
  - intro H. destruct p1, p2; simpl in H; try discriminate; reflexivity.
  - intro H. subst. destruct p2; reflexivity.
Qed.

Lemma prim_type_beq_false : forall p1 p2,
  prim_type_beq p1 p2 = false <-> p1 <> p2.
Proof.
  intros p1 p2. split.
  - intros H Heq. subst. destruct p2; simpl in H; discriminate.
  - intros H. destruct (prim_type_beq p1 p2) eqn:Heq.
    + exfalso. apply H. apply prim_type_beq_true. exact Heq.
    + reflexivity.
Qed.

Lemma reg_type_beq_true : forall r1 r2,
  reg_type_beq r1 r2 = true <-> r1 = r2.
Proof.
  intros r1 r2. split.
  - intro H. destruct r1, r2; simpl in H; try discriminate; reflexivity.
  - intro H. subst. destruct r2; reflexivity.
Qed.

Lemma reg_type_beq_false : forall r1 r2,
  reg_type_beq r1 r2 = false <-> r1 <> r2.
Proof.
  intros r1 r2. split.
  - intros H Heq. subst. destruct r2; simpl in H; discriminate.
  - intros H. destruct (reg_type_beq r1 r2) eqn:Heq.
    + exfalso. apply H. apply reg_type_beq_true. exact Heq.
    + reflexivity.
Qed.

(* ===== 7. Substitution utility lemma ===== *)

Lemma subst_lookup_head : forall s id t,
  subst_lookup ((id, t) :: s) id = Some t.
Proof.
  intros s id t. simpl. rewrite Nat.eqb_refl. reflexivity.
Qed.

(* ===== 8. Unfolding lemmas for unify_fun ===== *)

(* Ground-vs-ground: proved by reflexivity (follow Definition + struct fuel). *)
Lemma unify_fun_prim_prim : forall n s p1 p2,
  unify_fun (S n) s (PPrim p1) (PPrim p2) =
  if prim_type_beq p1 p2 then Some s else None.
Proof. intros. reflexivity. Qed.

Lemma unify_fun_reg_reg : forall n s r1 r2,
  unify_fun (S n) s (PReg r1) (PReg r2) =
  if reg_type_beq r1 r2 then Some s else None.
Proof. intros. reflexivity. Qed.

(* PVar-vs-ground: require follow n s (PVar id) = PVar id as a precondition
   (the variable is unbound at depth n). *)
Lemma unify_fun_var_prim : forall n s id p,
  follow n s (PVar id) = PVar id ->
  unify_fun (S n) s (PVar id) (PPrim p) =
  if occurs_in n s id (PPrim p) then None
  else Some ((id, PPrim p) :: s).
Proof.
  intros n s id p Hfoll. unfold follow in Hfoll. simpl. rewrite Hfoll. reflexivity.
Qed.

Lemma unify_fun_var_reg : forall n s id r,
  follow n s (PVar id) = PVar id ->
  unify_fun (S n) s (PVar id) (PReg r) =
  if occurs_in n s id (PReg r) then None
  else Some ((id, PReg r) :: s).
Proof.
  intros n s id r Hfoll. unfold follow in Hfoll. simpl. rewrite Hfoll. reflexivity.
Qed.

(* ===== 9. Zero-fuel lemma ===== *)

Lemma unify_zero_none : forall s t1 t2,
  unify_fun 0 s t1 t2 = None.
Proof. reflexivity. Qed.

(* ===== 10. Soundness and completeness lemmas ===== *)

Lemma unify_prim_sound : forall fuel s p1 p2 s',
  unify_fun fuel s (PPrim p1) (PPrim p2) = Some s' ->
  s' = s /\ p1 = p2.
Proof.
  intros fuel s p1 p2 s' H.
  destruct fuel as [| n]; [discriminate |].
  rewrite unify_fun_prim_prim in H.
  destruct (prim_type_beq p1 p2) eqn:Heq; [| discriminate].
  inversion H. subst.
  split; [reflexivity | apply prim_type_beq_true; exact Heq].
Qed.

Lemma prim_type_beq_refl : forall p,
  prim_type_beq p p = true.
Proof. intro p. destruct p; reflexivity. Qed.

Lemma reg_type_beq_refl : forall r,
  reg_type_beq r r = true.
Proof. intro r. destruct r; reflexivity. Qed.

Lemma unify_prim_complete : forall fuel s p,
  fuel > 0 ->
  unify_fun fuel s (PPrim p) (PPrim p) = Some s.
Proof.
  intros fuel s p Hfuel.
  destruct fuel as [| n]; [inversion Hfuel |].
  rewrite unify_fun_prim_prim.
  rewrite prim_type_beq_refl.
  reflexivity.
Qed.

Lemma unify_reg_sound : forall fuel s r1 r2 s',
  unify_fun fuel s (PReg r1) (PReg r2) = Some s' ->
  s' = s /\ r1 = r2.
Proof.
  intros fuel s r1 r2 s' H.
  destruct fuel as [| n]; [discriminate |].
  rewrite unify_fun_reg_reg in H.
  destruct (reg_type_beq r1 r2) eqn:Heq; [| discriminate].
  inversion H. subst.
  split; [reflexivity | apply reg_type_beq_true; exact Heq].
Qed.

Lemma unify_reg_complete : forall fuel s r,
  fuel > 0 ->
  unify_fun fuel s (PReg r) (PReg r) = Some s.
Proof.
  intros fuel s r Hfuel.
  destruct fuel as [| n]; [inversion Hfuel |].
  rewrite unify_fun_reg_reg.
  rewrite reg_type_beq_refl.
  reflexivity.
Qed.

(* unify_var_binds_prim: unifying a free PVar with PPrim adds the binding.
   The occurs check on PPrim always returns false (no PVar inside), so the
   binding is added unconditionally. *)
Lemma unify_var_binds_prim : forall n s id p s',
  follow n s (PVar id) = PVar id ->
  unify_fun (S n) s (PVar id) (PPrim p) = Some s' ->
  subst_lookup s' id = Some (PPrim p).
Proof.
  intros n s id p s' Hfoll Hunify.
  rewrite unify_fun_var_prim in Hunify; [| exact Hfoll].
  rewrite occurs_ground_false in Hunify.
  inversion Hunify. subst s'.
  apply subst_lookup_head.
Qed.

Lemma unify_var_binds_reg : forall n s id r s',
  follow n s (PVar id) = PVar id ->
  unify_fun (S n) s (PVar id) (PReg r) = Some s' ->
  subst_lookup s' id = Some (PReg r).
Proof.
  intros n s id r s' Hfoll Hunify.
  rewrite unify_fun_var_reg in Hunify; [| exact Hfoll].
  rewrite occurs_ground_reg_false in Hunify.
  inversion Hunify. subst s'.
  apply subst_lookup_head.
Qed.

(* occurs_check_blocks_prim/reg: PPrim/PReg contain no PVar, so the occurs
   check always returns false. Binding is never blocked for ground types. *)
Lemma occurs_check_blocks_prim : forall n s id p,
  occurs_in n s id (PPrim p) = false.
Proof. reflexivity. Qed.

Lemma occurs_check_blocks_reg : forall n s id r,
  occurs_in n s id (PReg r) = false.
Proof. reflexivity. Qed.

(* ===== 11. apply_subst (for extraction) ===== *)

(* apply_subst: apply substitution to a pre_type, following PVar chains. *)
Fixpoint apply_subst (fuel : nat) (s : pre_subst) (t : pre_type) : pre_type :=
  match t with
  | PPrim _   => t
  | PReg  _   => t
  | PVar id   => follow_pvar fuel s id
  | PTuple ts => PTuple (List.map (apply_subst fuel s) ts)
  end.
