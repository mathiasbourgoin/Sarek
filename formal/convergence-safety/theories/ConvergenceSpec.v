(******************************************************************************)
(* Rocq 9 spec for Sarek_convergence.ml — barrier safety analysis.
 *
 * Target: ~/dev/SPOC/sarek/ppx/Sarek_convergence.ml
 * Properties: dim_merge_monoid, check_seq_hom, diverged_clean_iff_barrier_free,
 *             mode_monotone, varying_if_flags_barriers, cdcf_check_agreement
 *
 * Abstract model: expr captures the convergence-relevant structure.
 * Elided: TEMatch (subsumed by EIf), TESuperstep, TELetRec, TEPragma, TEOpen,
 *         TELetShared — documented in ASSUMPTIONS.md.
 ******************************************************************************)

From Stdlib Require Import Bool List.
Import ListNotations.

(* ===== 1. Abstract expression type ===== *)

Inductive expr : Type :=
  | ELit     : expr
  | EVary    : expr
  | EBarrier : expr
  | EBinop   : expr -> expr -> expr
  | EUnop    : expr -> expr
  | EIf      : expr -> expr -> expr -> expr    (* cond, then, else *)
  | EWhile   : expr -> expr -> expr            (* cond, body *)
  | EFor     : expr -> expr -> expr -> expr    (* lo, hi, body *)
  | ESeq     : list expr -> expr
  | ELet     : expr -> expr -> expr            (* value, body *)
  | EApp     : list expr -> expr.

Inductive exec_mode : Type := Converged | Diverged.

Inductive error : Type := BarrierError.

(* ===== 2. is_varying ===== *)

Fixpoint is_varying (e : expr) : bool :=
  match e with
  | EVary         => true
  | ELit | EBarrier => false
  | EBinop a b    => is_varying a || is_varying b
  | EUnop e       => is_varying e
  | EIf c t el    => is_varying c || is_varying t || is_varying el
  | EWhile c b    => is_varying c || is_varying b
  | EFor lo hi b  => is_varying lo || is_varying hi || is_varying b
  | ESeq es       => existsb is_varying es
  | ELet v b      => is_varying v || is_varying b
  | EApp args     => existsb is_varying args
  end.

(* ===== 3. barrier_free ===== *)

Fixpoint barrier_free (e : expr) : bool :=
  match e with
  | EBarrier       => false
  | ELit | EVary   => true
  | EBinop a b     => barrier_free a && barrier_free b
  | EUnop e        => barrier_free e
  | EIf c t el     => barrier_free c && barrier_free t && barrier_free el
  | EWhile c b     => barrier_free c && barrier_free b
  | EFor lo hi b   => barrier_free lo && barrier_free hi && barrier_free b
  | ESeq es        => forallb barrier_free es
  | ELet v b       => barrier_free v && barrier_free b
  | EApp args      => forallb barrier_free args
  end.

(* ===== 4. has_diverging_cf ===== *)

Fixpoint has_diverging_cf (e : expr) : bool :=
  match e with
  | EIf c t el    => is_varying c || has_diverging_cf t || has_diverging_cf el
  | EWhile c b    => is_varying c || has_diverging_cf b
  | EFor lo hi b  => is_varying lo || is_varying hi || has_diverging_cf b
  | EBinop a b    => has_diverging_cf a || has_diverging_cf b
  | EUnop e       => has_diverging_cf e
  | ESeq es       => existsb has_diverging_cf es
  | ELet v b      => has_diverging_cf v || has_diverging_cf b
  | EApp args     => existsb has_diverging_cf args
  | _             => false
  end.

(* ===== 5. check ===== *)

Fixpoint check (m : exec_mode) (e : expr) : list error :=
  match e with
  | EBarrier =>
      match m with Diverged => [BarrierError] | Converged => [] end
  | ELit | EVary   => []
  | EBinop a b     => check m a ++ check m b
  | EUnop e        => check m e
  | EIf cond t el  =>
      let inner := if is_varying cond then Diverged else m in
      check m cond ++ check inner t ++ check inner el
  | EWhile cond b  =>
      let inner := if is_varying cond then Diverged else m in
      check m cond ++ check inner b
  | EFor lo hi b   =>
      let inner := if is_varying lo || is_varying hi then Diverged else m in
      check m lo ++ check m hi ++ check inner b
  | ESeq es        => concat (map (check m) es)
  | ELet v b       => check m v ++ check m b
  | EApp args      => concat (map (check m) args)
  end.

(* ===== 6. dim_usage ===== *)

Record dim_usage : Type := mk_dim_usage {
  uses_x : bool; uses_y : bool; uses_z : bool;
  uses_block_dim : bool; uses_grid_dim : bool;
  uses_thread_idx : bool; uses_block_idx : bool;
  uses_shared_mem : bool
}.

Definition empty_dim_usage :=
  mk_dim_usage false false false false false false false false.

Definition merge_dim_usage (a b : dim_usage) := mk_dim_usage
  (a.(uses_x)          || b.(uses_x))
  (a.(uses_y)          || b.(uses_y))
  (a.(uses_z)          || b.(uses_z))
  (a.(uses_block_dim)  || b.(uses_block_dim))
  (a.(uses_grid_dim)   || b.(uses_grid_dim))
  (a.(uses_thread_idx) || b.(uses_thread_idx))
  (a.(uses_block_idx)  || b.(uses_block_idx))
  (a.(uses_shared_mem) || b.(uses_shared_mem)).

(* ===== 7. Custom induction principle for expr / list expr ===== *)

Section ExprListInd.

Variable P     : expr -> Prop.
Variable Plist : list expr -> Prop.

Hypothesis IH_lit     : P ELit.
Hypothesis IH_vary    : P EVary.
Hypothesis IH_barrier : P EBarrier.
Hypothesis IH_binop   : forall a b, P a -> P b -> P (EBinop a b).
Hypothesis IH_unop    : forall e, P e -> P (EUnop e).
Hypothesis IH_if      : forall c t el, P c -> P t -> P el -> P (EIf c t el).
Hypothesis IH_while   : forall c b, P c -> P b -> P (EWhile c b).
Hypothesis IH_for     : forall lo hi b, P lo -> P hi -> P b -> P (EFor lo hi b).
Hypothesis IH_seq     : forall es, Plist es -> P (ESeq es).
Hypothesis IH_let     : forall v b, P v -> P b -> P (ELet v b).
Hypothesis IH_app     : forall args, Plist args -> P (EApp args).
Hypothesis IH_nil     : Plist [].
Hypothesis IH_cons    : forall e es, P e -> Plist es -> Plist (e :: es).

Fixpoint expr_list_rect (e : expr) : P e :=
  match e with
  | ELit     => IH_lit
  | EVary    => IH_vary
  | EBarrier => IH_barrier
  | EBinop a b   => IH_binop a b (expr_list_rect a) (expr_list_rect b)
  | EUnop u      => IH_unop u (expr_list_rect u)
  | EIf c t el   =>
      IH_if c t el (expr_list_rect c) (expr_list_rect t) (expr_list_rect el)
  | EWhile c b   => IH_while c b (expr_list_rect c) (expr_list_rect b)
  | EFor lo hi b =>
      IH_for lo hi b (expr_list_rect lo) (expr_list_rect hi) (expr_list_rect b)
  | ESeq es =>
      IH_seq es
        ((fix go (xs : list expr) : Plist xs :=
          match xs with
          | []     => IH_nil
          | x :: t => IH_cons x t (expr_list_rect x) (go t)
          end) es)
  | ELet v b     => IH_let v b (expr_list_rect v) (expr_list_rect b)
  | EApp args =>
      IH_app args
        ((fix go (xs : list expr) : Plist xs :=
          match xs with
          | []     => IH_nil
          | x :: t => IH_cons x t (expr_list_rect x) (go t)
          end) args)
  end.

End ExprListInd.

(* ===== 8. Helper lemmas ===== *)

Lemma incl_app_mono : forall (A : Type) (l1 l2 l1' l2' : list A),
  incl l1 l1' -> incl l2 l2' -> incl (l1 ++ l2) (l1' ++ l2').
Proof.
  intros A l1 l2 l1' l2' H1 H2 x Hx.
  apply in_app_iff in Hx. apply in_app_iff.
  destruct Hx as [Ha | Hb].
  - left.  exact (H1 x Ha).
  - right. exact (H2 x Hb).
Qed.

Lemma app_nil_iff : forall (A : Type) (l1 l2 : list A),
  l1 ++ l2 = [] <-> l1 = [] /\ l2 = [].
Proof.
  intros A l1 l2. split.
  - apply app_eq_nil.
  - intros [H1 H2]. subst. reflexivity.
Qed.

Lemma diverged_absorbing : forall (b : bool),
  (if b then Diverged else Diverged) = Diverged.
Proof. intros []; reflexivity. Qed.

(* ===== 9. Property 1: dim_merge_monoid ===== *)

Theorem merge_dim_comm : forall a b,
  merge_dim_usage a b = merge_dim_usage b a.
Proof.
  intros [x1 y1 z1 bd1 gd1 ti1 bi1 sm1]
         [x2 y2 z2 bd2 gd2 ti2 bi2 sm2].
  unfold merge_dim_usage; simpl.
  f_equal; apply Bool.orb_comm.
Qed.

Theorem merge_dim_assoc : forall a b c,
  merge_dim_usage a (merge_dim_usage b c) =
  merge_dim_usage (merge_dim_usage a b) c.
Proof.
  intros [x1 y1 z1 bd1 gd1 ti1 bi1 sm1]
         [x2 y2 z2 bd2 gd2 ti2 bi2 sm2]
         [x3 y3 z3 bd3 gd3 ti3 bi3 sm3].
  unfold merge_dim_usage; simpl.
  f_equal; apply Bool.orb_assoc.
Qed.

Theorem merge_dim_idempotent : forall a,
  merge_dim_usage a a = a.
Proof.
  intros [x y z bd gd ti bi sm].
  unfold merge_dim_usage; simpl.
  f_equal; apply Bool.orb_diag.
Qed.

Theorem merge_dim_empty_r : forall a,
  merge_dim_usage a empty_dim_usage = a.
Proof.
  intros [x y z bd gd ti bi sm].
  unfold merge_dim_usage, empty_dim_usage; simpl.
  f_equal; apply Bool.orb_false_r.
Qed.

Theorem merge_dim_empty_l : forall a,
  merge_dim_usage empty_dim_usage a = a.
Proof.
  intros [x y z bd gd ti bi sm].
  unfold merge_dim_usage, empty_dim_usage; simpl.
  f_equal; apply Bool.orb_false_l.
Qed.

(* ===== 10. Property 2: check_seq_hom ===== *)

Theorem check_seq_hom : forall m es1 es2,
  check m (ESeq (es1 ++ es2)) = check m (ESeq es1) ++ check m (ESeq es2).
Proof.
  intros m es1 es2.
  simpl. rewrite map_app, concat_app. reflexivity.
Qed.

(* ===== 11. Property 3: diverged_clean_iff_barrier_free ===== *)

Theorem diverged_clean_iff_barrier_free : forall e,
  check Diverged e = [] <-> barrier_free e = true.
Proof.
  apply (expr_list_rect
    (fun e    => check Diverged e = [] <-> barrier_free e = true)
    (fun es   =>
       concat (map (check Diverged) es) = [] <->
       forallb barrier_free es = true)).
  (* ELit *) - simpl. tauto.
  (* EVary *) - simpl. tauto.
  (* EBarrier — both sides false *)
  - simpl. split; intro H; discriminate.
  (* EBinop a b *)
  - intros a b IHa IHb. simpl.
    rewrite app_nil_iff, andb_true_iff. tauto.
  (* EUnop e *)
  - intros e IH. exact IH.
  (* EIf c t el — Diverged mode is absorbing: inner is always Diverged *)
  - intros c t el IHc IHt IHel. simpl.
    rewrite diverged_absorbing.
    rewrite !app_nil_iff, !andb_true_iff. tauto.
  (* EWhile c b *)
  - intros c b IHc IHb. simpl.
    rewrite diverged_absorbing.
    rewrite !app_nil_iff, !andb_true_iff. tauto.
  (* EFor lo hi b *)
  - intros lo hi b IHlo IHhi IHb. simpl.
    rewrite diverged_absorbing.
    rewrite !app_nil_iff, !andb_true_iff. tauto.
  (* ESeq es *) - intros es IH. exact IH.
  (* ELet v b *)
  - intros v b IHv IHb. simpl.
    rewrite app_nil_iff, andb_true_iff. tauto.
  (* EApp args *) - intros args IH. exact IH.
  (* nil *)
  - simpl. tauto.
  (* cons e es *)
  - intros e es IHe IHes. simpl.
    rewrite app_nil_iff, andb_true_iff. tauto.
Qed.

(* ===== 12. Property 4: mode_monotone ===== *)

Theorem mode_monotone : forall e,
  incl (check Converged e) (check Diverged e).
Proof.
  apply (expr_list_rect
    (fun e  => incl (check Converged e) (check Diverged e))
    (fun es =>
       incl (concat (map (check Converged) es))
            (concat (map (check Diverged) es)))).
  (* ELit *)   - simpl. apply incl_refl.
  (* EVary *)  - simpl. apply incl_refl.
  (* EBarrier: Converged→[], Diverged→[BarrierError] *)
  - simpl. apply incl_nil_l.
  (* EBinop a b: ++ is right-assoc; one incl_app_mono per level *)
  - intros a b IHa IHb. simpl.
    apply incl_app_mono; [exact IHa | exact IHb].
  (* EUnop e *)
  - intros e IH. exact IH.
  (* EIf c t el
     - Diverged side: rewrite (if X then D else D) → D with diverged_absorbing
     - Then: varying c → inner stays D; non-varying → inner upgrades C→D *)
  - intros c t el IHc IHt IHel. simpl.
    rewrite (diverged_absorbing (is_varying c)).
    destruct (is_varying c).
    + (* true: cv-inner=D, dv-inner=D, so t/el parts are identical *)
      apply incl_app_mono; [exact IHc | apply incl_refl].
    + (* false: cv-inner=C, dv-inner=D; need IH for t and el *)
      apply incl_app_mono; [exact IHc | apply incl_app_mono; [exact IHt | exact IHel]].
  (* EWhile c b *)
  - intros c b IHc IHb. simpl.
    rewrite (diverged_absorbing (is_varying c)).
    destruct (is_varying c).
    + apply incl_app_mono; [exact IHc | apply incl_refl].
    + apply incl_app_mono; [exact IHc | exact IHb].
  (* EFor lo hi b *)
  - intros lo hi b IHlo IHhi IHb. simpl.
    rewrite (diverged_absorbing (is_varying lo || is_varying hi)).
    destruct (is_varying lo || is_varying hi).
    + apply incl_app_mono; [exact IHlo | apply incl_app_mono; [exact IHhi | apply incl_refl]].
    + apply incl_app_mono; [exact IHlo | apply incl_app_mono; [exact IHhi | exact IHb]].
  (* ESeq es *) - intros es IH. exact IH.
  (* ELet v b *)
  - intros v b IHv IHb. simpl.
    apply incl_app_mono; [exact IHv | exact IHb].
  (* EApp args *) - intros args IH. exact IH.
  (* nil *)   - simpl. apply incl_refl.
  (* cons e es *)
  - intros e es IHe IHes. simpl.
    apply incl_app_mono; [exact IHe | exact IHes].
Qed.

(* ===== 13. Helper: is_varying = false implies check Converged = [] ===== *)

Lemma not_varying_converged_clean : forall e,
  is_varying e = false -> check Converged e = [].
Proof.
  apply (expr_list_rect
    (fun e  => is_varying e = false -> check Converged e = [])
    (fun es =>
       existsb is_varying es = false ->
       concat (map (check Converged) es) = [])).
  (* ELit *)    - intros _. reflexivity.
  (* EVary *)   - intros H. discriminate.
  (* EBarrier *) - intros _. reflexivity.
  (* EBinop a b *)
  - intros a b IHa IHb H. simpl in *.
    apply orb_false_iff in H. destruct H as [Ha Hb].
    rewrite (IHa Ha), (IHb Hb). reflexivity.
  (* EUnop e *)
  - intros e IH H. simpl in *. apply IH. exact H.
  (* EIf c t el *)
  - intros c t el IHc IHt IHel H. simpl in *.
    (* H : (is_varying c || is_varying t) || is_varying el = false *)
    apply orb_false_iff in H. destruct H as [Hct Hel].
    apply orb_false_iff in Hct. destruct Hct as [Hc Ht].
    rewrite (IHc Hc). simpl. rewrite Hc. simpl.
    rewrite (IHt Ht), (IHel Hel). reflexivity.
  (* EWhile c b *)
  - intros c b IHc IHb H. simpl in *.
    apply orb_false_iff in H. destruct H as [Hc Hb].
    rewrite (IHc Hc). simpl. rewrite Hc. simpl.
    exact (IHb Hb).
  (* EFor lo hi b *)
  - intros lo hi b IHlo IHhi IHb H. simpl in *.
    (* H : (is_varying lo || is_varying hi) || is_varying b = false *)
    apply orb_false_iff in H. destruct H as [Hlohi Hb].
    apply orb_false_iff in Hlohi. destruct Hlohi as [Hlo Hhi].
    rewrite (IHlo Hlo), (IHhi Hhi). simpl.
    rewrite Hlo, Hhi. simpl.
    exact (IHb Hb).
  (* ESeq es *) - intros es IH H. simpl in *. apply IH. exact H.
  (* ELet v b *)
  - intros v b IHv IHb H. simpl in *.
    apply orb_false_iff in H. destruct H as [Hv Hb].
    rewrite (IHv Hv), (IHb Hb). reflexivity.
  (* EApp args *) - intros args IH H. simpl in *. apply IH. exact H.
  (* nil *) - intros _. reflexivity.
  (* cons e es *)
  - intros e es IHe IHes H. simpl in *.
    apply orb_false_iff in H. destruct H as [He Hes].
    rewrite (IHe He), (IHes Hes). reflexivity.
Qed.

(* ===== 14. Property 6: cdcf_check_agreement ===== *)

Theorem cdcf_check_agreement : forall e,
  has_diverging_cf e = false -> check Converged e = [].
Proof.
  apply (expr_list_rect
    (fun e  => has_diverging_cf e = false -> check Converged e = [])
    (fun es =>
       existsb has_diverging_cf es = false ->
       concat (map (check Converged) es) = [])).
  (* ELit *)    - intros _. reflexivity.
  (* EVary *)   - intros _. reflexivity.
  (* EBarrier: check Converged EBarrier = [], and cdcf = false *)
  - intros _. reflexivity.
  (* EBinop a b *)
  - intros a b IHa IHb H. simpl in *.
    apply orb_false_iff in H. destruct H as [Ha Hb].
    rewrite (IHa Ha), (IHb Hb). reflexivity.
  (* EUnop e *)
  - intros e IH H. simpl in *. apply IH. exact H.
  (* EIf c t el *)
  - intros c t el IHc IHt IHel H. simpl in *.
    (* H : (is_varying c || has_diverging_cf t) || has_diverging_cf el = false *)
    apply orb_false_iff in H. destruct H as [Hct Hel].
    apply orb_false_iff in Hct. destruct Hct as [Hc Ht].
    rewrite (not_varying_converged_clean c Hc). simpl.
    rewrite Hc. simpl.
    rewrite (IHt Ht), (IHel Hel). reflexivity.
  (* EWhile c b *)
  - intros c b IHc IHb H. simpl in *.
    apply orb_false_iff in H. destruct H as [Hc Hb].
    rewrite (not_varying_converged_clean c Hc). simpl.
    rewrite Hc. simpl.
    exact (IHb Hb).
  (* EFor lo hi b *)
  - intros lo hi b IHlo IHhi IHb H. simpl in *.
    (* H : (is_varying lo || is_varying hi) || has_diverging_cf b = false *)
    apply orb_false_iff in H. destruct H as [Hlohi Hb].
    apply orb_false_iff in Hlohi. destruct Hlohi as [Hlo Hhi].
    rewrite (not_varying_converged_clean lo Hlo),
            (not_varying_converged_clean hi Hhi). simpl.
    rewrite Hlo, Hhi. simpl.
    exact (IHb Hb).
  (* ESeq es *) - intros es IH H. simpl in *. apply IH. exact H.
  (* ELet v b *)
  - intros v b IHv IHb H. simpl in *.
    apply orb_false_iff in H. destruct H as [Hv Hb].
    rewrite (IHv Hv), (IHb Hb). reflexivity.
  (* EApp args *) - intros args IH H. simpl in *. apply IH. exact H.
  (* nil *) - intros _. reflexivity.
  (* cons e es *)
  - intros e es IHe IHes H. simpl in *.
    apply orb_false_iff in H. destruct H as [He Hes].
    rewrite (IHe He), (IHes Hes). reflexivity.
Qed.

(* ===== 15. Property 5: varying_if_flags_barriers ===== *)

Theorem varying_if_flags_barriers : forall cond then_ else_ ctx,
  is_varying cond = true ->
  barrier_free then_ = false ->
  check ctx (EIf cond then_ else_) <> [].
Proof.
  intros cond then_ else_ ctx Hv Hnbf Hempty.
  simpl in Hempty. rewrite Hv in Hempty. simpl in Hempty.
  (* Hempty : check ctx cond ++ check Diverged then_ ++ check Diverged else_ = [] *)
  apply app_eq_nil in Hempty. destruct Hempty as [_ H1].
  apply app_eq_nil in H1. destruct H1 as [H2 _].
  apply diverged_clean_iff_barrier_free in H2.
  congruence.
Qed.

(* ===== Summary ===== *)
(*
  Print Assumptions merge_dim_comm.
  Print Assumptions check_seq_hom.
  Print Assumptions diverged_clean_iff_barrier_free.
  Print Assumptions mode_monotone.
  Print Assumptions cdcf_check_agreement.
  Print Assumptions varying_if_flags_barriers.
  (* Expected: no axioms beyond the Rocq kernel for all six. *)
*)
