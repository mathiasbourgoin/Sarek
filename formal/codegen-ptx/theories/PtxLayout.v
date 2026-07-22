(** PtxLayout.v — aligned (C-ABI) aggregate byte-layout model for Sarek PTX
 *  codegen.
 *
 * Standalone model of the layout function implemented in OCaml by
 * [spoc/ir/Sarek_ir_layout.ml] (which itself mirrors the host PPX layout in
 * sarek/ppx/Sarek_ppx.ml). Campaign item L8 migrated both from PACKED to
 * ALIGNED: record fields are placed at the next offset satisfying their natural
 * alignment (padding inserted), the total size is rounded up to the struct's
 * maximum member alignment, and a variant is [[tag:int32@0][payload@P]] with
 * [P = max(4, max payload-member alignment)] and size rounded to the overall
 * alignment. This is the standard C struct-layout ABI, so it agrees
 * byte-for-byte with the [typedef struct {...}] the C-family backends emit.
 *
 * Design notes (FR-040):
 * - This theory deliberately does NOT import or extend [PtxTypes.elttype];
 *   it defines its own small scalar universe [lty] carrying only what layout
 *   needs (byte size / natural alignment).  Field names are [nat] indices.
 * - "No variant below top level" is encoded STRUCTURALLY: [lfield] (the type
 *   of record fields and constructor payload slots) has no variant
 *   constructor at all, and variants appear only in the top-level [elayout]
 *   dispatch — exactly like [Sarek_ir_layout.elttype_layout], which rejects
 *   [TVariant] anywhere below the root ([Nested_variant]).
 * - With aligned placement, natural alignment now holds BY CONSTRUCTION:
 *   [record_leaf_aligned] / [variant_leaf_aligned] are unconditional, and the
 *   OCaml [Misaligned_field] rejection is dead ([record_always_accepted]).
 *
 * What changed from the packed model (the load-bearing restatement):
 * - The old master invariant [chain] said leaves TILE the byte range with NO
 *   gaps ([off' = off + leaf_size]).  Aligned layout inserts padding, so that
 *   is false.  It is replaced by [sorted_packed] (leaves are ordered and
 *   non-overlapping, gaps permitted) plus [end_of] (the running end).  The old
 *   [record_size_correct] ("size = sum of leaf sizes") is likewise false with
 *   padding; it is restated as [record_size_is_padded_end] (size = the aligned,
 *   padded cumulative end).  The non-overlap / in-bounds / alignment theorems
 *   survive (alignment becomes EASIER — unconditional).
 *
 * Lemmas proved with 0 admits (FR-041):
 * - non-overlap:      [record_leaf_nonoverlap], [variant_ctor_leaf_nonoverlap],
 *                     [variant_tag_payload_disjoint]
 * - in-bounds:        [record_leaf_in_bounds], [variant_leaf_in_bounds]
 * - size (restated):  [record_size_is_padded_end], [ctor_payload_size_correct]
 * - alignment:        [record_leaf_aligned], [variant_leaf_aligned],
 *                     [record_always_accepted], [variant_always_accepted]
 * - tag = index:      [ctor_tag_is_index]
 *
 * Conformance with the OCaml layout function is established by CMBT in
 * test/test_layout_conformance.ml (FR-042), which hand-mirrors the definitions
 * below.
 *
 * No [Admitted] is used anywhere in this file.
 *)

From Stdlib Require Import List.
From Stdlib Require Import Arith.Arith.
From Stdlib Require Import Bool.Bool.
From Stdlib Require Import Lia.
Import ListNotations.

(* ===================================================================== *)
(** * 0. Alignment arithmetic *)
(* ===================================================================== *)

(** [align_up off a] rounds [off] up to the next multiple of [a] — the C-ABI
    padding rule (Sarek_ir_layout.align_up: [(off + a - 1) / a * a]). *)
Definition align_up (off a : nat) : nat := a * ((off + a - 1) / a).

Lemma align_up_ge : forall off a, 0 < a -> off <= align_up off a.
Proof.
  intros off a Ha. unfold align_up.
  pose proof (Nat.div_mod (off + a - 1) a (Nat.neq_sym _ _ (Nat.lt_neq _ _ Ha))) as Hdm.
  pose proof (Nat.mod_upper_bound (off + a - 1) a (Nat.neq_sym _ _ (Nat.lt_neq _ _ Ha))) as Hub.
  nia.
Qed.

Lemma align_up_divide : forall off a, Nat.divide a (align_up off a).
Proof. intros off a. unfold align_up. exists ((off + a - 1) / a). lia. Qed.

(** Key translation identity: shifting the layout origin by a multiple of [a]
    shifts every aligned offset by the same amount.  This is what makes a
    record's byte size independent of where the record is embedded, once the
    embedding offset respects the record's alignment. *)
Lemma align_up_add : forall d x a,
  0 < a -> Nat.divide a d -> align_up (d + x) a = d + align_up x a.
Proof.
  intros d x a Ha [m Hm]. subst d. unfold align_up.
  assert (Hne : a <> 0) by lia.
  replace (m * a + x + a - 1) with (m * a + (x + a - 1)) by lia.
  rewrite Nat.div_add_l by exact Hne.
  nia.
Qed.

(* ===================================================================== *)
(** * 1. Scalar universe: size and natural alignment *)
(* ===================================================================== *)

Inductive lty : Type :=
  | L32 : lty
  | L64 : lty.

Definition scalar_size (t : lty) : nat :=
  match t with
  | L32 => 4
  | L64 => 8
  end.

Definition scalar_align (t : lty) : nat := scalar_size t.

Lemma scalar_align_pos : forall t, 0 < scalar_align t.
Proof. destruct t; simpl; lia. Qed.

Lemma scalar_align_cases : forall t, scalar_align t = 4 \/ scalar_align t = 8.
Proof. destruct t; simpl; auto. Qed.

(* ===================================================================== *)
(** * 2. Field trees: leaves and nested records (no nested variants) *)
(* ===================================================================== *)

Inductive lfield : Type :=
  | LLeaf : lty -> lfield
  | LRec : lfields -> lfield
with lfields : Type :=
  | LNil : lfields
  | LCons : nat -> lfield -> lfields -> lfields.

Scheme lfield_mut := Induction for lfield Sort Prop
  with lfields_mut := Induction for lfields Sort Prop.
Combined Scheme lfield_lfields_ind from lfield_mut, lfields_mut.

(** Natural alignment of a field / field sequence (Sarek_ir_layout.elttype_align
    / record_align).  Empty sequence has alignment 1. *)
Fixpoint falign (f : lfield) : nat :=
  match f with
  | LLeaf t => scalar_align t
  | LRec fs => fsalign fs
  end
with fsalign (fs : lfields) : nat :=
  match fs with
  | LNil => 1
  | LCons _ f r => Nat.max (falign f) (fsalign r)
  end.

(** Aligned (padded) byte size of a field, and the running end offset of a field
    sequence laid out from [off] (Sarek_ir_layout: each field rounded up to its
    alignment; a record's size padded to its own alignment). *)
Fixpoint fsize (f : lfield) : nat :=
  match f with
  | LLeaf t => scalar_size t
  | LRec fs => align_up (fsend 0 fs) (fsalign fs)
  end
with fsend (off : nat) (fs : lfields) : nat :=
  match fs with
  | LNil => off
  | LCons _ f r => fsend (align_up off (falign f) + fsize f) r
  end.

(** Alignment values live in the divisibility chain {1,4,8}. *)
Lemma falign_cases :
  (forall f, falign f = 1 \/ falign f = 4 \/ falign f = 8)
  /\ (forall fs, fsalign fs = 1 \/ fsalign fs = 4 \/ fsalign fs = 8).
Proof.
  apply lfield_lfields_ind.
  - intros t. simpl. destruct (scalar_align_cases t); auto.
  - intros fs IH. simpl. exact IH.
  - simpl. auto.
  - intros n f IHf r IHr. simpl.
    destruct IHf as [E1|[E1|E1]]; destruct IHr as [E2|[E2|E2]];
      rewrite E1, E2; simpl; auto.
Qed.

Lemma falign_pos : forall f, 0 < falign f.
Proof. intros f. destruct (proj1 falign_cases f) as [E|[E|E]]; rewrite E; lia. Qed.

Lemma fsalign_pos : forall fs, 0 < fsalign fs.
Proof. intros fs. destruct (proj2 falign_cases fs) as [ -> | [ -> | -> ] ]; lia. Qed.

(** Within {1,4,8}, [a <= b] implies [a | b] (a divisibility chain). *)
Lemma div_148 : forall a b,
  (a = 1 \/ a = 4 \/ a = 8) -> (b = 1 \/ b = 4 \/ b = 8) ->
  a <= b -> Nat.divide a b.
Proof.
  intros a b Ha Hb Hle.
  destruct Ha as [ -> | [ -> | -> ] ]; destruct Hb as [ -> | [ -> | -> ] ]; try lia;
    [ exists 1 | exists 4 | exists 8 | exists 1 | exists 2 | exists 1 ]; lia.
Qed.

Lemma falign_div_max_l : forall f r,
  Nat.divide (falign f) (Nat.max (falign f) (fsalign r)).
Proof.
  intros f r. apply div_148.
  - apply (proj1 falign_cases).
  - destruct (proj1 falign_cases f) as [ -> | [ -> | -> ] ];
    destruct (proj2 falign_cases r) as [ -> | [ -> | -> ] ]; simpl; auto.
  - apply Nat.le_max_l.
Qed.

Lemma fsalign_div_max_r : forall f r,
  Nat.divide (fsalign r) (Nat.max (falign f) (fsalign r)).
Proof.
  intros f r. apply div_148.
  - apply (proj2 falign_cases).
  - destruct (proj1 falign_cases f) as [ -> | [ -> | -> ] ];
    destruct (proj2 falign_cases r) as [ -> | [ -> | -> ] ]; simpl; auto.
  - apply Nat.le_max_r.
Qed.

(** Translation: [fsend] shifts by any multiple [d] of the sequence's
    alignment. *)
Lemma fsend_shift : forall fs d off,
  Nat.divide (fsalign fs) d -> fsend (d + off) fs = d + fsend off fs.
Proof.
  induction fs as [| n f r IH]; intros d off Hd; simpl in *.
  - reflexivity.
  - (* d divisible by max(falign f, fsalign r) => by each *)
    assert (Hf : Nat.divide (falign f) d)
      by (eapply Nat.divide_trans; [apply falign_div_max_l | exact Hd]).
    assert (Hr : Nat.divide (fsalign r) d)
      by (eapply Nat.divide_trans; [apply fsalign_div_max_r | exact Hd]).
    rewrite (align_up_add d off (falign f) (falign_pos f) Hf).
    replace (d + align_up off (falign f) + fsize f)
      with (d + (align_up off (falign f) + fsize f)) by lia.
    rewrite (IH d (align_up off (falign f) + fsize f) Hr).
    reflexivity.
Qed.

Lemma fsend_translate : forall fs off,
  Nat.divide (fsalign fs) off -> fsend off fs = off + fsend 0 fs.
Proof.
  intros fs off Hdiv.
  rewrite <- (fsend_shift fs off 0 Hdiv). rewrite Nat.add_0_r. reflexivity.
Qed.

Lemma fsend_ge : forall fs off, off <= fsend off fs.
Proof.
  induction fs as [| n f r IH]; intros off; simpl.
  - lia.
  - specialize (IH (align_up off (falign f) + fsize f)).
    pose proof (align_up_ge off (falign f) (falign_pos f)). lia.
Qed.

(* ===================================================================== *)
(** * 3. Flattening into scalar leaves at aligned offsets *)
(* ===================================================================== *)

Record leaf : Type := mkLeaf {
  lf_path : list nat;
  lf_ty : lty;
  lf_off : nat
}.

Definition leaf_size (l : leaf) : nat := scalar_size (lf_ty l).

(** Flatten a field / field sequence at absolute byte [off]
    (Sarek_ir_layout.flatten_field / flatten_fields): each field rounded up to
    its natural alignment before placement. *)
Fixpoint flatten (p : list nat) (off : nat) (f : lfield) : list leaf :=
  match f with
  | LLeaf t => [ mkLeaf p t off ]
  | LRec fs => flattens p off fs
  end
with flattens (p : list nat) (off : nat) (fs : lfields) : list leaf :=
  match fs with
  | LNil => []
  | LCons n f r =>
      let o := align_up off (falign f) in
      flatten (p ++ [n]) o f ++ flattens p (o + fsize f) r
  end.

(* ===================================================================== *)
(** * 4. The ordered-with-gaps invariant of aligned flattening *)
(* ===================================================================== *)

(** [sorted_packed lo ls]: the leaves [ls] are in strictly forward byte order —
    the first sits at or after [lo], and each subsequent leaf starts at or after
    the previous one's end.  Unlike the packed [chain], gaps (alignment padding)
    are permitted. *)
Fixpoint sorted_packed (lo : nat) (ls : list leaf) : Prop :=
  match ls with
  | [] => True
  | l :: r => lo <= lf_off l /\ sorted_packed (lf_off l + leaf_size l) r
  end.

(** Running end offset of a leaf list started at [lo]. *)
Fixpoint end_of (lo : nat) (ls : list leaf) : nat :=
  match ls with
  | [] => lo
  | l :: r => end_of (lf_off l + leaf_size l) r
  end.

Lemma end_of_app : forall a b lo,
  end_of lo (a ++ b) = end_of (end_of lo a) b.
Proof.
  induction a as [| l a IH]; intros b lo; simpl; [reflexivity | apply IH].
Qed.

(** [end_of] ignores its origin argument once the list is non-empty. *)
Lemma end_of_cons_indep : forall l r lo lo',
  end_of lo (l :: r) = end_of lo' (l :: r).
Proof. intros. simpl. reflexivity. Qed.

Lemma sorted_weaken : forall ls lo lo',
  lo' <= lo -> sorted_packed lo ls -> sorted_packed lo' ls.
Proof.
  intros [| l r] lo lo' Hle Hs; simpl in *; [exact I |].
  destruct Hs as [Hhead Hrest]. split; [lia | exact Hrest].
Qed.

Lemma sorted_app : forall a b lo,
  sorted_packed lo a ->
  sorted_packed (end_of lo a) b ->
  sorted_packed lo (a ++ b).
Proof.
  induction a as [| l a IH]; intros b lo Ha Hb; simpl in *.
  - exact Hb.
  - destruct Ha as [Hl Ha]. split; [exact Hl |].
    apply IH; [exact Ha | exact Hb].
Qed.

Lemma sorted_lower : forall ls lo l,
  sorted_packed lo ls -> In l ls -> lo <= lf_off l.
Proof.
  induction ls as [| a ls IH]; intros lo l Hs Hin; simpl in *.
  - contradiction.
  - destruct Hs as [Ha Hs].
    destruct Hin as [-> | Hin]; [lia |].
    specialize (IH _ _ Hs Hin). lia.
Qed.

Lemma end_of_ge : forall ls lo, sorted_packed lo ls -> lo <= end_of lo ls.
Proof.
  induction ls as [| a ls IH]; intros lo Hs; simpl in *.
  - lia.
  - destruct Hs as [Ha Hs]. specialize (IH _ Hs). lia.
Qed.

Lemma in_le_end : forall ls lo l,
  sorted_packed lo ls -> In l ls -> lf_off l + leaf_size l <= end_of lo ls.
Proof.
  induction ls as [| a ls IH]; intros lo l Hs Hin; simpl in *.
  - contradiction.
  - destruct Hs as [Ha Hs].
    destruct Hin as [-> | Hin].
    + pose proof (end_of_ge ls (lf_off l + leaf_size l) Hs). lia.
    + apply (IH _ _ Hs Hin).
Qed.

Lemma sorted_nth_le : forall ls lo i j li lj,
  sorted_packed lo ls -> i < j ->
  nth_error ls i = Some li -> nth_error ls j = Some lj ->
  lf_off li + leaf_size li <= lf_off lj.
Proof.
  induction ls as [| a ls IH]; intros lo i j li lj Hs Hij Hi Hj.
  - destruct i; discriminate.
  - destruct Hs as [Ha Hs].
    destruct i as [| i]; simpl in Hi, Hj.
    + injection Hi as ->.
      destruct j as [| j]; [lia |]. simpl in Hj.
      assert (lf_off li + leaf_size li <= lf_off lj)
        by (eapply sorted_lower; [exact Hs | eapply nth_error_In; eauto]).
      lia.
    + destruct j as [| j]; [lia |]. simpl in Hj.
      eapply IH with (lo := lf_off a + leaf_size a) (i := i) (j := j); eauto.
      lia.
Qed.

(* ===================================================================== *)
(** * 5. Flattening produces sorted, in-bounds, aligned leaves *)
(* ===================================================================== *)

(** Master lemma (sortedness + end bound).  The [flatten] branch needs the
    origin to respect the field's own alignment (always true at call sites,
    where [flattens] rounds up first); the [flattens] branch needs nothing. *)
Lemma flatten_sorted_mut :
  (forall f p off,
     Nat.divide (falign f) off ->
     sorted_packed off (flatten p off f)
     /\ end_of off (flatten p off f) <= off + fsize f)
  /\ (forall fs p off,
     sorted_packed off (flattens p off fs)
     /\ end_of off (flattens p off fs) <= fsend off fs).
Proof.
  apply lfield_lfields_ind.
  - (* LLeaf *)
    intros t p off Hdiv. simpl. unfold leaf_size. simpl.
    split; [split; [lia | exact I] | lia].
  - (* LRec fs *)
    intros fs IH p off Hdiv. simpl.
    destruct (IH p off) as [Hs Hend].
    split; [exact Hs |].
    (* end_of off (flattens..) <= fsend off fs = off + fsend 0 fs
       and off + fsend 0 fs <= off + align_up (fsend 0 fs) (fsalign fs) *)
    simpl in Hdiv.
    rewrite (fsend_translate fs off Hdiv) in Hend.
    pose proof (align_up_ge (fsend 0 fs) (fsalign fs) (fsalign_pos fs)). lia.
  - (* LNil *)
    intros p off. simpl. split; [exact I | lia].
  - (* LCons n f r *)
    intros n f IHf r IHr p off. simpl.
    set (o := align_up off (falign f)).
    assert (Hoge : off <= o) by (apply align_up_ge; apply falign_pos).
    assert (Hodiv : Nat.divide (falign f) o) by apply align_up_divide.
    destruct (IHf (p ++ [n]) o Hodiv) as [HsA HendA].
    destruct (IHr p (o + fsize f)) as [HsB HendB].
    (* sortedness of the concatenation *)
    assert (Hsort : sorted_packed off
             (flatten (p ++ [n]) o f ++ flattens p (o + fsize f) r)).
    { apply sorted_app.
      - apply (sorted_weaken _ o off Hoge HsA).
      - (* end_of off A ... need sorted_packed (end_of off A) B *)
        (* end_of off A <= o + fsize f <= (o + fsize f); weaken B's origin *)
        assert (HAend : end_of off (flatten (p ++ [n]) o f) <= o + fsize f).
        { destruct (flatten (p ++ [n]) o f) eqn:EA.
          - simpl. pose proof (fsend_ge r (o + fsize f)). (* off <= o + fsize f *)
            pose proof (align_up_ge off (falign f) (falign_pos f)). lia.
          - rewrite (end_of_cons_indep l l0 off o). exact HendA. }
        apply (sorted_weaken _ (o + fsize f) (end_of off (flatten (p++[n]) o f)) HAend HsB). }
    split; [exact Hsort |].
    (* end bound of the concatenation *)
    rewrite end_of_app.
    (* end_of (end_of off A) B <= fsend (o + fsize f) r *)
    assert (HAend : end_of off (flatten (p ++ [n]) o f) <= o + fsize f).
    { destruct (flatten (p ++ [n]) o f) eqn:EA.
      - simpl. pose proof (align_up_ge off (falign f) (falign_pos f)). lia.
      - rewrite (end_of_cons_indep l l0 off o). exact HendA. }
    destruct (flattens p (o + fsize f) r) eqn:EB.
    + simpl. pose proof (fsend_ge r (o + fsize f)). lia.
    + rewrite (end_of_cons_indep l l0 (end_of off (flatten (p++[n]) o f)) (o + fsize f)).
      exact HendB.
Qed.

Lemma flattens_sorted : forall fs p off, sorted_packed off (flattens p off fs).
Proof. intros fs p off. apply (proj2 flatten_sorted_mut). Qed.

Lemma flattens_end : forall fs p off,
  end_of off (flattens p off fs) <= fsend off fs.
Proof. intros fs p off. apply (proj2 flatten_sorted_mut). Qed.

(** Alignment by construction: every scalar leaf lands on its natural boundary,
    with NO acceptance precondition (the aligned layout can never misalign). *)
Lemma flatten_aligned_mut :
  (forall f p off,
     Nat.divide (falign f) off ->
     forall l, In l (flatten p off f) ->
       Nat.divide (scalar_align (lf_ty l)) (lf_off l))
  /\ (forall fs p off,
     forall l, In l (flattens p off fs) ->
       Nat.divide (scalar_align (lf_ty l)) (lf_off l)).
Proof.
  apply lfield_lfields_ind.
  - (* LLeaf *)
    intros t p off Hdiv l Hin. simpl in Hin.
    destruct Hin as [<- | []]. simpl. exact Hdiv.
  - (* LRec fs *)
    intros fs IH p off Hdiv l Hin. simpl in Hin.
    eapply IH; eauto.
  - (* LNil *)
    intros p off l Hin. simpl in Hin. contradiction.
  - (* LCons n f r *)
    intros n f IHf r IHr p off l Hin. simpl in Hin.
    apply in_app_or in Hin. destruct Hin as [Hin | Hin].
    + eapply IHf; [ apply align_up_divide | exact Hin ].
    + eapply IHr; eauto.
Qed.

Lemma flattens_aligned : forall fs p off l,
  In l (flattens p off fs) ->
  Nat.divide (scalar_align (lf_ty l)) (lf_off l).
Proof. intros. eapply (proj2 flatten_aligned_mut); eauto. Qed.

(* ===================================================================== *)
(** * 6. Record layout *)
(* ===================================================================== *)

Definition record_leaves (fs : lfields) : list leaf := flattens [] 0 fs.

(** Aligned record size: the padded cumulative end (Sarek_ir_layout.record_layout
    total = [align_up (end offset) (struct alignment)]). *)
Definition record_size (fs : lfields) : nat :=
  align_up (fsend 0 fs) (fsalign fs).

Fixpoint field_offsets (off : nat) (fs : lfields) : list (nat * nat) :=
  match fs with
  | LNil => []
  | LCons n f r =>
      let o := align_up off (falign f) in
      (n, o) :: field_offsets (o + fsize f) r
  end.

Definition record_field_offsets (fs : lfields) : list (nat * nat) :=
  field_offsets 0 fs.

Lemma record_sorted : forall fs, sorted_packed 0 (record_leaves fs).
Proof. intros fs. apply flattens_sorted. Qed.

(** (restated size) The record's size is exactly the aligned, padded cumulative
    end of its fields (replaces the packed [record_size_correct], which claimed
    size = sum of leaf sizes — false once padding is inserted). *)
Theorem record_size_is_padded_end : forall fs,
  record_size fs = align_up (fsend 0 fs) (fsalign fs).
Proof. reflexivity. Qed.

(** (b) In-bounds: every scalar leaf's byte range fits inside the record. *)
Theorem record_leaf_in_bounds : forall fs l,
  In l (record_leaves fs) ->
  lf_off l + leaf_size l <= record_size fs.
Proof.
  intros fs l Hin.
  pose proof (in_le_end (record_leaves fs) 0 l (record_sorted fs) Hin) as H.
  pose proof (flattens_end fs [] 0) as Hend. unfold record_leaves in *.
  pose proof (align_up_ge (fsend 0 fs) (fsalign fs) (fsalign_pos fs)).
  unfold record_size. lia.
Qed.

(** (a) Field non-overlap: distinct leaves of a record occupy disjoint byte
    ranges [offset, offset + size). *)
Theorem record_leaf_nonoverlap : forall fs i j li lj,
  i <> j ->
  nth_error (record_leaves fs) i = Some li ->
  nth_error (record_leaves fs) j = Some lj ->
  lf_off li + leaf_size li <= lf_off lj \/
  lf_off lj + leaf_size lj <= lf_off li.
Proof.
  intros fs i j li lj Hij Hi Hj.
  destruct (Nat.lt_ge_cases i j) as [Hlt | Hge].
  - left. eapply sorted_nth_le; eauto. apply record_sorted.
  - right. assert (j < i) as Hlt by lia.
    eapply sorted_nth_le; eauto. apply record_sorted.
Qed.

(** (d) Alignment, unconditional: every scalar leaf's absolute offset is a
    multiple of its natural alignment — no acceptance predicate required. *)
Theorem record_leaf_aligned : forall fs l,
  In l (record_leaves fs) ->
  Nat.divide (scalar_align (lf_ty l)) (lf_off l).
Proof. intros fs l Hin. eapply flattens_aligned; eauto. Qed.

(* ===================================================================== *)
(** * 7. Variant layout: [[tag:int32@0][payload@P]] *)
(* ===================================================================== *)

Definition tag_offset : nat := 0.
Definition tag_size : nat := 4.

(** Positional numbering of constructor arguments (OCaml [_0], [_1], ...). *)
Fixpoint number_args (i : nat) (args : list lfield) : lfields :=
  match args with
  | [] => LNil
  | a :: r => LCons i a (number_args (S i) r)
  end.

(** Alignment of one constructor's payload (max of its args, min 1). *)
Definition ctor_align (args : list lfield) : nat := fsalign (number_args 0 args).

(** Payload region offset = [max(4, max payload-member alignment)] over all
    constructors (Sarek_ir_layout variant_layout). *)
Definition variant_payload_offset (ctors : list (list lfield)) : nat :=
  fold_right (fun c acc => Nat.max (ctor_align c) acc) 4 ctors.

(** Padded payload (C union member) size of one constructor. *)
Definition payload_struct_size (args : list lfield) : nat :=
  align_up (fsend 0 (number_args 0 args)) (fsalign (number_args 0 args)).

Record ctor_layout : Type := mkCtor {
  cl_tag : nat;
  cl_leaves : list leaf;
  cl_payload_size : nat
}.

(** Per-constructor layout: payload args placed from [payoff], leaf paths
    qualified by the constructor tag. *)
Fixpoint ctor_layouts (payoff tag : nat) (ctors : list (list lfield))
  : list ctor_layout :=
  match ctors with
  | [] => []
  | c :: r =>
      mkCtor tag (flattens [tag] payoff (number_args 0 c)) (payload_struct_size c)
        :: ctor_layouts payoff (S tag) r
  end.

Definition max_payload (cls : list ctor_layout) : nat :=
  fold_right (fun c acc => Nat.max (cl_payload_size c) acc) 0 cls.

(** Variant element size: [round_up(payload_offset + max payload, payload_align)]
    (Sarek_ir_layout variant_layout vl_size). *)
Definition variant_size (ctors : list (list lfield)) : nat :=
  let p := variant_payload_offset ctors in
  align_up (p + max_payload (ctor_layouts p 0 ctors)) p.

(** [variant_payload_offset] is at least 4 (so the tag fits) and lies in
    {4,8}. *)
Lemma variant_payload_offset_ge4 : forall ctors,
  4 <= variant_payload_offset ctors.
Proof.
  induction ctors as [| c r IH]; simpl; [lia |].
  pose proof (Nat.le_max_r (ctor_align c) (variant_payload_offset r)). lia.
Qed.

Lemma ctor_align_cases : forall args,
  ctor_align args = 1 \/ ctor_align args = 4 \/ ctor_align args = 8.
Proof. intros args. unfold ctor_align. apply (proj2 falign_cases). Qed.

Lemma variant_payload_offset_cases : forall ctors,
  variant_payload_offset ctors = 4 \/ variant_payload_offset ctors = 8.
Proof.
  induction ctors as [| c r IH]; simpl; [auto |].
  destruct (ctor_align_cases c) as [ -> | [ -> | -> ] ];
    destruct IH as [ -> | -> ]; simpl; auto.
Qed.

Lemma variant_payload_offset_cases3 : forall ctors,
  variant_payload_offset ctors = 1 \/ variant_payload_offset ctors = 4
  \/ variant_payload_offset ctors = 8.
Proof. intros ctors. destruct (variant_payload_offset_cases ctors) as [H|H]; auto. Qed.

Lemma variant_payload_offset_pos : forall ctors, 0 < variant_payload_offset ctors.
Proof. intros ctors. pose proof (variant_payload_offset_ge4 ctors). lia. Qed.

(** One-step unfolding of the payload offset over a cons. *)
Lemma variant_payload_offset_cons : forall a r,
  variant_payload_offset (a :: r)
  = Nat.max (ctor_align a) (variant_payload_offset r).
Proof. reflexivity. Qed.

(** The payload origin respects each constructor's alignment (needed so payload
    args land on their natural boundary). *)
Lemma ctor_align_div_payload_offset : forall ctors c,
  In c ctors -> Nat.divide (ctor_align c) (variant_payload_offset ctors).
Proof.
  induction ctors as [| a r IH]; intros c Hin.
  - inversion Hin.
  - simpl in Hin. destruct Hin as [-> | Hin].
    + apply div_148.
      * apply ctor_align_cases.
      * apply variant_payload_offset_cases3.
      * rewrite variant_payload_offset_cons. apply Nat.le_max_l.
    + eapply Nat.divide_trans; [ apply IH; exact Hin |].
      apply div_148.
      * apply variant_payload_offset_cases3.
      * apply variant_payload_offset_cases3.
      * rewrite variant_payload_offset_cons. apply Nat.le_max_r.
Qed.

(** Structural facts about [ctor_layouts]: membership yields the flattened
    payload at [payoff] with the recorded (padded) payload size. *)
Lemma ctor_layouts_spec : forall ctors payoff t c,
  In c (ctor_layouts payoff t ctors) ->
  exists args, In args ctors /\
    cl_leaves c = flattens [cl_tag c] payoff (number_args 0 args) /\
    cl_payload_size c = payload_struct_size args.
Proof.
  induction ctors as [| a ctors IH]; intros payoff t c Hin; simpl in Hin.
  - contradiction.
  - destruct Hin as [<- | Hin].
    + simpl. exists a. auto.
    + destruct (IH payoff (S t) c Hin) as [args [Hin' [Hl Hs]]].
      exists args. split; [right; exact Hin' | auto].
Qed.

Lemma max_payload_ub : forall cls c,
  In c cls -> cl_payload_size c <= max_payload cls.
Proof.
  induction cls as [| a cls IH]; intros c Hin; simpl in *.
  - contradiction.
  - destruct Hin as [<- | Hin]; [apply Nat.le_max_l |].
    specialize (IH c Hin). pose proof (Nat.le_max_r (cl_payload_size a) (max_payload cls)). lia.
Qed.

(** Tag correctness: the tag stored for the i-th declared constructor is
    exactly i (host rule: tag = declaration index). *)
Theorem ctor_tag_is_index : forall ctors p i c,
  nth_error (ctor_layouts p 0 ctors) i = Some c ->
  cl_tag c = i.
Proof.
  assert (forall ctors p t i c,
            nth_error (ctor_layouts p t ctors) i = Some c ->
            cl_tag c = t + i) as H.
  { induction ctors as [| a ctors IH]; intros p t i c Hi.
    - destruct i; discriminate.
    - destruct i as [| i]; simpl in Hi.
      + injection Hi as <-. simpl. lia.
      + specialize (IH p (S t) i c Hi). lia. }
  intros ctors p i c Hi. now apply (H ctors p 0 i c) in Hi.
Qed.

(** (c) Per-constructor size correctness: the recorded payload size bounds the
    constructor's leaves (they fit inside the padded payload struct). *)
Theorem ctor_payload_size_correct : forall ctors c l,
  In c (ctor_layouts (variant_payload_offset ctors) 0 ctors) ->
  In l (cl_leaves c) ->
  lf_off l + leaf_size l <= variant_payload_offset ctors + cl_payload_size c.
Proof.
  intros ctors c l Hc Hl.
  destruct (ctor_layouts_spec ctors _ 0 c Hc) as [args [Hargs [Hleaves Hsize]]].
  rewrite Hleaves in Hl.
  set (p := variant_payload_offset ctors) in *.
  set (fs := number_args 0 args) in *.
  pose proof (in_le_end (flattens [cl_tag c] p fs) p l
                (flattens_sorted fs [cl_tag c] p) Hl) as Hb.
  pose proof (flattens_end fs [cl_tag c] p) as Hend.
  (* p divisible by fsalign fs => fsend p fs = p + fsend 0 fs *)
  assert (Hdiv : Nat.divide (fsalign fs) p).
  { unfold p, fs. change (fsalign (number_args 0 args)) with (ctor_align args).
    apply ctor_align_div_payload_offset; exact Hargs. }
  rewrite (fsend_translate fs p Hdiv) in Hend.
  rewrite Hsize. unfold payload_struct_size. fold fs.
  pose proof (align_up_ge (fsend 0 fs) (fsalign fs) (fsalign_pos fs)). lia.
Qed.

(** (a) Tag/payload disjointness: the int32 tag occupies [0,4) and every payload
    leaf starts at or after byte 4. *)
Theorem variant_tag_payload_disjoint : forall ctors c l,
  In c (ctor_layouts (variant_payload_offset ctors) 0 ctors) ->
  In l (cl_leaves c) ->
  tag_offset + tag_size <= lf_off l.
Proof.
  intros ctors c l Hc Hl.
  destruct (ctor_layouts_spec ctors _ 0 c Hc) as [args [Hargs [Hleaves _]]].
  rewrite Hleaves in Hl.
  set (p := variant_payload_offset ctors) in *.
  pose proof (sorted_lower _ p l (flattens_sorted (number_args 0 args) [cl_tag c] p) Hl).
  pose proof (variant_payload_offset_ge4 ctors).
  unfold tag_offset, tag_size, p in *. lia.
Qed.

(** (a) Non-overlap within a constructor. *)
Theorem variant_ctor_leaf_nonoverlap : forall ctors c i j li lj,
  In c (ctor_layouts (variant_payload_offset ctors) 0 ctors) ->
  i <> j ->
  nth_error (cl_leaves c) i = Some li ->
  nth_error (cl_leaves c) j = Some lj ->
  lf_off li + leaf_size li <= lf_off lj \/
  lf_off lj + leaf_size lj <= lf_off li.
Proof.
  intros ctors c i j li lj Hc Hij Hi Hj.
  destruct (ctor_layouts_spec ctors _ 0 c Hc) as [args [_ [Hleaves _]]].
  assert (Hsorted : sorted_packed (variant_payload_offset ctors) (cl_leaves c)).
  { rewrite Hleaves. apply flattens_sorted. }
  destruct (Nat.lt_ge_cases i j) as [Hlt | Hge].
  - left. eapply sorted_nth_le; eauto.
  - right. assert (j < i) as Hlt by lia. eapply sorted_nth_le; eauto.
Qed.

(** (b) In-bounds: every payload leaf of every constructor fits inside the
    variant element size. *)
Theorem variant_leaf_in_bounds : forall ctors c l,
  In c (ctor_layouts (variant_payload_offset ctors) 0 ctors) ->
  In l (cl_leaves c) ->
  lf_off l + leaf_size l <= variant_size ctors.
Proof.
  intros ctors c l Hc Hl.
  pose proof (ctor_payload_size_correct ctors c l Hc Hl) as Hb.
  pose proof (max_payload_ub _ _ Hc) as Hmax.
  unfold variant_size.
  set (p := variant_payload_offset ctors) in *.
  pose proof (align_up_ge (p + max_payload (ctor_layouts p 0 ctors)) p
                (variant_payload_offset_pos ctors)). lia.
Qed.

(** (d) Alignment, unconditional: every payload leaf is naturally aligned. *)
Theorem variant_leaf_aligned : forall ctors c l,
  In c (ctor_layouts (variant_payload_offset ctors) 0 ctors) ->
  In l (cl_leaves c) ->
  Nat.divide (scalar_align (lf_ty l)) (lf_off l).
Proof.
  intros ctors c l Hc Hl.
  destruct (ctor_layouts_spec ctors _ 0 c Hc) as [args [_ [Hleaves _]]].
  rewrite Hleaves in Hl. eapply flattens_aligned; eauto.
Qed.

(* ===================================================================== *)
(** * 8. The acceptance predicate is now trivially true *)
(* ===================================================================== *)

(** Natural-alignment acceptance (mirrors the OCaml boolean check).  With the
    aligned layout it is a tautology — the [Misaligned_field] rejection is
    dead. *)
Definition leaf_aligned (l : leaf) : bool :=
  Nat.eqb (Nat.modulo (lf_off l) (scalar_align (lf_ty l))) 0.

Definition record_accepted (fs : lfields) : bool :=
  forallb leaf_aligned (record_leaves fs).

Definition variant_accepted (ctors : list (list lfield)) : bool :=
  forallb (fun c => forallb leaf_aligned (cl_leaves c))
    (ctor_layouts (variant_payload_offset ctors) 0 ctors).

Lemma divide_mod0 : forall a b, 0 < a -> Nat.divide a b -> Nat.modulo b a = 0.
Proof.
  intros a b Ha Hd. apply Nat.Lcm0.mod_divide. exact Hd.
Qed.

Theorem record_always_accepted : forall fs, record_accepted fs = true.
Proof.
  intros fs. unfold record_accepted. apply forallb_forall.
  intros l Hin. unfold leaf_aligned. apply Nat.eqb_eq.
  apply divide_mod0; [apply scalar_align_pos |].
  eapply record_leaf_aligned; eauto.
Qed.

Theorem variant_always_accepted : forall ctors, variant_accepted ctors = true.
Proof.
  intros ctors. unfold variant_accepted. apply forallb_forall.
  intros c Hc. apply forallb_forall. intros l Hl.
  unfold leaf_aligned. apply Nat.eqb_eq.
  apply divide_mod0; [apply scalar_align_pos |].
  eapply variant_leaf_aligned; eauto.
Qed.

(* ===================================================================== *)
(** * 9. Assumption audit *)
(* ===================================================================== *)

(* Each theorem below must report "Closed under the global context". *)
Print Assumptions record_size_is_padded_end.
Print Assumptions record_leaf_in_bounds.
Print Assumptions record_leaf_nonoverlap.
Print Assumptions record_leaf_aligned.
Print Assumptions record_always_accepted.
Print Assumptions ctor_tag_is_index.
Print Assumptions variant_tag_payload_disjoint.
Print Assumptions variant_ctor_leaf_nonoverlap.
Print Assumptions ctor_payload_size_correct.
Print Assumptions variant_leaf_in_bounds.
Print Assumptions variant_leaf_aligned.
Print Assumptions variant_always_accepted.
