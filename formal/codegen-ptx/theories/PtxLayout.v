(** PtxLayout.v — packed aggregate byte-layout model for Sarek PTX codegen.
 *
 * Standalone model of the layout function implemented in OCaml by
 * [spoc/ir/Sarek_ir_layout.ml] (which itself mirrors the host PPX layout in
 * sarek/ppx/Sarek_ppx.ml: packed cumulative record offsets with no padding,
 * variants as [[tag:int32@0][payload@4]] with size [4 + max payload] and
 * tag = constructor declaration index).
 *
 * Design notes (FR-040):
 * - This theory deliberately does NOT import or extend [PtxTypes.elttype];
 *   it defines its own small scalar universe [lty] carrying only what layout
 *   needs (byte size / natural alignment).  Field names are [nat] indices.
 * - "No variant below top level" is encoded STRUCTURALLY: [lfield] (the type
 *   of record fields and constructor payload slots) has no variant
 *   constructor at all, and variants appear only in the top-level [elayout]
 *   dispatch — exactly like [Sarek_ir_layout.elttype_layout], which rejects
 *   [TVariant] anywhere below the root ([Nested_variant]).  TVec/TArray
 *   fields likewise have no counterpart here (OCaml: [Unsupported_field]).
 * - [record_accepted]/[variant_accepted] are boolean checkers mirroring the
 *   OCaml natural-alignment validation: a layout is accepted iff every
 *   scalar leaf's absolute packed offset is a multiple of its natural
 *   alignment.
 *
 * Lemmas proved with 0 admits (FR-041):
 * - non-overlap:      [record_leaf_nonoverlap], [variant_ctor_leaf_nonoverlap],
 *                     [variant_tag_payload_disjoint]
 * - in-bounds:        [record_leaf_in_bounds], [variant_leaf_in_bounds]
 * - size correctness: [record_size_correct], [ctor_payload_size_correct]
 * - alignment:        [record_accepted_aligned], [variant_accepted_aligned]
 * - tag = index:      [ctor_tag_is_index]
 *
 * Boundary note (recorded non-goal): the full emitter re-proof over
 * aggregates (relating emitted PTX ld/st offsets to this model) is a
 * follow-up task; conformance with the OCaml layout function is established
 * by CMBT in test/test_layout_conformance.ml (FR-042), which hand-mirrors
 * the definitions below.
 *
 * No [Admitted] is used anywhere in this file.
 *)

From Stdlib Require Import List.
From Stdlib Require Import Arith.Arith.
From Stdlib Require Import Bool.Bool.
From Stdlib Require Import Lia.
Import ListNotations.

(* ===================================================================== *)
(** * 1. Scalar universe: size and natural alignment *)
(* ===================================================================== *)

(** Layout only distinguishes 4-byte scalars (int32/float32/bool/unit on the
    OCaml side) from 8-byte scalars (int64/float64). *)
Inductive lty : Type :=
  | L32 : lty
  | L64 : lty.

Definition scalar_size (t : lty) : nat :=
  match t with
  | L32 => 4
  | L64 => 8
  end.

(** Natural alignment equals size for both supported scalar widths
    (Sarek_ir_layout.scalar_align). *)
Definition scalar_align (t : lty) : nat := scalar_size t.

Lemma scalar_size_pos : forall t, 0 < scalar_size t.
Proof. destruct t; simpl; lia. Qed.

(* ===================================================================== *)
(** * 2. Field trees: leaves and nested records (no nested variants) *)
(* ===================================================================== *)

(** A record field is either a scalar leaf or a nested record.  There is no
    variant case: variants below top level are structurally unrepresentable,
    mirroring the OCaml [Nested_variant] rejection. *)
Inductive lfield : Type :=
  | LLeaf : lty -> lfield
  | LRec : lfields -> lfield
with lfields : Type :=
  | LNil : lfields
  | LCons : nat -> lfield -> lfields -> lfields.

Scheme lfield_mut := Induction for lfield Sort Prop
  with lfields_mut := Induction for lfields Sort Prop.
Combined Scheme lfield_lfields_ind from lfield_mut, lfields_mut.

(** Packed byte size of a field / of a packed field sequence
    (Sarek_ir_layout.packed_size / cumulative [calc_offsets] rule:
    no padding, sizes just add up). *)
Fixpoint fsize (f : lfield) : nat :=
  match f with
  | LLeaf t => scalar_size t
  | LRec fs => fssize fs
  end
with fssize (fs : lfields) : nat :=
  match fs with
  | LNil => 0
  | LCons _ f r => fsize f + fssize r
  end.

(* ===================================================================== *)
(** * 3. Flattening into scalar leaves at absolute offsets *)
(* ===================================================================== *)

(** One scalar leaf of a flattened aggregate (Sarek_ir_layout.leaf).
    [lf_path] is the index path from the aggregate root (the OCaml side uses
    dotted string paths; layout does not depend on the naming scheme). *)
Record leaf : Type := mkLeaf {
  lf_path : list nat;
  lf_ty : lty;
  lf_off : nat
}.

Definition leaf_size (l : leaf) : nat := scalar_size (lf_ty l).

(** Flatten a field rooted at absolute byte [off]
    (Sarek_ir_layout.flatten_field / flatten_fields): leaves in declaration
    order, packed cumulative offsets, no padding. *)
Fixpoint flatten (p : list nat) (off : nat) (f : lfield) : list leaf :=
  match f with
  | LLeaf t => [ mkLeaf p t off ]
  | LRec fs => flattens p off fs
  end
with flattens (p : list nat) (off : nat) (fs : lfields) : list leaf :=
  match fs with
  | LNil => []
  | LCons n f r =>
      flatten (p ++ [n]) off f ++ flattens p (off + fsize f) r
  end.

Definition leaves_size (ls : list leaf) : nat :=
  fold_right (fun l acc => leaf_size l + acc) 0 ls.

(* ===================================================================== *)
(** * 4. Record layout *)
(* ===================================================================== *)

(** Packed record layout (Sarek_ir_layout.record_layout): leaves flattened
    from offset 0; total size is the padding-free sum of field sizes. *)
Definition record_leaves (fs : lfields) : list leaf := flattens [] 0 fs.
Definition record_size (fs : lfields) : nat := fssize fs.

(** Immediate-field offset table (Sarek_ir_layout.rl_fields / host
    [calc_offsets]): cumulative packed sums. *)
Fixpoint field_offsets (off : nat) (fs : lfields) : list (nat * nat) :=
  match fs with
  | LNil => []
  | LCons n f r => (n, off) :: field_offsets (off + fsize f) r
  end.

Definition record_field_offsets (fs : lfields) : list (nat * nat) :=
  field_offsets 0 fs.

(* ===================================================================== *)
(** * 5. Variant layout: [[tag:int32@0][payload@4]] *)
(* ===================================================================== *)

Definition tag_offset : nat := 0.
Definition tag_size : nat := 4.
Definition payload_offset : nat := 4.

(** Layout of one constructor's payload (Sarek_ir_layout.ctor_layout).
    [cl_tag] is the constructor declaration index. *)
Record ctor_layout : Type := mkCtor {
  cl_tag : nat;
  cl_leaves : list leaf;
  cl_payload_size : nat
}.

(** Positional numbering of constructor arguments (OCaml [_0], [_1], ...). *)
Fixpoint number_args (i : nat) (args : list lfield) : lfields :=
  match args with
  | [] => LNil
  | a :: r => LCons i a (number_args (S i) r)
  end.

(** Per-constructor layout: payload args packed from [payload_offset],
    leaf paths qualified by the constructor tag (OCaml: constructor-name
    prefix). *)
Definition ctor_layout_of (tag : nat) (args : list lfield) : ctor_layout :=
  let fs := number_args 0 args in
  mkCtor tag (flattens [tag] payload_offset fs) (fssize fs).

Fixpoint ctor_layouts (tag : nat) (ctors : list (list lfield))
  : list ctor_layout :=
  match ctors with
  | [] => []
  | c :: r => ctor_layout_of tag c :: ctor_layouts (S tag) r
  end.

Definition max_payload (cls : list ctor_layout) : nat :=
  fold_right (fun c acc => Nat.max (cl_payload_size c) acc) 0 cls.

(** Variant element size (host rule, Sarek_ppx.ml:750-755):
    4-byte int32 tag + max payload over all constructors. *)
Definition variant_size (ctors : list (list lfield)) : nat :=
  tag_size + max_payload (ctor_layouts 0 ctors).

(* ===================================================================== *)
(** * 6. Top-level dispatch and the acceptance predicate *)
(* ===================================================================== *)

(** Top-level element layouts (Sarek_ir_layout.elttype_layout): variants
    occur ONLY here, never inside [lfield]. *)
Inductive elayout : Type :=
  | ELScalar : lty -> elayout
  | ELRecord : lfields -> elayout
  | ELVariant : list (list lfield) -> elayout.

Definition elayout_size (e : elayout) : nat :=
  match e with
  | ELScalar t => scalar_size t
  | ELRecord fs => record_size fs
  | ELVariant cs => variant_size cs
  end.

(** Natural-alignment acceptance, mirroring the OCaml misalignment check in
    [Sarek_ir_layout.flatten_field]: every scalar leaf's absolute offset must
    be a multiple of its natural alignment. *)
Definition leaf_aligned (l : leaf) : bool :=
  Nat.eqb (Nat.modulo (lf_off l) (scalar_align (lf_ty l))) 0.

Definition record_accepted (fs : lfields) : bool :=
  forallb leaf_aligned (record_leaves fs).

Definition variant_accepted (ctors : list (list lfield)) : bool :=
  forallb (fun c => forallb leaf_aligned (cl_leaves c)) (ctor_layouts 0 ctors).

Definition accepted (e : elayout) : bool :=
  match e with
  | ELScalar _ => true
  | ELRecord fs => record_accepted fs
  | ELVariant cs => variant_accepted cs
  end.

(* ===================================================================== *)
(** * 7. The packed-chain invariant of flattening *)
(* ===================================================================== *)

(** [chain off ls]: the leaves [ls] tile the byte range starting at [off]
    consecutively — each leaf sits exactly at the running offset, which then
    advances by the leaf's size.  This is the master invariant from which
    non-overlap, in-bounds and size correctness all follow. *)
Fixpoint chain (off : nat) (ls : list leaf) : Prop :=
  match ls with
  | [] => True
  | l :: r => lf_off l = off /\ chain (off + leaf_size l) r
  end.

Lemma leaves_size_app : forall a b,
  leaves_size (a ++ b) = leaves_size a + leaves_size b.
Proof.
  induction a as [| l a IH]; intros b; simpl; [reflexivity |].
  rewrite IH. lia.
Qed.

Lemma chain_app : forall a off b,
  chain off a ->
  chain (off + leaves_size a) b ->
  chain off (a ++ b).
Proof.
  induction a as [| l a IH]; intros off b Ha Hb; simpl in *.
  - now rewrite Nat.add_0_r in Hb.
  - destruct Ha as [Hl Ha]. split; [exact Hl |].
    apply IH; [exact Ha |].
    now rewrite <- Nat.add_assoc.
Qed.

(** Master lemma: flattening any field (sequence) rooted at [off] yields a
    packed chain starting at [off] whose total leaf size is exactly the
    packed size of the field (sequence) — i.e. packed layout has no padding
    and no gaps. *)
Lemma flatten_chain_size :
  (forall f p off,
     chain off (flatten p off f) /\
     leaves_size (flatten p off f) = fsize f)
  /\
  (forall fs p off,
     chain off (flattens p off fs) /\
     leaves_size (flattens p off fs) = fssize fs).
Proof.
  apply lfield_lfields_ind.
  - (* LLeaf *)
    intros t p off; simpl. repeat split; try exact I.
    unfold leaf_size; simpl; lia.
  - (* LRec *)
    intros fs IH p off; simpl. apply IH.
  - (* LNil *)
    intros p off; simpl. repeat split; try exact I; lia.
  - (* LCons *)
    intros n f IHf r IHr p off; simpl.
    destruct (IHf (p ++ [n]) off) as [Hcf Hsf].
    destruct (IHr p (off + fsize f)) as [Hcr Hsr].
    split.
    + apply chain_app; [exact Hcf |]. now rewrite Hsf.
    + rewrite leaves_size_app, Hsf, Hsr. reflexivity.
Qed.

Lemma flattens_chain : forall fs p off, chain off (flattens p off fs).
Proof. intros fs p off. apply (proj2 flatten_chain_size). Qed.

Lemma flattens_leaves_size : forall fs p off,
  leaves_size (flattens p off fs) = fssize fs.
Proof. intros fs p off. apply (proj2 flatten_chain_size). Qed.

(* ===================================================================== *)
(** * 8. Consequences of the chain invariant *)
(* ===================================================================== *)

Lemma chain_in_lower : forall ls off l,
  chain off ls -> In l ls -> off <= lf_off l.
Proof.
  induction ls as [| a ls IH]; intros off l Hc Hin; simpl in *.
  - contradiction.
  - destruct Hc as [Ha Hc].
    destruct Hin as [-> | Hin]; [lia |].
    specialize (IH _ _ Hc Hin). lia.
Qed.

Lemma chain_in_bounds : forall ls off l,
  chain off ls -> In l ls ->
  lf_off l + leaf_size l <= off + leaves_size ls.
Proof.
  induction ls as [| a ls IH]; intros off l Hc Hin; simpl in *.
  - contradiction.
  - destruct Hc as [Ha Hc].
    destruct Hin as [-> | Hin]; [lia |].
    specialize (IH _ _ Hc Hin). lia.
Qed.

Lemma chain_nth_lower : forall ls off j lj,
  chain off ls -> nth_error ls j = Some lj -> off <= lf_off lj.
Proof.
  intros ls off j lj Hc Hj.
  apply (chain_in_lower ls off lj Hc).
  eapply nth_error_In; eauto.
Qed.

(** Ordered non-overlap: in a packed chain, an earlier leaf's byte range
    ends no later than a later leaf's range begins. *)
Lemma chain_nonoverlap : forall ls off i j li lj,
  chain off ls -> i < j ->
  nth_error ls i = Some li -> nth_error ls j = Some lj ->
  lf_off li + leaf_size li <= lf_off lj.
Proof.
  induction ls as [| a ls IH]; intros off i j li lj Hc Hij Hi Hj.
  - destruct i; discriminate.
  - destruct Hc as [Ha Hc].
    destruct i as [| i]; simpl in Hi, Hj.
    + injection Hi as ->.
      destruct j as [| j]; [lia |]. simpl in Hj.
      assert (off + leaf_size li <= lf_off lj)
        by (eapply chain_nth_lower; eauto).
      lia.
    + destruct j as [| j]; [lia |]. simpl in Hj.
      eapply IH with (off := off + leaf_size a) (i := i) (j := j); eauto.
      lia.
Qed.

(* ===================================================================== *)
(** * 9. Record layout theorems *)
(* ===================================================================== *)

(** (c) Size correctness: the packed record size is exactly the sum of its
    scalar leaf sizes — packed implies no padding. *)
Theorem record_size_correct : forall fs,
  leaves_size (record_leaves fs) = record_size fs.
Proof. intros fs. apply flattens_leaves_size. Qed.

(** (b) In-bounds: every scalar leaf's byte range fits inside the record. *)
Theorem record_leaf_in_bounds : forall fs l,
  In l (record_leaves fs) ->
  lf_off l + leaf_size l <= record_size fs.
Proof.
  intros fs l Hin.
  pose proof (chain_in_bounds (record_leaves fs) 0 l
                (flattens_chain fs [] 0) Hin) as H.
  rewrite record_size_correct in H. lia.
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
  - left. eapply chain_nonoverlap; eauto. apply flattens_chain.
  - right. assert (j < i) as Hlt by lia.
    eapply chain_nonoverlap; eauto. apply flattens_chain.
Qed.

(** (d) Alignment: in an accepted record layout, every scalar leaf's
    absolute offset is a multiple of its natural alignment. *)
Theorem record_accepted_aligned : forall fs l,
  record_accepted fs = true ->
  In l (record_leaves fs) ->
  Nat.modulo (lf_off l) (scalar_align (lf_ty l)) = 0.
Proof.
  intros fs l Hacc Hin.
  unfold record_accepted in Hacc.
  rewrite forallb_forall in Hacc.
  specialize (Hacc l Hin).
  unfold leaf_aligned in Hacc.
  now apply Nat.eqb_eq in Hacc.
Qed.

(* ===================================================================== *)
(** * 10. Variant layout theorems *)
(* ===================================================================== *)

Lemma ctor_layouts_ok : forall ctors t c,
  In c (ctor_layouts t ctors) ->
  chain payload_offset (cl_leaves c) /\
  leaves_size (cl_leaves c) = cl_payload_size c.
Proof.
  induction ctors as [| a ctors IH]; intros t c Hin; simpl in Hin.
  - contradiction.
  - destruct Hin as [<- | Hin].
    + simpl. split; [apply flattens_chain | apply flattens_leaves_size].
    + eapply IH; eauto.
Qed.

Lemma max_payload_ub : forall cls c,
  In c cls -> cl_payload_size c <= max_payload cls.
Proof.
  induction cls as [| a cls IH]; intros c Hin; simpl in *.
  - contradiction.
  - destruct Hin as [<- | Hin]; [lia |].
    specialize (IH c Hin). lia.
Qed.

(** Tag correctness: the tag stored for the i-th declared constructor is
    exactly i (host rule: tag = declaration index). *)
Theorem ctor_tag_is_index : forall ctors i c,
  nth_error (ctor_layouts 0 ctors) i = Some c ->
  cl_tag c = i.
Proof.
  assert (forall ctors t i c,
            nth_error (ctor_layouts t ctors) i = Some c ->
            cl_tag c = t + i) as H.
  { induction ctors as [| a ctors IH]; intros t i c Hi.
    - destruct i; discriminate.
    - destruct i as [| i]; simpl in Hi.
      + injection Hi as <-. simpl. lia.
      + specialize (IH (S t) i c Hi). lia. }
  intros ctors i c Hi. now apply (H ctors 0 i c) in Hi.
Qed.

(** (a) Tag/payload disjointness: the int32 tag occupies [0, 4) and every
    payload leaf starts at or after byte 4. *)
Theorem variant_tag_payload_disjoint : forall ctors c l,
  In c (ctor_layouts 0 ctors) ->
  In l (cl_leaves c) ->
  tag_offset + tag_size <= lf_off l.
Proof.
  intros ctors c l Hc Hl.
  destruct (ctor_layouts_ok ctors 0 c Hc) as [Hchain _].
  pose proof (chain_in_lower _ _ _ Hchain Hl).
  unfold tag_offset, tag_size, payload_offset in *. lia.
Qed.

(** (a) Non-overlap within a constructor: distinct payload leaves of the
    same constructor occupy disjoint byte ranges. *)
Theorem variant_ctor_leaf_nonoverlap : forall ctors c i j li lj,
  In c (ctor_layouts 0 ctors) ->
  i <> j ->
  nth_error (cl_leaves c) i = Some li ->
  nth_error (cl_leaves c) j = Some lj ->
  lf_off li + leaf_size li <= lf_off lj \/
  lf_off lj + leaf_size lj <= lf_off li.
Proof.
  intros ctors c i j li lj Hc Hij Hi Hj.
  destruct (ctor_layouts_ok ctors 0 c Hc) as [Hchain _].
  destruct (Nat.lt_ge_cases i j) as [Hlt | Hge].
  - left. eapply chain_nonoverlap; eauto.
  - right. assert (j < i) as Hlt by lia.
    eapply chain_nonoverlap; eauto.
Qed.

(** (c) Per-constructor size correctness: the recorded payload size is the
    sum of the constructor's leaf sizes (packed, no padding). *)
Theorem ctor_payload_size_correct : forall ctors c,
  In c (ctor_layouts 0 ctors) ->
  leaves_size (cl_leaves c) = cl_payload_size c.
Proof. intros ctors c Hc. apply (ctor_layouts_ok ctors 0 c Hc). Qed.

(** (b) In-bounds: every payload leaf of every constructor fits inside the
    variant element size [4 + max payload]. *)
Theorem variant_leaf_in_bounds : forall ctors c l,
  In c (ctor_layouts 0 ctors) ->
  In l (cl_leaves c) ->
  lf_off l + leaf_size l <= variant_size ctors.
Proof.
  intros ctors c l Hc Hl.
  destruct (ctor_layouts_ok ctors 0 c Hc) as [Hchain Hsize].
  pose proof (chain_in_bounds _ _ _ Hchain Hl) as Hb.
  pose proof (max_payload_ub _ _ Hc) as Hmax.
  unfold variant_size, tag_size, payload_offset in *. lia.
Qed.

(** (d) Alignment: in an accepted variant layout, every payload leaf's
    absolute offset is a multiple of its natural alignment.  (The tag is a
    4-byte scalar at offset 0, trivially aligned.) *)
Theorem variant_accepted_aligned : forall ctors c l,
  variant_accepted ctors = true ->
  In c (ctor_layouts 0 ctors) ->
  In l (cl_leaves c) ->
  Nat.modulo (lf_off l) (scalar_align (lf_ty l)) = 0.
Proof.
  intros ctors c l Hacc Hc Hl.
  unfold variant_accepted in Hacc.
  rewrite forallb_forall in Hacc.
  specialize (Hacc c Hc).
  rewrite forallb_forall in Hacc.
  specialize (Hacc l Hl).
  unfold leaf_aligned in Hacc.
  now apply Nat.eqb_eq in Hacc.
Qed.

(* ===================================================================== *)
(** * 11. Assumption audit *)
(* ===================================================================== *)

(* Each theorem below must report "Closed under the global context". *)
Print Assumptions record_size_correct.
Print Assumptions record_leaf_in_bounds.
Print Assumptions record_leaf_nonoverlap.
Print Assumptions record_accepted_aligned.
Print Assumptions ctor_tag_is_index.
Print Assumptions variant_tag_payload_disjoint.
Print Assumptions variant_ctor_leaf_nonoverlap.
Print Assumptions ctor_payload_size_correct.
Print Assumptions variant_leaf_in_bounds.
Print Assumptions variant_accepted_aligned.
