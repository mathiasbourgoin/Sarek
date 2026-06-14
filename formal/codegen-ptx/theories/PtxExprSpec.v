(** PtxExprSpec.v — PTX expression emission correctness specification.
 *
 * Defines [emit_ast_expr : ir_expr -> ptx_expr_ast], a structural translation
 * of the covered Sarek IR expression subset to the PTX AST defined in
 * [PtxTypes.v], and proves:
 *
 *   Theorem emit_expr_correct :
 *     forall e st v st',
 *       agpu_eval_ir st e = Some (v, st') ->
 *       agpu_eval_ptx st (emit_ast_expr e) = Some (v, st').
 *
 * Design notes:
 * - No [Admitted] is used anywhere in this file.
 * - [emit_ast_expr] is a total structural map from [ir_expr] to [ptx_expr_ast].
 * - Binary operators: arithmetic → [PtxBinop], comparison → [PtxCmp].
 * - FMA uses [PtxFma32]/[PtxFma64] which call the same [Parameter]s.
 * - [IEGlobalIdx] → [PtxBinop PAdd (PtxBinop PMul PtxBidx PtxBdim) PtxTidx].
 * - [IEBarrier] → [PtxLitU32 0] (both evaluate to [Some (U32 0, st)]).
 * - [IEArrayRead ms base_e idx_e] → [PtxGlobalRead/PtxSharedRead
 *   (PtxBinop PAdd (emit_ast_expr base_e) (emit_ast_expr idx_e))].
 *)

From CodegenPtx Require Import AGpuSemantics.
From CodegenPtx Require Import PtxTypes.
From Stdlib Require Import Strings.String.
From Stdlib Require Import Arith.Arith.
From Stdlib Require Import Bool.Bool.
From Stdlib Require Import Floats.

Open Scope string_scope.

(* ------------------------------------------------------------------ *)
(** * Binary-op classifier *)
(* ------------------------------------------------------------------ *)

Definition is_cmp_op (op : ir_binop) : bool :=
  match op with
  | Eq | Ne | Lt | Le | Gt | Ge => true
  | _ => false
  end.

Definition ir_binop_to_ptx_binop (op : ir_binop) : ptx_binop_tag :=
  match op with
  | Add    => PAdd
  | Sub    => PSub
  | Mul    => PMul
  | Div    => PDiv
  | Mod    => PMod
  | And    => PAnd
  | Or     => POr
  | Shl    => PShl
  | Shr    => PShr
  | BitAnd => PBitAnd
  | BitOr  => PBitOr
  | BitXor => PBitXor
  | _      => PAdd   (* unreachable when is_cmp_op op = false *)
  end.

Definition ir_binop_to_ptx_cmp (op : ir_binop) : ptx_cmp_tag :=
  match op with
  | Eq => PEq
  | Ne => PNe
  | Lt => PLt
  | Le => PLe
  | Gt => PGt
  | Ge => PGe
  | _  => PEq  (* unreachable when is_cmp_op op = true *)
  end.

(* ------------------------------------------------------------------ *)
(** * [emit_ast_expr] *)
(* ------------------------------------------------------------------ *)

Fixpoint emit_ast_expr (e : ir_expr) : ptx_expr_ast :=
  match e with
  | IEConst (CInt32  n)  => PtxLitU32 n
  | IEConst (CInt64  n)  => PtxLitU64 n
  | IEConst (CFloat32 f) => PtxLitF32 f
  | IEConst (CFloat64 f) => PtxLitF64 f
  | IEConst (CBool b)    => PtxLitU32 (bool_to_nat b)
  | IEConst CUnit        => PtxLitU32 0
  | IEVar name           => PtxReg name
  | IEBinop op e1 e2     =>
      if is_cmp_op op
      then PtxCmp (ir_binop_to_ptx_cmp op) (emit_ast_expr e1) (emit_ast_expr e2)
      else PtxBinop (ir_binop_to_ptx_binop op) (emit_ast_expr e1) (emit_ast_expr e2)
  | IEArrayRead MS_Global base_e idx_e =>
      PtxGlobalRead (PtxBinop PAdd (emit_ast_expr base_e) (emit_ast_expr idx_e))
  | IEArrayRead MS_Shared base_e idx_e =>
      PtxSharedRead (PtxBinop PAdd (emit_ast_expr base_e) (emit_ast_expr idx_e))
  | IEThreadIdxX  => PtxTidx
  | IEBlockIdxX   => PtxBidx
  | IEBlockDimX   => PtxBdim
  | IEGlobalIdx   =>
      PtxBinop PAdd (PtxBinop PMul PtxBidx PtxBdim) PtxTidx
  | IEBarrier     => PtxLitU32 0
  | IESin32 e1    => PtxIntrinsic PISin32  (emit_ast_expr e1)
  | IECos32 e1    => PtxIntrinsic PICos32  (emit_ast_expr e1)
  | IESqrt32 e1   => PtxIntrinsic PISqrt32 (emit_ast_expr e1)
  | IEFabs32 e1   => PtxIntrinsic PIFabs32 (emit_ast_expr e1)
  | IEFma32 ea eb ec =>
      PtxFma32 (emit_ast_expr ea) (emit_ast_expr eb) (emit_ast_expr ec)
  | IESin64 e1    => PtxIntrinsic PISin64  (emit_ast_expr e1)
  | IECos64 e1    => PtxIntrinsic PICos64  (emit_ast_expr e1)
  | IESqrt64 e1   => PtxIntrinsic PISqrt64 (emit_ast_expr e1)
  | IEFabs64 e1   => PtxIntrinsic PIFabs64 (emit_ast_expr e1)
  | IEFma64 ea eb ec =>
      PtxFma64 (emit_ast_expr ea) (emit_ast_expr eb) (emit_ast_expr ec)
  end.

(* ------------------------------------------------------------------ *)
(** * Auxiliary lemmas: binop/cmp agreement *)
(* ------------------------------------------------------------------ *)

Lemma ptx_binop_agrees :
  forall op v1 v2 v,
    is_cmp_op op = false ->
    agpu_eval_binop op v1 v2 = Some v ->
    agpu_eval_ptx_binop (ir_binop_to_ptx_binop op) v1 v2 = Some v.
Proof.
  intros op v1 v2 v Hnc Heval.
  destruct op; simpl in *; try discriminate Hnc;
    destruct v1; destruct v2; simpl in *; try discriminate Heval; exact Heval.
Qed.

Lemma ptx_cmp_agrees :
  forall op v1 v2 v,
    is_cmp_op op = true ->
    agpu_eval_binop op v1 v2 = Some v ->
    agpu_eval_ptx_cmp (ir_binop_to_ptx_cmp op) v1 v2 = Some v.
Proof.
  intros op v1 v2 v Hc Heval.
  destruct op; simpl in *; try discriminate Hc;
    destruct v1; destruct v2; simpl in *; try discriminate Heval; exact Heval.
Qed.

(* ------------------------------------------------------------------ *)
(** * Auxiliary lemma: single-step option pair extraction
 *
 * Unpacks [Some (a, b) = Some (c, d)] into [a = c] and [b = d].
 *)
(* ------------------------------------------------------------------ *)

Lemma some_pair_eq :
  forall {A B} (a c : A) (b d : B),
    Some (a, b) = Some (c, d) -> a = c /\ b = d.
Proof.
  intros A B a c b d H. injection H as H1 H2. split; assumption.
Qed.

(* ------------------------------------------------------------------ *)
(** * Main correctness theorem *)
(* ------------------------------------------------------------------ *)

(** Tactic: from [H : Some (a, b) = Some (v, st')], extract [v = a] and
    [st' = b] and substitute. *)
Ltac invert_some_pair H :=
  apply some_pair_eq in H;
  let Hv := fresh "Hv" in
  let Hst := fresh "Hst" in
  destruct H as [Hv Hst]; subst.

Theorem emit_expr_correct :
  forall e st v st',
    agpu_eval_ir st e = Some (v, st') ->
    agpu_eval_ptx st (emit_ast_expr e) = Some (v, st').
Proof.
  induction e; intros st v st' Heval.

  (* ---- IEConst ---- *)
  - destruct i; simpl in *; exact Heval.

  (* ---- IEVar ---- *)
  - simpl in *. exact Heval.

  (* ---- IEBinop ---- *)
  - simpl in Heval.
    destruct (agpu_eval_ir st e1) as [[v1 st1]|] eqn:He1; [|discriminate Heval].
    destruct (agpu_eval_ir st1 e2) as [[v2 st2]|] eqn:He2; [|discriminate Heval].
    destruct (agpu_eval_binop i v1 v2) as [vb|] eqn:Hbop; [|discriminate Heval].
    apply some_pair_eq in Heval. destruct Heval as [Hv Hst]. subst v st'.
    apply IHe1 in He1. apply IHe2 in He2.
    simpl.
    destruct (is_cmp_op i) eqn:Hcmp.
    + simpl. rewrite He1. rewrite He2.
      rewrite (ptx_cmp_agrees i v1 v2 vb Hcmp Hbop). reflexivity.
    + simpl. rewrite He1. rewrite He2.
      rewrite (ptx_binop_agrees i v1 v2 vb Hcmp Hbop). reflexivity.

  (* ---- IEArrayRead (handles both MS_Global and MS_Shared in ONE case) ---- *)
  - (* [i : ir_memspace]; IHe1, IHe2 : induction hypotheses for e1, e2.
     * Both base and index must be the same type (U32+U32 or U64+U64).
     * Other combinations are excluded by [agpu_eval_ir] returning None.
     *)
    destruct i.

    (* MS_Global *)
    + simpl in Heval.
      destruct (agpu_eval_ir st e1) as [[vb st1]|] eqn:He1; [|discriminate Heval].
      apply IHe1 in He1.
      destruct vb as [base|base|f1|f2|b1]; try discriminate Heval.
      * (* U32 base: index must be U32 *)
        destruct (agpu_eval_ir st1 e2) as [[vi st2]|] eqn:He2; [|discriminate Heval].
        apply IHe2 in He2.
        destruct vi as [idx|idx'|f3|f4|b2]; try discriminate Heval.
        apply some_pair_eq in Heval; destruct Heval as [Hv Hst]; subst.
        simpl. rewrite He1. simpl. rewrite He2. simpl. reflexivity.
      * (* U64 base: index must be U64 *)
        destruct (agpu_eval_ir st1 e2) as [[vi st2]|] eqn:He2; [|discriminate Heval].
        apply IHe2 in He2.
        destruct vi as [idx'|idx|f3|f4|b2]; try discriminate Heval.
        apply some_pair_eq in Heval; destruct Heval as [Hv Hst]; subst.
        simpl. rewrite He1. simpl. rewrite He2. simpl. reflexivity.

    (* MS_Shared *)
    + simpl in Heval.
      destruct (agpu_eval_ir st e1) as [[vb st1]|] eqn:He1; [|discriminate Heval].
      apply IHe1 in He1.
      destruct vb as [base|base|f1|f2|b1]; try discriminate Heval.
      * (* U32 base *)
        destruct (agpu_eval_ir st1 e2) as [[vi st2]|] eqn:He2; [|discriminate Heval].
        apply IHe2 in He2.
        destruct vi as [idx|idx'|f3|f4|b2]; try discriminate Heval.
        apply some_pair_eq in Heval; destruct Heval as [Hv Hst]; subst.
        simpl. rewrite He1. simpl. rewrite He2. simpl. reflexivity.
      * (* U64 base *)
        destruct (agpu_eval_ir st1 e2) as [[vi st2]|] eqn:He2; [|discriminate Heval].
        apply IHe2 in He2.
        destruct vi as [idx'|idx|f3|f4|b2]; try discriminate Heval.
        apply some_pair_eq in Heval; destruct Heval as [Hv Hst]; subst.
        simpl. rewrite He1. simpl. rewrite He2. simpl. reflexivity.

  (* ---- IEThreadIdxX ---- *)
  - simpl in *. exact Heval.

  (* ---- IEBlockIdxX ---- *)
  - simpl in *. exact Heval.

  (* ---- IEBlockDimX ---- *)
  - simpl in *. exact Heval.

  (* ---- IEGlobalIdx ---- *)
  - simpl in Heval.
    apply some_pair_eq in Heval. destruct Heval as [Hv Hst]. subst.
    simpl. reflexivity.

  (* ---- IEBarrier ---- *)
  - simpl in Heval.
    apply some_pair_eq in Heval. destruct Heval as [Hv Hst]. subst.
    simpl. reflexivity.

  (* ---- IESin32 ---- *)
  - simpl in Heval.
    destruct (agpu_eval_ir st e) as [[v1 st1]|] eqn:He; [|discriminate Heval].
    destruct v1; try discriminate Heval.
    apply some_pair_eq in Heval. destruct Heval as [Hv Hst]. subst.
    apply IHe in He.
    simpl. rewrite He. simpl. reflexivity.

  (* ---- IECos32 ---- *)
  - simpl in Heval.
    destruct (agpu_eval_ir st e) as [[v1 st1]|] eqn:He; [|discriminate Heval].
    destruct v1; try discriminate Heval.
    apply some_pair_eq in Heval. destruct Heval as [Hv Hst]. subst.
    apply IHe in He.
    simpl. rewrite He. simpl. reflexivity.

  (* ---- IESqrt32 ---- *)
  - simpl in Heval.
    destruct (agpu_eval_ir st e) as [[v1 st1]|] eqn:He; [|discriminate Heval].
    destruct v1; try discriminate Heval.
    apply some_pair_eq in Heval. destruct Heval as [Hv Hst]. subst.
    apply IHe in He.
    simpl. rewrite He. simpl. reflexivity.

  (* ---- IEFabs32 ---- *)
  - simpl in Heval.
    destruct (agpu_eval_ir st e) as [[v1 st1]|] eqn:He; [|discriminate Heval].
    destruct v1; try discriminate Heval.
    apply some_pair_eq in Heval. destruct Heval as [Hv Hst]. subst.
    apply IHe in He.
    simpl. rewrite He. simpl. reflexivity.

  (* ---- IEFma32 ---- *)
  - simpl in Heval.
    destruct (agpu_eval_ir st e1) as [[va st1]|] eqn:Hea; [|discriminate Heval].
    destruct va; try discriminate Heval.
    destruct (agpu_eval_ir st1 e2) as [[vb st2]|] eqn:Heb; [|discriminate Heval].
    destruct vb; try discriminate Heval.
    destruct (agpu_eval_ir st2 e3) as [[vc st3]|] eqn:Hec; [|discriminate Heval].
    destruct vc; try discriminate Heval.
    apply some_pair_eq in Heval. destruct Heval as [Hv Hst]. subst.
    apply IHe1 in Hea. apply IHe2 in Heb. apply IHe3 in Hec.
    simpl. rewrite Hea. rewrite Heb. rewrite Hec. simpl. reflexivity.

  (* ---- IESin64 ---- *)
  - simpl in Heval.
    destruct (agpu_eval_ir st e) as [[v1 st1]|] eqn:He; [|discriminate Heval].
    destruct v1; try discriminate Heval.
    apply some_pair_eq in Heval. destruct Heval as [Hv Hst]. subst.
    apply IHe in He.
    simpl. rewrite He. simpl. reflexivity.

  (* ---- IECos64 ---- *)
  - simpl in Heval.
    destruct (agpu_eval_ir st e) as [[v1 st1]|] eqn:He; [|discriminate Heval].
    destruct v1; try discriminate Heval.
    apply some_pair_eq in Heval. destruct Heval as [Hv Hst]. subst.
    apply IHe in He.
    simpl. rewrite He. simpl. reflexivity.

  (* ---- IESqrt64 ---- *)
  - simpl in Heval.
    destruct (agpu_eval_ir st e) as [[v1 st1]|] eqn:He; [|discriminate Heval].
    destruct v1; try discriminate Heval.
    apply some_pair_eq in Heval. destruct Heval as [Hv Hst]. subst.
    apply IHe in He.
    simpl. rewrite He. simpl. reflexivity.

  (* ---- IEFabs64 ---- *)
  - simpl in Heval.
    destruct (agpu_eval_ir st e) as [[v1 st1]|] eqn:He; [|discriminate Heval].
    destruct v1; try discriminate Heval.
    apply some_pair_eq in Heval. destruct Heval as [Hv Hst]. subst.
    apply IHe in He.
    simpl. rewrite He. simpl. reflexivity.

  (* ---- IEFma64 ---- *)
  - simpl in Heval.
    destruct (agpu_eval_ir st e1) as [[va st1]|] eqn:Hea; [|discriminate Heval].
    destruct va; try discriminate Heval.
    destruct (agpu_eval_ir st1 e2) as [[vb st2]|] eqn:Heb; [|discriminate Heval].
    destruct vb; try discriminate Heval.
    destruct (agpu_eval_ir st2 e3) as [[vc st3]|] eqn:Hec; [|discriminate Heval].
    destruct vc; try discriminate Heval.
    apply some_pair_eq in Heval. destruct Heval as [Hv Hst]. subst.
    apply IHe1 in Hea. apply IHe2 in Heb. apply IHe3 in Hec.
    simpl. rewrite Hea. rewrite Heb. rewrite Hec. simpl. reflexivity.

Qed.
