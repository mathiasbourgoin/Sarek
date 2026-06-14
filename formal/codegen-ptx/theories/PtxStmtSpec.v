(** PtxStmtSpec.v — PTX statement emission correctness specification.
 *
 * Defines a loop-free subset of the Sarek IR statement language as [ir_stmt],
 * its execution semantics [agpu_exec_ir], a mirror PTX statement AST
 * [ptx_stmt_ast] with execution [agpu_exec_ptx_stmt], the translation
 * [emit_ast_stmt], and proves:
 *
 *   Theorem emit_stmt_correct :
 *     forall s st,
 *       agpu_exec_ir st s = agpu_exec_ptx_stmt st (emit_ast_stmt s).
 *
 * Design notes:
 * - No [Admitted] is used anywhere in this file.
 * - [ir_stmt] covers: ISEmpty, ISSeq, ISLet, ISLetMut, ISAssign, ISIf,
 *   ISBarrier.  Loops are excluded; their termination argument requires fuel
 *   or well-founded recursion and is deferred to a future extension.
 * - Register assignment ([ISAssign name e]) models [SAssign (LVar _, e)].
 * - [reg_write] is the single state-update primitive.
 *)

From CodegenPtx Require Import AGpuSemantics.
From CodegenPtx Require Import PtxTypes.
From CodegenPtx Require Import PtxExprSpec.
From Stdlib Require Import Strings.String.
From Stdlib Require Import Arith.Arith.
From Stdlib Require Import Bool.Bool.

Open Scope string_scope.

(* ------------------------------------------------------------------ *)
(** * Register-write helper *)
(* ------------------------------------------------------------------ *)

Definition reg_write (name : string) (v : ptx_val) (st : agpu_state)
    : agpu_state :=
  {| regs := fun n => if String.eqb n name then Some v else st.(regs) n;
     tc   := st.(tc);
     mem  := st.(mem) |}.

(* ------------------------------------------------------------------ *)
(** * IR statement subset *)
(* ------------------------------------------------------------------ *)

Inductive ir_stmt :=
  | ISEmpty   : ir_stmt
  | ISSeq     : ir_stmt  -> ir_stmt -> ir_stmt
  | ISLet     : string   -> ir_expr -> ir_stmt -> ir_stmt
  | ISLetMut  : string   -> ir_expr -> ir_stmt -> ir_stmt
  | ISAssign  : string   -> ir_expr -> ir_stmt
  | ISIf      : ir_expr  -> ir_stmt -> ir_stmt -> ir_stmt
  | ISBarrier : ir_stmt.

(* ------------------------------------------------------------------ *)
(** * IR statement execution semantics *)
(* ------------------------------------------------------------------ *)

Fixpoint agpu_exec_ir (st : agpu_state) (s : ir_stmt)
    : option agpu_state :=
  match s with
  | ISEmpty => Some st
  | ISSeq s1 s2 =>
      match agpu_exec_ir st s1 with
      | None    => None
      | Some st1 => agpu_exec_ir st1 s2
      end
  | ISLet name e body | ISLetMut name e body =>
      match agpu_eval_ir st e with
      | None         => None
      | Some (v, st1) => agpu_exec_ir (reg_write name v st1) body
      end
  | ISAssign name e =>
      match agpu_eval_ir st e with
      | None         => None
      | Some (v, st1) => Some (reg_write name v st1)
      end
  | ISIf cond s1 s2 =>
      match agpu_eval_ir st cond with
      | None => None
      | Some (U32 0, st1) => agpu_exec_ir st1 s2
      | Some (U32 _, st1) => agpu_exec_ir st1 s1
      | Some (Pred false, st1) => agpu_exec_ir st1 s2
      | Some (Pred true,  st1) => agpu_exec_ir st1 s1
      | _ => None
      end
  | ISBarrier => Some st
  end.

(* ------------------------------------------------------------------ *)
(** * PTX statement AST *)
(* ------------------------------------------------------------------ *)

Inductive ptx_stmt_ast :=
  | PSEmpty   : ptx_stmt_ast
  | PSSeq     : ptx_stmt_ast -> ptx_stmt_ast -> ptx_stmt_ast
  | PSLet     : string -> ptx_expr_ast -> ptx_stmt_ast -> ptx_stmt_ast
  | PSLetMut  : string -> ptx_expr_ast -> ptx_stmt_ast -> ptx_stmt_ast
  | PSAssign  : string -> ptx_expr_ast -> ptx_stmt_ast
  | PSIf      : ptx_expr_ast -> ptx_stmt_ast -> ptx_stmt_ast -> ptx_stmt_ast
  | PSBarrier : ptx_stmt_ast.

(* ------------------------------------------------------------------ *)
(** * PTX statement execution semantics *)
(* ------------------------------------------------------------------ *)

Fixpoint agpu_exec_ptx_stmt (st : agpu_state) (s : ptx_stmt_ast)
    : option agpu_state :=
  match s with
  | PSEmpty => Some st
  | PSSeq s1 s2 =>
      match agpu_exec_ptx_stmt st s1 with
      | None    => None
      | Some st1 => agpu_exec_ptx_stmt st1 s2
      end
  | PSLet name e body | PSLetMut name e body =>
      match agpu_eval_ptx st e with
      | None         => None
      | Some (v, st1) => agpu_exec_ptx_stmt (reg_write name v st1) body
      end
  | PSAssign name e =>
      match agpu_eval_ptx st e with
      | None         => None
      | Some (v, st1) => Some (reg_write name v st1)
      end
  | PSIf cond s1 s2 =>
      match agpu_eval_ptx st cond with
      | None => None
      | Some (U32 0, st1) => agpu_exec_ptx_stmt st1 s2
      | Some (U32 _, st1) => agpu_exec_ptx_stmt st1 s1
      | Some (Pred false, st1) => agpu_exec_ptx_stmt st1 s2
      | Some (Pred true,  st1) => agpu_exec_ptx_stmt st1 s1
      | _ => None
      end
  | PSBarrier => Some st
  end.

(* ------------------------------------------------------------------ *)
(** * [emit_ast_stmt] *)
(* ------------------------------------------------------------------ *)

Fixpoint emit_ast_stmt (s : ir_stmt) : ptx_stmt_ast :=
  match s with
  | ISEmpty          => PSEmpty
  | ISSeq s1 s2      => PSSeq (emit_ast_stmt s1) (emit_ast_stmt s2)
  | ISLet n e body   => PSLet    n (emit_ast_expr e) (emit_ast_stmt body)
  | ISLetMut n e body => PSLetMut n (emit_ast_expr e) (emit_ast_stmt body)
  | ISAssign n e     => PSAssign n (emit_ast_expr e)
  | ISIf cond s1 s2  =>
      PSIf (emit_ast_expr cond) (emit_ast_stmt s1) (emit_ast_stmt s2)
  | ISBarrier        => PSBarrier
  end.

(** Full equality: [agpu_eval_ir] and [agpu_eval_ptx] agree for all expressions.
 *
 * We prove this directly by induction on [e], mirroring the [emit_ast_expr]
 * structure.  For each case we reduce both sides and show they are equal.
 * This is cleaner than separate Some/None lemmas.
 *)
Lemma eval_ir_ptx_eq :
  forall e st,
    agpu_eval_ir st e = agpu_eval_ptx st (emit_ast_expr e).
Proof.
  induction e; intros st.

  (* ---- IEConst ---- *)
  - destruct i; simpl; reflexivity.

  (* ---- IEVar ---- *)
  - simpl. reflexivity.

  (* ---- IEBinop ---- *)
  - (* Destruct is_cmp_op to expose the [if] in emit_ast_expr *)
    destruct (is_cmp_op i) eqn:Hcmp.
    + (* Comparison → PtxCmp *)
      assert (Hemit : emit_ast_expr (IEBinop i e1 e2) =
              PtxCmp (ir_binop_to_ptx_cmp i) (emit_ast_expr e1) (emit_ast_expr e2)).
      { simpl. rewrite Hcmp. reflexivity. }
      rewrite Hemit. simpl.
      rewrite <- IHe1.
      destruct (agpu_eval_ir st e1) as [[v1 st1]|]; simpl; [|reflexivity].
      rewrite <- IHe2.
      destruct (agpu_eval_ir st1 e2) as [[v2 st2]|]; simpl; [|reflexivity].
      destruct (agpu_eval_binop i v1 v2) as [vb|] eqn:Hbop.
      * rewrite (ptx_cmp_agrees i v1 v2 vb Hcmp Hbop). reflexivity.
      * destruct (agpu_eval_ptx_cmp (ir_binop_to_ptx_cmp i) v1 v2)
            as [vp|] eqn:Hpcmp; [|reflexivity].
        exfalso.
        destruct i; simpl in *; try discriminate Hcmp;
          destruct v1; destruct v2; simpl in *;
            try discriminate Hbop; try discriminate Hpcmp.
    + (* Arithmetic → PtxBinop *)
      assert (Hemit : emit_ast_expr (IEBinop i e1 e2) =
              PtxBinop (ir_binop_to_ptx_binop i) (emit_ast_expr e1) (emit_ast_expr e2)).
      { simpl. rewrite Hcmp. reflexivity. }
      rewrite Hemit. simpl.
      rewrite <- IHe1.
      destruct (agpu_eval_ir st e1) as [[v1 st1]|]; simpl; [|reflexivity].
      rewrite <- IHe2.
      destruct (agpu_eval_ir st1 e2) as [[v2 st2]|]; simpl; [|reflexivity].
      destruct (agpu_eval_binop i v1 v2) as [vb|] eqn:Hbop.
      * rewrite (ptx_binop_agrees i v1 v2 vb Hcmp Hbop). reflexivity.
      * destruct (agpu_eval_ptx_binop (ir_binop_to_ptx_binop i) v1 v2)
            as [vp|] eqn:Hpbop; [|reflexivity].
        exfalso.
        destruct i; simpl in *; try discriminate Hcmp;
          destruct v1; destruct v2; simpl in *;
            try discriminate Hbop; try discriminate Hpbop.

  (* ---- IEArrayRead — ONE case, destruct ir_memspace inside ---- *)
  - destruct i.
    + (* MS_Global *)
      simpl.
      rewrite <- IHe1.
      destruct (agpu_eval_ir st e1) as [[vb st1]|]; simpl; [|reflexivity].
      rewrite <- IHe2.
      destruct vb as [base|base|f1|f2|b1]; simpl.
      * (* U32 base *)
        destruct (agpu_eval_ir st1 e2) as [[vi st2]|]; simpl; [|reflexivity].
        destruct vi as [idx|idx'|f3|f4|b2]; simpl; reflexivity.
      * (* U64 base *)
        destruct (agpu_eval_ir st1 e2) as [[vi st2]|]; simpl; [|reflexivity].
        destruct vi as [idx'|idx|f3|f4|b2]; simpl; reflexivity.
      * (* F32 base: IR gives None; PTX: PAdd F32 v2 then global read = None *)
        destruct (agpu_eval_ir st1 e2) as [[vi st2]|]; simpl.
        -- destruct vi; simpl; reflexivity.
        -- reflexivity.
      * (* F64 base *)
        destruct (agpu_eval_ir st1 e2) as [[vi st2]|]; simpl.
        -- destruct vi; simpl; reflexivity.
        -- reflexivity.
      * (* Pred base *)
        destruct (agpu_eval_ir st1 e2) as [[vi st2]|]; simpl.
        -- destruct vi; simpl; reflexivity.
        -- reflexivity.
    + (* MS_Shared *)
      simpl.
      rewrite <- IHe1.
      destruct (agpu_eval_ir st e1) as [[vb st1]|]; simpl; [|reflexivity].
      rewrite <- IHe2.
      destruct vb as [base|base|f1|f2|b1]; simpl.
      * destruct (agpu_eval_ir st1 e2) as [[vi st2]|]; simpl; [|reflexivity].
        destruct vi as [idx|idx'|f3|f4|b2]; simpl; reflexivity.
      * destruct (agpu_eval_ir st1 e2) as [[vi st2]|]; simpl; [|reflexivity].
        destruct vi as [idx'|idx|f3|f4|b2]; simpl; reflexivity.
      * destruct (agpu_eval_ir st1 e2) as [[vi st2]|]; simpl.
        -- destruct vi; simpl; reflexivity.
        -- reflexivity.
      * destruct (agpu_eval_ir st1 e2) as [[vi st2]|]; simpl.
        -- destruct vi; simpl; reflexivity.
        -- reflexivity.
      * destruct (agpu_eval_ir st1 e2) as [[vi st2]|]; simpl.
        -- destruct vi; simpl; reflexivity.
        -- reflexivity.

  (* ---- Thread intrinsics ---- *)
  - simpl. reflexivity.
  - simpl. reflexivity.
  - simpl. reflexivity.

  (* ---- IEGlobalIdx ---- *)
  - simpl. reflexivity.

  (* ---- IEBarrier ---- *)
  - simpl. reflexivity.

  (* ---- IESin32 ---- *)
  - simpl. rewrite <- IHe.
    destruct (agpu_eval_ir st e) as [[v st']|]; [destruct v|]; simpl; reflexivity.

  (* ---- IECos32 ---- *)
  - simpl. rewrite <- IHe.
    destruct (agpu_eval_ir st e) as [[v st']|]; [destruct v|]; simpl; reflexivity.

  (* ---- IESqrt32 ---- *)
  - simpl. rewrite <- IHe.
    destruct (agpu_eval_ir st e) as [[v st']|]; [destruct v|]; simpl; reflexivity.

  (* ---- IEFabs32 ---- *)
  - simpl. rewrite <- IHe.
    destruct (agpu_eval_ir st e) as [[v st']|]; [destruct v|]; simpl; reflexivity.

  (* ---- IEFma32 ---- *)
  - simpl.
    rewrite <- IHe1.
    destruct (agpu_eval_ir st e1) as [[va st1]|]; [|simpl; reflexivity].
    destruct va; simpl; try reflexivity.
    rewrite <- IHe2.
    destruct (agpu_eval_ir st1 e2) as [[vb st2]|]; [|simpl; reflexivity].
    destruct vb; simpl; try reflexivity.
    rewrite <- IHe3.
    destruct (agpu_eval_ir st2 e3) as [[vc st3]|]; [|simpl; reflexivity].
    destruct vc; simpl; reflexivity.

  (* ---- IESin64 ---- *)
  - simpl. rewrite <- IHe.
    destruct (agpu_eval_ir st e) as [[v st']|]; [destruct v|]; simpl; reflexivity.

  (* ---- IECos64 ---- *)
  - simpl. rewrite <- IHe.
    destruct (agpu_eval_ir st e) as [[v st']|]; [destruct v|]; simpl; reflexivity.

  (* ---- IESqrt64 ---- *)
  - simpl. rewrite <- IHe.
    destruct (agpu_eval_ir st e) as [[v st']|]; [destruct v|]; simpl; reflexivity.

  (* ---- IEFabs64 ---- *)
  - simpl. rewrite <- IHe.
    destruct (agpu_eval_ir st e) as [[v st']|]; [destruct v|]; simpl; reflexivity.

  (* ---- IEFma64 ---- *)
  - simpl.
    rewrite <- IHe1.
    destruct (agpu_eval_ir st e1) as [[va st1]|]; [|simpl; reflexivity].
    destruct va; simpl; try reflexivity.
    rewrite <- IHe2.
    destruct (agpu_eval_ir st1 e2) as [[vb st2]|]; [|simpl; reflexivity].
    destruct vb; simpl; try reflexivity.
    rewrite <- IHe3.
    destruct (agpu_eval_ir st2 e3) as [[vc st3]|]; [|simpl; reflexivity].
    destruct vc; simpl; reflexivity.

Qed.

(* ------------------------------------------------------------------ *)
(** * Main correctness theorem *)
(* ------------------------------------------------------------------ *)

Theorem emit_stmt_correct :
  forall s st,
    agpu_exec_ir st s = agpu_exec_ptx_stmt st (emit_ast_stmt s).
Proof.
  induction s as [ | s1 IHs1 s2 IHs2
                  | nm e_let body_let IHs
                  | nm e_letm body_letm IHs
                  | nm e_asgn
                  | e_cond s1 IHs1 s2 IHs2
                  | ]; intros st.

  (* ---- ISEmpty ---- *)
  - simpl. reflexivity.

  (* ---- ISSeq ---- *)
  - simpl. rewrite IHs1.
    destruct (agpu_exec_ptx_stmt st (emit_ast_stmt s1)) as [st1|]; [|reflexivity].
    apply IHs2.

  (* ---- ISLet ---- *)
  - simpl.
    rewrite <- (eval_ir_ptx_eq e_let st).
    destruct (agpu_eval_ir st e_let) as [[v st1]|]; [|reflexivity].
    apply IHs.

  (* ---- ISLetMut ---- *)
  - simpl.
    rewrite <- (eval_ir_ptx_eq e_letm st).
    destruct (agpu_eval_ir st e_letm) as [[v st1]|]; [|reflexivity].
    apply IHs.

  (* ---- ISAssign ---- *)
  - simpl.
    rewrite <- (eval_ir_ptx_eq e_asgn st).
    reflexivity.

  (* ---- ISIf ---- *)
  - simpl.
    rewrite <- (eval_ir_ptx_eq e_cond st).
    destruct (agpu_eval_ir st e_cond) as [[vc st1]|]; [|reflexivity].
    destruct vc as [n|n|f|f|b].
    + destruct n.
      * apply IHs2.
      * apply IHs1.
    + reflexivity.
    + reflexivity.
    + reflexivity.
    + destruct b.
      * apply IHs1.
      * apply IHs2.

  (* ---- ISBarrier ---- *)
  - simpl. reflexivity.

Qed.
