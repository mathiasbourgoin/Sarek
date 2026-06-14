(** PtxKernelSpec.v — Top-level PTX kernel correctness specification.
 *
 * Defines an [ir_kernel] record (the Rocq mirror of the covered portion of
 * [Sarek_ir_types.kernel]), its execution semantics [agpu_exec_ir_kernel],
 * a PTX kernel AST [ptx_kernel_ast] with execution [agpu_exec_ptx_kernel],
 * the translation [emit_ast_kernel], and proves:
 *
 *   Theorem emit_kernel_correct :
 *     forall k st,
 *       agpu_exec_ir_kernel st k =
 *       agpu_exec_ptx_kernel st (emit_ast_kernel k).
 *
 * Uses [emit_stmt_correct] from [PtxStmtSpec] and [eval_ir_ptx_eq]
 * from [PtxStmtSpec] (which in turn uses [emit_expr_correct] from
 * [PtxExprSpec]).
 *
 * Design notes:
 * - No [Admitted] is used anywhere in this file.
 * - [ir_kernel] is a simplified record: kernel name (ignored in semantics),
 *   parameter names (list of register names to bind from the initial state),
 *   and body ([ir_stmt]).  This captures the essential structure of
 *   [Sarek_ir_types.kernel] that the PTX emitter formalizes.
 * - The proof is immediate from [emit_stmt_correct]: the kernel evaluator
 *   simply executes the body statement, so correctness lifts directly.
 *)

From CodegenPtx Require Import AGpuSemantics.
From CodegenPtx Require Import PtxTypes.
From CodegenPtx Require Import PtxExprSpec.
From CodegenPtx Require Import PtxStmtSpec.
From Stdlib Require Import Strings.String.
From Stdlib Require Import List.

Open Scope string_scope.

(* ------------------------------------------------------------------ *)
(** * IR kernel record
 *
 * Mirrors the portion of [Sarek_ir_types.kernel] that matters for PTX
 * code generation correctness.
 *)
(* ------------------------------------------------------------------ *)

Record ir_kernel := {
  kern_name   : string;        (** kernel name (for documentation only) *)
  kern_params : list string;   (** parameter register names             *)
  kern_body   : ir_stmt;       (** kernel body                          *)
}.

(* ------------------------------------------------------------------ *)
(** * PTX kernel AST *)
(* ------------------------------------------------------------------ *)

Record ptx_kernel_ast := {
  ptx_kern_name   : string;
  ptx_kern_params : list string;
  ptx_kern_body   : ptx_stmt_ast;
}.

(* ------------------------------------------------------------------ *)
(** * Kernel execution semantics
 *
 * Execute the kernel body in the given initial state.  Parameters are
 * already assumed to be bound in [st.(regs)] before the kernel runs
 * (the kernel launch mechanism is out of scope for this model).
 *)
(* ------------------------------------------------------------------ *)

Definition agpu_exec_ir_kernel (st : agpu_state) (k : ir_kernel)
    : option agpu_state :=
  agpu_exec_ir st k.(kern_body).

Definition agpu_exec_ptx_kernel (st : agpu_state) (k : ptx_kernel_ast)
    : option agpu_state :=
  agpu_exec_ptx_stmt st k.(ptx_kern_body).

(* ------------------------------------------------------------------ *)
(** * [emit_ast_kernel] — structural translation *)
(* ------------------------------------------------------------------ *)

Definition emit_ast_kernel (k : ir_kernel) : ptx_kernel_ast :=
  {| ptx_kern_name   := k.(kern_name);
     ptx_kern_params := k.(kern_params);
     ptx_kern_body   := emit_ast_stmt k.(kern_body) |}.

(* ------------------------------------------------------------------ *)
(** * Top-level correctness theorem
 *
 * [emit_kernel_correct]: the PTX kernel AST produced by [emit_ast_kernel]
 * is semantically equivalent to the source IR kernel under [agpu] semantics.
 *
 * Proof: unfold both kernel evaluators; the goal reduces to
 *   [agpu_exec_ir st k.(kern_body) =
 *    agpu_exec_ptx_stmt st (emit_ast_stmt k.(kern_body))]
 * which is exactly [emit_stmt_correct].
 *)
(* ------------------------------------------------------------------ *)

Theorem emit_kernel_correct :
  forall k st,
    agpu_exec_ir_kernel st k =
    agpu_exec_ptx_kernel st (emit_ast_kernel k).
Proof.
  intros k st.
  unfold agpu_exec_ir_kernel, agpu_exec_ptx_kernel, emit_ast_kernel.
  simpl.
  apply emit_stmt_correct.
Qed.

(* ------------------------------------------------------------------ *)
(** * Summary: theorems proved in this project
 *
 * 1. [emit_expr_correct] (PtxExprSpec.v):
 *      forall e st v st',
 *        agpu_eval_ir st e = Some (v, st') ->
 *        agpu_eval_ptx st (emit_ast_expr e) = Some (v, st').
 *
 * 2. [emit_stmt_correct] (PtxStmtSpec.v):
 *      forall s st,
 *        agpu_exec_ir st s = agpu_exec_ptx_stmt st (emit_ast_stmt s).
 *
 * 3. [emit_kernel_correct] (this file):
 *      forall k st,
 *        agpu_exec_ir_kernel st k =
 *        agpu_exec_ptx_kernel st (emit_ast_kernel k).
 *
 * Admits: 0.  All proofs are closed without [Admitted] or [admit].
 *)
(* ------------------------------------------------------------------ *)
