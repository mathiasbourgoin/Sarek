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
 * - [ir_kernel] is a simplified record: kernel name, parameter names, shared
 *   memory declarations (carried through translation; the abstract [agpu_state]
 *   already has a [shared_mem] field so no additional setup is needed), and body.
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
Open Scope list_scope.

(* ------------------------------------------------------------------ *)
(** * Shared memory declaration record
 *
 * Mirrors the [DShared] IR constructor: a name and a static element count.
 * The element type is not modelled here because the abstract semantics in
 * [AGpuSemantics] use a uniform [ptx_val] domain (no stride information is
 * needed for the kernel-level correctness theorem).
 *)
(* ------------------------------------------------------------------ *)

Record ir_shared_decl := {
  shdecl_name : string;
  shdecl_size : nat;
}.

(* ------------------------------------------------------------------ *)
(** * IR kernel record
 *
 * Mirrors the portion of [Sarek_ir_types.kernel] that matters for PTX
 * code generation correctness.
 *)
(* ------------------------------------------------------------------ *)

Record ir_kernel := {
  kern_name   : string;
  kern_params : list string;
  kern_shared : list ir_shared_decl;
  kern_body   : ir_stmt;
}.

(* ------------------------------------------------------------------ *)
(** * PTX kernel AST *)
(* ------------------------------------------------------------------ *)

Record ptx_kernel_ast := {
  ptx_kern_name   : string;
  ptx_kern_params : list string;
  ptx_kern_shared : list ir_shared_decl;
  ptx_kern_body   : ptx_stmt_ast;
}.

(* ------------------------------------------------------------------ *)
(** * Kernel execution semantics
 *
 * Execute the kernel body in the given initial state.  Parameters are
 * already assumed to be bound in [st.(regs)] before the kernel runs.
 * [kern_shared] / [ptx_kern_shared] are carried through but not
 * interpreted: the abstract [agpu_state] already models shared memory
 * as a flat word-addressed [shared_mem] array.
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
     ptx_kern_shared := k.(kern_shared);
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
 *
 * Note: [kern_shared] / [ptx_kern_shared] do not appear in either
 * semantics function, so the proof is unaffected by the field extension.
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
(** * Non-vacuousness witnesses (Rule 11)
 *
 * Three concrete Examples proving the theorem is non-vacuous:
 * (A) kernel with no shared declarations;
 * (B) kernel with one DShared-derived declaration (the theorem still holds);
 * (C) [emit_ast_kernel] faithfully copies [kern_shared] into
 *     [ptx_kern_shared] — the field is not dropped.
 *)
(* ------------------------------------------------------------------ *)

(** Concrete initial state: empty register file, zero thread constants,
    zero-initialised global and shared memory. *)
Definition ex_empty_regs : string -> option ptx_val :=
  fun _ => None.

Definition ex_zero_tc : thread_const :=
  {| tidx := 0; bidx := 0; bdim := 0 |}.

Definition ex_zero_mem : agpu_mem :=
  {| global_mem := fun _ => U32 0;
     shared_mem := fun _ => U32 0 |}.

Definition ex_st : agpu_state :=
  {| regs := ex_empty_regs; tc := ex_zero_tc; mem := ex_zero_mem |}.

(** Witness A: kernel with no shared declarations, empty body. *)
Definition ex_k_no_shared : ir_kernel :=
  {| kern_name := "no_shared";
     kern_params := nil;
     kern_shared := nil;
     kern_body := ISEmpty |}.

Example example_kernel_no_shared :
  agpu_exec_ir_kernel ex_st ex_k_no_shared =
  agpu_exec_ptx_kernel ex_st (emit_ast_kernel ex_k_no_shared).
Proof. apply emit_kernel_correct. Qed.

(** Witness B: kernel with one DShared-derived declaration.
    [kern_shared] is carried through; the empty body executes correctly. *)
Definition ex_shared_decl : ir_shared_decl :=
  {| shdecl_name := "shmem"; shdecl_size := 256 |}.

Definition ex_k_with_shared : ir_kernel :=
  {| kern_name := "with_shared";
     kern_params := nil;
     kern_shared := ex_shared_decl :: nil;
     kern_body := ISEmpty |}.

Example example_kernel_with_shared :
  agpu_exec_ir_kernel ex_st ex_k_with_shared =
  agpu_exec_ptx_kernel ex_st (emit_ast_kernel ex_k_with_shared).
Proof. apply emit_kernel_correct. Qed.

(** Witness C: [emit_ast_kernel] faithfully copies [kern_shared]
    into [ptx_kern_shared] — the field is not dropped. *)
Example example_shared_field_copied :
  (emit_ast_kernel ex_k_with_shared).(ptx_kern_shared) =
  ex_k_with_shared.(kern_shared).
Proof. reflexivity. Qed.

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
