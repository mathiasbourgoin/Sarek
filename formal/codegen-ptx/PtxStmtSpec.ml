open AGpuSemantics
open PtxExprSpec
open PtxTypes
open String

(** val reg_write : string -> ptx_val -> agpu_state -> agpu_state **)

let reg_write name v st =
  { regs = (fun n -> if eqb n name then Some v else st.regs n); tc = st.tc;
    mem = st.mem }

type ir_stmt =
| ISEmpty
| ISSeq of ir_stmt * ir_stmt
| ISLet of string * ir_expr * ir_stmt
| ISLetMut of string * ir_expr * ir_stmt
| ISAssign of string * ir_expr
| ISIf of ir_expr * ir_stmt * ir_stmt
| ISBarrier

(** val ir_stmt_rect :
    'a1 -> (ir_stmt -> 'a1 -> ir_stmt -> 'a1 -> 'a1) -> (string -> ir_expr ->
    ir_stmt -> 'a1 -> 'a1) -> (string -> ir_expr -> ir_stmt -> 'a1 -> 'a1) ->
    (string -> ir_expr -> 'a1) -> (ir_expr -> ir_stmt -> 'a1 -> ir_stmt ->
    'a1 -> 'a1) -> 'a1 -> ir_stmt -> 'a1 **)

let rec ir_stmt_rect f f0 f1 f2 f3 f4 f5 = function
| ISEmpty -> f
| ISSeq (i0, i1) ->
  f0 i0 (ir_stmt_rect f f0 f1 f2 f3 f4 f5 i0) i1
    (ir_stmt_rect f f0 f1 f2 f3 f4 f5 i1)
| ISLet (s, i0, i1) -> f1 s i0 i1 (ir_stmt_rect f f0 f1 f2 f3 f4 f5 i1)
| ISLetMut (s, i0, i1) -> f2 s i0 i1 (ir_stmt_rect f f0 f1 f2 f3 f4 f5 i1)
| ISAssign (s, i0) -> f3 s i0
| ISIf (i0, i1, i2) ->
  f4 i0 i1 (ir_stmt_rect f f0 f1 f2 f3 f4 f5 i1) i2
    (ir_stmt_rect f f0 f1 f2 f3 f4 f5 i2)
| ISBarrier -> f5

(** val ir_stmt_rec :
    'a1 -> (ir_stmt -> 'a1 -> ir_stmt -> 'a1 -> 'a1) -> (string -> ir_expr ->
    ir_stmt -> 'a1 -> 'a1) -> (string -> ir_expr -> ir_stmt -> 'a1 -> 'a1) ->
    (string -> ir_expr -> 'a1) -> (ir_expr -> ir_stmt -> 'a1 -> ir_stmt ->
    'a1 -> 'a1) -> 'a1 -> ir_stmt -> 'a1 **)

let rec ir_stmt_rec f f0 f1 f2 f3 f4 f5 = function
| ISEmpty -> f
| ISSeq (i0, i1) ->
  f0 i0 (ir_stmt_rec f f0 f1 f2 f3 f4 f5 i0) i1
    (ir_stmt_rec f f0 f1 f2 f3 f4 f5 i1)
| ISLet (s, i0, i1) -> f1 s i0 i1 (ir_stmt_rec f f0 f1 f2 f3 f4 f5 i1)
| ISLetMut (s, i0, i1) -> f2 s i0 i1 (ir_stmt_rec f f0 f1 f2 f3 f4 f5 i1)
| ISAssign (s, i0) -> f3 s i0
| ISIf (i0, i1, i2) ->
  f4 i0 i1 (ir_stmt_rec f f0 f1 f2 f3 f4 f5 i1) i2
    (ir_stmt_rec f f0 f1 f2 f3 f4 f5 i2)
| ISBarrier -> f5

(** val agpu_exec_ir : agpu_state -> ir_stmt -> agpu_state option **)

let rec agpu_exec_ir st = function
| ISSeq (s1, s2) ->
  (match agpu_exec_ir st s1 with
   | Some st1 -> agpu_exec_ir st1 s2
   | None -> None)
| ISLet (name, e, body) ->
  (match agpu_eval_ir st e with
   | Some p -> let (v, st1) = p in agpu_exec_ir (reg_write name v st1) body
   | None -> None)
| ISLetMut (name, e, body) ->
  (match agpu_eval_ir st e with
   | Some p -> let (v, st1) = p in agpu_exec_ir (reg_write name v st1) body
   | None -> None)
| ISAssign (name, e) ->
  (match agpu_eval_ir st e with
   | Some p -> let (v, st1) = p in Some (reg_write name v st1)
   | None -> None)
| ISIf (cond, s1, s2) ->
  (match agpu_eval_ir st cond with
   | Some p ->
     let (p0, st1) = p in
     (match p0 with
      | U32 n ->
        ((fun fO fS n -> if n=0 then fO () else fS (n-1))
           (fun _ -> agpu_exec_ir st1 s2)
           (fun _ -> agpu_exec_ir st1 s1)
           n)
      | Pred b -> if b then agpu_exec_ir st1 s1 else agpu_exec_ir st1 s2
      | _ -> None)
   | None -> None)
| _ -> Some st

type ptx_stmt_ast =
| PSEmpty
| PSSeq of ptx_stmt_ast * ptx_stmt_ast
| PSLet of string * ptx_expr_ast * ptx_stmt_ast
| PSLetMut of string * ptx_expr_ast * ptx_stmt_ast
| PSAssign of string * ptx_expr_ast
| PSIf of ptx_expr_ast * ptx_stmt_ast * ptx_stmt_ast
| PSBarrier

(** val ptx_stmt_ast_rect :
    'a1 -> (ptx_stmt_ast -> 'a1 -> ptx_stmt_ast -> 'a1 -> 'a1) -> (string ->
    ptx_expr_ast -> ptx_stmt_ast -> 'a1 -> 'a1) -> (string -> ptx_expr_ast ->
    ptx_stmt_ast -> 'a1 -> 'a1) -> (string -> ptx_expr_ast -> 'a1) ->
    (ptx_expr_ast -> ptx_stmt_ast -> 'a1 -> ptx_stmt_ast -> 'a1 -> 'a1) ->
    'a1 -> ptx_stmt_ast -> 'a1 **)

let rec ptx_stmt_ast_rect f f0 f1 f2 f3 f4 f5 = function
| PSEmpty -> f
| PSSeq (p0, p1) ->
  f0 p0 (ptx_stmt_ast_rect f f0 f1 f2 f3 f4 f5 p0) p1
    (ptx_stmt_ast_rect f f0 f1 f2 f3 f4 f5 p1)
| PSLet (s, p0, p1) -> f1 s p0 p1 (ptx_stmt_ast_rect f f0 f1 f2 f3 f4 f5 p1)
| PSLetMut (s, p0, p1) ->
  f2 s p0 p1 (ptx_stmt_ast_rect f f0 f1 f2 f3 f4 f5 p1)
| PSAssign (s, p0) -> f3 s p0
| PSIf (p0, p1, p2) ->
  f4 p0 p1 (ptx_stmt_ast_rect f f0 f1 f2 f3 f4 f5 p1) p2
    (ptx_stmt_ast_rect f f0 f1 f2 f3 f4 f5 p2)
| PSBarrier -> f5

(** val ptx_stmt_ast_rec :
    'a1 -> (ptx_stmt_ast -> 'a1 -> ptx_stmt_ast -> 'a1 -> 'a1) -> (string ->
    ptx_expr_ast -> ptx_stmt_ast -> 'a1 -> 'a1) -> (string -> ptx_expr_ast ->
    ptx_stmt_ast -> 'a1 -> 'a1) -> (string -> ptx_expr_ast -> 'a1) ->
    (ptx_expr_ast -> ptx_stmt_ast -> 'a1 -> ptx_stmt_ast -> 'a1 -> 'a1) ->
    'a1 -> ptx_stmt_ast -> 'a1 **)

let rec ptx_stmt_ast_rec f f0 f1 f2 f3 f4 f5 = function
| PSEmpty -> f
| PSSeq (p0, p1) ->
  f0 p0 (ptx_stmt_ast_rec f f0 f1 f2 f3 f4 f5 p0) p1
    (ptx_stmt_ast_rec f f0 f1 f2 f3 f4 f5 p1)
| PSLet (s, p0, p1) -> f1 s p0 p1 (ptx_stmt_ast_rec f f0 f1 f2 f3 f4 f5 p1)
| PSLetMut (s, p0, p1) -> f2 s p0 p1 (ptx_stmt_ast_rec f f0 f1 f2 f3 f4 f5 p1)
| PSAssign (s, p0) -> f3 s p0
| PSIf (p0, p1, p2) ->
  f4 p0 p1 (ptx_stmt_ast_rec f f0 f1 f2 f3 f4 f5 p1) p2
    (ptx_stmt_ast_rec f f0 f1 f2 f3 f4 f5 p2)
| PSBarrier -> f5

(** val agpu_exec_ptx_stmt :
    agpu_state -> ptx_stmt_ast -> agpu_state option **)

let rec agpu_exec_ptx_stmt st = function
| PSSeq (s1, s2) ->
  (match agpu_exec_ptx_stmt st s1 with
   | Some st1 -> agpu_exec_ptx_stmt st1 s2
   | None -> None)
| PSLet (name, e, body) ->
  (match agpu_eval_ptx st e with
   | Some p ->
     let (v, st1) = p in agpu_exec_ptx_stmt (reg_write name v st1) body
   | None -> None)
| PSLetMut (name, e, body) ->
  (match agpu_eval_ptx st e with
   | Some p ->
     let (v, st1) = p in agpu_exec_ptx_stmt (reg_write name v st1) body
   | None -> None)
| PSAssign (name, e) ->
  (match agpu_eval_ptx st e with
   | Some p -> let (v, st1) = p in Some (reg_write name v st1)
   | None -> None)
| PSIf (cond, s1, s2) ->
  (match agpu_eval_ptx st cond with
   | Some p ->
     let (p0, st1) = p in
     (match p0 with
      | U32 n ->
        ((fun fO fS n -> if n=0 then fO () else fS (n-1))
           (fun _ -> agpu_exec_ptx_stmt st1 s2)
           (fun _ -> agpu_exec_ptx_stmt st1 s1)
           n)
      | Pred b ->
        if b then agpu_exec_ptx_stmt st1 s1 else agpu_exec_ptx_stmt st1 s2
      | _ -> None)
   | None -> None)
| _ -> Some st

(** val emit_ast_stmt : ir_stmt -> ptx_stmt_ast **)

let rec emit_ast_stmt = function
| ISEmpty -> PSEmpty
| ISSeq (s1, s2) -> PSSeq ((emit_ast_stmt s1), (emit_ast_stmt s2))
| ISLet (n, e, body) -> PSLet (n, (emit_ast_expr e), (emit_ast_stmt body))
| ISLetMut (n, e, body) ->
  PSLetMut (n, (emit_ast_expr e), (emit_ast_stmt body))
| ISAssign (n, e) -> PSAssign (n, (emit_ast_expr e))
| ISIf (cond, s1, s2) ->
  PSIf ((emit_ast_expr cond), (emit_ast_stmt s1), (emit_ast_stmt s2))
| ISBarrier -> PSBarrier
