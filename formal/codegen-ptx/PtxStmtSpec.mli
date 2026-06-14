open AGpuSemantics
open PtxExprSpec
open PtxTypes
open String

val reg_write : string -> ptx_val -> agpu_state -> agpu_state

type ir_stmt =
  | ISEmpty
  | ISSeq of ir_stmt * ir_stmt
  | ISLet of string * ir_expr * ir_stmt
  | ISLetMut of string * ir_expr * ir_stmt
  | ISAssign of string * ir_expr
  | ISIf of ir_expr * ir_stmt * ir_stmt
  | ISBarrier

val ir_stmt_rect :
  'a1 ->
  (ir_stmt -> 'a1 -> ir_stmt -> 'a1 -> 'a1) ->
  (string -> ir_expr -> ir_stmt -> 'a1 -> 'a1) ->
  (string -> ir_expr -> ir_stmt -> 'a1 -> 'a1) ->
  (string -> ir_expr -> 'a1) ->
  (ir_expr -> ir_stmt -> 'a1 -> ir_stmt -> 'a1 -> 'a1) ->
  'a1 ->
  ir_stmt ->
  'a1

val ir_stmt_rec :
  'a1 ->
  (ir_stmt -> 'a1 -> ir_stmt -> 'a1 -> 'a1) ->
  (string -> ir_expr -> ir_stmt -> 'a1 -> 'a1) ->
  (string -> ir_expr -> ir_stmt -> 'a1 -> 'a1) ->
  (string -> ir_expr -> 'a1) ->
  (ir_expr -> ir_stmt -> 'a1 -> ir_stmt -> 'a1 -> 'a1) ->
  'a1 ->
  ir_stmt ->
  'a1

val agpu_exec_ir : agpu_state -> ir_stmt -> agpu_state option

type ptx_stmt_ast =
  | PSEmpty
  | PSSeq of ptx_stmt_ast * ptx_stmt_ast
  | PSLet of string * ptx_expr_ast * ptx_stmt_ast
  | PSLetMut of string * ptx_expr_ast * ptx_stmt_ast
  | PSAssign of string * ptx_expr_ast
  | PSIf of ptx_expr_ast * ptx_stmt_ast * ptx_stmt_ast
  | PSBarrier

val ptx_stmt_ast_rect :
  'a1 ->
  (ptx_stmt_ast -> 'a1 -> ptx_stmt_ast -> 'a1 -> 'a1) ->
  (string -> ptx_expr_ast -> ptx_stmt_ast -> 'a1 -> 'a1) ->
  (string -> ptx_expr_ast -> ptx_stmt_ast -> 'a1 -> 'a1) ->
  (string -> ptx_expr_ast -> 'a1) ->
  (ptx_expr_ast -> ptx_stmt_ast -> 'a1 -> ptx_stmt_ast -> 'a1 -> 'a1) ->
  'a1 ->
  ptx_stmt_ast ->
  'a1

val ptx_stmt_ast_rec :
  'a1 ->
  (ptx_stmt_ast -> 'a1 -> ptx_stmt_ast -> 'a1 -> 'a1) ->
  (string -> ptx_expr_ast -> ptx_stmt_ast -> 'a1 -> 'a1) ->
  (string -> ptx_expr_ast -> ptx_stmt_ast -> 'a1 -> 'a1) ->
  (string -> ptx_expr_ast -> 'a1) ->
  (ptx_expr_ast -> ptx_stmt_ast -> 'a1 -> ptx_stmt_ast -> 'a1 -> 'a1) ->
  'a1 ->
  ptx_stmt_ast ->
  'a1

val agpu_exec_ptx_stmt : agpu_state -> ptx_stmt_ast -> agpu_state option

val emit_ast_stmt : ir_stmt -> ptx_stmt_ast
