open AGpuSemantics
open PtxTypes

val is_cmp_op : ir_binop -> bool

val ir_binop_to_ptx_binop : ir_binop -> ptx_binop_tag

val ir_binop_to_ptx_cmp : ir_binop -> ptx_cmp_tag

val emit_ast_expr : ir_expr -> ptx_expr_ast
