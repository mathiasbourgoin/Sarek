open AGpuSemantics
open PtxStmtSpec
open String

type ir_kernel = { kern_name : string; kern_params : string list;
                   kern_body : ir_stmt }

val kern_name : ir_kernel -> string

val kern_params : ir_kernel -> string list

val kern_body : ir_kernel -> ir_stmt

type ptx_kernel_ast = { ptx_kern_name : string;
                        ptx_kern_params : string list;
                        ptx_kern_body : ptx_stmt_ast }

val ptx_kern_name : ptx_kernel_ast -> string

val ptx_kern_params : ptx_kernel_ast -> string list

val ptx_kern_body : ptx_kernel_ast -> ptx_stmt_ast

val agpu_exec_ir_kernel : agpu_state -> ir_kernel -> agpu_state option

val agpu_exec_ptx_kernel : agpu_state -> ptx_kernel_ast -> agpu_state option

val emit_ast_kernel : ir_kernel -> ptx_kernel_ast
