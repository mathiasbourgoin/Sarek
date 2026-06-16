open AGpuSemantics
open Ascii
open PtxStmtSpec
open String

type ir_shared_decl = {shdecl_name : string; shdecl_size : int}

val shdecl_name : ir_shared_decl -> string

val shdecl_size : ir_shared_decl -> int

type ir_kernel = {
  kern_name : string;
  kern_params : string list;
  kern_shared : ir_shared_decl list;
  kern_body : ir_stmt;
}

val kern_name : ir_kernel -> string

val kern_params : ir_kernel -> string list

val kern_shared : ir_kernel -> ir_shared_decl list

val kern_body : ir_kernel -> ir_stmt

type ptx_kernel_ast = {
  ptx_kern_name : string;
  ptx_kern_params : string list;
  ptx_kern_shared : ir_shared_decl list;
  ptx_kern_body : ptx_stmt_ast;
}

val ptx_kern_name : ptx_kernel_ast -> string

val ptx_kern_params : ptx_kernel_ast -> string list

val ptx_kern_shared : ptx_kernel_ast -> ir_shared_decl list

val ptx_kern_body : ptx_kernel_ast -> ptx_stmt_ast

val agpu_exec_ir_kernel : agpu_state -> ir_kernel -> agpu_state option

val agpu_exec_ptx_kernel : agpu_state -> ptx_kernel_ast -> agpu_state option

val emit_ast_kernel : ir_kernel -> ptx_kernel_ast

val ex_empty_regs : string -> ptx_val option

val ex_zero_tc : thread_const

val ex_zero_mem : agpu_mem

val ex_st : agpu_state

val ex_k_no_shared : ir_kernel

val ex_shared_decl : ir_shared_decl

val ex_k_with_shared : ir_kernel
