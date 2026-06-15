open AGpuSemantics
open PtxStmtSpec
open String

type ir_kernel = { kern_name : string; kern_params : string list;
                   kern_body : ir_stmt }

(** val kern_name : ir_kernel -> string **)

let kern_name i =
  i.kern_name

(** val kern_params : ir_kernel -> string list **)

let kern_params i =
  i.kern_params

(** val kern_body : ir_kernel -> ir_stmt **)

let kern_body i =
  i.kern_body

type ptx_kernel_ast = { ptx_kern_name : string;
                        ptx_kern_params : string list;
                        ptx_kern_body : ptx_stmt_ast }

(** val ptx_kern_name : ptx_kernel_ast -> string **)

let ptx_kern_name p =
  p.ptx_kern_name

(** val ptx_kern_params : ptx_kernel_ast -> string list **)

let ptx_kern_params p =
  p.ptx_kern_params

(** val ptx_kern_body : ptx_kernel_ast -> ptx_stmt_ast **)

let ptx_kern_body p =
  p.ptx_kern_body

(** val agpu_exec_ir_kernel : agpu_state -> ir_kernel -> agpu_state option **)

let agpu_exec_ir_kernel st k =
  agpu_exec_ir st k.kern_body

(** val agpu_exec_ptx_kernel :
    agpu_state -> ptx_kernel_ast -> agpu_state option **)

let agpu_exec_ptx_kernel st k =
  agpu_exec_ptx_stmt st k.ptx_kern_body

(** val emit_ast_kernel : ir_kernel -> ptx_kernel_ast **)

let emit_ast_kernel k =
  { ptx_kern_name = k.kern_name; ptx_kern_params = k.kern_params;
    ptx_kern_body = (emit_ast_stmt k.kern_body) }
