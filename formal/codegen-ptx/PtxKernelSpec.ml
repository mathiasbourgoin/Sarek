open AGpuSemantics
open Ascii
open PtxStmtSpec
open String

type ir_shared_decl = {shdecl_name : string; shdecl_size : int}

(** val shdecl_name : ir_shared_decl -> string **)

let shdecl_name i = i.shdecl_name

(** val shdecl_size : ir_shared_decl -> int **)

let shdecl_size i = i.shdecl_size

type ir_kernel = {
  kern_name : string;
  kern_params : string list;
  kern_shared : ir_shared_decl list;
  kern_body : ir_stmt;
}

(** val kern_name : ir_kernel -> string **)

let kern_name i = i.kern_name

(** val kern_params : ir_kernel -> string list **)

let kern_params i = i.kern_params

(** val kern_shared : ir_kernel -> ir_shared_decl list **)

let kern_shared i = i.kern_shared

(** val kern_body : ir_kernel -> ir_stmt **)

let kern_body i = i.kern_body

type ptx_kernel_ast = {
  ptx_kern_name : string;
  ptx_kern_params : string list;
  ptx_kern_shared : ir_shared_decl list;
  ptx_kern_body : ptx_stmt_ast;
}

(** val ptx_kern_name : ptx_kernel_ast -> string **)

let ptx_kern_name p = p.ptx_kern_name

(** val ptx_kern_params : ptx_kernel_ast -> string list **)

let ptx_kern_params p = p.ptx_kern_params

(** val ptx_kern_shared : ptx_kernel_ast -> ir_shared_decl list **)

let ptx_kern_shared p = p.ptx_kern_shared

(** val ptx_kern_body : ptx_kernel_ast -> ptx_stmt_ast **)

let ptx_kern_body p = p.ptx_kern_body

(** val agpu_exec_ir_kernel : agpu_state -> ir_kernel -> agpu_state option **)

let agpu_exec_ir_kernel st k = agpu_exec_ir st k.kern_body

(** val agpu_exec_ptx_kernel : agpu_state -> ptx_kernel_ast -> agpu_state option
    **)

let agpu_exec_ptx_kernel st k = agpu_exec_ptx_stmt st k.ptx_kern_body

(** val emit_ast_kernel : ir_kernel -> ptx_kernel_ast **)

let emit_ast_kernel k =
  {
    ptx_kern_name = k.kern_name;
    ptx_kern_params = k.kern_params;
    ptx_kern_shared = k.kern_shared;
    ptx_kern_body = emit_ast_stmt k.kern_body;
  }

(** val ex_empty_regs : string -> ptx_val option **)

let ex_empty_regs _ = None

(** val ex_zero_tc : thread_const **)

let ex_zero_tc = {tidx = 0; bidx = 0; bdim = 0}

(** val ex_zero_mem : agpu_mem **)

let ex_zero_mem = {global_mem = (fun _ -> U32 0); shared_mem = (fun _ -> U32 0)}

(** val ex_st : agpu_state **)

let ex_st = {regs = ex_empty_regs; tc = ex_zero_tc; mem = ex_zero_mem}

(** val ex_k_no_shared : ir_kernel **)

let ex_k_no_shared =
  {
    kern_name =
      String
        ( Ascii (false, true, true, true, false, true, true, false),
          String
            ( Ascii (true, true, true, true, false, true, true, false),
              String
                ( Ascii (true, true, true, true, true, false, true, false),
                  String
                    ( Ascii (true, true, false, false, true, true, true, false),
                      String
                        ( Ascii
                            (false, false, false, true, false, true, true, false),
                          String
                            ( Ascii
                                ( true,
                                  false,
                                  false,
                                  false,
                                  false,
                                  true,
                                  true,
                                  false ),
                              String
                                ( Ascii
                                    ( false,
                                      true,
                                      false,
                                      false,
                                      true,
                                      true,
                                      true,
                                      false ),
                                  String
                                    ( Ascii
                                        ( true,
                                          false,
                                          true,
                                          false,
                                          false,
                                          true,
                                          true,
                                          false ),
                                      String
                                        ( Ascii
                                            ( false,
                                              false,
                                              true,
                                              false,
                                              false,
                                              true,
                                              true,
                                              false ),
                                          EmptyString ) ) ) ) ) ) ) ) );
    kern_params = [];
    kern_shared = [];
    kern_body = ISEmpty;
  }

(** val ex_shared_decl : ir_shared_decl **)

let ex_shared_decl =
  {
    shdecl_name =
      String
        ( Ascii (true, true, false, false, true, true, true, false),
          String
            ( Ascii (false, false, false, true, false, true, true, false),
              String
                ( Ascii (true, false, true, true, false, true, true, false),
                  String
                    ( Ascii (true, false, true, false, false, true, true, false),
                      String
                        ( Ascii
                            (true, false, true, true, false, true, true, false),
                          EmptyString ) ) ) ) );
    shdecl_size =
      Stdlib.Int.succ
        (Stdlib.Int.succ
           (Stdlib.Int.succ
              (Stdlib.Int.succ
                 (Stdlib.Int.succ
                    (Stdlib.Int.succ
                       (Stdlib.Int.succ
                          (Stdlib.Int.succ
                             (Stdlib.Int.succ
                                (Stdlib.Int.succ
                                   (Stdlib.Int.succ
                                      (Stdlib.Int.succ
                                         (Stdlib.Int.succ
                                            (Stdlib.Int.succ
                                               (Stdlib.Int.succ
                                                  (Stdlib.Int.succ
                                                     (Stdlib.Int.succ
                                                        (Stdlib.Int.succ
                                                           (Stdlib.Int.succ
                                                              (Stdlib.Int.succ
                                                                 (Stdlib.Int
                                                                  .succ
                                                                    (Stdlib.Int
                                                                     .succ
                                                                       (Stdlib
                                                                        .Int
                                                                        .succ
                                                                          (Stdlib
                                                                           .Int
                                                                           .succ
                                                                             (Stdlib
                                                                              .Int
                                                                              .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                (
                                                                                Stdlib
                                                                                .Int
                                                                                .succ
                                                                                0)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))));
  }

(** val ex_k_with_shared : ir_kernel **)

let ex_k_with_shared =
  {
    kern_name =
      String
        ( Ascii (true, true, true, false, true, true, true, false),
          String
            ( Ascii (true, false, false, true, false, true, true, false),
              String
                ( Ascii (false, false, true, false, true, true, true, false),
                  String
                    ( Ascii (false, false, false, true, false, true, true, false),
                      String
                        ( Ascii
                            (true, true, true, true, true, false, true, false),
                          String
                            ( Ascii
                                ( true,
                                  true,
                                  false,
                                  false,
                                  true,
                                  true,
                                  true,
                                  false ),
                              String
                                ( Ascii
                                    ( false,
                                      false,
                                      false,
                                      true,
                                      false,
                                      true,
                                      true,
                                      false ),
                                  String
                                    ( Ascii
                                        ( true,
                                          false,
                                          false,
                                          false,
                                          false,
                                          true,
                                          true,
                                          false ),
                                      String
                                        ( Ascii
                                            ( false,
                                              true,
                                              false,
                                              false,
                                              true,
                                              true,
                                              true,
                                              false ),
                                          String
                                            ( Ascii
                                                ( true,
                                                  false,
                                                  true,
                                                  false,
                                                  false,
                                                  true,
                                                  true,
                                                  false ),
                                              String
                                                ( Ascii
                                                    ( false,
                                                      false,
                                                      true,
                                                      false,
                                                      false,
                                                      true,
                                                      true,
                                                      false ),
                                                  EmptyString ) ) ) ) ) ) ) ) )
            ) );
    kern_params = [];
    kern_shared = ex_shared_decl :: [];
    kern_body = ISEmpty;
  }
