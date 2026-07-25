(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_ir_opencl - OpenCL Code Generation from Sarek IR
 *
 * Generates OpenCL C source code from Sarek_ir.kernel.
 * This is the Phase 4 replacement for the Kirc_Ast-based Gen.ml generator.
 *
 * Features:
 * - Direct generation from clean IR (not legacy Kirc_Ast)
 * - Intrinsic registry support for extensible builtins
 * - Record/variant type support with C struct generation
 * - Pragma support for optimization hints
 ******************************************************************************)

open Sarek_ir_types

(** Local error module — same raised exception as the package-level
    [Opencl_error]. *)
module Codegen_error = Sarek_backend_error.Backend_error.Make (struct
  let name = "OpenCL"
end)

module Dispatch = Sarek_ir_intrinsic_dispatch

(** Raise a located invalid-argument-count error (atomic-arity helper for the
    shared {!Dispatch.emit_atomic}). *)
let bad_arity n e g =
  Codegen_error.raise_error (Codegen_error.invalid_arg_count n e g)

(** {1 Constants} *)

(** Buffer size for small temporary string buffers *)
let small_buffer_size = 64

(** Buffer size for large code generation buffers *)
let large_buffer_size = 4096

(** Format float with full precision (17 digits for double precision) *)
let format_float f = Printf.sprintf "%.17g" f

(** Current framework string for SNative code generation. Always [None] in
    normal use; SNative branches check this ref and error if None. *)
let current_framework : string option ref = ref None

(** Current kernel's variant definitions (set during generate) *)
let current_variants : (string * (string * elttype list) list) list ref = ref []

(** {1 Type Mapping} *)

let mangle_name = Sarek_ir_codegen.mangle_name

(** Map Sarek IR element type to OpenCL C type string *)
let rec opencl_type_of_elttype = function
  | TInt32 -> "int"
  | TInt64 -> "long"
  | TFloat16 ->
      (* Deferred to #57 slice 2: OpenCL spells this "half" but needs
         `#pragma OPENCL EXTENSION cl_khr_fp16 : enable` prepended (mirroring
         generate_with_fp64) plus the `h` literal suffix. Reject explicitly
         rather than emit a type the kernel preamble has not enabled. *)
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "f16"
           "OpenCL: float16 not yet supported (#57 slice 2 — needs cl_khr_fp16 \
            enablement)")
  | TFloat32 -> "float"
  | TFloat64 -> "double"
  | TBool -> "int"
  | TUnit -> "void"
  | TRecord (name, _) -> mangle_name name
  | TVariant (name, _) -> mangle_name name
  | TArray (elt, _) -> opencl_type_of_elttype elt ^ "*"
  | TVec elt -> opencl_type_of_elttype elt ^ "*"

(** Map memory space to OpenCL qualifier *)
let opencl_memspace = function
  | Global -> "__global"
  | Shared -> "__local"
  | Local -> ""

(** Map Sarek IR element type to OpenCL C type for kernel parameters *)
let opencl_param_type = function
  | TVec elt -> "__global " ^ opencl_type_of_elttype elt ^ "* restrict"
  | TArray (elt, ms) ->
      opencl_memspace ms ^ " " ^ opencl_type_of_elttype elt ^ "*"
  | t -> opencl_type_of_elttype t

(** {1 Thread Intrinsics} *)

let opencl_thread_intrinsic = function
  (* Support both idx and id naming conventions *)
  | "thread_id_x" | "thread_idx_x" -> "get_local_id(0)"
  | "thread_id_y" | "thread_idx_y" -> "get_local_id(1)"
  | "thread_id_z" | "thread_idx_z" -> "get_local_id(2)"
  | "block_id_x" | "block_idx_x" -> "get_group_id(0)"
  | "block_id_y" | "block_idx_y" -> "get_group_id(1)"
  | "block_id_z" | "block_idx_z" -> "get_group_id(2)"
  | "block_dim_x" -> "get_local_size(0)"
  | "block_dim_y" -> "get_local_size(1)"
  | "block_dim_z" -> "get_local_size(2)"
  | "grid_dim_x" -> "get_num_groups(0)"
  | "grid_dim_y" -> "get_num_groups(1)"
  | "grid_dim_z" -> "get_num_groups(2)"
  | "global_thread_id" | "global_idx" | "global_idx_x" -> "get_global_id(0)"
  | "global_idx_y" -> "get_global_id(1)"
  | "global_idx_z" -> "get_global_id(2)"
  | "global_size" -> "get_global_size(0)"
  | name -> Codegen_error.raise_error (Codegen_error.unknown_intrinsic name)

(** {1 Expression Generation} *)

let rec gen_expr buf = function
  | EConst (CInt32 n) -> Buffer.add_string buf (Int32.to_string n)
  | EConst (CInt64 n) -> Buffer.add_string buf (Int64.to_string n ^ "L")
  | EConst (CFloat32 f) ->
      let s = format_float f in
      (* Ensure decimal point for OpenCL compatibility *)
      let s =
        if String.contains s '.' || String.contains s 'e' then s else s ^ ".0"
      in
      Buffer.add_string buf (s ^ "f")
  | EConst (CFloat64 f) -> Buffer.add_string buf (format_float f)
  | EConst (CBool true) -> Buffer.add_string buf "1"
  | EConst (CBool false) -> Buffer.add_string buf "0"
  | EConst CUnit -> Buffer.add_string buf "(void)0"
  | EVar v -> Buffer.add_string buf v.var_name
  | EBinop (op, e1, e2) ->
      Buffer.add_char buf '(' ;
      gen_expr buf e1 ;
      Buffer.add_string buf (gen_binop op) ;
      gen_expr buf e2 ;
      Buffer.add_char buf ')'
  | EUnop (op, e) ->
      Buffer.add_char buf '(' ;
      Buffer.add_string buf (gen_unop op) ;
      gen_expr buf e ;
      Buffer.add_char buf ')'
  | EArrayRead (arr, idx) ->
      Buffer.add_string buf arr ;
      Buffer.add_char buf '[' ;
      gen_expr buf idx ;
      Buffer.add_char buf ']'
  | EArrayReadExpr (base, idx) ->
      Buffer.add_char buf '(' ;
      gen_expr buf base ;
      Buffer.add_string buf ")[" ;
      gen_expr buf idx ;
      Buffer.add_char buf ']'
  | ERecordField (e, field) ->
      gen_expr buf e ;
      Buffer.add_char buf '.' ;
      Buffer.add_string buf field
  | EIntrinsic (path, name, args) ->
      Dispatch.gen_intrinsic opencl_backend buf path name args
  | ECast (ty, e) ->
      Buffer.add_char buf '(' ;
      Buffer.add_string buf (opencl_type_of_elttype ty) ;
      Buffer.add_char buf ')' ;
      gen_expr buf e
  | ETuple exprs ->
      Buffer.add_string buf "{" ;
      List.iteri
        (fun i e ->
          if i > 0 then Buffer.add_string buf ", " ;
          gen_expr buf e)
        exprs ;
      Buffer.add_string buf "}"
  | EApp (fn, args) ->
      gen_expr buf fn ;
      Buffer.add_char buf '(' ;
      List.iteri
        (fun i e ->
          if i > 0 then Buffer.add_string buf ", " ;
          gen_expr buf e)
        args ;
      Buffer.add_char buf ')'
  | ERecord (name, fields) ->
      Buffer.add_string buf ("(" ^ mangle_name name ^ "){") ;
      List.iteri
        (fun i (f, e) ->
          if i > 0 then Buffer.add_string buf ", " ;
          Buffer.add_string buf ("." ^ f ^ " = ") ;
          gen_expr buf e)
        fields ;
      Buffer.add_string buf "}"
  | EVariant (type_name, constr, []) ->
      (* Nullary constructor - use constructor function for proper struct init *)
      let mangled = mangle_name type_name in
      Buffer.add_string buf ("make_" ^ mangled ^ "_" ^ constr ^ "()")
  | EVariant (type_name, constr, args) ->
      let mangled = mangle_name type_name in
      Buffer.add_string buf ("make_" ^ mangled ^ "_" ^ constr ^ "(") ;
      List.iteri
        (fun i e ->
          if i > 0 then Buffer.add_string buf ", " ;
          gen_expr buf e)
        args ;
      Buffer.add_char buf ')'
  | EArrayLen arr -> Buffer.add_string buf ("sarek_" ^ arr ^ "_length")
  | EArrayCreate _ ->
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "EArrayCreate"
           "should be handled in gen_stmt SLet")
  | EIf (cond, then_, else_) ->
      (* Ternary operator for value-returning if *)
      Buffer.add_char buf '(' ;
      gen_expr buf cond ;
      Buffer.add_string buf " ? " ;
      gen_expr buf then_ ;
      Buffer.add_string buf " : " ;
      gen_expr buf else_ ;
      Buffer.add_char buf ')'
  | EMatch (_, []) ->
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct "EMatch" "empty match expression")
  | EMatch (_, [(_, body)]) ->
      (* Single case - just emit the body *)
      gen_expr buf body
  | EMatch (e, cases) ->
      (* Multi-case match as nested ternary - check tag field *)
      let rec gen_cases = function
        | [] ->
            Codegen_error.raise_error
              (Codegen_error.unsupported_construct
                 "EMatch"
                 "empty match cases list")
        | [(_, body)] -> gen_expr buf body
        | (pat, body) :: rest ->
            Buffer.add_char buf '(' ;
            (match pat with
            | PConstr (name, _) ->
                Buffer.add_char buf '(' ;
                gen_expr buf e ;
                Buffer.add_string buf (".tag == " ^ name ^ ")")
            | PWild -> Buffer.add_string buf "1") ;
            Buffer.add_string buf " ? " ;
            gen_expr buf body ;
            Buffer.add_string buf " : " ;
            gen_cases rest ;
            Buffer.add_char buf ')'
      in
      gen_cases cases

and gen_binop = function
  | Add -> " + "
  | Sub -> " - "
  | Mul -> " * "
  | Div -> " / "
  | Mod -> " % "
  | Eq -> " == "
  | Ne -> " != "
  | Lt -> " < "
  | Le -> " <= "
  | Gt -> " > "
  | Ge -> " >= "
  | And -> " && "
  | Or -> " || "
  | Shl -> " << "
  | Shr -> " >> "
  | BitAnd -> " & "
  | BitOr -> " | "
  | BitXor -> " ^ "

and gen_unop = function Neg -> "-" | Not -> "!" | BitNot -> "~"

and opencl_backend =
  {
    Dispatch.framework =
      (fun () -> Option.value ~default:"OpenCL" !current_framework);
    gen_expr;
    thread_intrinsic = opencl_thread_intrinsic;
    pre_hook = (fun _ ~full_name:_ _ _ _ -> false);
    post_hook =
      (fun buf path name args ->
        Dispatch.emit_registry_template ~gen_expr buf path name args);
    on_unknown =
      (fun full ->
        Codegen_error.raise_error (Codegen_error.unknown_intrinsic full));
    arm =
      (fun name ->
        match name with
        | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh"
        | "tanh" | "exp" | "exp2" | "log" | "log2" | "log10" | "sqrt" | "rsqrt"
        | "cbrt" | "floor" | "ceil" | "round" | "trunc" | "fabs" | "atan2"
        | "pow" | "fma" | "min" | "max" ->
            Some (fun buf args -> Dispatch.emit_call ~gen_expr buf name args)
        | "block_barrier" ->
            Some
              (fun buf _ ->
                Buffer.add_string buf "barrier(CLK_LOCAL_MEM_FENCE)")
        | "atomic_add" | "atomic_add_int32" | "atomic_add_global_int32" ->
            Some
              (fun buf args ->
                Dispatch.emit_atomic
                  ~gen_expr
                  ~invalid_arg_count:bad_arity
                  buf
                  ~callee:"atomic_add"
                  ~prefix:"&"
                  ~suffix:")"
                  ~opname:"atomic_add"
                  ~expected:3
                  ~allow_array:true
                  args)
        | "atomic_sub" ->
            Some
              (fun buf args ->
                Dispatch.emit_atomic
                  ~gen_expr
                  ~invalid_arg_count:bad_arity
                  buf
                  ~callee:"atomic_sub"
                  ~prefix:"&"
                  ~suffix:")"
                  ~opname:"atomic_sub"
                  ~expected:2
                  ~allow_array:false
                  args)
        | "atomic_min" ->
            Some
              (fun buf args ->
                Dispatch.emit_atomic
                  ~gen_expr
                  ~invalid_arg_count:bad_arity
                  buf
                  ~callee:"atomic_min"
                  ~prefix:"&"
                  ~suffix:")"
                  ~opname:"atomic_min"
                  ~expected:2
                  ~allow_array:false
                  args)
        | "atomic_max" ->
            Some
              (fun buf args ->
                Dispatch.emit_atomic
                  ~gen_expr
                  ~invalid_arg_count:bad_arity
                  buf
                  ~callee:"atomic_max"
                  ~prefix:"&"
                  ~suffix:")"
                  ~opname:"atomic_max"
                  ~expected:2
                  ~allow_array:false
                  args)
        | _ -> None);
  }

(** {1 L-value Generation} *)

let gen_lvalue buf lv = Sarek_ir_codegen.gen_lvalue ~gen_expr buf lv

(** {1 Statement Generation} *)

let rec gen_stmt buf indent = function
  | SEmpty -> ()
  | SSeq stmts -> List.iter (gen_stmt buf indent) stmts
  | SAssign (lv, e) ->
      Buffer.add_string buf indent ;
      gen_lvalue buf lv ;
      Buffer.add_string buf " = " ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n"
  | SIf (cond, then_, else_opt) -> (
      Buffer.add_string buf indent ;
      Buffer.add_string buf "if (" ;
      gen_expr buf cond ;
      Buffer.add_string buf ") {\n" ;
      gen_stmt buf (indent ^ "  ") then_ ;
      Buffer.add_string buf indent ;
      Buffer.add_string buf "}" ;
      match else_opt with
      | None -> Buffer.add_char buf '\n'
      | Some else_ ->
          Buffer.add_string buf " else {\n" ;
          gen_stmt buf (indent ^ "  ") else_ ;
          Buffer.add_string buf indent ;
          Buffer.add_string buf "}\n")
  | SWhile (cond, body) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "while (" ;
      gen_expr buf cond ;
      Buffer.add_string buf ") {\n" ;
      gen_stmt buf (indent ^ "  ") body ;
      Buffer.add_string buf indent ;
      Buffer.add_string buf "}\n"
  | SFor (v, start, stop, dir, body) ->
      (* OCaml 'for i = a to b' is inclusive, so use <= not < *)
      let op, incr =
        match dir with Upto -> ("<=", "++") | Downto -> (">=", "--")
      in
      Buffer.add_string buf indent ;
      Buffer.add_string buf "for (" ;
      Buffer.add_string buf (opencl_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf " = " ;
      gen_expr buf start ;
      Buffer.add_string buf "; " ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf (" " ^ op ^ " ") ;
      gen_expr buf stop ;
      Buffer.add_string buf "; " ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf incr ;
      Buffer.add_string buf ") {\n" ;
      gen_stmt buf (indent ^ "  ") body ;
      Buffer.add_string buf indent ;
      Buffer.add_string buf "}\n"
  | SMatch (e, cases) ->
      (* Generate scrutinee into a temp buffer to get its string representation *)
      let scrutinee_buf = Buffer.create small_buffer_size in
      gen_expr scrutinee_buf e ;
      let scrutinee = Buffer.contents scrutinee_buf in
      Buffer.add_string buf indent ;
      Buffer.add_string buf "switch (" ;
      Buffer.add_string buf scrutinee ;
      Buffer.add_string buf ".tag) {\n" ;
      List.iter
        (fun (pattern, body) ->
          gen_match_case buf indent scrutinee pattern body)
        cases ;
      Buffer.add_string buf indent ;
      Buffer.add_string buf "}\n"
  | SReturn e ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "return " ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n"
  | SBarrier ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "barrier(CLK_LOCAL_MEM_FENCE);\n"
  | SWarpBarrier ->
      (* OpenCL doesn't have warp-level sync, use sub_group_barrier if available *)
      Buffer.add_string buf indent ;
      Buffer.add_string buf "sub_group_barrier(CLK_LOCAL_MEM_FENCE);\n"
  | SMemFence ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "mem_fence(CLK_GLOBAL_MEM_FENCE);\n"
  | SNative {gpu; ocaml = _} -> (
      match !current_framework with
      | Some framework ->
          let code = gpu ~framework in
          Buffer.add_string buf indent ;
          Buffer.add_string buf code ;
          if not (String.length code > 0 && code.[String.length code - 1] = '\n')
          then Buffer.add_char buf '\n'
      | None ->
          Codegen_error.raise_error
            (Codegen_error.unsupported_construct
               "SNative"
               "SNative requires device context (set current_framework before \
                calling generate)"))
  | SExpr e ->
      Buffer.add_string buf indent ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n"
  | SLet (v, EArrayCreate (elem_ty, size, mem), body) ->
      gen_array_decl buf indent v elem_ty size mem body
  | SLet (v, e, body) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf (opencl_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf " = " ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n" ;
      gen_stmt buf indent body
  | SLetMut (v, e, body) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf (opencl_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf " = " ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n" ;
      gen_stmt buf indent body
  | SPragma (hints, body) ->
      (* OpenCL uses #pragma for hints *)
      Buffer.add_string buf indent ;
      Buffer.add_string buf "#pragma " ;
      Buffer.add_string buf (String.concat " " hints) ;
      Buffer.add_char buf '\n' ;
      gen_stmt buf indent body
  | SBlock body ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "{\n" ;
      gen_stmt buf (indent ^ "  ") body ;
      Buffer.add_string buf indent ;
      Buffer.add_string buf "}\n"

(** Generate a pattern match case (extracted helper) *)
and gen_match_case buf indent scrutinee pattern body =
  let find_constr_types cname =
    List.find_map
      (fun (_vname, constrs) ->
        List.find_map
          (fun (cn, args) -> if cn = cname then Some args else None)
          constrs)
      !current_variants
  in
  Buffer.add_string buf indent ;
  (match pattern with
  | PConstr (cname, bindings) -> (
      Buffer.add_string buf ("  case " ^ cname ^ ": {\n") ;
      (* Generate bindings: extract payload from scrutinee *)
      match (bindings, find_constr_types cname) with
      | [var_name], Some [ty] ->
          (* Single payload: access data.Constructor_v *)
          Buffer.add_string buf (indent ^ "    ") ;
          Buffer.add_string buf (opencl_type_of_elttype ty) ;
          Buffer.add_string buf " " ;
          Buffer.add_string buf var_name ;
          Buffer.add_string buf " = " ;
          Buffer.add_string buf scrutinee ;
          Buffer.add_string buf ".data." ;
          Buffer.add_string buf cname ;
          Buffer.add_string buf "_v;\n"
      | vars, Some types when List.length vars = List.length types ->
          (* Multiple payloads: access data.Constructor_v._0, ._1, etc. *)
          List.iteri
            (fun i (var_name, ty) ->
              Buffer.add_string buf (indent ^ "    ") ;
              Buffer.add_string buf (opencl_type_of_elttype ty) ;
              Buffer.add_string buf " " ;
              Buffer.add_string buf var_name ;
              Buffer.add_string buf " = " ;
              Buffer.add_string buf scrutinee ;
              Buffer.add_string buf ".data." ;
              Buffer.add_string buf cname ;
              Buffer.add_string buf (Printf.sprintf "_v._%d;\n" i))
            (List.combine vars types)
      | [], _ | _, None | _, Some [] -> () (* No bindings needed *)
      | _ ->
          Codegen_error.raise_error
            (Codegen_error.type_error
               "pattern match"
               "matching bindings"
               "mismatched constructor args"))
  | PWild -> Buffer.add_string buf "  default: {\n") ;
  gen_stmt buf (indent ^ "    ") body ;
  Buffer.add_string buf (indent ^ "    break;\n") ;
  Buffer.add_string buf (indent ^ "  }\n")

(** Generate array declaration with optional __local qualifier (extracted
    helper) *)
and gen_array_decl buf indent v elem_ty size mem body =
  Buffer.add_string buf indent ;
  (match mem with Shared -> Buffer.add_string buf "__local " | _ -> ()) ;
  Buffer.add_string buf (opencl_type_of_elttype elem_ty) ;
  Buffer.add_char buf ' ' ;
  Buffer.add_string buf v.var_name ;
  Buffer.add_char buf '[' ;
  gen_expr buf size ;
  Buffer.add_string buf "];\n" ;
  gen_stmt buf indent body

(** {1 Declaration Generation} *)

let gen_param buf decl =
  Sarek_ir_codegen.gen_param
    ~param_type:opencl_param_type
    ~gen_array_param:
      (Sarek_ir_codegen.gen_global_array_param
         ~memspace:opencl_memspace
         ~type_of_elttype:opencl_type_of_elttype)
    ~invalid:(fun () ->
      Codegen_error.raise_error
        (Codegen_error.invalid_memory_space "gen_param" "DLocal or DShared"))
    buf
    decl

let gen_local buf indent = function
  | DLocal (v, None) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf (opencl_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf ";\n"
  | DLocal (v, Some e) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf (opencl_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf " = " ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n"
  | DShared (name, elt, None) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "__local " ;
      Buffer.add_string buf (opencl_type_of_elttype elt) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf name ;
      Buffer.add_string buf "[];\n"
  | DShared (name, elt, Some size) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "__local " ;
      Buffer.add_string buf (opencl_type_of_elttype elt) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf name ;
      Buffer.add_char buf '[' ;
      gen_expr buf size ;
      Buffer.add_string buf "];\n"
  | DParam _ ->
      Codegen_error.raise_error
        (Codegen_error.invalid_memory_space "gen_local" "DParam")

(** {1 Helper Function Generation} *)

(** Generate a helper function (OpenCL device function) *)
let gen_helper_func buf (hf : helper_func) =
  (* In OpenCL, helper functions don't need any special decoration *)
  Buffer.add_string buf (opencl_type_of_elttype hf.hf_ret_type) ;
  Buffer.add_char buf ' ' ;
  Buffer.add_string buf hf.hf_name ;
  Buffer.add_char buf '(' ;
  (* Parameters - use opencl_param_type to add __global for vector params *)
  List.iteri
    (fun i (v : var) ->
      if i > 0 then Buffer.add_string buf ", " ;
      Buffer.add_string buf (opencl_param_type v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name)
    hf.hf_params ;
  Buffer.add_string buf ") {\n" ;
  (* Body *)
  gen_stmt buf "  " hf.hf_body ;
  Buffer.add_string buf "}\n\n"

(** {1 Kernel Generation} *)

(** Generate complete OpenCL source for a kernel *)
let generate (k : kernel) : string =
  let buf = Buffer.create large_buffer_size in

  (* Generate helper functions before kernel *)
  List.iter (gen_helper_func buf) k.kern_funcs ;

  (* Kernel signature *)
  Buffer.add_string buf "__kernel void " ;
  Buffer.add_string buf k.kern_name ;
  Buffer.add_char buf '(' ;

  (* Parameters *)
  List.iteri
    (fun i p ->
      if i > 0 then Buffer.add_string buf ", " ;
      gen_param buf p)
    k.kern_params ;

  Buffer.add_string buf ") {\n" ;

  (* Local declarations *)
  List.iter (gen_local buf "  ") k.kern_locals ;

  (* Body *)
  gen_stmt buf "  " k.kern_body ;

  (* Close kernel *)
  Buffer.add_string buf "}\n" ;

  Buffer.contents buf

(** Generate variant type definition for OpenCL *)
let gen_variant_def buf v =
  Sarek_ir_codegen.gen_variant_def
    ~type_of_elttype:opencl_type_of_elttype
    ~constructor_prefix:"static inline"
    buf
    v

(** Generate OpenCL source with custom type definitions *)
let generate_with_types ~(types : (string * (string * elttype) list) list)
    (k : kernel) : string =
  (* Set current_variants for SMatch binding extraction *)
  current_variants := k.kern_variants ;
  let buf = Buffer.create large_buffer_size in

  (* Variant type definitions first (may be needed by records) *)
  List.iter (gen_variant_def buf) k.kern_variants ;

  (* Record type definitions *)
  Sarek_ir_codegen.gen_record_typedefs
    ~type_of_elttype:opencl_type_of_elttype
    buf
    types ;

  (* Generate helper functions before kernel *)
  List.iter (gen_helper_func buf) k.kern_funcs ;

  (* Kernel signature *)
  Buffer.add_string buf "__kernel void " ;
  Buffer.add_string buf k.kern_name ;
  Buffer.add_char buf '(' ;

  (* Parameters *)
  List.iteri
    (fun i p ->
      if i > 0 then Buffer.add_string buf ", " ;
      gen_param buf p)
    k.kern_params ;

  Buffer.add_string buf ") {\n" ;

  (* Local declarations *)
  List.iter (gen_local buf "  ") k.kern_locals ;

  (* Body *)
  gen_stmt buf "  " k.kern_body ;

  (* Close kernel *)
  Buffer.add_string buf "}\n" ;

  Buffer.contents buf

(** Generate OpenCL source with double precision extension if needed *)
let generate_with_fp64 (k : kernel) : string =
  let source = generate k in
  if Sarek_ir_analysis.kernel_uses_float64 k then
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n" ^ source
  else source
