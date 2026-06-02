(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_ir_metal - Metal Code Generation from Sarek IR
 *
 * Generates Metal C source code from Sarek_ir.kernel.
 * This is the Phase 4 replacement for the Kirc_Ast-based Gen.ml generator.
 *
 * Features:
 * - Direct generation from clean IR (not legacy Kirc_Ast)
 * - Intrinsic registry support for extensible builtins
 * - Record/variant type support with C struct generation
 * - Pragma support for optimization hints
 ******************************************************************************)

open Sarek_ir_types

(** Current framework string for SNative code generation. Set by
    [generate_for_device]; [generate] leaves this as [None]. *)
let current_framework : string option ref = ref None

(** Current kernel's variant definitions (set during generate) *)
let current_variants : (string * (string * elttype list) list) list ref = ref []

(** {1 Type Mapping} *)

let mangle_name = Sarek_ir_codegen.mangle_name

(** Map Sarek IR element type to Metal C type string *)
let rec metal_type_of_elttype = function
  | TInt32 -> "int"
  | TInt64 -> "long"
  | TFloat32 -> "float"
  | TFloat64 -> "float" (* Metal doesn't support double precision *)
  | TBool -> "bool"
  | TUnit -> "void"
  | TRecord (name, _) -> mangle_name name
  | TVariant (name, _) -> mangle_name name
  | TArray (elt, _) -> metal_type_of_elttype elt ^ "*"
  | TVec elt -> metal_type_of_elttype elt ^ "*"

(** Map memory space to Metal qualifier *)
let metal_memspace = function
  | Global -> "device"
  | Shared -> "threadgroup"
  | Local -> ""

(** Map Sarek IR element type to Metal C type for kernel parameters *)
let metal_param_type = function
  | TVec elt -> "device " ^ metal_type_of_elttype elt ^ "* restrict"
  | TArray (elt, ms) ->
      metal_memspace ms ^ " " ^ metal_type_of_elttype elt ^ "*"
  | t -> metal_type_of_elttype t

(** Map Sarek IR element type to Metal C type for helper function parameters *)
let metal_helper_param_type = function
  | TVec elt -> "device " ^ metal_type_of_elttype elt ^ "*"
  | TArray (elt, ms) ->
      metal_memspace ms ^ " " ^ metal_type_of_elttype elt ^ "*"
  | t -> metal_type_of_elttype t

(** Convert type to atomic type for Metal *)
let metal_atomic_type_of_elttype = function
  | TInt32 -> "atomic_int"
  | TInt64 -> "atomic_long"
  | TFloat32 ->
      "atomic_float" (* Metal doesn't support atomic float, but let's try *)
  | t -> metal_type_of_elttype t

(** {1 Thread Intrinsics} *)

let metal_thread_intrinsic = function
  (* Support both idx and id naming conventions *)
  | "thread_id_x" | "thread_idx_x" -> "__metal_tid.x"
  | "thread_id_y" | "thread_idx_y" -> "__metal_tid.y"
  | "thread_id_z" | "thread_idx_z" -> "__metal_tid.z"
  | "block_id_x" | "block_idx_x" -> "__metal_bid.x"
  | "block_id_y" | "block_idx_y" -> "__metal_bid.y"
  | "block_id_z" | "block_idx_z" -> "__metal_bid.z"
  | "block_dim_x" -> "__metal_tpg.x"
  | "block_dim_y" -> "__metal_tpg.y"
  | "block_dim_z" -> "__metal_tpg.z"
  | "grid_dim_x" -> "__metal_num_groups.x"
  | "grid_dim_y" -> "__metal_num_groups.y"
  | "grid_dim_z" -> "__metal_num_groups.z"
  | "global_thread_id" | "global_idx" | "global_idx_x" -> "__metal_gid.x"
  | "global_idx_y" -> "__metal_gid.y"
  | "global_idx_z" -> "__metal_gid.z"
  | "global_size" -> "__metal_tpg.x * __metal_num_groups.x"
  | name -> Metal_error.raise_error (Metal_error.unknown_intrinsic name)

(** {1 Expression Generation} *)

let rec gen_expr buf = function
  | EConst (CInt32 n) -> Buffer.add_string buf (Int32.to_string n)
  | EConst (CInt64 n) -> Buffer.add_string buf (Int64.to_string n ^ "L")
  | EConst (CFloat32 f) ->
      let s = Printf.sprintf "%.17g" f in
      (* Ensure decimal point for Metal compatibility *)
      let s =
        if String.contains s '.' || String.contains s 'e' then s else s ^ ".0"
      in
      Buffer.add_string buf (s ^ "f")
  | EConst (CFloat64 f) -> Buffer.add_string buf (Printf.sprintf "%.17g" f)
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
  | EIntrinsic (path, name, args) -> gen_intrinsic buf path name args
  | ECast (ty, e) ->
      Buffer.add_char buf '(' ;
      Buffer.add_string buf (metal_type_of_elttype ty) ;
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
      Metal_error.raise_error
        (Metal_error.unsupported_construct
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
      Metal_error.raise_error
        (Metal_error.unsupported_construct "match" "empty match expression")
  | EMatch (_, [(_, body)]) ->
      (* Single case - just emit the body *)
      gen_expr buf body
  | EMatch (e, cases) ->
      (* Multi-case match as nested ternary - check tag field *)
      let rec gen_cases = function
        | [] ->
            Metal_error.raise_error
              (Metal_error.unsupported_construct "match" "empty match cases")
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

and gen_intrinsic buf path name args =
  let full_name =
    match path with [] -> name | _ -> String.concat "." path ^ "." ^ name
  in
  (* Try thread intrinsics - support both idx and id naming *)
  if
    List.mem
      name
      [
        "thread_id_x";
        "thread_idx_x";
        "thread_id_y";
        "thread_idx_y";
        "thread_id_z";
        "thread_idx_z";
        "block_id_x";
        "block_idx_x";
        "block_id_y";
        "block_idx_y";
        "block_id_z";
        "block_idx_z";
        "block_dim_x";
        "block_dim_y";
        "block_dim_z";
        "grid_dim_x";
        "grid_dim_y";
        "grid_dim_z";
        "global_thread_id";
        "global_idx";
        "global_idx_x";
        "global_idx_y";
        "global_idx_z";
        "global_size";
      ]
  then Buffer.add_string buf (metal_thread_intrinsic name)
  else
    (* Standard math intrinsics - Metal uses same names *)
    match name with
    | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh"
    | "tanh" | "exp" | "exp2" | "log" | "log2" | "log10" | "sqrt" | "rsqrt"
    | "cbrt" | "floor" | "ceil" | "round" | "trunc" | "fabs" ->
        Buffer.add_string buf name ;
        Buffer.add_char buf '(' ;
        List.iteri
          (fun i e ->
            if i > 0 then Buffer.add_string buf ", " ;
            gen_expr buf e)
          args ;
        Buffer.add_char buf ')'
    | "atan2" | "pow" | "fma" | "min" | "max" ->
        Buffer.add_string buf name ;
        Buffer.add_char buf '(' ;
        List.iteri
          (fun i e ->
            if i > 0 then Buffer.add_string buf ", " ;
            gen_expr buf e)
          args ;
        Buffer.add_char buf ')'
    (* Barrier synchronization *)
    | "block_barrier" ->
        Buffer.add_string buf "threadgroup_barrier(mem_flags::mem_threadgroup)"
    | "atomic_add" | "atomic_add_int32" ->
        Buffer.add_string buf "atomic_fetch_add_explicit(" ;
        (match args with
        | [addr; value] ->
            (* Cast pointer to threadgroup atomic type *)
            Buffer.add_string buf "(volatile threadgroup atomic_int*)&" ;
            gen_expr buf addr ;
            Buffer.add_string buf ", " ;
            gen_expr buf value
        | [arr; idx; value] ->
            (* Array element atomic with threadgroup cast *)
            Buffer.add_string buf "(volatile threadgroup atomic_int*)&" ;
            gen_expr buf arr ;
            Buffer.add_char buf '[' ;
            gen_expr buf idx ;
            Buffer.add_string buf "], " ;
            gen_expr buf value
        | args ->
            Metal_error.raise_error
              (Metal_error.invalid_arg_count "atomic_add" 2 (List.length args))) ;
        Buffer.add_string buf ", memory_order_relaxed)"
    | "atomic_add_global_int32" ->
        Buffer.add_string buf "atomic_fetch_add_explicit(" ;
        (match args with
        | [addr; value] ->
            (* Cast pointer to device atomic type *)
            Buffer.add_string buf "(volatile device atomic_int*)&" ;
            gen_expr buf addr ;
            Buffer.add_string buf ", " ;
            gen_expr buf value
        | [arr; idx; value] ->
            (* Array element atomic with device cast *)
            Buffer.add_string buf "(volatile device atomic_int*)&" ;
            gen_expr buf arr ;
            Buffer.add_char buf '[' ;
            gen_expr buf idx ;
            Buffer.add_string buf "], " ;
            gen_expr buf value
        | args ->
            Metal_error.raise_error
              (Metal_error.invalid_arg_count
                 "atomic_add_global"
                 2
                 (List.length args))) ;
        (* Use relaxed memory order *)
        Buffer.add_string buf ", memory_order_relaxed)"
    | "atomic_sub" ->
        Buffer.add_string buf "atomic_sub(" ;
        (match args with
        | [addr; value] ->
            Buffer.add_char buf '&' ;
            gen_expr buf addr ;
            Buffer.add_string buf ", " ;
            gen_expr buf value
        | args ->
            Metal_error.raise_error
              (Metal_error.invalid_arg_count "atomic_sub" 2 (List.length args))) ;
        Buffer.add_char buf ')'
    | "atomic_min" ->
        Buffer.add_string buf "atomic_min(" ;
        (match args with
        | [addr; value] ->
            Buffer.add_char buf '&' ;
            gen_expr buf addr ;
            Buffer.add_string buf ", " ;
            gen_expr buf value
        | args ->
            Metal_error.raise_error
              (Metal_error.invalid_arg_count "atomic_min" 2 (List.length args))) ;
        Buffer.add_char buf ')'
    | "atomic_max" ->
        Buffer.add_string buf "atomic_max(" ;
        (match args with
        | [addr; value] ->
            Buffer.add_char buf '&' ;
            gen_expr buf addr ;
            Buffer.add_string buf ", " ;
            gen_expr buf value
        | args ->
            Metal_error.raise_error
              (Metal_error.invalid_arg_count "atomic_max" 2 (List.length args))) ;
        Buffer.add_char buf ')'
    | _ -> (
        (* Try registry lookup for intrinsics like float, int_of_float, etc. *)
        match Sarek_registry.fun_device_template ~module_path:path name with
        | Some template ->
            (* Generate argument strings *)
            let arg_strs =
              List.map
                (fun e ->
                  let b = Buffer.create 64 in
                  gen_expr b e ;
                  Buffer.contents b)
                args
            in
            (* Count %s placeholders in template *)
            let count_placeholders s =
              let rec count i acc =
                if i >= String.length s - 1 then acc
                else if s.[i] = '%' && s.[i + 1] = 's' then
                  count (i + 2) (acc + 1)
                else count (i + 1) acc
              in
              count 0 0
            in
            let num_placeholders = count_placeholders template in
            let result =
              if num_placeholders = 0 then
                (* Plain function/cast like "(float)" -> call as function *)
                template ^ "(" ^ String.concat ", " arg_strs ^ ")"
              else if num_placeholders = 1 && List.length arg_strs = 1 then
                Printf.sprintf
                  (Scanf.format_from_string template "%s")
                  (List.hd arg_strs)
              else if num_placeholders = 2 && List.length arg_strs = 2 then
                Printf.sprintf
                  (Scanf.format_from_string template "%s%s")
                  (List.nth arg_strs 0)
                  (List.nth arg_strs 1)
              else if num_placeholders = 3 && List.length arg_strs = 3 then
                Printf.sprintf
                  (Scanf.format_from_string template "%s%s%s")
                  (List.nth arg_strs 0)
                  (List.nth arg_strs 1)
                  (List.nth arg_strs 2)
              else
                (* Fallback: treat as function call *)
                template ^ "(" ^ String.concat ", " arg_strs ^ ")"
            in
            Buffer.add_string buf result
        | None ->
            (* Unknown intrinsic - emit as function call *)
            Buffer.add_string buf full_name ;
            Buffer.add_char buf '(' ;
            List.iteri
              (fun i e ->
                if i > 0 then Buffer.add_string buf ", " ;
                gen_expr buf e)
              args ;
            Buffer.add_char buf ')')

(** {1 L-value Generation} *)

let rec gen_lvalue buf = function
  | LVar v -> Buffer.add_string buf v.var_name
  | LArrayElem (arr, idx) ->
      Buffer.add_string buf arr ;
      Buffer.add_char buf '[' ;
      gen_expr buf idx ;
      Buffer.add_char buf ']'
  | LArrayElemExpr (base, idx) ->
      Buffer.add_char buf '(' ;
      gen_expr buf base ;
      Buffer.add_string buf ")[" ;
      gen_expr buf idx ;
      Buffer.add_char buf ']'
  | LRecordField (lv, field) ->
      gen_lvalue buf lv ;
      Buffer.add_char buf '.' ;
      Buffer.add_string buf field

(** {1 Statement Generation} *)

(** Nested indentation level *)
let indent_nested indent = indent ^ "  "

(** Generate match case pattern with variable bindings *)
and gen_match_pattern buf indent scrutinee cname bindings find_constr_types =
  Buffer.add_string buf ("  case " ^ cname ^ ": {\n") ;
  match (bindings, find_constr_types cname) with
  | [var_name], Some [ty] ->
      (* Single payload: access data.Constructor_v *)
      Buffer.add_string buf (indent ^ "    ") ;
      Buffer.add_string buf (metal_type_of_elttype ty) ;
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
          Buffer.add_string buf (metal_type_of_elttype ty) ;
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
      Metal_error.raise_error
        (Metal_error.unsupported_construct
           "pattern"
           "mismatch between pattern bindings and constructor args")

(** Generate variable declaration with initialization *)
and gen_var_decl buf indent v_name v_type init_expr =
  Buffer.add_string buf indent ;
  Buffer.add_string buf (metal_type_of_elttype v_type) ;
  Buffer.add_char buf ' ' ;
  Buffer.add_string buf v_name ;
  Buffer.add_string buf " = " ;
  gen_expr buf init_expr ;
  Buffer.add_string buf ";\n"

(** Generate array declaration *)
and gen_array_decl buf indent v_name elem_ty size memspace =
  Buffer.add_string buf indent ;
  if memspace <> "" then (
    Buffer.add_string buf memspace ;
    Buffer.add_char buf ' ') ;
  Buffer.add_string buf (metal_type_of_elttype elem_ty) ;
  Buffer.add_char buf ' ' ;
  Buffer.add_string buf v_name ;
  Buffer.add_char buf '[' ;
  gen_expr buf size ;
  Buffer.add_string buf "];\n"

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
      gen_stmt buf (indent_nested indent) then_ ;
      Buffer.add_string buf indent ;
      Buffer.add_string buf "}" ;
      match else_opt with
      | None -> Buffer.add_char buf '\n'
      | Some else_ ->
          Buffer.add_string buf " else {\n" ;
          gen_stmt buf (indent_nested indent) else_ ;
          Buffer.add_string buf indent ;
          Buffer.add_string buf "}\n")
  | SWhile (cond, body) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "while (" ;
      gen_expr buf cond ;
      Buffer.add_string buf ") {\n" ;
      gen_stmt buf (indent_nested indent) body ;
      Buffer.add_string buf indent ;
      Buffer.add_string buf "}\n"
  | SFor (v, start, stop, dir, body) ->
      (* OCaml 'for i = a to b' is inclusive, so use <= not < *)
      let op, incr =
        match dir with Upto -> ("<=", "++") | Downto -> (">=", "--")
      in
      Buffer.add_string buf indent ;
      Buffer.add_string buf "for (" ;
      Buffer.add_string buf (metal_type_of_elttype v.var_type) ;
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
      gen_stmt buf (indent_nested indent) body ;
      Buffer.add_string buf indent ;
      Buffer.add_string buf "}\n"
  | SMatch (e, cases) ->
      let scrutinee_buf = Buffer.create 64 in
      gen_expr scrutinee_buf e ;
      let scrutinee = Buffer.contents scrutinee_buf in
      let find_constr_types cname =
        List.find_map
          (fun (_vname, constrs) ->
            List.find_map
              (fun (cn, args) -> if cn = cname then Some args else None)
              constrs)
          !current_variants
      in
      Buffer.add_string buf indent ;
      Buffer.add_string buf "switch (" ;
      Buffer.add_string buf scrutinee ;
      Buffer.add_string buf ".tag) {\n" ;
      List.iter
        (fun (pattern, body) ->
          Buffer.add_string buf indent ;
          (match pattern with
          | PConstr (cname, bindings) ->
              gen_match_pattern
                buf
                indent
                scrutinee
                cname
                bindings
                find_constr_types
          | PWild -> Buffer.add_string buf "  default: {\n") ;
          gen_stmt buf (indent ^ "    ") body ;
          Buffer.add_string buf (indent ^ "    break;\n") ;
          Buffer.add_string buf (indent ^ "  }\n"))
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
      Buffer.add_string buf "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  | SWarpBarrier ->
      (* Metal doesn't have warp-level sync, use sub_group_barrier if available *)
      Buffer.add_string buf indent ;
      Buffer.add_string
        buf
        "sub_group_threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  | SMemFence ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "threadgroup_barrier(mem_flags::mem_device);\n"
  | SNative {gpu; ocaml = _} -> (
      match !current_framework with
      | Some framework ->
          let code = gpu ~framework in
          Buffer.add_string buf indent ;
          Buffer.add_string buf code ;
          if not (String.length code > 0 && code.[String.length code - 1] = '\n')
          then Buffer.add_char buf '\n'
      | None ->
          Metal_error.raise_error
            (Metal_error.no_device_selected
               "SNative (use generate_for_device instead of generate)"))
  | SExpr e ->
      Buffer.add_string buf indent ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n"
  | SLet (v, EArrayCreate (elem_ty, size, mem), body) ->
      let ms = match mem with Shared -> "threadgroup" | _ -> "" in
      gen_array_decl buf indent v.var_name elem_ty size ms ;
      gen_stmt buf indent body
  | SLet (v, e, body) ->
      gen_var_decl buf indent v.var_name v.var_type e ;
      gen_stmt buf indent body
  | SLetMut (v, e, body) ->
      gen_var_decl buf indent v.var_name v.var_type e ;
      gen_stmt buf indent body
  | SPragma (hints, body) ->
      (* Metal uses #pragma for hints *)
      Buffer.add_string buf indent ;
      Buffer.add_string buf "#pragma " ;
      Buffer.add_string buf (String.concat " " hints) ;
      Buffer.add_char buf '\n' ;
      gen_stmt buf indent body
  | SBlock body ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "{\n" ;
      gen_stmt buf (indent_nested indent) body ;
      Buffer.add_string buf indent ;
      Buffer.add_string buf "}\n"

(** {1 Declaration Generation} *)

(** Check if a type is a vector (requires length parameter) *)
let is_vec_type = function TVec _ -> true | _ -> false

(** Generate parameter with Metal buffer attributes, returns next buffer index
*)
let gen_param_metal buf atomic_vars idx = function
  | DParam (v, None) ->
      (* Scalar parameter - wrap in constant buffer *)
      Buffer.add_string buf "constant " ;
      Buffer.add_string buf (metal_type_of_elttype v.var_type) ;
      Buffer.add_string buf " &" ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf " [[buffer(" ;
      Buffer.add_string buf (string_of_int idx) ;
      Buffer.add_string buf ")]]" ;
      (* Add length parameter for vectors *)
      if is_vec_type v.var_type then begin
        Buffer.add_string buf ", constant int &sarek_" ;
        Buffer.add_string buf v.var_name ;
        Buffer.add_string buf "_length [[buffer(" ;
        Buffer.add_string buf (string_of_int (idx + 1)) ;
        Buffer.add_string buf ")]]" ;
        idx + 2
      end
      else idx + 1
  | DParam (v, Some arr) ->
      (* Array with explicit info - always needs length *)
      Buffer.add_string buf (metal_memspace arr.arr_memspace) ;
      Buffer.add_char buf ' ' ;
      (* Use atomic type if this variable is used with atomics *)
      let type_str =
        if List.mem v.var_name atomic_vars then
          metal_atomic_type_of_elttype arr.arr_elttype
        else metal_type_of_elttype arr.arr_elttype
      in
      Buffer.add_string buf type_str ;
      Buffer.add_string buf "* " ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf " [[buffer(" ;
      Buffer.add_string buf (string_of_int idx) ;
      Buffer.add_string buf ")]], constant int &sarek_" ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf "_length [[buffer(" ;
      Buffer.add_string buf (string_of_int (idx + 1)) ;
      Buffer.add_string buf ")]]" ;
      idx + 2
  | DLocal _ | DShared _ ->
      Metal_error.raise_error
        (Metal_error.unsupported_construct "gen_param_metal" "expected DParam")

let gen_param buf = function
  | DParam (v, None) ->
      Buffer.add_string buf (metal_param_type v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      (* Add length parameter for vectors *)
      if is_vec_type v.var_type then begin
        Buffer.add_string buf ", int sarek_" ;
        Buffer.add_string buf v.var_name ;
        Buffer.add_string buf "_length"
      end
  | DParam (v, Some arr) ->
      (* Array with explicit info - always needs length *)
      Buffer.add_string buf (metal_memspace arr.arr_memspace) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf (metal_type_of_elttype arr.arr_elttype) ;
      Buffer.add_string buf "* restrict " ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf ", int sarek_" ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf "_length"
  | DLocal _ | DShared _ ->
      Metal_error.raise_error
        (Metal_error.unsupported_construct "gen_param" "expected DParam")

(** Collect variable names used in atomic operations *)
let rec collect_atomic_vars_expr = function
  | EIntrinsic (_, name, args)
    when String.equal name "atomic_add"
         || String.equal name "atomic_add_int32"
         || String.equal name "atomic_add_global_int32"
         || String.equal name "atomic_sub" -> (
      match args with
      (* When atomic operation has 3 args, first is likely the array identifier *)
      | arr_expr :: _idx :: _value :: _ -> (
          match arr_expr with
          | EVar v -> [v.var_name]
          | EArrayRead (arr, _) -> [arr]
          | _ -> [])
      (* When atomic operation has 2 args, first is the lvalue *)
      | lvalue_expr :: _value :: _ -> (
          match lvalue_expr with
          | EVar v -> [v.var_name]
          | EArrayRead (arr, _) -> [arr]
          | _ -> [])
      | _ -> [])
  | EIntrinsic (_, _, args) -> List.concat_map collect_atomic_vars_expr args
  | EBinop (_, e1, e2) ->
      collect_atomic_vars_expr e1 @ collect_atomic_vars_expr e2
  | EUnop (_, e) -> collect_atomic_vars_expr e
  | EArrayRead (_, idx) -> collect_atomic_vars_expr idx
  | EArrayReadExpr (base, idx) ->
      collect_atomic_vars_expr base @ collect_atomic_vars_expr idx
  | ERecordField (e, _) -> collect_atomic_vars_expr e
  | ECast (_, e) -> collect_atomic_vars_expr e
  | ETuple exprs -> List.concat_map collect_atomic_vars_expr exprs
  | EApp (f, args) ->
      collect_atomic_vars_expr f @ List.concat_map collect_atomic_vars_expr args
  | ERecord (_, fields) ->
      List.concat_map (fun (_, e) -> collect_atomic_vars_expr e) fields
  | EVariant (_, _, args) -> List.concat_map collect_atomic_vars_expr args
  | EArrayCreate (_, size, _) -> collect_atomic_vars_expr size
  | EIf (c, t, e) ->
      collect_atomic_vars_expr c @ collect_atomic_vars_expr t
      @ collect_atomic_vars_expr e
  | EMatch (scrut, cases) ->
      collect_atomic_vars_expr scrut
      @ List.concat_map (fun (_, e) -> collect_atomic_vars_expr e) cases
  | _ -> []

let rec collect_atomic_vars_lvalue = function
  | LArrayElem (_, idx) -> collect_atomic_vars_expr idx
  | LArrayElemExpr (base, idx) ->
      collect_atomic_vars_expr base @ collect_atomic_vars_expr idx
  | LRecordField (lv, _) -> collect_atomic_vars_lvalue lv
  | _ -> []

let rec collect_atomic_vars_stmt = function
  | SAssign (lv, e) ->
      collect_atomic_vars_lvalue lv @ collect_atomic_vars_expr e
  | SSeq stmts -> List.concat_map collect_atomic_vars_stmt stmts
  | SIf (e, s1, Some s2) ->
      collect_atomic_vars_expr e
      @ collect_atomic_vars_stmt s1
      @ collect_atomic_vars_stmt s2
  | SIf (e, s, None) -> collect_atomic_vars_expr e @ collect_atomic_vars_stmt s
  | SWhile (e, s) -> collect_atomic_vars_expr e @ collect_atomic_vars_stmt s
  | SFor (_, start, stop, _, body) ->
      collect_atomic_vars_expr start
      @ collect_atomic_vars_expr stop
      @ collect_atomic_vars_stmt body
  | SMatch (e, cases) ->
      collect_atomic_vars_expr e
      @ List.concat_map (fun (_, s) -> collect_atomic_vars_stmt s) cases
  | SReturn e -> collect_atomic_vars_expr e
  | SExpr e -> collect_atomic_vars_expr e
  | SLet (_, e, s) -> collect_atomic_vars_expr e @ collect_atomic_vars_stmt s
  | SLetMut (_, e, s) -> collect_atomic_vars_expr e @ collect_atomic_vars_stmt s
  | SPragma (_, s) -> collect_atomic_vars_stmt s
  | SBlock s -> collect_atomic_vars_stmt s
  | _ -> []

let gen_local buf indent atomic_vars = function
  | DLocal (v, None) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf (metal_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf ";\n"
  | DLocal (v, Some e) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf (metal_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf " = " ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n"
  | DShared (name, elt, None) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "threadgroup " ;
      (* Use atomic type if this variable is used with atomics *)
      let type_str =
        if List.mem name atomic_vars then metal_atomic_type_of_elttype elt
        else metal_type_of_elttype elt
      in
      Buffer.add_string buf type_str ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf name ;
      Buffer.add_string buf "[];\n"
  | DShared (name, elt, Some size) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "threadgroup " ;
      (* Use atomic type if this variable is used with atomics *)
      let type_str =
        if List.mem name atomic_vars then metal_atomic_type_of_elttype elt
        else metal_type_of_elttype elt
      in
      Buffer.add_string buf type_str ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf name ;
      Buffer.add_char buf '[' ;
      gen_expr buf size ;
      Buffer.add_string buf "];\n"
  | DParam _ ->
      Metal_error.raise_error
        (Metal_error.unsupported_construct
           "gen_local"
           "expected DLocal or DShared")

(** {1 Helper Function Generation} *)

(** Generate a helper function (Metal device function) *)
let gen_helper_func buf (hf : helper_func) =
  (* In Metal, helper functions don't need any special decoration *)
  Buffer.add_string buf (metal_type_of_elttype hf.hf_ret_type) ;
  Buffer.add_char buf ' ' ;
  Buffer.add_string buf hf.hf_name ;
  Buffer.add_char buf '(' ;
  (* Parameters - use metal_helper_param_type *)
  List.iteri
    (fun i (v : var) ->
      if i > 0 then Buffer.add_string buf ", " ;
      Buffer.add_string buf (metal_helper_param_type v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name)
    hf.hf_params ;
  Buffer.add_string buf ") {\n" ;
  (* Body *)
  gen_stmt buf "  " hf.hf_body ;
  Buffer.add_string buf "}\n\n"

(** {1 Kernel Generation} *)

(** Pretty-print Metal source code *)
let pretty_print_metal (source : string) : string =
  let lines = String.split_on_char '\n' source in
  let buf = Buffer.create (String.length source + 1024) in
  let rec process_lines indent = function
    | [] -> ()
    | line :: rest ->
        let trimmed = String.trim line in
        (* Decrease indent for closing braces *)
        let new_indent =
          if String.length trimmed > 0 && trimmed.[0] = '}' then
            max 0 (indent - 2)
          else indent
        in
        (* Add indentation *)
        if String.length trimmed > 0 then (
          for _ = 1 to new_indent do
            Buffer.add_char buf ' '
          done ;
          Buffer.add_string buf trimmed ;
          Buffer.add_char buf '\n')
        else Buffer.add_char buf '\n' ;
        (* Increase indent for opening braces *)
        let next_indent =
          if
            String.length trimmed > 0
            && trimmed.[String.length trimmed - 1] = '{'
          then new_indent + 2
          else new_indent
        in
        process_lines next_indent rest
  in
  process_lines 0 lines ;
  Buffer.contents buf

(** Generate complete Metal source for a kernel *)
let generate (k : kernel) : string =
  let buf = Buffer.create 4096 in

  (* Collect variables used with atomic operations *)
  let atomic_vars = collect_atomic_vars_stmt k.kern_body in

  (* Metal header *)
  Buffer.add_string buf "#include <metal_stdlib>\n" ;
  Buffer.add_string buf "using namespace metal;\n\n" ;

  (* Generate helper functions before kernel *)
  List.iter (gen_helper_func buf) k.kern_funcs ;

  (* Kernel signature *)
  Buffer.add_string buf "kernel void " ;
  Buffer.add_string buf k.kern_name ;
  Buffer.add_char buf '(' ;

  (* Parameters with buffer attributes *)
  let buffer_idx = ref 0 in
  List.iteri
    (fun i p ->
      if i > 0 then Buffer.add_string buf ", " ;
      buffer_idx := gen_param_metal buf atomic_vars !buffer_idx p)
    k.kern_params ;

  (* Add thread position parameters *)
  if !buffer_idx > 0 then Buffer.add_string buf ", " ;
  Buffer.add_string buf "\n  uint3 __metal_gid [[thread_position_in_grid]],\n" ;
  Buffer.add_string
    buf
    "  uint3 __metal_tid [[thread_position_in_threadgroup]],\n" ;
  Buffer.add_string
    buf
    "  uint3 __metal_bid [[threadgroup_position_in_grid]],\n" ;
  Buffer.add_string buf "  uint3 __metal_tpg [[threads_per_threadgroup]],\n" ;
  Buffer.add_string buf "  uint3 __metal_num_groups [[threadgroups_per_grid]]" ;

  Buffer.add_string buf ") {\n" ;

  (* Local declarations *)
  List.iter (gen_local buf "  " atomic_vars) k.kern_locals ;

  (* Body *)
  gen_stmt buf "  " k.kern_body ;

  (* Close kernel *)
  Buffer.add_string buf "}\n" ;

  pretty_print_metal (Buffer.contents buf)

(** Generate complete Metal source with device context for SNative. Extracts
    only the framework string from [device]. *)
let generate_for_device ~(device : Spoc_core.Device.t) (k : kernel) : string =
  current_framework := Some device.Spoc_core.Device.framework ;
  let result = generate k in
  current_framework := None ;
  result

(** Generate variant type definition for Metal *)
let gen_variant_def buf v =
  Sarek_ir_codegen.gen_variant_def
    ~type_of_elttype:metal_type_of_elttype
    ~constructor_prefix:"static inline"
    buf
    v

(** Generate Metal source with custom type definitions *)
let generate_with_types ~(types : (string * (string * elttype) list) list)
    (k : kernel) : string =
  (* Set current_variants for SMatch binding extraction *)
  current_variants := k.kern_variants ;
  let buf = Buffer.create 4096 in

  (* Collect variables used with atomic operations *)
  let atomic_vars = collect_atomic_vars_stmt k.kern_body in

  (* Metal header *)
  Buffer.add_string buf "#include <metal_stdlib>\n" ;
  Buffer.add_string buf "using namespace metal;\n\n" ;

  (* Variant type definitions first (may be needed by records).
     Previously omitted here, unlike the CUDA/OpenCL backends, so Metal kernels
     using variant types emitted no typedef. Emitting them keeps Metal consistent
     with the other C-family backends. *)
  List.iter (gen_variant_def buf) k.kern_variants ;

  (* Record type definitions *)
  List.iter
    (fun (name, fields) ->
      Buffer.add_string buf "typedef struct {\n" ;
      List.iter
        (fun (fname, ftype) ->
          Buffer.add_string buf "  " ;
          Buffer.add_string buf (metal_type_of_elttype ftype) ;
          Buffer.add_char buf ' ' ;
          Buffer.add_string buf fname ;
          Buffer.add_string buf ";\n")
        fields ;
      Buffer.add_string buf "} " ;
      Buffer.add_string buf (mangle_name name) ;
      Buffer.add_string buf ";\n\n")
    types ;

  (* Generate helper functions before kernel *)
  List.iter (gen_helper_func buf) k.kern_funcs ;

  (* Kernel signature *)
  Buffer.add_string buf "kernel void " ;
  Buffer.add_string buf k.kern_name ;
  Buffer.add_char buf '(' ;

  (* Parameters with buffer attributes *)
  let buffer_idx = ref 0 in
  List.iteri
    (fun i p ->
      if i > 0 then Buffer.add_string buf ", " ;
      buffer_idx := gen_param_metal buf atomic_vars !buffer_idx p)
    k.kern_params ;

  (* Add thread position parameters *)
  if !buffer_idx > 0 then Buffer.add_string buf ", " ;
  Buffer.add_string buf "\n  uint3 __metal_gid [[thread_position_in_grid]],\n" ;
  Buffer.add_string
    buf
    "  uint3 __metal_tid [[thread_position_in_threadgroup]],\n" ;
  Buffer.add_string
    buf
    "  uint3 __metal_bid [[threadgroup_position_in_grid]],\n" ;
  Buffer.add_string buf "  uint3 __metal_tpg [[threads_per_threadgroup]],\n" ;
  Buffer.add_string buf "  uint3 __metal_num_groups [[threadgroups_per_grid]]" ;

  Buffer.add_string buf ") {\n" ;

  (* Local declarations *)
  List.iter (gen_local buf "  " atomic_vars) k.kern_locals ;

  (* Body *)
  gen_stmt buf "  " k.kern_body ;

  (* Close kernel *)
  Buffer.add_string buf "}\n" ;

  (* Pretty-print the generated code *)
  pretty_print_metal (Buffer.contents buf)

(** Generate Metal source with double precision (Metal supports FP64 natively)
*)
let generate_with_fp64 (k : kernel) : string = generate k
