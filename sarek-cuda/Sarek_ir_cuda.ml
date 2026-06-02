(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_ir_cuda - CUDA Code Generation from Sarek IR
 *
 * Generates CUDA C source code from Sarek_ir.kernel.
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

(** Map Sarek IR element type to CUDA C type string *)
let rec cuda_type_of_elttype = function
  | TInt32 -> "int"
  | TInt64 -> "long long"
  | TFloat32 -> "float"
  | TFloat64 -> "double"
  | TBool -> "int"
  | TUnit -> "void"
  | TRecord (name, _) -> mangle_name name
  | TVariant (name, _) -> mangle_name name
  | TArray (elt, _) -> cuda_type_of_elttype elt ^ "*"
  | TVec elt -> cuda_type_of_elttype elt ^ "*"

(** Map Sarek IR element type to CUDA C type for kernel parameters *)
let cuda_param_type = function
  | TVec elt -> cuda_type_of_elttype elt ^ "* __restrict__"
  | TArray (elt, _) -> cuda_type_of_elttype elt ^ "*"
  | t -> cuda_type_of_elttype t

(** {1 Thread Intrinsics} *)

let cuda_thread_intrinsic name =
  match name with
  (* Support both idx and id naming conventions *)
  | "thread_id_x" | "thread_idx_x" -> "threadIdx.x"
  | "thread_id_y" | "thread_idx_y" -> "threadIdx.y"
  | "thread_id_z" | "thread_idx_z" -> "threadIdx.z"
  | "block_id_x" | "block_idx_x" -> "blockIdx.x"
  | "block_id_y" | "block_idx_y" -> "blockIdx.y"
  | "block_id_z" | "block_idx_z" -> "blockIdx.z"
  | "block_dim_x" -> "blockDim.x"
  | "block_dim_y" -> "blockDim.y"
  | "block_dim_z" -> "blockDim.z"
  | "grid_dim_x" -> "gridDim.x"
  | "grid_dim_y" -> "gridDim.y"
  | "grid_dim_z" -> "gridDim.z"
  | "global_thread_id" | "global_idx" | "global_idx_x" ->
      "(threadIdx.x + blockIdx.x * blockDim.x)"
  | "global_idx_y" -> "(threadIdx.y + blockIdx.y * blockDim.y)"
  | "global_idx_z" -> "(threadIdx.z + blockIdx.z * blockDim.z)"
  | "global_size" -> "(blockDim.x * gridDim.x)"
  | name -> Cuda_error.raise_error (Cuda_error.unknown_intrinsic name)

(** {1 Expression Generation} *)

let rec gen_expr buf = function
  | EConst (CInt32 n) -> Buffer.add_string buf (Int32.to_string n)
  | EConst (CInt64 n) -> Buffer.add_string buf (Int64.to_string n ^ "LL")
  | EConst (CFloat32 f) ->
      let s = Printf.sprintf "%.17g" f in
      (* Ensure decimal point for valid C/CUDA float literal *)
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
      Buffer.add_string buf (cuda_type_of_elttype ty) ;
      Buffer.add_char buf ')' ;
      gen_expr buf e
  | ETuple exprs ->
      (* Tuples become struct literals in CUDA *)
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
      (* CUDA doesn't support designated initializers in compound literals.
         Use constructor-style initialization with positional values. *)
      Buffer.add_string buf ("(" ^ mangle_name name ^ "){") ;
      List.iteri
        (fun i (_f, e) ->
          if i > 0 then Buffer.add_string buf ", " ;
          gen_expr buf e)
        fields ;
      Buffer.add_string buf "}"
  | EVariant (_, constr, []) -> Buffer.add_string buf constr
  | EVariant (type_name, constr, args) ->
      Buffer.add_string buf ("make_" ^ type_name ^ "_" ^ constr ^ "(") ;
      List.iteri
        (fun i e ->
          if i > 0 then Buffer.add_string buf ", " ;
          gen_expr buf e)
        args ;
      Buffer.add_char buf ')'
  | EArrayLen arr -> Buffer.add_string buf ("sarek_" ^ arr ^ "_length")
  | EArrayCreate _ ->
      Cuda_error.raise_error
        (Cuda_error.unsupported_construct
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
      Cuda_error.raise_error
        (Cuda_error.unsupported_construct "EMatch" "empty match expression")
  | EMatch (_, [(_, body)]) ->
      (* Single case - just emit the body *)
      gen_expr buf body
  | EMatch (e, cases) ->
      (* Multi-case match as nested ternary - check tag field *)
      let rec gen_cases = function
        | [] ->
            Cuda_error.raise_error
              (Cuda_error.unsupported_construct
                 "EMatch"
                 "empty match cases after filtering")
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
  (* Check for thread intrinsics first *)
  let full_name =
    match path with [] -> name | _ -> String.concat "." path ^ "." ^ name
  in
  (* For path-qualified intrinsics, query the pure registry first.
     This handles Float32.sin -> sinf on CUDA, sin on others, etc. *)
  let pure_registry_hit =
    match path with
    | [] -> None
    | _ -> (
        let framework = Option.value ~default:"CUDA" !current_framework in
        match
          Sarek_pure_registry.fun_device_template ~module_path:path name
        with
        | Some f -> Some (f ~framework)
        | None -> None)
  in
  match pure_registry_hit with
  | Some device_name ->
      Buffer.add_string buf device_name ;
      Buffer.add_char buf '(' ;
      List.iteri
        (fun i e ->
          if i > 0 then Buffer.add_string buf ", " ;
          gen_expr buf e)
        args ;
      Buffer.add_char buf ')'
  | None -> (
      if
        (* Try thread intrinsics *)
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
      then Buffer.add_string buf (cuda_thread_intrinsic name)
      else
        (* Standard math intrinsics *)
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
        | "block_barrier" -> Buffer.add_string buf "__syncthreads()"
        | "atomic_add" | "atomic_add_int32" | "atomic_add_global_int32" ->
            Buffer.add_string buf "atomicAdd(" ;
            (match args with
            | [addr; value] ->
                Buffer.add_char buf '&' ;
                gen_expr buf addr ;
                Buffer.add_string buf ", " ;
                gen_expr buf value
            | [arr; idx; value] ->
                (* Array element atomic: atomicAdd(&arr[idx], value) *)
                Buffer.add_char buf '&' ;
                gen_expr buf arr ;
                Buffer.add_char buf '[' ;
                gen_expr buf idx ;
                Buffer.add_string buf "], " ;
                gen_expr buf value
            | _ ->
                Cuda_error.raise_error
                  (Cuda_error.invalid_arg_count
                     "atomic_add"
                     3
                     (List.length args))) ;
            Buffer.add_char buf ')'
        | "atomic_sub" ->
            Buffer.add_string buf "atomicSub(" ;
            (match args with
            | [addr; value] ->
                Buffer.add_char buf '&' ;
                gen_expr buf addr ;
                Buffer.add_string buf ", " ;
                gen_expr buf value
            | _ ->
                Cuda_error.raise_error
                  (Cuda_error.invalid_arg_count
                     "atomic_sub"
                     2
                     (List.length args))) ;
            Buffer.add_char buf ')'
        | "atomic_min" ->
            Buffer.add_string buf "atomicMin(" ;
            (match args with
            | [addr; value] ->
                Buffer.add_char buf '&' ;
                gen_expr buf addr ;
                Buffer.add_string buf ", " ;
                gen_expr buf value
            | _ ->
                Cuda_error.raise_error
                  (Cuda_error.invalid_arg_count
                     "atomic_min"
                     2
                     (List.length args))) ;
            Buffer.add_char buf ')'
        | "atomic_max" ->
            Buffer.add_string buf "atomicMax(" ;
            (match args with
            | [addr; value] ->
                Buffer.add_char buf '&' ;
                gen_expr buf addr ;
                Buffer.add_string buf ", " ;
                gen_expr buf value
            | _ ->
                Cuda_error.raise_error
                  (Cuda_error.invalid_arg_count
                     "atomic_max"
                     2
                     (List.length args))) ;
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
                    match arg_strs with
                    | [arg] ->
                        Printf.sprintf
                          (Scanf.format_from_string template "%s")
                          arg
                    | _ ->
                        (* This should never happen due to length check above *)
                        Cuda_error.raise_error
                          (Cuda_error.type_error
                             "intrinsic template"
                             "1 placeholder, 1 argument"
                             "unexpected list state")
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
                Buffer.add_char buf ')'))

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

let rec gen_stmt buf indent = function
  | SEmpty -> ()
  | SSeq stmts -> List.iter (gen_stmt buf indent) stmts
  | SAssign (lv, e) -> (
      (* Special case: CUDA doesn't support compound literals in assignments.
         For record assignments, generate field-by-field initialization. *)
      match e with
      | ERecord (_, fields) -> gen_record_assign buf indent lv fields
      | _ ->
          Buffer.add_string buf indent ;
          gen_lvalue buf lv ;
          Buffer.add_string buf " = " ;
          gen_expr buf e ;
          Buffer.add_string buf ";\n")
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
      Buffer.add_string buf (cuda_type_of_elttype v.var_type) ;
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
      let scrutinee_buf = Buffer.create 64 in
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
      Buffer.add_string buf "__syncthreads();\n"
  | SWarpBarrier ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "__syncwarp();\n"
  | SMemFence ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "__threadfence();\n"
  | SNative {gpu; ocaml = _} -> (
      match !current_framework with
      | Some framework ->
          let code = gpu ~framework in
          Buffer.add_string buf indent ;
          Buffer.add_string buf code ;
          if not (String.length code > 0 && code.[String.length code - 1] = '\n')
          then Buffer.add_char buf '\n'
      | None ->
          Cuda_error.raise_error
            (Cuda_error.unsupported_construct
               "SNative"
               "requires device context - use generate_for_device instead of \
                generate"))
  | SExpr e ->
      Buffer.add_string buf indent ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n"
  | SLet (v, EArrayCreate (elem_ty, size, mem), body) ->
      (* Array declaration: type arr[size]; *)
      Buffer.add_string buf indent ;
      gen_array_decl buf v elem_ty size mem ;
      gen_stmt buf indent body
  | SLet (v, e, body) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf (cuda_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf " = " ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n" ;
      gen_stmt buf indent body
  | SLetMut (v, e, body) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf (cuda_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf " = " ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n" ;
      gen_stmt buf indent body
  | SPragma (hints, body) ->
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

(** Helper: Generate record field assignment *)
and gen_record_assign buf indent lv fields =
  List.iter
    (fun (fname, fexpr) ->
      Buffer.add_string buf indent ;
      gen_lvalue buf lv ;
      Buffer.add_char buf '.' ;
      Buffer.add_string buf fname ;
      Buffer.add_string buf " = " ;
      gen_expr buf fexpr ;
      Buffer.add_string buf ";\n")
    fields

(** Helper: Generate switch case for pattern match *)
and gen_match_case buf indent scrutinee pattern body =
  Buffer.add_string buf indent ;
  (match pattern with
  | PConstr (cname, bindings) -> (
      (* Lookup constructor types from current_variants *)
      let find_constr_types cname =
        List.find_map
          (fun (_vname, constrs) ->
            List.find_map
              (fun (cn, args) -> if cn = cname then Some args else None)
              constrs)
          !current_variants
      in
      Buffer.add_string buf ("  case " ^ cname ^ ": {\n") ;
      (* Generate bindings: extract payload from scrutinee *)
      match (bindings, find_constr_types cname) with
      | [var_name], Some [ty] ->
          (* Single payload: access data.Constructor_v *)
          Buffer.add_string buf (indent ^ "    ") ;
          Buffer.add_string buf (cuda_type_of_elttype ty) ;
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
              Buffer.add_string buf (cuda_type_of_elttype ty) ;
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
          Cuda_error.raise_error
            (Cuda_error.type_error
               "pattern match"
               "matching bindings and constructor"
               "mismatched bindings/args"))
  | PWild -> Buffer.add_string buf "  default: {\n") ;
  gen_stmt buf (indent ^ "    ") body ;
  Buffer.add_string buf (indent ^ "    break;\n") ;
  Buffer.add_string buf (indent ^ "  }\n")

(** Helper: Generate array declaration for SLet with EArrayCreate *)
and gen_array_decl buf v elem_ty size mem =
  (match mem with Shared -> Buffer.add_string buf "__shared__ " | _ -> ()) ;
  Buffer.add_string buf (cuda_type_of_elttype elem_ty) ;
  Buffer.add_string buf " " ;
  Buffer.add_string buf v.var_name ;
  Buffer.add_char buf '[' ;
  gen_expr buf size ;
  Buffer.add_string buf "];\n"

(** {1 Declaration Generation} *)

(** Check if a type is a vector (requires length parameter) *)
let is_vec_type = function TVec _ -> true | _ -> false

let gen_param buf = function
  | DParam (v, None) ->
      Buffer.add_string buf (cuda_param_type v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      (* Add length parameter for vectors *)
      if is_vec_type v.var_type then begin
        Buffer.add_string buf ", int sarek_" ;
        Buffer.add_string buf v.var_name ;
        Buffer.add_string buf "_length"
      end
  | DParam (v, Some _arr) ->
      (* Array with explicit info - always needs length *)
      Buffer.add_string buf (cuda_param_type v.var_type) ;
      Buffer.add_string buf " " ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf ", int sarek_" ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf "_length"
  | DLocal _ | DShared _ ->
      Cuda_error.raise_error
        (Cuda_error.invalid_memory_space "gen_param" "DLocal or DShared")

let gen_local buf indent = function
  | DLocal (v, None) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf (cuda_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf ";\n"
  | DLocal (v, Some e) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf (cuda_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf " = " ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n"
  | DShared (name, elt, None) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "__shared__ " ;
      Buffer.add_string buf (cuda_type_of_elttype elt) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf name ;
      Buffer.add_string buf "[];\n"
  | DShared (name, elt, Some size) ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "__shared__ " ;
      Buffer.add_string buf (cuda_type_of_elttype elt) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf name ;
      Buffer.add_char buf '[' ;
      gen_expr buf size ;
      Buffer.add_string buf "];\n"
  | DParam _ ->
      Cuda_error.raise_error
        (Cuda_error.invalid_memory_space "gen_local" "DParam")

(** {1 Helper Function Generation} *)

(** Generate a device helper function *)
let gen_helper_func buf (hf : helper_func) =
  (* __device__ ret_type name(params) { body } *)
  Buffer.add_string buf "__device__ " ;
  Buffer.add_string buf (cuda_type_of_elttype hf.hf_ret_type) ;
  Buffer.add_char buf ' ' ;
  Buffer.add_string buf hf.hf_name ;
  Buffer.add_char buf '(' ;
  (* Parameters *)
  List.iteri
    (fun i (v : var) ->
      if i > 0 then Buffer.add_string buf ", " ;
      Buffer.add_string buf (cuda_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name)
    hf.hf_params ;
  Buffer.add_string buf ") {\n" ;
  (* Body *)
  gen_stmt buf "  " hf.hf_body ;
  Buffer.add_string buf "}\n\n"

(** {1 Kernel Generation} *)

(** Generate the CUDA kernel header *)
let cuda_header = {|
extern "C" {
|}

(** Generate complete CUDA source for a kernel *)
let generate (k : kernel) : string =
  let buf = Buffer.create 4096 in

  (* Header *)
  Buffer.add_string buf cuda_header ;

  (* Generate helper functions before kernel *)
  List.iter (gen_helper_func buf) k.kern_funcs ;

  (* Kernel signature *)
  Buffer.add_string buf "__global__ void " ;
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

  (* Close extern "C" *)
  Buffer.add_string buf "}\n" ;

  Buffer.contents buf

(** Generate complete CUDA source with device context for SNative. Extracts only
    the framework string from [device]; [current_device] is no longer used. *)
let generate_for_device ~(device : Spoc_core.Device.t) (k : kernel) : string =
  current_framework := Some device.Spoc_core.Device.framework ;
  let result = generate k in
  current_framework := None ;
  result

(** Generate CUDA variant type definition *)
let gen_variant_def buf v =
  Sarek_ir_codegen.gen_variant_def
    ~type_of_elttype:cuda_type_of_elttype
    ~constructor_prefix:"__device__ __host__ inline"
    buf
    v

(** Generate CUDA source with custom type definitions *)
let generate_with_types ~(types : (string * (string * elttype) list) list)
    (k : kernel) : string =
  (* Set current_variants for SMatch binding extraction *)
  current_variants := k.kern_variants ;
  let buf = Buffer.create 4096 in

  (* Header *)
  Buffer.add_string buf cuda_header ;

  (* Variant type definitions first (may be needed by records) *)
  List.iter (gen_variant_def buf) k.kern_variants ;

  (* Record type definitions *)
  List.iter
    (fun (name, fields) ->
      Buffer.add_string buf "typedef struct {\n" ;
      List.iter
        (fun (fname, ftype) ->
          Buffer.add_string buf "  " ;
          Buffer.add_string buf (cuda_type_of_elttype ftype) ;
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
  Buffer.add_string buf "__global__ void " ;
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

  (* Close extern "C" *)
  Buffer.add_string buf "}\n" ;

  Buffer.contents buf
