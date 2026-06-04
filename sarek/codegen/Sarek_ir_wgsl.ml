(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_ir_wgsl - WGSL Compute Shader Generation from Sarek IR
 *
 * Generates WebGPU Shading Language (WGSL) compute shader source code from
 * Sarek_ir.kernel. The output targets WebGPU compute pipelines.
 *
 * Features:
 * - Direct generation from clean IR
 * - Storage buffer bindings for vector parameters (array<T>)
 * - Uniform struct for scalar parameters (struct Params)
 * - Strict type-conversion enforcement (no implicit int<->float)
 * - Float64 / f64 unsupported — returns structured error
 * - Workgroup size from kernel block hint
 ******************************************************************************)

open Sarek_ir_types

(** Local error module — tagged as "WebGPU" in error messages. *)
module Codegen_error = Sarek_backend_error.Backend_error.Make (struct
  let name = "WebGPU"
end)

(** Current kernel's variant definitions (set during generate) *)
let current_variants : (string * (string * elttype list) list) list ref = ref []

(** Current framework name — mirrors the other generators so [set_framework] in
    Sarek_transpile can set it, and the pure registry resolves Float32 math with
    framework="WGSL" (falls through to the generic [sin] spelling). *)
let current_framework : string option ref = ref None

(** {1 Type Mapping} *)

let mangle_name = Sarek_ir_codegen.mangle_name

(** WGSL reserved keywords that cannot be used as identifiers *)
let wgsl_reserved_keywords =
  [
    (* Types *)
    "bool";
    "f32";
    "f16";
    "i32";
    "u32";
    "u64";
    "i64";
    "vec2";
    "vec3";
    "vec4";
    "mat2x2";
    "mat3x3";
    "mat4x4";
    "array";
    "atomic";
    "ptr";
    "sampler";
    "texture_2d";
    (* Storage qualifiers *)
    "var";
    "let";
    "const";
    "override";
    (* Control flow *)
    "if";
    "else";
    "for";
    "while";
    "loop";
    "break";
    "continue";
    "return";
    "switch";
    "case";
    "default";
    "fallthrough";
    "discard";
    (* Functions / structure *)
    "fn";
    "struct";
    "type";
    (* Address spaces *)
    "storage";
    "uniform";
    "workgroup";
    "private";
    "function";
    "read";
    "write";
    "read_write";
    (* Built-in decorators / attributes *)
    "compute";
    "vertex";
    "fragment";
    "builtin";
    "location";
    "group";
    "binding";
    "workgroup_size";
    (* Built-in values *)
    "true";
    "false";
    (* Entry point name — avoid shadowing *)
    "main";
    (* Params struct name we emit *)
    "Params";
    "params";
  ]

(** Escape reserved WGSL keywords by adding 'v' suffix. *)
let escape_wgsl_name name =
  if List.mem name wgsl_reserved_keywords then name ^ "v" else name

(** Map Sarek IR element type to WGSL type string. Float64 (f64) is not
    supported in WebGPU — callers must check for TFloat64 before reaching this
    function and raise [Codegen_error.unsupported_construct]. *)
let rec wgsl_type_of_elttype = function
  | TInt32 -> "i32"
  | TInt64 ->
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "i64"
           "WGSL: 64-bit integers unsupported in core WebGPU")
  | TFloat32 -> "f32"
  | TFloat64 ->
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "f64"
           "WGSL: f64 unsupported — WebGPU has no float64 type")
  | TBool -> "bool"
  | TUnit -> "/* unit */"
  | TRecord (name, _) -> mangle_name name
  | TVariant (name, _) -> mangle_name name
  | TArray (elt, _) -> wgsl_type_of_elttype elt
  | TVec elt -> wgsl_type_of_elttype elt

(** Check whether an elttype (recursively) uses Float64. *)
let rec has_float64 = function
  | TFloat64 -> true
  | TArray (t, _) | TVec t -> has_float64 t
  | TRecord (_, fields) -> List.exists (fun (_, t) -> has_float64 t) fields
  | TVariant (_, constrs) ->
      List.exists (fun (_, ts) -> List.exists has_float64 ts) constrs
  | TInt32 | TInt64 | TFloat32 | TBool | TUnit -> false

(** {1 Thread Intrinsics}

    WGSL uses three distinct builtins:
    - [local_invocation_id] (lid) — thread within workgroup
    - [workgroup_id] (wid) — workgroup index in the dispatch grid
    - [global_invocation_id] (gid) — globally unique thread index

    All are [vec3<u32>]; we cast to i32 to match the IR's i32 type for thread
    ids. The entry point declares all three builtins; unused ones are harmless
    (WGSL permits unused builtin params). *)
let wgsl_thread_intrinsic = function
  | "thread_id_x" | "thread_idx_x" -> "i32(lid.x)"
  | "thread_id_y" | "thread_idx_y" -> "i32(lid.y)"
  | "thread_id_z" | "thread_idx_z" -> "i32(lid.z)"
  | "block_id_x" | "block_idx_x" -> "i32(wid.x)"
  | "block_id_y" | "block_idx_y" -> "i32(wid.y)"
  | "block_id_z" | "block_idx_z" -> "i32(wid.z)"
  | "block_dim_x" -> "256i"
  | "block_dim_y" -> "1i"
  | "block_dim_z" -> "1i"
  | "grid_dim_x" -> "i32(nwg.x)"
  | "grid_dim_y" -> "i32(nwg.y)"
  | "grid_dim_z" -> "i32(nwg.z)"
  | "global_thread_id" | "global_idx" | "global_idx_x" -> "i32(gid.x)"
  | "global_idx_y" -> "i32(gid.y)"
  | "global_idx_z" -> "i32(gid.z)"
  | "global_size" -> "0i"
  | name -> Codegen_error.raise_error (Codegen_error.unknown_intrinsic name)

(** {1 Expression Generation} *)

(** Names of scalar kernel params — accessed as [params.<name>] in WGSL. *)
let scalar_param_names : string list ref = ref []

let rec gen_expr buf = function
  | EConst (CInt32 n) -> Buffer.add_string buf (Int32.to_string n ^ "i")
  | EConst (CInt64 _) ->
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "i64 literal"
           "WGSL: 64-bit integers unsupported in core WebGPU")
  | EConst (CFloat32 f) ->
      let s = Printf.sprintf "%.17g" f in
      let s =
        if String.contains s '.' || String.contains s 'e' then s else s ^ ".0"
      in
      Buffer.add_string buf (s ^ "f")
  | EConst (CFloat64 _) ->
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "f64 literal"
           "WGSL: f64 unsupported — WebGPU has no float64 type")
  | EConst (CBool true) -> Buffer.add_string buf "true"
  | EConst (CBool false) -> Buffer.add_string buf "false"
  | EConst CUnit -> Buffer.add_string buf "/* unit */"
  | EVar v ->
      let vn = escape_wgsl_name v.var_name in
      if List.mem v.var_name !scalar_param_names then begin
        Buffer.add_string buf "params." ;
        Buffer.add_string buf vn
      end
      else Buffer.add_string buf vn
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
      Buffer.add_string buf (escape_wgsl_name arr) ;
      Buffer.add_char buf '[' ;
      gen_expr buf idx ;
      Buffer.add_char buf ']'
  | EArrayReadExpr (base, idx) ->
      Buffer.add_char buf '(' ;
      gen_expr buf base ;
      Buffer.add_char buf ')' ;
      Buffer.add_char buf '[' ;
      gen_expr buf idx ;
      Buffer.add_char buf ']'
  | ERecordField (e, field) ->
      gen_expr buf e ;
      Buffer.add_char buf '.' ;
      Buffer.add_string buf field
  | EIntrinsic (path, name, args) -> gen_intrinsic buf path name args
  | ECast (ty, e) ->
      Buffer.add_string buf (wgsl_type_of_elttype ty) ;
      Buffer.add_char buf '(' ;
      gen_expr buf e ;
      Buffer.add_char buf ')'
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
      Buffer.add_string buf (mangle_name name ^ "(") ;
      List.iteri
        (fun i (_, e) ->
          if i > 0 then Buffer.add_string buf ", " ;
          gen_expr buf e)
        fields ;
      Buffer.add_char buf ')'
  | EVariant (type_name, constr, args) ->
      Buffer.add_string
        buf
        ("make_" ^ mangle_name type_name ^ "_" ^ constr ^ "(") ;
      List.iteri
        (fun i e ->
          if i > 0 then Buffer.add_string buf ", " ;
          gen_expr buf e)
        args ;
      Buffer.add_char buf ')'
  | EArrayLen arr ->
      Buffer.add_string buf ("params.sarek_" ^ escape_wgsl_name arr ^ "_length")
  | EArrayCreate _ ->
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "EArrayCreate"
           "should be handled in gen_stmt SLet")
  | EIf (cond, then_, else_) ->
      (* WGSL has no ternary operator — use select(false_val, true_val, cond) *)
      Buffer.add_string buf "select(" ;
      gen_expr buf else_ ;
      Buffer.add_string buf ", " ;
      gen_expr buf then_ ;
      Buffer.add_string buf ", " ;
      gen_expr buf cond ;
      Buffer.add_char buf ')'
  | EMatch (_, []) ->
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct "match" "empty match expression")
  | EMatch (_, [(_, body)]) -> gen_expr buf body
  | EMatch (e, cases) ->
      (* Nest select() calls: select(else_result, then_result, condition) *)
      let rec gen_cases = function
        | [] ->
            Codegen_error.raise_error
              (Codegen_error.unsupported_construct "match" "empty match cases")
        | [(_, body)] -> gen_expr buf body
        | (pat, body) :: rest ->
            Buffer.add_string buf "select(" ;
            (* false branch comes first in select() *)
            let rest_buf = Buffer.create 64 in
            gen_cases_into rest_buf rest ;
            Buffer.add_buffer buf rest_buf ;
            Buffer.add_string buf ", " ;
            gen_expr buf body ;
            Buffer.add_string buf ", " ;
            (match pat with
            | PConstr (name, _) ->
                Buffer.add_char buf '(' ;
                gen_expr buf e ;
                Buffer.add_string buf (".tag == " ^ name ^ ")")
            | PWild -> Buffer.add_string buf "true") ;
            Buffer.add_char buf ')'
      and gen_cases_into buf2 = function
        | [] ->
            Codegen_error.raise_error
              (Codegen_error.unsupported_construct "match" "empty match cases")
        | [(_, body)] -> gen_expr buf2 body
        | (pat, body) :: rest ->
            Buffer.add_string buf2 "select(" ;
            let rest_buf = Buffer.create 64 in
            gen_cases_into rest_buf rest ;
            Buffer.add_buffer buf2 rest_buf ;
            Buffer.add_string buf2 ", " ;
            gen_expr buf2 body ;
            Buffer.add_string buf2 ", " ;
            (match pat with
            | PConstr (name, _) ->
                Buffer.add_char buf2 '(' ;
                gen_expr buf2 e ;
                Buffer.add_string buf2 (".tag == " ^ name ^ ")")
            | PWild -> Buffer.add_string buf2 "true") ;
            Buffer.add_char buf2 ')'
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
  let framework = Option.value ~default:"WGSL" !current_framework in
  let pure_registry_hit =
    match path with
    | [] -> None
    | _ -> (
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
      then Buffer.add_string buf (wgsl_thread_intrinsic name)
      else
        match name with
        | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh"
        | "tanh" | "exp" | "exp2" | "log" | "log2" | "sqrt" | "floor" | "ceil"
        | "round" | "trunc" | "abs" ->
            Buffer.add_string buf name ;
            Buffer.add_char buf '(' ;
            List.iteri
              (fun i e ->
                if i > 0 then Buffer.add_string buf ", " ;
                gen_expr buf e)
              args ;
            Buffer.add_char buf ')'
        | "fabs" ->
            Buffer.add_string buf "abs" ;
            Buffer.add_char buf '(' ;
            List.iteri
              (fun i e ->
                if i > 0 then Buffer.add_string buf ", " ;
                gen_expr buf e)
              args ;
            Buffer.add_char buf ')'
        | "rsqrt" ->
            Buffer.add_string buf "(1.0f / sqrt(" ;
            (match args with
            | [e] -> gen_expr buf e
            | _ ->
                List.iteri
                  (fun i e ->
                    if i > 0 then Buffer.add_string buf ", " ;
                    gen_expr buf e)
                  args) ;
            Buffer.add_string buf "))"
        | "atan2" | "pow" | "min" | "max" ->
            Buffer.add_string buf name ;
            Buffer.add_char buf '(' ;
            List.iteri
              (fun i e ->
                if i > 0 then Buffer.add_string buf ", " ;
                gen_expr buf e)
              args ;
            Buffer.add_char buf ')'
        | "fma" ->
            Buffer.add_string buf "fma" ;
            Buffer.add_char buf '(' ;
            List.iteri
              (fun i e ->
                if i > 0 then Buffer.add_string buf ", " ;
                gen_expr buf e)
              args ;
            Buffer.add_char buf ')'
        | "block_barrier" -> Buffer.add_string buf "workgroupBarrier()"
        | "atomic_add" | "atomic_add_int32" | "atomic_add_global_int32" ->
            Buffer.add_string buf "atomicAdd(" ;
            (match args with
            | [addr; value] ->
                gen_expr buf addr ;
                Buffer.add_string buf ", " ;
                gen_expr buf value
            | [arr; idx; value] ->
                gen_expr buf arr ;
                Buffer.add_char buf '[' ;
                gen_expr buf idx ;
                Buffer.add_string buf "], " ;
                gen_expr buf value
            | args ->
                Codegen_error.raise_error
                  (Codegen_error.invalid_arg_count
                     "atomic_add"
                     2
                     (List.length args))) ;
            Buffer.add_char buf ')'
        | "atomic_min" ->
            Buffer.add_string buf "atomicMin(" ;
            (match args with
            | [addr; value] ->
                gen_expr buf addr ;
                Buffer.add_string buf ", " ;
                gen_expr buf value
            | args ->
                Codegen_error.raise_error
                  (Codegen_error.invalid_arg_count
                     "atomic_min"
                     2
                     (List.length args))) ;
            Buffer.add_char buf ')'
        | "atomic_max" ->
            Buffer.add_string buf "atomicMax(" ;
            (match args with
            | [addr; value] ->
                gen_expr buf addr ;
                Buffer.add_string buf ", " ;
                gen_expr buf value
            | args ->
                Codegen_error.raise_error
                  (Codegen_error.invalid_arg_count
                     "atomic_max"
                     2
                     (List.length args))) ;
            Buffer.add_char buf ')'
        | "float" ->
            Buffer.add_string buf "f32(" ;
            (match args with [e] -> gen_expr buf e | _ -> ()) ;
            Buffer.add_char buf ')'
        | "int_of_float" ->
            Buffer.add_string buf "i32(" ;
            (match args with [e] -> gen_expr buf e | _ -> ()) ;
            Buffer.add_char buf ')'
        | _ ->
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
  | LVar v -> Buffer.add_string buf (escape_wgsl_name v.var_name)
  | LArrayElem (arr, idx) ->
      Buffer.add_string buf (escape_wgsl_name arr) ;
      Buffer.add_char buf '[' ;
      gen_expr buf idx ;
      Buffer.add_char buf ']'
  | LArrayElemExpr (base, idx) ->
      Buffer.add_char buf '(' ;
      gen_expr buf base ;
      Buffer.add_char buf ')' ;
      Buffer.add_char buf '[' ;
      gen_expr buf idx ;
      Buffer.add_char buf ']'
  | LRecordField (lv, field) ->
      gen_lvalue buf lv ;
      Buffer.add_char buf '.' ;
      Buffer.add_string buf field

(** {1 Statement Generation} *)

let indent_nested indent = indent ^ "  "

and gen_match_pattern buf indent scrutinee cname bindings find_constr_types =
  Buffer.add_string buf ("  case " ^ cname ^ ": {\n") ;
  match (bindings, find_constr_types cname) with
  | [var_name], Some [ty] ->
      let vn = escape_wgsl_name var_name in
      Buffer.add_string buf (indent ^ "    ") ;
      Buffer.add_string buf "let " ;
      Buffer.add_string buf vn ;
      Buffer.add_string buf " : " ;
      Buffer.add_string buf (wgsl_type_of_elttype ty) ;
      Buffer.add_string buf " = " ;
      Buffer.add_string buf scrutinee ;
      Buffer.add_char buf '.' ;
      Buffer.add_string buf cname ;
      Buffer.add_string buf "_v;\n"
  | vars, Some types when List.length vars = List.length types ->
      List.iteri
        (fun i (var_name, ty) ->
          let vn = escape_wgsl_name var_name in
          Buffer.add_string buf (indent ^ "    ") ;
          Buffer.add_string buf "let " ;
          Buffer.add_string buf vn ;
          Buffer.add_string buf " : " ;
          Buffer.add_string buf (wgsl_type_of_elttype ty) ;
          Buffer.add_string buf " = " ;
          Buffer.add_string buf scrutinee ;
          Buffer.add_char buf '.' ;
          Buffer.add_string buf cname ;
          Buffer.add_string buf (Printf.sprintf "_v._%d;\n" i))
        (List.combine vars types)
  | [], _ | _, None | _, Some [] -> ()
  | _ ->
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "pattern"
           "mismatch between pattern bindings and constructor args")

and gen_var_decl buf indent ~mutable_ v_name v_type init_expr =
  let vn = escape_wgsl_name v_name in
  Buffer.add_string buf indent ;
  Buffer.add_string buf (if mutable_ then "var" else "let") ;
  Buffer.add_char buf ' ' ;
  Buffer.add_string buf vn ;
  Buffer.add_string buf " : " ;
  Buffer.add_string buf (wgsl_type_of_elttype v_type) ;
  Buffer.add_string buf " = " ;
  gen_expr buf init_expr ;
  Buffer.add_string buf ";\n"

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
      let op, step_expr =
        match dir with
        | Upto ->
            ("<=", Printf.sprintf " = %s + 1i" (escape_wgsl_name v.var_name))
        | Downto ->
            (">=", Printf.sprintf " = %s - 1i" (escape_wgsl_name v.var_name))
      in
      let loop_var = escape_wgsl_name v.var_name in
      Buffer.add_string buf indent ;
      Buffer.add_string buf "for (var " ;
      Buffer.add_string buf loop_var ;
      Buffer.add_string buf " : " ;
      Buffer.add_string buf (wgsl_type_of_elttype v.var_type) ;
      Buffer.add_string buf " = " ;
      gen_expr buf start ;
      Buffer.add_string buf "; " ;
      Buffer.add_string buf loop_var ;
      Buffer.add_string buf (" " ^ op ^ " ") ;
      gen_expr buf stop ;
      Buffer.add_string buf "; " ;
      Buffer.add_string buf loop_var ;
      Buffer.add_string buf step_expr ;
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
      Buffer.add_string buf "workgroupBarrier();\n"
  | SWarpBarrier ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "subgroupBarrier();\n"
  | SMemFence ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "storageBarrier();\n"
  | SNative _ ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "/* native code not supported in WGSL */\n"
  | SExpr e ->
      Buffer.add_string buf indent ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n"
  | SLet (_v, EArrayCreate (_, _, Shared), body) ->
      (* Shared arrays are hoisted to module scope by collect_workgroup_decls;
         only the body continuation needs to be emitted here. *)
      gen_stmt buf indent body
  | SLet (_v, EArrayCreate (_, _, Local), _) ->
      (* WGSL has no function-local dynamic arrays. Raise rather than emit
         invalid WGSL. Use Shared memspace for workgroup-scoped arrays. *)
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "EArrayCreate(Local)"
           "WGSL: local dynamic arrays unsupported; use Shared for workgroup \
            arrays")
  | SLet (_v, EArrayCreate (_, _, Global), _) ->
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "EArrayCreate(Global)"
           "WGSL: global dynamic array creation not supported in kernel body")
  | SLet (v, e, body) ->
      gen_var_decl buf indent ~mutable_:false v.var_name v.var_type e ;
      gen_stmt buf indent body
  | SLetMut (v, e, body) ->
      gen_var_decl buf indent ~mutable_:true v.var_name v.var_type e ;
      gen_stmt buf indent body
  | SPragma (_hints, body) -> gen_stmt buf indent body
  | SBlock body ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "{\n" ;
      gen_stmt buf (indent_nested indent) body ;
      Buffer.add_string buf indent ;
      Buffer.add_string buf "}\n"

(** {1 Helper Function Generation} *)

let gen_helper_func buf (hf : helper_func) =
  let non_vec_params =
    List.filter
      (fun (v : var) -> match v.var_type with TVec _ -> false | _ -> true)
      hf.hf_params
  in
  Buffer.add_string buf "fn " ;
  Buffer.add_string buf hf.hf_name ;
  Buffer.add_char buf '(' ;
  List.iteri
    (fun i (v : var) ->
      if i > 0 then Buffer.add_string buf ", " ;
      Buffer.add_string buf (escape_wgsl_name v.var_name) ;
      Buffer.add_string buf " : " ;
      Buffer.add_string buf (wgsl_type_of_elttype v.var_type))
    non_vec_params ;
  Buffer.add_string buf ") -> " ;
  Buffer.add_string buf (wgsl_type_of_elttype hf.hf_ret_type) ;
  Buffer.add_string buf " {\n" ;
  gen_stmt buf "  " hf.hf_body ;
  Buffer.add_string buf "}\n\n"

(** {1 Record and Variant Type Generation} *)

let gen_record_def buf (name, fields) =
  let mangled = mangle_name name in
  Buffer.add_string buf (Printf.sprintf "struct %s {\n" mangled) ;
  List.iter
    (fun (fname, ftype) ->
      Buffer.add_string buf "  " ;
      Buffer.add_string buf fname ;
      Buffer.add_string buf " : " ;
      Buffer.add_string buf (wgsl_type_of_elttype ftype) ;
      Buffer.add_string buf ",\n")
    fields ;
  Buffer.add_string buf "}\n\n"

(** Emit a WGSL variant type. WGSL has no enums or unions. We emit:
    - [const <CNAME> : i32 = N;] for each constructor tag
    - a struct with [tag : i32] and flat payload fields
    - [fn make_<Type>_<Constr>(...) -> <Type>] constructors *)
let gen_variant_def buf (name, constrs) =
  let mangled = mangle_name name in
  List.iteri
    (fun i (cname, _) ->
      Buffer.add_string buf (Printf.sprintf "const %s : i32 = %di;\n" cname i))
    constrs ;
  Buffer.add_char buf '\n' ;
  Buffer.add_string buf (Printf.sprintf "struct %s {\n  tag : i32,\n" mangled) ;
  let has_payload = List.exists (fun (_, args) -> args <> []) constrs in
  if has_payload then begin
    List.iter
      (fun (cname, args) ->
        match args with
        | [] -> ()
        | [ty] ->
            Buffer.add_string
              buf
              (Printf.sprintf "  %s_v : %s,\n" cname (wgsl_type_of_elttype ty))
        | _ ->
            List.iteri
              (fun i ty ->
                Buffer.add_string
                  buf
                  (Printf.sprintf
                     "  %s_v_%d : %s,\n"
                     cname
                     i
                     (wgsl_type_of_elttype ty)))
              args)
      constrs
  end ;
  Buffer.add_string buf "}\n\n" ;
  List.iteri
    (fun _i (cname, args) ->
      Buffer.add_string buf (Printf.sprintf "fn make_%s_%s(" mangled cname) ;
      (match args with
      | [] -> ()
      | [ty] -> Buffer.add_string buf ("v : " ^ wgsl_type_of_elttype ty)
      | _ ->
          List.iteri
            (fun j ty ->
              if j > 0 then Buffer.add_string buf ", " ;
              Buffer.add_string
                buf
                (Printf.sprintf "v%d : %s" j (wgsl_type_of_elttype ty)))
            args) ;
      Buffer.add_string buf (Printf.sprintf ") -> %s {\n" mangled) ;
      Buffer.add_string buf (Printf.sprintf "  var r : %s;\n" mangled) ;
      Buffer.add_string buf (Printf.sprintf "  r.tag = %s;\n" cname) ;
      (match args with
      | [] -> ()
      | [_] -> Buffer.add_string buf (Printf.sprintf "  r.%s_v = v;\n" cname)
      | _ ->
          List.iteri
            (fun j _ ->
              Buffer.add_string
                buf
                (Printf.sprintf "  r.%s_v_%d = v%d;\n" cname j j))
            args) ;
      Buffer.add_string buf "  return r;\n}\n\n")
    constrs

(** {1 Buffer / Uniform Binding Generation} *)

(** Collect workgroup shared array declarations from a statement tree. *)
let rec collect_workgroup_decls (s : stmt) : (string * elttype * expr) list =
  match s with
  | SLet (v, EArrayCreate (elem_ty, size, Shared), body) ->
      (escape_wgsl_name v.var_name, elem_ty, size)
      :: collect_workgroup_decls body
  | SLet (_, _, body) | SLetMut (_, _, body) -> collect_workgroup_decls body
  | SSeq stmts -> List.concat_map collect_workgroup_decls stmts
  | SFor (_, _, _, _, body) -> collect_workgroup_decls body
  | SWhile (_, body) -> collect_workgroup_decls body
  | SIf (_, st, sf_opt) ->
      let sf_decls =
        match sf_opt with Some sf -> collect_workgroup_decls sf | None -> []
      in
      collect_workgroup_decls st @ sf_decls
  | SBlock body -> collect_workgroup_decls body
  | SPragma (_, body) -> collect_workgroup_decls body
  | SMatch (_, cases) ->
      List.concat_map (fun (_, body) -> collect_workgroup_decls body) cases
  | SEmpty | SBarrier | SWarpBarrier | SMemFence | SNative _ | SExpr _
  | SAssign _ | SReturn _ ->
      []

let gen_workgroup_module_decls buf (decls : (string * elttype * expr) list) =
  if decls <> [] then begin
    Buffer.add_string buf "// Workgroup shared memory\n" ;
    List.iter
      (fun (name, elem_ty, size) ->
        Buffer.add_string buf "var<workgroup> " ;
        Buffer.add_string buf name ;
        Buffer.add_string buf " : array<" ;
        Buffer.add_string buf (wgsl_type_of_elttype elem_ty) ;
        Buffer.add_string buf ", " ;
        gen_expr buf size ;
        Buffer.add_string buf ">;\n")
      decls ;
    Buffer.add_char buf '\n'
  end

(** Separate kernel params into vectors (storage buffers) and scalars (uniform).
*)
let split_params params =
  let vectors = ref [] in
  let scalars = ref [] in
  List.iter
    (fun decl ->
      match decl with
      | DParam (v, _) -> (
          match v.var_type with
          | TVec _ -> vectors := v :: !vectors
          | _ -> scalars := v :: !scalars)
      | _ -> ())
    params ;
  (List.rev !vectors, List.rev !scalars)

(** Emit storage buffer bindings and the Params uniform struct. Returns the list
    of scalar param names (for [scalar_param_names] ref). *)
let gen_bindings buf params =
  let vectors, scalars = split_params params in
  let binding_idx = ref 0 in
  List.iter
    (fun (v : var) ->
      let name = escape_wgsl_name v.var_name in
      let elem_type =
        match v.var_type with
        | TVec elt -> wgsl_type_of_elttype elt
        | _ -> assert false
      in
      Buffer.add_string
        buf
        (Printf.sprintf
           "@group(0) @binding(%d) var<storage, read_write> %s : array<%s>;\n"
           !binding_idx
           name
           elem_type) ;
      incr binding_idx)
    vectors ;
  if vectors <> [] || scalars <> [] then begin
    Buffer.add_string buf "struct Params {\n" ;
    List.iter
      (fun (v : var) ->
        let name = escape_wgsl_name v.var_name in
        Buffer.add_string buf (Printf.sprintf "  sarek_%s_length : i32,\n" name))
      vectors ;
    List.iter
      (fun (v : var) ->
        let name = escape_wgsl_name v.var_name in
        Buffer.add_string
          buf
          (Printf.sprintf "  %s : %s,\n" name (wgsl_type_of_elttype v.var_type)))
      scalars ;
    Buffer.add_string buf "}\n" ;
    Buffer.add_string
      buf
      (Printf.sprintf
         "@group(0) @binding(%d) var<uniform> params : Params;\n"
         !binding_idx)
  end ;
  Buffer.add_char buf '\n' ;
  List.map (fun (v : var) -> v.var_name) scalars

(** {1 Main generate functions} *)

let wgsl_header ~kernel_name ?(block = (256, 1, 1)) () =
  let bx, by, bz = block in
  Printf.sprintf
    "@compute @workgroup_size(%d, %d, %d)\n\
     fn main(\n\
    \  @builtin(global_invocation_id) gid : vec3<u32>,\n\
    \  @builtin(local_invocation_id) lid : vec3<u32>,\n\
    \  @builtin(workgroup_id) wid : vec3<u32>,\n\
    \  @builtin(num_workgroups) nwg : vec3<u32>\n\
     ) {\n"
    bx
    by
    bz
  |> fun s ->
  Printf.sprintf "// Sarek-generated compute shader: %s\n%s" kernel_name s

(** Check if any kernel param uses Float64. *)
let params_have_float64 params =
  List.exists
    (fun decl ->
      match decl with DParam (v, _) -> has_float64 v.var_type | _ -> false)
    params

(** Generate complete WGSL source for a kernel. *)
let generate ?block ?(log : string -> unit = fun _ -> ()) (k : kernel) : string
    =
  if params_have_float64 k.kern_params then
    Codegen_error.raise_error
      (Codegen_error.unsupported_construct
         "f64 parameter"
         "WGSL: f64 unsupported — WebGPU has no float64 type") ;
  scalar_param_names := [] ;
  current_variants := k.kern_variants ;
  let buf = Buffer.create 1024 in
  let scalars = gen_bindings buf k.kern_params in
  scalar_param_names := scalars ;
  let wg_decls = collect_workgroup_decls k.kern_body in
  gen_workgroup_module_decls buf wg_decls ;
  List.iter (gen_helper_func buf) k.kern_funcs ;
  Buffer.add_string buf (wgsl_header ~kernel_name:k.kern_name ?block ()) ;
  gen_stmt buf "  " k.kern_body ;
  Buffer.add_string buf "}\n" ;
  let shader = Buffer.contents buf in
  log (Printf.sprintf "[WGSL] Generated shader:\n%s" shader) ;
  shader

(** Generate WGSL source with custom type definitions. *)
let generate_with_types ?block ?(log : string -> unit = fun _ -> ())
    ~(types : (string * (string * elttype) list) list) (k : kernel) : string =
  if params_have_float64 k.kern_params then
    Codegen_error.raise_error
      (Codegen_error.unsupported_construct
         "f64 parameter"
         "WGSL: f64 unsupported — WebGPU has no float64 type") ;
  scalar_param_names := [] ;
  current_variants := k.kern_variants ;
  let buf = Buffer.create 1024 in
  List.iter (gen_record_def buf) types ;
  List.iter (gen_variant_def buf) k.kern_variants ;
  let scalars = gen_bindings buf k.kern_params in
  scalar_param_names := scalars ;
  let wg_decls = collect_workgroup_decls k.kern_body in
  gen_workgroup_module_decls buf wg_decls ;
  List.iter (gen_helper_func buf) k.kern_funcs ;
  Buffer.add_string buf (wgsl_header ~kernel_name:k.kern_name ?block ()) ;
  gen_stmt buf "  " k.kern_body ;
  Buffer.add_string buf "}\n" ;
  let shader = Buffer.contents buf in
  log (Printf.sprintf "[WGSL] Generated shader:\n%s" shader) ;
  shader

(** {1 ABI descriptor} *)

(** Build the ABI descriptor for a kernel. Reuses [split_params] and
    [escape_wgsl_name] / [wgsl_type_of_elttype] so the descriptor cannot drift
    from [gen_bindings].

    Raises [Codegen_error.unsupported_construct] for f64 parameters (same error
    as [generate]). *)
let abi ?(block = (256, 1, 1)) (k : kernel) : Sarek_wgsl_abi.t =
  if params_have_float64 k.kern_params then
    Codegen_error.raise_error
      (Codegen_error.unsupported_construct
         "f64 parameter"
         "WGSL: f64 unsupported — WebGPU has no float64 type") ;
  let vectors, scalars = split_params k.kern_params in
  (* Storage buffer descriptors — one per vector, binding 0..k-1. *)
  let buffers =
    List.mapi
      (fun i (v : var) ->
        let elt = match v.var_type with TVec e -> e | _ -> assert false in
        let element_type =
          match wgsl_type_of_elttype elt with
          | "f32" -> Sarek_wgsl_abi.F32
          | "i32" -> Sarek_wgsl_abi.I32
          | "u32" -> Sarek_wgsl_abi.U32
          | other ->
              Codegen_error.raise_error
                (Codegen_error.unsupported_construct
                   other
                   "WGSL ABI: unsupported element type")
        in
        Sarek_wgsl_abi.
          {
            name = escape_wgsl_name v.var_name;
            binding = i;
            element_type;
            access = "read_write";
          })
      vectors
  in
  let num_vectors = List.length vectors in
  (* Params struct — present when there are any vectors or scalars. *)
  let params_opt =
    if vectors = [] && scalars = [] then None
    else begin
      (* Fields: one length i32 per vector, then each scalar. *)
      let length_fields =
        List.mapi
          (fun j (v : var) ->
            let vec_name = escape_wgsl_name v.var_name in
            Sarek_wgsl_abi.
              {
                name = Printf.sprintf "sarek_%s_length" vec_name;
                field_type = I32;
                offset = 4 * j;
                kind = Length vec_name;
              })
          vectors
      in
      let scalar_fields =
        List.mapi
          (fun j (v : var) ->
            let field_type =
              match wgsl_type_of_elttype v.var_type with
              | "f32" -> Sarek_wgsl_abi.F32
              | "i32" -> Sarek_wgsl_abi.I32
              | "u32" -> Sarek_wgsl_abi.U32
              | other ->
                  Codegen_error.raise_error
                    (Codegen_error.unsupported_construct
                       other
                       "WGSL ABI: unsupported scalar type")
            in
            Sarek_wgsl_abi.
              {
                name = escape_wgsl_name v.var_name;
                field_type;
                offset = 4 * (num_vectors + j);
                kind = Scalar;
              })
          scalars
      in
      let all_fields = length_fields @ scalar_fields in
      let num_fields = List.length all_fields in
      (* byteSize = total bytes rounded up to multiple of 16. *)
      let raw = num_fields * 4 in
      let byte_size =
        if raw mod 16 = 0 then raw else raw + (16 - (raw mod 16))
      in
      Some
        Sarek_wgsl_abi.{binding = num_vectors; byte_size; fields = all_fields}
    end
  in
  Sarek_wgsl_abi.
    {
      kernel_name = k.kern_name;
      workgroup_size = block;
      buffers;
      params = params_opt;
    }
