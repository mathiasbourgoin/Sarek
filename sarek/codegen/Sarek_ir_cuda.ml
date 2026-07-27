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

(** Local error module — same raised exception as the package-level
    [Cuda_error]. *)
module Codegen_error = Sarek_backend_error.Backend_error.Make (struct
  let name = "CUDA"
end)

module Dispatch = Sarek_ir_intrinsic_dispatch

(** Raise a located invalid-argument-count error (atomic-arity helper for the
    shared {!Dispatch.emit_atomic}). *)
let bad_arity n e g =
  Codegen_error.raise_error (Codegen_error.invalid_arg_count n e g)

(** Current framework string for SNative code generation. Always [None] in
    normal use; SNative branches check this ref and error if None. *)
let current_framework : string option ref = ref None

(** Current kernel's variant definitions (set during generate) *)
let current_variants : (string * (string * elttype list) list) list ref = ref []

(** {1 Type Mapping} *)

let mangle_name = Sarek_ir_codegen.mangle_name

(** Map Sarek IR element type to CUDA C type string *)
let rec cuda_type_of_elttype = function
  | TInt32 -> "int"
  | TInt64 -> "long long"
  | TFloat16 -> "__half"
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
  | name -> Codegen_error.raise_error (Codegen_error.unknown_intrinsic name)

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
  | EIntrinsic (path, name, args) ->
      Dispatch.gen_intrinsic cuda_backend buf path name args
  | ECast (TFloat16, e) ->
      (* f16 is a storage type: narrow through the documented intrinsic rather
         than a C cast, so the round-to-nearest-even mode is explicit and
         identical on CUDA and HIP. The widening direction (__half -> float,
         int, ...) needs no intrinsic: __half carries an implicit conversion
         operator to float in both toolchains, so a plain C cast is correct and
         is left to the generic arm below.

         The argument goes through [sarek_f32_barrier] because the narrowing is
         exactly where the AMDGPU backend fuses away the f32 intermediate the
         DSL promises. See [sarek_f32_barrier_decl] for the measured ISA and why
         neither -ffp-contract=off nor `#pragma clang fp contract` reaches it. *)
      Buffer.add_string buf "__float2half(sarek_f32_barrier(" ;
      gen_expr buf e ;
      Buffer.add_string buf "))"
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
  | EMatch (scrut, cases) when Sarek_ir_codegen.ematch_binds_payload cases ->
      (* #75: a match EXPRESSION lowers to a nested ternary, which has nowhere to
         declare a payload binder — bind it by substituting the same payload
         read the [SMatch] arm declares (the C-family tagged union), then emit the
         (now binder-free) match. One shared, capture-avoiding pass for every
         backend; see {!Sarek_ir_codegen.subst_ematch_payloads}. *)
      gen_expr
        buf
        (EMatch
           ( scrut,
             Sarek_ir_codegen.subst_ematch_payloads
               ~layout:Sarek_ir_codegen.c_family_payload_layout
               ~raise_:(fun msg ->
                 Codegen_error.raise_error
                   (Codegen_error.unsupported_construct
                      "match-expression payload binding"
                      msg))
               scrut
               cases ))
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

and cuda_backend =
  {
    Dispatch.framework =
      (fun () -> Option.value ~default:"CUDA" !current_framework);
    gen_expr;
    thread_intrinsic = cuda_thread_intrinsic;
    pre_hook = (fun _ ~full_name:_ _ _ _ -> false);
    post_hook =
      (fun buf path name args ->
        (* Same framework tag the pure-registry lookup uses: without it this
           fallback emitted the CUDA spelling on every backend. *)
        let framework = Option.value ~default:"CUDA" !current_framework in
        Dispatch.emit_registry_template
          ~gen_expr
          ~framework
          ~invalid_arg_count:bad_arity
          buf
          path
          name
          args);
    invalid_arg_count = bad_arity;
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
            Some (fun buf _ -> Buffer.add_string buf "__syncthreads()")
        | "atomic_add" | "atomic_add_int32" | "atomic_add_global_int32" ->
            Some
              (fun buf args ->
                Dispatch.emit_atomic
                  ~gen_expr
                  ~invalid_arg_count:bad_arity
                  buf
                  ~callee:"atomicAdd"
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
                  ~callee:"atomicSub"
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
                  ~callee:"atomicMin"
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
                  ~callee:"atomicMax"
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
          Buffer.add_string
            buf
            (Sarek_ir_codegen.payload_suffix
               Sarek_ir_codegen.c_family_payload_layout
               ~cname
               ~arity:1
               0) ;
          Buffer.add_string buf ";\n"
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
              Buffer.add_string
                buf
                (Sarek_ir_codegen.payload_suffix
                   Sarek_ir_codegen.c_family_payload_layout
                   ~cname
                   ~arity:(List.length vars)
                   i) ;
              Buffer.add_string buf ";\n")
            (List.combine vars types)
      | [], _ | _, None | _, Some [] -> () (* No bindings needed *)
      | _ ->
          Codegen_error.raise_error
            (Codegen_error.type_error
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

(* CUDA's array parameter uses [cuda_param_type] (which already carries the
   [__restrict__] spelling and pointer syntax) and, unlike OpenCL/Metal, emits
   no address-space qualifier, so it supplies its own [gen_array_param] rather
   than the shared {!Sarek_ir_codegen.gen_global_array_param}. *)
let gen_param buf decl =
  Sarek_ir_codegen.gen_param
    ~param_type:cuda_param_type
    ~gen_array_param:(fun buf (v : var) _arr ->
      Buffer.add_string buf (cuda_param_type v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf v.var_name)
    ~invalid:(fun () ->
      Codegen_error.raise_error
        (Codegen_error.invalid_memory_space "gen_param" "DLocal or DShared"))
    buf
    decl

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
      Codegen_error.raise_error
        (Codegen_error.invalid_memory_space "gen_local" "DParam")

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

(** f16 feature declaration, emitted only for kernels that actually use f16 —
    the same conditional-emission discipline the OpenCL/GLSL backends apply to
    fp64 via {!Sarek_ir_analysis.kernel_uses_float64}. Kernels without f16 are
    byte-identical to before.

    The guard is NOT decoration. This generator is shared verbatim by the HIP
    backend ([sarek-hip/Hip_shared.ml]), and the two toolchains disagree:

    - CUDA needs [#include <cuda_fp16.h>] for [__half] and [__float2half].
      Emitting the include is NECESSARY but not by itself SUFFICIENT under
      nvrtc: nvrtc is a library, not a driver, so it has no default include path
      and [__half] is not one of its builtins. Without an explicit
      [--include-path] the very include below fails with
      [NVRTC_ERROR_COMPILATION] / "could not open source file \"cuda_fp16.h\"
      (no directories in search list)". Supplying that path is
      [Cuda_nvrtc.cuda_include_paths]' job; nvcc, by contrast, adds the toolkit
      include directory itself.
    - HIP compiles through hiprtc, which pre-provides [__half], [half] and
      [__float2half] with no include at all, and which cannot resolve ANY f16
      header — neither [cuda_fp16.h] nor [hip/hip_fp16.h] exists on its search
      path. Emitting an unconditional include therefore breaks every HIP f16
      kernel. Verified empirically against hiprtc on gfx1100: bare [__half] +
      [__float2half] compile; both include forms fail with "file not found";
      this negative guard compiles.

    [__HIP__] / [__HIP_PLATFORM_AMD__] are both defined under hiprtc (also
    verified), so the include is skipped there and taken on CUDA. *)
let cuda_fp16_include =
  {|#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
#include <cuda_fp16.h>
#endif
|}

(** An optimisation barrier that forces an f32 value to be MATERIALISED as a
    correctly-rounded binary32 register before it is consumed.

    WHY THIS EXISTS. Sarek's f16 surface promises that arithmetic happens in f32
    and that narrowing to binary16 is a separate, explicit,
    round-to-nearest-even step — that is what makes the device agree with the
    interpreter bit-for-bit. The AMDGPU backend breaks that promise: it fuses a
    narrowing into the operation that feeds it. Measured on gfx1100 for
    [__float2half((float)__float2half((float)inp[tid] * 1.1f) + 1000.0f)]:

    v_fma_mixlo_f16 v0, v0, s2, 0 <- f32 multiply AND the narrowing, fused
    v_add_f16_e32 v0.l, 0x63d0, v0.l <- the f32 ADD demoted to binary16

    Both fusions skip a mandated rounding. The first is what made x = 5.68359375
    return 1006.5 on HIP where the interpreter, the native path and the host
    reference all return 1006.0: the fused form rounds the EXACT product once to
    binary16 instead of rounding to f32 first, and the exact product sits just
    above a binary16 tie that the correctly-rounded f32 value sits exactly on.

    Both halves are individually CORRECT — verified in isolation: the device's
    f32 product is bit-identical to the host's (0x40c81000), and the device's
    f32->f16 narrowing is round-to-nearest-even on exact ties in both directions
    and on negatives. The defect is purely the FUSION.

    WHY A BARRIER AND NOT A FLAG. Verified at ISA level, the emitted code is
    byte-identical under [-ffp-contract=off], [=on] and [=fast], and under
    [#pragma clang fp contract(off)]: this is an AMDGPU ISel combine that the
    standard FP-contraction controls do not reach. [-ffp-contract=off] is still
    set on the hiprtc path (see [Hip_rtc.base_options]) because it DOES fix
    ordinary f32 [a*b+c] contraction, but it is necessary-not-sufficient and
    does nothing for this pattern.

    COST. The [asm volatile] constraint pins the value in a register and
    clobbers nothing, so it costs no memory traffic — ScratchSize stays 0. A
    [volatile] local also works but spills to scratch, which is why it is not
    used. The price is the un-fused instruction count: 6 VALU ops instead of 2
    per f16 round-trip. Paid only inside kernels that actually narrow to f16 —
    the declaration is emitted only under [kernel_uses_float16].

    NVIDIA BRANCH: DELIBERATELY EMPTY, and that is a statement about NVIDIA, not
    an oversight. The non-HIP branch previously carried a PTX-flavoured
    [asm volatile("" : "+f"(x))]. AT A NARROWING it bought nothing, and it was
    removed because a call site reading [sarek_f32_barrier(...)] on the CUDA
    path advertised a protection that did not exist — the gap between assumed
    and actual FP semantics is exactly what produced this bug class.

    Measured for this file's current output on CUDA 13.3 (nvcc/ptxas/nvdisasm
    V13.3.73, host-side, no NVIDIA device) for sm_75, sm_80, sm_86, sm_89,
    sm_90, sm_100, sm_120 and sm_121: [cmp] on the cubin says byte-identical on
    all eight, and the arithmetic stream stays
    [HADD2.F32 / FMUL / F2FP.F16.F32 / HADD2.F32 / FADD / F2FP.F16.F32] — the
    f32 multiply and the f32 add both intact and the narrowings separate — with
    the asm and without it.

    WHY, precisely. The first version of this note got it wrong twice, and both
    corrections matter to anyone reusing this function:

    - NVVM does NOT erase the block. The barriered PTX keeps the
      [// begin/end inline asm] marker pair and allocates more virtual registers
      ([%f<9>] against [%f<5>]). What it contributes is ZERO PTX INSTRUCTIONS,
      so ptxas receives an identical instruction stream either way — which makes
      the identical cubins structural, not a 13.3 accident.
    - The barrier is NOT inert in general. On
      [out[i] = sarek_f32_barrier(a[i]*b[i]) + c[i]] it IS a real NVVM-level
      contraction barrier: [mul.f32]+[add.f32] with it, [fma.rn.f32] without.
      Measured at sm_90. But ptxas -O1 and above RE-CONTRACT that back to [FFMA]
      under the default [-fmad=true] ([-O0] and [--fmad=false] do not), so the
      cubins are byte-identical there too.

    CONSEQUENCE, and it is a trap: do NOT reach for [sarek_f32_barrier] to fix
    the caller-side df64 contraction hazard on NVIDIA. It protects the PTX and
    ptxas undoes it. [Sarek_df64]'s [mul_rn] works because an fma cannot be
    fused a second time — a property of the instruction, not of a barrier.

    At the f16 narrowing there is nothing for either level to fuse in the first
    place: NVIDIA has no fused multiply-and-convert-to-f16 instruction, which is
    why the emitted code is unchanged there under every flag tried.

    WHAT ACTUALLY HOLDS THE GUARANTEE ON NVIDIA: [ptxas] declines to absorb
    [cvt.rn.f16.f32] into the operation feeding it — hand-written PTX with no
    inline asm at all gives the same unfused SASS. That is a property of the
    assembler, not of anything Sarek emits, so it is machine-checked rather than
    assumed: [sarek-cuda/test/test_cuda_f16_sass.ml] walks generated CUDA ->
    nvrtc -> PTX -> ptxas -> cubin -> nvdisasm -> SASS on every architecture the
    local ptxas knows and fails if the discipline breaks. See
    [docs/fp-contraction-policy.md] and
    [docs/optimization/cuda-f16-fusion-sass-audit.md]. NO f16 kernel has been
    EXECUTED on NVIDIA hardware; the claim is a machine-code claim. *)
let sarek_f32_barrier_decl =
  {|#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float sarek_f32_barrier(float x) {
  asm volatile("" : "+v"(x));
  return x;
}
#else
/* NVIDIA: intentionally an identity. A PTX opacity barrier here contributes
   zero PTX instructions at a narrowing, so ptxas sees the same instruction
   stream and the cubins are byte-identical (measured on CUDA 13.3, sm_75..
   sm_121). What keeps the f32 multiply out of the narrowing on NVIDIA is ptxas
   itself, checked by test_cuda_f16_sass.ml — not this function. NOTE the same
   barrier IS load-bearing at PTX level for mul->add, but ptxas re-contracts
   that under the default -fmad=true; do not reuse it as a general contraction
   barrier here. See docs/fp-contraction-policy.md. */
__device__ __forceinline__ float sarek_f32_barrier(float x) {
  return x;
}
#endif
|}

(** Prefix for a kernel's generated source: the f16 include (only when the
    kernel uses f16) followed by the standard header. *)
let cuda_header_for (k : kernel) =
  if Sarek_ir_analysis.kernel_uses_float16 k then
    cuda_fp16_include ^ sarek_f32_barrier_decl ^ cuda_header
  else cuda_header

(** Generate complete CUDA source for a kernel *)
let generate (k : kernel) : string =
  let buf = Buffer.create 4096 in

  (* Header *)
  Buffer.add_string buf (cuda_header_for k) ;

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
  Buffer.add_string buf (cuda_header_for k) ;

  (* Variant type definitions first (may be needed by records) *)
  List.iter (gen_variant_def buf) k.kern_variants ;

  (* Record type definitions *)
  Sarek_ir_codegen.gen_record_typedefs
    ~type_of_elttype:cuda_type_of_elttype
    buf
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
