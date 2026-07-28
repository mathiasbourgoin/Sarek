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

(** Local error module — same raised exception as the package-level
    [Metal_error]. *)
module Codegen_error = Sarek_backend_error.Backend_error.Make (struct
  let name = "Metal"
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

(** Map Sarek IR element type to Metal C type string *)
let rec metal_type_of_elttype = function
  | TInt32 -> "int"
  | TInt64 -> "long"
  | TFloat16 ->
      (* Deferred to #57 slice 2: Metal has a native `half` needing no feature
         declaration, so this arm is a one-liner then — but it is untestable on
         this platform, so it is not landed unverified. *)
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "f16"
           "Metal: float16 not yet supported (#57 slice 2)")
  | TUint8 ->
      (* Unlike the f16 arm above this is not a deferral. MSL has `uchar` and
         even has simdgroup matrices, but [TUint8] is not a general 8-bit
         integer in this IR: it marks a cooperative-matrix operand buffer, and
         emitting `uchar` for it would produce a buffer no Metal statement can
         consume, because the [SCoopmat] that gives it meaning is refused. *)
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "uint8"
           "Metal: uint8 is a cooperative-matrix operand element type, emitted \
            only by the Vulkan backend, and Metal has no cooperative-matrix \
            path")
  | TFloat32 -> "float"
  | TFloat64 ->
      (* Until #64 slice 1 this arm was `"float"`, with a comment saying Metal
         does not support double precision — and no refusal anywhere on the
         path. Metal genuinely has no `double`, so this is Backend_structural
         and belongs at codegen, not at a launch gate: no device can supply it.

         #141 established that the cost is WORSE than the halved precision this
         comment originally claimed, and the correction matters because it moves
         the defect from quality-of-result to wrong-answer. The IR element type
         also fixes the BUFFER STRIDE: a `float64 vector` is 8 bytes per element
         on the host (Spoc_core.Vector.float64), so `device float* v` strode it
         at 4 and every element after the first was a bit-half of its neighbour.
         The kernel did not lose precision — it read a different array. Captured
         before the fix, the emitted kernel for out.(i) <- inp.(i) * 2.0 in
         float64 was, verbatim,
         `kernel void f64_scale(device float* out ..., device float* inp ...)`,
         with no diagnostic on any channel.

         Same family as the `[@@sarek.type]` payload emitted as `int` and the
         `Char` vector read at 4-byte stride: a type computed and then narrowed
         without saying so. The class validator over the whole backend
         type-mapping surface is
         sarek/tests/codegen_golden/test_backend_type_width_totality.ml. *)
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "f64"
           (Sarek_capability.explain
              ~target:"Metal"
              Sarek_capability.float64_absent_metal))
  | TBool ->
      (* "int", not "bool", and this is a width fix, not a style choice. The
         host gives a Sarek `bool` a 4-byte slot everywhere it has a layout:
         Sarek_ppx's field-size mapping returns 4 for `bool`, and
         Sarek_ir_layout.scalar_size TBool is 4 to match it. MSL `bool` is ONE
         byte (MSL spec, size/alignment of scalar data types), so a
         [@@sarek.type] record with bool fields desynced silently — host
         {bool;bool;int} lays out at 0/4/8, size 12, while the emitted
         `typedef struct { bool a; bool b; int n; }` lays out at 0/1/4, size 8.
         CUDA and OpenCL both already emit `int` here for exactly this reason;
         Metal was the odd one out. GLSL/WGSL also spell it `bool`, but neither
         is a silent case: glslang lowers a storage-buffer bool to a 32-bit uint
         (verified — OpMemberDecorate Offset 0/4, ArrayStride 8), and naga
         refuses a bool in a storage struct outright ("The type is not
         host-shareable").

         Deliberately NOT a {!Sarek_capability} entry. The test that decides it
         is docs/design/capability-model.md §5.1, and it is NOT "is it silent"
         nor "is it a width mismatch" — both are equally true of this arm and of
         the TFloat64 one above. It is: DOES A CORRECT LOWERING EXIST IN THE
         TARGET LANGUAGE? For f64 there is none, so it is a capability. For
         bool there is — `int`, at the host's width, which CUDA and OpenCL
         already emit — so it is a codegen bug and the fix belongs here. Filing
         it as Backend_structural would make the table claim Metal cannot
         express booleans, which is false, and would remove a working feature
         from users of this backend. *)
      "int"
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
  | name -> Codegen_error.raise_error (Codegen_error.unknown_intrinsic name)

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
  | EIntrinsic (path, name, args) ->
      Dispatch.gen_intrinsic metal_backend buf path name args
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
        (Codegen_error.unsupported_construct "match" "empty match expression")
  | EMatch (_, [(_, body)]) ->
      (* Single case - just emit the body *)
      gen_expr buf body
  | EMatch (e, cases) ->
      (* Multi-case match as nested ternary - check tag field *)
      let rec gen_cases = function
        | [] ->
            Codegen_error.raise_error
              (Codegen_error.unsupported_construct "match" "empty match cases")
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

(** MSL has no [cbrt]/[hypot]/[expm1]/[log1p] builtins under any name (unlike
    [fabs]/[rsqrt]/[atan2], which MSL does define — see Table 6.4 of the Metal
    Shading Language Specification). These need a multi-token expression instead
    of a function-name substitution, so they're special-cased here ahead of both
    the unqualified match arms and the pure registry, applying uniformly to
    qualified (Float32.cbrt) and unqualified calls alike. [cbrt] uses
    [sign(x)*pow(abs(x),...)] rather than bare [pow] because [pow] is undefined
    for a negative base. *)
and gen_metal_polyfill buf name args =
  match (name, args) with
  | "cbrt", [x] ->
      Buffer.add_string buf "(sign(" ;
      gen_expr buf x ;
      Buffer.add_string buf ") * pow(abs(" ;
      gen_expr buf x ;
      Buffer.add_string buf "), 1.0 / 3.0))"
  | "hypot", [x; y] ->
      Buffer.add_string buf "sqrt((" ;
      gen_expr buf x ;
      Buffer.add_string buf ") * (" ;
      gen_expr buf x ;
      Buffer.add_string buf ") + (" ;
      gen_expr buf y ;
      Buffer.add_string buf ") * (" ;
      gen_expr buf y ;
      Buffer.add_string buf "))"
  | "expm1", [x] ->
      Buffer.add_string buf "(exp(" ;
      gen_expr buf x ;
      Buffer.add_string buf ") - 1.0)"
  | "log1p", [x] ->
      Buffer.add_string buf "log(1.0 + (" ;
      gen_expr buf x ;
      Buffer.add_string buf "))"
  | _ ->
      Codegen_error.raise_error
        (Codegen_error.unknown_intrinsic
           (Printf.sprintf "%s (wrong arity for Metal polyfill)" name))

and metal_backend =
  {
    Dispatch.framework =
      (fun () -> Option.value ~default:"Metal" !current_framework);
    gen_expr;
    thread_intrinsic = metal_thread_intrinsic;
    pre_hook =
      (fun buf ~full_name:_ _path name args ->
        if List.mem name ["cbrt"; "hypot"; "expm1"; "log1p"] then (
          gen_metal_polyfill buf name args ;
          true)
        else false);
    post_hook =
      (fun buf path name args ->
        (* Same framework tag the pure-registry lookup uses: without it this
           fallback emitted the CUDA spelling on every backend. *)
        let framework = Option.value ~default:"Metal" !current_framework in
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
        | "floor" | "ceil" | "round" | "trunc" | "fabs" | "atan2" | "pow"
        | "fma" | "min" | "max" ->
            Some (fun buf args -> Dispatch.emit_call ~gen_expr buf name args)
        | "block_barrier" ->
            Some
              (fun buf _ ->
                Buffer.add_string
                  buf
                  "threadgroup_barrier(mem_flags::mem_threadgroup)")
        | "atomic_add" | "atomic_add_int32" ->
            Some
              (fun buf args ->
                Dispatch.emit_atomic
                  ~gen_expr
                  ~invalid_arg_count:bad_arity
                  buf
                  ~callee:"atomic_fetch_add_explicit"
                  ~prefix:"(volatile threadgroup atomic_int*)&"
                  ~suffix:", memory_order_relaxed)"
                  ~opname:"atomic_add"
                  ~expected:2
                  ~allow_array:true
                  args)
        | "atomic_add_global_int32" ->
            Some
              (fun buf args ->
                Dispatch.emit_atomic
                  ~gen_expr
                  ~invalid_arg_count:bad_arity
                  buf
                  ~callee:"atomic_fetch_add_explicit"
                  ~prefix:"(volatile device atomic_int*)&"
                  ~suffix:", memory_order_relaxed)"
                  ~opname:"atomic_add_global"
                  ~expected:2
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
          Buffer.add_string buf (metal_type_of_elttype ty) ;
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
        (Codegen_error.unsupported_construct
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
      (* MSL's warp-level ("SIMD-group") barrier is simdgroup_barrier. The old
         text here, sub_group_threadgroup_barrier, is an OpenCL-shaped name that
         MSL does not declare, so it would have been a hard MSL compile error.
         It survived because nothing constructs SWarpBarrier from PPX syntax
         yet, so the arm has never reached a Metal compiler --- and because the
         repo's own Metal plugin table (sarek-metal/Metal_plugin.ml) already
         said simdgroup_barrier, the two disagreed in silence. Pinned by
         sarek/tests/unit/test_sync_stmt_emission.ml. *)
      Buffer.add_string buf indent ;
      Buffer.add_string buf "simdgroup_barrier(mem_flags::mem_threadgroup);\n"
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
          Codegen_error.raise_error
            (Codegen_error.no_device_selected
               "SNative requires device context (set current_framework before \
                calling generate)"))
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
  | SCoopmat _ ->
      (* MSL's simdgroup_matrix is a plausible future home for this, but it has
         its own shapes, its own load/store convention and no measurement here;
         until one exists the statement has no lowering, and a silent skip would
         leave the accumulator buffer untouched rather than wrong-by-a-little,
         which is the failure mode hardest to notice. *)
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "cooperative matrix"
           "Metal: the Metal backend has no cooperative-matrix path; \
            cooperative-matrix statements are emitted only by the Vulkan \
            backend")

(** {1 Declaration Generation} *)

(** Check if a type is a vector (requires length parameter). Still used by the
    Metal-specific {!gen_param_metal} below (buffer-index variant). *)
let is_vec_type = Sarek_ir_codegen.is_vec_type

(* "Will [metal_type_of_elttype] hand back a pointer?" — a different question
   from {!is_vec_type}, which asks "does this carry a trailing length argument?"
   and answers [true] for [TVec] alone. Conflating the two is what let [TArray]
   parameters reach the scalar arm and emit [constant T* &v]. *)
let is_pointer_type (t : elttype) =
  match t with TVec _ | TArray _ -> true | _ -> false

(* A buffer parameter: pointer into an address space, plus its length.

   ADDRESS SPACE IS [device], NOT [constant], and that is a decision rather than
   an accident (#139). Metal has no default address space for a pointer
   parameter, so the choice has to be made explicitly; [constant] is the wrong
   half of it for a Sarek [vec]:

   - MSL 3.2 §4.2/§4.3: objects in [constant] are read-only for the whole
     lifetime of the kernel. Sarek vecs are read-write — [record_kernel] writes
     [pts[idx] = ...] and [variant_kernel] writes [out[idx] = ...] — so
     [constant] would not compile even if the reference form below were fixed.
   - [constant] additionally carries an implementation-defined size limit and
     wants argument data that does not change per dispatch; a Sarek vec is a
     general MTLBuffer bound with [setBuffer:].
   - Every other backend already maps a vec param to a mutable global pointer
     (CUDA [T* __restrict__], OpenCL [__global T* restrict]), and Metal's own
     {!metal_param_type} and [metal_memspace Global] already say [device]. The
     [DParam (_, None)] arm was the single place in the backend that disagreed.

   The defect this replaces: that arm treated EVERY [DParam (v, None)] as a
   scalar and emitted [constant <ty> &name]. For a vec-typed [v],
   [metal_type_of_elttype] returns a pointer type, so the emission was
   [constant Point2* &pts] — a reference to a pointer whose POINTEE has no
   address space, which Metal rejects outright ("must have address space
   qualifier"). Both such goldens (record_kernel, variant_kernel) had never
   compiled; nothing on Linux could see it. *)
let gen_buffer_param buf atomic_vars idx v ~memspace ~elttype ~with_length =
  (* [metal_memspace Local] is [""], so a Local buffer parameter would emit a
     leading space and then a pointer with NO address space — the other half of
     MSL 3.2 §4.2, and exactly the shape Metal_gate.Metal_addrspace rejects. The
     emitter would be producing source its own gate refuses.

     Unreachable from the current corpus, but reachable in principle, and newly
     so: the [DParam (v, None)] arm above derives the space from the variable's
     type and passes [TArray (elt, Local)] straight through. Rejecting here
     keeps the invariant in the emitter, where it belongs, rather than resting
     on the corpus happening not to contain the case. *)
  (match memspace with
  | Local ->
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "gen_buffer_param"
           "Metal kernel buffer parameters need an explicit address space \
            (device or threadgroup); Local has none")
  | Global | Shared -> ()) ;
  Buffer.add_string buf (metal_memspace memspace) ;
  Buffer.add_char buf ' ' ;
  (* Use atomic type if this variable is used with atomics *)
  let type_str =
    if List.mem v.var_name atomic_vars then metal_atomic_type_of_elttype elttype
    else metal_type_of_elttype elttype
  in
  Buffer.add_string buf type_str ;
  Buffer.add_string buf "* " ;
  Buffer.add_string buf v.var_name ;
  Buffer.add_string buf " [[buffer(" ;
  Buffer.add_string buf (string_of_int idx) ;
  Buffer.add_string buf ")]]" ;
  (* Only a [TVec] carries the implicit trailing [sarek_<name>_length] argument
     — that is the documented meaning of {!Sarek_ir_codegen.is_vec_type}. A
     [TArray] has a size known at codegen time and no length argument, so
     emitting one here would shift every following buffer index past what the
     host binds. *)
  if with_length then begin
    Buffer.add_string buf ", constant int &sarek_" ;
    Buffer.add_string buf v.var_name ;
    Buffer.add_string buf "_length [[buffer(" ;
    Buffer.add_string buf (string_of_int (idx + 1)) ;
    Buffer.add_string buf ")]]" ;
    idx + 2
  end
  else idx + 1

(** Generate parameter with Metal buffer attributes, returns next buffer index
*)
let gen_param_metal buf atomic_vars idx = function
  (* Any POINTER-typed parameter carrying no [array_info]. The element type and
     the memory space come from the variable's own type; a bare [TVec] is a
     global buffer, exactly as {!metal_param_type} already assumed.

     The test is deliberately NOT [is_vec_type]: that predicate means "carries a
     trailing length argument" and is [TVec] only, so guarding on it left
     [TArray] parameters falling through to the scalar arm below — which emits
     [constant float* &a], the very #139 reference-to-pointer shape this change
     exists to remove, in a second constructor. Caught by the Local-address-space
     test in sarek-metal/test/test_sarek_ir_metal.ml. What matters here is
     whether [metal_type_of_elttype] will produce a pointer, so that is what is
     asked. *)
  | DParam (v, None) when is_pointer_type v.var_type ->
      let memspace, elttype, with_length =
        match v.var_type with
        | TVec elt -> (Global, elt, true)
        | TArray (elt, ms) -> (ms, elt, false)
        | _ ->
            (* unreachable: [is_pointer_type] is exactly TVec | TArray *)
            Codegen_error.raise_error
              (Codegen_error.unsupported_construct
                 "gen_param_metal"
                 "is_pointer_type accepted a non-pointer type")
      in
      gen_buffer_param buf atomic_vars idx v ~memspace ~elttype ~with_length
  | DParam (v, None) ->
      (* Scalar parameter - wrap in constant buffer. A scalar genuinely is
         uniform and read-only per dispatch, so [constant T &] is right here. *)
      Buffer.add_string buf "constant " ;
      Buffer.add_string buf (metal_type_of_elttype v.var_type) ;
      Buffer.add_string buf " &" ;
      Buffer.add_string buf v.var_name ;
      Buffer.add_string buf " [[buffer(" ;
      Buffer.add_string buf (string_of_int idx) ;
      Buffer.add_string buf ")]]" ;
      idx + 1
  | DParam (v, Some arr) ->
      (* Array with explicit info - always needs length *)
      gen_buffer_param
        buf
        atomic_vars
        idx
        v
        ~memspace:arr.arr_memspace
        ~elttype:arr.arr_elttype
        ~with_length:true
  | DLocal _ | DShared _ ->
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "gen_param_metal"
           "expected DParam")

let gen_param buf decl =
  Sarek_ir_codegen.gen_param
    ~param_type:metal_param_type
    ~gen_array_param:
      (Sarek_ir_codegen.gen_global_array_param
         ~memspace:metal_memspace
         ~type_of_elttype:metal_type_of_elttype)
    ~invalid:(fun () ->
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct "gen_param" "expected DParam"))
    buf
    decl

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
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
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

(* Slice-2 deferral for this backend. The explanation of WHY a whole-kernel gate
   is needed (and not just the per-element-type arms) lives once, at
   {!Sarek_ir_codegen.reject_feature}. Named [_kernel] to distinguish it from
   [Sarek_typer.reject_float16], which rejects an f16 OPERAND — a different
   concept at a different layer. *)
let reject_float16_kernel =
  Sarek_ir_codegen.reject_feature
    ~raise_:(fun reason ->
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct "f16" reason))
    ~backend:"Metal"
    Sarek_ir_analysis.Float16

(* Whole-kernel f64 gate (#64 slice 1). The per-element-type arm above catches
   every type that reaches the emitter, but not every f64 in a kernel reaches
   it: the same reasoning that gave f16 a whole-kernel gate on top of its arm
   applies unchanged. Having both means the refusal cannot be routed around by
   a path that formats a type some other way.

   Unlike [reject_float16_kernel] this does NOT go through
   [Sarek_ir_codegen.reject_feature]: that composer says "not YET supported
   (#57 slice 2)", a claim about a queue position. Metal will never have
   `double`, so promising future support would be false.

   CONVERGENT: #141 arrived at this same second entry point independently, from
   the opposite direction — auditing the emitted source rather than the
   capability model — and reached the identical detector
   ([Sarek_ir_analysis.kernel_uses Float64], already the driver of the
   OpenCL/GLSL fp64 pragma/extension) with the identical placement at both
   [generate] entries. #141's route in was the f64 LOCAL, captured verbatim from
   the pre-fix emitter as `float x = 0.10000000000000001;`; #64's was the f64
   LITERAL assigned into an f32 buffer, which the type arm never sees at all.
   Two searches, two motivating shapes, one gate. That agreement is the reason
   to believe {arm, whole-kernel} is the COMPLETE set of entry points rather
   than the two somebody happened to think of. *)
let reject_float64_kernel =
  Sarek_capability.refuse_if_used
    ~raise_:(fun reason ->
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct "f64" reason))
    ~target:"Metal"
    Sarek_capability.float64_absent_metal
    Sarek_ir_analysis.Float64

(* Whole-kernel cooperative-matrix gate, for the reason spelled out at
   {!Sarek_ir_codegen.reject_feature}: the per-node arms above see only what the
   emitter reaches, and a kernel can carry the feature in a [TUint8] parameter
   it never loads from. Like the f64 gate and unlike the f16 one this does NOT
   go through [reject_feature], whose composed sentence hardcodes "#57 slice
   2" — the wrong issue to send a reader to for a cooperative-matrix refusal. *)
let reject_coopmat_kernel (k : kernel) : unit =
  if Sarek_ir_analysis.kernel_uses Sarek_ir_analysis.Coopmat k then
    Codegen_error.raise_error
      (Codegen_error.unsupported_construct
         "cooperative matrix"
         "Metal: the Metal backend has no cooperative-matrix path; cooperative \
          matrices and their uint8 operand buffers are emitted only by the \
          Vulkan backend (backlog-62)")

(** The ONLY thing measured to stop Metal contracting [a*b+c].

    Metal's compile options do NOT do it. Measured on Apple M4 / macOS 15.6.1
    (24G90) / Apple clang 17.0.0, on [o = a*b + c] over 65536 inputs, restricted
    to the 8773 elements where the DEVICE's own [fma] differs from the
    separately-rounded value (so contraction is observable at all):

    | build | contracted | |---|---| | default options | 8773 / 8773 | |
    [mathMode = MTLMathModeSafe] | **8773 / 8773** | | [mathMode=Safe] +
    [mathFloatingPointFunctions=Precise] | **8773 / 8773** | |
    [fastMathEnabled = NO] | **8773 / 8773** | | **this pragma** | **0 / 8773**
    |

    That is §1 corollary 2 again — "a flag that names the hazard is not a
    mechanism that prevents it" — and it is why the compile options set in
    [Metal_bindings.mtl_compile_options_conformant] are NOT a contraction
    defence and are not described as one. They buy math-function precision; this
    pragma buys the rounding.

    [#pragma clang fp contract(off)], a [volatile thread] local, a
    [threadgroup volatile] round-trip and an [as_type] bitcast round-trip were
    all measured to work too. The pragma is chosen because it is file-scoped,
    costs no register or memory traffic, and needs no per-expression codegen
    change — the same reasoning that put [precise] on GLSL locals (§6).
    [#pragma METAL fp math_mode(safe)] does NOT work: like the [mathMode]
    property it leaves contraction on. Sweep:
    [tools/probes/metal_contraction_barrier_probe.m].

    Sarek's rule is IEEE-754 with every operation rounded as written
    (docs/fp-contraction-policy.md §1), so this is a conformance requirement,
    not a tuning choice. *)
let metal_fp_contract_pragma = "#pragma METAL fp contract(off)\n"

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
  reject_float16_kernel k ;
  reject_float64_kernel k ;
  reject_coopmat_kernel k ;
  (* Set current_variants for SMatch binding extraction *)
  current_variants := k.kern_variants ;
  let buf = Buffer.create 4096 in

  (* Collect variables used with atomic operations *)
  let atomic_vars = collect_atomic_vars_stmt k.kern_body in

  (* Metal header *)
  Buffer.add_string buf "#include <metal_stdlib>\n" ;
  Buffer.add_string buf "using namespace metal;\n" ;
  Buffer.add_string buf metal_fp_contract_pragma ;
  Buffer.add_string buf "\n" ;

  (* Variant type definitions first (may be needed by records).
     Previously omitted here, unlike the CUDA/OpenCL backends, so Metal kernels
     using variant types emitted no typedef. Emitting them keeps Metal consistent
     with the other C-family backends. *)
  List.iter (gen_variant_def buf) k.kern_variants ;

  (* Record type definitions *)
  Sarek_ir_codegen.gen_record_typedefs
    ~type_of_elttype:metal_type_of_elttype
    buf
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

(** Generate complete Metal source for a kernel.

    A special case of {!generate_with_types} with the kernel's OWN type
    declarations, which is the only thing every production caller ever passed:
    [~types] has exactly the type of the [kern_types] field
    ([Sarek_ir_types.kernel]), so the parameter was redundant with the record it
    travels in. This used to be a separate 30-80 line copy of the emit sequence
    that silently omitted record typedefs, variant typedefs and
    [current_variants] — source referencing an undeclared struct, with no error.
    Delegating keeps one emit path per backend. *)
let generate (k : kernel) : string = generate_with_types ~types:k.kern_types k
