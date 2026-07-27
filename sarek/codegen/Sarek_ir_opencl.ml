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
      (* Still rejected after #57 slice 2a, but NOT for the reason originally
         recorded, and not merely "not implemented yet". The blocker is
         measured, not structural: the codegen itself is a two-line change
         ("half" here, a narrowing arm in gen_expr, and the cl_khr_fp16 pragma
         in the preamble).

         What blocks it is that OpenCL on this stack cannot hold Sarek's f16
         contract. Slice 1 defines f16 as "store f16, compute f32, round on
         every narrowing", and gates it by requiring the GPU to agree with the
         interpreter BIT-EXACTLY. On rusticl/radeonsi the ACO backend fuses the
         f32 multiply into the f32->f16 narrowing that consumes it, rounding
         ONCE where the DSL mandates twice — the same defect class HIP has, and
         620 of the 63488 finite binary16 inputs disagree because of it.

         The difference from HIP is that HIP has an affordable fix and OpenCL
         does not. Every source-level barrier that is expressible here was
         measured and does NOT work: `#pragma OPENCL FP_CONTRACT OFF`, a
         `volatile` local, a `volatile __private` pointer, an
         as_half/as_ushort bitcast round-trip, and convert_half_rte all still
         report 620/63488. HIP's `asm volatile("" : "+v"(x))` does not even
         compile (rusticl goes through SPIR-V, where AMDGPU register
         constraints do not exist). The only two barriers measured to work are
         a `volatile __global` round-trip and a `volatile __local` (LDS)
         round-trip, both 0/63488 — and both cost memory traffic per narrowing,
         with the LDS form additionally needing a workgroup-sized allocation
         this backend does not control (OpenCL fixes the workgroup size at
         launch, not at codegen).

         So enabling f16 here would mean shipping a backend that silently
         disagrees with the interpreter on 620 inputs — precisely the bug slice
         1 spent a review round removing. It stays rejected until either Mesa
         stops fusing, or a barrier that costs no memory traffic appears.

         Measured 2026-07-26 on RX 7900 XTX (navi31) AND the integrated Raphael
         iGPU (gfx1036), rusticl/radeonsi, Mesa via DRM 3.64 / kernel
         7.1.2-3-cachyos; both devices report 620/63488, first divergence at
         x=5.68359375 (got 1006.5, interpreter 1006). Reproducer:
         tools/probes/opencl_f16_contraction_probe.c. Full table and method:
         docs/fp-contraction-policy.md, "OpenCL / rusticl (f16 narrowing)". *)
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "f16"
           Sarek_ir_codegen.opencl_float16_refusal)
  | TUint8 ->
      (* OpenCL C spells an 8-bit unsigned integer `uchar` perfectly well; that
         is not what is being refused. [TUint8] reaches the IR only as the
         element type of a cooperative-matrix operand buffer, and OpenCL has no
         cooperative-matrix path to load it into. Mapping the type would let the
         buffer through while the matching [SCoopmat] statement is refused, i.e.
         it would move the diagnostic away from the construct that caused it. *)
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "uint8"
           "OpenCL: uint8 is a cooperative-matrix operand element type, \
            emitted only by the Vulkan backend, and OpenCL has no \
            cooperative-matrix path")
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
        (* Same framework tag the pure-registry lookup uses: without it this
           fallback emitted the CUDA spelling on every backend. *)
        let framework = Option.value ~default:"OpenCL" !current_framework in
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
  | SCoopmat _ ->
      (* Reached only if a kernel slipped past [reject_coopmat_kernel] — a
         helper body compiled outside [generate], say. The arm is kept a hard
         refusal rather than a no-op so that the failure mode is a diagnostic
         and not a kernel that quietly omits its matrix multiply. *)
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "cooperative matrix"
           "OpenCL: the OpenCL backend has no cooperative-matrix path; \
            cooperative-matrix statements are emitted only by the Vulkan \
            backend")

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
              Buffer.add_string buf (opencl_type_of_elttype ty) ;
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

(** Emit [ret name(params)] — shared by the prototype and the definition so the
    two can never drift apart. *)
let gen_helper_signature buf (hf : helper_func) =
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
    hf.hf_params

(** Forward declaration for a helper.

    Emitted for every helper BEFORE any definition, because [kern_funcs] carries
    no ordering guarantee: a caller listed before its callee produced
    [error: use of undeclared identifier 'g'] — invalid OpenCL C that depended
    purely on list order. Found by the #128 sweep once the recursion classifier
    stopped refusing helper-to-helper calls; no golden kernel has helpers, which
    is why the corpus never showed it. Declaring all of them up front makes the
    order irrelevant rather than relying on the IR producer to topologically
    sort. *)
let gen_helper_proto buf (hf : helper_func) =
  gen_helper_signature buf hf ;
  Buffer.add_string buf ");\n"

(** Generate a helper function (OpenCL device function) *)
let gen_helper_func buf (hf : helper_func) =
  gen_helper_signature buf hf ;
  Buffer.add_string buf ") {\n" ;
  (* Body *)
  gen_stmt buf "  " hf.hf_body ;
  Buffer.add_string buf "}\n\n"

(** Prototypes for every helper, then the definitions. A no-op when the kernel
    has no helpers, so kernels without them are byte-identical to before. *)
let gen_helpers buf (funcs : helper_func list) =
  if funcs <> [] then begin
    List.iter (gen_helper_proto buf) funcs ;
    Buffer.add_char buf '\n'
  end ;
  List.iter (gen_helper_func buf) funcs

(** {1 Kernel Generation} *)

(* OpenCL f16 refusal. This deliberately does NOT go through
   {!Sarek_ir_codegen.reject_feature}, and the divergence is the point.

   [reject_feature] composes "<backend>: float16 not yet supported (#57 slice
   2 — <hint>)". Both halves of that sentence are false here, and #57 slice 2a
   measured them false rather than inferring it:

   - "not YET supported" describes a queue position. OpenCL is not in the
     queue. The codegen is a two-line change; what is missing is not work.
   - the old hint, "needs cl_khr_fp16 enablement", named the wrong blocker.
     [cl_khr_fp16] is advertised and usable on both local devices. Enabling it
     changes nothing about why f16 is refused.

   The actual blocker: rusticl/radeonsi's ACO backend fuses the f32 multiply
   into the f32->f16 narrowing that consumes it, rounding once where Sarek's f16
   discipline mandates twice, so 620 of the 63488 finite binary16 inputs
   disagree with the interpreter — and no affordable source-level barrier
   exists on this path (measured: FP_CONTRACT OFF, volatile locals, volatile
   private pointers, bitcast round-trips and convert_half_rte all leave it at
   620; HIP's "+v" asm does not compile through SPIR-V). See
   docs/fp-contraction-policy.md and tools/probes/opencl_f16_contraction_probe.c.

   Keeping the shared wording here would have been actively harmful: it would
   tell a reader to go enable an extension that is already enabled, and it would
   file a measured, possibly-permanent refusal under the same heading as three
   backends that genuinely are just unimplemented. The other three keep the
   shared composer precisely so THEY still reword together.

   Named [_kernel] to distinguish it from [Sarek_typer.reject_float16], which
   rejects an f16 OPERAND — a different concept at a different layer. *)
let reject_float16_kernel (k : kernel) : unit =
  if Sarek_ir_analysis.kernel_uses Sarek_ir_analysis.Float16 k then
    Codegen_error.raise_error
      (Codegen_error.unsupported_construct
         "f16"
         Sarek_ir_codegen.opencl_float16_refusal)

(* The cooperative-matrix counterpart, at the same whole-kernel choke point and
   for the same reason the f16 gate is there: a refusal that only fires from
   inside the statement walk names whichever node happened to be reached first,
   while the thing the user has to change is a property of the KERNEL. Note this
   also catches a kernel that merely declares a [TUint8] buffer without ever
   reaching a multiply-add, which the per-statement arm cannot see.

   Deliberately not {!Sarek_ir_codegen.reject_feature}: that composer hardcodes
   "not yet supported (#57 slice 2)", and citing the f16 slice for a
   cooperative-matrix refusal would send a reader to the wrong history. *)
let reject_coopmat_kernel (k : kernel) : unit =
  if Sarek_ir_analysis.kernel_uses Sarek_ir_analysis.Coopmat k then
    Codegen_error.raise_error
      (Codegen_error.unsupported_construct
         "cooperative matrix"
         "OpenCL: the OpenCL backend has no cooperative-matrix path; \
          cooperative matrices and their uint8 operand buffers are emitted \
          only by the Vulkan backend (backlog-62)")

(** {1 Recursion Resolution}

    OpenCL C forbids recursion outright (OpenCL C 1.2 §6.9.e, 3.0 §6.9.5: "the
    OpenCL C programming language does not support recursion"). Unlike an
    undeclared identifier, no compiler in this project's reach diagnoses it:
    [clang -x cl -cl-std=CL1.2 -fsyntax-only] accepts a recursive device
    function silently, and rusticl/radeonsi (Mesa) does not diagnose it either —
    it overflows its own compiler stack inside [libRusticlOpenCL] and takes the
    host process down with SIGSEGV (~30 800 recursive frames on a [clctxworker]
    thread, zero OCaml frames in the backtrace). That crash is backlog #53, and
    its cause is this: [pragma ["sarek.inline N"]] bounds the UNROLLING, not the
    recursion, so the PPX leaves a residual self-call in the IR and this backend
    used to print it verbatim.

    Emitting a self-call and hoping the vendor rejects it is therefore not an
    option, and neither is a blanket refusal: [pragma ["sarek.inline N"]] is an
    advertised feature that the PTX backend already lowers correctly. So this
    pass takes the same two-part policy as PTX
    ({!Sarek_ir_ptx_expr.emit_app_recursive}), which keeps one semantics for one
    pragma across backends:

    - A self-recursive helper carrying [pragma ["sarek.inline N"]] has its
      residual self-calls replaced by a typed zero. The pragma is the author's
      contract that N levels cover every runtime input, so the residual call
      site is dynamically unreachable: it only has to be well-formed OpenCL C,
      never correct. Dropping the arguments is sound because IR expressions are
      pure by construction (see {!Sarek_ir_types.expr}), so there are no
      argument side effects to lose — PTX has to evaluate them only because its
      lowering emits into a register file.
    - Any other cycle — a recursive helper with no pragma, or mutual recursion
      between helpers, which the PPX's self-call-only inliner cannot bound
      anyway — is REFUSED with a located error, the way
      {!Sarek_ir_inline_vec.splice_call} refuses recursion for GLSL/WGSL.

    A partial unroll that silently leaves a self-call is the one outcome ruled
    out: it is neither bounded nor refused. *)

(** Typed zero for the result of a residual (dynamically unreachable) recursive
    call. Aggregates are zeroed field-by-field / through the first constructor,
    so the emitted expression is a real value of the helper's return type. *)
let rec zero_expr (t : elttype) : expr =
  match t with
  | TInt32 -> EConst (CInt32 0l)
  | TInt64 -> EConst (CInt64 0L)
  | TFloat32 -> EConst (CFloat32 0.0)
  | TFloat64 -> EConst (CFloat64 0.0)
  | TBool -> EConst (CBool false)
  | TUnit -> EConst CUnit
  | TRecord (name, fields) ->
      ERecord (name, List.map (fun (f, ft) -> (f, zero_expr ft)) fields)
  | TVariant (name, (cname, payload) :: _) ->
      EVariant (name, cname, List.map zero_expr payload)
  | TVariant (name, []) ->
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "recursion"
           (Printf.sprintf
              "OpenCL: recursive helper returns the uninhabited variant '%s', \
               so its residual call has no value to elide to"
              name))
  | TFloat16 | TArray _ | TVec _ ->
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "recursion"
           "OpenCL: a recursive helper returning an array, vector or f16 \
            cannot have its residual call elided; rewrite it without recursion")
  | TUint8 ->
      (* A zero of this type would be spellable — but a helper cannot return a
         cooperative-matrix operand element in the first place, since nothing
         but [CM_load]/[CM_store] ever touches one. Reaching here means the type
         escaped its intended scope, which is worth a diagnostic rather than a
         plausible-looking literal. *)
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "uint8"
           "OpenCL: uint8 is a cooperative-matrix operand element type, \
            emitted only by the Vulkan backend, and cannot be the return type \
            of a helper")

(** Inline budget declared by [hf], parsed from an [SPragma] at its body root.

    Deliberately a copy of {!Sarek_ir_ptx_expr.helper_inline_budget} rather than
    a shared symbol: both are minimal re-implementations of the option parsing
    in [Sarek_tailrec_pragma.parse_sarek_inline_pragma] (the source of truth for
    the "sarek.inline N" string format), which lives in the PPX — a separate
    library not linkable from codegen. Factoring the two copies together belongs
    with moving the parser into the IR, not with this fix. *)
let helper_inline_budget (hf : helper_func) : int option =
  let checked n_str =
    match int_of_string_opt n_str with
    | Some n when n < 0 ->
        Codegen_error.raise_error
          (Codegen_error.unsupported_construct
             "pragma"
             ("pragma [\"sarek.inline " ^ n_str
            ^ "\"]: the inline depth must be >= 0"))
    | v -> v
  in
  let parse = function
    | [opt] -> (
        match String.split_on_char ' ' opt with
        | ["sarek.inline"; n] -> checked n
        | _ -> None)
    | ["sarek.inline"; n] -> checked n
    | _ -> None
  in
  let rec root = function SBlock s -> root s | s -> s in
  match root hf.hf_body with SPragma (opts, _) -> parse opts | _ -> None

(** Bottom-up rewrite of every expression node reachable from a statement. One
    traversal serves both the call-graph scan (with [f] the identity plus a side
    effect) and the residual-call elision. *)
let rec map_expr (f : expr -> expr) (e : expr) : expr =
  let r = map_expr f in
  let e' =
    match e with
    | EConst _ | EVar _ | EArrayLen _ -> e
    | EBinop (op, a, b) -> EBinop (op, r a, r b)
    | EUnop (op, a) -> EUnop (op, r a)
    | EArrayRead (arr, i) -> EArrayRead (arr, r i)
    | EArrayReadExpr (b, i) -> EArrayReadExpr (r b, r i)
    | ERecordField (a, fld) -> ERecordField (r a, fld)
    | EIntrinsic (path, name, args) -> EIntrinsic (path, name, List.map r args)
    | ECast (t, a) -> ECast (t, r a)
    | ETuple es -> ETuple (List.map r es)
    | EApp (fn, args) -> EApp (r fn, List.map r args)
    | ERecord (n, fields) ->
        ERecord (n, List.map (fun (fl, a) -> (fl, r a)) fields)
    | EVariant (n, c, args) -> EVariant (n, c, List.map r args)
    | EArrayCreate (t, sz, ms) -> EArrayCreate (t, r sz, ms)
    | EIf (c, a, b) -> EIf (r c, r a, r b)
    | EMatch (s, cases) -> EMatch (r s, List.map (fun (p, a) -> (p, r a)) cases)
  in
  f e'

let rec map_lvalue f (lv : lvalue) : lvalue =
  match lv with
  | LVar _ -> lv
  | LArrayElem (a, i) -> LArrayElem (a, map_expr f i)
  | LArrayElemExpr (b, i) -> LArrayElemExpr (map_expr f b, map_expr f i)
  | LRecordField (b, fld) -> LRecordField (map_lvalue f b, fld)

let rec map_stmt (f : expr -> expr) (s : stmt) : stmt =
  let e = map_expr f and r = map_stmt f in
  match s with
  | SBarrier | SWarpBarrier | SEmpty | SMemFence | SNative _ -> s
  | SAssign (lv, x) -> SAssign (map_lvalue f lv, e x)
  | SSeq ss -> SSeq (List.map r ss)
  | SIf (c, t, el) -> SIf (e c, r t, Option.map r el)
  | SWhile (c, b) -> SWhile (e c, r b)
  | SFor (v, lo, hi, d, b) -> SFor (v, e lo, e hi, d, r b)
  | SMatch (sc, cases) -> SMatch (e sc, List.map (fun (p, b) -> (p, r b)) cases)
  | SReturn x -> SReturn (e x)
  | SExpr x -> SExpr (e x)
  | SLet (v, x, b) -> SLet (v, e x, r b)
  | SLetMut (v, x, b) -> SLetMut (v, e x, r b)
  | SPragma (h, b) -> SPragma (h, r b)
  | SBlock b -> SBlock (r b)
  | SCoopmat op ->
      (* This walk is used both to SCAN for helper calls and to REWRITE residual
         ones, so it must descend into every expression a statement holds — the
         index and the stride here. Fragment and buffer names are not
         expressions and stay as they are; substituting one would replace a
         named buffer with a term [CM_load] has no field to hold. *)
      SCoopmat
        (match op with
        | CM_decl _ | CM_muladd _ -> op
        | CM_load r -> CM_load {r with index = e r.index; stride = e r.stride}
        | CM_store r -> CM_store {r with index = e r.index; stride = e r.stride})

(** Names of helper functions called (directly) from [s]. *)
let called_helpers (helpers : string list) (s : stmt) : string list =
  let acc = ref [] in
  ignore
    (map_stmt
       (fun x ->
         (match x with
         | EApp (EVar v, _) when List.mem v.var_name helpers ->
             if not (List.mem v.var_name !acc) then acc := v.var_name :: !acc
         | _ -> ()) ;
         x)
       s) ;
  !acc

(** Every helper name reachable from [start] through the call graph [edges]. *)
let reachable (edges : (string * string list) list) (start : string) :
    string list =
  let seen = ref [] in
  let rec go n =
    if not (List.mem n !seen) then begin
      seen := n :: !seen ;
      List.iter go (try List.assoc n edges with Not_found -> [])
    end
  in
  List.iter go (try List.assoc start edges with Not_found -> []) ;
  !seen

let refuse_recursion name detail =
  Codegen_error.raise_error
    (Codegen_error.unsupported_construct
       "recursion"
       (Printf.sprintf
          "OpenCL: helper '%s' is %s. OpenCL C forbids recursion (OpenCL C 1.2 \
           §6.9.e), and no vendor compiler on this path diagnoses it — \
           rusticl/radeonsi crashes on it instead of rejecting it (#53/#127). \
           Annotate the helper body with pragma [\"sarek.inline N\"] for a \
           depth-bounded self-recursion, or rewrite it without recursion."
          name
          detail))

(** Replace residual self-calls in budgeted self-recursive helpers by a typed
    zero, and refuse every other cycle. Post-condition (asserted below): the
    returned kernel's helper call graph is acyclic, so no [EApp] this backend
    emits can ever be a recursive call. *)
let resolve_recursive_helpers (k : kernel) : kernel =
  let names = List.map (fun hf -> hf.hf_name) k.kern_funcs in
  let edges =
    List.map
      (fun hf -> (hf.hf_name, called_helpers names hf.hf_body))
      k.kern_funcs
  in
  let on_cycle n = List.mem n (reachable edges n) in
  let funcs =
    List.map
      (fun hf ->
        if not (on_cycle hf.hf_name) then hf
        else
          (* Mutual recursion: the PPX inliner only rewrites SELF-calls
             (Sarek_tailrec_analysis.is_self_call), so no pragma can bound this
             — refuse before looking at the budget.

             "Mutually recursive with [hf]" means SAME SCC: reachable from [hf]
             AND able to reach [hf] back. Testing only [on_cycle n] would be an
             over-approximation — it holds for any callee sitting on a cycle of
             its own — and that is not a cosmetic difference here. This
             backend's whole policy turns on the self-vs-other distinction:
             budgeted self-recursion is elided to a typed zero, everything else
             is refused. So a budgeted self-recursive [f] that merely calls an
             independently self-recursive [g] would have been REFUSED as
             "mutually recursive with 'g'" even though the two cycles never
             touch — a false refusal on a case the inliner can bound, i.e. a
             regression for anyone using the pragma. Each of [f] and [g] is
             resolved by its own iteration of this map. *)
          let direct = try List.assoc hf.hf_name edges with Not_found -> [] in
          let through =
            List.filter (fun n -> n <> hf.hf_name) (reachable edges hf.hf_name)
          in
          let mutual =
            List.filter
              (fun n -> List.mem hf.hf_name (reachable edges n))
              through
          in
          if mutual <> [] then
            refuse_recursion
              hf.hf_name
              (Printf.sprintf
                 "mutually recursive with %s"
                 (String.concat ", " (List.map (fun n -> "'" ^ n ^ "'") mutual)))
          else if not (List.mem hf.hf_name direct) then
            refuse_recursion hf.hf_name "recursive"
          else
            match helper_inline_budget hf with
            | None ->
                refuse_recursion
                  hf.hf_name
                  "self-recursive with no inline bound"
            | Some _ ->
                let zero = zero_expr hf.hf_ret_type in
                let body =
                  map_stmt
                    (function
                      | EApp (EVar v, _) when v.var_name = hf.hf_name -> zero
                      | x -> x)
                    hf.hf_body
                in
                {hf with hf_body = body})
      k.kern_funcs
  in
  let k = {k with kern_funcs = funcs} in
  (* Post-condition. Cheap, and it is the only thing standing between a future
     change to the elision above and another silent SIGSEGV inside the vendor
     compiler. *)
  let edges' =
    List.map (fun hf -> (hf.hf_name, called_helpers names hf.hf_body)) funcs
  in
  List.iter
    (fun hf ->
      if List.mem hf.hf_name (reachable edges' hf.hf_name) then
        refuse_recursion
          hf.hf_name
          "still recursive after residual-call elision (internal invariant \
           violated)")
    funcs ;
  k

(** Generate complete OpenCL source for a kernel *)
let generate (k : kernel) : string =
  reject_float16_kernel k ;
  reject_coopmat_kernel k ;
  let k = resolve_recursive_helpers k in
  let buf = Buffer.create large_buffer_size in

  (* Generate helper functions before kernel *)
  gen_helpers buf k.kern_funcs ;

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
  reject_float16_kernel k ;
  reject_coopmat_kernel k ;
  let k = resolve_recursive_helpers k in
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
  gen_helpers buf k.kern_funcs ;

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
