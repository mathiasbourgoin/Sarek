(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_ir_glsl - GLSL Compute Shader Generation from Sarek IR
 *
 * Generates GLSL compute shader source code from Sarek_ir.kernel.
 * The output is compatible with Vulkan compute shaders and can be compiled
 * to SPIR-V using glslangValidator.
 *
 * Features:
 * - Direct generation from clean IR
 * - Storage buffer bindings for arrays
 * - Record/variant type support with GLSL structs
 * - Workgroup size configuration
 ******************************************************************************)

open Sarek_ir_types

(** Local error module — same raised exception as the package-level
    [Vulkan_error]. *)
module Codegen_error = Sarek_backend_error.Backend_error.Make (struct
  let name = "Vulkan"
end)

(** Current kernel's variant definitions (set during generate) *)
let current_variants : (string * (string * elttype list) list) list ref = ref []

(** Name of the integer-remainder helper ([sarek_smod] by default) for the
    current kernel, set per-kernel during generate by {!compute_smod_name} so it
    cannot collide with a user identifier. Both the declaration
    ({!gen_smod_helper}) and the call site ({!gen_expr}'s [Mod] arm) read this,
    guaranteeing they agree. Mirrors the per-kernel {!current_variants} state.
*)
let current_smod_name = ref "sarek_smod"

(** Name of the sign-copy helper ([sarek_copysign] by default) for the current
    kernel, set per-kernel during generate by {!compute_copysign_name} so it
    cannot collide with a user identifier. Both the declaration
    ({!gen_copysign_helper}) and the call site ({!gen_intrinsic}'s [copysign]
    arm) read this, guaranteeing they agree. Mirrors {!current_smod_name}. *)
let current_copysign_name = ref "sarek_copysign"

(** Helper function vector parameter indices - maps function name to set of
    parameter indices that are vectors. In GLSL, vectors cannot be passed as
    function parameters, so these must be filtered out at call sites. *)
let helper_vec_param_indices : (string, int list) Hashtbl.t = Hashtbl.create 16

(** {1 Type Mapping} *)

let mangle_name = Sarek_ir_codegen.mangle_name

(** GLSL reserved keywords that cannot be used as identifiers *)
let glsl_reserved_keywords =
  [
    (* Storage qualifiers *)
    "input";
    "output";
    "uniform";
    "buffer";
    "shared";
    "attribute";
    "varying";
    "const";
    (* Types *)
    "void";
    "bool";
    "int";
    "uint";
    "float";
    "double";
    "vec2";
    "vec3";
    "vec4";
    "ivec2";
    "ivec3";
    "ivec4";
    "uvec2";
    "uvec3";
    "uvec4";
    "bvec2";
    "bvec3";
    "bvec4";
    "mat2";
    "mat3";
    "mat4";
    "sampler2D";
    "sampler3D";
    "samplerCube";
    (* Control flow *)
    "if";
    "else";
    "for";
    "while";
    "do";
    "switch";
    "case";
    "default";
    "break";
    "continue";
    "return";
    "discard";
    (* Other reserved *)
    "true";
    "false";
    "struct";
    "layout";
    "in";
    "out";
    "inout";
    "lowp";
    "mediump";
    "highp";
    "precision";
    "invariant";
    "flat";
    "smooth";
    "centroid";
    "noperspective";
    "patch";
    "sample";
    "subroutine";
    "common";
    "partition";
    "active";
    "asm";
    "class";
    "union";
    "enum";
    "typedef";
    "template";
    "this";
    "packed";
    "goto";
    "inline";
    "noinline";
    "volatile";
    "public";
    "static";
    "extern";
    "external";
    "interface";
    "long";
    "short";
    "half";
    "fixed";
    "unsigned";
    "superp";
    "cast";
    "namespace";
    "using";
    "row_major";
    "gl_FragCoord";
    "gl_FragColor";
    "main";
  ]

(** Escape reserved GLSL keywords by adding 'v' suffix (avoids double underscore
    with _len) *)
let escape_glsl_name name =
  if List.mem name glsl_reserved_keywords then name ^ "v" else name

(** Map Sarek IR element type to GLSL type string *)
let rec glsl_type_of_elttype = function
  | TInt32 -> "int"
  | TInt64 -> "int64_t" (* Requires GL_ARB_gpu_shader_int64 *)
  | TFloat32 -> "float"
  | TFloat64 -> "double" (* Requires GL_ARB_gpu_shader_fp64 *)
  | TBool -> "bool"
  | TUnit -> "void"
  | TRecord (name, _) -> mangle_name name
  | TVariant (name, _) -> mangle_name name
  | TArray (elt, _) -> glsl_type_of_elttype elt (* Arrays are special in GLSL *)
  | TVec elt -> glsl_type_of_elttype elt

(** {1 Thread Intrinsics} *)

let glsl_thread_intrinsic = function
  | "thread_id_x" | "thread_idx_x" -> "int(gl_LocalInvocationID.x)"
  | "thread_id_y" | "thread_idx_y" -> "int(gl_LocalInvocationID.y)"
  | "thread_id_z" | "thread_idx_z" -> "int(gl_LocalInvocationID.z)"
  | "block_id_x" | "block_idx_x" -> "int(gl_WorkGroupID.x)"
  | "block_id_y" | "block_idx_y" -> "int(gl_WorkGroupID.y)"
  | "block_id_z" | "block_idx_z" -> "int(gl_WorkGroupID.z)"
  | "block_dim_x" -> "int(gl_WorkGroupSize.x)"
  | "block_dim_y" -> "int(gl_WorkGroupSize.y)"
  | "block_dim_z" -> "int(gl_WorkGroupSize.z)"
  | "grid_dim_x" -> "int(gl_NumWorkGroups.x)"
  | "grid_dim_y" -> "int(gl_NumWorkGroups.y)"
  | "grid_dim_z" -> "int(gl_NumWorkGroups.z)"
  | "global_thread_id" | "global_idx" | "global_idx_x" ->
      "int(gl_GlobalInvocationID.x)"
  | "global_idx_y" -> "int(gl_GlobalInvocationID.y)"
  | "global_idx_z" -> "int(gl_GlobalInvocationID.z)"
  | "global_size" -> "int(gl_WorkGroupSize.x * gl_NumWorkGroups.x)"
  | name -> Codegen_error.raise_error (Codegen_error.unknown_intrinsic name)

(** {1 Expression Generation} *)

let rec gen_expr buf = function
  | EConst (CInt32 n) -> Buffer.add_string buf (Int32.to_string n)
  | EConst (CInt64 n) -> Buffer.add_string buf (Int64.to_string n ^ "L")
  | EConst (CFloat32 f) ->
      let s = Printf.sprintf "%.17g" f in
      let s =
        if String.contains s '.' || String.contains s 'e' then s else s ^ ".0"
      in
      Buffer.add_string buf s
  | EConst (CFloat64 f) ->
      let s = Printf.sprintf "%.17g" f in
      let s =
        if String.contains s '.' || String.contains s 'e' then s else s ^ ".0"
      in
      Buffer.add_string buf (s ^ "lf")
  | EConst (CBool true) -> Buffer.add_string buf "true"
  | EConst (CBool false) -> Buffer.add_string buf "false"
  | EConst CUnit -> Buffer.add_string buf "/* unit */"
  | EVar v -> Buffer.add_string buf (escape_glsl_name v.var_name)
  | EBinop (Mod, e1, e2) ->
      (* Integer remainder with C (dividend-signed, truncated) semantics.
         GLSL's [%] is undefined for negative operands and lowers to OpSMod
         (divisor-signed) on RADV, so [-7 % 2] yields +1 instead of C's -1.
         Delegate to the [sarek_smod] helper (emitted in the preamble by
         [gen_smod_helper]) rather than inlining [a - b * (a / b)]: a GLSL
         function call evaluates each argument exactly once, so operands that
         carry side effects (a value-returning atomic intrinsic, an effectful
         helper call) fire once - inlining the arithmetic form would emit each
         operand twice and double any such effect. Float [mod] never reaches
         this arm (the frontend lowers it to the [fmod]/[mod] intrinsic, not
         [Ir.Mod]). *)
      Buffer.add_string buf !current_smod_name ;
      Buffer.add_char buf '(' ;
      gen_expr buf e1 ;
      Buffer.add_string buf ", " ;
      gen_expr buf e2 ;
      Buffer.add_char buf ')'
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
      Buffer.add_string buf (escape_glsl_name arr) ;
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
      Buffer.add_string buf (glsl_type_of_elttype ty) ;
      Buffer.add_char buf '(' ;
      gen_expr buf e ;
      Buffer.add_char buf ')'
  | ETuple exprs ->
      (* Tuples become struct literals in GLSL *)
      Buffer.add_string buf "{" ;
      List.iteri
        (fun i e ->
          if i > 0 then Buffer.add_string buf ", " ;
          gen_expr buf e)
        exprs ;
      Buffer.add_string buf "}"
  | EApp (fn, args) ->
      (* Extract function name to check for vector parameter filtering *)
      let fn_name = match fn with EVar v -> Some v.var_name | _ -> None in
      let vec_indices =
        match fn_name with
        | Some name -> Hashtbl.find_opt helper_vec_param_indices name
        | None -> None
      in
      gen_expr buf fn ;
      Buffer.add_char buf '(' ;
      let filtered_args =
        match vec_indices with
        | Some indices ->
            (* Filter out vector arguments at registered indices *)
            List.mapi (fun i e -> (i, e)) args
            |> List.filter (fun (i, _) -> not (List.mem i indices))
            |> List.map snd
        | None -> args
      in
      List.iteri
        (fun i e ->
          if i > 0 then Buffer.add_string buf ", " ;
          gen_expr buf e)
        filtered_args ;
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
      Buffer.add_string buf ("sarek_" ^ escape_glsl_name arr ^ "_length")
  | EArrayCreate _ ->
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "EArrayCreate"
           "should be handled in gen_stmt SLet")
  | EIf (cond, then_, else_) ->
      Buffer.add_char buf '(' ;
      gen_expr buf cond ;
      Buffer.add_string buf " ? " ;
      gen_expr buf then_ ;
      Buffer.add_string buf " : " ;
      gen_expr buf else_ ;
      Buffer.add_char buf ')'
  | EMatch (_, []) ->
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct "match" "empty match expression")
  | EMatch (_, [(_, body)]) -> gen_expr buf body
  | EMatch (e, cases) ->
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
            | PWild -> Buffer.add_string buf "true") ;
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
  (* [Mod] is intercepted by [gen_expr] and lowered to the [sarek_smod] helper
     call; this arm is unreachable and kept only for match exhaustiveness. *)
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

(** GLSL has no [cbrt]/[hypot]/[expm1]/[log1p]/[log10] builtins under any name
    (unlike [fabs]/[rsqrt]/[atan2], which are simple renames — see
    [Sarek_pure_registry.glsl_override_name]). These need a multi-token
    expression instead of a function-name substitution, so they're special-cased
    here ahead of both the unqualified match arms and the pure registry,
    applying uniformly to qualified (Float32.cbrt) and unqualified calls alike.
    [cbrt] uses [sign(x)*pow(abs(x),...)] rather than bare [pow] because GLSL's
    [pow] is undefined for a negative base. [log10] is derived from the natural
    [log] builtin: GLSL exposes [log] (base e) and [log2] but no base-10 form,
    so [log10(x) = log(x)/log(10)]. Routing [log10] through here (ahead of the
    pure registry) is required: [log10] IS present in the pure-registry
    float32/float64 tables and would otherwise emit the raw un-suffixed
    [log10(...)] that glslang rejects — the same latent-invalid-GLSL class as
    [fabs]/[copysign] (#246/#256).

    [is_f64] carries the operand precision (derived from the intrinsic path — a
    [Float64] component). It governs the precision of any numeric LITERAL that a
    builtin then computes on: on the [double] route the literal must carry the
    GLSL double suffix (GLSL 4.5 §4.1.4), otherwise it defaults to [float] and
    pins the result to single precision even where the [double] overload of the
    surrounding builtin exists. The suffix is spelled lowercase [lf] to match
    the generator's own [EConst (CFloat64 _)] output (see {!gen_expr}); GLSL
    accepts [lf] and [LF] interchangeably, so a single casing keeps every double
    literal in the emitted shader uniform. This bites exactly the two polyfills
    that feed a literal into a transcendental / irrational computation:
    - [log10]: [log(10.0)] — an irrational, evaluated by the [float] [log]
      overload; [log(10.0lf)] uses the [double] overload.
    - [cbrt]: the exponent [1.0/3.0] — a non-terminating fraction; [1.0/3.0] is
      a [float] division (~7 digits), [1.0lf/3.0lf] a [double] one. [hypot] (no
      literal), [expm1] and [log1p] (their only literal is [1.0], which is
      exactly representable and promotes to [double] losslessly, and is never
      itself the argument of a builtin) are precision-safe as-is and left
      byte-for-byte unchanged so their existing goldens do not move. *)
and gen_glsl_polyfill buf ~is_f64 name args =
  let flit s = if is_f64 then s ^ "lf" else s in
  match (name, args) with
  | "log10", [x] ->
      Buffer.add_string buf "(log(" ;
      gen_expr buf x ;
      Buffer.add_string buf (Printf.sprintf ") / log(%s))" (flit "10.0"))
  | "cbrt", [x] ->
      Buffer.add_string buf "(sign(" ;
      gen_expr buf x ;
      Buffer.add_string buf ") * pow(abs(" ;
      gen_expr buf x ;
      Buffer.add_string
        buf
        (Printf.sprintf "), %s / %s))" (flit "1.0") (flit "3.0"))
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
           (Printf.sprintf "%s (wrong arity for GLSL polyfill)" name))

and gen_intrinsic buf path name args =
  let full_name =
    match path with [] -> name | _ -> String.concat "." path ^ "." ^ name
  in
  if List.mem name ["cbrt"; "hypot"; "expm1"; "log1p"; "log10"] then
    (* A [Float64] path component marks the double-precision route; the plain
       [Float32]/[Math.Float32] and unqualified (core-primitive, f32-typed)
       spellings stay single-precision. This mirrors how the pure registry keys
       the same intrinsics by their [Float64] vs [Float32] module path. *)
    let is_f64 = List.mem "Float64" path in
    gen_glsl_polyfill buf ~is_f64 name args
  else if name = "copysign" then (
    (* GLSL has no [copysign] builtin under any name, and [abs(x)*sign(y)] is
       wrong for [y=0] (GLSL [sign(0)=0]) and the [x=0]/NaN sign-transfer edge
       cases. Lower to the bit-level [sarek_copysign] helper emitted in the
       preamble by [gen_copysign_helper], ahead of both the pure registry
       (which would emit the raw un-suffixed [copysign(...)] for
       [Float32.copysign]) and the unqualified match arms (where
       [Float64.copysign] would fall through to the raw-name fallback and emit
       the swizzle-parsed [Float64.copysign(...)] that glslang rejects).
       Routing through a function (not inlining the bit twiddling) evaluates
       each argument exactly once — the same single-eval guarantee as the
       [sarek_smod] helper. *)
    Buffer.add_string buf !current_copysign_name ;
    Buffer.add_char buf '(' ;
    List.iteri
      (fun i e ->
        if i > 0 then Buffer.add_string buf ", " ;
        gen_expr buf e)
      args ;
    Buffer.add_char buf ')')
  else
    (* For path-qualified intrinsics, query the pure registry first.
     Float32.sin -> sin on GLSL (GLSL uses un-suffixed names). *)
    let pure_registry_hit =
      match path with
      | [] -> None
      | _ -> (
          match
            Sarek_pure_registry.fun_device_template ~module_path:path name
          with
          | Some f -> Some (f ~framework:"GLSL")
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
        then Buffer.add_string buf (glsl_thread_intrinsic name)
        else
          (* Standard math intrinsics - GLSL versions *)
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
          | "fabs" | "abs_float" ->
              (* GLSL has no [fabs]; its abs() has an fp64 overload under
                 GL_ARB_gpu_shader_fp64 (enabled on the f64 path). [abs_float]
                 (Float64.abs_float) reaches here rather than via the pure
                 registry — it is absent from [float64_list] because that list's
                 generic template would emit the raw name on CUDA/OpenCL, which
                 need [fabs]. See spoc/ir/Sarek_pure_registry.ml. *)
              Buffer.add_string buf "abs" ;
              Buffer.add_char buf '(' ;
              List.iteri
                (fun i e ->
                  if i > 0 then Buffer.add_string buf ", " ;
                  gen_expr buf e)
                args ;
              Buffer.add_char buf ')'
          | "rsqrt" ->
              Buffer.add_string buf "inversesqrt" ;
              Buffer.add_char buf '(' ;
              List.iteri
                (fun i e ->
                  if i > 0 then Buffer.add_string buf ", " ;
                  gen_expr buf e)
                args ;
              Buffer.add_char buf ')'
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
          (* Barrier synchronization *)
          | "block_barrier" -> Buffer.add_string buf "barrier()"
          (* Atomic operations - GLSL uses atomicAdd etc. *)
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
              Buffer.add_string buf "float(" ;
              (match args with [e] -> gen_expr buf e | _ -> ()) ;
              Buffer.add_char buf ')'
          | "int_of_float" ->
              Buffer.add_string buf "int(" ;
              (match args with [e] -> gen_expr buf e | _ -> ()) ;
              Buffer.add_char buf ')'
          | _ ->
              (* No GLSL lowering for this intrinsic. Unlike CUDA/OpenCL/Metal,
                 GLSL does NOT fall back to [Sarek_registry] (the FFI registry):
                 its device closures only branch CUDA-vs-OpenCL and have no GLSL
                 arm, so consulting it would splice OpenCL C — [get_global_id],
                 [barrier(CLK_LOCAL_MEM_FENCE)], [(float)] casts — into GLSL,
                 which glslang rejects just as cryptically as the old behaviour
                 of emitting the raw OCaml path [full_name(...)] ("vector
                 swizzle too long"). Raise a located error naming the intrinsic
                 and backend instead — strictly better than emitting garbage.

                 To give a future intrinsic a real GLSL lowering, extend one of
                 (a) [Sarek_pure_registry.glsl_override_name] for a plain rename,
                 (b) [gen_glsl_polyfill] above for a multi-token expression, or
                 (c) an explicit match arm here. The pure registry is already
                 GLSL-parameterised (it receives [~framework:"GLSL"]), so no
                 registry-type change is needed and existing registrations are
                 untouched. *)
              Codegen_error.raise_error
                (Codegen_error.unknown_intrinsic full_name))

(** {1 L-value Generation} *)

let rec gen_lvalue buf = function
  | LVar v -> Buffer.add_string buf (escape_glsl_name v.var_name)
  | LArrayElem (arr, idx) ->
      Buffer.add_string buf (escape_glsl_name arr) ;
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

(** Nested indentation level *)
let indent_nested indent = indent ^ "  "

(** Generate match case pattern with variable bindings *)
and gen_match_pattern buf indent scrutinee cname bindings find_constr_types =
  Buffer.add_string buf ("  case " ^ cname ^ ": {\n") ;
  match (bindings, find_constr_types cname) with
  | [var_name], Some [ty] ->
      let vn = escape_glsl_name var_name in
      Buffer.add_string buf (indent ^ "    ") ;
      Buffer.add_string buf (glsl_type_of_elttype ty) ;
      Buffer.add_string buf " " ;
      Buffer.add_string buf vn ;
      Buffer.add_string buf " = " ;
      Buffer.add_string buf scrutinee ;
      Buffer.add_char buf '.' ;
      Buffer.add_string buf cname ;
      Buffer.add_string buf "_v;\n"
  | vars, Some types when List.length vars = List.length types ->
      List.iteri
        (fun i (var_name, ty) ->
          let vn = escape_glsl_name var_name in
          Buffer.add_string buf (indent ^ "    ") ;
          Buffer.add_string buf (glsl_type_of_elttype ty) ;
          Buffer.add_string buf " " ;
          Buffer.add_string buf vn ;
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

(** Generate variable declaration with optional initialization *)
and gen_var_decl buf indent v_name v_type init_expr =
  let vn = escape_glsl_name v_name in
  Buffer.add_string buf indent ;
  (* [precise] forbids contraction/reassociation on float locals (SPIR-V
     NoContraction). Without it some drivers (observed: RADV via glslang)
     simplify error-free transformations like Dekker/Knuth TwoSum, breaking
     algorithms that rely on IEEE-exact rounding of each operation. This
     matches CUDA/OpenCL codegen semantics (no fast-math contraction). *)
  (match v_type with
  | TFloat32 | TFloat64 -> Buffer.add_string buf "precise "
  | _ -> ()) ;
  Buffer.add_string buf (glsl_type_of_elttype v_type) ;
  Buffer.add_char buf ' ' ;
  Buffer.add_string buf vn ;
  Buffer.add_string buf " = " ;
  gen_expr buf init_expr ;
  Buffer.add_string buf ";\n"

(** Generate array declaration *)
and gen_array_decl buf indent v_name elem_ty size =
  let vn = escape_glsl_name v_name in
  Buffer.add_string buf indent ;
  Buffer.add_string buf (glsl_type_of_elttype elem_ty) ;
  Buffer.add_char buf ' ' ;
  Buffer.add_string buf vn ;
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
      let op, incr =
        match dir with Upto -> ("<=", "++") | Downto -> (">=", "--")
      in
      let loop_var = escape_glsl_name v.var_name in
      Buffer.add_string buf indent ;
      Buffer.add_string buf "for (" ;
      Buffer.add_string buf (glsl_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf loop_var ;
      Buffer.add_string buf " = " ;
      gen_expr buf start ;
      Buffer.add_string buf "; " ;
      Buffer.add_string buf loop_var ;
      Buffer.add_string buf (" " ^ op ^ " ") ;
      gen_expr buf stop ;
      Buffer.add_string buf "; " ;
      Buffer.add_string buf loop_var ;
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
      Buffer.add_string buf "barrier();\n"
  | SWarpBarrier ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "subgroupBarrier();\n"
  | SMemFence ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "memoryBarrier();\n"
  | SNative _ ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "/* native code not supported in GLSL */\n"
  | SExpr e ->
      Buffer.add_string buf indent ;
      gen_expr buf e ;
      Buffer.add_string buf ";\n"
  | SLet (_v, EArrayCreate (_, _, Shared), body) ->
      (* Shared declarations are hoisted to module scope, so just emit the body *)
      gen_stmt buf indent body
  | SLet (v, EArrayCreate (elem_ty, size, _), body) ->
      gen_array_decl buf indent v.var_name elem_ty size ;
      gen_stmt buf indent body
  | SLet (v, e, body) ->
      gen_var_decl buf indent v.var_name v.var_type e ;
      gen_stmt buf indent body
  | SLetMut (v, e, body) ->
      gen_var_decl buf indent v.var_name v.var_type e ;
      gen_stmt buf indent body
  | SPragma (_hints, body) ->
      (* GLSL doesn't have #pragma in the same way *)
      gen_stmt buf indent body
  | SBlock body ->
      Buffer.add_string buf indent ;
      Buffer.add_string buf "{\n" ;
      gen_stmt buf (indent_nested indent) body ;
      Buffer.add_string buf indent ;
      Buffer.add_string buf "}\n"

(** {1 Helper Function Generation} *)

(** Generate helper function with #undef/#define guards to avoid macro
    collisions. Push constant macros (e.g., #define max_iter pc.max_iter) would
    otherwise expand function parameters with the same name, causing syntax
    errors.
    @param pc_names Set of push constant names that have macros defined *)
let gen_helper_func ~pc_names buf (hf : helper_func) =
  (* Filter out vector parameters - in GLSL, buffer arrays can't be passed as
     function parameters. They are accessed directly via global buffer names. *)
  let vec_indices =
    List.mapi (fun i (v : var) -> (i, v)) hf.hf_params
    |> List.filter_map (fun (i, v) ->
        match v.var_type with TVec _ -> Some i | _ -> None)
  in
  (* Register vector param indices for call site filtering *)
  Hashtbl.replace helper_vec_param_indices hf.hf_name vec_indices ;
  let non_vec_params =
    List.filter
      (fun (v : var) -> match v.var_type with TVec _ -> false | _ -> true)
      hf.hf_params
  in
  (* Find parameter names that collide with push constant macros *)
  let param_names =
    List.map (fun (v : var) -> escape_glsl_name v.var_name) non_vec_params
  in
  let colliding_names =
    List.filter (fun name -> List.mem name pc_names) param_names
  in
  (* #undef colliding names before the function *)
  List.iter
    (fun name -> Buffer.add_string buf (Printf.sprintf "#undef %s\n" name))
    colliding_names ;
  (* Generate function *)
  Buffer.add_string buf (glsl_type_of_elttype hf.hf_ret_type) ;
  Buffer.add_char buf ' ' ;
  Buffer.add_string buf hf.hf_name ;
  Buffer.add_char buf '(' ;
  List.iteri
    (fun i (v : var) ->
      if i > 0 then Buffer.add_string buf ", " ;
      Buffer.add_string buf (glsl_type_of_elttype v.var_type) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf (escape_glsl_name v.var_name))
    non_vec_params ;
  Buffer.add_string buf ") {\n" ;
  gen_stmt buf "  " hf.hf_body ;
  Buffer.add_string buf "}\n" ;
  (* Re-#define the colliding macros after the function *)
  List.iter
    (fun name ->
      Buffer.add_string buf (Printf.sprintf "#define %s pc.%s\n" name name))
    colliding_names ;
  Buffer.add_char buf '\n'

(** {1 Kernel Generation} *)

(** Count vector parameters for binding assignment *)
let count_vec_params params =
  List.fold_left
    (fun acc decl ->
      match decl with
      | DParam (v, _) -> ( match v.var_type with TVec _ -> acc + 1 | _ -> acc)
      | _ -> acc)
    0
    params

(** Generate GLSL compute shader header.
    @param block Optional workgroup dimensions (x, y, z). Defaults to 256x1x1.
    @param uses_float64
      Whether the kernel uses [double] (Sarek [float64]) anywhere. When [true],
      emits [#extension GL_ARB_gpu_shader_fp64 : require]. glslang does not
      strictly require this pragma to compile [double] under [#version 450]
      targeting SPIR-V (it auto-adds the SPIR-V [Float64] capability), but
      declaring it explicitly keeps the generated source correct against
      stricter/non-glslang GLSL compilers and documents the requirement in the
      emitted shader. Defaults to [false] so kernels that do not use float64
      never carry the extension. *)
let glsl_header ~kernel_name ?(block = (256, 1, 1)) ?(uses_float64 = false) () =
  let bx, by, bz = block in
  let fp64_extension =
    if uses_float64 then "#extension GL_ARB_gpu_shader_fp64 : require\n" else ""
  in
  Printf.sprintf
    {|#version 450
%s
// Sarek-generated compute shader: %s
layout(local_size_x = %d, local_size_y = %d, local_size_z = %d) in;

|}
    fp64_extension
    kernel_name
    bx
    by
    bz

(** Generate buffer binding for a vector parameter *)
let gen_buffer_binding buf binding_idx v elem_type =
  let name = escape_glsl_name v.var_name in
  Buffer.add_string
    buf
    (Printf.sprintf
       "layout(std430, set=0, binding = %d) buffer Buffer_%s {\n"
       binding_idx
       name) ;
  Buffer.add_string
    buf
    (Printf.sprintf "  %s %s[];\n" (glsl_type_of_elttype elem_type) name) ;
  Buffer.add_string buf "};\n"

let gen_push_constants buf params =
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
  let vectors = List.rev !vectors in
  let scalars = List.rev !scalars in
  (* Generate push constants if we have vectors (for lengths) or scalars *)
  if vectors <> [] || scalars <> [] then begin
    Buffer.add_string buf "layout(push_constant) uniform PushConstants {\n" ;
    (* Add length parameter for each vector *)
    List.iter
      (fun v ->
        let name = escape_glsl_name v.var_name in
        Buffer.add_string buf (Printf.sprintf "  int %s_len;\n" name))
      vectors ;
    (* Add user-defined scalar parameters *)
    List.iter
      (fun v ->
        let name = escape_glsl_name v.var_name in
        Buffer.add_string
          buf
          (Printf.sprintf "  %s %s;\n" (glsl_type_of_elttype v.var_type) name))
      scalars ;
    Buffer.add_string buf "} pc;\n\n" ;
    (* Define convenience aliases for push constants *)
    List.iter
      (fun v ->
        let name = escape_glsl_name v.var_name in
        Buffer.add_string
          buf
          (Printf.sprintf "#define %s_len pc.%s_len\n" name name))
      vectors ;
    List.iter
      (fun v ->
        let name = escape_glsl_name v.var_name in
        Buffer.add_string buf (Printf.sprintf "#define %s pc.%s\n" name name))
      scalars ;
    Buffer.add_string buf "\n"
  end

(** Collect shared array declarations from a statement tree. Returns list of
    (name, elem_type, size_expr) *)
let rec collect_shared_decls (s : stmt) : (string * elttype * expr) list =
  match s with
  | SLet (v, EArrayCreate (elem_ty, size, Shared), body) ->
      (escape_glsl_name v.var_name, elem_ty, size) :: collect_shared_decls body
  | SLet (_, _, body) | SLetMut (_, _, body) -> collect_shared_decls body
  | SSeq stmts -> List.concat_map collect_shared_decls stmts
  | SFor (_, _, _, _, body) -> collect_shared_decls body
  | SWhile (_, body) -> collect_shared_decls body
  | SIf (_, st, sf_opt) ->
      let sf_decls =
        match sf_opt with Some sf -> collect_shared_decls sf | None -> []
      in
      collect_shared_decls st @ sf_decls
  | SBlock body -> collect_shared_decls body
  | SPragma (_, body) -> collect_shared_decls body
  | SMatch (_, cases) ->
      List.concat_map (fun (_, body) -> collect_shared_decls body) cases
  | SEmpty | SBarrier | SWarpBarrier | SMemFence | SNative _ | SExpr _
  | SAssign _ | SReturn _ ->
      []

(** Generate shared declarations at module scope *)
let gen_shared_decls buf (decls : (string * elttype * expr) list) =
  if decls <> [] then begin
    Buffer.add_string buf "// Shared memory\n" ;
    List.iter
      (fun (name, elem_ty, size) ->
        Buffer.add_string buf "shared " ;
        Buffer.add_string buf (glsl_type_of_elttype elem_ty) ;
        Buffer.add_char buf ' ' ;
        Buffer.add_string buf name ;
        Buffer.add_char buf '[' ;
        gen_expr buf size ;
        Buffer.add_string buf "];\n")
      decls ;
    Buffer.add_char buf '\n'
  end

(** Emit the [sarek_smod] integer-remainder helper when the kernel uses [mod].

    Integer [Mod] is lowered (in [gen_expr]) to a call to this helper rather
    than to the GLSL [%] operator: [%] is undefined for negative operands and
    lowers to OpSMod (divisor-signed) on RADV, giving [-7 % 2 = +1] instead of
    C's [-1]. The helper computes the C-truncated, dividend-signed remainder as
    [a - b * (a / b)]; GLSL integer [/] truncates toward zero, so the result
    carries the dividend's sign and matches OCaml [Int32.rem] / the interpreter
    / PTX [rem.s32] / OpenCL [%]. Routing through a function (not inlining the
    arithmetic) guarantees each operand is evaluated exactly once - critical for
    operands with side effects (value-returning atomics, effectful helper calls)
    that are legal integer expressions and reach the [Mod] node unguarded.

    [int]-only for now: int64 on the Vulkan backend is unwired (no
    [GL_ARB_gpu_shader_int64] extension is emitted anywhere), so an int64 kernel
    already fails to compile independently of [mod]. If int64 Vulkan lands, add
    an [int64_t <name>(int64_t, int64_t)] overload here (GLSL resolves the
    overload by argument type at the call site) and gate the extension on int64
    usage, mirroring the float64 path in [glsl_header].

    The helper name is [!current_smod_name] (see {!compute_smod_name}), not a
    literal, so it cannot collide with a user param or helper identifier. *)
let gen_smod_helper buf (k : kernel) =
  if Sarek_ir_analysis.kernel_uses_int_mod k then
    Buffer.add_string
      buf
      (Printf.sprintf
         "int %s(int a, int b) { return a - b * (a / b); }\n\n"
         !current_smod_name)

(** Choose a collision-safe name for the integer-remainder helper of kernel [k].

    The helper is declared at GLSL top level and called from expressions, so its
    name must avoid every identifier sharing that scope. Two collision sources
    (both observed by CodeRabbit on PR #255):

    - {b param names}: a scalar param becomes a push-constant alias
      [#define <name> pc.<name>], which would macro-expand the helper
      declaration and every call; a vector param becomes the storage-buffer
      array identifier [<name>]. Both use [escape_glsl_name].
    - {b helper-function names}: emitted verbatim as [hf.hf_name] (see
      {!gen_helper_func}); a user helper named [sarek_smod] would duplicate the
      symbol.

    If the default [sarek_smod] is taken, return the first free [sarek_smod_1],
    [sarek_smod_2], ... Local ([SLet]) names are function-scoped, not top-level,
    and are left to the future reserved-prefix policy noted in the impl brief
    (the same class already affects the [sarek_<arr>_length] intrinsic name). *)
let compute_collision_safe_name (k : kernel) ~(base : string) : string =
  let reserved : (string, unit) Hashtbl.t = Hashtbl.create 16 in
  List.iter
    (fun decl ->
      match decl with
      | DParam (v, _) ->
          Hashtbl.replace reserved (escape_glsl_name v.var_name) ()
      | _ -> ())
    k.kern_params ;
  List.iter
    (fun (hf : helper_func) -> Hashtbl.replace reserved hf.hf_name ())
    k.kern_funcs ;
  if not (Hashtbl.mem reserved base) then base
  else
    let rec find i =
      let cand = Printf.sprintf "%s_%d" base i in
      if Hashtbl.mem reserved cand then find (i + 1) else cand
    in
    find 1

let compute_smod_name (k : kernel) : string =
  compute_collision_safe_name k ~base:"sarek_smod"

(** Choose a collision-safe name for the sign-copy helper of kernel [k]. Same
    scope and collision rules as {!compute_smod_name} (see its doc); the two
    helpers use distinct bases ([sarek_smod] / [sarek_copysign]) so they never
    collide with each other, only with user param/helper identifiers. *)
let compute_copysign_name (k : kernel) : string =
  compute_collision_safe_name k ~base:"sarek_copysign"

(** Emit the [sarek_copysign] sign-copy helper when the kernel uses [copysign].

    GLSL has no [copysign] builtin. The exact, branch-free lowering transfers
    the IEEE-754 sign bit of [y] onto the magnitude of [x] via integer bit ops,
    correct for every input including [±0] (where [abs(x)*sign(y)] fails, since
    GLSL [sign(0)=0]) and NaN sign transfer.

    Two overloads are emitted, resolved by argument type at the call site:

    - [float]: always emitted when [copysign] is used. [floatBitsToUint] /
      [uintBitsToFloat] are core since GLSL 3.30, so this needs no extension.
    - [double]: emitted only when the kernel also uses float64, because it uses
      [unpackDouble2x32] / [packDouble2x32] and the [double] type itself, all
      gated behind [GL_ARB_gpu_shader_fp64] — the extension [glsl_header]
      already emits under the same [kernel_uses_float64] condition. A
      [Float64.copysign] kernel is float64 by construction, so its call always
      finds the double overload; the (then-unused) float overload is harmless
      dead code. A [Float32.copysign]-only kernel gets just the float overload.

    The helper name is [!current_copysign_name] (see {!compute_copysign_name}),
    not a literal, so it cannot collide with a user param or helper identifier.
*)
let gen_copysign_helper buf (k : kernel) =
  if Sarek_ir_analysis.kernel_uses_copysign k then begin
    Buffer.add_string
      buf
      (Printf.sprintf
         "float %s(float x, float y) { return \
          uintBitsToFloat((floatBitsToUint(x) & 0x7FFFFFFFu) | \
          (floatBitsToUint(y) & 0x80000000u)); }\n\n"
         !current_copysign_name) ;
    if Sarek_ir_analysis.kernel_uses_float64 k then
      Buffer.add_string
        buf
        (Printf.sprintf
           "double %s(double x, double y) { uvec2 ux = unpackDouble2x32(x); \
            uvec2 uy = unpackDouble2x32(y); ux.y = (ux.y & 0x7FFFFFFFu) | \
            (uy.y & 0x80000000u); return packDouble2x32(ux); }\n\n"
           !current_copysign_name)
  end

(** Generate complete GLSL source for a kernel.
    @param block Optional workgroup dimensions (x, y, z). Defaults to 256x1x1.
*)
let generate ?block ?(log : string -> unit = fun _ -> ()) (k : kernel) : string
    =
  (* Clear per-kernel state *)
  Hashtbl.clear helper_vec_param_indices ;
  current_smod_name := compute_smod_name k ;
  current_copysign_name := compute_copysign_name k ;
  let buf = Buffer.create 1024 in
  Buffer.add_string
    buf
    (glsl_header
       ~kernel_name:k.kern_name
       ?block
       ~uses_float64:(Sarek_ir_analysis.kernel_uses_float64 k)
       ()) ;

  (* Generate buffer bindings *)
  let binding_idx = ref 0 in
  List.iter
    (fun decl ->
      match decl with
      | DParam (v, _) -> (
          match v.var_type with
          | TVec elem_type ->
              gen_buffer_binding buf !binding_idx v elem_type ;
              incr binding_idx
          | _ -> ())
      | _ -> ())
    k.kern_params ;

  (* Generate push constants and collect scalar names for macro collision handling *)
  gen_push_constants buf k.kern_params ;
  let pc_names =
    List.filter_map
      (fun decl ->
        match decl with
        | DParam (v, _) -> (
            match v.var_type with
            | TVec _ -> None (* vectors don't get macros, only their _len *)
            | _ -> Some (escape_glsl_name v.var_name))
        | _ -> None)
      k.kern_params
  in

  (* Generate shared declarations at module scope (GLSL requirement) *)
  let shared_decls = collect_shared_decls k.kern_body in
  gen_shared_decls buf shared_decls ;

  (* Emit the integer-remainder helper (before user helpers, which may call
     it) when the kernel uses [mod]. *)
  gen_smod_helper buf k ;

  (* Emit the sign-copy helper (before user helpers, which may call it) when
     the kernel uses [copysign]. *)
  gen_copysign_helper buf k ;

  (* Generate helper functions *)
  List.iter (gen_helper_func ~pc_names buf) k.kern_funcs ;

  (* Generate main function *)
  Buffer.add_string buf "void main() {\n" ;
  gen_stmt buf "  " k.kern_body ;
  Buffer.add_string buf "}\n" ;

  let shader = Buffer.contents buf in
  log (Printf.sprintf "[GLSL] Generated shader:\n%s" shader) ;
  shader

(** Generate GLSL record type definition - simple struct without tag *)
let gen_record_def buf (name, fields) =
  let mangled = mangle_name name in
  Buffer.add_string buf (Printf.sprintf "struct %s {\n" mangled) ;
  List.iter
    (fun (fname, ftype) ->
      Buffer.add_string buf "  " ;
      Buffer.add_string buf (glsl_type_of_elttype ftype) ;
      Buffer.add_char buf ' ' ;
      Buffer.add_string buf fname ;
      Buffer.add_string buf ";\n")
    fields ;
  Buffer.add_string buf "};\n\n"

(** Generate GLSL variant type definition *)
let gen_variant_def buf v =
  Sarek_ir_codegen.gen_variant_def_glsl
    ~type_of_elttype:glsl_type_of_elttype
    buf
    v

(** Generate GLSL source with custom type definitions.
    @param block Optional workgroup dimensions (x, y, z). Defaults to 256x1x1.
*)
let generate_with_types ?block ?(log : string -> unit = fun _ -> ())
    ~(types : (string * (string * elttype) list) list) (k : kernel) : string =
  (* Clear per-kernel state *)
  Hashtbl.clear helper_vec_param_indices ;
  current_smod_name := compute_smod_name k ;
  current_copysign_name := compute_copysign_name k ;
  (* Use variant types directly from kernel IR *)
  current_variants := k.kern_variants ;

  let buf = Buffer.create 1024 in
  Buffer.add_string
    buf
    (glsl_header
       ~kernel_name:k.kern_name
       ?block
       ~uses_float64:(Sarek_ir_analysis.kernel_uses_float64 k)
       ()) ;

  (* Generate record type definitions (simple structs without tag) *)
  List.iter (gen_record_def buf) types ;

  (* Generate variant type definitions (structs with tag) *)
  List.iter (gen_variant_def buf) k.kern_variants ;

  (* Generate buffer bindings *)
  let binding_idx = ref 0 in
  List.iter
    (fun decl ->
      match decl with
      | DParam (v, _) -> (
          match v.var_type with
          | TVec elem_type ->
              gen_buffer_binding buf !binding_idx v elem_type ;
              incr binding_idx
          | _ -> ())
      | _ -> ())
    k.kern_params ;

  (* Generate push constants and collect scalar names for macro collision handling *)
  gen_push_constants buf k.kern_params ;
  let pc_names =
    List.filter_map
      (fun decl ->
        match decl with
        | DParam (v, _) -> (
            match v.var_type with
            | TVec _ -> None (* vectors don't get macros, only their _len *)
            | _ -> Some (escape_glsl_name v.var_name))
        | _ -> None)
      k.kern_params
  in

  (* Generate shared declarations at module scope (GLSL requirement) *)
  let shared_decls = collect_shared_decls k.kern_body in
  gen_shared_decls buf shared_decls ;

  (* Emit the integer-remainder helper (before user helpers, which may call
     it) when the kernel uses [mod]. *)
  gen_smod_helper buf k ;

  (* Emit the sign-copy helper (before user helpers, which may call it) when
     the kernel uses [copysign]. *)
  gen_copysign_helper buf k ;

  (* Generate helper functions *)
  List.iter (gen_helper_func ~pc_names buf) k.kern_funcs ;

  (* Generate main function *)
  Buffer.add_string buf "void main() {\n" ;
  gen_stmt buf "  " k.kern_body ;
  Buffer.add_string buf "}\n" ;

  let shader = Buffer.contents buf in
  log (Printf.sprintf "[GLSL] Generated shader:\n%s" shader) ;
  shader
