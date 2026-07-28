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

module Dispatch = Sarek_ir_intrinsic_dispatch

(** Raise a located invalid-argument-count error (atomic-arity helper for the
    shared {!Dispatch.emit_atomic}). *)
let bad_arity n e g =
  Codegen_error.raise_error (Codegen_error.invalid_arg_count n e g)

(** Current kernel's variant definitions (set during generate) *)
let current_variants : (string * (string * elttype list) list) list ref = ref []

(** Current framework name — mirrors the other generators so [set_framework] in
    Sarek_transpile can set it, and the pure registry resolves Float32 math with
    framework="WGSL" (falls through to the generic [sin] spelling). *)
let current_framework : string option ref = ref None

(** {1 Type Mapping} *)

let mangle_name = Sarek_ir_codegen.mangle_name

(** Names that must never be emitted as a WGSL identifier.

    Three groups, kept separate because they are reserved for different reasons
    and a reader needs to know which ones are load-bearing:

    - {b Predeclared names} (types, address spaces, access modes, attribute
      names). WGSL {i does} accept these as user identifiers — they live in a
      scope outside module scope and are shadowable — so escaping them is
      defensive rather than mandatory: a kernel variable named [f32] parses, but
      then [f32] can no longer be spelled as a type in the same scope and the
      emitter would produce a shader that fails much later and much less
      legibly.
    - {b Keywords} (WGSL §2.3 "Keyword Summary"). Hard parse errors.
    - {b Reserved words} (WGSL §2.4 "Reserved Words"). Also hard parse errors,
      and this is where the previous table was materially incomplete: plausible
      OCaml variable names such as [ref], [set], [get], [from], [shared],
      [filter], [target] and [where] are all reserved in WGSL and were being
      emitted verbatim, producing a shader that no WebGPU implementation
      accepts.

    The keyword and reserved-word groups are not a transcription from memory:
    every entry below was verified to be {i actually} rejected by running [naga]
    30.0.0 (the validator [ci/assert-toolchain.sh] pins) over a minimal compute
    shader declaring [var <name> : i32]. The probe also established that the
    predeclared group is {i not} rejected, which is why it is labelled defensive
    above rather than merged into the other two. Re-run the probe when bumping
    naga: a word moving between these groups changes nothing (both are escaped),
    but a word being {i added} to WGSL must be added here. *)
let wgsl_reserved_keywords =
  [
    (* -- Predeclared names: escaped defensively, not rejected by WGSL. ----- *)
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
    (* Entry point name — avoid shadowing *)
    "main";
    (* Params struct name we emit *)
    "Params";
    "params";
    (* Internal builtin parameter names emitted by wgsl_header; must not
       be shadowed by user variable declarations in the kernel body.
       Redundant since [escape_wgsl_name] reserves the whole ["sarek_"]
       prefix, kept so the intent survives a change to that rule. *)
    "sarek_gid";
    "sarek_lid";
    "sarek_wid";
    "sarek_nwg";
    (* -- Keywords (WGSL §2.3). Hard parse errors. ------------------------- *)
    "alias";
    "break";
    "case";
    "const";
    "const_assert";
    "continue";
    "continuing";
    "default";
    "diagnostic";
    "discard";
    "else";
    "enable";
    "false";
    "fn";
    "for";
    "if";
    "let";
    "loop";
    "override";
    "requires";
    "return";
    "struct";
    "switch";
    "true";
    "var";
    "while";
    (* -- Reserved words (WGSL §2.4). Hard parse errors. ------------------- *)
    "NULL";
    "Self";
    "abstract";
    "active";
    "alignas";
    "alignof";
    "as";
    "asm";
    "asm_fragment";
    "async";
    "attribute";
    "auto";
    "await";
    "become";
    "cast";
    "catch";
    "class";
    "co_await";
    "co_return";
    "co_yield";
    "coherent";
    "column_major";
    "common";
    "compile";
    "compile_fragment";
    "concept";
    "consteval";
    "constexpr";
    "constinit";
    "crate";
    "do";
    "dynamic_cast";
    "enum";
    "explicit";
    "export";
    "extends";
    "extern";
    "external";
    "fallthrough";
    "filter";
    "final";
    "finally";
    "friend";
    "from";
    "fxgroup";
    "get";
    "goto";
    "groupshared";
    "highp";
    "impl";
    "implements";
    "import";
    "inline";
    "instanceof";
    "interface";
    "layout";
    "lowp";
    "macro";
    "macro_rules";
    "match";
    "mediump";
    "meta";
    "mod";
    "module";
    "move";
    "mut";
    "mutable";
    "namespace";
    "new";
    "nil";
    "noexcept";
    "noinline";
    "nointerpolation";
    "non_coherent";
    "noncoherent";
    "noperspective";
    "null";
    "nullptr";
    "of";
    "operator";
    "package";
    "packoffset";
    "partition";
    "pass";
    "patch";
    "pixelfragment";
    "precise";
    "precision";
    "premerge";
    "priv";
    "protected";
    "pub";
    "public";
    "readonly";
    "ref";
    "regardless";
    "register";
    "reinterpret_cast";
    "require";
    "resource";
    "restrict";
    "self";
    "set";
    "shared";
    "sizeof";
    "smooth";
    "snorm";
    "static";
    "static_assert";
    "static_cast";
    "std";
    "subroutine";
    "super";
    "target";
    "template";
    "this";
    "thread_local";
    "throw";
    "trait";
    "try";
    "type";
    "typedef";
    "typeid";
    "typename";
    "typeof";
    "union";
    "unless";
    "unorm";
    "unsafe";
    "unsized";
    "use";
    "using";
    "varying";
    "virtual";
    "volatile";
    "wgsl";
    "where";
    "with";
    "writeonly";
    "yield";
  ]

(** Escape identifiers that WGSL forbids.

    Four things make a name unusable as a WGSL identifier. All four are handled
    by the same rewrite — prefixing with ["sarek_"] — for the injectivity reason
    set out below.

    - {b Reserved prefix} (WGSL §4.4.1): an identifier must not start with a
      double underscore. The frontend's tail-recursion elimination
      ([Sarek_tailrec_elim]) renames every eliminated loop parameter to
      ["__" ^ name], and the native backend emits ["__v%d"]/["__m%d"]
      temporaries, so any kernel whose recursion is turned into a loop reached
      this emitter with a [__]-prefixed variable. C-family targets (CUDA,
      OpenCL, GLSL, Metal) accept those names, WGSL rejects them outright at
      parse time ("Identifier starts with a reserved prefix"), which made the
      emitted shader unusable on every WebGPU implementation. Prefixing keeps
      the name recognisable and moves the double underscore off the front, where
      it is legal.
    - {b Bare underscore} (same clause): a lone ["_"] is not an identifier in
      WGSL either — it is the phony-assignment target. An OCaml wildcard or
      generated placeholder reaching the emitter as ["_"] would produce
      [let _ : i32 = ...], which naga rejects.
    - {b Reserved names}: anything in {!wgsl_reserved_keywords} — WGSL keywords
      and reserved words, plus the predeclared and internal names this emitter
      escapes defensively.
    - {b The escaped namespace itself}: a source name that already starts with
      ["sarek_"] is escaped too, so it cannot be confused with the image of one
      that was. This is what makes the rewrite injective; see below.

    {2 Injectivity}

    An earlier form of this function rewrote each problem separately — ["_"] to
    ["sarek_"], a ["__"] prefix to ["sarek" ^ name], a keyword to [name ^ "v"] —
    and was {i not} injective: ["__i"] and ["sarek__i"] both emitted
    ["sarek__i"], ["_"] and ["sarek_"] both emitted ["sarek_"], and ["if"] and
    ["ifv"] both emitted ["ifv"]. Two source variables colliding on one WGSL
    name is not a cosmetic problem: the second [var] declaration shadows the
    first in the same block, every later read silently resolves to the wrong
    binding, and the shader still compiles. A wrong answer with no diagnostic is
    the worst failure mode available here.

    The rule below is a single unconditional one: the whole ["sarek_"] prefix is
    reserved, and any name that is reserved, or that could be confused with an
    escaped name, is prefixed with it.

    That this is injective on source identifiers is checkable by cases rather
    than by inspection:
    - the escaping branch only ever emits names starting with ["sarek_"];
    - the identity branch only ever emits names {i not} starting with
      ["sarek_"], because such a name would have taken the escaping branch;
    - the two images are therefore disjoint, and each branch is injective on its
      own ([name ↦ "sarek_" ^ name] and the identity).

    {2 Generator-produced names}

    {!wgsl_generated_prefixes} is exempt, and the exemption is structural rather
    than a convenience. [rename_scalar_shadowing_locals] mints
    [sarek_scalar_shadow_*] names and puts them into the IR as ordinary
    variables, so they reach this function again on the way out. A name in the
    escaped namespace cannot be a fixed point of the rule above — that is what
    reserving the prefix means — so re-escaping produced
    [sarek_sarek_scalar_shadow_width_1], and the alternative of choosing an
    internal name that {i is} a fixed point is self-defeating: every fixed point
    is, by definition, reachable from the identical source identifier. Internal
    names must therefore be minted in final form and left alone.

    The cost is one contrived residual: a user identifier spelled exactly like a
    generator-internal name (a local literally named
    [sarek_scalar_shadow_width_1]) is no longer pushed out of the namespace and
    can collide with the generated one. That is strictly smaller than what this
    function replaced, which collided on [__i]/[sarek__i], [_]/[sarek_] and
    [if]/[ifv] — three families of ordinary, plausible names.

    The output is also always a legal WGSL identifier: it never starts with
    ["__"] (an escaped name starts with ["sarek"], and an unescaped one cannot
    start with ["__"] or it would have been escaped), it is never a bare ["_"],
    and it is never reserved (no WGSL keyword or reserved word starts with
    ["sarek_"]).

    {2 Residual}

    [abi] builds the uniform-struct length fields as
    ["sarek_" ^ escape_wgsl_name v ^ "_length"], a second construction in the
    same namespace that this function cannot police. A vector named ["if"] and a
    scalar named ["sarek_if_length"] in one kernel still collide there. Closing
    it needs a length-prefixed encoding for that namespace, which changes the
    emitted ABI field names, so it is deliberately left for its own change
    rather than folded into this one. *)
let wgsl_escape_prefix = "sarek_"

(** Prefixes of names this generator mints itself, in already-final form. They
    round-trip unchanged through {!escape_wgsl_name}; see its
    "Generator-produced names" section for why the exemption is unavoidable
    rather than a shortcut. Anything added here must be a name a generator
    constructs, never a name that can arrive from source. *)
let wgsl_generated_prefixes = ["sarek_scalar_shadow_"]

let escape_wgsl_name name =
  if
    List.exists
      (fun p -> String.starts_with ~prefix:p name)
      wgsl_generated_prefixes
  then name
  else if
    name = "_"
    || String.starts_with ~prefix:"__" name
    || String.starts_with ~prefix:wgsl_escape_prefix name
    || List.mem name wgsl_reserved_keywords
  then wgsl_escape_prefix ^ name
  else name

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
  | TFloat16 ->
      (* Deferred to #57 slice 2: WGSL has a native `f16`, but it requires
         `enable f16;` at MODULE TOP — before the bindings the header emits —
         which is a structural change to the emitter, not a match arm. *)
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "f16"
           "WGSL: float16 not yet supported (#57 slice 2 — needs a module-top \
            `enable f16;` directive)")
  | TUint8 ->
      (* Core WebGPU has no 8-bit scalar at all, but even if it did this arm
         would refuse: [TUint8] is not a general integer type here, it is the
         element type of a cooperative-matrix operand buffer, and WGSL has no
         cooperative-matrix construct to consume one. *)
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "uint8"
           "WGSL: uint8 is a cooperative-matrix operand element type, emitted \
            only by the Vulkan backend, and WGSL has no cooperative-matrix \
            path")
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
  | TInt32 | TInt64 | TFloat16 | TFloat32 | TBool | TUnit | TUint8 -> false

(** {1 Thread Intrinsics}

    WGSL uses three distinct builtins:
    - [local_invocation_id] (sarek_lid) — thread within workgroup
    - [workgroup_id] (sarek_wid) — workgroup index in the dispatch grid
    - [global_invocation_id] (sarek_gid) — globally unique thread index

    All are [vec3<u32>]; we cast to i32 to match the IR's i32 type for thread
    ids. The entry point declares all three builtins; unused ones are harmless
    (WGSL permits unused builtin params). *)
let wgsl_thread_intrinsic = function
  | "thread_id_x" | "thread_idx_x" -> "i32(sarek_lid.x)"
  | "thread_id_y" | "thread_idx_y" -> "i32(sarek_lid.y)"
  | "thread_id_z" | "thread_idx_z" -> "i32(sarek_lid.z)"
  | "block_id_x" | "block_idx_x" -> "i32(sarek_wid.x)"
  | "block_id_y" | "block_idx_y" -> "i32(sarek_wid.y)"
  | "block_id_z" | "block_idx_z" -> "i32(sarek_wid.z)"
  | "block_dim_x" -> "256i"
  | "block_dim_y" -> "1i"
  | "block_dim_z" -> "1i"
  | "grid_dim_x" -> "i32(sarek_nwg.x)"
  | "grid_dim_y" -> "i32(sarek_nwg.y)"
  | "grid_dim_z" -> "i32(sarek_nwg.z)"
  | "global_thread_id" | "global_idx" | "global_idx_x" -> "i32(sarek_gid.x)"
  | "global_idx_y" -> "i32(sarek_gid.y)"
  | "global_idx_z" -> "i32(sarek_gid.z)"
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
  | EIntrinsic (path, name, args) ->
      Dispatch.gen_intrinsic wgsl_backend buf path name args
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
  | EMatch (scrut, cases) when Sarek_ir_codegen.ematch_binds_payload cases ->
      (* #75: a match EXPRESSION lowers to nested [select()] calls, which has nowhere to
         declare a payload binder — bind it by substituting the same payload
         read the [SMatch] arm declares (WGSL payloads are indexed sibling fields), then emit the
         (now binder-free) match. One shared, capture-avoiding pass for every
         backend; see {!Sarek_ir_codegen.subst_ematch_payloads}. *)
      gen_expr
        buf
        (EMatch
           ( scrut,
             Sarek_ir_codegen.subst_ematch_payloads
               ~layout:Sarek_ir_codegen.wgsl_payload_layout
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

and wgsl_backend =
  {
    Dispatch.framework =
      (fun () -> Option.value ~default:"WGSL" !current_framework);
    gen_expr;
    thread_intrinsic = wgsl_thread_intrinsic;
    pre_hook =
      (fun buf ~full_name:_ _path name args ->
        if name = "fmod" then
          match args with
          | [x; y] ->
              Buffer.add_string buf "sarek_fmod(" ;
              gen_expr buf x ;
              Buffer.add_string buf ", " ;
              gen_expr buf y ;
              Buffer.add_char buf ')' ;
              true
          | _ ->
              Codegen_error.raise_error
                (Codegen_error.invalid_arg_count "fmod" 2 (List.length args))
        else false);
    post_hook = (fun _ _ _ _ -> false);
    invalid_arg_count = bad_arity;
    on_unknown =
      (fun full ->
        Codegen_error.raise_error (Codegen_error.unknown_intrinsic full));
    arm =
      (fun name ->
        match name with
        | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh"
        | "tanh" | "exp" | "exp2" | "log" | "log2" | "sqrt" | "floor" | "ceil"
        | "round" | "trunc" | "abs" | "atan2" | "pow" | "min" | "max" | "fma" ->
            Some (fun buf args -> Dispatch.emit_call ~gen_expr buf name args)
        | "fabs" ->
            Some (fun buf args -> Dispatch.emit_call ~gen_expr buf "abs" args)
        | "rsqrt" ->
            Some
              (fun buf args ->
                (* Was: `| _ -> emit_args`, which on the wrong argument count
                   emitted `(1.0f / sqrt(a, b))` and returned Ok. *)
                Dispatch.emit_unary
                  ~gen_expr
                  ~invalid_arg_count:bad_arity
                  buf
                  ~prefix:"(1.0f / sqrt("
                  ~suffix:"))"
                  ~opname:"rsqrt"
                  args)
        | "block_barrier" ->
            Some (fun buf _ -> Buffer.add_string buf "workgroupBarrier()")
        | "atomic_add" | "atomic_add_int32" | "atomic_add_global_int32" ->
            Some
              (fun buf args ->
                Dispatch.emit_atomic
                  ~gen_expr
                  ~invalid_arg_count:bad_arity
                  buf
                  ~callee:"atomicAdd"
                  ~prefix:""
                  ~suffix:")"
                  ~opname:"atomic_add"
                  ~expected:2
                  ~allow_array:true
                  args)
        | "atomic_min" ->
            Some
              (fun buf args ->
                Dispatch.emit_atomic
                  ~gen_expr
                  ~invalid_arg_count:bad_arity
                  buf
                  ~callee:"atomicMin"
                  ~prefix:""
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
                  ~prefix:""
                  ~suffix:")"
                  ~opname:"atomic_max"
                  ~expected:2
                  ~allow_array:false
                  args)
        | "float" ->
            Some
              (fun buf args ->
                Dispatch.emit_unary
                  ~gen_expr
                  ~invalid_arg_count:bad_arity
                  buf
                  ~prefix:"f32("
                  ~suffix:")"
                  ~opname:"float"
                  args)
        | "int_of_float" ->
            Some
              (fun buf args ->
                Dispatch.emit_unary
                  ~gen_expr
                  ~invalid_arg_count:bad_arity
                  buf
                  ~prefix:"i32("
                  ~suffix:")"
                  ~opname:"int_of_float"
                  args)
        | _ -> None);
  }

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
      Buffer.add_string
        buf
        (Sarek_ir_codegen.payload_suffix
           Sarek_ir_codegen.wgsl_payload_layout
           ~cname
           ~arity:1
           0) ;
      Buffer.add_string buf ";\n"
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
          Buffer.add_string
            buf
            (Sarek_ir_codegen.payload_suffix
               Sarek_ir_codegen.wgsl_payload_layout
               ~cname
               ~arity:(List.length vars)
               i) ;
          Buffer.add_string buf ";\n")
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
      (* WGSL requires EXACTLY ONE default clause in every switch (WGSL spec
         §9.4.3, "must have exactly one default selector"); naga reports
         "missing default case" and rejects the whole entry point otherwise.
         C's switch needs no default and GLSL's likewise, so this is a WGSL-only
         obligation and the four other backends have nothing to do here.

         A Sarek match that is exhaustive over the constructors carries no
         [PWild] arm, so without this the emitter produced an invalid module for
         every such match — i.e. for the ordinary case. It was invisible because
         no WGSL [SMatch] anywhere in the tree was fed to naga (#132); the
         corpus case [smatch_multi_payload] in the wgsl_validation_sweep is what
         made it visible, and is what keeps it visible.

         An empty body is the right lowering: the arm is reachable only for a
         tag outside the declared constructor set, which the type system already
         excludes, and WGSL has no trap to put here. *)
      if not (List.exists (fun (p, _) -> p = PWild) cases) then begin
        Buffer.add_string buf indent ;
        Buffer.add_string buf "  default: {\n" ;
        Buffer.add_string buf indent ;
        Buffer.add_string buf "  }\n"
      end ;
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
  | SCoopmat _ ->
      (* Core WebGPU has no cooperative-matrix construct — the subgroup-matrix
         proposal is not in the shipped specification — so unlike the [SNative]
         arm above there is no comment-and-continue that would still produce a
         module computing what the kernel asked for. *)
      Codegen_error.raise_error
        (Codegen_error.unsupported_construct
           "cooperative matrix"
           "WGSL: the WGSL backend has no cooperative-matrix path; \
            cooperative-matrix statements are emitted only by the Vulkan \
            backend")

(** {1 Helper Function Generation} *)

let gen_helper_func buf (hf : helper_func) =
  let non_vec_params =
    List.filter
      (fun (v : var) -> match v.var_type with TVec _ -> false | _ -> true)
      hf.hf_params
  in
  Buffer.add_string buf "fn " ;
  (* Escaped, like every other identifier. A call to this helper is an
     [EApp (EVar _, _)], and [gen_expr]'s [EVar] case escapes — so emitting the
     definition raw does not merely risk an illegal name, it guarantees the
     definition and its call sites disagree. A helper named [__f] was defined as
     `fn __f(` (illegal: reserved double-underscore prefix) and called as
     `sarek___f(` (undefined function): two errors from one omission. *)
  Buffer.add_string buf (escape_wgsl_name hf.hf_name) ;
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

(** Emit the [sarek_fmod] C-fmod helper (f32; WGSL has no f64) when the kernel
    uses [fmod]. Replaces the earlier bare [%] lowering, which shared C-fmod's
    two divergences (both raised in review): the single-pass [x - y*trunc(x/y)]
    loses quotient precision for large [|x/y|], and an infinite divisor yields
    NaN where C defines [fmod(x, ±inf) = x].

    The body is a bounded exact reduction by power-of-two scaling (identical in
    shape to the GLSL [sarek_fmod] helper): scale [d = |y|] up by [×2] to the
    largest [|y|·2^k ≤ |x|], then walk back down subtracting whenever [r ≥ d].
    Every [×2]/[×0.5] is exact and each subtraction runs with [d ≤ r < 2d]
    (exact by Sterbenz), so [r] is the bit-exact remainder magnitude; the loops
    are bounded by the f32 exponent span (~277 iterations). The dividend's sign
    is restored by a bit-level copy.

    WGSL has no [isnan]/[isinf]; infinity is detected by a magnitude test
    against the largest finite f32 ([0x1.fffffep+127]). [|y| = inf] returns [x]
    (C-conformant). The genuine NaN-domain cases ([y = 0], [|x| = inf]) are NOT
    expressible in WGSL's float model — WGSL cannot synthesise a NaN — so they
    return [x] purely to keep the reduction loop terminating; this is the one
    residual divergence from C, unavoidable in WGSL and documented as such.

    The helper name is a fixed [sarek_fmod], emitted verbatim at both its
    definition here and its call site in [gen_expr]. A user helper of the same
    name no longer clashes with it: [escape_wgsl_name] reserves the whole
    ["sarek_"] prefix, so a user [sarek_fmod] is emitted as [sarek_sarek_fmod]
    at its definition and at every call. *)
let gen_fmod_helper buf (k : kernel) =
  if Sarek_ir_analysis.kernel_uses_intrinsic "fmod" k then
    Buffer.add_string
      buf
      "fn sarek_fmod(x: f32, y: f32) -> f32 {\n\
      \  let ax = abs(x); let ay = abs(y);\n\
      \  if (ay > 0x1.fffffep+127f) { return x; }\n\
      \  if (ax > 0x1.fffffep+127f || ay == 0.0) { return x; }\n\
      \  if (ax < ay) { return x; }\n\
      \  var r = ax; var d = ay;\n\
      \  loop { if (!(d <= 0.5 * r)) { break; } d = d * 2.0; }\n\
      \  loop { if (r >= d) { r = r - d; } if (d == ay) { break; } d = d * \
       0.5; }\n\
      \  return bitcast<f32>((bitcast<u32>(r) & 0x7fffffffu) | \
       (bitcast<u32>(x) & 0x80000000u));\n\
       }\n\n"

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
  | SCoopmat _ ->
      (* No workgroup storage to hoist: a fragment is a subgroup-cooperative
         value, not an array, and the buffers a load names are parameters. The
         statement itself is refused later by [gen_stmt]; returning [] here
         keeps that refusal the one the user sees. *)
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

(** {1 Scalar-param shadow renaming} *)

(** Alpha-rename kernel-body binders whose name collides with a scalar kernel
    param.

    Scalar params are accessed in the body as [params.<name>]; {!gen_expr}
    decides this per-[EVar] by checking the {e global} [scalar_param_names] ref,
    which ignores local scope. A local [let width = …] (or [let mut width = …])
    that shadows a scalar param [width] therefore has every body reference to
    [width] wrongly emitted as [params.width] — reading the uniform instead of
    the local. For an immutable self-binding local ([let width = params.width])
    this is accidentally correct; for a {e mutated} shadowing local it is a
    silent wrong result (valid WGSL, no error): the declaration uses the bare
    name ([var width : i32 = params.width;]) so writes hit the local, but every
    read is redirected to the immutable uniform.

    This mirrors the GLSL backend's {!Sarek_ir_glsl.rename_pc_shadowing_locals};
    both delegate the shared traversal to
    {!Sarek_ir_codegen.rename_shadowing_locals}. Each colliding binder (and its
    in-scope references) is rewritten to a fresh [sarek_scalar_shadow_*] name
    that is not a scalar param, so [gen_expr]'s [scalar_param_names] check never
    matches it. The initializer is evaluated in the outer scope, so it still
    expands to [params.<name>], preserving semantics. Unlike GLSL there is no
    vector-length collision: both spell the length [sarek_<arr>_length]
    ({!EArrayLen}), but WGSL emits it as the field access
    [params.sarek_<arr>_length], hardcoded with a [params.] prefix independent
    of any local, so a local cannot alias it — the collision set is scalar
    params only. WGSL-only. *)
let rename_scalar_shadowing_locals ~scalar_names body =
  Sarek_ir_codegen.rename_shadowing_locals
    ~collides:(fun name ->
      (* A local collides if its name matches a scalar param — the exact check
         [gen_expr] performs on [EVar] against [scalar_param_names] (raw
         names). *)
      List.mem name scalar_names)
    ~fresh_name:(fun orig n ->
      Printf.sprintf "sarek_scalar_shadow_%s_%d" (escape_wgsl_name orig) n)
    body

(** {1 Main generate functions} *)

let wgsl_header ~kernel_name ?(block = (256, 1, 1)) () =
  let bx, by, bz = block in
  Printf.sprintf
    "@compute @workgroup_size(%d, %d, %d)\n\
     fn main(\n\
    \  @builtin(global_invocation_id) sarek_gid : vec3<u32>,\n\
    \  @builtin(local_invocation_id) sarek_lid : vec3<u32>,\n\
    \  @builtin(workgroup_id) sarek_wid : vec3<u32>,\n\
    \  @builtin(num_workgroups) sarek_nwg : vec3<u32>\n\
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
    ~backend:"WGSL"
    ~hint:"needs a module-top `enable f16;` directive"
    Sarek_ir_analysis.Float16

(* Whole-kernel cooperative-matrix gate, at the same choke point and for the
   reason given at {!Sarek_ir_codegen.reject_feature}: the per-node arms see
   only what the emitter reaches, and a [TUint8] parameter that is never loaded
   from carries the feature without reaching any of them. Not routed through
   [reject_feature] because its composed sentence hardcodes "#57 slice 2", which
   is the f16 history and not this one. *)
let reject_coopmat_kernel (k : kernel) : unit =
  if Sarek_ir_analysis.kernel_uses Sarek_ir_analysis.Coopmat k then
    Codegen_error.raise_error
      (Codegen_error.unsupported_construct
         "cooperative matrix"
         "WGSL: the WGSL backend has no cooperative-matrix path; cooperative \
          matrices and their uint8 operand buffers are emitted only by the \
          Vulkan backend (backlog-62)")

(** Generate complete WGSL source for a kernel. *)
let generate ?block ?(log : string -> unit = fun _ -> ()) (k : kernel) : string
    =
  reject_float16_kernel k ;
  reject_coopmat_kernel k ;
  if params_have_float64 k.kern_params then
    Codegen_error.raise_error
      (Codegen_error.unsupported_construct
         "f64 parameter"
         "WGSL: f64 unsupported — WebGPU has no float64 type") ;
  (* Inline vector-parameter helpers (buffers cannot be passed as WGSL function
     arguments — see Sarek_ir_inline_vec). *)
  let k = Sarek_ir_inline_vec.inline_vec_helpers ~backend:"WGSL" k in
  scalar_param_names := [] ;
  current_variants := k.kern_variants ;
  let buf = Buffer.create 1024 in
  let scalars = gen_bindings buf k.kern_params in
  scalar_param_names := scalars ;
  let wg_decls = collect_workgroup_decls k.kern_body in
  gen_workgroup_module_decls buf wg_decls ;
  gen_fmod_helper buf k ;
  List.iter (gen_helper_func buf) k.kern_funcs ;
  Buffer.add_string buf (wgsl_header ~kernel_name:k.kern_name ?block ()) ;
  (* Alpha-rename any body local that shadows a scalar param first (see
     rename_scalar_shadowing_locals). *)
  gen_stmt
    buf
    "  "
    (rename_scalar_shadowing_locals ~scalar_names:scalars k.kern_body) ;
  Buffer.add_string buf "}\n" ;
  let shader = Buffer.contents buf in
  log (Printf.sprintf "[WGSL] Generated shader:\n%s" shader) ;
  shader

(** Generate WGSL source with custom type definitions. *)
let generate_with_types ?block ?(log : string -> unit = fun _ -> ())
    ~(types : (string * (string * elttype) list) list) (k : kernel) : string =
  reject_float16_kernel k ;
  reject_coopmat_kernel k ;
  if params_have_float64 k.kern_params then
    Codegen_error.raise_error
      (Codegen_error.unsupported_construct
         "f64 parameter"
         "WGSL: f64 unsupported — WebGPU has no float64 type") ;
  (* Inline vector-parameter helpers (buffers cannot be passed as WGSL function
     arguments — see Sarek_ir_inline_vec). *)
  let k = Sarek_ir_inline_vec.inline_vec_helpers ~backend:"WGSL" k in
  scalar_param_names := [] ;
  current_variants := k.kern_variants ;
  let buf = Buffer.create 1024 in
  List.iter (gen_record_def buf) types ;
  List.iter (gen_variant_def buf) k.kern_variants ;
  let scalars = gen_bindings buf k.kern_params in
  scalar_param_names := scalars ;
  let wg_decls = collect_workgroup_decls k.kern_body in
  gen_workgroup_module_decls buf wg_decls ;
  gen_fmod_helper buf k ;
  List.iter (gen_helper_func buf) k.kern_funcs ;
  Buffer.add_string buf (wgsl_header ~kernel_name:k.kern_name ?block ()) ;
  (* Alpha-rename any body local that shadows a scalar param first (see
     rename_scalar_shadowing_locals). *)
  gen_stmt
    buf
    "  "
    (rename_scalar_shadowing_locals ~scalar_names:scalars k.kern_body) ;
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
