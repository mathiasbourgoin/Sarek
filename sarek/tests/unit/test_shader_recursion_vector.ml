(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Shader-validation gate for recursion + vector-parameter helpers on the GLSL
    and WGSL backends.

    Before this gate existed the pipeline returned [Ok] while emitting an
    invalid shader for any helper taking a vector parameter: the helper was
    emitted as a real GLSL/WGSL function with the vector parameter stripped from
    its signature, so the body referenced an identifier that no longer existed
    (GLSL), and the WGSL call site still forwarded the stripped argument (arity
    mismatch). Nothing assembled the generated source, so the breakage was
    silent.

    This module builds a post-tail-recursion vector-reduction kernel (the shape
    the PPX tail-recursion pass produces: a bounded [while] loop over a vector
    parameter inside a helper) and, when a validator is on PATH:

    - GLSL: assembles with [glslangValidator] (present since the copysign work);
    - WGSL: validates with [naga] if available, otherwise skips cleanly with a
      message (mirrors the ptxas gate in test_ptx_snapshot.ml).

    Substring markers additionally pin that the vector-parameter helper was
    inlined away (no residual function definition, the global buffer is read
    directly). *)

open Sarek_ir_types
open Sarek_codegen

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let base_kernel name params body funcs =
  {
    kern_name = name;
    kern_params = params;
    kern_locals = [];
    kern_body = body;
    kern_types = [];
    kern_variants = [];
    kern_funcs = funcs;
    kern_native_fn = None;
  }

(** A vector-reduction helper in the exact shape the tail-recursion elimination
    pass emits: the recursion is already a bounded [while] loop over the vector
    parameter [arr], accumulating into a mutable local and returning it. The
    helper keeps its [TVec] parameter, which is what breaks the GLSL/WGSL
    backends unless it is inlined at the call site. *)
let sum_range_helper () =
  let arr = make_var "arr" (TVec TFloat32) in
  let n = make_var "n" TInt32 in
  (* Internal loop state uses names distinct from the parameters, exactly like
     the tail-recursion elimination pass (which prefixes loop vars). *)
  let i = {(make_var "__i" TInt32) with var_mutable = true} in
  let acc = {(make_var "_result" TFloat32) with var_mutable = true} in
  let body =
    SLetMut
      ( i,
        EConst (CInt32 0l),
        SLetMut
          ( acc,
            EConst (CFloat32 0.0),
            SSeq
              [
                SWhile
                  ( EBinop (Lt, EVar i, EVar n),
                    SSeq
                      [
                        SAssign
                          ( LVar acc,
                            EBinop (Add, EVar acc, EArrayRead ("arr", EVar i))
                          );
                        SAssign
                          (LVar i, EBinop (Add, EVar i, EConst (CInt32 1l)));
                      ] );
                SReturn (EVar acc);
              ] ) )
  in
  {
    hf_name = "sum_range";
    hf_params = [arr; n];
    hf_ret_type = TFloat32;
    hf_body = body;
  }

(** Kernel: out[tid] = sum_range(data, tid + 1) — a prefix-sum-ish reduction
    that calls the vector-parameter helper once per thread. *)
let recursion_vector_kernel () =
  let data = make_var "data" (TVec TFloat32) in
  let out = make_var "out" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let helper = sum_range_helper () in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("out", EVar tid),
            EApp
              ( EVar (make_var "sum_range" TFloat32),
                [EVar data; EBinop (Add, EVar tid, EConst (CInt32 1l))] ) ) )
  in
  base_kernel
    "recursion_vector"
    [
      DParam (data, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
    ]
    body
    [helper]

(** A vector-parameter helper whose own LOCAL accumulator is named [data] — the
    same name as the kernel's buffer the call passes for the vector parameter.
    Without alpha-renaming, substituting the vector parameter [arr] -> [data]
    makes the spliced body read the local scalar [data] as an array ([data[i]])
    and shadows the global buffer: invalid GLSL. The inliner must rename the
    colliding local to a fresh [sarek_]-prefixed name. *)
let collision_helper () =
  let arr = make_var "arr" (TVec TFloat32) in
  let n = make_var "n" TInt32 in
  let i = {(make_var "__i" TInt32) with var_mutable = true} in
  (* local named EXACTLY like the kernel buffer parameter below *)
  let data_local = {(make_var "data" TFloat32) with var_mutable = true} in
  let body =
    SLetMut
      ( i,
        EConst (CInt32 0l),
        SLetMut
          ( data_local,
            EConst (CFloat32 0.0),
            SSeq
              [
                SWhile
                  ( EBinop (Lt, EVar i, EVar n),
                    SSeq
                      [
                        SAssign
                          ( LVar data_local,
                            EBinop
                              (Add, EVar data_local, EArrayRead ("arr", EVar i))
                          );
                        SAssign
                          (LVar i, EBinop (Add, EVar i, EConst (CInt32 1l)));
                      ] );
                SReturn (EVar data_local);
              ] ) )
  in
  {
    hf_name = "sum_range";
    hf_params = [arr; n];
    hf_ret_type = TFloat32;
    hf_body = body;
  }

(** Same call shape, but the buffer parameter is literally named [data] and the
    helper's local accumulator is also [data]. *)
let collision_kernel () =
  let data = make_var "data" (TVec TFloat32) in
  let out = make_var "out" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("out", EVar tid),
            EApp
              ( EVar (make_var "sum_range" TFloat32),
                [EVar data; EBinop (Add, EVar tid, EConst (CInt32 1l))] ) ) )
  in
  base_kernel
    "collision_vector"
    [
      DParam (data, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
    ]
    body
    [collision_helper ()]

(** transpose_naive regression (#65): a body local that SHADOWS a scalar kernel
    param — the exact shape a [[@sarek.module]] helper produces when its formal
    is named like a kernel scalar. The inliner binds the formal to the argument
    ([let width = width]), so the emitted GLSL declares [int width = ...;].

    GLSL exposes scalar params as preprocessor macros
    ([#define width pc.width]), which rewrite EVERY [width] token in [main] —
    including that declaration's name, yielding [int pc.width = pc.width;]:
    glslangValidator rejects it with "unexpected DOT". HIP/OpenCL/Metal have no
    such macro, so the same kernel assembled fine there (perf sweep 2026-07-24).
    The GLSL emitter must alpha-rename the shadowing locals. *)
let transpose_naive_kernel () =
  let input = make_var "input" (TVec TFloat32) in
  let output = make_var "output" (TVec TFloat32) in
  let width = make_var "width" TInt32 in
  let height = make_var "height" TInt32 in
  (* Locals named EXACTLY like the scalar params — the module-inline shape. *)
  let width_l = make_var "width" TInt32 in
  let height_l = make_var "height" TInt32 in
  let tid = make_var "tid" TInt32 in
  let n = make_var "n" TInt32 in
  let col = make_var "col" TInt32 in
  let row = make_var "row" TInt32 in
  let in_idx = make_var "in_idx" TInt32 in
  let out_idx = make_var "out_idx" TInt32 in
  let inner =
    SLet
      ( col,
        EBinop (Mod, EVar tid, EVar width_l),
        SLet
          ( row,
            EBinop (Div, EVar tid, EVar width_l),
            SLet
              ( in_idx,
                EBinop (Add, EBinop (Mul, EVar row, EVar width_l), EVar col),
                SLet
                  ( out_idx,
                    EBinop (Add, EBinop (Mul, EVar col, EVar height_l), EVar row),
                    SAssign
                      ( LArrayElem ("output", EVar out_idx),
                        EArrayRead ("input", EVar in_idx) ) ) ) ) )
  in
  let body =
    SBlock
      (SLet
         ( width_l,
           EVar width,
           SLet
             ( height_l,
               EVar height,
               SLet
                 ( tid,
                   EIntrinsic ([], "global_thread_id", []),
                   SLet
                     ( n,
                       EBinop (Mul, EVar width_l, EVar height_l),
                       SIf (EBinop (Lt, EVar tid, EVar n), inner, None) ) ) ) ))
  in
  base_kernel
    "transpose_naive"
    [
      DParam (input, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (output, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (width, None);
      DParam (height, None);
    ]
    body
    []

(** Match-pattern-binder shadow (#71, same class as #65): a [match] whose
    variant destructuring binds a name equal to a scalar push-constant param
    ([width]). [gen_match_pattern] emits the binding as a real declaration
    [float width = scrut.OptSome_v;]; the scalar macro [#define width pc.width]
    rewrites it to [float pc.width = pc.width;] → glslangValidator "unexpected
    DOT". The pre-pass must alpha-rename the pattern binder (and its body refs)
    to [sarek_pc_shadow_*], consistently with the SLet path. *)
let match_pc_shadow_kernel () =
  let opt_constrs = [("OptNone", []); ("OptSome", [TFloat32])] in
  let opt_type = TVariant ("Opt", opt_constrs) in
  let data = make_var "data" (TVec opt_type) in
  let out = make_var "out" (TVec TFloat32) in
  (* scalar param named EXACTLY like the pattern binder below *)
  let width = make_var "width" TFloat32 in
  let tid = make_var "tid" TInt32 in
  let scrut = make_var "scrut" opt_type in
  (* the [OptSome] payload binder — same name as the scalar param *)
  let width_bind = make_var "width" TFloat32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( scrut,
            EArrayRead ("data", EVar tid),
            SMatch
              ( EVar scrut,
                [
                  ( PConstr ("OptSome", ["width"]),
                    SAssign (LArrayElem ("out", EVar tid), EVar width_bind) );
                  ( PConstr ("OptNone", []),
                    SAssign (LArrayElem ("out", EVar tid), EConst (CFloat32 0.0))
                  );
                ] ) ) )
  in
  let k =
    base_kernel
      "match_pc_shadow"
      [
        DParam (data, None);
        DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (width, None);
      ]
      body
      []
  in
  {k with kern_variants = [("Opt", opt_constrs)]}

(** Vector-length-macro shadow (#71, Minor): each vector param [v] emits a
    length macro [#define v_len pc.v_len]. A local named [<vec>_len] collides
    with it: [int data_len = ...;] → [int pc.data_len = ...;] → "unexpected
    DOT". [pc_names] excludes vectors, so the pre-pass must additionally treat
    the [_len] macro names as collisions. *)
let vec_len_shadow_kernel () =
  let data = make_var "data" (TVec TFloat32) in
  let out = make_var "out" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  (* local named EXACTLY like the [data] vector's length macro *)
  let data_len_l = make_var "data_len" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( data_len_l,
            EConst (CInt32 0l),
            SAssign
              ( LArrayElem ("out", EVar tid),
                EArrayRead ("data", EVar (make_var "data_len" TInt32)) ) ) )
  in
  base_kernel
    "vec_len_shadow"
    [
      DParam (data, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
    ]
    body
    []

(** EMatch env-drop asymmetry (#71 gap #2, silent-bug once gap #1 lands). An
    outer local [width] shadows the scalar param and is alpha-renamed to
    [sarek_pc_shadow_width_1]. An expression-position [match] then binds [width]
    again in its [OptSome] arm and reads it, while the fallback arm reads the
    OUTER [width].

    [EMatch] emits a ternary and NO binder declaration, so unlike [SMatch] the
    fix's manifestation is a {e substitution} bug, not a syntax error: if the
    [EMatch] arm does not rebind the pattern binder (as [SMatch] does), the
    outer [width -> sarek_pc_shadow_width_1] mapping leaks into the [OptSome]
    arm and its [width] reference is silently substituted with the outer local's
    name — a wrong-variable read. With the fix the pattern binder gets its own
    fresh name ([sarek_pc_shadow_width_2]).

    This is a semantic assertion, not a glslangValidator case: an [EMatch] arm
    that references its payload binder is a separate pre-existing backend
    limitation (no destructuring declaration is emitted for expression-position
    matches), so the shader is not expected to assemble regardless. *)
let ematch_shadow_kernel () =
  let opt_constrs = [("OptNone", []); ("OptSome", [TFloat32])] in
  let opt_type = TVariant ("Opt", opt_constrs) in
  let data = make_var "data" (TVec opt_type) in
  let out = make_var "out" (TVec TFloat32) in
  let width = make_var "width" TFloat32 in
  let tid = make_var "tid" TInt32 in
  let scrut = make_var "scrut" opt_type in
  (* outer local named like the scalar param -> renamed to _1 *)
  let width_outer = make_var "width" TFloat32 in
  (* inner OptSome payload binder, same name -> must get its own _2 *)
  let width_bind = make_var "width" TFloat32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( width_outer,
            EConst (CFloat32 5.0),
            SLet
              ( scrut,
                EArrayRead ("data", EVar tid),
                SAssign
                  ( LArrayElem ("out", EVar tid),
                    EMatch
                      ( EVar scrut,
                        [
                          (PConstr ("OptSome", ["width"]), EVar width_bind);
                          (PConstr ("OptNone", []), EVar width_outer);
                        ] ) ) ) ) )
  in
  let k =
    base_kernel
      "ematch_shadow"
      [
        DParam (data, None);
        DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (width, None);
      ]
      body
      []
  in
  {k with kern_variants = [("Opt", opt_constrs)]}

(** WGSL scalar-param-shadowing MUTATED local (#72). WGSL accesses scalar params
    as [params.<name>] field reads, decided per-[EVar] by the global
    [scalar_param_names] ref, which ignores local scope. A local [let mut width]
    that shadows a scalar param [width] therefore has every body {e read} of
    [width] emitted as [params.width] (the immutable uniform), while the
    declaration and the assignment {e target} use the bare name (writing the
    local). For a mutated local the two diverge: the writes hit a local nobody
    reads, and the reads see the never-updated uniform — a silent wrong result
    (valid WGSL, no error).

    The local's initializer is a constant (NOT [params.width]), so [params.width]
    appears in the emitted shader {e only} through the buggy body reads: with the
    scalar-shadow rename pass it must not appear at all, and the local must carry
    a fresh [sarek_scalar_shadow_*] name. *)
let wgsl_scalar_shadow_mut_kernel () =
  let out = make_var "out" (TVec TFloat32) in
  let width = make_var "width" TInt32 in
  let tid = make_var "tid" TInt32 in
  (* mutable local named EXACTLY like the scalar param *)
  let width_l = {(make_var "width" TInt32) with var_mutable = true} in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLetMut
          ( width_l,
            (* init from a constant, NOT from the param — so any params.width in
               the output can only come from the buggy body reads below *)
            EConst (CInt32 1l),
            SSeq
              [
                (* mutate the local: width := width + 1 *)
                SAssign
                  (LVar width_l, EBinop (Add, EVar width_l, EConst (CInt32 1l)));
                (* read the local — the bug emits params.width here *)
                SAssign
                  ( LArrayElem ("out", EVar tid),
                    ECast (TFloat32, EVar width_l) );
              ] ) )
  in
  base_kernel
    "wgsl_scalar_shadow_mut"
    [
      DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (width, None);
    ]
    body
    []

(* ---- validator plumbing (mirrors the ptxas gate) ---- *)

let tool_available cmd =
  match Unix.system (Printf.sprintf "command -v %s >/dev/null 2>&1" cmd) with
  | Unix.WEXITED 0 -> true
  | _ -> false

let glslang_available = lazy (tool_available "glslangValidator")

let naga_available = lazy (tool_available "naga")

let read_file f =
  try
    let ic = open_in f in
    let n = in_channel_length ic in
    let s = really_input_string ic n in
    close_in ic ;
    s
  with _ -> ""

(** Assemble GLSL compute source with glslangValidator (same invocation as
    Vulkan_api_base: [-V -S comp], entry point [main], no --target-env). Returns
    [Ok ()] on a clean assemble, [Error stderr] otherwise. *)
let glslang_ok glsl =
  let base = Filename.temp_file "sarek_glsl_" "" in
  let src = base ^ ".comp" in
  let spv = base ^ ".spv" in
  let err = base ^ ".err" in
  let oc = open_out src in
  output_string oc glsl ;
  close_out oc ;
  let cmd =
    Printf.sprintf
      "glslangValidator -V -S comp -o %s %s >%s 2>&1"
      (Filename.quote spv)
      (Filename.quote src)
      (Filename.quote err)
  in
  let rc = Unix.system cmd in
  let out = read_file err in
  List.iter (fun f -> try Sys.remove f with _ -> ()) [src; spv; err; base] ;
  match rc with Unix.WEXITED 0 -> Ok () | _ -> Error out

(** Validate WGSL with naga (front-end + validation). naga infers the input
    language from the [.wgsl] extension. *)
let naga_ok wgsl =
  let base = Filename.temp_file "sarek_wgsl_" "" in
  let src = base ^ ".wgsl" in
  let err = base ^ ".err" in
  let oc = open_out src in
  output_string oc wgsl ;
  close_out oc ;
  let cmd =
    Printf.sprintf
      "naga --validate all %s >%s 2>&1"
      (Filename.quote src)
      (Filename.quote err)
  in
  let rc = Unix.system cmd in
  let out = read_file err in
  List.iter (fun f -> try Sys.remove f with _ -> ()) [src; err; base] ;
  match rc with Unix.WEXITED 0 -> Ok () | _ -> Error out

let contains s sub =
  let sl = String.length sub in
  let found = ref false in
  for i = 0 to String.length s - sl do
    if String.sub s i sl = sub then found := true
  done ;
  !found

(* ---- tests ---- *)

let test_glsl_recursion_vector_validates () =
  let k = recursion_vector_kernel () in
  let glsl = Sarek_ir_glsl.generate k in
  (* The vector-parameter helper must be inlined away: no `float sum_range(`
     definition, and the global buffer `data` must be read directly in main. *)
  if contains glsl "sum_range(" then
    Alcotest.failf
      "vector-parameter helper must be inlined, found a residual call/def:\n%s"
      glsl ;
  if not (Lazy.force glslang_available) then
    Printf.printf "  SKIP: glslangValidator not on PATH\n%!"
  else
    match glslang_ok glsl with
    | Ok () -> Printf.printf "  glslangValidator OK: recursion_vector\n%!"
    | Error e ->
        Alcotest.failf
          "glslangValidator rejected recursion_vector GLSL:\n\
           %s\n\
           --- shader ---\n\
           %s"
          e
          glsl

let test_wgsl_recursion_vector_validates () =
  let k = recursion_vector_kernel () in
  let wgsl = Sarek_ir_wgsl.generate k in
  if contains wgsl "sum_range(" then
    Alcotest.failf
      "vector-parameter helper must be inlined, found a residual call/def:\n%s"
      wgsl ;
  if not (Lazy.force naga_available) then
    Printf.printf "  SKIP: naga not on PATH (WGSL validation skipped)\n%!"
  else
    match naga_ok wgsl with
    | Ok () -> Printf.printf "  naga OK: recursion_vector\n%!"
    | Error e ->
        Alcotest.failf
          "naga rejected recursion_vector WGSL:\n%s\n--- shader ---\n%s"
          e
          wgsl

(* Alpha-capture regression: a helper local named like the kernel buffer. The
   fixed inliner renames the local to a fresh sarek_-prefixed name; the buffer
   read survives; glslangValidator accepts. Before the fix this GLSL was invalid
   (a scalar `data` indexed as `data[i]`, shadowing the global buffer). *)
let test_glsl_local_buffer_name_collision () =
  let k = collision_kernel () in
  let glsl = Sarek_ir_glsl.generate k in
  if contains glsl "sum_range(" then
    Alcotest.failf "helper must be inlined, found residual call/def:\n%s" glsl ;
  (* The colliding local must have been renamed. *)
  if not (contains glsl "sarek_inl_local_") then
    Alcotest.failf
      "expected the colliding local to be alpha-renamed (sarek_inl_local_*):\n\
       %s"
      glsl ;
  if not (Lazy.force glslang_available) then
    Printf.printf "  SKIP: glslangValidator not on PATH\n%!"
  else
    match glslang_ok glsl with
    | Ok () -> Printf.printf "  glslangValidator OK: collision_vector\n%!"
    | Error e ->
        Alcotest.failf
          "glslangValidator rejected collision_vector GLSL (alpha-capture?):\n\
           %s\n\
           --- shader ---\n\
           %s"
          e
          glsl

let test_wgsl_local_buffer_name_collision () =
  let k = collision_kernel () in
  let wgsl = Sarek_ir_wgsl.generate k in
  if contains wgsl "sum_range(" then
    Alcotest.failf "helper must be inlined, found residual call/def:\n%s" wgsl ;
  if not (contains wgsl "sarek_inl_local_") then
    Alcotest.failf
      "expected the colliding local to be alpha-renamed (sarek_inl_local_*):\n\
       %s"
      wgsl ;
  if not (Lazy.force naga_available) then
    Printf.printf "  SKIP: naga not on PATH (WGSL validation skipped)\n%!"
  else
    match naga_ok wgsl with
    | Ok () -> Printf.printf "  naga OK: collision_vector\n%!"
    | Error e ->
        Alcotest.failf
          "naga rejected collision_vector WGSL (alpha-capture?):\n\
           %s\n\
           --- shader ---\n\
           %s"
          e
          wgsl

(* transpose_naive (#65): a scalar-param-shadowing local must be alpha-renamed
   so the push-constant macro does not mangle its declaration into `pc.width`.
   Red-on-mutation: revert the rename pass in Sarek_ir_glsl and glslangValidator
   fails here with "unexpected DOT". *)
let test_glsl_transpose_naive_pc_shadow_validates () =
  let k = transpose_naive_kernel () in
  let glsl = Sarek_ir_glsl.generate k in
  if not (contains glsl "sarek_pc_shadow_") then
    Alcotest.failf
      "expected the scalar-param-shadowing locals to be alpha-renamed \
       (sarek_pc_shadow_*):\n\
       %s"
      glsl ;
  (* The colliding raw declaration `int width = ` must NOT survive (it would be
     macro-rewritten to `int pc.width = `). *)
  if contains glsl "int width =" || contains glsl "int height =" then
    Alcotest.failf
      "a scalar-param-shadowing local declaration survived unrenamed:\n%s"
      glsl ;
  if not (Lazy.force glslang_available) then
    Printf.printf "  SKIP: glslangValidator not on PATH\n%!"
  else
    match glslang_ok glsl with
    | Ok () -> Printf.printf "  glslangValidator OK: transpose_naive\n%!"
    | Error e ->
        Alcotest.failf
          "glslangValidator rejected transpose_naive GLSL (unexpected DOT?):\n\
           %s\n\
           --- shader ---\n\
           %s"
          e
          glsl

(* WGSL is NOT affected by the "unexpected DOT" defect: it exposes scalars as
   `params.<name>` field access, not macros, so `let width : i32 = params.width;`
   is valid. This case documents that WGSL still assembles (skips if naga is
   absent, as elsewhere). *)
let test_wgsl_transpose_naive_validates () =
  let k = transpose_naive_kernel () in
  let wgsl = Sarek_ir_wgsl.generate k in
  if not (Lazy.force naga_available) then
    Printf.printf "  SKIP: naga not on PATH (WGSL validation skipped)\n%!"
  else
    match naga_ok wgsl with
    | Ok () -> Printf.printf "  naga OK: transpose_naive\n%!"
    | Error e ->
        Alcotest.failf
          "naga rejected transpose_naive WGSL:\n%s\n--- shader ---\n%s"
          e
          wgsl

(* #72: a MUTATED local shadowing a scalar param must be alpha-renamed so its
   body reads resolve to the local, not the immutable `params.<name>` uniform.
   Codegen-golden (naga absent): asserts the emitted WGSL tokens directly, no
   runtime execution. Were naga present this kernel would also be a semantic
   positive control (writes-then-reads the local). Red-on-mutation: revert the
   rename pass in Sarek_ir_wgsl and the body reads emit `params.width` (the
   silent wrong result) while `sarek_scalar_shadow_` never appears. *)
let test_wgsl_scalar_shadow_mut_local () =
  let wgsl = Sarek_ir_wgsl.generate (wgsl_scalar_shadow_mut_kernel ()) in
  (* The mutated shadowing local must be alpha-renamed. *)
  if not (contains wgsl "sarek_scalar_shadow_width_1") then
    Alcotest.failf
      "expected the mutated scalar-param-shadowing local to be alpha-renamed \
       (sarek_scalar_shadow_width_1):\n\
       %s"
      wgsl ;
  (* The local's init is a constant, so `params.width` can appear ONLY through
     the buggy body reads of the shadowed local. With the fix it must not. *)
  if contains wgsl "params.width" then
    Alcotest.failf
      "a body reference to the mutated shadowing local was emitted as \
       `params.width` (reads the uniform, not the local — silent wrong \
       result):\n\
       %s"
      wgsl ;
  (* The declaration and the mutation must both use the renamed local. *)
  if not (contains wgsl "var sarek_scalar_shadow_width_1 : i32 = 1i") then
    Alcotest.failf
      "expected the mutable local declaration to use the renamed name:\n%s"
      wgsl ;
  if not (Lazy.force naga_available) then
    Printf.printf
      "  codegen-golden OK: wgsl_scalar_shadow_mut (naga absent — WGSL \
       assertion is on emitted text, not runtime)\n\
       %!"
  else
    match naga_ok wgsl with
    | Ok () -> Printf.printf "  naga OK: wgsl_scalar_shadow_mut\n%!"
    | Error e ->
        Alcotest.failf
          "naga rejected wgsl_scalar_shadow_mut WGSL:\n%s\n--- shader ---\n%s"
          e
          wgsl

(* #71 gap #1: a match variant binder named like a scalar param must be
   alpha-renamed (gen_match_pattern emits it as a real declaration). Red-on-
   mutation: revert the pattern-binder rename and glslangValidator fails with
   "unexpected DOT" on `float pc.width = ...`. *)
let test_glsl_match_pattern_pc_shadow_validates () =
  let k =
    Sarek_ir_glsl.generate_with_types ~types:[] (match_pc_shadow_kernel ())
  in
  if not (contains k "sarek_pc_shadow_") then
    Alcotest.failf
      "expected the match pattern binder to be alpha-renamed \
       (sarek_pc_shadow_*):\n\
       %s"
      k ;
  (* The raw destructuring declaration `float width = ` must NOT survive (it
     would be macro-rewritten to `float pc.width = `). *)
  if contains k "float width =" then
    Alcotest.failf
      "a scalar-param-shadowing match binder declaration survived unrenamed:\n\
       %s"
      k ;
  if not (Lazy.force glslang_available) then
    Printf.printf "  SKIP: glslangValidator not on PATH\n%!"
  else
    match glslang_ok k with
    | Ok () -> Printf.printf "  glslangValidator OK: match_pc_shadow\n%!"
    | Error e ->
        Alcotest.failf
          "glslangValidator rejected match_pc_shadow GLSL (unexpected DOT?):\n\
           %s\n\
           --- shader ---\n\
           %s"
          e
          k

(* #71 gap #3: a local named like a vector-length macro (`data_len`) must be
   alpha-renamed. Red-on-mutation: drop the `_len` names from the pre-pass
   collision set and glslangValidator fails with "unexpected DOT" on
   `int pc.data_len = ...`. *)
let test_glsl_vec_len_shadow_validates () =
  let k = Sarek_ir_glsl.generate (vec_len_shadow_kernel ()) in
  if not (contains k "sarek_pc_shadow_") then
    Alcotest.failf
      "expected the vector-_len-shadowing local to be alpha-renamed \
       (sarek_pc_shadow_*):\n\
       %s"
      k ;
  if contains k "int data_len =" then
    Alcotest.failf
      "a vector-_len-shadowing local declaration survived unrenamed:\n%s"
      k ;
  if not (Lazy.force glslang_available) then
    Printf.printf "  SKIP: glslangValidator not on PATH\n%!"
  else
    match glslang_ok k with
    | Ok () -> Printf.printf "  glslangValidator OK: vec_len_shadow\n%!"
    | Error e ->
        Alcotest.failf
          "glslangValidator rejected vec_len_shadow GLSL (unexpected DOT?):\n\
           %s\n\
           --- shader ---\n\
           %s"
          e
          k

(* #71 gap #2 (silent wrong-var read). Semantic assertion, not a validator gate
   (EMatch expression binders emit no declaration — see kernel docstring). The
   OptSome arm binds `width` shadowing the outer renamed local; with the EMatch
   arm made consistent with SMatch, the pattern binder gets its OWN fresh name
   (sarek_pc_shadow_width_2), distinct from the outer local
   (sarek_pc_shadow_width_1). Red-on-mutation: revert the EMatch arm to reuse
   the outer env and both ternary branches collapse to _1 — the OptSome arm
   silently reads the outer local (no _2 is ever minted), failing this check. *)
let test_glsl_ematch_pattern_shadow_rebinds () =
  let k =
    Sarek_ir_glsl.generate_with_types ~types:[] (ematch_shadow_kernel ())
  in
  if not (contains k "sarek_pc_shadow_width_1") then
    Alcotest.failf
      "expected the outer scalar-shadowing local to be renamed \
       (sarek_pc_shadow_width_1):\n\
       %s"
      k ;
  if not (contains k "sarek_pc_shadow_width_2") then
    Alcotest.failf
      "EMatch pattern binder was not rebound: the OptSome arm reuses the outer \
       local (sarek_pc_shadow_width_1) instead of its own binder \
       (sarek_pc_shadow_width_2) — silent wrong-variable read:\n\
       %s"
      k ;
  Printf.printf
    "  semantic OK: ematch_shadow (pattern binder rebound to _2)\n%!"

let () =
  Alcotest.run
    "shader_recursion_vector"
    [
      ( "validation-gate",
        [
          Alcotest.test_case
            "GLSL recursion+vector validates"
            `Quick
            test_glsl_recursion_vector_validates;
          Alcotest.test_case
            "WGSL recursion+vector validates"
            `Quick
            test_wgsl_recursion_vector_validates;
          Alcotest.test_case
            "GLSL helper-local vs buffer-name collision (alpha-capture)"
            `Quick
            test_glsl_local_buffer_name_collision;
          Alcotest.test_case
            "WGSL helper-local vs buffer-name collision (alpha-capture)"
            `Quick
            test_wgsl_local_buffer_name_collision;
          Alcotest.test_case
            "GLSL transpose_naive scalar-param-shadowing local (unexpected DOT)"
            `Quick
            test_glsl_transpose_naive_pc_shadow_validates;
          Alcotest.test_case
            "WGSL transpose_naive scalar-param-shadowing local"
            `Quick
            test_wgsl_transpose_naive_validates;
          Alcotest.test_case
            "WGSL mutated local shadows scalar param (silent wrong result)"
            `Quick
            test_wgsl_scalar_shadow_mut_local;
          Alcotest.test_case
            "GLSL match pattern binder shadows scalar param (unexpected DOT)"
            `Quick
            test_glsl_match_pattern_pc_shadow_validates;
          Alcotest.test_case
            "GLSL local shadows vector _len macro (unexpected DOT)"
            `Quick
            test_glsl_vec_len_shadow_validates;
          Alcotest.test_case
            "GLSL EMatch pattern binder rebinds (silent wrong-var read)"
            `Quick
            test_glsl_ematch_pattern_shadow_rebinds;
        ] );
    ]
