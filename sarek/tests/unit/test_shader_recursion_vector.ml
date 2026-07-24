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
        ] );
    ]
