(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * A local inside a HELPER FUNCTION whose name matches a scalar kernel
 * parameter, on GLSL and WGSL.
 *
 * Both backends expose scalar kernel parameters by name substitution — GLSL as
 * `#define <name> pc.<name>`, WGSL as a `params.<name>` struct field — and both
 * ran their shadow-rename pre-pass over `kern_body` ONLY. Helper functions were
 * believed covered: on GLSL by the #undef/#define dance in gen_helper_func, and
 * that dance is real, but it is computed from PARAMETER names alone. A helper
 * whose BODY declares a colliding local was covered by neither.
 *
 * The two symptoms differ, and the WGSL one is why this test exists:
 *
 *   GLSL — `int n = (q + 1);` becomes `int pc.n = (q + 1);` and glslang
 *          rejects the shader outright. Loud.
 *   WGSL — `let n : i32 = (q + 1i);` is emitted and never read, while
 *          `return (n * 2i)` becomes `return (params.n * 2i)`. Valid WGSL,
 *          compiles, runs, returns the WRONG NUMBER. Silent.
 *
 * This asserts on the emitted source rather than on a device result, because
 * the point is what the generator produces: a device check would only catch the
 * GLSL half here (the WGSL half needs a WGSL device, and there is none in this
 * suite), and would catch it as "shader failed to compile" rather than as this
 * specific defect.
 *
 * NOT covered here, and the reason is the interesting part: a helper formal
 * colliding with a VECTOR-LENGTH macro (`sarek_<vec>_length`) cannot be written.
 * The PPX rejects any identifier beginning with `sarek_`
 * (sarek/ppx/Sarek_error.ml) — the reserved namespace IS enforced, at the
 * frontend rather than in the backend escapers. `gen_helper_func` guards that
 * family anyway, so the guard does not depend on that check holding, but there
 * is no DSL-expressible test for it and a test asserting otherwise would be
 * fiction.
 *
 * Run with: dune exec sarek/tests/codegen_golden/test_helper_local_shadowing.exe
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std

(* `bump`'s body binds `n`; the kernel's scalar parameter is also `n`. `bump`
   itself has no parameter named `n`, which is exactly what put this case
   outside the #undef guard's reach. *)
let shadowing_kernel =
  [%kernel
    let open Std in
    let bump (q : int32) : int32 =
      let n = q + 1l in
      n * 2l
    in
    fun (out : int32 vector) (n : int32) ->
      let t = global_thread_id in
      if t < 1l then out.(0l) <- bump n]

(* CodeRabbit's third case: the helper's FORMAL is named like the scalar param.
   `twice`'s parameter is `n`; so is the kernel's scalar parameter. *)
let formal_kernel =
  [%kernel
    let open Std in
    let twice (n : int32) : int32 = n * 3l in
    fun (out : int32 vector) (n : int32) ->
      let t = global_thread_id in
      if t < 1l then out.(0l) <- twice n]

let contains hay needle =
  let nh = String.length hay and nn = String.length needle in
  let rec go i = i + nn <= nh && (String.sub hay i nn = needle || go (i + 1)) in
  nn > 0 && go 0

let ir =
  let _, kirc = shadowing_kernel in
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "shadowing kernel has no IR"

let formal_ir =
  let _, kirc = formal_kernel in
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "formal kernel has no IR"

let failures = ref 0

let expect label cond =
  Printf.printf "  %-58s %s\n%!" label (if cond then "OK" else "FAIL") ;
  if not cond then incr failures

(* The real gate. The macro expansion that breaks the shader happens in the GLSL
   COMPILER, not in the text we emit — our output literally contains
   [int n = (q + 1);]. An earlier version of this test asserted the absence of
   [int pc.n =] in the emitted source, which can never appear and so could never
   fail; prove-red caught it. Compiling is what actually observes the defect. *)
let glslang_ok src =
  let f = Filename.temp_file "sarek_shadow" ".comp" in
  let oc = open_out f in
  output_string oc src ;
  close_out oc ;
  let rc =
    Sys.command
      (Printf.sprintf
         "glslangValidator -V -S comp -o /dev/null %s > /dev/null 2>&1"
         (Filename.quote f))
  in
  Sys.remove f ;
  rc = 0

let glslang_available =
  Sys.command "command -v glslangValidator > /dev/null 2>&1" = 0

(* Textual companion, usable with no toolchain: the shader must not DECLARE an
   identifier that a [#define] is rewriting. Both halves are required — the
   macro must be present (otherwise the check is trivially satisfied by a
   shader that never exposes the parameter at all) and the declaration must be
   absent. *)
let declares_macro_shadowed_local glsl =
  contains glsl "#define n pc.n" && contains glsl "  int n = "

let () =
  print_endline "=== helper-body local shadowing a scalar kernel param ===" ;
  let glsl = Sarek_codegen.Sarek_ir_glsl.generate ir in
  let wgsl = Sarek_codegen.Sarek_ir_wgsl.generate ir in
  expect
    "GLSL: the scalar macro is emitted (so the check is not trivial)"
    (contains glsl "#define n pc.n") ;
  expect
    "GLSL: no helper local is declared under a live macro name"
    (not (declares_macro_shadowed_local glsl)) ;
  if glslang_available then
    expect "GLSL: glslangValidator accepts the shader" (glslang_ok glsl)
  else
    print_endline
      "  GLSL: glslangValidator absent — compile gate SKIPPED (not a pass)" ;
  expect
    "WGSL: params.n is not substituted for the helper's local"
    (not (contains wgsl "return (params.n * 2i)")) ;
  expect
    "WGSL: the scalar field is still reachable (guard not disabled)"
    (contains wgsl "params.n") ;
  (* The USE must resolve to the renamed local, not merely the declaration.
     An earlier version asserted only [contains "* 2)"], which matches both the
     correct output AND a half-rename where the declaration moved but the use
     stayed [n]: GLSL would preprocess that use to [pc.n], and
     [return (pc.n * 2);] COMPILES — so neither the textual check nor the
     glslangValidator gate would have caught it. Only the declaration side is
     syntactically invalid. CodeRabbit caught this; it is the same
     passes-for-a-narrower-reason shape as the rest of this branch.

     Asserting the declaration and the use carry the SAME renamed identifier is
     what actually pins the behaviour, so the renamed name is read out of the
     emitted source rather than hardcoded. *)
  let renamed_use ~decl_prefix ~use_suffix src =
    match String.index_opt src 's' with
    | None -> false
    | Some _ -> (
        let rec find i =
          if i + String.length decl_prefix > String.length src then None
          else if String.sub src i (String.length decl_prefix) = decl_prefix
          then (
            let j = ref (i + String.length decl_prefix) in
            while
              !j < String.length src
              &&
              match src.[!j] with
              | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' -> true
              | _ -> false
            do
              incr j
            done ;
            Some (String.sub src i (!j - i)))
          else find (i + 1)
        in
        match find 0 with
        | None -> false
        | Some name -> contains src ("return (" ^ name ^ use_suffix))
  in
  expect
    "GLSL: the USE resolves to the same renamed local as the declaration"
    (renamed_use ~decl_prefix:"sarek_pc_shadow_" ~use_suffix:" * 2)" glsl) ;
  expect
    "WGSL: the USE resolves to the same renamed local as the declaration"
    (renamed_use ~decl_prefix:"sarek_scalar_shadow_" ~use_suffix:" * 2i)" wgsl) ;
  expect
    "GLSL: the helper never reads the push constant"
    (not (contains glsl "pc.n *")) ;
  expect
    "WGSL: the helper never reads the uniform"
    (not (contains wgsl "params.n *")) ;
  (* Formal-parameter collision. *)
  let fglsl = Sarek_codegen.Sarek_ir_glsl.generate formal_ir in
  let fwgsl = Sarek_codegen.Sarek_ir_wgsl.generate formal_ir in
  if glslang_available then
    expect "GLSL formal: glslangValidator accepts the shader" (glslang_ok fglsl) ;
  expect
    "WGSL formal: the formal is used, not params.n"
    (not (contains fwgsl "return (params.n * 3i)")) ;
  if Sys.getenv_opt "DUMPF" <> None then begin
    print_endline "--- GLSL formal ---" ;
    print_endline fglsl ;
    print_endline "--- WGSL formal ---" ;
    print_endline fwgsl
  end ;
  if Sys.getenv_opt "DUMP" <> None then begin
    print_endline "--- GLSL ---" ;
    print_endline glsl ;
    print_endline "--- WGSL ---" ;
    print_endline wgsl
  end ;
  if !failures > 0 then exit 1
