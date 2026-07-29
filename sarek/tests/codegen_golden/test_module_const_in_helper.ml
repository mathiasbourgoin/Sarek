(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * A module-level constant referenced from a HELPER body (backlog-160).
 *
 * THE DEFECT. A module constant is lexically visible inside a helper — the
 * frontend accepts it — but `Sarek_lower_ir` lowers `TMConst` to an `SLet`
 * prepended to the KERNEL body, while helpers are emitted as out-of-line device
 * functions. So the helper's emitted body names an identifier that its
 * translation unit never declares.
 *
 * SEVEN paths break, not one. The item was filed as a "Vulkan/GLSL" defect;
 * CUDA-C, OpenCL, Metal, GLSL, WGSL, PTX and the Interpreter all break, and only
 * Native works — because `Sarek_native_gen` emits the constant as an OCaml `let`
 * in a scope the helper closes over. That asymmetry is the interesting part: it
 * is a CPU-passes / device-fails divergence, the shape that gets a kernel
 * shipped.
 *
 * WHY THIS TEST EXISTS BEFORE THE FIX. Of the 129 files in this tree containing
 * `[%kernel]`, exactly ONE declares a module constant (test_module_const.ml) and
 * it has no helper. The broken combination is covered nowhere, which is why the
 * defect is reachable from the surface language and still invisible.
 *
 * WHAT IT ASSERTS, AND WHY NOT A DEVICE RESULT. The emitted source, not a
 * computed value: the failure is a translation-unit-scope error, so on the
 * backends with a compiler available it is "undeclared identifier" rather than a
 * wrong number, and on the others there is no device in this suite to run. For
 * GLSL the real compiler is the oracle (glslangValidator); for the C-family and
 * WGSL the assertion is declaration-BEFORE-use ordering, because a declaration
 * emitted after the helper is exactly as broken as one not emitted at all and a
 * mere `contains "scale"` would accept it.
 *
 * THE SECOND KERNEL IS THE DISCRIMINATOR, not extra coverage. `dynamic` binds
 * its constant to `thread_idx_x`. The remedy originally approved for this item —
 * hoist module constants to TOP-LEVEL `const`/`__constant`/`constant`
 * declarations — cannot express that: every one of those storage classes
 * requires a compile-time-constant initializer, so hoisting would turn a kernel
 * that works today into one that does not compile. The remedy actually taken
 * (prefix the referenced constants' `SLet`s into the helper body) handles both,
 * at the cost of evaluating the initializer once per helper call. A test that
 * only covered the static case would have passed under either, and would have
 * let the wrong one ship.
 *
 * Run with: dune exec sarek/tests/codegen_golden/test_module_const_in_helper.exe
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std

(* `scale` is a module constant with a CONSTANT initializer, referenced from the
   body of `apply` — a helper, hence out-of-line. The kernel body itself never
   names `scale`, deliberately: if it did, the kernel-body `SLet` would make the
   name appear in the output and a naive "is it declared" check would pass while
   the helper stayed broken. *)
let static_const_kernel =
  [%kernel
    let open Std in
    let (scale : float32) = 2.0 in
    let apply (x : float32) : float32 = x *. scale in
    fun (out : float32 vector) (src : float32 vector) ->
      let t = global_thread_id in
      if t < 1l then out.(0l) <- apply src.(0l)]

(* Same shape, but the initializer is THREAD-DEPENDENT. This is what a
   top-level-constant remedy cannot express. *)
let dynamic_const_kernel =
  [%kernel
    let open Std in
    let (base : int32) = thread_idx_x in
    let bump (x : int32) : int32 = x + base in
    fun (out : int32 vector) ->
      let t = global_thread_id in
      if t < 1l then out.(0l) <- bump 10l]

(* THE NEGATIVE CASE, from review on #362. `gain` is a module constant AND the
   name of a local inside `twice`, which does not reference the constant at all.

   The first version of the fix collected every `EVar` name in the helper body,
   locally-bound ones included, on the reasoning that over-approximating only
   costs a dead declaration. It does not: the backends emit `SLet` FLAT, so the
   prefixed constant and the local became two declarations of `gain` in one
   block — a redeclaration error on a helper that compiled before the fix. The
   kernel body references `gain` too, so the constant is still live and still
   declared there; only the helper must not get a copy. *)
let shadow_const_kernel =
  [%kernel
    let open Std in
    let (gain : float32) = 3.0 in
    let twice (x : float32) : float32 =
      let gain = 2.0 in
      x *. gain
    in
    fun (out : float32 vector) (src : float32 vector) ->
      let t = global_thread_id in
      if t < 1l then out.(0l) <- twice src.(0l) +. gain]

(* Index of the first occurrence, or -1. Ordering is the property: a declaration
   emitted AFTER the helper that uses it is exactly as broken as none. *)
let index_of hay needle =
  let nh = String.length hay and nn = String.length needle in
  let rec go i =
    if i + nn > nh then -1
    else if String.sub hay i nn = needle then i
    else go (i + 1)
  in
  if nn = 0 then -1 else go 0

let ir_of k =
  let _, kirc = k in
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let failures = ref 0

let expect label cond =
  Printf.printf "  %-62s %s\n%!" label (if cond then "OK" else "FAIL") ;
  if not cond then incr failures

let glslang_available =
  Sys.command "command -v glslangValidator > /dev/null 2>&1" = 0

let glslang_ok src =
  let f = Filename.temp_file "sarek_modconst" ".comp" in
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

(* THE PROPERTY, and my first version of it was wrong — recorded because the
   mistake is the interesting part.

   I first asserted that the constant is declared BEFORE the helper's signature,
   on the assumption that visibility means file-scope-before-use. glslangValidator
   then accepted a shader this predicate called broken: the shipped fix declares
   the constant INSIDE the helper body, which is textually after the signature and
   perfectly visible. The assertion encoded ONE valid shape and the correct fix
   uses another — an assertion narrower than the property, in a test written to
   catch exactly that class.

   The property actually is: within the helper's own body, the first mention of
   the constant is its DECLARATION, not a bare use. That covers the shipped shape
   (a prefixed `SLet`) and still fails the defect (a bare use with the only
   declaration in the kernel body, elsewhere in the file).

   The helper body is taken from its signature to the end of the source rather
   than to a matched brace: WGSL, GLSL and the C family close differently, and the
   first-mention test does not need the exact end — anything after the helper can
   only add later mentions, which cannot make a bare first use look like a
   declaration. *)
let declares_before_use ~src ~const_name ~helper_name =
  let sig_at = index_of src (helper_name ^ "(") in
  if sig_at < 0 then false
  else
    let body = String.sub src sig_at (String.length src - sig_at) in
    let first = index_of body const_name in
    if first < 0 then false
    else
      (* A declaration binds; a bare use does not. TWO spellings, because the
         emitted languages differ and asserting only one made this fail on WGSL
         while glslangValidator was already accepting the equivalent GLSL:

           C family / GLSL   `float scale = 2.0f;`     -> name then " = "
           WGSL              `let scale : f32 = 2.0f;` -> name then " : "

         A bare use is followed by an operator or a delimiter (`)`, `*`, `;`,
         `,`), never by either of these, so the predicate still fails the defect
         it was written for. Enumerated rather than reduced to "contains an =",
         which `(x * scale)` inside `return (x * scale) = ...` could not produce
         but a looser predicate would eventually accept. *)
      let after = first + String.length const_name in
      let starts_with tok =
        let n = String.length tok in
        after + n <= String.length body && String.sub body after n = tok
      in
      starts_with " = " || starts_with " : "

(* The helper's own body, brace-matched. The first-mention predicate above
   deliberately runs to end-of-source because a later mention cannot make a bare
   first use look like a declaration. COUNTING is the opposite: the kernel body
   declares the constant too, so an unbounded region would always read 2 and the
   check would be red on the fix and on the defect alike. All five emitted
   languages brace their function bodies, so matching is enough. *)
let helper_body ~src ~helper_name =
  let sig_at = index_of src (helper_name ^ "(") in
  if sig_at < 0 then None
  else
    let n = String.length src in
    let rec find_open i =
      if i >= n then None
      else if src.[i] = '{' then Some i
      else find_open (i + 1)
    in
    match find_open sig_at with
    | None -> None
    | Some o ->
        let rec close i depth =
          if i >= n then None
          else
            match src.[i] with
            | '{' -> close (i + 1) (depth + 1)
            | '}' -> if depth = 1 then Some i else close (i + 1) (depth - 1)
            | _ -> close (i + 1) depth
        in
        Option.map (fun c -> String.sub src o (c - o + 1)) (close o 0)

(* Occurrences of [name] in declaration position — the same two spellings
   [declares_before_use] enumerates, for the same reason. *)
let declaration_count body name =
  let nb = String.length body and nn = String.length name in
  let starts_at i tok =
    let nt = String.length tok in
    i + nt <= nb && String.sub body i nt = tok
  in
  let rec go i acc =
    if i + nn > nb then acc
    else if
      String.sub body i nn = name
      && (starts_at (i + nn) " = " || starts_at (i + nn) " : ")
    then go (i + nn) (acc + 1)
    else go (i + 1) acc
  in
  go 0 0

(* The negative case, from review on #362: a helper whose local merely SHARES a
   module constant's name must get exactly ONE declaration of it — its own. Two
   is the redeclaration the over-approximating collector produced; zero would
   mean the local itself went missing. *)
let check_shadow name gen ~const_name ~helper_name ir =
  match gen ir with
  | src -> (
      match helper_body ~src ~helper_name with
      | None ->
          expect
            (Printf.sprintf
               "%s/shadow: helper %s found in output"
               name
               helper_name)
            false
      | Some body ->
          let c = declaration_count body const_name in
          expect
            (Printf.sprintf
               "%s/shadow: %s declared exactly once in %s (got %d)"
               name
               const_name
               helper_name
               c)
            (c = 1) ;
          if name = "GLSL" && glslang_available then
            expect
              (Printf.sprintf
                 "%s/shadow: glslangValidator accepts the shader"
                 name)
              (glslang_ok src))
  | exception e ->
      Printf.printf
        "  %-62s REFUSED (%s)\n%!"
        (Printf.sprintf "%s/shadow" name)
        ( Printexc.to_string e |> fun s ->
          String.sub s 0 (min 60 (String.length s)) )

let check_backend name gen ~const_name ~helper_name kernel_label ir =
  match gen ir with
  | src ->
      expect
        (Printf.sprintf
           "%s/%s: %s is visible to %s"
           name
           kernel_label
           const_name
           helper_name)
        (declares_before_use ~src ~const_name ~helper_name) ;
      if name = "GLSL" && glslang_available then
        expect
          (Printf.sprintf
             "%s/%s: glslangValidator accepts the shader"
             name
             kernel_label)
          (glslang_ok src)
  | exception e ->
      (* A REFUSAL is an acceptable outcome and must be distinguished from a
         silent miscompile: a backend that cannot express the construct should
         say so. It is recorded, not counted as a failure, so that adding an
         explicit refusal to a backend does not turn this test red. *)
      Printf.printf
        "  %-62s REFUSED (%s)\n%!"
        (Printf.sprintf "%s/%s" name kernel_label)
        ( Printexc.to_string e |> fun s ->
          String.sub s 0 (min 60 (String.length s)) )

let backends =
  [
    ("CUDA", fun ir -> Sarek_codegen.Sarek_ir_cuda.generate ir);
    ("OpenCL", fun ir -> Sarek_codegen.Sarek_ir_opencl.generate ir);
    ("Metal", fun ir -> Sarek_codegen.Sarek_ir_metal.generate ir);
    ("GLSL", fun ir -> Sarek_codegen.Sarek_ir_glsl.generate ir);
    ("WGSL", fun ir -> Sarek_codegen.Sarek_ir_wgsl.generate ir);
  ]

let () =
  print_endline "=== a module constant referenced from a helper body ===" ;
  if not glslang_available then
    print_endline
      "  glslangValidator absent — the GLSL compile gate is SKIPPED (not a \
       pass)" ;
  List.iter
    (fun (name, gen) ->
      check_backend
        name
        gen
        ~const_name:"scale"
        ~helper_name:"apply"
        "static"
        (ir_of static_const_kernel))
    backends ;
  print_endline "" ;
  print_endline "  --- dynamic initializer: the discriminator ---" ;
  List.iter
    (fun (name, gen) ->
      check_backend
        name
        gen
        ~const_name:"base"
        ~helper_name:"bump"
        "dynamic"
        (ir_of dynamic_const_kernel))
    backends ;
  print_endline "" ;
  print_endline "  --- a helper-local that only SHARES the constant's name ---" ;
  List.iter
    (fun (name, gen) ->
      check_shadow
        name
        gen
        ~const_name:"gain"
        ~helper_name:"twice"
        (ir_of shadow_const_kernel))
    backends ;
  if Sys.getenv_opt "DUMP" <> None then
    List.iter
      (fun (name, gen) ->
        Printf.printf
          "--- %s (static) ---\n%s\n"
          name
          (try gen (ir_of static_const_kernel) with e -> Printexc.to_string e))
      backends ;
  if !failures > 0 then exit 1
