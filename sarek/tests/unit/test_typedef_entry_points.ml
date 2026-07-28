(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Every public emission entry point must declare the record types it uses
    (backlog-155).

    Each backend has a [generate] that emits the kernel body but NO record
    typedefs and no variant definitions, and a [generate_with_types] that emits
    both. [generate] is not wrong — it is the "I have no custom types" emitter.
    What is wrong is a PUBLIC entry point that accepts a kernel carrying
    [kern_types] and quietly routes to [generate], because the result is source
    that names struct types nobody declared. That had happened in three places:

    - [Sarek_ir_opencl.generate_with_fp64] (re-exported as
      [Opencl_plugin.generate_with_fp64]) delegated to [generate];
    - [Metal_plugin.generate_source] aliased [Sarek_ir_metal.generate], under a
      name identical to the runtime path that does use [generate_with_types];
    - [Sarek_transpile.of_source] — the whole public transpiler API — emitted
      through [generate] for all five backends, even though [conv_kernel]
      faithfully carries [kern_types] across.

    Each case below asserts the typedef IS emitted, and is paired with a
    positive control asserting the typedef-less emitter does NOT emit it. The
    control is what makes the assertion falsifiable: without it, a check for a
    substring in a large generated blob could be passing for any reason. *)

open Sarek_ir_types
module Ocl = Sarek_codegen.Sarek_ir_opencl

let () = Sarek_stdlib_meta.force_init ()

let contains haystack needle =
  let nl = String.length needle and hl = String.length haystack in
  let rec go i =
    i + nl <= hl && (String.sub haystack i nl = needle || go (i + 1))
  in
  nl = 0 || go 0

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

(** A kernel whose body mentions a [point] record, with the record declared in
    [kern_types] exactly as the frontend would leave it. *)
let point_types = [("point", [("x", TFloat32); ("y", TFloat32)])]

let point_kernel () =
  let out = make_var "out" (TVec TFloat32) in
  let p = make_var "p" (TRecord ("point", List.assoc "point" point_types)) in
  {
    kern_name = "k";
    kern_params = [DParam (out, None)];
    kern_locals = [DLocal (p, None)];
    kern_body =
      SSeq
        [
          SAssign
            ( LArrayElem ("out", EConst (CInt32 0l)),
              EBinop
                (Add, ERecordField (EVar p, "x"), ERecordField (EVar p, "y")) );
        ];
    kern_types = point_types;
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

let typedef_line = "} point;"

(* ------------------------------------------------------------------ *)
(* OpenCL: the third entry point                                       *)
(* ------------------------------------------------------------------ *)

let test_opencl_fp64_emits_typedefs () =
  Ocl.current_variants := [] ;
  let src = Ocl.generate_with_fp64 ~types:point_types (point_kernel ()) in
  Alcotest.(check bool)
    "generate_with_fp64 must declare the record it uses"
    true
    (contains src typedef_line)

(** POSITIVE CONTROL for the assertion above: [generate] is the typedef-less
    emitter, so the same check must come out FALSE against it. If this ever
    passes, [typedef_line] has stopped discriminating and the test above is
    vacuous. *)
let test_opencl_plain_generate_omits_typedefs () =
  Ocl.current_variants := [] ;
  let src = Ocl.generate (point_kernel ()) in
  Alcotest.(check bool)
    "control: bare `generate` emits no typedef, so the check discriminates"
    false
    (contains src typedef_line)

(** The pragma is the other half of what [generate_with_fp64] is for; assert it
    is still composed on top of the type-aware body, not lost in the rewiring.
*)
let test_opencl_fp64_still_emits_the_pragma () =
  Ocl.current_variants := [] ;
  let out = make_var "out" (TVec TFloat64) in
  let k =
    {
      (point_kernel ()) with
      kern_params = [DParam (out, None)];
      kern_locals = [];
      kern_body =
        SAssign (LArrayElem ("out", EConst (CInt32 0l)), EConst (CFloat64 1.0));
    }
  in
  let src = Ocl.generate_with_fp64 ~types:point_types k in
  Alcotest.(check bool)
    "fp64 kernel keeps the cl_khr_fp64 pragma"
    true
    (contains src "#pragma OPENCL EXTENSION cl_khr_fp64 : enable") ;
  Alcotest.(check bool)
    "...and the typedef is still there alongside it"
    true
    (contains src typedef_line)

(* ------------------------------------------------------------------ *)
(* The transpiler: same gap, five backends, no plugin involved         *)
(* ------------------------------------------------------------------ *)

let record_kernel_src =
  "let module M = struct\n\
  \  type point = {x : float32; y : float32} [@@sarek.type]\n\
   end in\n\
   fun (out : float32 vector) ->\n\
  \  let i = global_thread_id in\n\
  \  let p = {x = 1.0; y = 2.0} in\n\
  \  out.(i) <- p.x +. p.y"

let transpile backend =
  match Sarek_transpile.of_source backend record_kernel_src with
  | Ok s -> s
  | Error e ->
      Alcotest.failf
        "transpile failed, so this test proves nothing: %s"
        (Sarek_transpile.string_of_error e)

(* [(backend, declaration marker, use marker)]. The C-family backends close the
   typedef with "} point;" and spell the use "point p"; GLSL and WGSL have no
   typedef and emit "struct point" / "let p : point". Keyed per backend so a
   backend that emits nothing cannot pass on another backend's marker. *)
let transpile_cases =
  [
    ("CUDA", Sarek_transpile.CUDA, typedef_line, "point p");
    ("OpenCL", Sarek_transpile.OpenCL, typedef_line, "point p");
    ("Metal", Sarek_transpile.Metal, typedef_line, "point p");
    ("GLSL", Sarek_transpile.GLSL, "struct point", "point p");
    ("WGSL", Sarek_transpile.WGSL, "struct point", "p : point");
  ]

let test_transpile_declares_records () =
  List.iter
    (fun (name, backend, decl_marker, use_marker) ->
      let src = transpile backend in
      (* Check the USE first: if the body does not mention the type, declaring
         it proves nothing and the assertion below would be vacuous. *)
      Alcotest.(check bool)
        (Printf.sprintf
           "%s: body must actually use `point` (marker %S), got:\n%s"
           name
           use_marker
           src)
        true
        (contains src use_marker) ;
      Alcotest.(check bool)
        (Printf.sprintf
           "%s: transpiled source must declare `point` (marker %S), got:\n%s"
           name
           decl_marker
           src)
        true
        (contains src decl_marker))
    transpile_cases

let () =
  Alcotest.run
    "typedef_entry_points"
    [
      ( "opencl_generate_with_fp64",
        [
          Alcotest.test_case
            "emits record typedefs"
            `Quick
            test_opencl_fp64_emits_typedefs;
          Alcotest.test_case
            "control: bare generate does not"
            `Quick
            test_opencl_plain_generate_omits_typedefs;
          Alcotest.test_case
            "still emits the fp64 pragma"
            `Quick
            test_opencl_fp64_still_emits_the_pragma;
        ] );
      ( "transpiler",
        [
          Alcotest.test_case
            "of_source declares records on all five backends"
            `Quick
            test_transpile_declares_records;
        ] );
    ]
