(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Codegen carries no state between generations (backlog-185/200).

    Every source backend used to keep the facts about "the kernel being emitted"
    in module-level [ref]s — the framework tag the intrinsic registries are
    queried with, the kernel's variant table, GLSL's per-kernel helper names,
    WGSL's scalar-parameter list. Two generations therefore shared one set of
    cells, and the second one to start won.

    The cases below cover the two ways that was reachable, and each is the
    executed version of a claim rather than a restatement of it.

    SEQUENTIALLY, WITH NO CONCURRENCY AT ALL. [Sarek_transpile.of_source] wrote
    the framework tag into FOUR emitters at once and never cleared any of them,
    so transpiling anything to OpenCL left [Sarek_ir_cuda]'s tag reading
    ["OpenCL"] for the rest of the process. A later generation on the runtime
    path — e.g. [Cuda_c_plugin.generate_source], which calls
    [generate_with_types] and passes no framework — then queried the registry
    under the wrong tag. Note what these cases do NOT assert: merely that the
    two outputs are equal would also be satisfied by both being wrong, so the
    spelling is pinned directly in each.

    The sharpest instances are the two that emit an identifier the target
    language does not have — OpenCL given CUDA's [sinf], CUDA given GLSL's
    [inversesqrt] — because the device compiler rejects that source outright.
    See the section heading further down. The CUDA [sin]-vs-[sinf] case is kept
    but is the weaker one: whether the emitted [sin] selects the
    double-precision function is decided by nvcc's implicit preinclude, which no
    host here can settle, so this file does not claim it does.

    CONCURRENTLY. Two domains generating two different kernels interleave their
    writes to the shared cells, and the observable is a kernel emitted with
    ANOTHER kernel's variant payload types. Multi-domain execution is a
    supported use case of this library — it is the reason
    [Spoc_framework.Guarded_cache] exists, and that module's own interface says
    the expensive compile runs outside its lock — but codegen sits BEFORE any
    cache on the launch path ([Sarek.Execute.run] calls [B.generate_source] and
    only then reaches [run_source]), so nothing serialized it. The case is
    written as a loop because the pre-fix failure is a race: measured on the
    unfixed tree it mismatched 94, 888 and 1136 times out of 4000 across three
    runs — never zero, but not once per iteration either, which is exactly why
    it is a loop and not a single call. *)

open Sarek_ir_types

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

(** [b.(i) <- Float32.sin a.(i)] as IR — the shape whose CUDA spelling depends
    on the framework tag, because [Float32.sin] is registered in the pure
    registry as ["sinf"] on CUDA and ["sin"] everywhere else. *)
let sin_kernel =
  let a = make_var "a" (TVec TFloat32) in
  let b = make_var "b" (TVec TFloat32) in
  let i = make_var "i" TInt32 in
  {
    default_kernel with
    kern_name = "sin_kernel";
    kern_params =
      [
        DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ];
    kern_body =
      SLet
        ( i,
          EIntrinsic ([], "global_thread_id", []),
          SAssign
            ( LArrayElem ("b", EVar i),
              EIntrinsic (["Float32"], "sin", [EArrayRead ("a", EVar i)]) ) );
  }

(** The same kernel as OCaml source, for the transpiler. Deliberately the same
    kernel: the leak does not need the two to differ, and using one shape keeps
    the case about the tag rather than about the kernel. *)
let sin_kernel_src =
  "fun (a : float32 vector) (b : float32 vector) ->\n\
  \  let i = global_thread_id in\n\
  \  b.(i) <- Float32.sin a.(i)"

let contains ~needle haystack =
  let nl = String.length needle and hl = String.length haystack in
  let rec go i =
    i + nl <= hl && (String.sub haystack i nl = needle || go (i + 1))
  in
  nl = 0 || go 0

(** CUDA emission of [sin_kernel] on the runtime path — no [~framework], which
    is what every backend plugin passes. *)
let cuda_runtime_source () = Sarek_codegen.Sarek_ir_cuda.generate sin_kernel

let check_single_precision ~when_ source =
  Alcotest.check
    Alcotest.bool
    (Printf.sprintf
       "%s: CUDA emission of a float32 Float32.sin must call sinf, not the \
        double-precision sin. Emitted:\n\
        %s"
       when_
       source)
    true
    (contains ~needle:"sinf(" source)

(** [Sarek_transpile.of_source] for [backend] must not change what a LATER,
    unrelated CUDA generation emits. *)
let transpile_does_not_leak backend name () =
  let before = cuda_runtime_source () in
  check_single_precision ~when_:"before any transpile" before ;
  (match Sarek_transpile.of_source backend sin_kernel_src with
  | Ok _ -> ()
  | Error e ->
      Alcotest.failf
        "the transpile this case contaminates with must itself succeed, \
         otherwise the case proves nothing; %s failed: %s"
        name
        (Sarek_transpile.string_of_error e)) ;
  let after = cuda_runtime_source () in
  check_single_precision
    ~when_:("after transpiling an unrelated kernel to " ^ name)
    after ;
  Alcotest.check
    Alcotest.string
    (Printf.sprintf
       "CUDA emission must be identical before and after an unrelated %s \
        transpile; codegen kept the framework tag in module state"
       name)
    before
    after

(** {1 Leaks that produce source the device compiler rejects}

    The [sinf] case above is the WEAK instance of the leak: whether [sin(a[i])]
    in the emitted CUDA actually selects the double-precision function is
    decided by nvcc's implicit preinclude (the emitted source carries no math
    header), and there is no CUDA toolchain on this host to settle it against.
    Both cases below are unambiguous instead: each emits an identifier that DOES
    NOT EXIST in the target language, so the device compiler rejects the source
    outright. They are the two directions measured on the unfixed tree. *)

(** [b.(i) <- Float32.rsqrt a.(i)]. [Float32.rsqrt] is registered as ["rsqrtf"]
    on CUDA and ["rsqrt"] elsewhere, and {!Sarek_pure_registry} overrides GLSL
    to the GLSL builtin ["inversesqrt"] — so this shape spells three ways and
    tells the CUDA and GLSL tags apart. *)
let rsqrt_kernel =
  let a = make_var "a" (TVec TFloat32) in
  let b = make_var "b" (TVec TFloat32) in
  let i = make_var "i" TInt32 in
  {
    default_kernel with
    kern_name = "rsqrt_kernel";
    kern_params =
      [
        DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ];
    kern_body =
      SLet
        ( i,
          EIntrinsic ([], "global_thread_id", []),
          SAssign
            ( LArrayElem ("b", EVar i),
              EIntrinsic (["Float32"], "rsqrt", [EArrayRead ("a", EVar i)]) ) );
  }

let rsqrt_kernel_src =
  "fun (a : float32 vector) (b : float32 vector) ->\n\
  \  let i = global_thread_id in\n\
  \  b.(i) <- Float32.rsqrt a.(i)"

(** A transpile to [via] must not change what a LATER, unrelated generation on
    [emit] spells.

    [expected] is the spelling that target legitimately uses; [foreign] is the
    spelling belonging to [via]'s language, which does not exist in [emit]'s.
    Both polarities are pinned: asserting only the absence of [foreign] would
    also be satisfied by an emitter that stopped emitting the call at all, and
    asserting only before = after would be satisfied by both being wrong. *)
let foreign_identifier_does_not_leak ~emitter ~emit ~kernel_src ~via ~via_name
    ~expected ~foreign () =
  let describe when_ source =
    Printf.sprintf
      "%s: %s emission must spell this call %s — %s is %s's, and does not \
       exist in %s. Emitted:\n\
       %s"
      when_
      emitter
      expected
      foreign
      via_name
      emitter
      source
  in
  let check when_ source =
    Alcotest.check
      Alcotest.bool
      (describe when_ source)
      true
      (contains ~needle:expected source) ;
    Alcotest.check
      Alcotest.bool
      (describe when_ source)
      false
      (contains ~needle:foreign source)
  in
  let before = emit () in
  check "before any transpile" before ;
  (match Sarek_transpile.of_source via kernel_src with
  | Ok _ -> ()
  | Error e ->
      Alcotest.failf
        "the transpile this case contaminates with must itself succeed, \
         otherwise the case proves nothing; %s failed: %s"
        via_name
        (Sarek_transpile.string_of_error e)) ;
  let after = emit () in
  check ("after transpiling an unrelated kernel to " ^ via_name) after ;
  Alcotest.check
    Alcotest.string
    (Printf.sprintf
       "%s emission must be identical before and after an unrelated %s \
        transpile; codegen kept the framework tag in module state"
       emitter
       via_name)
    before
    after

(** {1 Every source backend refuses [%native]}

    An [SNative] node carries a closure that produces device source for a named
    target, and no caller of these five generators supplies one. They used to
    disagree about that: CUDA, OpenCL and Metal raised, while GLSL and WGSL
    emitted [/* native code not supported in <lang> */] and CONTINUED — handing
    back a shader that silently lacks the operation the kernel asked for. These
    cases pin the single refusal that replaced both behaviours.

    Each asserts the MESSAGE, not merely that something raised. An
    exception-type-only check cannot tell this refusal from an unrelated failure
    on the same path — and on GLSL and WGSL, where the arm previously returned
    normally, it could not tell a refusal from no refusal at all.

    Not covered here, deliberately: the PTX emitter, which passes the closure
    its own ["PTX"] tag and really does emit. Refusing there would remove a
    working path. *)

let native_kernel =
  let out = make_var "out" (TVec TInt32) in
  {
    default_kernel with
    kern_name = "native_kernel";
    kern_params =
      [DParam (out, Some {arr_elttype = TInt32; arr_memspace = Global})];
    kern_body =
      SNative
        {
          gpu = (fun ~framework -> "/* " ^ framework ^ " */");
          ocaml = {run = (fun ~block:_ ~grid:_ _args -> ())};
        };
  }

(** The substring every backend's refusal must carry. Taken from the shared
    constant rather than retyped, so rewording the message cannot leave this
    test passing against text no backend emits any more — but kept to a
    distinctive fragment so incidental rewrapping does not fail it. *)
let refusal_fragment = "Express the operation in Sarek"

let refuses_native_block ~emitter ~generate () =
  match generate native_kernel with
  | source ->
      Alcotest.failf
        "%s must refuse a [%%native] block, not emit for it. Before          \
         backlog-185/200 GLSL and WGSL returned a comment here and \
         continued,          which is exactly the silent-wrong output this \
         case exists to catch.          Emitted:\n\
         %s"
        emitter
        source
  | exception e ->
      let msg = Printexc.to_string e in
      Alcotest.check
        Alcotest.bool
        (Printf.sprintf
           "%s's refusal must carry the shared [%%native] diagnostic, so \
            the             user is told what is unsupported and what to do \
            instead. A raise             alone is not enough — any unrelated \
            failure on this path also             raises. Got: %s"
           emitter
           msg)
        true
        (contains ~needle:refusal_fragment msg)

(** {1 Concurrent generations} *)

(** A kernel matching on a one-payload constructor of a variant named [t]. The
    payload TYPE comes from [kern_variants], which the emitters used to read
    from a module-level ref — so two concurrent generations with different
    tables emit each other's payload types. *)
let variant_kernel name payload =
  let out = make_var "out" (TVec TInt32) in
  let i = make_var "i" TInt32 in
  {
    default_kernel with
    kern_name = name;
    kern_params =
      [DParam (out, Some {arr_elttype = TInt32; arr_memspace = Global})];
    kern_variants = [("t", [("A", [payload]); ("B", [])])];
    kern_body =
      SLet
        ( i,
          EIntrinsic ([], "global_thread_id", []),
          SMatch
            ( EVariant ("t", "A", [EConst (CInt32 1l)]),
              [
                ( PConstr ("A", ["x"]),
                  SAssign
                    ( LArrayElem ("out", EVar i),
                      ECast (TInt32, EVar (make_var "x" payload)) ) );
                (PWild, SEmpty);
              ] ) );
  }

(** Iterations per domain. Sized from the measured pre-fix mismatch rate (see
    the header): the smallest of three observed runs was 94 mismatches in 4000
    generations, so a run of this size going green by luck is not a risk worth
    trading more test time against. *)
let concurrent_iterations = 2000

let generations_do_not_interleave (name, generate) () =
  let k1 = variant_kernel "conc_k1" TInt32 in
  let k2 = variant_kernel "conc_k2" TFloat32 in
  let ref1 = generate k1 and ref2 = generate k2 in
  Alcotest.check
    Alcotest.bool
    (Printf.sprintf
       "%s: the two kernels must emit DIFFERENT source, or the case cannot \
        observe one generation reading the other's variant table"
       name)
    false
    (String.equal ref1 ref2) ;
  let mismatches = Atomic.make 0 in
  let run k expected () =
    for _ = 1 to concurrent_iterations do
      if not (String.equal (generate k) expected) then
        ignore (Atomic.fetch_and_add mismatches 1)
    done
  in
  let d1 = Domain.spawn (run k1 ref1) in
  let d2 = Domain.spawn (run k2 ref2) in
  Domain.join d1 ;
  Domain.join d2 ;
  Alcotest.check
    Alcotest.int
    (Printf.sprintf
       "%s: two domains generating two different kernels must each get its own \
        source; a mismatch is one generation reading state the other wrote"
       name)
    0
    (Atomic.get mismatches)

let concurrent_backends =
  [
    ("CUDA", fun k -> Sarek_codegen.Sarek_ir_cuda.generate k);
    ("OpenCL", fun k -> Sarek_codegen.Sarek_ir_opencl.generate k);
    ("Metal", fun k -> Sarek_codegen.Sarek_ir_metal.generate k);
    ("GLSL", fun k -> Sarek_codegen.Sarek_ir_glsl.generate k);
    ("WGSL", fun k -> Sarek_codegen.Sarek_ir_wgsl.generate k);
  ]

let () =
  Sarek_stdlib_meta.force_init () ;
  Alcotest.run
    "codegen generation state"
    [
      ( "framework_tag_is_not_module_state",
        [
          Alcotest.test_case
            "an OpenCL transpile does not change later CUDA emission"
            `Quick
            (transpile_does_not_leak Sarek_transpile.OpenCL "OpenCL");
          Alcotest.test_case
            "a Metal transpile does not change later CUDA emission"
            `Quick
            (transpile_does_not_leak Sarek_transpile.Metal "Metal");
          Alcotest.test_case
            "a GLSL transpile does not change later CUDA emission"
            `Quick
            (transpile_does_not_leak Sarek_transpile.GLSL "GLSL");
          Alcotest.test_case
            "a CUDA transpile does not put sinf into later OpenCL emission"
            `Quick
            (foreign_identifier_does_not_leak
               ~emitter:"OpenCL"
               ~emit:(fun () ->
                 Sarek_codegen.Sarek_ir_opencl.generate sin_kernel)
               ~kernel_src:sin_kernel_src
               ~via:Sarek_transpile.CUDA
               ~via_name:"CUDA"
               ~expected:"sin("
               ~foreign:"sinf(");
          Alcotest.test_case
            "a GLSL transpile does not put inversesqrt into later CUDA emission"
            `Quick
            (foreign_identifier_does_not_leak
               ~emitter:"CUDA"
               ~emit:(fun () ->
                 Sarek_codegen.Sarek_ir_cuda.generate rsqrt_kernel)
               ~kernel_src:rsqrt_kernel_src
               ~via:Sarek_transpile.GLSL
               ~via_name:"GLSL"
               ~expected:"rsqrtf("
               ~foreign:"inversesqrt(");
        ] );
      ( "native_block_is_refused",
        List.map
          (fun (emitter, generate) ->
            Alcotest.test_case
              (emitter ^ " refuses a [%native] block")
              `Quick
              (refuses_native_block ~emitter ~generate))
          [
            ("CUDA", fun k -> Sarek_codegen.Sarek_ir_cuda.generate k);
            ("OpenCL", fun k -> Sarek_codegen.Sarek_ir_opencl.generate k);
            ("Metal", fun k -> Sarek_codegen.Sarek_ir_metal.generate k);
            ("GLSL", fun k -> Sarek_codegen.Sarek_ir_glsl.generate k);
            ("WGSL", fun k -> Sarek_codegen.Sarek_ir_wgsl.generate k);
          ] );
      ( "generations_do_not_interleave",
        List.map
          (fun ((name, _) as backend) ->
            Alcotest.test_case
              (name ^ ": two domains, two kernels, no crosstalk")
              `Quick
              (generations_do_not_interleave backend))
          concurrent_backends );
    ]
