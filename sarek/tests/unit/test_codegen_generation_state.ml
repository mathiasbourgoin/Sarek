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

    The two cases below are the two ways that was reachable, and each is the
    executed version of a claim rather than a restatement of it.

    SEQUENTIALLY, WITH NO CONCURRENCY AT ALL. [Sarek_transpile.of_source] wrote
    the framework tag into FOUR emitters at once and never cleared any of them,
    so transpiling anything to OpenCL left [Sarek_ir_cuda]'s tag reading
    ["OpenCL"] for the rest of the process. A later CUDA generation on the
    runtime path — [Cuda_c_plugin.generate_source], which passes no framework —
    then queried the registry under the wrong tag and emitted [sin(a[i])] for a
    [float32] kernel where it had emitted [sinf(a[i])] moments before. That is
    the double-precision function, in valid CUDA C, with no diagnostic anywhere:
    the kernel keeps computing, more slowly and to a different result. Note what
    the case does NOT assert: merely that the two outputs are equal would also
    be satisfied by both being wrong, so the [sinf] spelling is pinned directly.

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
