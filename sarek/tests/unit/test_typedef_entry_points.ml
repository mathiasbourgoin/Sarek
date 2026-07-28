(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Every public emission entry point must declare the record types and variant
    types it uses, ABOVE the code that uses them (backlog-155).

    Each backend has a [generate] that emits the kernel body but NO record
    typedefs and no variant definitions, and a [generate_with_types] that emits
    both. [generate] is not wrong — it is the "I have no custom types" emitter.
    What is wrong is a PUBLIC entry point that accepts a kernel carrying
    [kern_types]/[kern_variants] and quietly routes to [generate], because the
    result is source that names struct types nobody declared. That had happened
    in three places:

    - [Sarek_ir_opencl.generate_with_fp64] (re-exported as
      [Opencl_plugin.generate_with_fp64]) delegated to [generate];
    - [Metal_plugin.generate_source] aliased [Sarek_ir_metal.generate], under a
      name identical to the runtime path that does use [generate_with_types] —
      covered by [sarek-metal/test/test_metal_plugin_entry_points.ml], which is
      where the [sarek_metal] library is linkable;
    - [Sarek_transpile.of_source] AND [of_source_with_abi] — the whole public
      transpiler API — emitted through [generate], even though [conv_kernel]
      faithfully carries [kern_types]/[kern_variants] across.

    THREE THINGS THIS FILE DOES THAT THE FIRST VERSION OF IT DID NOT, each added
    because a mutation of the fixed code left the first version GREEN:

    1. ORDER, NOT PRESENCE. Every marker pair is checked with
    [check_declared_before_use], which fails unless the DECLARATION's offset is
    strictly below the USE's. Presence alone is not a gate: moving the typedef
    block of [Sarek_ir_opencl.generate_with_types] from the prologue to the
    epilogue — emitting [point p = (point){…}] above [} point;], which no OpenCL
    C compiler accepts — left the presence-only version of this file green, exit
    0, 4 cases OK.

    2. [of_source_with_abi] HAS ITS OWN ROW. Starving its [~types] argument
    ([~types:[]] rather than [ir_kernel.kern_types]) changed nothing in the
    whole 1701-case suite: the ABI entry point was reached by no test at all, so
    making the label required bought coverage nowhere.

    3. A VARIANT-CARRYING FIXTURE. Every fixture here used to have
    [kern_variants = []], so the "and no variant definitions" half of the
    sentence above was not falsifiable by anything in this file — the
    [grep-absent] shape, where a test's prose is wider than its assertions.
    [Sarek_ir_opencl.generate_with_types] also sets [current_variants] from
    [k.kern_variants], which is the binding source for [SMatch], so a variant
    fixture is cheap and load-bearing.

    Each positive assertion is paired with a control asserting the typedef-less
    emitter does NOT emit the declaration. The control is what makes the
    assertion falsifiable: without it, a check for a substring in a large
    generated blob could be passing for any reason. *)

open Sarek_ir_types
module Ocl = Sarek_codegen.Sarek_ir_opencl

let () = Sarek_stdlib_meta.force_init ()

(** Offset of the first occurrence of [needle], or [None]. Offsets — not a
    boolean — because the defect this file exists to catch is an ORDERING one
    and [contains] cannot see it. *)
let index_of haystack needle =
  let nl = String.length needle and hl = String.length haystack in
  let rec go i =
    if i + nl > hl then None
    else if String.sub haystack i nl = needle then Some i
    else go (i + 1)
  in
  if nl = 0 then Some 0 else go 0

let contains haystack needle = index_of haystack needle <> None

(** The assertion this whole file is built on: [decl] must be present, [use]
    must be present, and [decl] must come FIRST.

    All three legs are load-bearing and each fails with its own message.
    - no [use]: the body never mentions the type, so declaring it would prove
      nothing and the [decl] leg would be vacuous;
    - no [decl]: the entry point routed to the typedef-less emitter — the
      original backlog-155 defect;
    - [use] before [decl]: the declaration is emitted, but below the code that
      needs it, which is invalid source just the same. *)
let check_declared_before_use ~ctx ~decl ~use src =
  match (index_of src decl, index_of src use) with
  | _, None ->
      Alcotest.failf
        "%s: the body never uses the type (marker %S), so this case proves \
         nothing. Emitted source:\n\
         %s"
        ctx
        use
        src
  | None, Some _ ->
      Alcotest.failf
        "%s: emitted source uses the type but never declares it (missing \
         marker %S). Emitted source:\n\
         %s"
        ctx
        decl
        src
  | Some d, Some u ->
      Alcotest.(check bool)
        (Printf.sprintf
           "%s: declaration %S is at offset %d but the use %S is at offset %d \
            — the declaration must come FIRST, this source is rejected by any \
            compiler. Emitted source:\n\
            %s"
           ctx
           decl
           d
           use
           u
           src)
        true
        (d < u)

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

(** The variant counterpart. [kern_types] is empty here on purpose: this fixture
    exercises the [kern_variants] arm of [generate_with_types] on its own, so
    deleting only that arm is red here and the record cases stay green. *)
let shape_constrs = [("Circle", [TFloat32]); ("Square", [])]

let variant_kernel () =
  let out = make_var "out" (TVec (TVariant ("shape", shape_constrs))) in
  {
    kern_name = "kv";
    kern_params = [DParam (out, None)];
    kern_locals = [];
    kern_body =
      SAssign
        ( LArrayElem ("out", EConst (CInt32 0l)),
          EVariant ("shape", "Circle", [EConst (CFloat32 1.0)]) );
    kern_types = [];
    kern_variants = [("shape", shape_constrs)];
    kern_funcs = [];
    kern_native_fn = None;
  }

let typedef_line = "} point;"

let record_use = "point p"

let variant_typedef_line = "} shape;"

(* Deliberately NOT the bare constructor name: [gen_variant_def] emits the
   constructor's DEFINITION (`static inline shape make_shape_Circle(`) as part
   of the declaration block, so a bare `make_shape_Circle` would be found
   inside the declaration itself and the ordering check would hold vacuously.
   The `= ` prefix pins the CALL in the kernel body. *)
let variant_use = "= make_shape_Circle("

(* ------------------------------------------------------------------ *)
(* OpenCL: the third entry point                                       *)
(* ------------------------------------------------------------------ *)

let test_opencl_fp64_emits_typedefs () =
  Ocl.current_variants := [] ;
  let src = Ocl.generate_with_fp64 ~types:point_types (point_kernel ()) in
  check_declared_before_use
    ~ctx:"generate_with_fp64, record"
    ~decl:typedef_line
    ~use:record_use
    src

(** POSITIVE CONTROL for the assertion above: [generate] is the typedef-less
    emitter, so the same check must come out FALSE against it. If this ever
    passes, [typedef_line] has stopped discriminating and the test above is
    vacuous. *)
let test_opencl_plain_generate_omits_typedefs () =
  Ocl.current_variants := [] ;
  let src = Ocl.generate (point_kernel ()) in
  Alcotest.(check bool)
    "control: bare `generate` uses the record"
    true
    (contains src record_use) ;
  Alcotest.(check bool)
    "control: bare `generate` emits no typedef, so the check discriminates"
    false
    (contains src typedef_line)

(** The other half of the sentence at the top of this file: [generate] emits no
    VARIANT definitions either, and [generate_with_fp64] must. Without this
    case, deleting the [kern_variants] arm of [generate_with_types] left every
    assertion in this file green, because every other fixture has
    [kern_variants = []]. *)
let test_opencl_fp64_emits_variant_defs () =
  Ocl.current_variants := [] ;
  let src = Ocl.generate_with_fp64 ~types:[] (variant_kernel ()) in
  check_declared_before_use
    ~ctx:"generate_with_fp64, variant"
    ~decl:variant_typedef_line
    ~use:variant_use
    src

(** Control for the variant case, same shape as the record one. *)
let test_opencl_plain_generate_omits_variant_defs () =
  Ocl.current_variants := [] ;
  let src = Ocl.generate (variant_kernel ()) in
  Alcotest.(check bool)
    "control: bare `generate` calls the variant constructor"
    true
    (contains src variant_use) ;
  Alcotest.(check bool)
    "control: bare `generate` emits no variant definition"
    false
    (contains src variant_typedef_line)

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
(* The transpiler: same gap, five backends + the ABI entry point       *)
(* ------------------------------------------------------------------ *)

let record_kernel_src =
  "let module M = struct\n\
  \  type point = {x : float32; y : float32} [@@sarek.type]\n\
   end in\n\
   fun (out : float32 vector) ->\n\
  \  let i = global_thread_id in\n\
  \  let p = {x = 1.0; y = 2.0} in\n\
  \  out.(i) <- p.x +. p.y"

let transpile backend () =
  match Sarek_transpile.of_source backend record_kernel_src with
  | Ok s -> s
  | Error e ->
      Alcotest.failf
        "transpile failed, so this test proves nothing: %s"
        (Sarek_transpile.string_of_error e)

(** [of_source_with_abi] is a SEPARATE emission entry point, not a wrapper over
    [of_source]: it re-runs the pipeline and calls [Sarek_ir_wgsl] itself
    (Sarek_transpile.ml:344-352). Nothing exercised it, so starving its [~types]
    argument was invisible to the entire suite. *)
let transpile_with_abi backend () =
  match Sarek_transpile.of_source_with_abi backend record_kernel_src with
  | Ok (code, _abi) -> code
  | Error e ->
      Alcotest.failf
        "transpile-with-abi failed, so this test proves nothing: %s"
        (Sarek_transpile.string_of_error e)

(* [(label, emit, declaration marker, use marker)]. The C-family backends close
   the typedef with "} point;" and spell the use "point p"; GLSL and WGSL have
   no typedef and emit "struct point" / "let p : point". Keyed per row so a
   backend that emits nothing cannot pass on another backend's marker, and
   carrying the emitter as a closure rather than a backend tag so the ABI entry
   point gets a row of its own. *)
let transpile_cases =
  [
    ("CUDA", transpile Sarek_transpile.CUDA, typedef_line, record_use);
    ("OpenCL", transpile Sarek_transpile.OpenCL, typedef_line, record_use);
    ("Metal", transpile Sarek_transpile.Metal, typedef_line, record_use);
    ("GLSL", transpile Sarek_transpile.GLSL, "struct point", record_use);
    ("WGSL", transpile Sarek_transpile.WGSL, "struct point", "p : point");
    ( "WGSL/of_source_with_abi",
      transpile_with_abi Sarek_transpile.WGSL,
      "struct point",
      "p : point" );
  ]

let test_transpile_declares_records () =
  List.iter
    (fun (name, emit, decl_marker, use_marker) ->
      let src = emit () in
      check_declared_before_use
        ~ctx:(Printf.sprintf "transpiler/%s" name)
        ~decl:decl_marker
        ~use:use_marker
        src)
    transpile_cases

(* ------------------------------------------------------------------ *)
(* The transpiler's OpenCL arm: the fp64 pragma, not just the typedefs *)
(* ------------------------------------------------------------------ *)

(** [Sarek_transpile.emit_backend]'s OpenCL arm was fixed to
    [generate_with_types] along with the other four, which restored the typedefs
    and left the fp64 pragma still missing — [generate_with_fp64] is the entry
    point that composes both, and nothing on the transpile path added the pragma
    anywhere (grep for `cl_khr` over Sarek_transpile.ml returned nothing).

    That a float64 kernel REACHES that arm is the thing this pair pins. It is
    not obvious: the transpiler could plausibly have refused fp64 upstream, at
    the typer or the lowering, in which case the pragma would be dead code and
    the arm's choice would not matter. It does not refuse — the first case here
    fails at [transpile_ocl] rather than at an assertion if that ever changes,
    and the message says so.

    The float32 control is the falsifier. Without it, "the output contains the
    pragma" is also satisfied by prefixing every OpenCL kernel unconditionally,
    which would be a different bug (a pragma naming an extension the program
    does not use, on every float32 kernel the transpiler emits). *)

let fp64_kernel_src =
  "fun (a : float64 vector) (b : float64 vector) ->\n\
  \  let i = global_thread_id in\n\
  \  b.(i) <- a.(i) +. a.(i)"

let f32_kernel_src =
  "fun (a : float32 vector) (b : float32 vector) ->\n\
  \  let i = global_thread_id in\n\
  \  b.(i) <- a.(i) +. a.(i)"

let fp64_pragma = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable"

let transpile_ocl ~ctx src =
  match Sarek_transpile.of_source Sarek_transpile.OpenCL src with
  | Ok s -> s
  | Error e ->
      Alcotest.failf
        "%s: transpile failed, so this case proves nothing: %s"
        ctx
        (Sarek_transpile.string_of_error e)

let test_transpile_opencl_fp64_emits_pragma () =
  let src = transpile_ocl ~ctx:"transpiler/OpenCL fp64" fp64_kernel_src in
  (* Vacuity guard: if the emitted kernel does not actually use `double`, the
     pragma assertion below is about a kernel that never needed it. *)
  Alcotest.(check bool)
    (Printf.sprintf
       "the fp64 kernel really does emit `double` — otherwise the pragma check \
        is vacuous. Emitted source:\n\
        %s"
       src)
    true
    (contains src "double") ;
  (* Order matters as much as presence: an extension pragma below the code that
     uses the extension enables nothing. *)
  check_declared_before_use
    ~ctx:"transpiler/OpenCL fp64"
    ~decl:fp64_pragma
    ~use:"double"
    src

let test_transpile_opencl_float32_omits_pragma () =
  let src = transpile_ocl ~ctx:"transpiler/OpenCL float32" f32_kernel_src in
  Alcotest.(check bool)
    (Printf.sprintf
       "control: a float32 kernel must NOT carry the fp64 pragma — the prefix \
        is conditional on the kernel using float64, not unconditional. Emitted \
        source:\n\
        %s"
       src)
    false
    (contains src fp64_pragma)

let () =
  Alcotest.run
    "typedef_entry_points"
    [
      ( "opencl_generate_with_fp64",
        [
          Alcotest.test_case
            "emits record typedefs, above the use"
            `Quick
            test_opencl_fp64_emits_typedefs;
          Alcotest.test_case
            "control: bare generate does not"
            `Quick
            test_opencl_plain_generate_omits_typedefs;
          Alcotest.test_case
            "emits variant definitions, above the use"
            `Quick
            test_opencl_fp64_emits_variant_defs;
          Alcotest.test_case
            "control: bare generate does not"
            `Quick
            test_opencl_plain_generate_omits_variant_defs;
          Alcotest.test_case
            "still emits the fp64 pragma"
            `Quick
            test_opencl_fp64_still_emits_the_pragma;
        ] );
      ( "transpiler",
        [
          Alcotest.test_case
            "of_source and of_source_with_abi declare records before using them"
            `Quick
            test_transpile_declares_records;
          Alcotest.test_case
            "of_source OpenCL enables cl_khr_fp64 above the first `double`"
            `Quick
            test_transpile_opencl_fp64_emits_pragma;
          Alcotest.test_case
            "control: a float32 kernel gets no fp64 pragma"
            `Quick
            test_transpile_opencl_float32_omits_pragma;
        ] );
    ]
