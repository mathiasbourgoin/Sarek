(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * IR-level test for match-EXPRESSION payload bindings on every source backend
 * (task #75; supersedes the #73 fail-loud stopgap on GLSL/WGSL).
 *
 * THE DEFECT THIS PINS: a match-EXPRESSION ([EMatch]) whose case pattern binds
 * a constructor payload used to lower to a nested ternary (CUDA/OpenCL/Metal),
 * a nested ternary (GLSL) or nested [select()] (WGSL) that DISCARDED the
 * binder. Expression position has nowhere to put a declaration, so the case
 * body was emitted with the binder name left dangling:
 *
 *   out[i] = (opt[i].tag == OptSome) ? (y + 1.0f) : 0.0f;   <- 'y' undeclared
 *
 * On the C-family backends that is SILENT-WRONG whenever a same-named variable
 * happens to be in scope (the kernel then computes with an unrelated value and
 * returns a plausible wrong answer, with no error anywhere); otherwise it is a
 * device-compiler error far from the cause. On GLSL/WGSL #73 converted it into
 * a loud [Unsupported_construct].
 *
 * THE FIX: {!Sarek_ir_codegen.subst_ematch_payloads} — one shared,
 * backend-parameterised substitution that rewrites each payload binder in the
 * case body into the SAME payload access path the [SMatch] declaration path
 * already emits ([<scrut>.data.<C>_v] for the C family, [<scrut>.<C>_v] for the
 * shader backends). No per-backend copy (#94).
 *
 * Every assertion below is red on the pre-fix generators: the C-family ones
 * because the payload access is absent and the bare binder is present, the
 * shader ones because generation raises instead of emitting anything.
 ******************************************************************************)

open Sarek_ir_types
module Cuda = Sarek_codegen.Sarek_ir_cuda
module Opencl = Sarek_codegen.Sarek_ir_opencl
module Metal = Sarek_codegen.Sarek_ir_metal
module Glsl = Sarek_codegen.Sarek_ir_glsl
module Wgsl = Sarek_codegen.Sarek_ir_wgsl

let make_var ?(mut = false) name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = mut}

(* --- string helpers ------------------------------------------------------ *)

let contains ~haystack ~needle =
  let hl = String.length haystack and nl = String.length needle in
  let rec loop i =
    if i + nl > hl then false
    else if String.sub haystack i nl = needle then true
    else loop (i + 1)
  in
  nl = 0 || loop 0

let is_ident_char c =
  (c >= 'a' && c <= 'z')
  || (c >= 'A' && c <= 'Z')
  || (c >= '0' && c <= '9')
  || c = '_'

(* Whether [src] mentions [name] as a whole identifier (not as a substring of a
   longer one). This is what distinguishes the DEFECT (a dangling binder name
   standing alone in an expression) from the FIX (the binder gone, replaced by
   a field access path). *)
let mentions_identifier ~src ~name =
  let sl = String.length src and nl = String.length name in
  let rec loop i =
    if i + nl > sl then false
    else if
      String.sub src i nl = name
      && (i = 0 || not (is_ident_char src.[i - 1]))
      && (i + nl = sl || not (is_ident_char src.[i + nl]))
    then true
    else loop (i + 1)
  in
  nl > 0 && loop 0

(* --- kernels ------------------------------------------------------------- *)

let opt_constrs = [("OptNone", []); ("OptSome", [TFloat32])]

let opt_type = TVariant ("Opt", opt_constrs)

let pair_constrs = [("MkOne", [TFloat32]); ("MkPair", [TFloat32; TFloat32])]

let pair_type = TVariant ("Pair", pair_constrs)

(* out.[idx] <- (match scrut.[idx] with <cases>) *)
let kernel_with ~vname ~constrs ~elt cases =
  let scrut = make_var "opt" (TVec elt) in
  let out = make_var "out" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("out", EVar idx),
            EMatch (EArrayRead ("opt", EVar idx), cases) ) )
  in
  {
    kern_name = "ematch_payload_probe";
    kern_params =
      [
        DParam (scrut, Some {arr_elttype = elt; arr_memspace = Global});
        DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ];
    kern_locals = [];
    kern_body = body;
    kern_types = [];
    kern_variants = [(vname, constrs)];
    kern_funcs = [];
    kern_native_fn = None;
  }

let v name = EVar (make_var name TFloat32)

(* `match opt.(idx) with OptSome y -> y +. 1.0 | OptNone -> 0.0` *)
let single_payload_kernel =
  kernel_with
    ~vname:"Opt"
    ~constrs:opt_constrs
    ~elt:opt_type
    [
      (PConstr ("OptSome", ["y"]), EBinop (Add, v "y", EConst (CFloat32 1.0)));
      (PConstr ("OptNone", []), EConst (CFloat32 0.0));
    ]

(* `match opt.(idx) with MkPair (a, b) -> a +. b | MkOne c -> c` *)
let multi_payload_kernel =
  kernel_with
    ~vname:"Pair"
    ~constrs:pair_constrs
    ~elt:pair_type
    [
      (PConstr ("MkPair", ["a"; "b"]), EBinop (Add, v "a", v "b"));
      (PConstr ("MkOne", ["c"]), v "c");
    ]

(* A SINGLE-case match expression: the backends short-circuit this shape to
   "just emit the body", which discarded the binder just as silently. *)
let single_case_kernel =
  kernel_with
    ~vname:"Opt"
    ~constrs:opt_constrs
    ~elt:opt_type
    [(PConstr ("OptSome", ["y"]), EBinop (Mul, v "y", v "y"))]

(* Shadowing: the INNER match rebinds `y`, so the inner body's `y` must resolve
   to the inner scrutinee's payload, not the outer one. Capture-avoiding
   substitution is the whole reason this is a traversal and not a string
   replace. *)
let shadowing_kernel =
  let inner =
    EMatch
      ( EArrayRead ("opt", EVar (make_var "idx" TInt32)),
        [
          (PConstr ("OptSome", ["y"]), v "y");
          (PConstr ("OptNone", []), EConst (CFloat32 0.0));
        ] )
  in
  kernel_with
    ~vname:"Opt"
    ~constrs:opt_constrs
    ~elt:opt_type
    [
      (PConstr ("OptSome", ["y"]), EBinop (Add, v "y", inner));
      (PConstr ("OptNone", []), EConst (CFloat32 0.0));
    ]

(* THE SILENT-WRONG SHAPE, and it is plainly reachable: the binder name comes
   from user source (`match s with Circle r -> ..`), so any same-named value
   already in scope — a kernel parameter, an enclosing `let`, a loop index —
   satisfies the dropped reference. The generated source then COMPILES, and the
   kernel returns a plausible wrong answer with no diagnostic on any backend or
   in any vendor compiler. Here an enclosing `let r = 100.0` shadows the
   `OptSome r` payload: pre-fix the arm emits `(r * 2.0)`, reading 100.0.

   This is why "the device compiler rejects it" is not a safety net: it only
   holds for the subset where nothing happens to be named the same. *)
let silent_wrong_kernel =
  let scrut = make_var "opt" (TVec opt_type) in
  let out = make_var "out" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  let r = make_var "r" TFloat32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( r,
            EConst (CFloat32 100.0),
            SAssign
              ( LArrayElem ("out", EVar idx),
                EMatch
                  ( EArrayRead ("opt", EVar idx),
                    [
                      ( PConstr ("OptSome", ["r"]),
                        EBinop (Mul, v "r", EConst (CFloat32 2.0)) );
                      (PConstr ("OptNone", []), EConst (CFloat32 0.0));
                    ] ) ) ) )
  in
  {
    kern_name = "ematch_silent_wrong_probe";
    kern_params =
      [
        DParam (scrut, Some {arr_elttype = opt_type; arr_memspace = Global});
        DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ];
    kern_locals = [];
    kern_body = body;
    kern_types = [];
    kern_variants = [("Opt", opt_constrs)];
    kern_funcs = [];
    kern_native_fn = None;
  }

(* `match e with x -> ...`: the PPX lowers a plain variable pattern to
   PConstr ("", [x]), which binds the WHOLE scrutinee rather than a payload.
   Same discard, and #73's fail-loud guard rejected it outright. *)
let var_pattern_kernel =
  let whole = EVar (make_var "whole" opt_type) in
  kernel_with
    ~vname:"Opt"
    ~constrs:opt_constrs
    ~elt:opt_type
    [
      ( PConstr ("", ["whole"]),
        (* The bound whole value is then matched on, so the binder is genuinely
           referenced: `match opt.(idx) with whole -> match whole with ..`. *)
        EMatch
          ( whole,
            [
              (PConstr ("OptSome", ["y"]), v "y");
              (PConstr ("OptNone", []), EConst (CFloat32 0.0));
            ] ) );
    ]

(* Tag-only, throwaway-binder and PWild shapes — unaffected by the fix, pinned
   so it cannot regress them (this is the coverage the #73-era
   test_shader_ematch_payload held for GLSL/WGSL, here widened to all five
   backends). *)
let tag_only_kernel =
  kernel_with
    ~vname:"Opt"
    ~constrs:opt_constrs
    ~elt:opt_type
    [
      (PConstr ("OptSome", []), EConst (CFloat32 1.0));
      (PConstr ("OptNone", []), EConst (CFloat32 0.0));
    ]

(* `OptSome _`: binds the throwaway name "_", which is never referenced. *)
let wildcard_payload_kernel =
  kernel_with
    ~vname:"Opt"
    ~constrs:opt_constrs
    ~elt:opt_type
    [
      (PConstr ("OptSome", ["_"]), EConst (CFloat32 1.0));
      (PConstr ("OptNone", []), EConst (CFloat32 0.0));
    ]

let pwild_kernel =
  kernel_with
    ~vname:"Opt"
    ~constrs:opt_constrs
    ~elt:opt_type
    [
      (PConstr ("OptSome", []), EConst (CFloat32 1.0));
      (PWild, EConst (CFloat32 0.0));
    ]

(* --- backends ------------------------------------------------------------ *)

(* [union] is the extra hop the C-family tagged union needs ([.data.]); the
   shader backends flatten payloads straight into the variant struct. *)
type backend = {label : string; gen : kernel -> string; union : string}

let backends =
  [
    {
      label = "CUDA";
      gen = (fun k -> Cuda.generate_with_types ~types:[] k);
      union = ".data.";
    };
    {
      label = "OpenCL";
      gen = (fun k -> Opencl.generate_with_types ~types:[] k);
      union = ".data.";
    };
    {
      label = "Metal";
      gen = (fun k -> Metal.generate_with_types ~types:[] k);
      union = ".data.";
    };
    {
      label = "GLSL";
      gen = (fun k -> Glsl.generate_with_types ~types:[] k);
      union = ".";
    };
    {
      label = "WGSL";
      gen = (fun k -> Wgsl.generate_with_types ~types:[] k);
      union = ".";
    };
  ]

let generate b k =
  try b.gen k
  with e ->
    Alcotest.failf
      "%s: generation raised instead of emitting the payload binding: %s"
      b.label
      (Printexc.to_string e)

(* THE ASSIGNMENT LINE, not the whole module, is what every assertion below
   inspects. The variant preamble each backend emits already contains the
   payload access path (the constructor function assigns [r.data.C_v = v]), so
   a whole-source search would be satisfied by the preamble alone and would
   stay green with the match lowering still broken. The kernel body here is a
   single [<out>[idx] = <match>;] line ([out] is spelled [outv] on GLSL, where
   [out] is reserved — hence matching on the index, not the name). *)
let assign_line b k =
  let src = generate b k in
  let lines = String.split_on_char '\n' src in
  match
    List.filter (fun l -> contains ~haystack:l ~needle:"[idx] = ") lines
  with
  | [l] -> l
  | [] ->
      Alcotest.failf
        "%s: no kernel assignment line in the generated source:\n%s"
        b.label
        src
  | ls ->
      Alcotest.failf
        "%s: expected exactly one kernel assignment line, got %d"
        b.label
        (List.length ls)

(* --- assertions ---------------------------------------------------------- *)

let check_payload_bound b ~kernel ~binders ~accesses () =
  let line = assign_line b kernel in
  List.iter
    (fun needle ->
      Alcotest.(check bool)
        (Printf.sprintf
           "%s: the match reads the payload as %S — found: %s"
           b.label
           needle
           line)
        true
        (contains ~haystack:line ~needle))
    (List.map (fun (c, suffix) -> b.union ^ c ^ "_v" ^ suffix) accesses) ;
  List.iter
    (fun name ->
      Alcotest.(check bool)
        (Printf.sprintf
           "%s: binder %S is NOT left dangling (a match expression has nowhere \
            to declare it, so an occurrence means the kernel reads an \
            undefined or unrelated same-named value) — found: %s"
           b.label
           name
           line)
        false
        (mentions_identifier ~src:line ~name))
    binders

(* The inner arm's `y` must read the INNER scrutinee. Both matches share the
   same scrutinee expression here, so correctness is pinned by: no dangling
   `y`, and a payload read per arm (>= 2 on the one assignment line). *)
let check_shadowing b () =
  let line = assign_line b shadowing_kernel in
  Alcotest.(check bool)
    (Printf.sprintf
       "%s: no dangling binder after nested rebinding — found: %s"
       b.label
       line)
    false
    (mentions_identifier ~src:line ~name:"y") ;
  let needle = b.union ^ "OptSome_v" in
  let count =
    let nl = String.length needle and sl = String.length line in
    let rec loop i acc =
      if i + nl > sl then acc
      else if String.sub line i nl = needle then loop (i + 1) (acc + 1)
      else loop (i + 1) acc
    in
    loop 0 0
  in
  Alcotest.(check bool)
    (Printf.sprintf
       "%s: both the outer and the inner arm read a payload (found %d, \
        expected >= 2) — %s"
       b.label
       count
       line)
    true
    (count >= 2)

(* The variable pattern binds the whole scrutinee, so it substitutes to the
   scrutinee expression itself — no payload field, no dangling binder. *)
let check_var_pattern b () =
  let line = assign_line b var_pattern_kernel in
  List.iter
    (fun name ->
      Alcotest.(check bool)
        (Printf.sprintf
           "%s: binder %S of `match e with x -> ..` is not left dangling — \
            found: %s"
           b.label
           name
           line)
        false
        (mentions_identifier ~src:line ~name))
    ["whole"; "y"] ;
  Alcotest.(check bool)
    (Printf.sprintf
       "%s: the whole-value binder resolves to the scrutinee, so the inner \
        match reads its payload — found: %s"
       b.label
       line)
    true
    (contains ~haystack:line ~needle:(b.union ^ "OptSome_v"))

(* Shapes the fix must leave alone: they bind nothing usable, so the assignment
   must still be a pure tag dispatch with no payload read at all. *)
let check_unaffected ~what kernel b () =
  let line = assign_line b kernel in
  Alcotest.(check bool)
    (Printf.sprintf "%s: %s still dispatches on the tag — %s" b.label what line)
    true
    (contains ~haystack:line ~needle:".tag == OptSome") ;
  Alcotest.(check bool)
    (Printf.sprintf
       "%s: %s emits no payload read (nothing binds one) — %s"
       b.label
       what
       line)
    false
    (contains ~haystack:line ~needle:(b.union ^ "OptSome_v"))

let () =
  let open Alcotest in
  let per_backend name f =
    List.map (fun b -> test_case b.label `Quick (f b)) backends |> fun cases ->
    (name, cases)
  in
  run
    "EMatch payload binding (#75)"
    [
      per_backend "single-payload" (fun b ->
          check_payload_bound
            b
            ~kernel:single_payload_kernel
            ~binders:["y"]
            ~accesses:[("OptSome", "")]);
      per_backend "multi-payload" (fun b ->
          check_payload_bound
            b
            ~kernel:multi_payload_kernel
            ~binders:["a"; "b"; "c"]
            ~accesses:[("MkPair", "._0"); ("MkPair", "._1"); ("MkOne", "")]);
      per_backend "single-case-shortcut" (fun b ->
          check_payload_bound
            b
            ~kernel:single_case_kernel
            ~binders:["y"]
            ~accesses:[("OptSome", "")]);
      per_backend "nested-shadowing" check_shadowing;
      per_backend "silent-wrong-shadowed-binder" (fun b ->
          check_payload_bound
            b
            ~kernel:silent_wrong_kernel
            ~binders:["r"]
            ~accesses:[("OptSome", "")]);
      per_backend "variable-pattern" check_var_pattern;
      per_backend
        "tag-only-unaffected"
        (check_unaffected ~what:"a tag-only match" tag_only_kernel);
      per_backend
        "wildcard-payload-unaffected"
        (check_unaffected
           ~what:"a throwaway-binder match (`OptSome _`)"
           wildcard_payload_kernel);
      per_backend
        "pwild-unaffected"
        (check_unaffected ~what:"a PWild catch-all" pwild_kernel);
    ]
