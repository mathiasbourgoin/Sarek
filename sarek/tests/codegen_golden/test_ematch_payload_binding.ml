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
module Backend_error = Sarek_backend_error.Backend_error

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
    default_kernel with
    kern_name = "ematch_payload_probe";
    kern_params =
      [
        DParam (scrut, Some {arr_elttype = elt; arr_memspace = Global});
        DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ];
    kern_body = body;
    kern_variants = [(vname, constrs)];
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
    default_kernel with
    kern_name = "ematch_silent_wrong_probe";
    kern_params =
      [
        DParam (scrut, Some {arr_elttype = opt_type; arr_memspace = Global});
        DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ];
    kern_body = body;
    kern_variants = [("Opt", opt_constrs)];
  }

(* CAPTURE. The outer arm's replacement term is built from the OUTER scrutinee
   and therefore has FREE VARIABLES — here the index `idx` inside
   `shp.(idx)`. The inner match then rebinds that very name (`Pick idx`).

   Substituting the inner binder into the already-injected outer term captures
   it: the arm ends up reading `shp[p.data.Pick_v]` where the source says
   `shp.(idx)`. That is valid code on every backend, it compiles clean, and it
   returns wrong numbers — the exact failure mode this whole change exists to
   remove, reintroduced by the fix itself if the rewrite is applied per-node as
   the backend walks down instead of once over the whole subtree.

   Handling only the other half of capture (an inner binder shadowing an outer
   MAPPING) is not enough, and a test whose two matches share one scrutinee
   cannot tell the two halves apart. *)
let capture_kernel =
  let shp = make_var "shp" (TVec opt_type) in
  let out = make_var "out" (TVec TFloat32) in
  let pick_constrs = [("NoPick", []); ("Pick", [TInt32])] in
  let pick_type = TVariant ("Choice", pick_constrs) in
  let p = make_var "p" pick_type in
  let idx = make_var "idx" TInt32 in
  let inner =
    EMatch
      ( EVar p,
        [
          (* rebinds `idx`, the free variable of the injected outer term *)
          (PConstr ("Pick", ["idx"]), v "y");
          (PConstr ("NoPick", []), EConst (CFloat32 0.0));
        ] )
  in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( p,
            EVariant ("Choice", "Pick", [EConst (CInt32 0l)]),
            SAssign
              ( LArrayElem ("out", EVar idx),
                EMatch
                  ( EArrayRead ("shp", EVar idx),
                    [
                      (PConstr ("OptSome", ["y"]), inner);
                      (PConstr ("OptNone", []), EConst (CFloat32 0.0));
                    ] ) ) ) )
  in
  {
    default_kernel with
    kern_name = "ematch_capture_probe";
    kern_params =
      [
        DParam (shp, Some {arr_elttype = opt_type; arr_memspace = Global});
        DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ];
    kern_body = body;
    kern_variants = [("Opt", opt_constrs); ("Choice", pick_constrs)];
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

(* SHAPES WITH NO LOWERING. #73's guard checked these explicitly (its
   [expr_mentions] covered [EArrayLen] and [EArrayRead] of a binder); replacing
   that guard with a real fix must not quietly drop the coverage. All three need
   a vector-typed payload or an effectful scrutinee, so they are unreachable
   from today's DSL — which is a reason to assert the refusal, not to assume it.
   Emitting for them would produce an undeclared identifier (the first two) or
   silently run an atomic once per re-emitted copy of the scrutinee (the
   third). *)
let array_len_of_binder_kernel =
  kernel_with
    ~vname:"Opt"
    ~constrs:opt_constrs
    ~elt:opt_type
    [
      (PConstr ("OptSome", ["y"]), ECast (TFloat32, EArrayLen "y"));
      (PConstr ("OptNone", []), EConst (CFloat32 0.0));
    ]

let array_read_of_binder_kernel =
  kernel_with
    ~vname:"Opt"
    ~constrs:opt_constrs
    ~elt:opt_type
    [
      (PConstr ("OptSome", ["y"]), EArrayRead ("y", EConst (CInt32 0l)));
      (PConstr ("OptNone", []), EConst (CFloat32 0.0));
    ]

(* An atomic in the scrutinee would be performed once per emitted copy: the tag
   test already re-emits it per case, and each substituted binder adds another.
   IR expressions are NOT pure in general — [EIntrinsic] covers the atomics — so
   this is refused rather than duplicated. *)
let atomic_scrutinee_kernel =
  let scrut = make_var "opt" (TVec opt_type) in
  let out = make_var "out" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  {
    default_kernel with
    kern_name = "ematch_atomic_scrutinee";
    kern_params =
      [
        DParam (scrut, Some {arr_elttype = opt_type; arr_memspace = Global});
        DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ];
    kern_body =
      SLet
        ( idx,
          EIntrinsic ([], "global_thread_id", []),
          SAssign
            ( LArrayElem ("out", EVar idx),
              EMatch
                ( EIntrinsic ([], "atomic_add", [EVar scrut; EVar idx]),
                  [
                    ( PConstr ("OptSome", ["y"]),
                      EBinop (Add, v "y", EConst (CFloat32 1.0)) );
                    (PConstr ("OptNone", []), EConst (CFloat32 0.0));
                  ] ) ) );
    kern_variants = [("Opt", opt_constrs)];
  }

(* The SAME multi-payload destructuring as a match STATEMENT. Both paths now
   read their accessor from one [payload_layout], so this pins the property the
   fix actually rests on: EMatch and SMatch cannot drift apart. Asserting only
   "EMatch matches SMatch" would be satisfied by both being wrong together —
   which is exactly what happened on WGSL, where both spelled a nested [_v._0]
   that the variant declaration never emitted — so each is also compared against
   a literal written out per backend. *)
let smatch_multi_kernel =
  let scrut = make_var "opt" (TVec pair_type) in
  let out = make_var "out" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  {
    default_kernel with
    kern_name = "smatch_multi_probe";
    kern_params =
      [
        DParam (scrut, Some {arr_elttype = pair_type; arr_memspace = Global});
        DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ];
    kern_body =
      SLet
        ( idx,
          EIntrinsic ([], "global_thread_id", []),
          SMatch
            ( EArrayRead ("opt", EVar idx),
              [
                ( PConstr ("MkPair", ["a"; "b"]),
                  SAssign
                    (LArrayElem ("out", EVar idx), EBinop (Add, v "a", v "b"))
                );
                ( PConstr ("MkOne", ["c"]),
                  SAssign (LArrayElem ("out", EVar idx), v "c") );
              ] ) );
    kern_variants = [("Pair", pair_constrs)];
  }

(* --- backends ------------------------------------------------------------ *)

(* The expected accessor spellings are written out LITERALLY here, per backend,
   rather than derived from Sarek_ir_codegen.payload_suffix — deriving them from
   the function under test would make this file agree with any spelling that
   function produces, including a wrong one. That is precisely how the first
   version of this test pinned WGSL's invalid `MkPair_v._0`: it applied the
   C-family/GLSL spelling uniformly to all five backends and stayed green while
   naga rejected the shader.

   The three layouts really do differ. WGSL flattens a multi-payload
   constructor into indexed SIBLING fields; the C family and GLSL nest. *)
type backend = {
  label : string;
  gen : kernel -> string;
  single : string -> string;  (** [cname] -> accessor for a 1-payload ctor *)
  multi : string -> int -> string;  (** [cname] -> [i] -> accessor *)
  validate : (string -> (unit, string) result) option;
      (** External shader validator, when one exists for this backend. *)
}

(* --- external validators -------------------------------------------------- *)

let read_file path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let s = really_input_string ic n in
  close_in ic ;
  s

let run_validator ~exe ~ext ~args src =
  let base = Filename.temp_file "sarek_payload_" "" in
  let file = base ^ ext in
  let err = base ^ ".err" in
  let oc = open_out file in
  output_string oc src ;
  close_out oc ;
  let cmd =
    Printf.sprintf
      "%s %s %s >%s 2>&1"
      exe
      args
      (Filename.quote file)
      (Filename.quote err)
  in
  let rc = Unix.system cmd in
  let out = read_file err in
  List.iter (fun f -> try Sys.remove f with _ -> ()) [file; err; base] ;
  match rc with Unix.WEXITED 0 -> Ok () | _ -> Error out

let tool_available exe =
  lazy
    (Unix.system (Printf.sprintf "command -v %s >/dev/null 2>&1" exe)
    = Unix.WEXITED 0)

let glslang_available = tool_available "glslangValidator"

let naga_available = tool_available "naga"

let glslang_ok src =
  run_validator
    ~exe:"glslangValidator"
    ~ext:".comp"
    ~args:"-V -S comp -o /dev/null"
    src

let naga_ok src = run_validator ~exe:"naga" ~ext:".wgsl" ~args:"" src

let c_family label gen =
  {
    label;
    gen;
    single = (fun c -> ".data." ^ c ^ "_v");
    multi = (fun c i -> Printf.sprintf ".data.%s_v._%d" c i);
    validate = None;
  }

let backends =
  [
    c_family "CUDA" (fun k -> Cuda.generate_with_types ~types:[] k);
    c_family "OpenCL" (fun k -> Opencl.generate_with_types ~types:[] k);
    c_family "Metal" (fun k -> Metal.generate_with_types ~types:[] k);
    {
      label = "GLSL";
      gen = (fun k -> Glsl.generate_with_types ~types:[] k);
      single = (fun c -> "." ^ c ^ "_v");
      multi = (fun c i -> Printf.sprintf ".%s_v._%d" c i);
      validate = Some glslang_ok;
    };
    {
      label = "WGSL";
      gen = (fun k -> Wgsl.generate_with_types ~types:[] k);
      single = (fun c -> "." ^ c ^ "_v");
      (* Sibling fields, NOT a nested struct — see the WGSL variant emitter. *)
      multi = (fun c i -> Printf.sprintf ".%s_v_%d" c i);
      validate = Some naga_ok;
    };
  ]

let available b =
  match b.label with
  | "GLSL" -> Lazy.force glslang_available
  | "WGSL" -> Lazy.force naga_available
  | _ -> false

(* --- assertions ---------------------------------------------------------- *)

let generate b k =
  try b.gen k
  with e ->
    Alcotest.failf
      "%s: generation raised instead of emitting the payload binding: %s"
      b.label
      (Printexc.to_string e)

(* THE ASSIGNMENT LINE, not the whole module, is what the string assertions
   inspect. The variant preamble each backend emits already contains the payload
   access path (the constructor function assigns [r.data.C_v = v]), so a
   whole-source search would be satisfied by the preamble alone and would stay
   green with the match lowering still broken. The kernel body here is a single
   [<out>[idx] = <match>;] line ([out] is spelled [outv] on GLSL, where [out] is
   reserved — hence matching on the index, not the name). *)
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

let expected_access b (cname, arity, i) =
  if arity <= 1 then b.single cname else b.multi cname i

let check_payload_bound b ~kernel ~binders ~accesses () =
  let line = assign_line b kernel in
  List.iter
    (fun spec ->
      let needle = expected_access b spec in
      Alcotest.(check bool)
        (Printf.sprintf
           "%s: the match reads the payload as %S — found: %s"
           b.label
           needle
           line)
        true
        (contains ~haystack:line ~needle))
    accesses ;
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

(* THE VALIDATOR GATE. String assertions cannot tell a valid accessor from an
   invalid one — the first version of this file happily pinned WGSL's
   `MkPair_v._0`, a field that does not exist, and naga rejects it with
   "invalid field accessor". Every kernel that binds a payload is therefore also
   handed to the real shader compiler. Reported as SKIP, never as a green OK,
   when the tool is absent: without it the check did not happen. *)
let check_validates b ~what kernel () =
  match b.validate with
  | None -> Alcotest.skip () (* no external validator for this backend *)
  | Some validate -> (
      let src = generate b kernel in
      if not (available b) then begin
        Printf.printf "  SKIP: no validator on PATH for %s\n%!" b.label ;
        Alcotest.skip ()
      end
      else
        match validate src with
        | Ok () -> ()
        | Error e ->
            Alcotest.failf
              "%s: the shader compiler rejected the generated %s payload \
               binding:\n\
               %s\n\
               --- shader ---\n\
               %s"
              b.label
              what
              e
              src)

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
  let needle = b.single "OptSome" in
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
    (contains ~haystack:line ~needle:(b.single "OptSome"))

(* The injected term must still index with `tid`. Capture shows up as the inner
   payload read appearing in INDEX position. *)
let check_no_capture b () =
  let line = assign_line b capture_kernel in
  Alcotest.(check bool)
    (Printf.sprintf
       "%s: the outer arm still indexes with the enclosing `idx`, not the \
        inner match's binder — found: %s"
       b.label
       line)
    true
    (contains ~haystack:line ~needle:"shp[idx]") ;
  Alcotest.(check bool)
    (Printf.sprintf
       "%s: the inner binder must NOT have been substituted into the outer \
        replacement term (capture) — found: %s"
       b.label
       line)
    false
    (contains ~haystack:line ~needle:(b.single "Pick" ^ "]"))

(* These must be REFUSED, loudly and with a located error — never emitted. *)
let check_refused b ~what kernel () =
  match b.gen kernel with
  | (_ : string) ->
      Alcotest.failf
        "%s: %s has no correct lowering, but generation succeeded — that emits \
         an undeclared identifier or a duplicated side effect"
        b.label
        what
  | exception
      Backend_error.Backend_error
        (Backend_error.Codegen
           {error = Backend_error.Unsupported_construct {construct; _}; _}) ->
      Alcotest.(check string)
        (Printf.sprintf "%s: names the construct" b.label)
        "match-expression payload binding"
        construct

(* The statement path must declare its payloads from the same accessor the
   expression path substitutes. *)
let check_smatch_agrees b () =
  let src = generate b smatch_multi_kernel in
  (* Only the arm DECLARATIONS, which are the lines that project out of the
     scrutinee [opt[idx]]. The constructor preamble also mentions every
     accessor ([r.MkPair_v_0 = v0;]), so searching the whole module here would
     be vacuous — and was: it stayed green under a mutation that changed the
     emitted accessor on both paths at once. *)
  let decls =
    List.filter
      (fun l -> contains ~haystack:l ~needle:"opt[idx].")
      (String.split_on_char '\n' src)
  in
  if decls = [] then
    Alcotest.failf
      "%s: no SMatch destructuring declarations in:\n%s"
      b.label
      src ;
  let decls = String.concat "\n" decls in
  List.iter
    (fun spec ->
      let needle = expected_access b spec in
      Alcotest.(check bool)
        (Printf.sprintf
           "%s: the SMatch declaration reads %S — the same accessor the EMatch \
            substitution uses, and the one the variant declaration emits — \
            found:\n\
            %s"
           b.label
           needle
           decls)
        true
        (contains ~haystack:decls ~needle))
    [("MkPair", 2, 0); ("MkPair", 2, 1); ("MkOne", 1, 0)]

(* WHY EVERY STRING ASSERTION IN THIS FILE IS SCOPED TO THE ASSIGNMENT LINE.
   This is not a style preference, it is the difference between a check and a
   no-op, so it is asserted rather than left as a comment: a kernel that binds
   NO payload at all still produces a module containing the payload accessor,
   because the generated constructor function assigns it
   ([r.data.OptSome_v = v;]). A whole-module search for that string is therefore
   satisfied by the preamble alone and holds no matter what the match arm
   actually reads — which is exactly how the first version of this work left a
   vacuous assertion in test_shader_recursion_vector.ml. *)
let check_whole_module_search_would_be_vacuous b () =
  let accessor = b.single "OptSome" in
  let src = generate b tag_only_kernel in
  Alcotest.(check bool)
    (Printf.sprintf
       "%s: the preamble alone contains %S, so a whole-module search is \
        vacuous — assertions must read the assignment line"
       b.label
       accessor)
    true
    (contains ~haystack:src ~needle:accessor) ;
  Alcotest.(check bool)
    (Printf.sprintf
       "%s: ...while the assignment line of that same payload-free kernel does \
        NOT contain it"
       b.label)
    false
    (contains ~haystack:(assign_line b tag_only_kernel) ~needle:accessor)

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
    (contains ~haystack:line ~needle:(b.single "OptSome"))

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
            ~accesses:[("OptSome", 1, 0)]);
      per_backend "multi-payload" (fun b ->
          check_payload_bound
            b
            ~kernel:multi_payload_kernel
            ~binders:["a"; "b"; "c"]
            ~accesses:[("MkPair", 2, 0); ("MkPair", 2, 1); ("MkOne", 1, 0)]);
      per_backend "single-case-shortcut" (fun b ->
          check_payload_bound
            b
            ~kernel:single_case_kernel
            ~binders:["y"]
            ~accesses:[("OptSome", 1, 0)]);
      per_backend "nested-shadowing" check_shadowing;
      per_backend "no-capture-of-injected-free-vars" check_no_capture;
      per_backend "smatch-uses-the-same-accessor" check_smatch_agrees;
      per_backend
        "whole-module-search-is-vacuous"
        check_whole_module_search_would_be_vacuous;
      per_backend "refuses-array-len-of-binder" (fun b ->
          check_refused
            b
            ~what:"EArrayLen of a payload binder"
            array_len_of_binder_kernel);
      per_backend "refuses-array-read-of-binder" (fun b ->
          check_refused
            b
            ~what:"EArrayRead of a payload binder"
            array_read_of_binder_kernel);
      per_backend "refuses-atomic-scrutinee" (fun b ->
          check_refused
            b
            ~what:"an atomic in the match scrutinee"
            atomic_scrutinee_kernel);
      per_backend "validates-single-payload" (fun b ->
          check_validates b ~what:"single-payload" single_payload_kernel);
      per_backend "validates-multi-payload" (fun b ->
          check_validates b ~what:"multi-payload" multi_payload_kernel);
      per_backend "validates-no-capture" (fun b ->
          check_validates b ~what:"nested/capture" capture_kernel);
      per_backend "silent-wrong-shadowed-binder" (fun b ->
          check_payload_bound
            b
            ~kernel:silent_wrong_kernel
            ~binders:["r"]
            ~accesses:[("OptSome", 1, 0)]);
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
