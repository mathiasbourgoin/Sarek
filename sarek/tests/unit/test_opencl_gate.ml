(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Negative controls for the OpenCL validation gate (#127/#128).

    The sweep in test_codegen_golden.ml is the gate; this file is the proof that
    the gate can go red. Every case here mutates an input and requires the
    corresponding layer to fail with the right diagnostic, because a check that
    has never been observed failing is not evidence of anything.

    Also pins the #127 backend policy: budgeted self-recursion is bounded, every
    other cycle is refused, and neither outcome is "emit a self-call and hope".
*)

open Sarek_ir_types
module Ocl = Sarek_codegen.Sarek_ir_opencl
module Recursion = Opencl_gate.Opencl_recursion
module Clang = Opencl_gate.Opencl_clang
module Uniquify = Opencl_gate.Ir_uniquify

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let empty_kernel name params locals body =
  {
    kern_name = name;
    kern_params = params;
    kern_locals = locals;
    kern_body = body;
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

(** A kernel calling helper [pow2], whose body is [hf_body]. *)
let recursive_helper_kernel ~budgeted =
  let n = make_var "n" TInt32 in
  let self =
    EApp
      (EVar (make_var "pow2" TInt32), [EBinop (Sub, EVar n, EConst (CInt32 1l))])
  in
  let core =
    SReturn
      (EIf
         ( EBinop (Le, EVar n, EConst (CInt32 0l)),
           EConst (CInt32 1l),
           EBinop (Mul, EConst (CInt32 2l), self) ))
  in
  let hf =
    {
      hf_name = "pow2";
      hf_params = [n];
      hf_ret_type = TInt32;
      hf_body = (if budgeted then SPragma (["sarek.inline 0"], core) else core);
    }
  in
  let out = make_var "out" (TVec TInt32) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("out", EVar idx),
            EApp (EVar (make_var "pow2" TInt32), [EConst (CInt32 5l)]) ) )
  in
  let k =
    empty_kernel
      "rec_kernel"
      [DParam (out, Some {arr_elttype = TInt32; arr_memspace = Global})]
      []
      body
  in
  {k with kern_funcs = [hf]}

(** Two INDEPENDENTLY self-recursive helpers, both budgeted, where [f] also
    calls [g]. Their cycles never touch, so this is not mutual recursion and
    both are bounded by their own pragma.

    The distinction is load-bearing: this backend elides budgeted self-recursion
    but REFUSES everything else, so an over-approximating "is the callee on some
    cycle?" test turns this into a false refusal — a regression for anyone using
    the pragma. *)
let nested_selfrec_kernel () =
  let n = make_var "n" TInt32 in
  let dec v = EBinop (Sub, EVar v, EConst (CInt32 1l)) in
  let selfrec name extra =
    let self = EApp (EVar (make_var name TInt32), [dec n]) in
    let rhs =
      match extra with
      | None -> self
      | Some other ->
          EBinop (Add, self, EApp (EVar (make_var other TInt32), [dec n]))
    in
    {
      hf_name = name;
      hf_params = [n];
      hf_ret_type = TInt32;
      hf_body =
        SPragma
          ( ["sarek.inline 0"],
            SReturn
              (EIf
                 ( EBinop (Le, EVar n, EConst (CInt32 0l)),
                   EConst (CInt32 1l),
                   rhs )) );
    }
  in
  (* g is self-recursive only; f is self-recursive AND calls g. *)
  let g = selfrec "g" None in
  let f = selfrec "f" (Some "g") in
  let out = make_var "out" (TVec TInt32) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("out", EVar idx),
            EApp (EVar (make_var "f" TInt32), [EConst (CInt32 4l)]) ) )
  in
  let k =
    empty_kernel
      "nested_selfrec"
      [DParam (out, Some {arr_elttype = TInt32; arr_memspace = Global})]
      []
      body
  in
  {k with kern_funcs = [f; g]}

(** Two helpers that call each other — no pragma can bound this, because the PPX
    inliner only ever rewrites SELF-calls. *)
let mutual_kernel () =
  let x = make_var "x" TInt32 in
  let call name = EApp (EVar (make_var name TInt32), [EVar x]) in
  let f =
    {
      hf_name = "f";
      hf_params = [x];
      hf_ret_type = TInt32;
      hf_body = SPragma (["sarek.inline 0"], SReturn (call "g"));
    }
  in
  let g =
    {
      hf_name = "g";
      hf_params = [x];
      hf_ret_type = TInt32;
      hf_body = SPragma (["sarek.inline 0"], SReturn (call "f"));
    }
  in
  let out = make_var "out" (TVec TInt32) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("out", EVar idx),
            EApp (EVar (make_var "f" TInt32), [EConst (CInt32 3l)]) ) )
  in
  let k =
    empty_kernel
      "mutual_kernel"
      [DParam (out, Some {arr_elttype = TInt32; arr_memspace = Global})]
      []
      body
  in
  {k with kern_funcs = [f; g]}

let plain_kernel () =
  let a = make_var "a" (TVec TFloat32) in
  let b = make_var "b" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("b", EVar idx),
            EBinop (Mul, EArrayRead ("a", EVar idx), EConst (CFloat32 2.0)) ) )
  in
  empty_kernel
    "plain"
    [
      DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
      DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
    ]
    []
    body

let contains hay needle =
  let n = String.length needle and h = String.length hay in
  let rec go i = i + n <= h && (String.sub hay i n = needle || go (i + 1)) in
  n = 0 || go 0

(* ------------------------------------------------------------------ *)
(* Layer 1 — recursion detector                                        *)
(* ------------------------------------------------------------------ *)

(** RED: a self-recursive device function must be reported. This is the case no
    compiler on this path reports — see the header of opencl_recursion.ml. *)
let test_recursion_self () =
  let src =
    "int pow2(int n) { if (n <= 0) return 1; return 2 * pow2(n - 1); }\n\
     __kernel void k(__global int* o) { o[get_global_id(0)] = pow2(3); }\n"
  in
  match Recursion.cycles src with
  | [c] ->
      Alcotest.(check string)
        "cycle names the function"
        "'pow2' calls itself"
        (Recursion.describe c)
  | cs -> Alcotest.failf "expected exactly one cycle, got %d" (List.length cs)

(** RED: mutual recursion too — a self-call check alone would miss it. *)
let test_recursion_mutual () =
  let src =
    "int g(int n);\n\
     int f(int n) { return g(n - 1); }\n\
     int g(int n) { return f(n - 1); }\n\
     __kernel void k(__global int* o) { o[get_global_id(0)] = f(3); }\n"
  in
  match Recursion.cycles src with
  | [] -> Alcotest.fail "mutual recursion not detected"
  | c :: _ ->
      let d = Recursion.describe c in
      Alcotest.(check bool)
        (Printf.sprintf "cycle mentions both helpers (%s)" d)
        true
        (contains d "'f'" && contains d "'g'")

(** GREEN control: real generator output must be reported clean, so the layer is
    not simply always-red. *)
let test_recursion_clean () =
  Ocl.current_variants := [] ;
  let src = Ocl.generate_with_types ~types:[] (plain_kernel ()) in
  Alcotest.(check int)
    "no cycles in real output"
    0
    (List.length (Recursion.cycles src)) ;
  (* And a struct initialiser / typedef must not be mistaken for a definition. *)
  let src2 =
    "typedef struct { int tag; } Opt;\n\
     static inline Opt make_Opt(int v) { Opt r; r.tag = v; return r; }\n\
     __kernel void k(__global int* o) { o[0] = make_Opt(1).tag; }\n"
  in
  Alcotest.(check int)
    "no false positive on constructors"
    0
    (List.length (Recursion.cycles src2))

(* ------------------------------------------------------------------ *)
(* Layer 2 — clang compile gate                                        *)
(* ------------------------------------------------------------------ *)

let skip_without_clang () =
  if not (Clang.available ()) then begin
    Printf.printf "  SKIP: %s\n%!" (Clang.why_unavailable ()) ;
    Alcotest.skip ()
  end

(** RED-ON-MUTATION: delete the declaration of [idx] from real generator output
    and require clang to name it. This is the exact diagnostic the gate relies
    on for the whole dropped-binder class. *)
let test_clang_red_on_mutation () =
  skip_without_clang () ;
  Ocl.current_variants := [] ;
  let src = Ocl.generate_with_types ~types:[] (plain_kernel ()) in
  (match Clang.run_clang src with
  | Ok () -> ()
  | Error e -> Alcotest.failf "unmutated source must compile:\n%s\n%s" e src) ;
  let lines = String.split_on_char '\n' src in
  let mutated =
    String.concat
      "\n"
      (List.filter (fun l -> not (contains l "int idx = ")) lines)
  in
  Alcotest.(check bool)
    "mutation actually changed the source"
    true
    (mutated <> src) ;
  match Clang.run_clang mutated with
  | Ok () ->
      Alcotest.failf
        "clang accepted a source with the declaration of 'idx' removed — the \
         compile layer is not checking anything:\n\
         %s"
        mutated
  | Error e ->
      Alcotest.(check bool)
        (Printf.sprintf
           "diagnostic names the undeclared binder (%s)"
           (String.trim e))
        true
        (contains e "undeclared identifier" && contains e "idx")

(* ------------------------------------------------------------------ *)
(* Layer 3 — binder canary                                             *)
(* ------------------------------------------------------------------ *)

(** The canary's whole premise: after α-conversion, two binders that shared a
    name no longer can. Checked on the IR AND on the emitted source, because it
    is the emitted source the compiler sees. *)
let test_canary_makes_names_unique () =
  let outer = make_var "r" TFloat32 in
  let inner = make_var "r" TFloat32 in
  let out = make_var "out" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( outer,
            EConst (CFloat32 1.0),
            SBlock
              (SLet
                 ( inner,
                   EConst (CFloat32 2.0),
                   SAssign (LArrayElem ("out", EVar idx), EVar inner) )) ) )
  in
  let k =
    empty_kernel
      "shadow"
      [DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      []
      body
  in
  Ocl.current_variants := [] ;
  let plain = Ocl.generate_with_types ~types:[] k in
  (* Pre-condition: as written, the same name really is declared twice — this is
     what makes a dropped binder invisible. *)
  let count_decl s =
    List.length
      (List.filter
         (fun l -> contains l "float r = ")
         (String.split_on_char '\n' s))
  in
  Alcotest.(check int) "as written, 'r' is declared twice" 2 (count_decl plain) ;
  let ku = Uniquify.uniquify_kernel k in
  let canary = Ocl.generate_with_types ~types:[] ku in
  Alcotest.(check int)
    "after α-conversion, no bare 'r' declaration remains"
    0
    (count_decl canary) ;
  Alcotest.(check bool)
    "α-converted source still compiles (renaming is sound)"
    true
    (if Clang.available () then Clang.run_clang canary = Ok () else true)

(** α-conversion must not change what the kernel computes; if it did, a canary
    failure would be uninformative. Structural proxy: the two sources are
    identical once binder names are erased to a canonical token. *)
let test_canary_is_semantics_preserving () =
  Ocl.current_variants := [] ;
  let k = plain_kernel () in
  let plain = Ocl.generate_with_types ~types:[] k in
  let canary = Ocl.generate_with_types ~types:[] (Uniquify.uniquify_kernel k) in
  Alcotest.(check bool) "α-conversion changed the text" true (plain <> canary) ;
  let erase s = Str.global_replace (Str.regexp "sk[0-9]+_") "" s in
  Alcotest.(check string)
    "identical modulo the sk<n>_ prefix"
    plain
    (erase canary)

(* ------------------------------------------------------------------ *)
(* #127 — backend recursion policy                                     *)
(* ------------------------------------------------------------------ *)

(** A budgeted self-recursive helper must generate, and the generated source
    must contain no self-call at all. Emitting a residual call is the outcome
    that crashed rusticl; a blanket refusal would have regressed the pragma. *)
let test_budgeted_recursion_is_bounded () =
  Ocl.current_variants := [] ;
  let src =
    Ocl.generate_with_types ~types:[] (recursive_helper_kernel ~budgeted:true)
  in
  Alcotest.(check int)
    "generated source has no call cycle"
    0
    (List.length (Recursion.cycles src)) ;
  Alcotest.(check bool)
    "residual call replaced by a typed zero"
    true
    (contains src "2 * 0" || contains src "(2 * 0)") ;
  if Clang.available () then
    match Clang.run_clang src with
    | Ok () -> ()
    | Error e -> Alcotest.failf "clang rejected the bounded form:\n%s\n%s" e src

(** RED: no pragma, no bound — refuse loudly rather than emit recursion. *)
let test_unbudgeted_recursion_refused () =
  Ocl.current_variants := [] ;
  match
    Ocl.generate_with_types ~types:[] (recursive_helper_kernel ~budgeted:false)
  with
  | src ->
      Alcotest.failf
        "expected a refusal for an unbounded recursive helper, got:\n%s"
        src
  | exception e ->
      let m = Printexc.to_string e in
      Alcotest.(check bool)
        (Printf.sprintf "error names recursion and the helper (%s)" m)
        true
        (contains m "recursion" && contains m "pow2")

(** A self-recursive callee must NOT make its caller "mutually recursive".

    Pins the classifier against over-approximation: [f] and [g] are each
    budgeted and self-recursive, and [f] calls [g], but the two cycles are
    independent. Both must be bounded, not refused. An "is the callee on some
    cycle?" test refuses [f] here with "mutually recursive with 'g'", which is
    false and costs a pragma user a working kernel — the failure direction that
    matters, since a compiler refusing valid input is a regression. *)
let test_independent_selfrec_not_mutual () =
  Ocl.current_variants := [] ;
  match Ocl.generate_with_types ~types:[] (nested_selfrec_kernel ()) with
  | exception e ->
      Alcotest.failf
        "independently self-recursive helpers must both be bounded, not \
         refused — got: %s"
        (Printexc.to_string e)
  | src -> (
      Alcotest.(check int)
        "no call cycle survives"
        0
        (List.length (Recursion.cycles src)) ;
      (* f must still really call g: the elision may only remove SELF-calls, and
         a pass that "fixed" this by deleting the cross-call would also pass the
         cycle check above while silently changing what the kernel computes. *)
      Alcotest.(check bool)
        "f still calls g (only self-calls were elided)"
        true
        (contains src "g(") ;
      (* [kern_funcs] has no ordering guarantee, and here the caller [f] comes
         first. Without a forward declaration that is `error: use of undeclared
         identifier 'g'` — invalid OpenCL C decided purely by list order. Pins
         the prototype block emitted by Sarek_ir_opencl.gen_helpers. *)
      Alcotest.(check bool)
        "helpers are forward-declared, so emission order cannot matter"
        true
        (contains src "int g(int n);" && contains src "int f(int n);") ;
      if Clang.available () then
        match Clang.run_clang src with
        | Ok () -> ()
        | Error e -> Alcotest.failf "clang rejected:\n%s\n%s" e src)

(** RED: mutual recursion is refused even though both helpers carry a pragma —
    the pragma cannot bound a cycle the inliner never rewrites. *)
let test_mutual_recursion_refused () =
  Ocl.current_variants := [] ;
  match Ocl.generate_with_types ~types:[] (mutual_kernel ()) with
  | src ->
      Alcotest.failf "expected a refusal for mutual recursion, got:\n%s" src
  | exception e ->
      let m = Printexc.to_string e in
      Alcotest.(check bool)
        (Printf.sprintf "error says mutually recursive (%s)" m)
        true
        (contains m "mutually recursive")

(* fp64 capability predicate (#140).

   The sweep asks [Clang.fp64_available ()] before deciding what to do with a
   float64 kernel, and on this machine the answer is normally "yes" — so the
   interesting branch is the one that never runs here. That is the branch that
   split the M4 into two verdicts, so it is the one that has to be driven.

   [SAREK_OPENCL_GATE_NO_FP64=1] removes cl_khr_fp64 from the compiler itself
   ([-cl-ext=-cl_khr_fp64]), producing the same diagnostic Apple clang gives.
   This case re-executes the test binary with that variable set and requires the
   predicate to flip — a self-check on the check, because a suppression switch
   that quietly does nothing would make every "SKIP (no fp64)" a lie. *)
let subprocess_reports_no_fp64 () =
  let exe = Sys.executable_name in
  let out = Filename.temp_file "sarek_fp64_probe_" ".txt" in
  let rc =
    Unix.system
      (Printf.sprintf
         "SAREK_OPENCL_GATE_NO_FP64=1 %s --fp64-report >%s 2>&1"
         (Filename.quote exe)
         (Filename.quote out))
  in
  let text =
    try
      let ic = open_in out in
      let n = in_channel_length ic in
      let s = really_input_string ic n in
      close_in ic ;
      s
    with _ -> ""
  in
  (try Sys.remove out with _ -> ()) ;
  match rc with Unix.WEXITED 0 -> Some (String.trim text) | _ -> None

let test_fp64_predicate_goes_red_under_suppression () =
  if not (Clang.available ()) then
    Printf.printf "  SKIP: %s\n%!" (Clang.why_unavailable ())
  else begin
    (* Positive control first: without suppression this clang HAS fp64, so the
       negative result below is attributable to the switch and not to a
       toolchain that never had it. *)
    Alcotest.(check bool)
      "unsuppressed: this clang compiles a double kernel"
      true
      (Clang.fp64_available ()) ;
    match subprocess_reports_no_fp64 () with
    | None ->
        Alcotest.fail "could not re-run this executable with the switch set"
    | Some report ->
        if report = "" then
          Alcotest.fail
            "SAREK_OPENCL_GATE_NO_FP64=1 left fp64 AVAILABLE. The suppression \
             switch does nothing, so every \"SKIP (no fp64)\" the sweep prints \
             is unverifiable." ;
        Alcotest.(check bool)
          "the stated reason names cl_khr_fp64"
          true
          (try
             ignore
               (Str.search_forward (Str.regexp_string "cl_khr_fp64") report 0) ;
             true
           with Not_found -> false)
  end

(* Sub-mode used by the case above. Prints the reason fp64 is unavailable (empty
   line if it IS available) and exits, so the parent can read the predicate out
   of a fresh process with a different environment. *)
let () =
  if Array.length Sys.argv > 1 && Sys.argv.(1) = "--fp64-report" then begin
    print_string (Clang.why_no_fp64 ()) ;
    exit 0
  end

let () =
  Alcotest.run
    "opencl_gate"
    [
      ( "layer1_recursion",
        [
          Alcotest.test_case "self-recursion detected" `Quick test_recursion_self;
          Alcotest.test_case
            "mutual recursion detected"
            `Quick
            test_recursion_mutual;
          Alcotest.test_case "no false positives" `Quick test_recursion_clean;
        ] );
      ( "layer2_clang",
        [
          Alcotest.test_case
            "red on a removed declaration"
            `Quick
            test_clang_red_on_mutation;
          Alcotest.test_case
            "the fp64 predicate goes red under suppression"
            `Quick
            test_fp64_predicate_goes_red_under_suppression;
        ] );
      ( "layer3_binder_canary",
        [
          Alcotest.test_case
            "α-conversion removes name collisions"
            `Quick
            test_canary_makes_names_unique;
          Alcotest.test_case
            "α-conversion is semantics-preserving"
            `Quick
            test_canary_is_semantics_preserving;
        ] );
      ( "backend_recursion_policy",
        [
          Alcotest.test_case
            "budgeted self-recursion is fully bounded"
            `Quick
            test_budgeted_recursion_is_bounded;
          Alcotest.test_case
            "unbounded recursion is refused"
            `Quick
            test_unbudgeted_recursion_refused;
          Alcotest.test_case
            "mutual recursion is refused"
            `Quick
            test_mutual_recursion_refused;
          Alcotest.test_case
            "an independently self-recursive callee is not mutual recursion"
            `Quick
            test_independent_selfrec_not_mutual;
        ] );
    ]
