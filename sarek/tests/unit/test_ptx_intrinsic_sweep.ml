(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** ptxas gate over the WHOLE PTX intrinsic surface.

    The hand-picked ptxas gate in test_ptx_snapshot.ml assembles five kernels;
    the intrinsic surface is 80 names. That gap let an invalid opcode ship: the
    f32→f64 widening of [Float32.{asin,acos,atan,atan2,expm1,log1p}] emitted
    [cvt.rn.f64.f32], which PTX forbids on an exact widening — the kernels
    generated fine, passed a substring snapshot test that asserted the invalid
    opcode, and died at [cuModuleLoadData].

    So this gate is driven by the emitter's own dispatch registry
    ({!Sarek_ir_ptx_expr.intrinsic_registry}) rather than a hand-picked list: it
    builds and assembles at least one kernel per intrinsic NAME, and an
    intrinsic added to the emitter with no kernel recipe here FAILS the recipe
    test. Kernel GENERATION always runs; only the [ptxas] assembly step
    self-skips (with a printed reason saying what is still checked) when the
    tool is absent, so the suite stays green — and still meaningful — off-CUDA
    machines. *)

open Sarek_ir_types
open Sarek_codegen

(* ---- ptxas plumbing (mirrors test_ptx_snapshot.ml's gate) ---------------- *)

let ptxas_available =
  lazy
    (match Unix.system "command -v ptxas >/dev/null 2>&1" with
    | Unix.WEXITED 0 -> true
    | _ -> false)

let assemble_ok ptx =
  let base = Filename.temp_file "sarek_sweep_" "" in
  let src = base ^ ".ptx" in
  let obj = base ^ ".cubin" in
  let oc = open_out src in
  output_string oc ptx ;
  close_out oc ;
  (* ptxas assumes a low default SM and rejects any PTX whose [.target] is
     higher, so pass the module's own target explicitly. *)
  let gpu_name =
    let target_re = Str.regexp "\\.target[ \t]+\\(sm_[0-9]+\\)" in
    try
      ignore (Str.search_forward target_re ptx 0) ;
      Str.matched_group 1 ptx
    with Not_found -> "sm_86"
  in
  let cmd =
    Printf.sprintf
      "ptxas --compile-only --gpu-name %s -o %s %s 2>%s.err"
      (Filename.quote gpu_name)
      (Filename.quote obj)
      (Filename.quote src)
      (Filename.quote base)
  in
  let rc = Unix.system cmd in
  let err =
    try
      let ic = open_in (base ^ ".err") in
      let n = in_channel_length ic in
      let s = really_input_string ic n in
      close_in ic ;
      s
    with _ -> ""
  in
  List.iter
    (fun f -> try Sys.remove f with _ -> ())
    [src; obj; base; base ^ ".err"] ;
  match rc with Unix.WEXITED 0 -> Ok () | _ -> Error err

(* ---- kernel fixtures ----------------------------------------------------- *)

let mk name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let param n ty =
  DParam (mk n (TVec ty), Some {arr_elttype = ty; arr_memspace = Global})

(* One in and one out array per scalar width: every recipe below writes its
   result to the out array of the matching width. *)
let params =
  [
    param "of32" TFloat32;
    param "of64" TFloat64;
    param "oi32" TInt32;
    param "oi64" TInt64;
    param "af32" TFloat32;
    param "af64" TFloat64;
    param "ai32" TInt32;
    param "ai64" TInt64;
  ]

let tid = mk "tid" TInt32

let rd a = EArrayRead (a, EVar tid)

let arr a ty = EVar (mk a (TVec ty))

let kernel label out e =
  {
    kern_name = "k_" ^ label;
    kern_params = params;
    kern_locals = [];
    kern_body =
      SLet
        ( tid,
          EIntrinsic ([], "global_thread_id", []),
          SAssign (LArrayElem (out, EVar tid), e) );
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

let f32_path = ["Sarek_stdlib"; "Float32"]

let f64_path = ["Sarek_stdlib"; "Float64"]

let stdlib_path = ["Sarek_stdlib"]

(* Arity of every math intrinsic (transcendentals + elementary float ops). An
   entry is REQUIRED: a name added to the emitter's tables with no arity here
   has no recipe and fails [test_every_name_has_a_recipe]. *)
let math_arity =
  [
    (* transcendentals *)
    ("sin", 1);
    ("cos", 1);
    ("tan", 1);
    ("sqrt", 1);
    ("exp", 1);
    ("log", 1);
    ("log10", 1);
    ("pow", 2);
    ("sinh", 1);
    ("cosh", 1);
    ("tanh", 1);
    ("asin", 1);
    ("acos", 1);
    ("atan", 1);
    ("atan2", 2);
    ("expm1", 1);
    ("log1p", 1);
    ("rsqrt", 1);
    (* elementary float ops *)
    ("fabs", 1);
    ("abs_float", 1);
    ("copysign", 2);
    ("fmod", 2);
    ("hypot", 2);
    ("fma", 3);
    ("min", 2);
    ("max", 2);
    ("floor", 1);
    ("ceil", 1);
  ]

(* Conversions are per-name (source and destination widths differ), and
   "of_int"/"to_int" exist in both the Float32 and the Float64 module. Each
   entry is (label, module path, argument array, out array). *)
let convert_recipes =
  [
    ("float", [("float", stdlib_path, "ai32", "of32")]);
    ("float_of_int", [("float_of_int", stdlib_path, "ai32", "of32")]);
    ("float64", [("float64", stdlib_path, "ai32", "of64")]);
    ("float64_of_int", [("float64_of_int", stdlib_path, "ai32", "of64")]);
    ("int_of_float", [("int_of_float", stdlib_path, "af32", "oi32")]);
    ("int_of_float64", [("int_of_float64", stdlib_path, "af64", "oi32")]);
    ( "of_int",
      [
        ("of_int_f32", f32_path, "ai32", "of32");
        ("of_int_f64", f64_path, "ai32", "of64");
      ] );
    ( "to_int",
      [
        ("to_int_f32", f32_path, "af32", "oi32");
        ("to_int_f64", f64_path, "af64", "oi32");
      ] );
  ]

(* Atomics: (argument shape, element type). The array operand comes first, then
   the index, then the value(s) — [atomic_inc/dec] take no value. *)
type atomic_shape = Rmw | Cas | Incdec

let starts_with p s =
  String.length s >= String.length p && String.sub s 0 (String.length p) = p

let ends_with suf s =
  let ls = String.length s and lf = String.length suf in
  ls >= lf && String.sub s (ls - lf) lf = suf

let atomic_recipe name =
  let ty, a, out =
    if ends_with "int32" name then (TInt32, "ai32", "oi32")
    else if ends_with "int64" name then (TInt64, "ai64", "oi64")
    else if ends_with "float32" name then (TFloat32, "af32", "of32")
    else (TFloat64, "af64", "of64")
  in
  let shape =
    if starts_with "atomic_cas" name then Cas
    else if starts_with "atomic_inc" name || starts_with "atomic_dec" name then
      Incdec
    else Rmw
  in
  let args =
    match shape with
    | Rmw -> [arr a ty; EVar tid; rd a]
    | Cas -> [arr a ty; EVar tid; rd a; rd a]
    | Incdec -> [arr a ty; EVar tid]
  in
  [(name, kernel name out (EIntrinsic (stdlib_path, name, args)))]

(** [cases_for name] is every (label, kernel) pair exercising intrinsic [name],
    or [None] when the name has no recipe (a drift alarm, not a skip). *)
let cases_for name =
  if List.mem name Sarek_ir_ptx_expr.index_intrinsic_names then
    Some [(name, kernel name "oi32" (EIntrinsic ([], name, [])))]
  else if List.mem name Sarek_ir_ptx_expr.bitcast_intrinsic_names then
    match name with
    | "f64_bits" ->
        Some
          [
            ( "f64_bits",
              kernel
                "f64_bits"
                "oi64"
                (EIntrinsic (f64_path, name, [rd "af64"])) );
          ]
    | "bits_f64" ->
        Some
          [
            ( "bits_f64",
              kernel
                "bits_f64"
                "of64"
                (EIntrinsic (f64_path, name, [rd "ai64"])) );
          ]
    | _ -> None
  else if List.mem name Sarek_ir_ptx_expr.convert_intrinsic_names then
    match List.assoc_opt name convert_recipes with
    | None -> None
    | Some rs ->
        Some
          (List.map
             (fun (label, path, a, out) ->
               (label, kernel label out (EIntrinsic (path, name, [rd a]))))
             rs)
  else if List.mem name Sarek_ir_ptx_expr.atomic_intrinsic_names then
    Some (atomic_recipe name)
  else if
    List.mem name Sarek_ir_ptx_expr.transcendental_intrinsic_names
    || List.mem name Sarek_ir_ptx_expr.float_ops_intrinsic_names
  then
    match List.assoc_opt name math_arity with
    | None -> None
    | Some n ->
        (* Both widths: the f32 path (where the widen-to-f64 transcendentals
           live) and the f64 path (native f64 ops or softmath helpers). *)
        let width (label, path, a, out) =
          ( label,
            kernel
              label
              out
              (EIntrinsic (path, name, List.init n (fun _ -> rd a))) )
        in
        let widths =
          [
            (name ^ "_f32", f32_path, "af32", "of32");
            (name ^ "_f64", f64_path, "af64", "of64");
          ]
          (* min/max are also the integer min/max intrinsics. *)
          @
          if name = "min" || name = "max" then
            [
              (name ^ "_i32", stdlib_path, "ai32", "oi32");
              (name ^ "_i64", stdlib_path, "ai64", "oi64");
            ]
          else []
        in
        Some (List.map width widths)
  else None

(* ---- tests --------------------------------------------------------------- *)

let named_sets =
  [
    ("index", Sarek_ir_ptx_expr.index_intrinsic_names);
    ("transcendental", Sarek_ir_ptx_expr.transcendental_intrinsic_names);
    ("float_ops", Sarek_ir_ptx_expr.float_ops_intrinsic_names);
    ("bitcast", Sarek_ir_ptx_expr.bitcast_intrinsic_names);
    ("convert", Sarek_ir_ptx_expr.convert_intrinsic_names);
    ("atomic", Sarek_ir_ptx_expr.atomic_intrinsic_names);
  ]

(** The six emitter name sets must PARTITION the intrinsic surface. Dispatch
    picks an emitter by name, so a name claimed twice is ambiguous: before the
    dispatch table it was silently won by whichever emitter came first in the
    candidate list (no warning, no failure, wrong lowering). *)
let test_name_sets_disjoint () =
  let rec pairs = function
    | [] -> []
    | x :: rest -> List.map (fun y -> (x, y)) rest @ pairs rest
  in
  List.iter
    (fun ((na, a), (nb, b)) ->
      let common = List.filter (fun n -> List.mem n b) a in
      if common <> [] then
        Alcotest.fail
          (Printf.sprintf
             "intrinsic name(s) [%s] are claimed by both the %s and the %s \
              emitter — dispatch would be ambiguous"
             (String.concat "; " common)
             na
             nb))
    (pairs named_sets) ;
  (* No name may repeat inside one set either. *)
  List.iter
    (fun (n, names) ->
      let sorted = List.sort compare names in
      let rec dups = function
        | a :: (b :: _ as rest) -> if a = b then a :: dups rest else dups rest
        | _ -> []
      in
      match dups sorted with
      | [] -> ()
      | ds ->
          Alcotest.fail
            (Printf.sprintf
               "%s emitter lists [%s] more than once"
               n
               (String.concat "; " ds)))
    named_sets

(* ---- why there is no "tables match the registry" test --------------------

   There used to be one, and before that a worse one: the exported name tables
   and the dispatch arms were two sources for one fact, hundreds of lines apart,
   compared by SCANNING THE EMITTER SOURCE for [match] arms — a textual check
   that passes whenever the text moves, and fails when it moves harmlessly.

   Replacing the scan with a value-level comparison against the handler registry
   fixed the textual coupling but kept two lists. The tables are now COMPUTED
   from the handlers ([Sarek_ir_ptx_expr.names_of_category]), so "the tables
   agree with dispatch" is true by construction and there is nothing left to
   compare: such a test would assert [x = x] and could never fail. A test that
   cannot fail is worse than no test, so it is gone rather than kept for
   appearances.

   What derivation does NOT give is that the registries PARTITION the surface —
   two handlers, in the same category or different ones, can still claim the
   same name, and dispatch raises an internal error on it. That property has
   real content and is checked by [test_name_sets_disjoint] above. The sweep
   below, which lowers a kernel per registered name, is what proves each entry's
   handler actually claims and emits it. *)

(** Every dispatched intrinsic name has an assembling kernel recipe above. This
    is the anti-drift check: add an intrinsic to the emitter and this fails
    until the sweep covers it. *)
let test_every_name_has_a_recipe () =
  let missing =
    List.filter
      (fun n -> match cases_for n with None | Some [] -> true | _ -> false)
      Sarek_ir_ptx_expr.intrinsic_names
  in
  if missing <> [] then
    Alcotest.fail
      (Printf.sprintf
         "no ptxas-sweep kernel recipe for intrinsic(s) [%s] — add one to \
          test_ptx_intrinsic_sweep.ml so the new intrinsic is assembled"
         (String.concat "; " missing))

(* All (intrinsic name, label, kernel) triples, in dispatch order. *)
let all_cases () =
  List.concat_map
    (fun n ->
      match cases_for n with
      | None -> []
      | Some cs -> List.map (fun (label, k) -> (n, label, k)) cs)
    Sarek_ir_ptx_expr.intrinsic_names

(* Opcode forms known to be invalid PTX, checked on the generated text so a
   CPU-only run catches this class too — without a CUDA toolkit the assembler
   cannot say so, and this gate exists precisely because such a form shipped.
   Keep it to forms proven illegal against ptxas, one line per form. *)
let illegal_opcodes =
  [
    ( "cvt.rn.f64.f32",
      "a rounding modifier on the EXACT f32->f64 widening (ptxas: Illegal \
       rounding modifier for instruction 'cvt'); emit cvt.f64.f32" );
  ]

let contains haystack needle =
  match Str.search_forward (Str.regexp_string needle) haystack 0 with
  | _ -> true
  | exception Not_found -> false

(** Generate — and, where [ptxas] exists, assemble — one kernel per case,
    reporting per-name pass/fail so a failure names the culprit intrinsic. The
    whole sweep (114 kernels) costs ~1.3s of ptxas, so it stays on the default
    [runtest] alias.

    GENERATION ALWAYS RUNS, including on CPU-only machines: it is what proves
    every registered name is actually claimed by an emitter, that the emitter
    does not decline it, and that the recipe's arguments are accepted. Only the
    [ptxas] assembly step — the extra layer that catches invalid opcodes such as
    the [cvt.rn.f64.f32] this gate was built for — is skipped when the tool is
    absent. *)
let sweep () =
  let have_ptxas = Lazy.force ptxas_available in
  if not have_ptxas then
    Printf.printf
      "  NOTE: ptxas not on PATH — assembly is skipped, but every intrinsic \
       kernel is still enumerated and GENERATED (an unclaimed name, a \
       declining emitter or any codegen exception still fails this test)\n\
       %!" ;
  let cases = all_cases () in
  let failures = ref [] in
  let generated = ref 0 in
  let assembled = ref 0 in
  List.iter
    (fun (name, label, k) ->
      match
        try Ok (Sarek_ir_ptx.generate k)
        with e -> Error ("codegen raised " ^ Printexc.to_string e)
      with
      | Error e ->
          failures := (name, label, e) :: !failures ;
          Printf.printf "  FAIL %-28s %s\n%!" label e
      | Ok ptx -> (
          incr generated ;
          List.iter
            (fun (op, why) ->
              if contains ptx op then begin
                let e = Printf.sprintf "emits illegal %s — %s" op why in
                failures := (name, label, e) :: !failures ;
                Printf.printf "  FAIL %-28s %s\n%!" label e
              end)
            illegal_opcodes ;
          if not have_ptxas then Printf.printf "  gen  %s\n%!" label
          else
            match assemble_ok ptx with
            | Ok () ->
                incr assembled ;
                Printf.printf "  ok   %s\n%!" label
            | Error err ->
                failures := (name, label, String.trim err) :: !failures ;
                Printf.printf "  FAIL %-28s %s\n%!" label (String.trim err)))
    cases ;
  Printf.printf
    "  intrinsic sweep: %d/%d kernels generated, %d assembled%s\n%!"
    !generated
    (List.length cases)
    !assembled
    (if have_ptxas then "" else " (ptxas absent)") ;
  match List.rev !failures with
  | [] -> ()
  | fs ->
      let names = List.sort_uniq compare (List.map (fun (n, _, _) -> n) fs) in
      Alcotest.fail
        (Printf.sprintf
           "%d intrinsic kernel(s) rejected — broken intrinsic(s): %s\n%s"
           (List.length fs)
           (String.concat ", " names)
           (String.concat
              "\n"
              (List.map
                 (fun (_, label, e) -> Printf.sprintf "  %s: %s" label e)
                 fs)))

let () =
  Alcotest.run
    "ptx_intrinsic_sweep"
    [
      ( "dispatch",
        [
          Alcotest.test_case
            "emitter name sets partition the intrinsic surface"
            `Quick
            test_name_sets_disjoint;
          Alcotest.test_case
            "every dispatched intrinsic has a sweep kernel"
            `Quick
            test_every_name_has_a_recipe;
        ] );
      ( "ptxas",
        [
          Alcotest.test_case
            "generates (always) and assembles (if ptxas) a kernel per \
             intrinsic name"
            `Quick
            sweep;
        ] );
    ]
