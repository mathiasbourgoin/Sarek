(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Intrinsic-surface reconciliation.
 *
 * A Sarek stdlib intrinsic is declared ONCE (let%sarek_intrinsic) but has to be
 * honoured by three independent tables, each of which had drifted:
 *
 *   1. Sarek_registry  (FFI)   - written by the PPX from the declaration itself,
 *                                so it is the GROUND TRUTH for "which names
 *                                exist and what device symbol each one means".
 *   2. Sarek_pure_registry     - a hand-maintained copy used by GPU codegen for
 *                                path-qualified calls. Float64 entries were a
 *                                bare NAME list with no name -> symbol mapping,
 *                                and five declared names were simply absent.
 *   3. The native (cpu_kern)   - Sarek_native_intrinsics.map_stdlib_path picks a
 *      target module              module, then the intrinsic name is copied
 *                                VERBATIM onto it. Any name the target module
 *                                does not export is a compile error inside
 *                                PPX-generated code.
 *
 * Fixing the reported thirteen names is not enough on its own: nothing stopped
 * the fourteenth. This test derives the expectation from (1) at link time, so a
 * new let%sarek_intrinsic that is not routable goes red without anyone
 * remembering to update a list.
 *
 * How it can fail (i.e. it is not vacuous):
 *   - a stdlib name absent from Sarek_pure_registry     -> test_pure_registry_*
 *   - a pure-registry symbol disagreeing with the       -> test_symbol_agreement
 *     declaration's own CUDA/OpenCL spelling
 *   - a stdlib name with no native target symbol        -> BUILD failure in the
 *     (the witness below names it explicitly)              witness table
 *   - a stdlib name missing from the witness table      -> test_native_witness_*
 *
 * The witness tables are deliberately hand-written rather than generated: each
 * entry is an OCaml reference to the real runtime function, so a missing one
 * cannot compile, and the completeness check below proves the table is not
 * merely a subset someone stopped extending.
 ******************************************************************************)

let () = Sarek_stdlib.force_init ()

(* Force sarek_float64's module initialiser, which is what registers the
   Float64 intrinsics into Sarek_registry. Without this the registry is empty
   for Float64 and every check below would pass vacuously. *)
let () = ignore (Sarek_float64.Float64.sin 0.0)

(******************************************************************************
 * Ground truth
 ******************************************************************************)

(** Every [(name, cuda_template, opencl_template)] the linked stdlib registered
    under [ffi_path]. *)
let declared ffi_path =
  Hashtbl.fold
    (fun (path, name) (fi : Sarek_registry.fun_info) acc ->
      if path = ffi_path then
        (name, fi.fi_device "CUDA", fi.fi_device "OpenCL") :: acc
      else acc)
    Sarek_registry.fun_registry
    []
  |> List.sort compare

(** A device template containing [%s] is an operator/cast form: codegen expands
    it structurally instead of emitting [symbol(args)], so it needs no
    pure-registry entry. Everything else is a plain call, and a plain call is
    exactly what the pure registry exists to name. *)
let is_plain_call template =
  let rec scan i =
    if i + 1 >= String.length template then true
    else if template.[i] = '%' && template.[i + 1] = 's' then false
    else scan (i + 1)
  in
  scan 0

let plain_calls ffi_path =
  List.filter (fun (_, cuda, _) -> is_plain_call cuda) (declared ffi_path)

(******************************************************************************
 * 1. Pure registry completeness
 ******************************************************************************)

(* (label, ffi registration path, user-visible paths a kernel may write) *)
let float32_surface =
  ( "Float32",
    ["Sarek_stdlib"; "Float32"],
    [["Float32"]; ["Math"; "Float32"]; ["Sarek_stdlib_meta"; "Float32"]] )

let float64_surface = ("Float64", ["Sarek_float64"; "Float64"], [["Float64"]])

let check_pure_completeness (label, ffi_path, user_paths) () =
  let names = plain_calls ffi_path in
  Alcotest.(check bool)
    (label ^ ": stdlib declared at least one plain-call intrinsic")
    true
    (names <> []) ;
  List.iter
    (fun user_path ->
      let missing =
        List.filter
          (fun (name, _, _) ->
            Sarek_pure_registry.fun_device_template ~module_path:user_path name
            = None)
          names
      in
      let printable = String.concat "." user_path in
      Alcotest.(check (list string))
        (Printf.sprintf
           "%s: every name declared by the stdlib resolves under path %s (an \
            unresolved name reaches GPU codegen as an Unknown intrinsic error, \
            or falls through to a backend arm that may spell it differently)"
           label
           printable)
        []
        (List.map (fun (n, _, _) -> n) missing))
    user_paths

(******************************************************************************
 * 2. Symbol agreement
 *
 * The pure registry must emit the SAME device symbol the declaration itself
 * names. This is the check that catches a bare name list: adding "abs_float"
 * to a list of names emits a call to abs_float(), while the declaration says
 * the symbol is fabs.
 ******************************************************************************)

let check_symbol_agreement (label, ffi_path, user_paths) () =
  let names = plain_calls ffi_path in
  let user_path = List.hd user_paths in
  List.iter
    (fun (name, cuda, opencl) ->
      match
        Sarek_pure_registry.fun_device_template ~module_path:user_path name
      with
      | None ->
          () (* completeness is checked separately; do not double-report *)
      | Some device ->
          Alcotest.(check string)
            (Printf.sprintf "%s.%s: CUDA symbol" label name)
            cuda
            (device ~framework:"CUDA") ;
          Alcotest.(check string)
            (Printf.sprintf "%s.%s: OpenCL symbol" label name)
            opencl
            (device ~framework:"OpenCL"))
    names

(******************************************************************************
 * 3. Native witness
 *
 * One entry per declared intrinsic, referencing the runtime function the native
 * backend will emit a call to. The reference is what makes a missing runtime
 * function a BUILD error rather than a silent gap; the completeness check makes
 * a missing witness entry a test failure.
 *
 * Read the module aliases as the assertion: these are exactly the modules
 * Sarek_native_intrinsics.map_stdlib_path resolves ["Float32"] / ["Float64"] to.
 ******************************************************************************)

module F32 = Sarek.Sarek_cpu_runtime.Float32
module F64 = Sarek.Sarek_cpu_runtime.Float64

let u f = fun () -> ignore (f 1.0 : float)

let b f = fun () -> ignore (f 1.0 2.0 : float)

let t f = fun () -> ignore (f 1.0 2.0 3.0 : float)

let float32_witness : (string * (unit -> unit)) list =
  [
    ("sin", u F32.sin);
    ("cos", u F32.cos);
    ("tan", u F32.tan);
    ("asin", u F32.asin);
    ("acos", u F32.acos);
    ("atan", u F32.atan);
    ("sinh", u F32.sinh);
    ("cosh", u F32.cosh);
    ("tanh", u F32.tanh);
    ("exp", u F32.exp);
    ("log", u F32.log);
    ("log10", u F32.log10);
    ("sqrt", u F32.sqrt);
    ("ceil", u F32.ceil);
    ("floor", u F32.floor);
    ("rsqrt", u F32.rsqrt);
    ("pow", b F32.pow);
    ("atan2", b F32.atan2);
    ("fma", t F32.fma);
    ("of_int", fun () -> ignore (F32.of_int 1 : float));
    ("to_int", fun () -> ignore (F32.to_int 1.0 : int));
    ("add", b F32.add);
    ("mul", b F32.mul);
    ("div", b F32.div);
    (* The seven that had no native target before this change. *)
    ("abs_float", u F32.abs_float);
    ("expm1", u F32.expm1);
    ("log1p", u F32.log1p);
    ("hypot", b F32.hypot);
    ("copysign", b F32.copysign);
    ("fmod", b F32.fmod);
    ("minus", b F32.minus);
    ("add_float32", b F32.add_float32);
    ("sub_float32", b F32.sub_float32);
    ("mul_float32", b F32.mul_float32);
    ("div_float32", b F32.div_float32);
  ]

let float64_witness : (string * (unit -> unit)) list =
  [
    ("sin", u F64.sin);
    ("cos", u F64.cos);
    ("tan", u F64.tan);
    ("asin", u F64.asin);
    ("acos", u F64.acos);
    ("atan", u F64.atan);
    ("sinh", u F64.sinh);
    ("cosh", u F64.cosh);
    ("tanh", u F64.tanh);
    ("exp", u F64.exp);
    ("log", u F64.log);
    ("log10", u F64.log10);
    ("sqrt", u F64.sqrt);
    ("ceil", u F64.ceil);
    ("floor", u F64.floor);
    ("expm1", u F64.expm1);
    ("log1p", u F64.log1p);
    ("abs_float", u F64.abs_float);
    ("rsqrt", u F64.rsqrt);
    ("pow", b F64.pow);
    ("atan2", b F64.atan2);
    ("hypot", b F64.hypot);
    ("copysign", b F64.copysign);
    ("fmod", b F64.fmod);
    ("add_float64", b F64.add_float64);
    ("sub_float64", b F64.sub_float64);
    ("mul_float64", b F64.mul_float64);
    ("div_float64", b F64.div_float64);
    ("of_int", fun () -> ignore (F64.of_int 1 : float));
    ("of_int32", fun () -> ignore (F64.of_int32 1l : float));
    ("to_int", fun () -> ignore (F64.to_int 1.0 : int));
    ("to_int32", fun () -> ignore (F64.to_int32 1.0 : int32));
    ("of_float32", u F64.of_float32);
    ("to_float32", u F64.to_float32);
    ("+.", b F64.( +. ));
    ("-.", b F64.( -. ));
    ("*.", b F64.( *. ));
    ("/.", b F64.( /. ));
    ("<=", fun () -> ignore (F64.( <= ) 1.0 2.0 : bool));
    (">=", fun () -> ignore (F64.( >= ) 1.0 2.0 : bool));
    ("<", fun () -> ignore (F64.( < ) 1.0 2.0 : bool));
    (">", fun () -> ignore (F64.( > ) 1.0 2.0 : bool));
  ]

let check_native_witness label ffi_path witness () =
  let declared_names =
    List.map (fun (n, _, _) -> n) (declared ffi_path) |> List.sort compare
  in
  let witness_names = List.map fst witness |> List.sort compare in
  Alcotest.(check (list string))
    (Printf.sprintf
       "%s: every declared intrinsic has a native witness (a name missing here \
        has no Sarek_cpu_runtime.%s.<name> to lower to, so any native kernel \
        calling it fails to compile)"
       label
       label)
    []
    (List.filter (fun n -> not (List.mem n witness_names)) declared_names) ;
  Alcotest.(check (list string))
    (Printf.sprintf
       "%s: the witness table has no entry the stdlib does not declare"
       label)
    []
    (List.filter (fun n -> not (List.mem n declared_names)) witness_names) ;
  (* Actually call each one: proves the entries are live values, not just
     names that typecheck. *)
  List.iter (fun (_, thunk) -> thunk ()) witness

(******************************************************************************
 * 4. The native mapping the witness assumes is the one the PPX emits
 ******************************************************************************)

let check_native_target_paths () =
  Alcotest.(check (list string))
    "Float32 native target module"
    ["Sarek"; "Sarek_cpu_runtime"; "Float32"]
    (Sarek_native_intrinsics.map_stdlib_path ["Float32"]) ;
  Alcotest.(check (list string))
    "Float64 native target module (Stdlib.Float does not export the whole \
     Float64 surface)"
    ["Sarek"; "Sarek_cpu_runtime"; "Float64"]
    (Sarek_native_intrinsics.map_stdlib_path ["Float64"])

(******************************************************************************
 * Registration
 ******************************************************************************)

(******************************************************************************
 * 5. The FFI fallback emits the CALLING backend's spelling
 *
 * A PPX kernel writes Float32.sin, and the PPX records the path as
 * ["Sarek_stdlib"; "Float32"] — the DECLARATION-SITE path, which
 * Sarek_pure_registry does not register. Such calls therefore land on the
 * dispatcher's FFI fallback (Sarek_ir_intrinsic_dispatch.emit_registry_template),
 * which used to ask Sarek_registry for the "generic" framework. cuda_or_opencl
 * resolves "generic" to the CUDA branch, so an OpenCL or Metal kernel calling
 * Float32.abs_float was emitted as fabsf(...) — a name neither OpenCL C nor MSL
 * declares.
 *
 * The clang gate below is the positive/negative control pair: it compiles the
 * emitted OpenCL, and separately confirms that the PRE-FIX spelling is rejected
 * by the same compiler. Without the negative control this check could pass
 * while testing nothing.
 ******************************************************************************)

let f32_call_kernel name arity : Sarek_ir_types.kernel =
  let open Sarek_ir_types in
  let a =
    {var_name = "a"; var_id = 0; var_type = TVec TFloat32; var_mutable = false}
  in
  let b =
    {var_name = "b"; var_id = 1; var_type = TVec TFloat32; var_mutable = false}
  in
  let idx =
    {var_name = "idx"; var_id = 2; var_type = TInt32; var_mutable = false}
  in
  let arg = EArrayRead ("a", EVar idx) in
  {
    default_kernel with
    kern_name = "ffi_fallback_" ^ name;
    kern_params =
      [
        DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ];
    kern_body =
      SLet
        ( idx,
          EIntrinsic ([], "global_thread_id", []),
          SAssign
            ( LArrayElem ("b", EVar idx),
              EIntrinsic
                ( ["Sarek_stdlib"; "Float32"],
                  name,
                  if arity = 1 then [arg] else [arg; arg] ) ) );
  }

(* Names reached ONLY through the FFI fallback: absent from the pure registry
   under this path and from the OpenCL backend's hardcoded arm. expm1 and log1p
   belong here for the same reason as the other four -- they are declared
   [dev "expm1f" "expm1"] / [dev "log1pf" "log1p"], so the discarded framework
   was observable on them too. (Metal polyfills those two in its pre_hook, which
   is why they were easy to overlook; OpenCL has no pre_hook and took the CUDA
   spelling straight through.) *)
let fallback_names =
  [
    ("abs_float", 1);
    ("copysign", 2);
    ("hypot", 2);
    ("fmod", 2);
    ("expm1", 1);
    ("log1p", 1);
  ]

let opencl_source name arity =
  Sarek_codegen.Sarek_ir_opencl.generate_with_types
    ~types:[]
    (f32_call_kernel name arity)

let check_opencl_spelling () =
  List.iter
    (fun (name, arity) ->
      let src = opencl_source name arity in
      let cuda_spelling =
        match
          Sarek_registry.fun_device_template
            ~module_path:["Sarek_stdlib"; "Float32"]
            ~framework:"CUDA"
            name
        with
        | Some s -> s
        | None -> Alcotest.fail (name ^ ": not in the FFI registry")
      in
      let opencl_spelling =
        match
          Sarek_registry.fun_device_template
            ~module_path:["Sarek_stdlib"; "Float32"]
            ~framework:"OpenCL"
            name
        with
        | Some s -> s
        | None -> Alcotest.fail (name ^ ": not in the FFI registry")
      in
      (* Guard against a vacuous test: these names must actually differ between
         the two branches, or the check below proves nothing. *)
      Alcotest.(check bool)
        (Printf.sprintf
           "%s: the CUDA and OpenCL declarations differ (%s vs %s), so the \
            framework the fallback passes is observable"
           name
           cuda_spelling
           opencl_spelling)
        true
        (cuda_spelling <> opencl_spelling) ;
      let contains hay needle =
        let n = String.length needle and h = String.length hay in
        let rec go i =
          i + n <= h && (String.sub hay i n = needle || go (i + 1))
        in
        go 0
      in
      Alcotest.(check bool)
        (Printf.sprintf
           "OpenCL kernel calling Float32.%s emits %s"
           name
           opencl_spelling)
        true
        (contains src (opencl_spelling ^ "(")) ;
      Alcotest.(check bool)
        (Printf.sprintf
           "OpenCL kernel calling Float32.%s does NOT emit the CUDA spelling %s"
           name
           cuda_spelling)
        false
        (contains src (cuda_spelling ^ "(")))
    fallback_names

let check_opencl_compiles () =
  if not (Opencl_gate.Opencl_clang.available ()) then Alcotest.skip ()
  else
    List.iter
      (fun (name, arity) ->
        let src = opencl_source name arity in
        (* Positive control: what we emit now compiles. *)
        (match Opencl_gate.Opencl_clang.run_clang src with
        | Ok () -> ()
        | Error log ->
            Alcotest.failf
              "emitted OpenCL for Float32.%s does not compile:\n\
               %s\n\
               --- source ---\n\
               %s"
              name
              log
              src) ;
        (* Negative control: the PRE-FIX spelling is rejected by the same
           compiler. Proves the positive control above is not vacuous — that
           clang would in fact have caught the old output. *)
        let cuda_spelling =
          Option.get
            (Sarek_registry.fun_device_template
               ~module_path:["Sarek_stdlib"; "Float32"]
               ~framework:"CUDA"
               name)
        in
        let opencl_spelling =
          Option.get
            (Sarek_registry.fun_device_template
               ~module_path:["Sarek_stdlib"; "Float32"]
               ~framework:"OpenCL"
               name)
        in
        let regressed =
          Str.global_replace
            (Str.regexp_string (opencl_spelling ^ "("))
            (cuda_spelling ^ "(")
            src
        in
        match Opencl_gate.Opencl_clang.run_clang regressed with
        | Error _ -> ()
        | Ok () ->
            Alcotest.failf
              "NEGATIVE CONTROL FAILED: clang accepted the pre-fix spelling %s \
               for Float32.%s, so this gate cannot detect the regression it \
               exists to catch"
              cuda_spelling
              name)
      fallback_names

let () =
  Alcotest.run
    "intrinsic-surface"
    [
      ( "pure-registry completeness",
        [
          Alcotest.test_case
            "Float32"
            `Quick
            (check_pure_completeness float32_surface);
          Alcotest.test_case
            "Float64"
            `Quick
            (check_pure_completeness float64_surface);
        ] );
      ( "symbol agreement",
        [
          Alcotest.test_case
            "Float32"
            `Quick
            (check_symbol_agreement float32_surface);
          Alcotest.test_case
            "Float64"
            `Quick
            (check_symbol_agreement float64_surface);
        ] );
      ( "native surface",
        [
          Alcotest.test_case
            "Float32"
            `Quick
            (check_native_witness
               "Float32"
               ["Sarek_stdlib"; "Float32"]
               float32_witness);
          Alcotest.test_case
            "Float64"
            `Quick
            (check_native_witness
               "Float64"
               ["Sarek_float64"; "Float64"]
               float64_witness);
          Alcotest.test_case "target modules" `Quick check_native_target_paths;
        ] );
      ( "framework-aware FFI fallback",
        [
          Alcotest.test_case "OpenCL spelling" `Quick check_opencl_spelling;
          Alcotest.test_case
            "emitted OpenCL compiles (clang gate)"
            `Quick
            check_opencl_compiles;
        ] );
    ]
