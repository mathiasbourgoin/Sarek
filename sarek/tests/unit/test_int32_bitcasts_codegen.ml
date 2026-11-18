(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Code generation of the stdlib bit reinterpretations Int32.float_of_bits and
 * Int32.bits_of_float on every source/assembly backend, without a device.
 *
 * 1. Both registrations (Sarek_stdlib and the FFI-free Sarek_stdlib_meta twin)
 *    declare the same three C-family spellings, and the registry still refuses
 *    GLSL and WGSL rather than handing them a C spelling.
 * 2. Each backend emits its own spelling, from a PPX-built kernel (so the
 *    module path is the one the PPX really records) and from hand-built IR
 *    under every stdlib Int32 path; Metal never gets OpenCL's [as_float].
 * 3. The emitted source is accepted by the validator for that language when
 *    the tool is on PATH (glslangValidator, naga, clang for OpenCL); each
 *    validator is also shown to REJECT a deliberately wrong spelling, so a
 *    pass is not vacuous.
 * 4. The path-scoped lowerings (GLSL, WGSL, PTX) do not capture a same-named
 *    intrinsic from another module, and every backend refuses a call with two
 *    arguments instead of emitting source.
 ******************************************************************************)

open Sarek_ir_types
open Sarek_codegen
module Std = Sarek_stdlib.Std

let () = Sarek_stdlib.force_init ()

let () = Sarek_stdlib_meta.force_init ()

let contains s sub =
  let n = String.length s and m = String.length sub in
  let rec go i = i + m <= n && (String.sub s i m = sub || go (i + 1)) in
  go 0

(******************************************************************************
 * 1. Registry templates
 ******************************************************************************)

let spellings =
  [
    ("float_of_bits", "CUDA", "__int_as_float(%s)");
    ("float_of_bits", "OpenCL", "as_float(%s)");
    ("float_of_bits", "Metal", "as_type<float>(%s)");
    ("bits_of_float", "CUDA", "__float_as_int(%s)");
    ("bits_of_float", "OpenCL", "as_int(%s)");
    ("bits_of_float", "Metal", "as_type<int>(%s)");
  ]

let registrations = [["Sarek_stdlib"; "Int32"]; ["Sarek_stdlib_meta"; "Int32"]]

let check_templates () =
  List.iter
    (fun module_path ->
      List.iter
        (fun (name, framework, expected) ->
          Alcotest.(check (option string))
            (Printf.sprintf
               "%s.%s on %s"
               (String.concat "." module_path)
               name
               framework)
            (Some expected)
            (Sarek_registry.fun_device_template ~module_path ~framework name))
        spellings ;
      List.iter
        (fun framework ->
          List.iter
            (fun name ->
              match
                Sarek_registry.fun_device_template ~module_path ~framework name
              with
              | exception Sarek_registry.No_device_spelling _ -> ()
              | s ->
                  Alcotest.failf
                    "%s.%s: the FFI registry answered %s with %s instead of \
                     refusing"
                    (String.concat "." module_path)
                    name
                    framework
                    (Option.value s ~default:"None"))
            ["float_of_bits"; "bits_of_float"])
        ["GLSL"; "WGSL"])
    registrations

(* The wrapping arithmetic templates (Sarek_ir_c_arith): the stdlib and its
   FFI-free twin must register the same spelling, and it must go through the
   unsigned type, on every C-family framework. *)
(******************************************************************************
 * 2. Spellings, from a PPX kernel and from hand-built IR
 ******************************************************************************)

let ppx_kernel =
  [%kernel
    fun (a : int32 vector)
        (b : float32 vector)
        (c : int32 vector)
        (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then begin
        b.(tid) <- Int32.float_of_bits a.(tid) ;
        c.(tid) <- Int32.bits_of_float b.(tid)
      end]

let ppx_ir =
  let _, kirc = ppx_kernel in
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let var name id ty =
  {var_name = name; var_id = id; var_type = ty; var_mutable = false}

let param name id ty =
  DParam (var name id (TVec ty), Some {arr_elttype = ty; arr_memspace = Global})

(* b.(tid) <- float_of_bits a.(tid) ; c.(tid) <- bits_of_float b.(tid), with
   both calls under [path]. [extra] appends arguments to test arity. *)
let ir_kernel ?(extra = []) path : kernel =
  let tid = var "tid" 3 TInt32 in
  let rd a = EArrayRead (a, EVar tid) in
  {
    default_kernel with
    kern_name = "bitcasts";
    kern_params = [param "a" 0 TInt32; param "b" 1 TFloat32; param "c" 2 TInt32];
    kern_body =
      SLet
        ( tid,
          EIntrinsic ([], "global_thread_id", []),
          SSeq
            [
              SAssign
                ( LArrayElem ("b", EVar tid),
                  EIntrinsic (path, "float_of_bits", rd "a" :: extra) );
              SAssign
                ( LArrayElem ("c", EVar tid),
                  EIntrinsic (path, "bits_of_float", rd "b" :: extra) );
            ] );
  }

(* float_of_bits applied to the literal Int32.min_int (the -0.0 pattern). The
   C family used to print it as [-2147483648] — unary minus on a [long] — so
   [as_float] saw an 8-byte operand; WGSL's [-2147483648i] does not fit i32. *)
let literal_kernel : kernel =
  let tid = var "tid" 3 TInt32 in
  {
    default_kernel with
    kern_name = "bitcast_literal";
    kern_params = [param "b" 1 TFloat32; param "c" 2 TInt32];
    kern_body =
      SLet
        ( tid,
          EIntrinsic ([], "global_thread_id", []),
          SSeq
            [
              SAssign
                ( LArrayElem ("b", EVar tid),
                  EIntrinsic
                    ( ["Sarek_stdlib"; "Int32"],
                      "float_of_bits",
                      [EConst (CInt32 Int32.min_int)] ) );
              SAssign (LArrayElem ("c", EVar tid), EConst (CInt32 Int32.min_int));
            ] );
  }

let backends : (string * (kernel -> string) * string list * string list) list =
  [
    ("CUDA", Sarek_ir_cuda.generate, ["__int_as_float("; "__float_as_int("], []);
    ("OpenCL", Sarek_ir_opencl.generate, ["as_float("; "as_int("], []);
    ( "Metal",
      Sarek_ir_metal.generate,
      ["as_type<float>("; "as_type<int>("],
      [" as_float("; "(as_float("; " as_int("; "(as_int("] );
    ( "GLSL",
      (fun k -> Sarek_ir_glsl.generate k),
      ["intBitsToFloat("; "floatBitsToInt("],
      [] );
    ( "WGSL",
      (fun k -> Sarek_ir_wgsl.generate k),
      ["bitcast<f32>("; "bitcast<i32>("],
      [] );
    ("PTX", (fun k -> Sarek_ir_ptx.generate k), ["mov.b32"], []);
  ]

let check_spellings ?(only = []) label k () =
  List.iter
    (fun (backend, gen, must, must_not) ->
      if only = [] || List.mem backend only then begin
        let src = gen k in
        List.iter
          (fun s ->
            Alcotest.(check bool)
              (Printf.sprintf "%s %s emits %s" label backend s)
              true
              (contains src s))
          must ;
        List.iter
          (fun s ->
            Alcotest.(check bool)
              (Printf.sprintf "%s %s does not emit %s" label backend s)
              false
              (contains src s))
          (must_not @ ["Int32."; "float_of_bits"; "bits_of_float"])
      end)
    backends

(* The PPX records the declaration-site path; it must be one the path-scoped
   lowerings accept. *)
let check_ppx_path () =
  let paths = ref [] in
  let rec expr = function
    | EIntrinsic (p, ("float_of_bits" | "bits_of_float"), args) ->
        paths := p :: !paths ;
        List.iter expr args
    | EIntrinsic (_, _, args) -> List.iter expr args
    | EArrayRead (_, e) -> expr e
    | _ -> ()
  in
  let rec stmt = function
    | SAssign (_, e) -> expr e
    | SSeq l -> List.iter stmt l
    | SLet (_, e, s) ->
        expr e ;
        stmt s
    | SIf (c, a, b) ->
        expr c ;
        stmt a ;
        Option.iter stmt b
    | _ -> ()
  in
  stmt ppx_ir.kern_body ;
  Alcotest.(check int) "both calls found in the PPX IR" 2 (List.length !paths) ;
  List.iter
    (fun p ->
      Alcotest.(check bool)
        (Printf.sprintf
           "PPX path [%s] is a stdlib Int32 path"
           (String.concat "; " p))
        true
        (Sarek_ir_intrinsic_dispatch.is_stdlib_int32_path p))
    !paths

(******************************************************************************
 * 3. Validators (positive and negative control)
 ******************************************************************************)

let tool_available cmd =
  match Unix.system (Printf.sprintf "command -v %s >/dev/null 2>&1" cmd) with
  | Unix.WEXITED 0 -> true
  | _ -> false

let run_tool ~ext ~cmd src =
  let base = Filename.temp_file "sarek_bitcast_" "" in
  let file = base ^ ext in
  let err = base ^ ".err" in
  let oc = open_out file in
  output_string oc src ;
  close_out oc ;
  let rc =
    Unix.system
      (Printf.sprintf
         "%s >%s 2>&1"
         (cmd (Filename.quote file) (Filename.quote (base ^ ".out")))
         (Filename.quote err))
  in
  let log =
    try In_channel.with_open_bin err In_channel.input_all with _ -> ""
  in
  List.iter
    (fun f -> try Sys.remove f with _ -> ())
    [file; err; base; base ^ ".out"] ;
  match rc with Unix.WEXITED 0 -> Ok () | _ -> Error log

let glslang =
  run_tool ~ext:".comp" ~cmd:(fun src out ->
      Printf.sprintf "glslangValidator -V -S comp -o %s %s" out src)

let naga =
  run_tool ~ext:".wgsl" ~cmd:(fun src _ -> Printf.sprintf "naga %s" src)

let replace ~sub ~by s = Str.global_replace (Str.regexp_string sub) by s

(* (label, tool, available, generator, a correct spelling and a wrong one the
   language does not declare) *)
let validators =
  [
    ( "GLSL/glslangValidator",
      lazy (tool_available "glslangValidator"),
      glslang,
      (fun k -> Sarek_ir_glsl.generate k),
      ("intBitsToFloat(", "as_float(") );
    ( "WGSL/naga",
      lazy (tool_available "naga"),
      naga,
      (fun k -> Sarek_ir_wgsl.generate k),
      ("bitcast<f32>(", "as_float(") );
    ( "OpenCL/clang",
      lazy (Opencl_gate.Opencl_clang.available ()),
      Opencl_gate.Opencl_clang.run_clang,
      Sarek_ir_opencl.generate,
      ("as_float(", "__int_as_float(") );
  ]

let check_validators () =
  List.iter
    (fun (label, available, tool, gen, (good, bad)) ->
      if not (Lazy.force available) then
        Printf.printf "  SKIP %s: tool not available\n%!" label
      else begin
        List.iter
          (fun (what, k) ->
            let src = gen k in
            (match tool src with
            | Ok () -> ()
            | Error log ->
                Alcotest.failf
                  "%s rejected the %s source:\n%s\n--- source ---\n%s"
                  label
                  what
                  log
                  src) ;
            match tool (replace ~sub:good ~by:bad src) with
            | Error _ -> ()
            | Ok () ->
                Alcotest.failf
                  "NEGATIVE CONTROL FAILED: %s accepted %s in place of %s"
                  label
                  bad
                  good)
          [
            ("PPX", ppx_ir);
            ("IR", ir_kernel ["Sarek_stdlib"; "Int32"]);
            ("min_int literal", literal_kernel);
          ]
      end)
    validators

(******************************************************************************
 * 4. Refusals
 ******************************************************************************)

let raises gen k = match gen k with _ -> false | exception _ -> true

let check_off_path () =
  let k = ir_kernel ["Dstest"; "Bits"] in
  List.iter
    (fun (backend, gen) ->
      Alcotest.(check bool)
        (Printf.sprintf
           "%s refuses Dstest.Bits.float_of_bits (not lowered as the stdlib \
            bitcast)"
           backend)
        true
        (raises gen k))
    [
      ("GLSL", fun k -> Sarek_ir_glsl.generate k);
      ("WGSL", fun k -> Sarek_ir_wgsl.generate k);
      ("PTX", fun k -> Sarek_ir_ptx.generate k);
    ] ;
  (* PTX has no registry fallback: its refusal names the module and says which
     float_of_bits it does lower. *)
  match Sarek_ir_ptx.generate k with
  | _ -> Alcotest.fail "PTX lowered Dstest.Bits.float_of_bits"
  | exception e ->
      let msg = Printexc.to_string e in
      Alcotest.(check bool)
        (Printf.sprintf "PTX refusal names the path and the stdlib (%s)" msg)
        true
        (contains msg "Dstest.Bits.float_of_bits"
        && contains msg "stdlib Int32.float_of_bits")

let check_arity () =
  let k = ir_kernel ~extra:[EConst (CInt32 1l)] ["Sarek_stdlib"; "Int32"] in
  List.iter
    (fun (backend, gen, _, _) ->
      Alcotest.(check bool)
        (Printf.sprintf "%s refuses a two-argument call" backend)
        true
        (raises gen k))
    backends

let () =
  Alcotest.run
    "int32-bitcasts-codegen"
    [
      ("registry", [Alcotest.test_case "templates" `Quick check_templates]);
      ( "spelling",
        [
          Alcotest.test_case "PPX path" `Quick check_ppx_path;
          Alcotest.test_case "PPX kernel" `Quick (check_spellings "PPX" ppx_ir);
        ]
        @ List.map
            (fun (p, only) ->
              Alcotest.test_case
                (String.concat "." p)
                `Quick
                (check_spellings ~only (String.concat "." p) (ir_kernel p)))
            [
              (* The bare [Int32] path is accepted by the path-scoped lowerings
                 (as by Native and the Interpreter); the C-family backends
                 resolve through the FFI registry, which knows only the
                 declaration-site paths below — as for every Int32 intrinsic. *)
              (["Int32"], ["GLSL"; "WGSL"; "PTX"]);
              (["Sarek_stdlib"; "Int32"], []);
              (["Sarek_stdlib_meta"; "Int32"], []);
            ] );
      ("validators", [Alcotest.test_case "compile" `Quick check_validators]);
      ( "refusal",
        [
          Alcotest.test_case "other module" `Quick check_off_path;
          Alcotest.test_case "arity" `Quick check_arity;
        ] );
    ]
