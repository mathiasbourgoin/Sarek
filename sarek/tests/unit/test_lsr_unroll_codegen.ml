(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Code generation of [lsr] (Lshr32/Lshr64) and of the automatic unroll hint
 * (Sarek_ir_unroll), without a device. specs/codegen-lsr-unroll.md.
 *
 * 1. Every backend spells [lsr] as its native unsigned shift, and none of
 *    them still contains the old arithmetic-shift/xor emulation.
 * 2. The unroll rule: constant bounds, trip count 1..16, nest expansion <= 64,
 *    explicit pragmas win (SAREK_OPT_DISABLE=unroll: sarek/tests/policy_env).
 * 3. The hint as each backend prints it (#pragma unroll, [[unroll]]).
 * 4. The emitted source is accepted by glslangValidator, naga and clang (as
 *    OpenCL C) when they are on PATH.
 ******************************************************************************)

open Sarek_ir_types
open Sarek_codegen

let contains s sub =
  let n = String.length s and m = String.length sub in
  let rec go i = i + m <= n && (String.sub s i m = sub || go (i + 1)) in
  go 0

let count s sub =
  let n = String.length s and m = String.length sub in
  let rec go i acc =
    if i + m > n then acc
    else if String.sub s i m = sub then go (i + m) (acc + 1)
    else go (i + 1) acc
  in
  go 0 0

let ir_of kern =
  let _, kirc = kern in
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

(******************************************************************************
 * Kernels
 ******************************************************************************)

let lsr32_kernel =
  [%kernel
    fun (a : int32 vector)
        (s : int32 vector)
        (out : int32 vector)
        (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then begin
        out.(tid) <- a.(tid) lsr s.(tid) ;
        a.(tid) <- (a.(tid) + 1l) lsr 3l
      end]

let lsr64_kernel =
  [%kernel
    fun (a : int64 vector)
        (s : int64 vector)
        (out : int64 vector)
        (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then out.(tid) <- a.(tid) lsr s.(tid)]

(* q5_K-like: two nested constant loops (8 x 4) extracting bytes with lsr, a
   downto loop, a loop with a parameter bound, a trip-32 loop, an explicitly
   annotated loop, and a three-deep 4 x 4 x 8 nest (expansion 128). *)
let loops_kernel =
  [%kernel
    fun (w : int32 vector) (out : int32 vector) (n : int32) (m : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then begin
        let acc = mut 0l in
        for wi = 0 to 7 do
          let word = w.((tid * 8) + wi) in
          for k = 0 to 3 do
            acc := acc + ((word lsr (8 * k)) land 255l)
          done
        done ;
        for j = 3 downto 0 do
          acc := acc + j
        done ;
        for j = 0 to m - 1 do
          acc := acc + j
        done ;
        for j = 0 to 31 do
          acc := acc lxor j
        done ;
        pragma
          ["unroll 1"]
          (for j = 0 to 3 do
             acc := acc + (j * 3l)
           done) ;
        for a = 0 to 3 do
          for b = 0 to 3 do
            for c = 0 to 7 do
              acc := acc + (a * b * c)
            done
          done
        done ;
        out.(tid) <- acc
      end]

(******************************************************************************
 * 1. lsr spellings
 ******************************************************************************)

let gens : (string * (kernel -> string)) list =
  [
    ("CUDA", Sarek_ir_cuda.generate);
    ("OpenCL", Sarek_ir_opencl.generate);
    ("Metal", Sarek_ir_metal.generate);
    ("GLSL", fun k -> Sarek_ir_glsl.generate k);
    ("WGSL", fun k -> Sarek_ir_wgsl.generate k);
    ("PTX", fun k -> Sarek_ir_ptx.generate k);
  ]

let gen name = List.assoc name gens

(* The old expansion's fingerprints. *)
let emulation = ["(32 - "; "& 31)"; ">> 31)"; "xor.b32"; "shr.s32"]

let check_lsr32 () =
  let k = ir_of lsr32_kernel in
  List.iter
    (fun (backend, must) ->
      let src = gen backend k in
      let expected =
        match backend with "CUDA" | "OpenCL" | "Metal" -> 1 | _ -> 2
      in
      Alcotest.(check int)
        (Printf.sprintf "%s: logical shifts (%s)" backend must)
        expected
        (count src must) ;
      List.iter
        (fun bad ->
          Alcotest.(check bool)
            (Printf.sprintf "%s: no emulation fingerprint %S" backend bad)
            false
            (contains src bad))
        emulation)
    [
      (* The C dialects also print the wrapping [+] through the unsigned type
         (Sarek_ir_c_arith), so each of their two shifts is pinned in full. *)
      ("CUDA", "((int)((unsigned int)(a[tid]) >> (s[tid])))");
      ( "CUDA",
        "((int)((unsigned int)(((int)((unsigned int)a[tid] + 1u))) >> (3)))" );
      ("OpenCL", "((int)((uint)(a[tid]) >> (s[tid])))");
      ("OpenCL", "((int)((uint)(as_int(((uint)a[tid] + 1u))) >> (3)))");
      ("Metal", "((int)((uint)(a[tid]) >> (s[tid])))");
      ("Metal", "((int)((uint)(as_type<int>(((uint)a[tid] + 1u))) >> (3)))");
      ("GLSL", "int(uint(");
      ("WGSL", "bitcast<i32>(bitcast<u32>(");
      ("PTX", "shr.b32");
    ]

let check_lsr64 () =
  let k = ir_of lsr64_kernel in
  List.iter
    (fun (backend, must) ->
      Alcotest.(check bool)
        (Printf.sprintf "%s emits %s" backend must)
        true
        (contains (gen backend k) must))
    [
      ("CUDA", "((long long)((unsigned long long)(");
      ("OpenCL", "((long)((ulong)(");
      ("Metal", "((long)((ulong)(");
      ("GLSL", "int64_t(uint64_t(");
      ("PTX", "shr.b64");
    ] ;
  (* WGSL has no 64-bit integers: the kernel is refused (by its int64
     vectors before the shift is reached). *)
  match gen "WGSL" k with
  | _ -> Alcotest.fail "WGSL emitted a 64-bit kernel"
  | exception _ -> ()

(******************************************************************************
 * 2. The unroll rule, on hand-built IR
 ******************************************************************************)

let i32 n = EConst (CInt32 (Int32.of_int n))

let ivar name id =
  {var_name = name; var_id = id; var_type = TInt32; var_mutable = false}

let loop ?(dir = Upto) id lo hi body = SFor (ivar "i" id, lo, hi, dir, body)

let leaf = SExpr (EConst (CInt32 0l))

let rec value = function
  | EConst (CInt32 n) -> Int32.to_int n
  | EBinop (Add, a, b) -> value a + value b
  | EBinop (Sub, a, b) -> value a - value b
  | EBinop (Mul, a, b) -> value a * value b
  | _ -> Alcotest.fail "hinted loop with a non-constant bound"

(* Trip counts of the hinted loops, outermost first. *)
let rec hinted = function
  | SPragma (["unroll"], SFor (_, lo, hi, dir, body)) ->
      let n =
        match dir with
        | Upto -> value hi - value lo + 1
        | Downto -> value lo - value hi + 1
      in
      n :: hinted body
  | SPragma (_, b) | SBlock b | SLet (_, _, b) | SLetMut (_, _, b) -> hinted b
  | SFor (_, _, _, _, b) | SWhile (_, b) -> hinted b
  | SSeq l -> List.concat_map hinted l
  | SIf (_, a, b) -> hinted a @ Option.fold ~none:[] ~some:hinted b
  | _ -> []

let annotate = Sarek_ir_unroll.annotate_stmt

let check_rule () =
  let case label stmt expected =
    Alcotest.(check (list int)) label expected (hinted (annotate stmt))
  in
  case
    "0..7 around 0..3"
    (loop 1 (i32 0) (i32 7) (loop 2 (i32 0) (i32 3) leaf))
    [8; 4] ;
  case "3 downto 0" (loop ~dir:Downto 1 (i32 3) (i32 0) leaf) [4] ;
  case "0..15 (trip 16)" (loop 1 (i32 0) (i32 15) leaf) [16] ;
  case "0..16 (trip 17)" (loop 1 (i32 0) (i32 16) leaf) [] ;
  case "0..31" (loop 1 (i32 0) (i32 31) leaf) [] ;
  case
    "parameter bound"
    (loop 1 (i32 0) (EBinop (Sub, EVar (ivar "m" 9), i32 1)) leaf)
    [] ;
  case
    "constant arithmetic bound"
    (loop 1 (i32 0) (EBinop (Sub, EBinop (Mul, i32 2, i32 4), i32 1)) leaf)
    [8] ;
  case "zero trip (5..4)" (loop 1 (i32 5) (i32 4) leaf) [] ;
  case
    "8 x 8 x 8: inner two only"
    (loop
       1
       (i32 0)
       (i32 7)
       (loop 2 (i32 0) (i32 7) (loop 3 (i32 0) (i32 7) leaf)))
    [8; 8] ;
  case
    "4 x 4 x 4 = 64: all three"
    (loop
       1
       (i32 0)
       (i32 3)
       (loop 2 (i32 0) (i32 3) (loop 3 (i32 0) (i32 3) leaf)))
    [4; 4; 4] ;
  case
    "explicit pragma wins"
    (SPragma (["unroll 2"], loop 1 (i32 0) (i32 7) leaf))
    [] ;
  case
    "explicit pragma governs only its own loop"
    (SPragma (["unroll 2"], loop 1 (i32 0) (i32 7) (loop 2 (i32 0) (i32 3) leaf)))
    [4] ;
  case
    "sarek.inline is not a loop pragma"
    (SPragma (["sarek.inline 3"], loop 1 (i32 0) (i32 3) leaf))
    [4] ;
  case
    "two-string sarek.inline is not a loop pragma"
    (SPragma (["sarek.inline"; "3"], loop 1 (i32 0) (i32 3) leaf))
    [4] ;
  case
    "an explicit pragma's loop counts toward the outer bound (16 x 16)"
    (loop
       1
       (i32 0)
       (i32 15)
       (SPragma (["unroll 16"], loop 2 (i32 0) (i32 15) leaf)))
    [] ;
  case
    "a bound that wraps in int32 is not counted"
    (loop
       1
       (EConst (CInt32 Int32.max_int))
       (EBinop (Add, EConst (CInt32 Int32.max_int), i32 2))
       leaf)
    [] ;
  case
    "unroll inside a while"
    (SWhile (EConst (CBool true), loop 1 (i32 0) (i32 3) leaf))
    [4]

(******************************************************************************
 * 3. The hint in each backend
 ******************************************************************************)

(* loops_kernel: wi (8) and k (4), j downto (4), and the inner two of the
   4 x 4 x 8 nest (8 and 32) get the automatic hint; the [m] loop, the trip-32
   loop, the explicit "unroll 1" loop and the outer nest loop (128) do not. *)
let auto_hints = 5

let check_backend_hints () =
  let k = ir_of loops_kernel in
  List.iter
    (fun backend ->
      let src = gen backend k in
      Alcotest.(check int)
        (backend ^ ": automatic #pragma unroll")
        auto_hints
        (count src "#pragma unroll\n") ;
      Alcotest.(check int)
        (backend ^ ": the explicit pragma is kept")
        1
        (count src "#pragma unroll 1\n"))
    ["CUDA"; "OpenCL"; "Metal"] ;
  let glsl = gen "GLSL" k in
  Alcotest.(check int) "GLSL: [[unroll]]" auto_hints (count glsl "[[unroll]]") ;
  Alcotest.(check bool)
    "GLSL declares GL_EXT_control_flow_attributes"
    true
    (contains glsl "#extension GL_EXT_control_flow_attributes : require") ;
  Alcotest.(check bool)
    "GLSL without a hinted loop does not declare the extension"
    false
    (contains (gen "GLSL" (ir_of lsr32_kernel)) "control_flow_attributes") ;
  List.iter
    (fun backend ->
      Alcotest.(check bool)
        (backend ^ ": no unroll spelling")
        false
        (contains (gen backend k) "unroll"))
    ["WGSL"; "PTX"]

(******************************************************************************
 * 4. Validators
 ******************************************************************************)

let tool_available cmd =
  match Unix.system (Printf.sprintf "command -v %s >/dev/null 2>&1" cmd) with
  | Unix.WEXITED 0 -> true
  | _ -> false

let run_tool ~ext ~cmd src =
  let base = Filename.temp_file "sarek_lsr_unroll_" "" in
  let file = base ^ ext in
  let err = base ^ ".err" in
  Out_channel.with_open_bin file (fun oc -> output_string oc src) ;
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

let validators =
  [
    ( "GLSL/glslangValidator",
      lazy (tool_available "glslangValidator"),
      glslang,
      (fun k -> Sarek_ir_glsl.generate k),
      [lsr32_kernel; lsr64_kernel; loops_kernel] );
    ( "WGSL/naga",
      lazy (tool_available "naga"),
      naga,
      (fun k -> Sarek_ir_wgsl.generate k),
      [lsr32_kernel; loops_kernel] );
    ( "OpenCL/clang",
      lazy (Opencl_gate.Opencl_clang.available ()),
      Opencl_gate.Opencl_clang.run_clang,
      Sarek_ir_opencl.generate,
      [lsr32_kernel; lsr64_kernel; loops_kernel] );
  ]

let check_validators () =
  List.iter
    (fun (label, available, tool, gen, kernels) ->
      if not (Lazy.force available) then
        Printf.printf "  SKIP %s: tool not available\n%!" label
      else
        List.iter
          (fun kern ->
            let src = gen (ir_of kern) in
            match tool src with
            | Ok () -> ()
            | Error log ->
                Alcotest.failf
                  "%s rejected:\n%s\n--- source ---\n%s"
                  label
                  log
                  src)
          kernels)
    validators

(* Negative control: the GLSL [[unroll]] really depends on the extension line,
   so glslang accepting the source above is not vacuous. *)
let check_glslang_needs_extension () =
  if not (tool_available "glslangValidator") then
    print_endline "  SKIP glslang negative control: tool not available"
  else
    let src = Sarek_ir_glsl.generate (ir_of loops_kernel) in
    let stripped =
      Str.global_replace
        (Str.regexp_string
           "#extension GL_EXT_control_flow_attributes : require\n")
        ""
        src
    in
    match glslang stripped with
    | Error _ -> ()
    | Ok () ->
        Alcotest.fail
          "NEGATIVE CONTROL FAILED: glslang accepted [[unroll]] without \
           GL_EXT_control_flow_attributes"

let () =
  Alcotest.run
    "lsr-unroll-codegen"
    [
      ( "lsr",
        [
          Alcotest.test_case "int32 spellings" `Quick check_lsr32;
          Alcotest.test_case "int64 spellings" `Quick check_lsr64;
        ] );
      ( "unroll",
        [
          Alcotest.test_case "rule" `Quick check_rule;
          Alcotest.test_case "backends" `Quick check_backend_hints;
        ] );
      ( "validators",
        [
          Alcotest.test_case "accept" `Quick check_validators;
          Alcotest.test_case
            "glslang needs the extension"
            `Quick
            check_glslang_needs_extension;
        ] );
    ]
