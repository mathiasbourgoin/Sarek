(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * An emitter must declare every name it uses (backlog-156).
 *
 * WHAT THIS IS FOR
 *
 * A backend emitter has two halves that never meet in the type system: the part
 * that DECLARES a generated identifier (a kernel parameter, a push-constant
 * field, a struct member) and the part that USES it (an expression arm). Each
 * half builds the identifier by string concatenation, at its own site, and
 * nothing in OCaml relates the two. Two spellings of one concept is therefore a
 * shape the compiler cannot see, and the only witness is a device compiler
 * rejecting the output.
 *
 * backlog-156 is exactly that: `Sarek_ir_glsl` declared the length of a vector
 * parameter as `<name>_len` (push-constant field plus `#define` alias) and used
 * it, for `EArrayLen`, as `sarek_<name>_length`. Any kernel taking a length
 * emitted GLSL naming an identifier the shader never declared; glslangValidator
 * exits 2 with "'sarek_a_length' : undeclared identifier". It survived because
 * nothing reaches `EArrayLen` on a GLSL path: no test, no example, and no
 * surface syntax — the Sarek frontend has no `len` primitive at all, so the
 * constructor is reachable only by hand-written IR. An unreachable feature
 * cannot be observed to be broken.
 *
 * HOW IT CHECKS, AND WHY THIS WAY
 *
 * The property "every identifier used is declared" needs a parser to state
 * directly, and a per-backend list of language builtins to approximate — and a
 * hand-maintained builtin list is the same unowned-list shape that produced the
 * bug. So this checks a corollary that needs neither:
 *
 *   Generate the SAME kernel twice, differing only in the construct under test.
 *   In each output, collect the identifiers occurring EXACTLY ONCE. An identifier
 *   used once and declared nowhere is a singleton; an identifier that is declared
 *   is not. Builtins, keywords and the kernel's own scaffolding appear in both
 *   outputs and cancel in the difference.
 *
 * What remains — a singleton the construct introduced — is a name the emitter
 * wrote and never bound. No allowlist, and it applies to any construct on any
 * backend, which is why the probe table below is a list rather than one case.
 *
 * WHY NOT EXTEND THE ARM-PARITY MATRIX (#94)
 *
 * test_backend_arm_parity.ml compares FIVE BACKENDS against each other on one
 * axis (does each know this intrinsic name). The defect here is within ONE
 * backend, between two of its own halves, and is invisible to a cross-backend
 * comparison: all five agree on `sarek_<a>_length` at the use site, which is
 * precisely why the arm-parity matrix was green throughout — five of the six
 * emitters spell it that way, GLSL having been the odd one out. Its lexical
 * companion, scripts/check-arm-parity-coverage.sh, extracts `arm` match-arm
 * literals and cannot see push-constant emission either. Different axis, so a
 * different instrument rather than a widened one.
 *
 * WHAT IT DOES NOT COVER
 *
 * Only the constructs listed in [probes], and only for the parameter shapes
 * those probes use. It sees a name that is used and never declared; it does not
 * see a name declared with the wrong type, wrong layout, or wrong offset.
 *
 * EXACTLY ONCE IS A CONDITION, NOT A DETAIL. The corollary above detects a free
 * name only while that name occurs EXACTLY ONCE in the output. A name the
 * emitter writes TWICE and binds nowhere has count 2, is not a singleton, and is
 * invisible here — the check reads green on a shader the device compiler
 * rejects. Measured, on this file: emitting `(sarek_a_undeclared_len -
 * sarek_a_undeclared_len)` for [EArrayLen] on GLSL leaves both `singleton-free`
 * GLSL cases green and is caught only by the `device-compiler` case below. So
 * this instrument covers single-use identifiers; a repeated one needs the
 * validator.
 *
 * AND THE VALIDATOR SKIPS AS PASS. [check_probe_validator] returns unit when
 * glslangValidator is not on PATH: the `device-compiler` cases then report [OK],
 * not a skip, and the suite exits 0. The "[skipped]" line it prints goes to the
 * per-case log alcotest does not surface on success. On a runner without the
 * tool the GLSL row is therefore SINGLETON-ONLY, and the two limits compose —
 * measured: with the mutation above and glslangValidator off PATH, all 12 cases
 * report [OK] and the suite exits 0 on a shader that does not compile.
 ******************************************************************************)

open Sarek_ir_types
module Wgsl = Sarek_codegen.Sarek_ir_wgsl
module Metal = Sarek_codegen.Sarek_ir_metal
module Cuda = Sarek_codegen.Sarek_ir_cuda
module Opencl = Sarek_codegen.Sarek_ir_opencl
module Glsl = Sarek_codegen.Sarek_ir_glsl
module SS = Set.Make (String)

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

(* Two float32 vector params, [a] read and [c] written; the body is a single
   assignment whose right-hand side is the only thing that varies. Keeping the
   body to one statement keeps the singleton sets small and the diff sharp. *)
let kernel_with rhs =
  let a = make_var "a" (TVec TFloat32) in
  let c = make_var "c" (TVec TFloat32) in
  {
    kern_name = "emitted_name_probe";
    kern_params =
      [
        DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (c, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ];
    kern_locals = [];
    kern_body = SAssign (LArrayElem ("c", EConst (CInt32 0l)), rhs);
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

(* ------------------------------------------------------------------------ *)
(* Identifier tokenising                                                     *)

let is_id_char ch =
  (ch >= 'a' && ch <= 'z')
  || (ch >= 'A' && ch <= 'Z')
  || (ch >= '0' && ch <= '9')
  || ch = '_'

(** Identifiers in [src], in order, with duplicates.

    Numeric literals are consumed whole — including any type suffix and any
    fractional part — rather than skipped a character at a time. Without that,
    WGSL's suffixed integers ([0i], [2u]) each contribute a bare [i]/[u]
    "identifier", whose singleton status then flips with the literal count and
    reports a free name on WGSL that is not there. Measured: it did. *)
let identifiers src =
  let n = String.length src in
  let out = ref [] and i = ref 0 in
  while !i < n do
    let ch = src.[!i] in
    if ch >= '0' && ch <= '9' then
      while !i < n && (is_id_char src.[!i] || src.[!i] = '.') do
        incr i
      done
    else if is_id_char ch then begin
      let s = !i in
      while !i < n && is_id_char src.[!i] do
        incr i
      done ;
      out := String.sub src s (!i - s) :: !out
    end
    else incr i
  done ;
  List.rev !out

(** Identifiers occurring exactly once in [src]. *)
let singletons src =
  let tbl = Hashtbl.create 64 in
  List.iter
    (fun id ->
      Hashtbl.replace tbl id (1 + try Hashtbl.find tbl id with Not_found -> 0))
    (identifiers src) ;
  Hashtbl.fold (fun k v acc -> if v = 1 then SS.add k acc else acc) tbl SS.empty

(* ------------------------------------------------------------------------ *)
(* The matrix                                                                *)

type backend = {
  label : string;
  gen : kernel -> string;
  validate : (string -> (unit, string) result) option;
      (** External compiler for this backend's output, when one exists and is
          the real oracle for "undeclared identifier". *)
}

let read_file path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let s = really_input_string ic n in
  close_in ic ;
  s

let run_validator ~exe ~ext ~args src =
  let base = Filename.temp_file "sarek_name_decl_" "" in
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

let glslang_ok src =
  run_validator
    ~exe:"glslangValidator"
    ~ext:".comp"
    ~args:"-V -S comp -o /dev/null"
    src

let plain label gen = {label; gen; validate = None}

(* Five of the six emitters. PTX is deliberately absent, on two grounds.

   The class is not expressible there: both halves already go through the one
   {!Sarek_ir_ptx_types.length_param_name} definition — Sarek_ir_ptx_kernel
   declares the length param with it, Sarek_ir_ptx_expr looks [EArrayLen] up with
   it — and anything it cannot name that way it refuses, raising [unsupported]
   ("only parameter arrays have a length") rather than emitting a guessed
   spelling. There is no second string construction to disagree with the first.

   And the instrument does not transfer: PTX is an assembly, where the singleton
   corollary's premise (an identifier is either declared or free) does not hold.
   Measured — adding PTX to this table makes the [EArrayLen] case fail with "PTX
   emits mov but declares no such name": a once-used INSTRUCTION MNEMONIC, not a
   free name. Covering PTX needs a different oracle, not this table. *)
let backends =
  [
    plain "CUDA" (fun k -> Cuda.generate_with_types ~types:[] k);
    plain "OpenCL" (fun k -> Opencl.generate_with_types ~types:[] k);
    plain "Metal" (fun k -> Metal.generate_with_types ~types:[] k);
    plain "WGSL" (fun k -> Wgsl.generate_with_types ~types:[] k);
    {
      label = "GLSL";
      gen = (fun k -> Glsl.generate_with_types ~types:[] k);
      validate = Some glslang_ok;
    };
  ]

(** A construct under test: [rhs] uses it, [baseline] does not, and the two are
    otherwise the same kernel. Anything [rhs] introduces that is used once and
    bound nowhere is a free identifier the emitter wrote. *)
type probe = {name : string; rhs : expr; baseline : expr}

let probes =
  [
    {
      (* backlog-156: GLSL declared [a_len] and used [sarek_a_length]. *)
      name = "EArrayLen of a vector parameter";
      rhs = ECast (TFloat32, EArrayLen "a");
      baseline = ECast (TFloat32, EConst (CInt32 0l));
    };
    {
      (* Control: a construct whose name IS declared on all five, so a red here
         means the instrument itself started reporting names that are bound. *)
      name = "EArrayRead of a vector parameter";
      rhs = EArrayRead ("a", EConst (CInt32 0l));
      baseline = EConst (CFloat32 0.0);
    };
  ]

let check_probe b p () =
  let src = b.gen (kernel_with p.rhs) in
  let base = b.gen (kernel_with p.baseline) in
  let free = SS.diff (singletons src) (singletons base) in
  if not (SS.is_empty free) then
    Alcotest.failf
      "%s emits %s but declares no such name.\n\n\
       Free identifier(s): %s\n\n\
       The emitter wrote %s into its output for %S and bound it nowhere: it is \
       used exactly once in the whole shader. Two spellings of one concept in \
       one emitter — declare it and use it through a single definition rather \
       than concatenating the name twice.\n\n\
       Generated source:\n\
       %s"
      b.label
      (String.concat ", " (SS.elements free))
      (String.concat ", " (SS.elements free))
      (String.concat ", " (SS.elements free))
      p.name
      src

(* The real oracle, where one is installed: the device compiler. The singleton
   check exists because it runs everywhere; this exists because it is the thing
   the singleton check is a proxy for. Skips (rather than passes) when the tool
   is absent, and says so. *)
let check_probe_validator b p () =
  match b.validate with
  | None -> ()
  | Some validate ->
      if not (Lazy.force glslang_available) then
        Printf.printf
          "  [skipped] glslangValidator not installed — %s/%S not compiled\n"
          b.label
          p.name
      else begin
        let src = b.gen (kernel_with p.rhs) in
        match validate src with
        | Ok () -> ()
        | Error out ->
            Alcotest.failf
              "%s output for %S does not compile.\n\n%s\nGenerated source:\n%s"
              b.label
              p.name
              out
              src
      end

let () =
  Alcotest.run
    "emitted names are declared"
    [
      ( "singleton-free",
        List.concat_map
          (fun b ->
            List.map
              (fun p ->
                Alcotest.test_case
                  (b.label ^ ": " ^ p.name)
                  `Quick
                  (check_probe b p))
              probes)
          backends );
      ( "device-compiler",
        List.concat_map
          (fun b ->
            if b.validate = None then []
            else
              List.map
                (fun p ->
                  Alcotest.test_case
                    (b.label ^ ": " ^ p.name)
                    `Quick
                    (check_probe_validator b p))
                probes)
          backends );
    ]
