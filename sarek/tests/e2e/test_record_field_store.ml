(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * In-place record-field store on a vector element: `v.(i).f <- e`.
 *
 * backlog-172. The construct is documented, is used by a shipped kernel
 * (p3_scale_y_kernel in test_soa_emitter_equiv), and worked on CUDA/PTX — while
 * the two CPU backends did something else entirely:
 *
 *   Interpreter: REFUSED, raising Unsupported_operation "record field
 *                assignment" / "not fully supported".
 *   Native:      ACCEPTED and silently dropped the store. The generated OCaml
 *                was a setfield on the fresh record Vector.get had just
 *                marshalled out of storage, so the write hit a temporary. No
 *                error on any path; the vector simply kept its old values.
 *
 * The Native half is the dangerous one, and it is why this test asserts on
 * EVERY available device rather than on the one that was broken loudly. A
 * silently-dropped store is indistinguishable from a kernel that did not run,
 * so the only thing that catches it is reading the values back and comparing.
 *
 * What is checked, per device:
 *   1. The written field holds the new value.
 *   2. The OTHER fields are untouched — a read-modify-write that rebuilt the
 *      record from defaults would satisfy (1) and destroy the rest.
 *   3. A store into the SECOND field of a mixed record lands in that field and
 *      not in the first, which is what a wrong field index looks like.
 *   4. A CHAINED target, v.(i).mid.b <- e, lands on the two CPU backends — with
 *      witnesses at BOTH levels. The depth-1 fix did not cover this: the
 *      interpreter read the intermediate record through the registry's copying
 *      get_field, and Native matched only the depth-1 shape, so both silently
 *      dropped it. Nesting is where "it works now" and "the shape I fixed works
 *      now" come apart.
 *
 *      On the C-family backends this case cannot run AT ALL yet, for a reason
 *      that has nothing to do with the store: the record typedefs are emitted
 *      in [kern_types] order with no dependency sort, so [outer] — which names
 *      [triple] as a field type — is written out BEFORE [triple] and the kernel
 *      does not compile. The dependency is emitted, just too late; both
 *      generators feed the same unsorted list through
 *      Sarek_ir_codegen.gen_record_typedefs / Sarek_ir_glsl.gen_record_def.
 *      Measured to affect a READ-ONLY nested kernel identically, so it is a
 *      struct-ordering gap, not a store gap (backlog-203).
 *
 *      That expectation is pinned in BOTH directions rather than skipped: the
 *      CPU backends must land the store, and a C-family backend must fail with
 *      exactly that error. A C-family device that started compiling it, or one
 *      that failed differently, is a change this test reports. The two C-family
 *      compilers word it completely differently — clang names the type, glslang
 *      reports a bare parse error and never echoes the identifier — so the pin
 *      is a two-clause disjunction; each clause states what it admits and what
 *      it excludes at the predicate.
 *
 * Every device failure makes the process exit non-zero.
 ******************************************************************************)

module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer

(* Explicit registration: linking a plugin does not enumerate its devices. *)
let () =
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init () ;
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_vulkan.Vulkan_plugin.init ()

type float32 = float

type ('a, 'b) vector = ('a, 'b) Vector.t

(* Three same-width fields: the store target plus two witnesses on either side,
   so a store that overruns in either direction is visible. *)
type triple = {a : float32; b : float32; c : float32} [@@sarek.type]

(* The SAME record with mutable fields, because Native's pre-fix behaviour was
   TWO different failures and only this one is silent.

   With immutable fields the old codegen did not compile at all: the emitted
   setfield produced "The record field b is not mutable" — loud, but
   misdiagnosed, since the problem was never mutability. That error is what
   pushed a user to add [mutable], and THEN the store compiled and was silently
   discarded (point3d in test_soa_emitter_equiv carries exactly that [mutable]
   with a comment describing it as necessary "to write a leaf in place").

   So the immutable case alone would prove-red as a build failure and never
   exercise the silent path. Both are pinned, and after the fix [mutable] is not
   required for either. *)
type mtriple = {
  mutable ma : float32;
  mutable mb : float32;
  mutable mc : float32;
}
[@@sarek.type]

(* Nested target. [outer] holds a whole [triple], so v.(i).mid.b <- e has to
   rebuild TWO levels on the way back into vector storage. The [tag] and the
   sibling fields of [mid] are witnesses: rebuilding either level from defaults
   satisfies "the target changed" and destroys something else. *)
type outer = {tag : float32; mid : triple} [@@sarek.type]

let scale_b =
  snd
    [%kernel
      fun (v : triple vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then v.(tid).b <- v.(tid).b *. 2.0]

let scale_mb =
  snd
    [%kernel
      fun (v : mtriple vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then v.(tid).mb <- v.(tid).mb *. 2.0]

let scale_nested =
  snd
    [%kernel
      fun (v : outer vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then v.(tid).mid.b <- v.(tid).mid.b *. 2.0]

(* No String.index_sub in the stdlib; spelled out so the message matches below
   are exact substring tests rather than looser regexes. Offsets (not just a
   bool) because the GLSL clause has to compare the POSITION of a type's use
   against the position of its declaration. *)
let find_sub (hay : string) (needle : string) : int option =
  let nh = String.length needle and hl = String.length hay in
  if nh = 0 then Some 0
  else
    let rec go i =
      if i + nh > hl then None
      else if String.equal (String.sub hay i nh) needle then Some i
      else go (i + 1)
    in
    go 0

let contains_sub (hay : string) (needle : string) : bool =
  Option.is_some (find_sub hay needle)

let read_file (path : string) : string option =
  try
    let ic = open_in_bin path in
    let s = really_input_string ic (in_channel_length ic) in
    close_in ic ;
    Some s
  with _ -> None

(* The shader path glslangValidator echoed. It appears twice in the log — once
   alone on its own line, once as the "<path>:<line>:" prefix — and this picks
   the bare occurrence by requiring the whole whitespace-delimited token to end
   in ".comp". Returns None when the log carries no such token, which is the
   case whenever Vulkan compiled through libshaderc instead of the CLI. *)
let glsl_shader_path (msg : string) : string option =
  String.split_on_char '\n' msg
  |> List.concat_map (String.split_on_char ' ')
  |> List.find_opt (fun tok ->
      let n = String.length tok in
      n > 6
      && Char.equal tok.[0] '/'
      && String.equal (String.sub tok (n - 5) 5) ".comp")

(* Does [src] USE the type name before it DECLARES it? This is the textual
   signature of backlog-203: the record typedefs are emitted in [kern_types]
   order with no dependency sort, so the enclosing struct — which names the
   inner one as a field type — is written out first. *)
let uses_type_before_declaring (src : string) (ty : string) : bool =
  match (find_sub src ty, find_sub src ("struct " ^ ty)) with
  | Some use, Some decl -> use < decl
  | _ -> false

(* backlog-203. The ONLY tolerated failure is the struct-ordering gap,
   and it has to be recognised in two different dialects because the two
   C-family compilers describe it with completely different words.
   Tolerated on the C-family backends only — on Interpreter or Native it
   would mean the fix regressed.

   Clause A — clang-family diagnostics (OpenCL here; CUDA/HIP/Metal
   would land here too). Verbatim, from the OpenCL log:

     input.cl:3:3: error: unknown type name
     'Test_record_field_store_triple'

   ADMITS: a diagnostic that both says "unknown type name" and names the
   inner struct's emitted symbol. EXCLUDES: every other clang
   diagnostic, and an unknown-type error about any OTHER type — which is
   what a store lowered to a bogus type would look like.

   Clause B — glslang. Its message is the reason the original predicate
   read Vulkan as a hard failure: glslang never echoes the offending
   identifier. Verbatim, from the Vulkan log:

     ERROR: /tmp/sarek_deea56.comp:8: '' :  syntax error, unexpected
     IDENTIFIER, expecting RIGHT_BRACE

   There is nothing in that text tying it to this gap, so the message
   alone cannot carry the pin and clause B is a conjunction of two:

     B1 the glslang parse-error shape an undeclared type produces when
        used as a struct member: the parser is inside a brace body and
        meets an IDENTIFIER where a `}` or a known type keyword must be.
        ADMITS only that one wording. EXCLUDES every semantic error
        (including glslang's own "undeclared identifier"), every other
        syntax error, and both link-stage errors — the "Missing entry
        point" line is downstream of this one and is not matched.

     B2 the shader source itself — recovered from the path glslang
        printed, which is still on disk because the Vulkan CLI path
        raises before it unlinks the .comp on a failed compile — uses
        Test_record_field_store_triple at an earlier byte offset than it
        declares it. ADMITS only a shader with THIS type emitted after
        its user. EXCLUDES a syntax error in a shader whose structs are
        in dependency order, a syntax error about some other type, and
        the case where the shader cannot be read at all: B2 is then
        false and the device FAILS, because an unsubstantiated claim
        must not pass as a substantiated one.

   What B1 ∧ B2 does NOT exclude, stated rather than papered over: a
   second, unrelated glslang parse error of exactly that shape occurring
   in a shader that ALSO has the ordering problem. Binding the reported
   line number to the line of the first use would close that, and is not
   worth the parsing until it happens. *)
let is_backlog203_struct_gap (msg : string) : bool =
  let clause_a =
    contains_sub msg "unknown type name"
    && contains_sub msg "Test_record_field_store_triple"
  in
  let clause_b1 =
    contains_sub
      msg
      "syntax error, unexpected IDENTIFIER, expecting RIGHT_BRACE"
  in
  let clause_b2 =
    match Option.bind (glsl_shader_path msg) read_file with
    | None -> false
    | Some src ->
        uses_type_before_declaring src "Test_record_field_store_triple"
  in
  let clause_b = clause_b1 && clause_b2 in
  clause_a || clause_b

(* Both polarities of {!is_backlog203_struct_gap}, pinned mechanically rather
   than in prose. A tolerated-failure predicate is exactly the shape that rots
   into "any failure passes": the accept cases would still hold if the whole
   predicate were [fun _ -> true], so the REJECT cases are what actually
   constrains it, and the two accept cases alone would have let the widening go
   unnoticed. The two shaders are assembled here at runtime — the accept case
   needs a real file on disk for clause B2 to read, and a fixture checked in
   beside the test could drift from the wording it is supposed to match.

   Runs before any device: a broken predicate must not be discovered halfway
   through a device sweep. *)
let () =
  let write suffix contents =
    let path = Filename.temp_file "sarek_selfcheck_" suffix in
    let oc = open_out path in
    output_string oc contents ;
    close_out oc ;
    path
  in
  (* Emitted the way backlog-203 emits it: user before dependency. *)
  let bad_order =
    write
      ".comp"
      "#version 450\n\
       struct Test_record_field_store_outer {\n\
      \  float tag;\n\
      \  Test_record_field_store_triple mid;\n\
       };\n\
       struct Test_record_field_store_triple {\n\
      \  float a;\n\
       };\n"
  in
  (* The same two structs in dependency order — what a fixed emitter writes. *)
  let good_order =
    write
      ".comp"
      "#version 450\n\
       struct Test_record_field_store_triple {\n\
      \  float a;\n\
       };\n\
       struct Test_record_field_store_outer {\n\
      \  float tag;\n\
      \  Test_record_field_store_triple mid;\n\
       };\n"
  in
  let glslang_syntax_error path =
    Printf.sprintf
      "[Vulkan Runtime] Compilation failed for:\n\n\n\
       Compiler log:\n\
       glslangValidator failed:\n\
       %s\n\
       ERROR: %s:4: '' :  syntax error, unexpected IDENTIFIER, expecting \
       RIGHT_BRACE\n\
       ERROR: 1 compilation errors.  No code generated.\n"
      path
      path
  in
  let cases =
    [
      (* ACCEPT — clause A, the OpenCL log verbatim. *)
      ( true,
        "opencl: verbatim",
        "[OpenCL Runtime] Compilation failed for:\n\
         OpenCL kernel\n\n\
         Compiler log:\n\
         input.cl:3:3: error: unknown type name 'Test_record_field_store_triple'\n\
        \    3 |   Test_record_field_store_triple mid;\n\
        \      |   ^\n\
         Error executing LLVM compilation action.\n" );
      (* ACCEPT — clause B, the Vulkan log verbatim, shader present and
         out of order. *)
      (true, "vulkan: verbatim", glslang_syntax_error bad_order);
      (* REJECT — clause A's wording about a DIFFERENT type. This is what a
         store lowered against a bogus record type would look like, and it must
         not hide behind backlog-203. *)
      ( false,
        "opencl: unknown type name, other type",
        "input.cl:3:3: error: unknown type name 'Test_record_field_store_quad'"
      );
      (* REJECT — clause A's type name in some OTHER clang diagnostic. *)
      ( false,
        "opencl: other diagnostic naming the type",
        "input.cl:9:5: error: no member named 'q' in \
         'Test_record_field_store_triple'" );
      (* REJECT — glslang's OTHER way of reporting an undeclared name. Close
         enough to the tolerated one to be worth pinning: same root cause
         family, different message, so B1 must not admit it. *)
      ( false,
        "vulkan: undeclared identifier",
        Printf.sprintf
          "glslangValidator failed:\n\
           %s\n\
           ERROR: %s:4: 'Test_record_field_store_triple' : undeclared identifier\n"
          bad_order
          bad_order );
      (* REJECT — B1's exact wording, but the shader declares its structs in
         dependency order, so the parse error is about something else. *)
      ( false,
        "vulkan: syntax error, shader well-ordered",
        glslang_syntax_error good_order );
      (* REJECT — B1's exact wording, but the shader is gone. Unverifiable is
         not the same as verified. *)
      ( false,
        "vulkan: syntax error, shader unreadable",
        glslang_syntax_error "/tmp/sarek_selfcheck_absent_00000.comp" );
      (* REJECT — a refusal from somewhere else entirely. *)
      ( false,
        "unrelated refusal",
        "Unsupported_operation(\"record field assignment\")" );
    ]
  in
  let bad =
    List.filter
      (fun (want, _, msg) ->
        not (Bool.equal (is_backlog203_struct_gap msg) want))
      cases
  in
  List.iter (fun p -> try Sys.remove p with _ -> ()) [bad_order; good_order] ;
  if bad <> [] then begin
    List.iter
      (fun (want, label, _) ->
        Printf.printf
          "backlog-203 predicate self-check: %s should be %b and is not\n%!"
          label
          want)
      bad ;
    exit 1
  end ;
  Printf.printf
    "backlog-203 predicate self-check: %d case(s) OK (%d accept, %d reject)\n%!"
    (List.length cases)
    (List.length (List.filter (fun (w, _, _) -> w) cases))
    (List.length (List.filter (fun (w, _, _) -> not w) cases))

let ir_of kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let n = 64

let orig i =
  {a = float_of_int i; b = float_of_int (i + 1); c = float_of_int (i + 2)}

let morig i =
  {ma = float_of_int i; mb = float_of_int (i + 1); mc = float_of_int (i + 2)}

(* One checker over both record shapes. [read] returns the three field values in
   (target, witness1, witness2) order so the assertions below are shape-agnostic
   and cannot drift apart between the two cases. *)
let run_case (dev : Device.t) ~(label : string) ~kernel ~make ~read : bool =
  Printf.printf
    "field-store %-9s [%s] %s: %!"
    label
    dev.Device.framework
    dev.Device.name ;
  let v = make () in
  match
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir:(ir_of kernel)
      ~args:[Vec v; Int n]
      ~block:(Sarek.Execute.dims1d (min 64 n))
      ~grid:(Sarek.Execute.dims1d ((n + 63) / 64))
      ()
  with
  | exception e ->
      (* A refusal is a FAILURE here, not a skip. The construct is part of the
         DSL and every backend in this list executes ordinary custom-vector
         kernels; a backend that cannot do this one is the defect. *)
      Printf.printf "FAILED (raised: %s)\n%!" (Printexc.to_string e) ;
      false
  | () ->
      Transfer.flush dev ;
      let ok = ref true in
      let reported = ref 0 in
      for i = 0 to n - 1 do
        let tgt, w1, w2 = read v i in
        let bad name got want =
          if Float.abs (got -. want) > 1e-4 then begin
            ok := false ;
            if !reported < 3 then begin
              incr reported ;
              Printf.printf "\n  @%d field %s: got %g want %g" i name got want
            end
          end
        in
        (* The written field doubled; the two witnesses untouched. Checking the
           witnesses is not padding: a read-modify-write that rebuilt the record
           from defaults would satisfy the first assertion and destroy the
           rest. *)
        bad "target" tgt (float_of_int (i + 1) *. 2.0) ;
        bad "witness 1 (must be untouched)" w1 (float_of_int i) ;
        bad "witness 2 (must be untouched)" w2 (float_of_int (i + 2))
      done ;
      if !ok then Printf.printf "OK\n%!" else Printf.printf "\n  FAILED\n%!" ;
      !ok

let run_on (dev : Device.t) : bool =
  let immutable_ok =
    run_case
      dev
      ~label:"immutable"
      ~kernel:scale_b
      ~make:(fun () ->
        let v = Vector.create_custom triple_custom n in
        for i = 0 to n - 1 do
          Vector.set v i (orig i)
        done ;
        v)
      ~read:(fun v i ->
        let p = Vector.get v i in
        (p.b, p.a, p.c))
  in
  let mutable_ok =
    run_case
      dev
      ~label:"mutable"
      ~kernel:scale_mb
      ~make:(fun () ->
        let v = Vector.create_custom mtriple_custom n in
        for i = 0 to n - 1 do
          Vector.set v i (morig i)
        done ;
        v)
      ~read:(fun v i ->
        let p = Vector.get v i in
        (p.mb, p.ma, p.mc))
  in
  (* The nested case carries its own checker: [run_case] reports one target and
     two witnesses, and this needs three witnesses across two levels. It is also
     the only case with a per-family expectation (see backlog-203 in the header),
     so it cannot reuse [run_case]'s "a refusal is always a failure" rule. *)
  let nested_ok =
    Printf.printf
      "field-store %-9s [%s] %s: %!"
      "nested"
      dev.Device.framework
      dev.Device.name ;
    let v = Vector.create_custom outer_custom n in
    for i = 0 to n - 1 do
      Vector.set v i {tag = float_of_int (1000 + i); mid = orig i}
    done ;
    match
      Sarek.Execute.run_vectors
        ~device:dev
        ~ir:(ir_of scale_nested)
        ~args:[Vec v; Int n]
        ~block:(Sarek.Execute.dims1d (min 64 n))
        ~grid:(Sarek.Execute.dims1d ((n + 63) / 64))
        ()
    with
    | exception e ->
        let msg = Printexc.to_string e in
        let is_struct_gap = is_backlog203_struct_gap msg in
        let c_family =
          match dev.Device.framework with
          | "Interpreter" | "Native" -> false
          | _ -> true
        in
        if is_struct_gap && c_family then begin
          Printf.printf
            "known gap (backlog-203: nested struct emitted after its user) — \
             expected\n\
             %!" ;
          true
        end
        else begin
          Printf.printf "FAILED (raised: %s)\n%!" msg ;
          false
        end
    | () ->
        if
          match dev.Device.framework with
          | "Interpreter" | "Native" -> false
          | _ -> true
        then
          (* Not a failure — the C-family gap is fixed and this test's
             expectation is now stale. Say so loudly rather than quietly
             passing, because the header claims this cannot compile here. *)
          Printf.printf
            "\n\
            \  NOTE: compiled on %s, so backlog-203 is fixed — drop the \
             known-gap branch above\n\
             %!"
            dev.Device.framework ;
        Transfer.flush dev ;
        let ok = ref true in
        let reported = ref 0 in
        for i = 0 to n - 1 do
          let p = Vector.get v i in
          let bad name got want =
            if Float.abs (got -. want) > 1e-4 then begin
              ok := false ;
              if !reported < 3 then begin
                incr reported ;
                Printf.printf "\n  @%d %s: got %g want %g" i name got want
              end
            end
          in
          bad "mid.b (target)" p.mid.b (float_of_int (i + 1) *. 2.0) ;
          (* Same level as the target. *)
          bad "mid.a (untouched)" p.mid.a (float_of_int i) ;
          bad "mid.c (untouched)" p.mid.c (float_of_int (i + 2)) ;
          (* OUTER level: a rebuild that dropped the enclosing record would show
             up here and nowhere else. *)
          bad "tag (untouched, outer level)" p.tag (float_of_int (1000 + i))
        done ;
        if !ok then Printf.printf "OK\n%!" else Printf.printf "\n  FAILED\n%!" ;
        !ok
  in
  immutable_ok && mutable_ok && nested_ok

let () =
  let devs =
    Device.init
      ~frameworks:
        ["Interpreter"; "Native"; "CUDA"; "OpenCL"; "Vulkan"; "Metal"; "HIP"]
      ()
  in
  if Array.length devs = 0 then begin
    print_endline "No devices found — nothing asserted, and that is a gap" ;
    (* Exit non-zero: a run that asserted nothing must not read as a pass. *)
    exit 1
  end ;
  let any_failure = ref false in
  Array.iter (fun dev -> if not (run_on dev) then any_failure := true) devs ;
  Printf.printf "%d device(s) exercised\n%!" (Array.length devs) ;
  if !any_failure then exit 1
