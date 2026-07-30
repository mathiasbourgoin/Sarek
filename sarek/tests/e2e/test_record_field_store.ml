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
 *   4. A CHAINED target, v.(i).mid.b <- e, lands — with witnesses at BOTH levels
 *      — on every device that can compile it, which is both CPU backends, both
 *      CUDA/PTX devices, and (for the case whose structs are emitted in
 *      dependency order) OpenCL and Vulkan too. The depth-1 fix did not cover
 *      this: the interpreter read the intermediate record through the registry's
 *      copying get_field, and Native matched only the depth-1 shape, so both
 *      silently dropped it. Nesting is where "it works now" and "the shape I
 *      fixed works now" come apart.
 *
 *      One of the two nested cases does not COMPILE on OpenCL or Vulkan, for a
 *      reason that has nothing to do with the store: the record typedefs are
 *      emitted in [kern_types] order with no dependency sort, so an enclosing
 *      struct can be written out BEFORE the struct it names as a field type, and
 *      the kernel does not compile. The dependency is emitted, just too late;
 *      both generators feed the same unsorted list through
 *      Sarek_ir_codegen.gen_record_typedefs / Sarek_ir_glsl.gen_record_def.
 *      Measured to affect a READ-ONLY nested kernel identically, so it is a
 *      struct-ordering gap, not a store gap (backlog-203).
 *
 *      "Nested records do not compile on the C family" would be the tidy
 *      statement and it is FALSE. Whether the gap bites depends on the order
 *      that unsorted list happens to hold, and the two nested cases here land on
 *      opposite sides of it: [outer] is emitted before [triple] (does not
 *      compile on OpenCL or Vulkan), [mouter] is emitted after [mtriple]
 *      (compiles, runs and passes on both). So the expectation is not assumed
 *      per backend — the emitted source is inspected per kernel, and a compile
 *      failure is tolerated only when that source genuinely uses the inner
 *      struct before declaring it (see [predict_struct_gap]).
 *
 *      CUDA/PTX does NOT have the gap at all: the direct PTX emitter declares no
 *      C structs, so there is no declaration order to get wrong. Measured under
 *      ZLUDA on an RX 7900 XTX and on a Ryzen 9 7950X (ZLUDA-on-AMD both times;
 *      there is no NVIDIA hardware on this machine) — both CUDA/PTX devices
 *      compile BOTH nested kernels and produce correct values at both levels.
 *      That measurement is what the dune rule's LD_LIBRARY_PATH exists for: the
 *      gate enumerated 7 devices and printed no CUDA/PTX row at all, so the
 *      sentence above was not reproducible by the test that states it. The run
 *      now also prints a named NOT-MEASURED-HERE line for any framework this
 *      header makes a claim about and the run did not enumerate.
 *      Metal and HIP are untested here (no such device on this machine), so
 *      nothing is claimed about them; they are not on the allowlist, and a
 *      failure on either is reported as a failure.
 *
 *      The expectation is pinned in BOTH directions rather than skipped: a
 *      kernel whose emitted structs are in dependency order must land the store
 *      on every backend, and one whose structs are out of order must fail on the
 *      struct-emitting backends with exactly that error. A device that starts
 *      compiling the out-of-order source, or that fails differently, is a change
 *      this test reports. The two C-family compilers word the gap completely
 *      differently — clang names the type, glslang reports a bare parse error and
 *      never echoes the identifier — so the pin is a two-clause disjunction; each
 *      clause states what it admits and what it excludes at the predicate.
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

(* The nested target with a MUTABLE leaf, for exactly the reason [mtriple] exists
   at depth 1 — and it was missing, so the committed nested case only ever
   exercised the LOUD half.

   Measured: revert only the nested half of the fix and the [outer] kernel does
   not build ("The record field \"b\" is not mutable"), because the chained
   lvalue falls through to a setfield on the record [Vector.get] marshalled out.
   That is a build failure, not the silent drop the header describes. With a
   mutable leaf the same revert COMPILES and silently drops the store — measured
   "got 1 want 2" on Native at both levels. Both halves are now committed. *)
type mouter = {mtag : float32; mmid : mtriple} [@@sarek.type]

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

let scale_nested_mut =
  snd
    [%kernel
      fun (v : mouter vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then v.(tid).mmid.mb <- v.(tid).mmid.mb *. 2.0]

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

(* Where [src] DECLARES [ty], in either dialect this test has to read:

     GLSL / the .comp fixtures    struct Ty { ... };
     OpenCL C and CUDA C          typedef struct { ... } Ty;

   Both forms are needed and having only the GLSL one is a way this can read
   green while checking nothing: the OpenCL generator emits the anonymous-typedef
   form, so "struct Ty" is absent from its output, the declaration is never
   found, and a genuinely out-of-order source reports as in-order.

   That hazard was named here and then left uncovered: every source the
   message-predicate self-check below assembles is GLSL, so reverting to the
   GLSL-only form kept printing "11 case(s) OK" and the regression appeared only
   on a machine with a live OpenCL device. The self-check attached to
   {!uses_type_before_declaring} covers BOTH spellings without a device. *)
let declaration_offset (src : string) (ty : string) : int option =
  match (find_sub src ("struct " ^ ty), find_sub src ("} " ^ ty ^ ";")) with
  | Some a, Some b -> Some (min a b)
  | Some a, None -> Some a
  | None, Some b -> Some b
  | None, None -> None

(* Does [src] USE the type name before it DECLARES it? This is the textual
   signature of backlog-203: the record typedefs are emitted in [kern_types]
   order with no dependency sort, so an enclosing struct — which names the inner
   one as a field type — can be written out first. Whether it IS depends on the
   order that list happens to hold, which is why this is measured per kernel
   rather than assumed per backend. *)
let uses_type_before_declaring (src : string) (ty : string) : bool =
  match (find_sub src ty, declaration_offset src ty) with
  | Some use, Some decl -> use < decl
  | _ -> false

(* Both polarities of {!uses_type_before_declaring} / {!declaration_offset}, over
   synthetic sources, with NO device involved.

   This exists because the two-dialect [declaration_offset] landed behind a gate
   that could not fail without a live OpenCL device. Revert it to the GLSL-only
   form — drop the [find_sub src ("} " ^ ty ^ ";")] arm — and the 11-case
   predicate self-check below still printed "11 case(s) OK": every shader it
   assembles is GLSL, so the typedef arm was never reached, and the regression
   surfaced only on a machine that enumerates an OpenCL device. The neighbouring
   MESSAGE predicate got 11 device-free cases and this one got zero.

   So case 3 is the load-bearing one here, and it is the ONLY one of the four:
   measured by replacing the typedef arm with [None], cases 1, 2 AND 4 still
   pass and only case 3 goes red ("opencl: typedef struct } Ty;, used before
   declared should be true and is not"). Case 4 cannot constrain that arm — with
   the arm gone [declaration_offset ocl_good] finds nothing,
   [uses_type_before_declaring] falls to its [| _ -> false] arm, and [false] is
   the answer case 4 wanted. It still rules out a [fun _ _ -> true] predicate on
   this dialect, which is what its own comment claims and all it claims.

   That leaves the arm constrained in one direction only, so the FOUND-IT
   direction is asserted separately and first, on [declaration_offset] itself
   rather than through the offset comparison — see the block below.

   Runs before the message-predicate self-check and before any device, for the
   same reason that one does: a broken source-inspection predicate must not be
   discovered halfway through a device sweep. *)
let () =
  (* The real emitted symbols, so a case cannot pass on a name shape the
     generators never produce. [..._outer] does not contain [..._triple], which
     is what makes "first occurrence of the needle" a use of the inner type. *)
  let inner = "Test_record_field_store_triple" in
  let outer = "Test_record_field_store_outer" in
  (* GLSL, the .comp dialect: `struct Ty { ... };`. *)
  let glsl_bad =
    Printf.sprintf
      "#version 450\n\
       struct %s {\n\
      \  float tag;\n\
      \  %s mid;\n\
       };\n\
       struct %s {\n\
      \  float a;\n\
       };\n"
      outer
      inner
      inner
  in
  let glsl_good =
    Printf.sprintf
      "#version 450\n\
       struct %s {\n\
      \  float a;\n\
       };\n\
       struct %s {\n\
      \  float tag;\n\
      \  %s mid;\n\
       };\n"
      inner
      outer
      inner
  in
  (* OpenCL C and CUDA C, the anonymous-typedef dialect: `typedef struct { ... }
     Ty;`. The string "struct Ty" NEVER appears, which is exactly why the
     GLSL-only form reported this out-of-order source as in-order. *)
  let ocl_bad =
    Printf.sprintf
      "typedef struct {\n\
      \  float tag;\n\
      \  %s mid;\n\
       } %s;\n\
       typedef struct {\n\
      \  float a;\n\
       } %s;\n"
      inner
      outer
      inner
  in
  let ocl_good =
    Printf.sprintf
      "typedef struct {\n\
      \  float a;\n\
       } %s;\n\
       typedef struct {\n\
      \  float tag;\n\
      \  %s mid;\n\
       } %s;\n"
      inner
      inner
      outer
  in
  (* The typedef arm, asserted DIRECTLY and in the FOUND-IT direction, before
     the case list below.

     [uses_type_before_declaring] answers a comparison of two offsets, so every
     [false] case of it conflates "the declaration was found, and it comes
     first" with "no declaration was found at all" — which is exactly why case 4
     survives the arm's removal and why the four cases together pin the arm in
     one direction only. This asserts the offset itself: in the anonymous-typedef
     dialect the declaration must be LOCATED in both sources, in-order and
     out-of-order alike, and neither may answer [None].

     FIRST, not last, so it has its own red: the case loop below exits 1 on case
     3 the moment the arm is gone, and an assertion placed after it would never
     run to be observed failing.

     BOTH sources reported before exiting, not the first one only. Exiting inside
     the loop would make the in-order source — the one carrying the coverage case
     4 cannot give — unobservable behind the out-of-order source, which is the
     same short-circuit this block was added to escape one level up. *)
  let unlocated =
    List.filter
      (fun (_, src) -> declaration_offset src inner = None)
      [
        ("opencl: out-of-order source", ocl_bad);
        ("opencl: in-order source", ocl_good);
      ]
  in
  if unlocated <> [] then begin
    List.iter
      (fun (label, _) ->
        Printf.printf
          "declaration_offset self-check: %s — the anonymous-typedef \
           declaration of %s was not located at all, so the `} Ty;` arm is \
           missing or narrowed\n\
           %!"
          label
          inner)
      unlocated ;
    exit 1
  end ;
  Printf.printf
    "declaration_offset self-check: the OpenCL typedef arm located %s in both \
     typedef sources\n\
     %!"
    inner ;
  let cases =
    [
      (* 1. GLSL, enclosing struct emitted first: the gap. *)
      (true, "glsl: struct Ty, used before declared", glsl_bad, inner);
      (* 2. GLSL, dependency order: no gap. *)
      (false, "glsl: struct Ty, declared before used", glsl_good, inner);
      (* 3. OpenCL typedef, enclosing struct emitted first: the gap, in the
         spelling the OpenCL generator actually emits. THIS is the case the
         GLSL-only form gets wrong — it finds no declaration at all and answers
         false, so a genuinely out-of-order OpenCL source reads as in-order and
         a real compile failure stops being tolerated (or, in
         [predict_struct_gap], a tolerated one stops being predicted). *)
      ( true,
        "opencl: typedef struct } Ty;, used before declared",
        ocl_bad,
        inner );
      (* 4. OpenCL typedef, dependency order: no gap. Pinned as well as (3):
         with only (3) the predicate could be [fun _ _ -> true] on this
         dialect. *)
      ( false,
        "opencl: typedef struct } Ty;, declared before used",
        ocl_good,
        inner );
      (* 5. Used and NEVER declared, both dialects. FALSE is the answer this
         must give, and it is the fail-closed direction rather than an
         oversight: backlog-203 is a mis-ORDERING of a declaration that IS
         emitted, so a source that never declares the type at all is a
         DIFFERENT defect — a dropped typedef — and must not be tolerated as
         the known gap. Answering true here would let a generator that stopped
         emitting the struct entirely hide behind backlog-203. *)
      ( false,
        "glsl: used, never declared",
        Printf.sprintf "#version 450\nstruct %s {\n  %s mid;\n};\n" outer inner,
        inner );
      ( false,
        "opencl: used, never declared",
        Printf.sprintf "typedef struct {\n  %s mid;\n} %s;\n" inner outer,
        inner );
      (* 6. Declared and never used: no gap. *)
      ( false,
        "glsl: declared, never used",
        Printf.sprintf "#version 450\nstruct %s {\n  float a;\n};\n" inner,
        inner );
      ( false,
        "opencl: declared, never used",
        Printf.sprintf "typedef struct {\n  float a;\n} %s;\n" inner,
        inner );
      (* 7. The type is absent from the source entirely — a PTX-style emitter
         that declares no structs. No prediction, so no gap. *)
      (false, "no structs at all", "#version 450\nvoid main() {}\n", inner);
      (* 8. A LONGER name containing the needle must not stand in for it: the
         source declares and uses only [..._triple_extra], and nothing is
         claimed about [..._triple]. [find_sub] is a substring search, so the
         use offset and the declaration offset both land inside the longer name
         and the comparison comes out false either way — pinned so that a future
         "search for the bare name" rewrite that answers true here is caught. *)
      ( false,
        "glsl: only a longer name containing the needle",
        Printf.sprintf
          "#version 450\n\
           struct %s_extra {\n\
          \  float a;\n\
           };\n\
           struct %s {\n\
          \  %s_extra mid;\n\
           };\n"
          inner
          outer
          inner,
        inner );
    ]
  in
  let bad =
    List.filter
      (fun (want, _, src, ty) ->
        not (Bool.equal (uses_type_before_declaring src ty) want))
      cases
  in
  if bad <> [] then begin
    List.iter
      (fun (want, label, _, _) ->
        Printf.printf
          "uses_type_before_declaring self-check: %s should be %b and is not\n\
           %!"
          label
          want)
      bad ;
    exit 1
  end ;
  Printf.printf
    "uses_type_before_declaring self-check: %d case(s) OK (%d gap, %d no-gap; \
     %d GLSL-spelling, %d OpenCL-typedef-spelling)\n\
     %!"
    (List.length cases)
    (List.length (List.filter (fun (w, _, _, _) -> w) cases))
    (List.length (List.filter (fun (w, _, _, _) -> not w) cases))
    (List.length
       (List.filter (fun (_, l, _, _) -> contains_sub l "glsl:") cases))
    (List.length
       (List.filter (fun (_, l, _, _) -> contains_sub l "opencl:") cases))

(* backlog-203. The ONLY tolerated failure is the struct-ordering gap,
   and it has to be recognised in two different dialects because the two
   C-family compilers describe it with completely different words.
   Tolerated on the two backends MEASURED to have the gap and nowhere
   else (see [gap_frameworks]) — on any other backend it would mean
   either the fix regressed or a new backend acquired the gap silently.

   Clause A — clang-family diagnostics (OpenCL here). Verbatim, from the
   OpenCL log:

     input.cl:3:3: error: unknown type name
     'Test_record_field_store_triple'

   ADMITS: a log whose clang error diagnostics are ALL of that one
   shape — "unknown type name" and the inner struct's emitted symbol, in
   the SAME diagnostic line. EXCLUDES: a log carrying any other clang
   error diagnostic ALONGSIDE this one (the tolerated wording must not
   be a licence for whatever else the compiler said — a real store
   regression reads "input.cl:17:5: error: expression is not
   assignable"), every other clang diagnostic on its own, an unknown-type
   error about any OTHER type, and a log where the symbol appears only in
   a note or a caret line while the error names something else.

   Both exclusions are properties of the CODE, not of this comment: the
   predicate binds the wording to the symbol within one line, and
   quantifies over every error line rather than searching the whole log.
   An earlier version was a substring conjunction over the whole text and
   excluded NEITHER; the two attack logs it admitted are reject cases in
   the self-check below.

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
        ADMITS only a log whose glslang diagnostics are ALL that one
        wording. EXCLUDES every semantic error (including glslang's own
        "undeclared identifier"), every other syntax error, any other
        diagnostic reported alongside this one, and the link-stage tail:
        the "N compilation errors" / "No code generated" / "Missing entry
        point" lines are downstream of the first error and carry no
        diagnostic of their own, so they are removed from the set being
        quantified over rather than admitted — a log containing ONLY such
        a tail has no diagnostic at all and B1 is false.

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

let lines (msg : string) : string list = String.split_on_char '\n' msg

(* The clang DIAGNOSTIC lines: `<file>:<line>:<col>: error: <text>`. Matching on
   ": error: " keeps out the source echo and caret lines, the `note:` lines, the
   warnings, and clang's "Error executing LLVM compilation action." tail — none
   of which is a diagnostic about a symbol. *)
let clang_error_lines (msg : string) : string list =
  List.filter (fun l -> contains_sub l ": error: ") (lines msg)

(* The glslang DIAGNOSTIC lines, minus the link-stage tail (see clause B1). *)
let glslang_error_lines (msg : string) : string list =
  lines msg
  |> List.filter (fun l -> contains_sub l "ERROR: ")
  |> List.filter (fun l ->
      not
        (contains_sub l "compilation errors"
        || contains_sub l "No code generated"
        || contains_sub l "Missing entry point"))

(* The emitted symbol for the inner struct, quoted the way both compilers quote
   an identifier, so "names this type" cannot be satisfied by a longer name that
   merely CONTAINS it — which is not hypothetical here: the mutable-leaf nested
   case's inner struct is [..._mtriple], and an unquoted needle would let a
   diagnostic about one stand in for the other. *)
let quoted (ty : string) = "'" ^ ty ^ "'"

(* WHEN THIS PREDICATE MAY BE DELETED, and why not yet.

   PR #394 (the dependency sort in [Sarek_ir_codegen]) asks for this predicate
   and the sibling tolerance in [test_record_local_alias_agreement.ml] to go,
   on the grounds that it fixes the gap they tolerate. That is the right
   follow-up in the wrong order, and the order is the whole content of this
   note: #394 is OPEN and its fix is on NO branch merged into `main`
   ([sort_record_types_by_dependency] does not occur in `origin/main`), and the
   gap is still measured live here — 4 device rows (OpenCL ×2, Vulkan ×2) took
   the known-gap branch on the rebase of this PR onto current `main`.

   Deleting it before #394 lands would not surface the gap, it would hide it in
   the one place nobody looks: CI enumerates no GPU, so those 4 rows do not run
   there at all. CI would stay green while every OpenCL or Vulkan host went red
   — a deletion whose only observable effect is on the machines that are not
   watching. So the deletion is gated on #394 being merged, not on #376 landing,
   and it is not performed here. *)

(* [~inner] is the emitted struct symbol of the type that is used before it is
   declared — one per nested record shape under test. *)
let is_backlog203_struct_gap ~(inner : string) (msg : string) : bool =
  let clause_a =
    match clang_error_lines msg with
    | [] -> false
    | errs ->
        List.for_all
          (fun l ->
            contains_sub l "unknown type name" && contains_sub l (quoted inner))
          errs
  in
  let clause_b1 =
    match glslang_error_lines msg with
    | [] -> false
    | errs ->
        List.for_all
          (fun l ->
            contains_sub
              l
              "syntax error, unexpected IDENTIFIER, expecting RIGHT_BRACE")
          errs
  in
  let clause_b2 =
    match Option.bind (glsl_shader_path msg) read_file with
    | None -> false
    | Some src -> uses_type_before_declaring src inner
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
      (* REJECT — ATTACK 1. The tolerated ordering error AND a genuine store
         regression in the SAME log. The whole-log substring conjunction this
         predicate used to be admitted this: both of its needles are present, so
         a real "expression is not assignable" hid behind backlog-203. The
         tolerated wording licenses only itself, never whatever else the
         compiler said. *)
      ( false,
        "opencl: ordering error PLUS a store regression",
        "[OpenCL Runtime] Compilation failed for:\n\
         OpenCL kernel\n\n\
         Compiler log:\n\
         input.cl:3:3: error: unknown type name 'Test_record_field_store_triple'\n\
        \    3 |   Test_record_field_store_triple mid;\n\
        \      |   ^\n\
         input.cl:17:5: error: expression is not assignable\n\
        \   17 |     v[tid].mid.b = 2.0f;\n\
        \      |     ~~~~~~~~~~~~ ^\n\
         Error executing LLVM compilation action.\n" );
      (* REJECT — ATTACK 2. An unknown-type error about a DIFFERENT type, with
         this test's symbol appearing only in a note. Also admitted by the
         whole-log conjunction, because both needles were somewhere in the text;
         the diagnostic and the symbol must be the same line. *)
      ( false,
        "opencl: unknown other type, this symbol only in a note",
        "input.cl:5:3: error: unknown type name 'Some_other_type'\n\
        \    5 |   Some_other_type z;\n\
        \      |   ^\n\
         input.cl:3:3: note: while declaring 'Test_record_field_store_triple'\n"
      );
      (* REJECT — the glslang half of ATTACK 1: the tolerated parse error in a
         genuinely out-of-order shader, PLUS a second, unrelated diagnostic.
         Same rule on the same footing as clause A. *)
      ( false,
        "vulkan: tolerated parse error PLUS an unrelated error",
        Printf.sprintf
          "glslangValidator failed:\n\
           %s\n\
           ERROR: %s:4: '' :  syntax error, unexpected IDENTIFIER, expecting \
           RIGHT_BRACE\n\
           ERROR: %s:9: 'assign' :  l-value required\n\
           ERROR: 2 compilation errors.  No code generated.\n"
          bad_order
          bad_order
          bad_order );
    ]
  in
  let bad =
    List.filter
      (fun (want, _, msg) ->
        not
          (Bool.equal
             (is_backlog203_struct_gap
                ~inner:"Test_record_field_store_triple"
                msg)
             want))
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

(* An explicit ALLOWLIST of the backends that can have the backlog-203
   struct-ordering gap, paired with the generator that produces the source they
   compile — so the expectation is PREDICTED per kernel from the emitted text,
   not assumed per backend.

   An allowlist, because the previous "anything but the CPU pair" complement was
   wrong twice over. CUDA/PTX was inside the tolerated set and does not have the
   gap at all: the direct PTX emitter declares no C structs, so there is no
   declaration order to get wrong. It compiles the nested kernel and passes
   (measured under ZLUDA on both devices) — and the complement form then printed
   a "backlog-203 is fixed, drop the known-gap branch" note that told a
   maintainer to delete a branch OpenCL and Vulkan still need, while a future
   clang-shaped regression on CUDA/PTX would have been swallowed as expected.

   Metal and HIP are untested here (no such device on this machine) and are NOT
   on the list: nothing is claimed about them, and a failure on either is
   reported as a failure rather than tolerated.

   Predicted per kernel, because "a nested record kernel cannot compile here" is
   ALSO wrong. The gap is an unsorted emission order, and whether it bites
   depends on the order [kern_types] happens to hold — measured in this very
   file: [outer] is emitted before [triple] (does not compile on OpenCL or
   Vulkan), while [mouter] is emitted AFTER [mtriple] (compiles, runs, and
   passes on both). So the source is inspected and the failure is tolerated only
   when the source genuinely has the defect. *)
let struct_emitting_frameworks :
    (string
    * (types:(string * (string * Sarek_ir_types.elttype) list) list ->
      Sarek_ir_types.kernel ->
      string))
    list =
  [
    ("OpenCL", Sarek_codegen.Sarek_ir_opencl.generate_with_types);
    ( "Vulkan",
      fun ~types ir -> Sarek_codegen.Sarek_ir_glsl.generate_with_types ~types ir
    );
  ]

(* [Some true]  — this backend emits C-like struct declarations AND this kernel's
                  emitted source uses [inner] before declaring it: backlog-203
                  applies, a compile failure with that wording is expected.
   [Some false] — this backend emits struct declarations and this kernel's are in
                  dependency order: it must compile and pass.
   [None]       — no generator on the allowlist for this backend, so no
                  prediction: any failure is a failure. *)
let predict_struct_gap (dev : Device.t) ~kernel ~(inner : string) : bool option
    =
  match List.assoc_opt dev.Device.framework struct_emitting_frameworks with
  | None -> None
  | Some generate ->
      let ir = ir_of kernel in
      let types = ir.Sarek_ir_types.kern_types in
      Some (uses_type_before_declaring (generate ~types ir) inner)

(* One nested case. [read] returns (target, same-level witness 1, same-level
   witness 2, outer-level witness) so the two shapes cannot drift apart in what
   they assert. [~inner] is the emitted symbol of the struct the gap is about. *)
let nested_case (dev : Device.t) ~(label : string) ~kernel ~(inner : string)
    ~make ~read : bool =
  Printf.printf
    "field-store %-10s [%s] %s: %!"
    label
    dev.Device.framework
    dev.Device.name ;
  let predicted_gap = predict_struct_gap dev ~kernel ~inner in
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
      let msg = Printexc.to_string e in
      if predicted_gap = Some true && is_backlog203_struct_gap ~inner msg then begin
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
      if predicted_gap = Some true then
        (* Not a failure, but a stale expectation: the emitted source DOES use
           the inner struct before declaring it and the compiler accepted it
           anyway. Say so loudly rather than quietly passing. Printed only in
           that case — a kernel whose structs are in dependency order, or a
           backend that emits no struct declarations at all, has nothing to
           report. *)
        Printf.printf
          "\n\
          \  NOTE: %s compiled a source that uses %s before declaring it — \
           backlog-203 no longer bites here\n\
           %!"
          dev.Device.framework
          inner ;
      Transfer.flush dev ;
      let ok = ref true in
      let reported = ref 0 in
      for i = 0 to n - 1 do
        let tgt, w1, w2, outer_w = read v i in
        let bad name got want =
          if Float.abs (got -. want) > 1e-4 then begin
            ok := false ;
            if !reported < 3 then begin
              incr reported ;
              Printf.printf "\n  @%d %s: got %g want %g" i name got want
            end
          end
        in
        bad "leaf target" tgt (float_of_int (i + 1) *. 2.0) ;
        (* Same level as the target. *)
        bad "leaf witness 1 (untouched)" w1 (float_of_int i) ;
        bad "leaf witness 2 (untouched)" w2 (float_of_int (i + 2)) ;
        (* OUTER level: a rebuild that dropped the enclosing record would show up
           here and nowhere else. *)
        bad "outer witness (untouched)" outer_w (float_of_int (1000 + i))
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
  (* The nested cases carry their own checker: [run_case] reports one target and
     two witnesses, and this needs three witnesses across two levels. They are
     also the only cases with a per-backend expectation (see backlog-203 in the
     header), so they cannot reuse [run_case]'s "a refusal is always a failure"
     rule. *)
  let nested_ok =
    nested_case
      dev
      ~label:"nested"
      ~kernel:scale_nested
      ~inner:"Test_record_field_store_triple"
      ~make:(fun () ->
        let v = Vector.create_custom outer_custom n in
        for i = 0 to n - 1 do
          Vector.set v i {tag = float_of_int (1000 + i); mid = orig i}
        done ;
        v)
      ~read:(fun v i ->
        let p = Vector.get v i in
        (p.mid.b, p.mid.a, p.mid.c, p.tag))
  in
  (* The same shape with a MUTABLE leaf. Depth 1 needed both polarities
     ([triple] and [mtriple]) and nesting needs them for the same reason: with an
     immutable leaf the pre-fix Native codegen does not BUILD, so the immutable
     case alone can never exercise the silent drop. *)
  let nested_mut_ok =
    nested_case
      dev
      ~label:"nested-mut"
      ~kernel:scale_nested_mut
      ~inner:"Test_record_field_store_mtriple"
      ~make:(fun () ->
        let v = Vector.create_custom mouter_custom n in
        for i = 0 to n - 1 do
          Vector.set v i {mtag = float_of_int (1000 + i); mmid = morig i}
        done ;
        v)
      ~read:(fun v i ->
        let p = Vector.get v i in
        (p.mmid.mb, p.mmid.ma, p.mmid.mc, p.mtag))
  in
  immutable_ok && mutable_ok && nested_ok && nested_mut_ok

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
  (* A device class this file's header makes a claim ABOUT must not be able to
     go missing quietly.

     The header states that CUDA/PTX compiles BOTH nested kernels and produces
     correct values at both levels, "measured under ZLUDA on an RX 7900 XTX and
     on a Ryzen 9 7950X", and [predict_struct_gap]'s reason for keeping CUDA/PTX
     OFF the tolerated allowlist rests on that measurement. Under `dune runtest`
     without ZLUDA on the loader path, this file enumerated 7 devices, printed
     ZERO CUDA/PTX rows, and exited 0 — the claim was not reproduced and nothing
     said so. The dune rule now sets LD_LIBRARY_PATH so the device is present
     where it exists; this line is what makes its absence visible where it does
     not.

     NOT a failure: a machine with no ZLUDA and no CUDA driver legitimately has
     no such device, and failing there would make the suite unrunnable rather
     than honest. It is a loud named skip, which is the difference between "the
     header's measurement was not reproduced here" and a false green.

     EVERY framework the header makes a claim about, not CUDA/PTX alone. The list
     held only CUDA/PTX while the sentence introducing it said "any framework
     this header makes a claim about", and the header makes measured claims about
     five — so on a CPU-only host the OpenCL and Vulkan claims went unreproduced
     in exactly the silence this mechanism exists to break, and the sentence was
     wider than the list under it. Each framework carries its OWN claim text,
     because one generic sentence stretched over five different measurements
     would be that same defect again, one level down. *)
  let frameworks_seen =
    Array.to_list devs |> List.map (fun (d : Device.t) -> d.Device.framework)
  in
  let claimed_frameworks =
    [
      ( "Interpreter",
        "the store lands at depth 1 and nested, the pre-fix \
         Unsupported_operation \"record field assignment\" refusal being gone"
      );
      ( "Native",
        "the store lands at depth 1 and nested rather than being dropped into \
         the temporary record Vector.get marshalled out" );
      ( "CUDA/PTX",
        "both nested kernels compile and land at both levels, which is also \
         why CUDA/PTX is off the tolerated allowlist \
         [struct_emitting_frameworks] — put ZLUDA on LD_LIBRARY_PATH, or run \
         on a CUDA host, to exercise it" );
      ( "OpenCL",
        "[mouter] compiles, runs and passes while [outer] fails with clang's \
         \"unknown type name\" — the two sides of the backlog-203 ordering gap \
         that [predict_struct_gap] decides between" );
      ( "Vulkan",
        "[mouter] compiles, runs and passes while [outer] fails with glslang's \
         bare parse error — the two sides of the backlog-203 ordering gap that \
         [predict_struct_gap] decides between" );
    ]
  in
  List.iter
    (fun (fw, claim) ->
      if not (List.mem fw frameworks_seen) then
        Printf.printf
          "NOT MEASURED HERE: no %s device was enumerated, so the header's %s \
           claim (%s) is NOT reproduced by this run.\n\
           %!"
          fw
          fw
          claim)
    claimed_frameworks ;
  Printf.printf
    "%d device(s) exercised (frameworks: %s)\n%!"
    (Array.length devs)
    (String.concat ", " (List.sort_uniq String.compare frameworks_seen)) ;
  if !any_failure then exit 1
