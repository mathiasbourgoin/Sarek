(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for type-declaration dependency ordering (backlog-203 for the
 * record-to-record edge, backlog-211 for the edges that CROSS between a record
 * and a variant).
 *
 * [kern_types] list order is NOT dependency order — the PPX prepends the types
 * reachable through the registry to the ones the kernel payload declares — so a
 * record with a record-typed field could have its struct emitted AFTER the
 * struct that uses it. Every C-family backend then failed to compile
 * (`unknown type name '<Inner>'`) and GLSL/WGSL failed to parse the field line.
 * backlog-203 sorted records inside each family's own emission loop, which left
 * the cross-kind edge ordered by neither loop: the C family emitted variants
 * first (so a variant with a RECORD payload was red) and GLSL/WGSL emitted
 * records first (so a record with a VARIANT-typed field was red).
 *
 * These tests pin the sort at the level the device e2e test cannot reach:
 *   - the ORDER produced for every edge direction, including the two cross-kind
 *     ones, with no device and no backend generator involved;
 *   - the TIE-BREAK, which no device can observe: a list already in a valid
 *     emission order must come back unchanged, or every committed golden churns
 *     on unrelated edits — together with the case pinning what is deliberately
 *     NOT guaranteed, namely that a BLOCKED declaration is overtaken by later
 *     independent ones, so this is not stability in the general sense;
 *   - the CYCLE refusal, which the PPX front end cannot produce at all.
 *
 * The five real backend generators are pinned separately, in
 * sarek/tests/codegen_golden/test_decl_order_all_backends.ml — that is where
 * CUDA, HIP, Metal and WGSL get their coverage, none of which this host can
 * execute.
 ******************************************************************************)

open Sarek_ir_types

let decl_name = function
  | Sarek_ir_codegen.Record_decl (n, _) | Sarek_ir_codegen.Variant_decl (n, _)
    ->
      n

(* A name tagged with its kind. Cross-kind tests read better when the expected
   list says which loop each entry used to come from, and it also means a test
   cannot pass by placing a record where a variant was expected. *)
let tagged_name = function
  | Sarek_ir_codegen.Record_decl (n, _) -> "record:" ^ n
  | Sarek_ir_codegen.Variant_decl (n, _) -> "variant:" ^ n

let names decls = List.map decl_name decls

let tagged_names decls = List.map tagged_name decls

let rec_decls types =
  List.map (fun (n, f) -> Sarek_ir_codegen.Record_decl (n, f)) types

let var_decls variants =
  List.map (fun (n, c) -> Sarek_ir_codegen.Variant_decl (n, c)) variants

(* The two family tie-break orders, spelled out here rather than imported: the
   shared module expresses them as the [tie_break] argument to [gen_type_decls]
   and does not export a list-building helper, so these mirror what
   [gen_type_decls] does internally. The dispatch tests below drive the real
   thing through [gen_type_decls] itself, so a divergence between these two lines
   and the shared module cannot hide a defect — it would show up there. *)
let variants_first ~records ~variants = var_decls variants @ rec_decls records

let records_first ~records ~variants = rec_decls records @ var_decls variants

(* The records-only entry point the backlog-203 cases were written against:
   wrap each record, sort, read the names back. *)
let sort_records types =
  names (Sarek_ir_codegen.sort_type_decls_by_dependency (rec_decls types))

let check_order what expected got =
  if expected <> got then
    failwith
      (Printf.sprintf
         "%s: expected [%s] got [%s]"
         what
         (String.concat "; " expected)
         (String.concat "; " got))

exception Timed_out

(* Run [f] under a hard deadline and FAIL if it does not return.
   [referenced_type_names] walks a graph, so its regression mode is a
   non-terminating tail-recursive loop, not a wrong answer. Left to itself that
   reads as a hung test rather than a failing one, and `dune runtest` imposes no
   timeout of its own — the only thing that would eventually notice is a CI job
   limit, with no message naming the cause. SIGALRM turns the hang into a named
   assertion.

   Do NOT justify that by "the walk allocates on every step". It does today, but
   the regression this deadline exists to catch is a guard that stops recording
   nodes, and under that guard the [TVec] cycle allocates nothing at all — the
   justification would be absent for exactly the shape that hangs. What makes the
   signal deliverable is OCaml's safepoints (>= 4.14) at function entry and loop
   back-edges, which a non-allocating recursive walk hits regardless. Measured:
   with the visited set narrowed to named nodes only, this fires on the [TVec]
   case with its own message. *)
let within_deadline ~seconds ~what f =
  let prev =
    Sys.signal Sys.sigalrm (Sys.Signal_handle (fun _ -> raise Timed_out))
  in
  let restore () =
    ignore (Unix.alarm 0) ;
    Sys.set_signal Sys.sigalrm prev
  in
  ignore (Unix.alarm seconds) ;
  match f () with
  | v ->
      restore () ;
      v
  | exception Timed_out ->
      restore () ;
      failwith
        (Printf.sprintf
           "%s: did not terminate within %ds — the cyclic-value guard no \
            longer closes this shape"
           what
           seconds)
  | exception e ->
      restore () ;
      raise e

let triple_fields = [("a", TFloat32); ("b", TFloat32); ("c", TFloat32)]

let triple = TRecord ("triple", triple_fields)

let leaf_fields = [("p", TFloat32)]

let leaf = TRecord ("leaf", leaf_fields)

let mid_fields = [("r", leaf)]

let mid = TRecord ("mid", mid_fields)

(* A dependent record placed BEFORE its dependency in the input list — the
   shape the PPX actually produced. *)
let test_one_level () =
  let input =
    [("outer", [("tag", TFloat32); ("mid", triple)]); ("triple", triple_fields)]
  in
  check_order "one-level nesting" ["triple"; "outer"] (sort_records input) ;
  print_endline "  one-level: inner struct first: OK"

(* Three levels, worst-case input order: each entry depends on the next. *)
let test_three_level_chain () =
  let input =
    [("top", [("s", mid)]); ("mid", mid_fields); ("leaf", leaf_fields)]
  in
  check_order "three-level chain" ["leaf"; "mid"; "top"] (sort_records input) ;
  print_endline "  three-level chain: leaf, mid, top: OK"

(* Two independent nested types in one record: both dependencies must precede
   the user, and the two independent ones must keep their INPUT order. The
   names here are deliberately not in alphabetical order, so a name-keyed sort
   would produce ["leaf"; "triple"; ...] and fail this test. *)
let test_two_independent_nested () =
  let input =
    [
      ("twin", [("left", triple); ("right", leaf)]);
      ("triple", triple_fields);
      ("leaf", leaf_fields);
    ]
  in
  check_order
    "two independent nested types"
    ["triple"; "leaf"; "twin"]
    (sort_records input) ;
  print_endline "  two independent nested types: both before the user: OK"

(* The tie-break, stated as its own property: an input with no dependencies at
   all comes back byte-identical, in a name order no comparison function would
   produce. This is what keeps committed goldens from churning. *)
let test_stable_on_independent_records () =
  let input =
    [
      ("zeta", leaf_fields);
      ("alpha", leaf_fields);
      ("mu", leaf_fields);
      ("beta", leaf_fields);
    ]
  in
  check_order "stability" ["zeta"; "alpha"; "mu"; "beta"] (sort_records input) ;
  print_endline "  independent records keep their input order: OK"

(* An already dependency-ordered list is returned unchanged — the sort is a
   no-op on every type list that compiled before this change. Together with the
   previous case this is the WHOLE ordering guarantee; the next case pins what
   is deliberately NOT guaranteed. *)
let test_noop_on_sorted_input () =
  let input =
    [("leaf", leaf_fields); ("mid", mid_fields); ("top", [("s", mid)])]
  in
  check_order "already sorted" ["leaf"; "mid"; "top"] (sort_records input) ;
  print_endline "  already-ordered input is unchanged: OK"

(* NOT stable in the general sense, and pinned so nobody documents that it is.
   A BLOCKED declaration is overtaken by later independent ones, so two entries
   with no dependency between them can still swap: [outer] needs [triple], so
   [lone] and [triple] are both placed ahead of it and [lone] ends up before
   [outer] though it started after. The .mli says exactly this, and this case is
   what stops that sentence from drifting back to "stable". *)
let test_not_stable_when_an_entry_is_blocked () =
  let input =
    [
      ("outer", [("mid", triple)]);
      ("lone", leaf_fields);
      ("triple", triple_fields);
    ]
  in
  check_order
    "blocked entry is overtaken"
    ["lone"; "triple"; "outer"]
    (sort_records input) ;
  print_endline "  a blocked entry is overtaken by later independent ones: OK"

(* A record referenced through an array or a variant payload is still a
   dependency: the struct has to exist before the type that names it. *)
let test_dependency_through_array_and_variant () =
  let input =
    [("holder", [("xs", TArray (triple, Global))]); ("triple", triple_fields)]
  in
  check_order
    "dependency through an array field"
    ["triple"; "holder"]
    (sort_records input) ;
  let input =
    [
      ("wrapper", [("v", TVariant ("opt", [("None", []); ("Some", [leaf])]))]);
      ("leaf", leaf_fields);
    ]
  in
  check_order
    "dependency through a variant payload"
    ["leaf"; "wrapper"]
    (sort_records input) ;
  print_endline "  array and variant-payload field types count as deps: OK"

(* ---------------------------------------------------------------------------
   backlog-211: the two CROSS-KIND edge directions.
   --------------------------------------------------------------------------- *)

let probe_constrs = [("Nowhere", []); ("At", [leaf])]

let flagv_name = "flagv"

let flagv = TVariant (flagv_name, [("Off", []); ("Level", [TFloat32])])

(* Direction 1 — a VARIANT whose payload is a record, handed to the sort in the
   order the C family emitted its loops (variants first). This is the C-family
   half: OpenCL failed with `unknown type name` on the union member. *)
let test_variant_payload_record_is_ordered () =
  let input =
    variants_first
      ~records:[("leaf", leaf_fields)]
      ~variants:[("probe", probe_constrs)]
  in
  check_order
    "variant with a record payload, variants-first input"
    ["record:leaf"; "variant:probe"]
    (tagged_names (Sarek_ir_codegen.sort_type_decls_by_dependency input)) ;
  print_endline "  variant payload record is declared first: OK"

(* Direction 2 — a RECORD whose field type is a variant, handed to the sort in
   the order GLSL/WGSL emitted their loops (records first). This is the mirror
   half: Vulkan failed with a syntax error at the field line. *)
let test_record_variant_field_is_ordered () =
  let input =
    records_first
      ~records:[("gauge", [("gk", flagv); ("gv", TFloat32)])]
      ~variants:[(flagv_name, [("Off", []); ("Level", [TFloat32])])]
  in
  check_order
    "record with a variant field, records-first input"
    ["variant:flagv"; "record:gauge"]
    (tagged_names (Sarek_ir_codegen.sort_type_decls_by_dependency input)) ;
  print_endline "  variant field type is declared first: OK"

(* Both cross edges at once, in one declaration set, in the worst input order
   for each: the variant that needs a record comes before that record, and the
   record that needs a variant comes before that variant. A pass that ordered
   only one direction would leave the other inverted. *)
let test_both_cross_edges_in_one_pass () =
  let input =
    [
      Sarek_ir_codegen.Variant_decl ("probe", probe_constrs);
      Sarek_ir_codegen.Record_decl ("gauge", [("gk", flagv); ("gv", TFloat32)]);
      Sarek_ir_codegen.Variant_decl
        (flagv_name, [("Off", []); ("Level", [TFloat32])]);
      Sarek_ir_codegen.Record_decl ("leaf", leaf_fields);
    ]
  in
  let got =
    tagged_names (Sarek_ir_codegen.sort_type_decls_by_dependency input)
  in
  let index n =
    let rec go i = function
      | [] -> failwith (Printf.sprintf "both-cross-edges: %s not emitted" n)
      | x :: _ when x = n -> i
      | _ :: tl -> go (i + 1) tl
    in
    go 0 got
  in
  if index "record:leaf" > index "variant:probe" then
    failwith
      (Printf.sprintf
         "both-cross-edges: leaf must precede probe, got [%s]"
         (String.concat "; " got)) ;
  if index "variant:flagv" > index "record:gauge" then
    failwith
      (Printf.sprintf
         "both-cross-edges: flagv must precede gauge, got [%s]"
         (String.concat "; " got)) ;
  if List.length got <> 4 then
    failwith
      (Printf.sprintf
         "both-cross-edges: expected 4 declarations, got [%s]"
         (String.concat "; " got)) ;
  print_endline "  both cross directions ordered in one pass: OK"

(* Node identity is the POSITION, not the name, and this is the case that
   distinguishes the two. A record and a variant that share a mangled name — and
   they can, because [mangle_name] maps [M.t] and [M_t] to the same string, and
   fusion concatenates two kernels' type lists without deduplicating — form a
   GENUINE cross-kind edge when the record's field type is the variant. A
   name-keyed self-edge drop reads that edge as a self-reference and discards it,
   leaving the record emitted first: exactly the ordering this whole change
   exists to prevent, reintroduced by the identity choice alone. *)
let test_same_name_record_and_variant_edge_survives () =
  let same = "t" in
  let same_variant = TVariant (same, [("Off", []); ("Level", [TFloat32])]) in
  let input =
    records_first
      ~records:[(same, [("k", same_variant); ("v", TFloat32)])]
      ~variants:[(same, [("Off", []); ("Level", [TFloat32])])]
  in
  check_order
    "same-named record and variant"
    ["variant:t"; "record:t"]
    (tagged_names (Sarek_ir_codegen.sort_type_decls_by_dependency input)) ;
  print_endline
    "  a record's edge to a SAME-NAMED variant is kept, not read as a \
     self-edge: OK"

(* A cross-kind cycle: the record's field type is the variant and the variant's
   payload is the record. Unorderable, so it must be refused rather than
   emitted in some order — and the exception must name BOTH declarations, not
   only the record one. *)
let test_cross_kind_cycle_is_refused () =
  let rec rec_g = TRecord ("g", [("v", var_v)])
  and var_v = TVariant ("v", [("C", [rec_g])]) in
  let input =
    [
      Sarek_ir_codegen.Record_decl ("g", [("v", var_v)]);
      Sarek_ir_codegen.Variant_decl ("v", [("C", [rec_g])]);
    ]
  in
  match
    within_deadline ~seconds:10 ~what:"cross-kind cycle" (fun () ->
        Sarek_ir_codegen.sort_type_decls_by_dependency input)
  with
  | order ->
      failwith
        (Printf.sprintf
           "cross-kind cycle: expected Type_decl_cycle, got [%s]"
           (String.concat "; " (tagged_names order)))
  | exception Sarek_ir_codegen.Type_decl_cycle unplaced ->
      if List.sort String.compare unplaced <> ["g"; "v"] then
        failwith
          (Printf.sprintf
             "cross-kind cycle: expected both names unplaced, got [%s]"
             (String.concat "; " unplaced)) ;
      print_endline "  a record/variant cycle raises Type_decl_cycle: OK"

(* A variant whose own payload is itself is the variant-side mirror of the
   self-referencing record field: not a cycle between distinct declarations, so
   the self-edge is dropped and the declaration is emitted in place. *)
let test_variant_self_payload_does_not_raise () =
  let rec selfv = TVariant ("selfv", [("Wrap", [selfv])]) in
  let input = [Sarek_ir_codegen.Variant_decl ("selfv", [("Wrap", [selfv])])] in
  check_order
    "variant self payload"
    ["variant:selfv"]
    (tagged_names
       (within_deadline ~seconds:10 ~what:"variant self payload" (fun () ->
            Sarek_ir_codegen.sort_type_decls_by_dependency input))) ;
  print_endline "  a self-payload variant is not reported as a cycle: OK"

(* A cycle has no valid emission order. It must be refused, not emitted in
   input order — a silently wrong order is the defect this sort removes.
   Unreachable from the PPX (it resolves a field's alignment from a registry
   populated by the field type's own earlier declaration, so a self- or
   forward-referencing field is refused there), so this is the only place the
   refusal can be exercised. *)
let test_cycle_is_refused () =
  let rec_a = TRecord ("a", [("to_b", TRecord ("b", []))]) in
  let input =
    rec_decls [("a", [("to_b", TRecord ("b", []))]); ("b", [("to_a", rec_a)])]
  in
  match Sarek_ir_codegen.sort_type_decls_by_dependency input with
  | order ->
      failwith
        (Printf.sprintf
           "cycle: expected Type_decl_cycle, got [%s]"
           (String.concat "; " (names order)))
  | exception Sarek_ir_codegen.Type_decl_cycle unplaced ->
      if List.sort String.compare unplaced <> ["a"; "b"] then
        failwith
          (Printf.sprintf
             "cycle: expected both names unplaced, got [%s]"
             (String.concat "; " unplaced)) ;
      print_endline "  a record cycle raises Type_decl_cycle: OK"

(* A record field of the record's OWN type is not orderable either, but it is
   not a cycle between distinct declarations: the self-edge is dropped so the
   diagnostic stays about real cycles, and the C backend's own field-type
   emission is what reports it. Pinned so the drop is deliberate, not
   incidental. *)
let test_self_reference_does_not_raise () =
  let input = [("selfy", [("me", TRecord ("selfy", []))])] in
  check_order "self reference" ["selfy"] (sort_records input) ;
  print_endline "  a self-referencing field is not reported as a cycle: OK"

(* The generic emission driver, not just the sort: each entry must be handed to
   the emitter for its OWN kind, and in the sorted order. Both cross shapes are
   driven through it with stand-in emitters, so a dispatch that sent a variant
   to the record emitter (or emitted in input order) is a named failure here,
   with no backend generator and no device in the picture. *)
let test_gen_type_decls_dispatch_and_order () =
  let emit_record buf (name, _) =
    Buffer.add_string buf (Printf.sprintf "R(%s) " name)
  in
  let emit_variant buf (name, _) =
    Buffer.add_string buf (Printf.sprintf "V(%s) " name)
  in
  let run ~tie_break ~records ~variants =
    let buf = Buffer.create 64 in
    Sarek_ir_codegen.gen_type_decls
      ~emit_record
      ~emit_variant
      ~tie_break
      buf
      ~records
      ~variants ;
    String.trim (Buffer.contents buf)
  in
  (* C-family shape: variant payload is a record, Variants_first tie-break. *)
  check_order
    "gen_type_decls on a variant-with-record-payload"
    ["R(leaf) V(probe)"]
    [
      run
        ~tie_break:Sarek_ir_codegen.Variants_first
        ~records:[("leaf", leaf_fields)]
        ~variants:[("probe", probe_constrs)];
    ] ;
  (* GLSL/WGSL shape: record field is a variant, Records_first tie-break. *)
  check_order
    "gen_type_decls on a record-with-variant-field"
    ["V(flagv) R(gauge)"]
    [
      run
        ~tie_break:Sarek_ir_codegen.Records_first
        ~records:[("gauge", [("gk", flagv); ("gv", TFloat32)])]
        ~variants:[(flagv_name, [("Off", []); ("Level", [TFloat32])])];
    ] ;
  (* THE GOLDEN-NON-CHURN PROPERTY, on a MIXED edge-free set, once per
     tie-break. This is the property three docstrings and a changelog entry rest
     on — "an edge-free type list is emitted byte-identically, so no committed
     golden moves" — and until now nothing exercised it with BOTH kinds present.
     The record-only stability cases above cannot: they never mix. *)
  let edge_free_records = [("zeta", leaf_fields); ("alpha", leaf_fields)] in
  let edge_free_variants =
    [("vone", [("A", [])]); ("vtwo", [("B", [TFloat32])])]
  in
  check_order
    "edge-free mixed set, Variants_first, is emitted in input order"
    ["V(vone) V(vtwo) R(zeta) R(alpha)"]
    [
      run
        ~tie_break:Sarek_ir_codegen.Variants_first
        ~records:edge_free_records
        ~variants:edge_free_variants;
    ] ;
  check_order
    "edge-free mixed set, Records_first, is emitted in input order"
    ["R(zeta) R(alpha) V(vone) V(vtwo)"]
    [
      run
        ~tie_break:Sarek_ir_codegen.Records_first
        ~records:edge_free_records
        ~variants:edge_free_variants;
    ] ;
  print_endline "  gen_type_decls dispatches by kind, in sorted order: OK" ;
  print_endline
    "  an edge-free mixed set is emitted in input order, both tie-breaks: OK"

(* The C-family record emitter itself, at the level the shared module owns: one
   [typedef struct] per record, and the inner struct before the field naming
   it. *)
let test_c_family_emission_order () =
  let type_of_elttype = function
    | TFloat32 -> "float"
    | TRecord (n, _) -> Sarek_ir_codegen.mangle_name n
    | _ -> "int"
  in
  let buf = Buffer.create 256 in
  Sarek_ir_codegen.gen_c_type_decls
    ~type_of_elttype
    ~constructor_prefix:"static inline"
    buf
    ~records:[("outer", [("mid", triple)]); ("triple", triple_fields)]
    ~variants:[] ;
  let src = Buffer.contents buf in
  let idx needle =
    let nl = String.length needle and hl = String.length src in
    let rec go i =
      if i + nl > hl then -1
      else if String.sub src i nl = needle then i
      else go (i + 1)
    in
    go 0
  in
  let decl_triple = idx "} triple;" and use_triple = idx "  triple mid;" in
  if decl_triple < 0 || use_triple < 0 then
    failwith (Printf.sprintf "C-family emission: unexpected source:\n%s" src) ;
  if decl_triple > use_triple then
    failwith
      (Printf.sprintf
         "C-family emission: `triple` used at %d before its declaration at %d:\n\
          %s"
         use_triple
         decl_triple
         src) ;
  print_endline "  gen_c_type_decls declares the inner struct first: OK"

(* [referenced_type_names] must terminate on a cyclic elttype VALUE, not just
   on a finite tree — the sort calls it on IR nobody promised is acyclic.

   All THREE cycle shapes OCaml's recursive-value rule admits are pinned. The
   first version of the guard was keyed on the record NAME, which closes a cycle
   only if a [TRecord] sits on it; the vec and variant cycles below have no
   [TRecord] on the cycle at all and looped forever (they are tail-recursive, so
   they hang rather than overflow — a test that only covered the record shape
   reported a general "safe on a cyclic value" it had never exercised). Each
   case runs under {!within_deadline} so a regression is a named failure, not a
   hung process.

   The expected lists changed with backlog-211: this function now reports
   VARIANT names as well as record names, because a variant declaration is a
   declaration the sort has to place. *)
let test_referenced_names_terminates_on_cyclic_value () =
  let check what expected ty =
    let got =
      within_deadline ~seconds:10 ~what (fun () ->
          Sarek_ir_codegen.referenced_type_names ty)
    in
    if got <> expected then
      failwith
        (Printf.sprintf
           "%s: expected [%s] got [%s]"
           what
           (String.concat "; " expected)
           (String.concat "; " got))
  in
  (* Shape 1: the cycle passes through a TRecord. *)
  let rec node = TRecord ("node", [("self", node)]) in
  check "referenced_type_names on a TRecord cycle" ["node"] node ;
  (* Shape 2: the cycle is closed by TVec alone — neither a record nor a
     variant is on it, so no name-keyed guard of either kind closes it. *)
  let rec vec_cycle = TVec vec_cycle in
  check "referenced_type_names on a TVec cycle" [] vec_cycle ;
  (* Shape 3: the cycle is closed by a TVariant payload. The variant itself is
     now reported, and the record hanging off the same variant still is. *)
  let rec var_cycle =
    TVariant ("loop", [("Stop", [leaf]); ("Go", [TArray (var_cycle, Global)])])
  in
  check "referenced_type_names on a TVariant cycle" ["leaf"; "loop"] var_cycle ;
  print_endline
    "  referenced_type_names terminates on record/vec/variant cycles: OK"

let () =
  print_endline
    "=== type declaration dependency order (backlog-203, backlog-211) ===" ;
  test_one_level () ;
  test_three_level_chain () ;
  test_two_independent_nested () ;
  test_stable_on_independent_records () ;
  test_noop_on_sorted_input () ;
  test_not_stable_when_an_entry_is_blocked () ;
  test_dependency_through_array_and_variant () ;
  test_variant_payload_record_is_ordered () ;
  test_record_variant_field_is_ordered () ;
  test_both_cross_edges_in_one_pass () ;
  test_same_name_record_and_variant_edge_survives () ;
  test_cross_kind_cycle_is_refused () ;
  test_variant_self_payload_does_not_raise () ;
  test_cycle_is_refused () ;
  test_self_reference_does_not_raise () ;
  test_gen_type_decls_dispatch_and_order () ;
  test_c_family_emission_order () ;
  test_referenced_names_terminates_on_cyclic_value () ;
  print_endline "All declaration-order tests passed!"
