(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for record-declaration dependency ordering (backlog-203).
 *
 * [kern_types] list order is NOT dependency order — the PPX prepends the types
 * reachable through the registry to the ones the kernel payload declares — so a
 * record with a record-typed field could have its struct emitted AFTER the
 * struct that uses it. Every C-family backend then failed to compile
 * (`unknown type name '<Inner>'`) and GLSL/WGSL failed to parse the field line.
 *
 * These tests pin the sort at the level the device e2e test cannot reach:
 *   - the emission ORDER for the C family, including Metal and HIP, for which
 *     this host has no hardware (the e2e test can only prove OpenCL/Vulkan);
 *   - STABILITY, which no device can observe: two independent records must keep
 *     their input order, or every committed golden churns on unrelated edits;
 *   - the CYCLE refusal, which the PPX front end cannot produce at all.
 ******************************************************************************)

open Sarek_ir_types

let names types = List.map fst types

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
   [referenced_record_names] walks a graph, so its regression mode is a
   non-terminating tail-recursive loop, not a wrong answer. Left to itself that
   reads as a hung test rather than a failing one, and `dune runtest` imposes no
   timeout of its own — the only thing that would eventually notice is a CI job
   limit, with no message naming the cause. SIGALRM turns the hang into a named
   assertion: the walk allocates on every step (it conses onto the visited set),
   so there is a poll point for the signal to be delivered at. *)
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
  check_order
    "one-level nesting"
    ["triple"; "outer"]
    (names (Sarek_ir_codegen.sort_record_types_by_dependency input)) ;
  print_endline "  one-level: inner struct first: OK"

(* Three levels, worst-case input order: each entry depends on the next. *)
let test_three_level_chain () =
  let input =
    [("top", [("s", mid)]); ("mid", mid_fields); ("leaf", leaf_fields)]
  in
  check_order
    "three-level chain"
    ["leaf"; "mid"; "top"]
    (names (Sarek_ir_codegen.sort_record_types_by_dependency input)) ;
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
    (names (Sarek_ir_codegen.sort_record_types_by_dependency input)) ;
  print_endline "  two independent nested types: both before the user: OK"

(* Stability, stated as its own property: an input with no dependencies at all
   comes back byte-identical, in a name order no comparison function would
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
  check_order
    "stability"
    ["zeta"; "alpha"; "mu"; "beta"]
    (names (Sarek_ir_codegen.sort_record_types_by_dependency input)) ;
  print_endline "  independent records keep their input order: OK"

(* An already dependency-ordered list is returned unchanged — the sort is a
   no-op on every type list that compiled before this change. *)
let test_noop_on_sorted_input () =
  let input =
    [("leaf", leaf_fields); ("mid", mid_fields); ("top", [("s", mid)])]
  in
  check_order
    "already sorted"
    ["leaf"; "mid"; "top"]
    (names (Sarek_ir_codegen.sort_record_types_by_dependency input)) ;
  print_endline "  already-ordered input is unchanged: OK"

(* A record referenced through an array or a variant payload is still a
   dependency: the struct has to exist before the type that names it. *)
let test_dependency_through_array_and_variant () =
  let input =
    [("holder", [("xs", TArray (triple, Global))]); ("triple", triple_fields)]
  in
  check_order
    "dependency through an array field"
    ["triple"; "holder"]
    (names (Sarek_ir_codegen.sort_record_types_by_dependency input)) ;
  let input =
    [
      ("wrapper", [("v", TVariant ("opt", [("None", []); ("Some", [leaf])]))]);
      ("leaf", leaf_fields);
    ]
  in
  check_order
    "dependency through a variant payload"
    ["leaf"; "wrapper"]
    (names (Sarek_ir_codegen.sort_record_types_by_dependency input)) ;
  print_endline "  array and variant-payload field types count as deps: OK"

(* A cycle has no valid emission order. It must be refused, not emitted in
   input order — a silently wrong order is the defect this sort removes.
   Unreachable from the PPX (it resolves a field's alignment from a registry
   populated by the field type's own earlier declaration, so a self- or
   forward-referencing field is refused there), so this is the only place the
   refusal can be exercised. *)
let test_cycle_is_refused () =
  let rec_a = TRecord ("a", [("to_b", TRecord ("b", []))]) in
  let input =
    [("a", [("to_b", TRecord ("b", []))]); ("b", [("to_a", rec_a)])]
  in
  match Sarek_ir_codegen.sort_record_types_by_dependency input with
  | order ->
      failwith
        (Printf.sprintf
           "cycle: expected Record_type_cycle, got [%s]"
           (String.concat "; " (names order)))
  | exception Sarek_ir_codegen.Record_type_cycle unplaced ->
      if List.sort String.compare unplaced <> ["a"; "b"] then
        failwith
          (Printf.sprintf
             "cycle: expected both names unplaced, got [%s]"
             (String.concat "; " unplaced)) ;
      print_endline "  a record cycle raises Record_type_cycle: OK"

(* A record field of the record's OWN type is not orderable either, but it is
   not a cycle between distinct declarations: the self-edge is dropped so the
   diagnostic stays about real cycles, and the C backend's own field-type
   emission is what reports it. Pinned so the drop is deliberate, not
   incidental. *)
let test_self_reference_does_not_raise () =
  let input = [("selfy", [("me", TRecord ("selfy", []))])] in
  check_order
    "self reference"
    ["selfy"]
    (names (Sarek_ir_codegen.sort_record_types_by_dependency input)) ;
  print_endline "  a self-referencing field is not reported as a cycle: OK"

(* The C-family emitter itself, not just the sort: this is the only coverage
   Metal and HIP get on a host with neither. *)
let test_c_family_emission_order () =
  let type_of_elttype = function
    | TFloat32 -> "float"
    | TRecord (n, _) -> Sarek_ir_codegen.mangle_name n
    | _ -> "int"
  in
  let buf = Buffer.create 256 in
  Sarek_ir_codegen.gen_record_typedefs
    ~type_of_elttype
    buf
    [("outer", [("mid", triple)]); ("triple", triple_fields)] ;
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
  print_endline "  gen_record_typedefs declares the inner struct first: OK"

(* [referenced_record_names] must terminate on a cyclic elttype VALUE, not just
   on a finite tree — the sort calls it on IR nobody promised is acyclic.

   All THREE cycle shapes OCaml's recursive-value rule admits are pinned, not
   just the record one. The first version of the guard was keyed on the record
   NAME, which closes a cycle only if a [TRecord] sits on it; the vec and variant
   cycles below have no [TRecord] on the cycle at all and looped forever (they
   are tail-recursive, so they hang rather than overflow — a test that only
   covered the record shape reported a general "safe on a cyclic value" it had
   never exercised). Each case runs under {!within_deadline} so a regression is a
   named failure, not a hung process. *)
let test_referenced_names_terminates_on_cyclic_value () =
  let check what expected ty =
    let got =
      within_deadline ~seconds:10 ~what (fun () ->
          Sarek_ir_codegen.referenced_record_names ty)
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
  check "referenced_record_names on a TRecord cycle" ["node"] node ;
  (* Shape 2: the cycle is closed by TVec alone — no TRecord on it. *)
  let rec vec_cycle = TVec vec_cycle in
  check "referenced_record_names on a TVec cycle" [] vec_cycle ;
  (* Shape 3: the cycle is closed by a TVariant payload — no TRecord on it, but
     a record hangs off the same variant and must still be reported. *)
  let rec var_cycle =
    TVariant ("loop", [("Stop", [leaf]); ("Go", [TArray (var_cycle, Global)])])
  in
  check "referenced_record_names on a TVariant cycle" ["leaf"] var_cycle ;
  print_endline
    "  referenced_record_names terminates on record/vec/variant cycles: OK"

let () =
  print_endline "=== record declaration dependency order (backlog-203) ===" ;
  test_one_level () ;
  test_three_level_chain () ;
  test_two_independent_nested () ;
  test_stable_on_independent_records () ;
  test_noop_on_sorted_input () ;
  test_dependency_through_array_and_variant () ;
  test_cycle_is_refused () ;
  test_self_reference_does_not_raise () ;
  test_c_family_emission_order () ;
  test_referenced_names_terminates_on_cyclic_value () ;
  print_endline "All record-order tests passed!"
