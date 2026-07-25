(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_ir_wgsl.escape_wgsl_name.
 *
 * Two properties, both of which were violated before this suite existed:
 *
 *   1. INJECTIVITY. Distinct source identifiers must not collide on one WGSL
 *      name. This is the property with teeth: two colliding `var` declarations
 *      in one block produce a shader that is perfectly VALID, so naga accepts
 *      it, the validation sweep passes, and every read after the second
 *      declaration silently resolves to the wrong binding. There is no
 *      diagnostic anywhere in the pipeline for this failure — only a wrong
 *      numeric answer at runtime — which is why it is pinned here rather than
 *      left to the shader-validation gate.
 *
 *   2. LEGALITY. The escaped name must actually be a WGSL identifier: not a
 *      keyword or reserved word, not `_`, and not `__`-prefixed.
 *
 * The reserved-word table these tests exercise is not a transcription from
 * memory. It was built by running every candidate through `naga` (the same
 * validator ci/assert-toolchain.sh pins) as `var <name> : i32` in a minimal
 * compute shader and keeping the ones it rejected. `test_shader_recursion_
 * vector.ml` runs naga over whole generated shaders; this suite is the fast,
 * naga-free complement that runs on a developer machine with no tooling.
 ******************************************************************************)

module W = Sarek_codegen.Sarek_ir_wgsl

let esc = W.escape_wgsl_name

(* --------------------------------------------------------------------- *)
(* 1. Injectivity                                                        *)
(* --------------------------------------------------------------------- *)

(* The exact pairs the previous escaping merged. Each is a distinct pair of
   legal source identifiers that mapped to one WGSL name:
     "__i" and "sarek__i" -> "sarek__i"   (double-underscore rule)
     "_"   and "sarek_"   -> "sarek_"     (bare-underscore rule)
     "if"  and "ifv"      -> "ifv"        (keyword 'v'-suffix rule)
   Each rule contributed its own collision, so all three are pinned. *)
let collision_pairs =
  [
    ("__i", "sarek__i");
    ("_", "sarek_");
    ("if", "ifv");
    ("__v0", "sarek__v0");
    ("fallthrough", "fallthroughv");
    ("sarek_gid", "gid");
    ("sarek_fmod", "fmod");
  ]

let test_no_collisions () =
  List.iter
    (fun (a, b) ->
      Alcotest.(check bool)
        (Printf.sprintf "%S and %S must not collide (both -> %S)" a b (esc a))
        false
        (esc a = esc b))
    collision_pairs

(* Injectivity over a corpus, rather than only over the known-bad pairs: every
   reserved word, each one prefixed, and the shapes the frontend actually
   generates. A duplicate anywhere in the image is a failure. *)
let corpus =
  W.wgsl_reserved_keywords
  @ List.map (fun k -> "sarek_" ^ k) W.wgsl_reserved_keywords
  @ List.map (fun k -> k ^ "v") W.wgsl_reserved_keywords
  @ ["_"; "__"; "___"; "__i"; "__v0"; "__m3"; "sarek_"; "sarek__"; "sarek___i"]
  @ ["i"; "x"; "acc"; "gid"; "idx"; "tmp0"; "sarek"; "sarekx"; "sarek_x"]

let test_injective_over_corpus () =
  let seen = Hashtbl.create 512 in
  List.iter
    (fun name ->
      let e = esc name in
      match Hashtbl.find_opt seen e with
      | Some other when other <> name ->
          Alcotest.failf
            "escape_wgsl_name is not injective: %S and %S both map to %S"
            other
            name
            e
      | _ -> Hashtbl.replace seen e name)
    corpus

(* --------------------------------------------------------------------- *)
(* 2. Legality of the result                                             *)
(* --------------------------------------------------------------------- *)

let test_result_is_never_reserved () =
  List.iter
    (fun k ->
      Alcotest.(check bool)
        (Printf.sprintf "esc %S = %S must not itself be reserved" k (esc k))
        false
        (List.mem (esc k) W.wgsl_reserved_keywords))
    W.wgsl_reserved_keywords

let test_result_has_no_reserved_prefix () =
  List.iter
    (fun name ->
      let e = esc name in
      Alcotest.(check bool)
        (Printf.sprintf "esc %S = %S must not start with __" name e)
        false
        (String.starts_with ~prefix:"__" e) ;
      Alcotest.(check bool)
        (Printf.sprintf "esc %S = %S must not be a bare _" name e)
        false
        (e = "_"))
    corpus

(* --------------------------------------------------------------------- *)
(* 3. The keyword table itself                                           *)
(* --------------------------------------------------------------------- *)

(* Regression for the six current WGSL keywords the table omitted. A source
   variable named `alias` or `enable` was emitted verbatim and no WebGPU
   implementation accepts the result. *)
let newly_covered_keywords =
  [
    "alias";
    "const_assert";
    "continuing";
    "diagnostic";
    "enable";
    "requires";
    (* reserved words — same class, and these are plausible OCaml names *)
    "ref";
    "set";
    "get";
    "from";
    "shared";
    "filter";
    "target";
    "where";
    "match";
    "use";
    "union";
    "static";
  ]

let test_keyword_table_covers () =
  List.iter
    (fun k ->
      Alcotest.(check bool)
        (Printf.sprintf "%S must be in the reserved table" k)
        true
        (List.mem k W.wgsl_reserved_keywords) ;
      Alcotest.(check bool)
        (Printf.sprintf "%S must be rewritten, not emitted verbatim" k)
        false
        (esc k = k))
    newly_covered_keywords

let test_no_duplicates_in_table () =
  let sorted = List.sort compare W.wgsl_reserved_keywords in
  let rec dup = function
    | a :: (b :: _ as tl) -> if a = b then Some a else dup tl
    | _ -> None
  in
  match dup sorted with
  | Some d -> Alcotest.failf "duplicate entry in wgsl_reserved_keywords: %S" d
  | None -> ()

(* Generator-minted names must round-trip unchanged. rename_scalar_shadowing_
   locals puts sarek_scalar_shadow_* names back into the IR as ordinary
   variables, so they reach escape_wgsl_name a second time on the way out;
   without the exemption they came out as sarek_sarek_scalar_shadow_width_1 and
   the WGSL shadowing gate in test_shader_recursion_vector.ml failed. Escaping
   must be idempotent on them specifically. *)
let generated_names =
  [
    "sarek_scalar_shadow_width_1";
    "sarek_scalar_shadow_if_0";
    "sarek_scalar_shadow_sarek_x_2";
  ]

let test_generated_names_are_fixed_points () =
  List.iter
    (fun n ->
      Alcotest.(check string)
        (Printf.sprintf "%S must round-trip unchanged" n)
        n
        (esc n) ;
      Alcotest.(check string)
        (Printf.sprintf "%S must be idempotent under a second escape" n)
        (esc n)
        (esc (esc n)))
    generated_names

(* Ordinary identifiers must survive untouched — an escaping pass that renames
   everything would satisfy injectivity and be useless to read. *)
let test_plain_names_untouched () =
  List.iter
    (fun n ->
      Alcotest.(check string) (Printf.sprintf "%S is left alone" n) n (esc n))
    ["i"; "x"; "acc"; "idx"; "tmp0"; "my_var"; "gid"; "sarek"; "sarekx"]

let () =
  let open Alcotest in
  run
    "Sarek_ir_wgsl escaping"
    [
      ( "injectivity",
        [
          test_case "known collision pairs" `Quick test_no_collisions;
          test_case "injective over corpus" `Quick test_injective_over_corpus;
        ] );
      ( "legality",
        [
          test_case "result never reserved" `Quick test_result_is_never_reserved;
          test_case
            "result has no reserved prefix"
            `Quick
            test_result_has_no_reserved_prefix;
        ] );
      ( "keyword_table",
        [
          test_case "covers WGSL keywords" `Quick test_keyword_table_covers;
          test_case "no duplicates" `Quick test_no_duplicates_in_table;
          test_case "plain names untouched" `Quick test_plain_names_untouched;
          test_case
            "generated names are fixed points"
            `Quick
            test_generated_names_are_fixed_points;
        ] );
    ]
