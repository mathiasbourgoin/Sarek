(******************************************************************************)
(* test_layout_conformance.ml
 *
 * CMBT conformance harness for the packed aggregate layout model (FR-042).
 *
 * Strategy: [Model] is theories/PtxLayout.v itself, extracted to OCaml by
 * extraction/LayoutExtract.v (task #46 — it was previously a hand
 * transcription). The suite checks that the extracted model and the
 * production layout module [Sarek_ir_layout] agree on accept/reject AND on
 * all numeric offsets/sizes:
 *
 *   1. exhaustively, on every record shape with 1..4 fields over the scalar
 *      universe {int32, float32, bool, int64, float64} and every variant
 *      shape with 1..3 constructors of 0..2 scalar args;
 *   2. on 500 seeded qcheck-random nested-record shapes (depth <= 3);
 *   3. on the pinned host e2e types: point, point3d, color, particle
 *      (literal offset/size asserts on BOTH sides).
 *
 * Encoding note: PtxLayout.v names fields by [nat] index and abstracts the
 * scalar universe to [lty] (L32/L64: byte size + natural alignment only).
 * The agreement checked here is therefore on numeric offsets, sizes and
 * accept/reject verdicts, mapped through that encoding — never on the string
 * paths, which the theory deliberately does not model. *)

open Sarek_ir_types

(* ======================================================================= *)
(** * 1. The layout model — EXTRACTED from theories/PtxLayout.v *)
(* ======================================================================= *)

(* [Model] is Rocq's own extraction of PtxLayout.v, built by
   extraction/LayoutExtract.v and compiled as the [sarek_ptx_layout_model]
   library. It is not written; `make -f CoqMakefile` regenerates it and
   scripts/check-formal-proofs.sh fails if the committed copy differs.

   It replaces a 130-line module named [RocqMirror] that opened this file, whose
   own header called it "a line-by-line OCaml transcription of the definitions
   in theories/PtxLayout.v". That transcription was the weak link: every
   theorem in PtxLayout.v was proved about the Rocq definitions and then checked
   against a copy of them that nothing compared to the original, so an edit to
   the theory that nobody propagated left this suite green while it tested a
   model that had stopped being the model. Extracting removes the hop rather
   than watching it.

   The names below are unchanged from the transcription because the
   transcription used the theory's names; the substitution is [Model] for
   [RocqMirror] and nothing else. *)
module Model = Sarek_ptx_layout_model

(* ======================================================================= *)
(** * 2. Bridge: Sarek_ir_types.elttype shapes -> Model encoding *)
(* ======================================================================= *)

(* Scalar universe of the conformance domain (TUnit excluded: not part of
   the pinned e2e types nor of the theory's motivating universe). *)
let scalar_universe = [TInt32; TFloat32; TBool; TInt64; TFloat64]

let rec to_lfield (t : elttype) : Model.lfield =
  match t with
  | TInt32 | TFloat32 | TBool -> Model.LLeaf Model.L32
  | TInt64 | TFloat64 -> Model.LLeaf Model.L64
  | TRecord (_, fields) -> Model.LRec (to_lfields (List.map snd fields))
  (* TFloat16 is a 2-byte leaf; the extracted layout model only models L32/L64, and
     f16 aggregate fields are rejected by Sarek_ir_layout.flatten_field anyway
     (#57 slice 1), so f16 is outside the conformance domain. It is likewise
     absent from [scalar_universe] above, so no generated case reaches here. *)
  | TFloat16 | TVariant _ | TArray _ | TVec _ | TUnit ->
      invalid_arg "to_lfield: outside the conformance domain"

and to_lfields (ts : elttype list) : Model.lfields =
  List.fold_right
    (fun (i, t) acc -> Model.LCons (i, to_lfield t, acc))
    (List.mapi (fun i t -> (i, t)) ts)
    Model.LNil

let rec pp_elttype = function
  | TInt32 -> "i32"
  | TInt64 -> "i64"
  | TFloat16 -> "f16"
  | TFloat32 -> "f32"
  | TFloat64 -> "f64"
  | TBool -> "bool"
  | TUnit -> "unit"
  | TRecord (n, fs) ->
      Printf.sprintf
        "%s{%s}"
        n
        (String.concat ";" (List.map (fun (f, t) -> f ^ ":" ^ pp_elttype t) fs))
  | TVariant (n, _) -> n ^ "<variant>"
  | TArray _ -> "array"
  | TVec _ -> "vec"

let pp_fields fields =
  String.concat ";" (List.map (fun (n, t) -> n ^ ":" ^ pp_elttype t) fields)

let pp_ctors ctors =
  String.concat
    " | "
    (List.map
       (fun (n, args) ->
         n ^ "(" ^ String.concat "," (List.map pp_elttype args) ^ ")")
       ctors)

(* ======================================================================= *)
(** * 3. Agreement checks *)
(* ======================================================================= *)

(* [List.iteri2] does not exist in the stdlib; positional pairwise iteration
   with an index, failing loudly on length mismatch. *)
let iteri2 what f a b =
  Alcotest.(check int) (what ^ ": length") (List.length a) (List.length b) ;
  List.iteri (fun i (x, y) -> f i x y) (List.combine a b)

(* In the conformance domain (scalars + nested records only), the only legal
   OCaml rejection is [Misaligned_field]; any other error is a divergence. *)
let check_rejection_kind what = function
  | Sarek_ir_layout.Misaligned_field _ -> ()
  | e ->
      Alcotest.failf
        "%s: unexpected rejection kind: %s"
        what
        (Sarek_ir_layout.layout_error_message e)

let check_leaves what (m_leaves : Model.leaf list)
    (o_leaves : Sarek_ir_layout.leaf list) =
  iteri2
    (what ^ ": leaves")
    (fun k (m : Model.leaf) (o : Sarek_ir_layout.leaf) ->
      Alcotest.(check int)
        (Printf.sprintf "%s: leaf %d offset" what k)
        m.Model.lf_off
        o.Sarek_ir_layout.leaf_offset ;
      Alcotest.(check int)
        (Printf.sprintf "%s: leaf %d size" what k)
        (Model.leaf_size m)
        o.Sarek_ir_layout.leaf_size)
    m_leaves
    o_leaves

(* Model-vs-OCaml agreement for one record shape. *)
let check_record_agreement (fields : (string * elttype) list) =
  let what = "record{" ^ pp_fields fields ^ "}" in
  let fs = to_lfields (List.map snd fields) in
  let m_ok = Model.record_accepted fs in
  match Sarek_ir_layout.record_layout ~type_name:"t" fields with
  | Error e ->
      check_rejection_kind what e ;
      if m_ok then
        Alcotest.failf "%s: model accepts but Sarek_ir_layout rejects" what
  | Ok rl ->
      if not m_ok then
        Alcotest.failf "%s: Sarek_ir_layout accepts but model rejects" what ;
      Alcotest.(check int)
        (what ^ ": total size")
        (Model.record_size fs)
        rl.Sarek_ir_layout.rl_size ;
      check_leaves what (Model.record_leaves fs) rl.Sarek_ir_layout.rl_leaves ;
      iteri2
        (what ^ ": field offsets")
        (fun k (idx, m_off) (_name, o_off) ->
          Alcotest.(check int)
            (Printf.sprintf "%s: field %d decl index" what k)
            k
            idx ;
          Alcotest.(check int)
            (Printf.sprintf "%s: field %d offset" what k)
            m_off
            o_off)
        (Model.record_field_offsets fs)
        rl.Sarek_ir_layout.rl_fields

(* Model-vs-OCaml agreement for one variant shape. *)
let check_variant_agreement (ctors : (string * elttype list) list) =
  let what = "variant[" ^ pp_ctors ctors ^ "]" in
  let m_ctors = List.map (fun (_, args) -> List.map to_lfield args) ctors in
  let m_ok = Model.variant_accepted m_ctors in
  match Sarek_ir_layout.variant_layout ~type_name:"t" ctors with
  | Error e ->
      check_rejection_kind what e ;
      if m_ok then
        Alcotest.failf "%s: model accepts but Sarek_ir_layout rejects" what
  | Ok vl ->
      if not m_ok then
        Alcotest.failf "%s: Sarek_ir_layout accepts but model rejects" what ;
      let m_payoff = Model.variant_payload_offset m_ctors in
      Alcotest.(check int)
        (what ^ ": tag offset")
        Model.tag_offset
        vl.Sarek_ir_layout.vl_tag_offset ;
      Alcotest.(check int)
        (what ^ ": payload offset")
        m_payoff
        vl.Sarek_ir_layout.vl_payload_offset ;
      Alcotest.(check int)
        (what ^ ": total size")
        (Model.variant_size m_ctors)
        vl.Sarek_ir_layout.vl_size ;
      iteri2
        (what ^ ": ctors")
        (fun k (m : Model.ctor_layout) (o : Sarek_ir_layout.ctor_layout) ->
          Alcotest.(check int)
            (Printf.sprintf "%s: ctor %d tag" what k)
            m.Model.cl_tag
            o.Sarek_ir_layout.ctor_tag ;
          Alcotest.(check int)
            (Printf.sprintf "%s: ctor %d tag=index" what k)
            k
            o.Sarek_ir_layout.ctor_tag ;
          Alcotest.(check int)
            (Printf.sprintf "%s: ctor %d payload size" what k)
            m.Model.cl_payload_size
            o.Sarek_ir_layout.ctor_payload_size ;
          check_leaves
            (Printf.sprintf "%s: ctor %d" what k)
            m.Model.cl_leaves
            o.Sarek_ir_layout.ctor_leaves)
        (Model.ctor_layouts m_payoff 0 m_ctors)
        vl.Sarek_ir_layout.vl_ctors

(* ======================================================================= *)
(** * 4. Exhaustive small-shape enumeration *)
(* ======================================================================= *)

(* All lists of length [n] over [universe]. *)
let rec tuples universe n =
  if n = 0 then [[]]
  else
    List.concat_map
      (fun t -> List.map (fun r -> t :: r) (tuples universe (n - 1)))
      universe

let name_fields ts = List.mapi (fun i t -> ("f" ^ string_of_int i, t)) ts

let test_exhaustive_records () =
  let count = ref 0 in
  List.iter
    (fun n ->
      List.iter
        (fun ts ->
          incr count ;
          check_record_agreement (name_fields ts))
        (tuples scalar_universe n))
    [1; 2; 3; 4] ;
  Alcotest.(check int) "record shape count" 780 !count

let test_exhaustive_variants () =
  (* Constructor arg lists: length 0..2 over the scalar universe. *)
  let arg_options =
    tuples scalar_universe 0 @ tuples scalar_universe 1
    @ tuples scalar_universe 2
  in
  let count = ref 0 in
  List.iter
    (fun k ->
      List.iter
        (fun ctor_args ->
          incr count ;
          check_variant_agreement
            (List.mapi (fun i args -> ("C" ^ string_of_int i, args)) ctor_args))
        (tuples arg_options k))
    [1; 2; 3] ;
  Alcotest.(check int) "variant shape count" 30783 !count

(* ======================================================================= *)
(** * 5. Seeded qcheck-random nested-record shapes (depth <= 3) *)
(* ======================================================================= *)

let scalar_gen = QCheck.Gen.oneof_list scalar_universe

(* Field type at remaining nesting [depth]: scalar, or a nested record. *)
let rec field_gen depth =
  if depth = 0 then scalar_gen
  else
    QCheck.Gen.oneof_weighted
      [
        (3, scalar_gen);
        ( 1,
          QCheck.Gen.map
            (fun fs -> TRecord ("nested", fs))
            (fields_gen (depth - 1)) );
      ]

and fields_gen depth =
  QCheck.Gen.map
    name_fields
    (QCheck.Gen.list_size (QCheck.Gen.int_range 1 4) (field_gen depth))

let test_random_nested_records () =
  let rand = Random.State.make [|0x5a7e; 2026; 7; 22|] in
  (* Top level + two nested levels = depth <= 3. *)
  let shapes = QCheck.Gen.generate ~rand ~n:500 (fields_gen 2) in
  Alcotest.(check int) "random case count" 500 (List.length shapes) ;
  List.iter check_record_agreement shapes

(* ======================================================================= *)
(** * 6. Host pins — literal offsets/sizes for the e2e test types *)
(* ======================================================================= *)

(* Asserts a record layout literally on BOTH the extracted model and Sarek_ir_layout. *)
let pin_record name fields expected_offsets expected_size =
  let fs = to_lfields (List.map snd fields) in
  Alcotest.(check bool)
    (name ^ ": model accepts")
    true
    (Model.record_accepted fs) ;
  Alcotest.(check (list int))
    (name ^ ": model offsets")
    expected_offsets
    (List.map (fun (l : Model.leaf) -> l.Model.lf_off) (Model.record_leaves fs)) ;
  Alcotest.(check int)
    (name ^ ": model size")
    expected_size
    (Model.record_size fs) ;
  match Sarek_ir_layout.record_layout ~type_name:name fields with
  | Error e ->
      Alcotest.failf
        "%s: rejected: %s"
        name
        (Sarek_ir_layout.layout_error_message e)
  | Ok rl ->
      Alcotest.(check (list int))
        (name ^ ": OCaml offsets")
        expected_offsets
        (List.map
           (fun (l : Sarek_ir_layout.leaf) -> l.Sarek_ir_layout.leaf_offset)
           rl.Sarek_ir_layout.rl_leaves) ;
      Alcotest.(check int)
        (name ^ ": OCaml size")
        expected_size
        rl.Sarek_ir_layout.rl_size

let test_pin_point () =
  pin_record "point" [("x", TFloat32); ("y", TFloat32)] [0; 4] 8

let test_pin_point3d () =
  pin_record
    "point3d"
    [("x", TFloat32); ("y", TFloat32); ("z", TFloat32)]
    [0; 4; 8]
    12

let test_pin_particle () =
  pin_record
    "particle"
    [
      ("px", TFloat32);
      ("py", TFloat32);
      ("vx", TFloat32);
      ("vy", TFloat32);
      ("mass", TFloat32);
    ]
    [0; 4; 8; 12; 16]
    20

(* color = Red | Value of float32 : tag@0, payload@4, size 8, decl-order tags. *)
let test_pin_color () =
  let ctors = [("Red", []); ("Value", [TFloat32])] in
  let m_ctors = [[]; [Model.LLeaf Model.L32]] in
  Alcotest.(check bool)
    "color: model accepts"
    true
    (Model.variant_accepted m_ctors) ;
  Alcotest.(check int) "color: model size" 8 (Model.variant_size m_ctors) ;
  Alcotest.(check (list int))
    "color: model tags"
    [0; 1]
    (List.map
       (fun (c : Model.ctor_layout) -> c.Model.cl_tag)
       (Model.ctor_layouts (Model.variant_payload_offset m_ctors) 0 m_ctors)) ;
  match Sarek_ir_layout.variant_layout ~type_name:"color" ctors with
  | Error e ->
      Alcotest.failf
        "color: rejected: %s"
        (Sarek_ir_layout.layout_error_message e)
  | Ok vl ->
      Alcotest.(check int)
        "color: tag offset"
        0
        vl.Sarek_ir_layout.vl_tag_offset ;
      Alcotest.(check int)
        "color: payload offset"
        4
        vl.Sarek_ir_layout.vl_payload_offset ;
      Alcotest.(check int) "color: size" 8 vl.Sarek_ir_layout.vl_size ;
      Alcotest.(check (list int))
        "color: tags decl-order"
        [0; 1]
        (List.map
           (fun (c : Sarek_ir_layout.ctor_layout) -> c.Sarek_ir_layout.ctor_tag)
           vl.Sarek_ir_layout.vl_ctors) ;
      let value_ctor = List.nth vl.Sarek_ir_layout.vl_ctors 1 in
      Alcotest.(check (list int))
        "color: Value payload offsets"
        [4]
        (List.map
           (fun (l : Sarek_ir_layout.leaf) -> l.Sarek_ir_layout.leaf_offset)
           value_ctor.Sarek_ir_layout.ctor_leaves) ;
      Alcotest.(check int)
        "color: Value payload size"
        4
        value_ctor.Sarek_ir_layout.ctor_payload_size

(* L8: mixed-alignment record {a:i32; b:f64} — a@0, b@8 (pad), size 16. Both
   the extracted model and Sarek_ir_layout must agree on the aligned offsets. *)
let test_pin_mixed_i32_f64 () =
  pin_record "mixed_i32_f64" [("a", TInt32); ("b", TFloat64)] [0; 8] 16

(* L8: {flag:bool; d:f64} — bool is 4 bytes on the host, d@8, size 16. *)
let test_pin_bool_f64 () =
  pin_record "bool_f64" [("flag", TBool); ("d", TFloat64)] [0; 8] 16

(* L8: reordered largest-first {b:f64; a:i32} — b@0, a@8, size 16. *)
let test_pin_f64_i32 () =
  pin_record "f64_i32" [("b", TFloat64); ("a", TInt32)] [0; 8] 16

(* L8: variant with an f64 payload — payload region @8, Some_._0@8, size 16. *)
let test_pin_f64_variant () =
  let ctors = [("None_", []); ("Some_", [TFloat64])] in
  let m_ctors = [[]; [Model.LLeaf Model.L64]] in
  Alcotest.(check bool)
    "f64_variant: model accepts"
    true
    (Model.variant_accepted m_ctors) ;
  Alcotest.(check int)
    "f64_variant: model payload offset"
    8
    (Model.variant_payload_offset m_ctors) ;
  Alcotest.(check int) "f64_variant: model size" 16 (Model.variant_size m_ctors) ;
  match Sarek_ir_layout.variant_layout ~type_name:"f64_variant" ctors with
  | Error e ->
      Alcotest.failf
        "f64_variant: rejected: %s"
        (Sarek_ir_layout.layout_error_message e)
  | Ok vl ->
      Alcotest.(check int)
        "f64_variant: payload offset"
        8
        vl.Sarek_ir_layout.vl_payload_offset ;
      Alcotest.(check int) "f64_variant: size" 16 vl.Sarek_ir_layout.vl_size ;
      let some = List.nth vl.Sarek_ir_layout.vl_ctors 1 in
      Alcotest.(check (list int))
        "f64_variant: Some_ payload offsets"
        [8]
        (List.map
           (fun (l : Sarek_ir_layout.leaf) -> l.Sarek_ir_layout.leaf_offset)
           some.Sarek_ir_layout.ctor_leaves)

(* ======================================================================= *)
(** * 7. Suite *)
(* ======================================================================= *)

let () =
  Alcotest.run
    "layout-conformance"
    [
      ( "exhaustive",
        [
          Alcotest.test_case
            "records <=4 fields over 5 scalars (780 shapes)"
            `Quick
            test_exhaustive_records;
          Alcotest.test_case
            "variants <=3 ctors x <=2 args (30783 shapes)"
            `Quick
            test_exhaustive_variants;
        ] );
      ( "random",
        [
          Alcotest.test_case
            "nested records depth <=3 (500 seeded cases)"
            `Quick
            test_random_nested_records;
        ] );
      ( "host-pins",
        [
          Alcotest.test_case "point" `Quick test_pin_point;
          Alcotest.test_case "point3d" `Quick test_pin_point3d;
          Alcotest.test_case "color" `Quick test_pin_color;
          Alcotest.test_case "particle" `Quick test_pin_particle;
        ] );
      ( "mixed-alignment pins (L8)",
        [
          Alcotest.test_case
            "{i32;f64} a@0 b@8 size 16"
            `Quick
            test_pin_mixed_i32_f64;
          Alcotest.test_case "{bool;f64} d@8 size 16" `Quick test_pin_bool_f64;
          Alcotest.test_case
            "{f64;i32} reordered b@0 a@8 size 16"
            `Quick
            test_pin_f64_i32;
          Alcotest.test_case
            "variant f64 payload@8 size 16"
            `Quick
            test_pin_f64_variant;
        ] );
    ]
