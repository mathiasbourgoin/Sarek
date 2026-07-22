(******************************************************************************)
(* test_layout_conformance.ml
 *
 * CMBT conformance harness for the packed aggregate layout model (FR-042).
 *
 * Strategy: [RocqMirror] below is a line-by-line OCaml transcription of the
 * definitions in theories/PtxLayout.v (each function comments its .v source).
 * The suite then checks that the mirror and the production layout module
 * [Sarek_ir_layout] agree on accept/reject AND on all numeric offsets/sizes:
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
(** * 1. RocqMirror — hand transcription of theories/PtxLayout.v *)
(* ======================================================================= *)

module RocqMirror = struct
  (* PtxLayout.v [lty]: scalar universe, 4-byte vs 8-byte. *)
  type lty = L32 | L64

  (* PtxLayout.v [scalar_size]. *)
  let scalar_size = function L32 -> 4 | L64 -> 8

  (* PtxLayout.v [scalar_align] (= size for both widths). *)
  let scalar_align = scalar_size

  (* PtxLayout.v [lfield]/[lfields]: record fields are scalar leaves or
     nested records; no variant constructor exists below top level. *)
  type lfield = LLeaf of lty | LRec of lfields

  and lfields = LNil | LCons of int * lfield * lfields

  (* PtxLayout.v [fsize]/[fssize]: packed byte size, no padding. *)
  let rec fsize = function LLeaf t -> scalar_size t | LRec fs -> fssize fs

  and fssize = function LNil -> 0 | LCons (_, f, r) -> fsize f + fssize r

  (* PtxLayout.v [leaf] (lf_path / lf_ty / lf_off). *)
  type leaf = {lf_path : int list; lf_ty : lty; lf_off : int}

  (* PtxLayout.v [leaf_size]. *)
  let leaf_size l = scalar_size l.lf_ty

  (* PtxLayout.v [flatten]/[flattens]: declaration-order leaves at packed
     cumulative offsets. *)
  let rec flatten p off = function
    | LLeaf t -> [{lf_path = p; lf_ty = t; lf_off = off}]
    | LRec fs -> flattens p off fs

  and flattens p off = function
    | LNil -> []
    | LCons (n, f, r) -> flatten (p @ [n]) off f @ flattens p (off + fsize f) r

  (* PtxLayout.v [record_leaves] / [record_size]. *)
  let record_leaves fs = flattens [] 0 fs

  let record_size = fssize

  (* PtxLayout.v [field_offsets] / [record_field_offsets]. *)
  let rec field_offsets off = function
    | LNil -> []
    | LCons (n, f, r) -> (n, off) :: field_offsets (off + fsize f) r

  let record_field_offsets fs = field_offsets 0 fs

  (* PtxLayout.v [tag_offset]/[tag_size]/[payload_offset]. *)
  let tag_offset = 0

  let tag_size = 4

  let payload_offset = 4

  (* PtxLayout.v [ctor_layout]. *)
  type ctor_layout = {
    cl_tag : int;
    cl_leaves : leaf list;
    cl_payload_size : int;
  }

  (* PtxLayout.v [number_args]: positional slots _0, _1, ... *)
  let rec number_args i = function
    | [] -> LNil
    | a :: r -> LCons (i, a, number_args (i + 1) r)

  (* PtxLayout.v [ctor_layout_of]: payload packed from [payload_offset]. *)
  let ctor_layout_of tag args =
    let fs = number_args 0 args in
    {
      cl_tag = tag;
      cl_leaves = flattens [tag] payload_offset fs;
      cl_payload_size = fssize fs;
    }

  (* PtxLayout.v [ctor_layouts]: tag = declaration index. *)
  let rec ctor_layouts tag = function
    | [] -> []
    | c :: r -> ctor_layout_of tag c :: ctor_layouts (tag + 1) r

  (* PtxLayout.v [max_payload]. *)
  let max_payload cls =
    List.fold_right (fun c acc -> max c.cl_payload_size acc) cls 0

  (* PtxLayout.v [variant_size]: 4-byte int32 tag + max payload. *)
  let variant_size ctors = tag_size + max_payload (ctor_layouts 0 ctors)

  (* PtxLayout.v [leaf_aligned]: natural alignment of the absolute offset. *)
  let leaf_aligned l = l.lf_off mod scalar_align l.lf_ty = 0

  (* PtxLayout.v [record_accepted]. *)
  let record_accepted fs = List.for_all leaf_aligned (record_leaves fs)

  (* PtxLayout.v [variant_accepted]. *)
  let variant_accepted ctors =
    List.for_all
      (fun c -> List.for_all leaf_aligned c.cl_leaves)
      (ctor_layouts 0 ctors)
end

(* ======================================================================= *)
(** * 2. Bridge: Sarek_ir_types.elttype shapes -> RocqMirror encoding *)
(* ======================================================================= *)

(* Scalar universe of the conformance domain (TUnit excluded: not part of
   the pinned e2e types nor of the theory's motivating universe). *)
let scalar_universe = [TInt32; TFloat32; TBool; TInt64; TFloat64]

let rec to_lfield (t : elttype) : RocqMirror.lfield =
  match t with
  | TInt32 | TFloat32 | TBool -> RocqMirror.LLeaf RocqMirror.L32
  | TInt64 | TFloat64 -> RocqMirror.LLeaf RocqMirror.L64
  | TRecord (_, fields) -> RocqMirror.LRec (to_lfields (List.map snd fields))
  | TVariant _ | TArray _ | TVec _ | TUnit ->
      invalid_arg "to_lfield: outside the conformance domain"

and to_lfields (ts : elttype list) : RocqMirror.lfields =
  List.fold_right
    (fun (i, t) acc -> RocqMirror.LCons (i, to_lfield t, acc))
    (List.mapi (fun i t -> (i, t)) ts)
    RocqMirror.LNil

let rec pp_elttype = function
  | TInt32 -> "i32"
  | TInt64 -> "i64"
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

let check_leaves what (m_leaves : RocqMirror.leaf list)
    (o_leaves : Sarek_ir_layout.leaf list) =
  iteri2
    (what ^ ": leaves")
    (fun k (m : RocqMirror.leaf) (o : Sarek_ir_layout.leaf) ->
      Alcotest.(check int)
        (Printf.sprintf "%s: leaf %d offset" what k)
        m.RocqMirror.lf_off
        o.Sarek_ir_layout.leaf_offset ;
      Alcotest.(check int)
        (Printf.sprintf "%s: leaf %d size" what k)
        (RocqMirror.leaf_size m)
        o.Sarek_ir_layout.leaf_size)
    m_leaves
    o_leaves

(* Mirror-vs-OCaml agreement for one record shape. *)
let check_record_agreement (fields : (string * elttype) list) =
  let what = "record{" ^ pp_fields fields ^ "}" in
  let fs = to_lfields (List.map snd fields) in
  let m_ok = RocqMirror.record_accepted fs in
  match Sarek_ir_layout.record_layout ~type_name:"t" fields with
  | Error e ->
      check_rejection_kind what e ;
      if m_ok then
        Alcotest.failf "%s: mirror accepts but Sarek_ir_layout rejects" what
  | Ok rl ->
      if not m_ok then
        Alcotest.failf "%s: Sarek_ir_layout accepts but mirror rejects" what ;
      Alcotest.(check int)
        (what ^ ": total size")
        (RocqMirror.record_size fs)
        rl.Sarek_ir_layout.rl_size ;
      check_leaves
        what
        (RocqMirror.record_leaves fs)
        rl.Sarek_ir_layout.rl_leaves ;
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
        (RocqMirror.record_field_offsets fs)
        rl.Sarek_ir_layout.rl_fields

(* Mirror-vs-OCaml agreement for one variant shape. *)
let check_variant_agreement (ctors : (string * elttype list) list) =
  let what = "variant[" ^ pp_ctors ctors ^ "]" in
  let m_ctors = List.map (fun (_, args) -> List.map to_lfield args) ctors in
  let m_ok = RocqMirror.variant_accepted m_ctors in
  match Sarek_ir_layout.variant_layout ~type_name:"t" ctors with
  | Error e ->
      check_rejection_kind what e ;
      if m_ok then
        Alcotest.failf "%s: mirror accepts but Sarek_ir_layout rejects" what
  | Ok vl ->
      if not m_ok then
        Alcotest.failf "%s: Sarek_ir_layout accepts but mirror rejects" what ;
      Alcotest.(check int)
        (what ^ ": tag offset")
        RocqMirror.tag_offset
        vl.Sarek_ir_layout.vl_tag_offset ;
      Alcotest.(check int)
        (what ^ ": payload offset")
        RocqMirror.payload_offset
        vl.Sarek_ir_layout.vl_payload_offset ;
      Alcotest.(check int)
        (what ^ ": total size")
        (RocqMirror.variant_size m_ctors)
        vl.Sarek_ir_layout.vl_size ;
      iteri2
        (what ^ ": ctors")
        (fun k (m : RocqMirror.ctor_layout) (o : Sarek_ir_layout.ctor_layout) ->
          Alcotest.(check int)
            (Printf.sprintf "%s: ctor %d tag" what k)
            m.RocqMirror.cl_tag
            o.Sarek_ir_layout.ctor_tag ;
          Alcotest.(check int)
            (Printf.sprintf "%s: ctor %d tag=index" what k)
            k
            o.Sarek_ir_layout.ctor_tag ;
          Alcotest.(check int)
            (Printf.sprintf "%s: ctor %d payload size" what k)
            m.RocqMirror.cl_payload_size
            o.Sarek_ir_layout.ctor_payload_size ;
          check_leaves
            (Printf.sprintf "%s: ctor %d" what k)
            m.RocqMirror.cl_leaves
            o.Sarek_ir_layout.ctor_leaves)
        (RocqMirror.ctor_layouts 0 m_ctors)
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

(* Asserts a record layout literally on BOTH the mirror and Sarek_ir_layout. *)
let pin_record name fields expected_offsets expected_size =
  let fs = to_lfields (List.map snd fields) in
  Alcotest.(check bool)
    (name ^ ": mirror accepts")
    true
    (RocqMirror.record_accepted fs) ;
  Alcotest.(check (list int))
    (name ^ ": mirror offsets")
    expected_offsets
    (List.map
       (fun (l : RocqMirror.leaf) -> l.RocqMirror.lf_off)
       (RocqMirror.record_leaves fs)) ;
  Alcotest.(check int)
    (name ^ ": mirror size")
    expected_size
    (RocqMirror.record_size fs) ;
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
  let m_ctors = [[]; [RocqMirror.LLeaf RocqMirror.L32]] in
  Alcotest.(check bool)
    "color: mirror accepts"
    true
    (RocqMirror.variant_accepted m_ctors) ;
  Alcotest.(check int) "color: mirror size" 8 (RocqMirror.variant_size m_ctors) ;
  Alcotest.(check (list int))
    "color: mirror tags"
    [0; 1]
    (List.map
       (fun (c : RocqMirror.ctor_layout) -> c.RocqMirror.cl_tag)
       (RocqMirror.ctor_layouts 0 m_ctors)) ;
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
    ]
