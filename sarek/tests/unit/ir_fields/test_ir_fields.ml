(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** [custom_type.ir_fields] — the SoA field list, checked against the bytes the
    generated accessors actually touch (backlog-54 slice 2).

    {1 What this file is defending}

    [ir_fields] is untyped metadata sitting next to [elem_size], [get] and
    [set] in a plain record. Nothing in the type system relates them. If the
    PPX's [ir_elttype_of_core_type] maps a field to the wrong [elttype], the
    result is not a type error and not a crash: [Soa.plan] computes a different
    offset or a different width than [set] wrote to, the SoA transposition moves
    the wrong bytes, and the kernel reads plausible garbage. That is the
    wrong-width failure family (backlog-85/-96/-139/-141/-142), and prose
    review does not catch it.

    Two arms of the mapping are traps and are the reason this file exists:

    - OCaml [float] is 8 bytes {i in OCaml} but this framework marshals a
      [float] record field as a 32-bit GPU float ([read_float32], size 4). It
      maps to [TFloat32]. Reading the OCaml type and answering [TFloat64] is
      the natural mistake and doubles every stride below it.
    - [int] marshals through [read_int], which is
      [Int32.to_int (read_int32 ...)] — 4 bytes, not an OCaml 63-bit int. It
      maps to [TInt32].

    {1 The gate}

    [check_layout] does not compare the mapping against a second copy of the
    mapping — that would be a tautology, and two copies of a wrong table agree
    perfectly. It compares against the {b observable behaviour of the generated
    [set]}: write one field with an all-[0x7F] bit pattern into a zeroed
    element and record which byte indices became nonzero. The set of dirtied
    bytes must be exactly the half-open range [[aos_offset, aos_offset + size)]
    that [Soa.plan] derived from [ir_fields]. Field {i order} is pinned the same
    way, because a swapped pair dirties the other field's byte range.

    So the assertion chain is

    [ir_fields] -> [Soa.plan] -> predicted (offset, size) per leaf
    vs. bytes [set] really wrote, and [aos_stride] vs. [elem_size].

    Both ends are independently derived; agreement is evidence, not restatement.

    {1 Coverage}

    Enumerated, not sampled, over the mapping's domain: all six field types the
    marshaller supports appear as the sole field of a record, which pins each
    width in isolation ({!singleton_cases}). On top of that, {!pair_cases}
    covers every ordered 4-byte/8-byte alignment-class pair, which is where
    padding is inserted and where an offset error shows up; plus a three-field
    record for the general case. [bool] and [unit] are absent because
    [gen_field_read] has no arm for them — they are unusable as record fields
    today (see the note in [Sarek_ppx.ir_elttype_of_core_type]).

    Not covered here: any device. This is a pure host-layout test. *)

module Soa = Spoc_core.Soa
module Vector = Spoc_core.Vector
module Helpers = Spoc_core.Vector_types.Custom_helpers
open Sarek_ir_types

type float32 = float

type float64 = float

(** {1 Element types} *)

(* One field each: pins the width of every arm of the mapping in isolation. *)

type r_i32 = {i32_f : int32} [@@sarek.type]

type r_int = {int_f : int} [@@sarek.type]

type r_i64 = {i64_f : int64} [@@sarek.type]

type r_f32 = {f32_f : float32} [@@sarek.type]

type r_flt = {flt_f : float} [@@sarek.type]

type r_f64 = {f64_f : float64} [@@sarek.type]

(* Ordered 4/8-byte pairs: padding is inserted only when an 8-byte field
   follows a 4-byte one, and the struct tail is padded only when an 8-byte
   member is present, so both orders of each pair are distinct layouts. *)

type p_i32_i64 = {a_i32 : int32; b_i64 : int64} [@@sarek.type]

type p_i64_i32 = {a_i64 : int64; b_i32 : int32} [@@sarek.type]

type p_f32_f64 = {a_f32 : float32; b_f64 : float64} [@@sarek.type]

type p_f64_f32 = {a_f64 : float64; b_f32 : float32} [@@sarek.type]

type p_int_f64 = {a_int : int; b_f64d : float64} [@@sarek.type]

type p_flt_i64 = {a_flt : float; b_i64d : int64} [@@sarek.type]

(* Three fields, mixed alignment: the general case. *)
type t_mixed = {m_i32 : int32; m_f64 : float64; m_f32 : float32} [@@sarek.type]

(* All-4-byte: the historically common shape, where aligned == packed. *)
type t_flat3 = {v_x : float32; v_y : float32; v_z : float32} [@@sarek.type]

(* Not SoA-derivable: a nested custom-type field. [gen_field_read] handles it
   (via the nested-descriptor branch) so it compiles, but Sarek_ir_layout
   rejects nested records for v1 SoA, so the field list must be withheld. *)
type nested_outer = {n_head : float32; n_body : t_flat3} [@@sarek.type]

(* Not SoA-derivable: a variant is a tagged union, not a flat scalar record. *)
type v_tag = VA | VB of float32 [@@sarek.type]

(** {1 Byte probing} *)

(* All bytes 0x7F: nonzero in every byte, and a finite (non-NaN) float in both
   widths, so nothing normalises it away on the way through [set]. *)
let hot_i32 = 0x7F7F7F7Fl

let hot_i64 = 0x7F7F7F7F7F7F7F7FL

let hot_int = 0x7F7F7F7F

let hot_f32 = Int32.float_of_bits hot_i32

let hot_f64 = Int64.float_of_bits hot_i64

let alloc_zeroed n =
  let p = Ctypes.allocate_n Ctypes.uint8_t ~count:n in
  for i = 0 to n - 1 do
    Ctypes.(p +@ i <-@ Unsigned.UInt8.zero)
  done ;
  Ctypes.to_voidp p

(* Byte indices of [ptr.(0 .. n-1)] that are nonzero. *)
let dirty_bytes ptr n =
  let bp = Ctypes.from_voidp Ctypes.uint8_t ptr in
  let acc = ref [] in
  for i = n - 1 downto 0 do
    if Unsigned.UInt8.compare Ctypes.(!@(bp +@ i)) Unsigned.UInt8.zero <> 0 then
      acc := i :: !acc
  done ;
  !acc

let range_list off size = List.init size (fun k -> off + k)

(* Every constructor spelled out, no wildcard: a new elttype must be considered
   here (is it a legal record field? does the PPX map it?) rather than silently
   absorbed into a catch-all. TUint8 arriving with backlog-62 slice 3 made this
   file fail to compile, which is the intended behaviour. *)
let string_of_elttype = function
  | TInt32 -> "TInt32"
  | TInt64 -> "TInt64"
  | TFloat16 -> "TFloat16"
  | TUint8 -> "TUint8"
  | TFloat32 -> "TFloat32"
  | TFloat64 -> "TFloat64"
  | TBool -> "TBool"
  | TUnit -> "TUnit"
  | TRecord (n, _) -> "TRecord " ^ n
  | TVariant (n, _) -> "TVariant " ^ n
  | TArray _ -> "TArray"
  | TVec _ -> "TVec"

(** [check_layout ~custom ~expected ~probes] is the whole gate.

    [probes] must be in the same order as the type's fields: [probes.(k)] writes
    element 0 with field [k] hot and every other field zero. *)
let check_layout ~name ~custom ~expected ~probes () =
  (* 1. The PPX emitted a field list at all, and it is the expected one. *)
  let fields =
    match custom.Vector.ir_fields with
    | Some f -> f
    | None ->
        Alcotest.failf "%s: ir_fields is None, expected a derivable record" name
  in
  let show = List.map (fun (n, t) -> (n, string_of_elttype t)) in
  Alcotest.(check (list (pair string string)))
    (name ^ ": ir_fields")
    (show expected)
    (show fields) ;

  (* 2. The plan derived from that list must agree with the element size the
     accessors index by. A stride disagreement corrupts every element past the
     first, so it is checked separately from the per-field offsets. *)
  let plan = Soa.plan ~name:custom.Vector.name fields in
  Alcotest.(check int)
    (name ^ ": aos_stride = elem_size")
    custom.Vector.elem_size
    plan.Soa.aos_stride ;

  let leaves = Array.of_list plan.Soa.leaves in
  Alcotest.(check int)
    (name ^ ": one leaf per field")
    (List.length fields)
    (Array.length leaves) ;
  Alcotest.(check int)
    (name ^ ": one probe per field")
    (List.length fields)
    (Array.length probes) ;

  (* 3. The bytes [set] actually writes must be exactly the bytes the plan
     predicted. This is the part that cannot be satisfied by a second copy of a
     wrong mapping table. *)
  Array.iteri
    (fun k (leaf : Soa.leaf) ->
      let ptr = alloc_zeroed custom.Vector.elem_size in
      probes.(k) ptr ;
      let got = dirty_bytes ptr custom.Vector.elem_size in
      let want = range_list leaf.Soa.aos_offset leaf.Soa.size in
      Alcotest.(check (list int))
        (Printf.sprintf
           "%s: field %d (%s) dirties bytes [%d,%d)"
           name
           k
           leaf.Soa.path
           leaf.Soa.aos_offset
           (leaf.Soa.aos_offset + leaf.Soa.size))
        want
        got)
    leaves

(** {1 Cases} *)

let singleton_cases =
  [
    ( "r_i32",
      check_layout
        ~name:"r_i32"
        ~custom:r_i32_custom
        ~expected:[("i32_f", TInt32)]
        ~probes:[|(fun p -> r_i32_custom.Vector.set p 0 {i32_f = hot_i32})|] );
    ( "r_int",
      check_layout
        ~name:"r_int"
        ~custom:r_int_custom
        ~expected:[("int_f", TInt32)]
        ~probes:[|(fun p -> r_int_custom.Vector.set p 0 {int_f = hot_int})|] );
    ( "r_i64",
      check_layout
        ~name:"r_i64"
        ~custom:r_i64_custom
        ~expected:[("i64_f", TInt64)]
        ~probes:[|(fun p -> r_i64_custom.Vector.set p 0 {i64_f = hot_i64})|] );
    ( "r_f32",
      check_layout
        ~name:"r_f32"
        ~custom:r_f32_custom
        ~expected:[("f32_f", TFloat32)]
        ~probes:[|(fun p -> r_f32_custom.Vector.set p 0 {f32_f = hot_f32})|] );
    ( "r_flt (OCaml float marshals as GPU float32)",
      check_layout
        ~name:"r_flt"
        ~custom:r_flt_custom
        ~expected:[("flt_f", TFloat32)]
        ~probes:[|(fun p -> r_flt_custom.Vector.set p 0 {flt_f = hot_f32})|] );
    ( "r_f64",
      check_layout
        ~name:"r_f64"
        ~custom:r_f64_custom
        ~expected:[("f64_f", TFloat64)]
        ~probes:[|(fun p -> r_f64_custom.Vector.set p 0 {f64_f = hot_f64})|] );
  ]

let pair_cases =
  [
    ( "p_i32_i64",
      check_layout
        ~name:"p_i32_i64"
        ~custom:p_i32_i64_custom
        ~expected:[("a_i32", TInt32); ("b_i64", TInt64)]
        ~probes:
          [|
            (fun p ->
              p_i32_i64_custom.Vector.set p 0 {a_i32 = hot_i32; b_i64 = 0L});
            (fun p ->
              p_i32_i64_custom.Vector.set p 0 {a_i32 = 0l; b_i64 = hot_i64});
          |] );
    ( "p_i64_i32",
      check_layout
        ~name:"p_i64_i32"
        ~custom:p_i64_i32_custom
        ~expected:[("a_i64", TInt64); ("b_i32", TInt32)]
        ~probes:
          [|
            (fun p ->
              p_i64_i32_custom.Vector.set p 0 {a_i64 = hot_i64; b_i32 = 0l});
            (fun p ->
              p_i64_i32_custom.Vector.set p 0 {a_i64 = 0L; b_i32 = hot_i32});
          |] );
    ( "p_f32_f64",
      check_layout
        ~name:"p_f32_f64"
        ~custom:p_f32_f64_custom
        ~expected:[("a_f32", TFloat32); ("b_f64", TFloat64)]
        ~probes:
          [|
            (fun p ->
              p_f32_f64_custom.Vector.set p 0 {a_f32 = hot_f32; b_f64 = 0.0});
            (fun p ->
              p_f32_f64_custom.Vector.set p 0 {a_f32 = 0.0; b_f64 = hot_f64});
          |] );
    ( "p_f64_f32",
      check_layout
        ~name:"p_f64_f32"
        ~custom:p_f64_f32_custom
        ~expected:[("a_f64", TFloat64); ("b_f32", TFloat32)]
        ~probes:
          [|
            (fun p ->
              p_f64_f32_custom.Vector.set p 0 {a_f64 = hot_f64; b_f32 = 0.0});
            (fun p ->
              p_f64_f32_custom.Vector.set p 0 {a_f64 = 0.0; b_f32 = hot_f32});
          |] );
    ( "p_int_f64",
      check_layout
        ~name:"p_int_f64"
        ~custom:p_int_f64_custom
        ~expected:[("a_int", TInt32); ("b_f64d", TFloat64)]
        ~probes:
          [|
            (fun p ->
              p_int_f64_custom.Vector.set p 0 {a_int = hot_int; b_f64d = 0.0});
            (fun p ->
              p_int_f64_custom.Vector.set p 0 {a_int = 0; b_f64d = hot_f64});
          |] );
    ( "p_flt_i64",
      check_layout
        ~name:"p_flt_i64"
        ~custom:p_flt_i64_custom
        ~expected:[("a_flt", TFloat32); ("b_i64d", TInt64)]
        ~probes:
          [|
            (fun p ->
              p_flt_i64_custom.Vector.set p 0 {a_flt = hot_f32; b_i64d = 0L});
            (fun p ->
              p_flt_i64_custom.Vector.set p 0 {a_flt = 0.0; b_i64d = hot_i64});
          |] );
  ]

let wide_cases =
  [
    ( "t_mixed",
      check_layout
        ~name:"t_mixed"
        ~custom:t_mixed_custom
        ~expected:[("m_i32", TInt32); ("m_f64", TFloat64); ("m_f32", TFloat32)]
        ~probes:
          [|
            (fun p ->
              t_mixed_custom.Vector.set
                p
                0
                {m_i32 = hot_i32; m_f64 = 0.0; m_f32 = 0.0});
            (fun p ->
              t_mixed_custom.Vector.set
                p
                0
                {m_i32 = 0l; m_f64 = hot_f64; m_f32 = 0.0});
            (fun p ->
              t_mixed_custom.Vector.set
                p
                0
                {m_i32 = 0l; m_f64 = 0.0; m_f32 = hot_f32});
          |] );
    ( "t_flat3",
      check_layout
        ~name:"t_flat3"
        ~custom:t_flat3_custom
        ~expected:[("v_x", TFloat32); ("v_y", TFloat32); ("v_z", TFloat32)]
        ~probes:
          [|
            (fun p ->
              t_flat3_custom.Vector.set
                p
                0
                {v_x = hot_f32; v_y = 0.0; v_z = 0.0});
            (fun p ->
              t_flat3_custom.Vector.set
                p
                0
                {v_x = 0.0; v_y = hot_f32; v_z = 0.0});
            (fun p ->
              t_flat3_custom.Vector.set
                p
                0
                {v_x = 0.0; v_y = 0.0; v_z = hot_f32});
          |] );
  ]

(** {1 Withholding} *)

(* [None] must mean "no SoA plan derivable", and a consumer must never read it
   as "a record with no fields" — the difference between falling back to AoS and
   transposing a zero-leaf plan over real data. *)

let test_nested_withheld () =
  Alcotest.(check bool)
    "nested-record field: ir_fields withheld"
    true
    (nested_outer_custom.Vector.ir_fields = None) ;
  (* And the withholding is not over-cautious noise: Sarek_ir_layout would in
     fact refuse this type, so [None] is the truthful answer rather than a
     missed opportunity. *)
  Alcotest.check_raises
    "and Soa.plan would have rejected it"
    (Soa.Unsupported
       "nested-record field \"n_body\" in \"nested_outer\": v1 SoA supports \
        flat records only")
    (fun () ->
      ignore
        (Soa.plan
           ~name:"nested_outer"
           [
             ("n_head", TFloat32);
             ( "n_body",
               TRecord
                 ( "t_flat3",
                   [("v_x", TFloat32); ("v_y", TFloat32); ("v_z", TFloat32)] )
             );
           ]))

let test_variant_withheld () =
  Alcotest.(check bool)
    "variant: ir_fields withheld"
    true
    (v_tag_custom.Vector.ir_fields = None)

(** {1 Runner} *)

let () =
  let case (n, f) = Alcotest.test_case n `Quick f in
  Alcotest.run
    "ir_fields"
    [
      ("width-per-type (enumerated)", List.map case singleton_cases);
      ("alignment pairs", List.map case pair_cases);
      ("multi-field", List.map case wide_cases);
      ( "not derivable",
        [
          Alcotest.test_case "nested record" `Quick test_nested_withheld;
          Alcotest.test_case "variant" `Quick test_variant_withheld;
        ] );
    ]
