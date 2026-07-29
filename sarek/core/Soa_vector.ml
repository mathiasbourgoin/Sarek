(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* See Soa_vector.mli for the design rationale. *)

type packed_leaf = Leaf : ('e, 'f) Vector.t -> packed_leaf

type 'a t = {
  aos : ('a, unit) Vector.t;
  plan : Soa.plan;
  leaves : packed_leaf array;
  length : int;
}

(* A leaf host buffer is a bit-preserving byte transport sized to the leaf's
   scalar width; Soa.scatter/gather copy raw 4/8-byte words, so an int32/int64
   scalar vector of the right width is a correct container for any leaf type
   (f32/f64/i32/i64). *)
let leaf_vector_of_size length size : packed_leaf =
  match size with
  | 4 -> Leaf (Vector.create Vector.int32 length)
  | 8 -> Leaf (Vector.create Vector.int64 length)
  | _ ->
      invalid_arg
        (Printf.sprintf
           "Soa_vector: unsupported leaf byte size %d (expected 4 or 8)"
           size)

(* The leaf layout is DERIVED from the element type, never supplied. [create]
   used to take a [~fields] list, on the stated premise that "the PPX
   [custom_type] carries no layout" — which stopped being true once
   [custom_type.ir_fields] landed. The PPX populates it for every
   [[@@sarek.type]] record (Sarek_ppx.ml), from the same
   [aligned_record_offsets] call that produces [elem_size]/[get]/[set], and
   [test_ir_fields.ml] pins that agreement against the bytes [set] actually
   writes.

   Deriving it removes a hazard rather than documenting one: a caller-supplied
   list that disagreed with the real record (wrong order, wrong widths,
   missing/extra field) made scatter/gather transpose against the wrong byte
   offsets, which is silently corrupted data and not an error. That failure mode
   is now unreachable — there is no longer a second description of the layout to
   disagree with the first.

   [ir_fields = None] means "no derivable flat-scalar layout", which the
   [custom_type] doc requires consumers to read as "no SoA". The only producer
   that sets it is the variant deriver, and [Soa.plan] rejects a non-flat-record
   anyway, so this refuses exactly the element types SoA could never represent. *)
let create (custom : 'a Vector.custom_type) (length : int) : 'a t =
  let fields =
    match custom.Vector.ir_fields with
    | Some fields -> fields
    | None ->
        raise
          (Soa.Unsupported
             (Printf.sprintf
                "element type %S has no derivable flat-record layout \
                 (custom_type.ir_fields is None), so it cannot be stored as \
                 Structure-of-Arrays"
                custom.Vector.name))
  in
  let plan = Soa.plan ~name:custom.Vector.name fields in
  let aos = Vector.create_custom custom length in
  let leaves =
    Array.of_list
      (List.map
         (fun (l : Soa.leaf) -> leaf_vector_of_size length l.Soa.size)
         plan.Soa.leaves)
  in
  {aos; plan; leaves; length}

let aos_vector t = t.aos

let plan t = t.plan

let leaves t = t.leaves

let num_leaves t = Array.length t.leaves

let length t = t.length

let set t i v = Vector.set t.aos i v

let get t i = Vector.get t.aos i

let leaf_ptrs t = Array.map (fun (Leaf v) -> Vector.to_ctypes_ptr v) t.leaves

let scatter t =
  (* Make the AoS host copy authoritative before transposing out of it. *)
  Vector.ensure_cpu_sync t.aos ;
  Soa.scatter
    t.plan
    ~aos:(Vector.to_ctypes_ptr t.aos)
    ~length:t.length
    ~leaves:(leaf_ptrs t)

let gather t =
  Soa.gather
    t.plan
    ~leaves:(leaf_ptrs t)
    ~length:t.length
    ~aos:(Vector.to_ctypes_ptr t.aos)
