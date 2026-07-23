(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * L13 — Tuple-typed vectors (host side).
 *
 * OCaml tuples are structural, so there is no declaration site to hang
 * [@@sarek.type] off and no PPX-generated [<name>_custom] to feed to
 * [Vector.create_custom]. This module builds the missing [custom_type] values
 * at runtime by composition: a tuple element is a packed record with
 * positional fields [_0], [_1], ... and its byte layout is taken from the
 * shared layout authority [Sarek_ir_layout], byte-for-byte identical to the
 * layout the device codegen computes for the synthesized record
 * ([Sarek_lower_ir.vector_elem_elttype]). Host [get]/[set] therefore marshal
 * to/from exactly the bytes the kernel reads and writes on device.
 *
 * Scope (this tier): tuples of two or three scalar-primitive components,
 * running on CUDA/PTX, OpenCL, Vulkan and Native. Native shares the [Type_id]
 * with generated kernel code through the process-wide registry below. The
 * Interpreter path needs value-model unification and is a follow-up (see
 * roster/ptx-limits-campaign/L13-tuple-vectors.md).
 ******************************************************************************)

module Vector = Spoc_core.Vector
module CH = Spoc_core.Vector.Custom_helpers
module Layout = Sarek_ir_layout
module Type_id = Sarek_ir_types.Type_id

(** A scalar component of a tuple element: its IR element type (for layout) and
    the typed raw-pointer read/write primitives used to marshal it at a byte
    offset. *)
type 'a component = {
  c_elttype : Sarek_ir_types.elttype;
  c_read : unit Ctypes.ptr -> int -> 'a;
  c_write : unit Ctypes.ptr -> int -> 'a -> unit;
  c_tag : string;
}

let float32 : float component =
  {
    c_elttype = Sarek_ir_types.TFloat32;
    c_read = CH.read_float32;
    c_write = CH.write_float32;
    c_tag = "float32";
  }

let float64 : float component =
  {
    c_elttype = Sarek_ir_types.TFloat64;
    c_read = CH.read_float64;
    c_write = CH.write_float64;
    c_tag = "float64";
  }

let int32 : int32 component =
  {
    c_elttype = Sarek_ir_types.TInt32;
    c_read = CH.read_int32;
    c_write = CH.write_int32;
    c_tag = "int32";
  }

let int64 : int64 component =
  {
    c_elttype = Sarek_ir_types.TInt64;
    c_read = CH.read_int64;
    c_write = CH.write_int64;
    c_tag = "int64";
  }

(** Mangled record name for a component tag list; kept in sync with
    [Sarek_lower_ir.tuple_record_name] so host and device agree on the element
    identity. *)
let mangled_name (tags : string list) : string =
  "_tup" ^ String.concat "" (List.map (fun t -> "_" ^ t) tags)

let field_name i = Printf.sprintf "_%d" i

let layout_of fields name =
  match Layout.record_layout ~type_name:name fields with
  | Ok rl -> rl
  | Error e ->
      failwith
        ("Sarek_tuple_vec: could not lay out " ^ name ^ ": "
        ^ Layout.layout_error_message e)

let offset_of rl fname =
  match List.assoc_opt fname rl.Layout.rl_fields with
  | Some off -> off
  | None -> failwith ("Sarek_tuple_vec: missing field offset for " ^ fname)

(* Process-global registry mapping a mangled tuple-shape name to its canonical
   [custom_type]. A shape is instantiated once and shared: the host builds a
   vector from it and the Native execution path looks up the very same
   descriptor (via [descriptor_by_name], emitted by the native code generator),
   so both sides carry the identical [Type_id] token that [vec_get_custom]
   compares with [Type_id.equal]. The value is stored type-erased and recovered
   at the call-site type; the mangled name uniquely determines the OCaml type,
   so the coercion is sound. *)
let registry : (string, Obj.t) Hashtbl.t = Hashtbl.create 16

(* Guards first-time registration: [find_opt] + [replace] is not atomic, and
   two concurrent callers racing on the same shape would each build a fresh
   [Type_id] with only one winning the registry slot — the loser's vector
   would then fail the [Type_id.equal] check against the canonical descriptor
   later resolved by generated Native code ([descriptor_by_name]). The mutex
   makes build-and-insert single-winner; the double-check inside the critical
   section resolves the race deterministically. *)
let registry_mutex = Mutex.create ()

let memoize name (build : unit -> 'a Vector.custom_type) : 'a Vector.custom_type
    =
  match Hashtbl.find_opt registry name with
  | Some obj -> Obj.obj obj
  | None ->
      Mutex.protect registry_mutex (fun () ->
          match Hashtbl.find_opt registry name with
          | Some obj -> Obj.obj obj
          | None ->
              let custom = build () in
              Hashtbl.replace registry name (Obj.repr custom) ;
              custom)

(** [descriptor_by_name name] returns the canonical [custom_type] previously
    registered for the mangled shape [name] (built by [pair]/[triple]). Used by
    generated Native kernel code to obtain the host-shared [Type_id]. Raises if
    the shape was never instantiated on the host. *)
let descriptor_by_name (name : string) : 'a Vector.custom_type =
  match Hashtbl.find_opt registry name with
  | Some obj -> Obj.obj obj
  | None ->
      failwith
        ("Sarek_tuple_vec: no tuple custom_type registered for shape '" ^ name
       ^ "'; build it on the host (e.g. Sarek_tuple_vec.pair ...) before \
          running the kernel.")

(** [pair a b] is the [custom_type] for a [(a, b)] tuple vector element. *)
let pair (a : 'a component) (b : 'b component) : ('a * 'b) Vector.custom_type =
  let name = mangled_name [a.c_tag; b.c_tag] in
  memoize name @@ fun () ->
  let fields = [(field_name 0, a.c_elttype); (field_name 1, b.c_elttype)] in
  let rl = layout_of fields name in
  let size = rl.Layout.rl_size in
  let o0 = offset_of rl (field_name 0) in
  let o1 = offset_of rl (field_name 1) in
  {
    Vector.elem_size = size;
    type_id = Type_id.create ();
    vector_type_id = Type_id.create ();
    name;
    get =
      (fun ptr idx ->
        let base = idx * size in
        (a.c_read ptr (base + o0), b.c_read ptr (base + o1)));
    set =
      (fun ptr idx (x, y) ->
        let base = idx * size in
        a.c_write ptr (base + o0) x ;
        b.c_write ptr (base + o1) y);
  }

(** [triple a b c] is the [custom_type] for a [(a, b, c)] tuple vector element.
*)
let triple (a : 'a component) (b : 'b component) (c : 'c component) :
    ('a * 'b * 'c) Vector.custom_type =
  let name = mangled_name [a.c_tag; b.c_tag; c.c_tag] in
  memoize name @@ fun () ->
  let fields =
    [
      (field_name 0, a.c_elttype);
      (field_name 1, b.c_elttype);
      (field_name 2, c.c_elttype);
    ]
  in
  let rl = layout_of fields name in
  let size = rl.Layout.rl_size in
  let o0 = offset_of rl (field_name 0) in
  let o1 = offset_of rl (field_name 1) in
  let o2 = offset_of rl (field_name 2) in
  {
    Vector.elem_size = size;
    type_id = Type_id.create ();
    vector_type_id = Type_id.create ();
    name;
    get =
      (fun ptr idx ->
        let base = idx * size in
        ( a.c_read ptr (base + o0),
          b.c_read ptr (base + o1),
          c.c_read ptr (base + o2) ));
    set =
      (fun ptr idx (x, y, z) ->
        let base = idx * size in
        a.c_write ptr (base + o0) x ;
        b.c_write ptr (base + o1) y ;
        c.c_write ptr (base + o2) z);
  }
