val fst : ('a1 * 'a2) -> 'a1

val snd : ('a1 * 'a2) -> 'a2

val app : 'a1 list -> 'a1 list -> 'a1 list

val add : int -> int -> int

val mul : int -> int -> int

val sub : int -> int -> int

module Nat : sig val sub : int -> int -> int

val max : int -> int -> int

val divmod : int -> int -> int -> int -> int * int

val div : int -> int -> int

val modulo : int -> int -> int end

val fold_right : ('a2 -> 'a1 -> 'a1) -> 'a1 -> 'a2 list -> 'a1

val forallb : ('a1 -> bool) -> 'a1 list -> bool

val align_up : int -> int -> int

type lty = | L32 | L64

val scalar_size : lty -> int

val scalar_align : lty -> int

type lfield = | LLeaf of lty | LRec of lfields and lfields = | LNil | LCons of
  int * lfield * lfields

val falign : lfield -> int

val fsalign : lfields -> int

val fsize : lfield -> int

val fsend : int -> lfields -> int

type leaf = { lf_path : int list; lf_ty : lty; lf_off : int }

val leaf_size : leaf -> int

val flatten : int list -> int -> lfield -> leaf list

val flattens : int list -> int -> lfields -> leaf list

val record_leaves : lfields -> leaf list

val record_size : lfields -> int

val field_offsets : int -> lfields -> (int * int) list

val record_field_offsets : lfields -> (int * int) list

val tag_offset : int

val tag_size : int

val number_args : int -> lfield list -> lfields

val ctor_align : lfield list -> int

val variant_payload_offset : lfield list list -> int

val payload_struct_size : lfield list -> int

type ctor_layout = { cl_tag : int; cl_leaves : leaf list; cl_payload_size :
  int }

val ctor_layouts : int -> int -> lfield list list -> ctor_layout list

val max_payload : ctor_layout list -> int

val variant_size : lfield list list -> int

val leaf_aligned : leaf -> bool

val record_accepted : lfields -> bool

val variant_accepted : lfield list list -> bool
