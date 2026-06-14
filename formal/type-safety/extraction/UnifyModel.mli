
val negb : bool -> bool

val length : 'a1 list -> int

module Nat :
 sig
 end

val map : ('a1 -> 'a2) -> 'a1 list -> 'a2 list

val existsb : ('a1 -> bool) -> 'a1 list -> bool

type prim_type =
| TUnit
| TBool
| TInt32

type reg_type =
| RInt
| RInt64
| RFloat32
| RFloat64
| RChar

val prim_type_beq : prim_type -> prim_type -> bool

val reg_type_beq : reg_type -> reg_type -> bool

type pre_type =
| PVar of int
| PPrim of prim_type
| PReg of reg_type
| PTuple of pre_type list

type pre_subst = (int * pre_type) list

val subst_lookup : pre_subst -> int -> pre_type option

val follow_pvar : int -> pre_subst -> int -> pre_type

val follow : int -> pre_subst -> pre_type -> pre_type

val recurse_occurs : int -> pre_type -> bool

val occurs_in : int -> pre_subst -> int -> pre_type -> bool

val unify_list_with :
  (pre_subst -> pre_type -> pre_type -> pre_subst option) -> pre_subst ->
  pre_type list -> pre_type list -> pre_subst option

val unify_fun : int -> pre_subst -> pre_type -> pre_type -> pre_subst option

val apply_subst : int -> pre_subst -> pre_type -> pre_type
