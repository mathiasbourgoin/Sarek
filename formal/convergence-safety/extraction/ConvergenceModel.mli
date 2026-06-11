
val app : 'a1 list -> 'a1 list -> 'a1 list

val map : ('a1 -> 'a2) -> 'a1 list -> 'a2 list

val concat : 'a1 list list -> 'a1 list

val existsb : ('a1 -> bool) -> 'a1 list -> bool

val forallb : ('a1 -> bool) -> 'a1 list -> bool

type expr =
| ELit
| EVary
| EBarrier
| EBinop of expr * expr
| EUnop of expr
| EIf of expr * expr * expr
| EWhile of expr * expr
| EFor of expr * expr * expr
| ESeq of expr list
| ELet of expr * expr
| EApp of expr list

type exec_mode =
| Converged
| Diverged

type error =
| BarrierError

val is_varying : expr -> bool

val barrier_free : expr -> bool

val has_diverging_cf : expr -> bool

val check : exec_mode -> expr -> error list

type dim_usage = { uses_x : bool; uses_y : bool; uses_z : bool;
                   uses_block_dim : bool; uses_grid_dim : bool;
                   uses_thread_idx : bool; uses_block_idx : bool;
                   uses_shared_mem : bool }

val empty_dim_usage : dim_usage

val merge_dim_usage : dim_usage -> dim_usage -> dim_usage
