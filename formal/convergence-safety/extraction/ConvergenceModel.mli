val fst : 'a1 * 'a2 -> 'a1

val app : 'a1 list -> 'a1 list -> 'a1 list

val add : int -> int -> int

val sub : int -> int -> int

module Nat : sig end

val map : ('a1 -> 'a2) -> 'a1 list -> 'a2 list

val concat : 'a1 list list -> 'a1 list

val existsb : ('a1 -> bool) -> 'a1 list -> bool

val forallb : ('a1 -> bool) -> 'a1 list -> bool

val find : ('a1 -> bool) -> 'a1 list -> 'a1 option

type expr =
  | ELit
  | EVary
  | EBarrier
  | EWarpPoint
  | EVar of int
  | EBinop of expr * expr
  | EUnop of expr
  | EIf of expr * expr * expr
  | EWhile of expr * expr
  | EFor of expr * expr * expr
  | ESeq of expr list
  | ELet of int * expr * expr
  | ESuperstep of bool * expr * expr
  | EApp of expr list
  | EReturn of expr

type exec_mode = Converged | Diverged

type error = BarrierError | WarpError

val is_varying : expr -> bool

val barrier_free : expr -> bool

val has_diverging_cf : expr -> bool

val check : exec_mode -> expr -> error list

type dim_usage = {
  uses_x : bool;
  uses_y : bool;
  uses_z : bool;
  uses_block_dim : bool;
  uses_grid_dim : bool;
  uses_thread_idx : bool;
  uses_block_idx : bool;
  uses_shared_mem : bool;
}

val empty_dim_usage : dim_usage

val merge_dim_usage : dim_usage -> dim_usage -> dim_usage

val check_warp : exec_mode -> expr -> error list

type tid = int

type value = int

type venv = (int * value) list

val venv_lookup : venv -> int -> value

val venv_extend : venv -> int -> value -> venv

type event = EvBarrier | EvWarp

type trace = event list

type outcome = ONorm of value | ORet of value

val eval :
  (tid -> value) -> int -> tid -> venv -> expr -> (outcome * trace) option

val eval_concrete : int -> tid -> venv -> expr -> (outcome * trace) option
