(** val app : 'a1 list -> 'a1 list -> 'a1 list **)

let rec app l m = match l with [] -> m | a :: l1 -> a :: app l1 m

(** val map : ('a1 -> 'a2) -> 'a1 list -> 'a2 list **)

let rec map f = function [] -> [] | a :: l0 -> f a :: map f l0

(** val concat : 'a1 list list -> 'a1 list **)

let rec concat = function [] -> [] | x :: l0 -> app x (concat l0)

(** val existsb : ('a1 -> bool) -> 'a1 list -> bool **)

let rec existsb f = function [] -> false | a :: l0 -> f a || existsb f l0

(** val forallb : ('a1 -> bool) -> 'a1 list -> bool **)

let rec forallb f = function [] -> true | a :: l0 -> f a && forallb f l0

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

(** val is_varying : expr -> bool **)

let rec is_varying = function
  | EVary -> true
  | EBinop (a, b) -> is_varying a || is_varying b
  | EUnop e0 -> is_varying e0
  | EIf (c, t, el) -> (is_varying c || is_varying t) || is_varying el
  | EWhile (c, b) -> is_varying c || is_varying b
  | EFor (lo, hi, b) -> (is_varying lo || is_varying hi) || is_varying b
  | ESeq es -> existsb is_varying es
  | ELet (_, v, b) -> is_varying v || is_varying b
  | ESuperstep (_, body, cont) -> is_varying body || is_varying cont
  | EApp args -> existsb is_varying args
  | EReturn e0 -> is_varying e0
  | _ -> false

(** val barrier_free : expr -> bool **)

let rec barrier_free = function
  | EBarrier -> false
  | EBinop (a, b) -> barrier_free a && barrier_free b
  | EUnop e0 -> barrier_free e0
  | EIf (c, t, el) -> (barrier_free c && barrier_free t) && barrier_free el
  | EWhile (c, b) -> barrier_free c && barrier_free b
  | EFor (lo, hi, b) -> (barrier_free lo && barrier_free hi) && barrier_free b
  | ESeq es -> forallb barrier_free es
  | ELet (_, v, b) -> barrier_free v && barrier_free b
  | ESuperstep (divergent, body, cont) ->
      (divergent && barrier_free body) && barrier_free cont
  | EApp args -> forallb barrier_free args
  | EReturn e0 -> barrier_free e0
  | _ -> true

(** val has_diverging_cf : expr -> bool **)

let rec has_diverging_cf = function
  | EBinop (a, b) -> has_diverging_cf a || has_diverging_cf b
  | EUnop e0 -> has_diverging_cf e0
  | EIf (c, t, el) ->
      (is_varying c || has_diverging_cf t) || has_diverging_cf el
  | EWhile (c, b) -> is_varying c || has_diverging_cf b
  | EFor (lo, hi, b) -> (is_varying lo || is_varying hi) || has_diverging_cf b
  | ESeq es -> existsb has_diverging_cf es
  | ELet (_, v, b) -> has_diverging_cf v || has_diverging_cf b
  | ESuperstep (_, body, cont) -> has_diverging_cf body || has_diverging_cf cont
  | EApp args -> existsb has_diverging_cf args
  | EReturn e0 -> has_diverging_cf e0
  | _ -> false

(** val check : exec_mode -> expr -> error list **)

let rec check m = function
  | EBarrier -> (
      match m with Converged -> [] | Diverged -> BarrierError :: [])
  | EBinop (a, b) -> app (check m a) (check m b)
  | EUnop e0 -> check m e0
  | EIf (cond, t, el) ->
      let inner = if is_varying cond then Diverged else m in
      app (check m cond) (app (check inner t) (check inner el))
  | EWhile (cond, b) ->
      let inner = if is_varying cond then Diverged else m in
      app (check m cond) (check inner b)
  | EFor (lo, hi, b) ->
      let inner = if is_varying lo || is_varying hi then Diverged else m in
      app (check m lo) (app (check m hi) (check inner b))
  | ESeq es -> concat (map (check m) es)
  | ELet (_, v, b) -> app (check m v) (check m b)
  | ESuperstep (divergent, body, cont) ->
      let entry_errors =
        match m with
        | Converged -> []
        | Diverged -> if divergent then [] else BarrierError :: []
      in
      app entry_errors (app (check m body) (check m cont))
  | EApp args -> concat (map (check m) args)
  | EReturn e0 -> check m e0
  | _ -> []

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

(** val empty_dim_usage : dim_usage **)

let empty_dim_usage =
  {
    uses_x = false;
    uses_y = false;
    uses_z = false;
    uses_block_dim = false;
    uses_grid_dim = false;
    uses_thread_idx = false;
    uses_block_idx = false;
    uses_shared_mem = false;
  }

(** val merge_dim_usage : dim_usage -> dim_usage -> dim_usage **)

let merge_dim_usage a b =
  {
    uses_x = a.uses_x || b.uses_x;
    uses_y = a.uses_y || b.uses_y;
    uses_z = a.uses_z || b.uses_z;
    uses_block_dim = a.uses_block_dim || b.uses_block_dim;
    uses_grid_dim = a.uses_grid_dim || b.uses_grid_dim;
    uses_thread_idx = a.uses_thread_idx || b.uses_thread_idx;
    uses_block_idx = a.uses_block_idx || b.uses_block_idx;
    uses_shared_mem = a.uses_shared_mem || b.uses_shared_mem;
  }

(** val check_warp : exec_mode -> expr -> error list **)

let rec check_warp m = function
  | EBarrier -> (
      match m with Converged -> [] | Diverged -> BarrierError :: [])
  | EWarpPoint -> (
      match m with Converged -> [] | Diverged -> WarpError :: [])
  | EBinop (a, b) -> app (check_warp m a) (check_warp m b)
  | EUnop e0 -> check_warp m e0
  | EIf (cond, t, el) ->
      let inner = if is_varying cond then Diverged else m in
      app (check_warp m cond) (app (check_warp inner t) (check_warp inner el))
  | EWhile (cond, b) ->
      let inner = if is_varying cond then Diverged else m in
      app (check_warp m cond) (check_warp inner b)
  | EFor (lo, hi, b) ->
      let inner = if is_varying lo || is_varying hi then Diverged else m in
      app (check_warp m lo) (app (check_warp m hi) (check_warp inner b))
  | ESeq es -> concat (map (check_warp m) es)
  | ELet (_, v, b) -> app (check_warp m v) (check_warp m b)
  | ESuperstep (divergent, body, cont) ->
      let entry_errors =
        match m with
        | Converged -> []
        | Diverged -> if divergent then [] else BarrierError :: []
      in
      app entry_errors (app (check_warp m body) (check_warp m cont))
  | EApp args -> concat (map (check_warp m) args)
  | EReturn e0 -> check_warp m e0
  | _ -> []
