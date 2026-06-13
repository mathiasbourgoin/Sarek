(** val fst : ('a1 * 'a2) -> 'a1 **)

let fst = function x, _ -> x

(** val app : 'a1 list -> 'a1 list -> 'a1 list **)

let rec app l m = match l with [] -> m | a :: l1 -> a :: app l1 m

(** val add : int -> int -> int **)

let rec add = ( + )

(** val sub : int -> int -> int **)

let rec sub = fun n m -> Stdlib.max 0 (n - m)

module Nat = struct end

(** val map : ('a1 -> 'a2) -> 'a1 list -> 'a2 list **)

let rec map f = function [] -> [] | a :: l0 -> f a :: map f l0

(** val concat : 'a1 list list -> 'a1 list **)

let rec concat = function [] -> [] | x :: l0 -> app x (concat l0)

(** val existsb : ('a1 -> bool) -> 'a1 list -> bool **)

let rec existsb f = function [] -> false | a :: l0 -> f a || existsb f l0

(** val forallb : ('a1 -> bool) -> 'a1 list -> bool **)

let rec forallb f = function [] -> true | a :: l0 -> f a && forallb f l0

(** val find : ('a1 -> bool) -> 'a1 list -> 'a1 option **)

let rec find f = function
  | [] -> None
  | x :: tl -> if f x then Some x else find f tl

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

type tid = int

type value = int

type venv = (int * value) list

(** val venv_lookup : venv -> int -> value **)

let venv_lookup rho x =
  match find (fun p -> fst p = x) rho with
  | Some p ->
      let _, v = p in
      v
  | None -> 0

(** val venv_extend : venv -> int -> value -> venv **)

let venv_extend rho x v = (x, v) :: rho

type event = EvBarrier | EvWarp

type trace = event list

type outcome = ONorm of value | ORet of value

(** val eval : (tid -> value) -> int -> tid -> venv -> expr -> (outcome * trace)
    option **)

let rec eval vary_val fuel t rho e =
  (fun fO fS n -> if n = 0 then fO () else fS (n - 1))
    (fun _ -> None)
    (fun fuel' ->
      match e with
      | ELit -> Some (ONorm 0, [])
      | EVary -> Some (ONorm (vary_val t), [])
      | EBarrier -> Some (ONorm 0, EvBarrier :: [])
      | EWarpPoint -> Some (ONorm 0, EvWarp :: [])
      | EVar x -> Some (ONorm (venv_lookup rho x), [])
      | EBinop (e1, e2) -> (
          match eval vary_val fuel' t rho e1 with
          | Some p -> (
              let o, tr1 = p in
              match o with
              | ONorm v1 -> (
                  match eval vary_val fuel' t rho e2 with
                  | Some p0 -> (
                      let o0, tr2 = p0 in
                      match o0 with
                      | ONorm v2 -> Some (ONorm (add v1 v2), app tr1 tr2)
                      | ORet v2 -> Some (ORet v2, app tr1 tr2))
                  | None -> None)
              | ORet v -> Some (ORet v, tr1))
          | None -> None)
      | EUnop e1 -> eval vary_val fuel' t rho e1
      | EIf (cond, e_then, e_else) -> (
          match eval vary_val fuel' t rho cond with
          | Some p -> (
              let o, tr_c = p in
              match o with
              | ONorm cv -> (
                  let branch = if cv = 0 then e_else else e_then in
                  match eval vary_val fuel' t rho branch with
                  | Some p0 ->
                      let o0, tr_b = p0 in
                      Some (o0, app tr_c tr_b)
                  | None -> None)
              | ORet v -> Some (ORet v, tr_c))
          | None -> None)
      | EWhile (cond, body) -> (
          match eval vary_val fuel' t rho cond with
          | Some p -> (
              let o, tr_c = p in
              match o with
              | ONorm cv -> (
                  if cv = 0 then Some (ONorm 0, tr_c)
                  else
                    match eval vary_val fuel' t rho body with
                    | Some p0 -> (
                        let o0, tr_b = p0 in
                        match o0 with
                        | ONorm _ -> (
                            match
                              eval vary_val fuel' t rho (EWhile (cond, body))
                            with
                            | Some p1 ->
                                let o1, tr_loop = p1 in
                                Some (o1, app tr_c (app tr_b tr_loop))
                            | None -> None)
                        | ORet v -> Some (ORet v, app tr_c tr_b))
                    | None -> None)
              | ORet v -> Some (ORet v, tr_c))
          | None -> None)
      | EFor (lo, hi, body) -> (
          match eval vary_val fuel' t rho lo with
          | Some p -> (
              let o, tr_lo = p in
              match o with
              | ONorm lo_v -> (
                  match eval vary_val fuel' t rho hi with
                  | Some p0 -> (
                      let o0, tr_hi = p0 in
                      match o0 with
                      | ONorm hi_v ->
                          if hi_v <= lo_v then Some (ONorm 0, app tr_lo tr_hi)
                          else
                            let steps = sub hi_v lo_v in
                            let rec loop k acc_tr =
                              (fun fO fS n ->
                                if n = 0 then fO () else fS (n - 1))
                                (fun _ -> Some (ONorm 0, acc_tr))
                                (fun k' ->
                                  match eval vary_val fuel' t rho body with
                                  | Some p1 -> (
                                      let o1, tr_b = p1 in
                                      match o1 with
                                      | ONorm _ -> loop k' (app acc_tr tr_b)
                                      | ORet v -> Some (ORet v, app acc_tr tr_b)
                                      )
                                  | None -> None)
                                k
                            in
                            loop steps (app tr_lo tr_hi)
                      | ORet v -> Some (ORet v, app tr_lo tr_hi))
                  | None -> None)
              | ORet v -> Some (ORet v, tr_lo))
          | None -> None)
      | ESeq es ->
          let rec eval_seq xs acc_tr =
            match xs with
            | [] -> Some (ONorm 0, acc_tr)
            | x :: rest -> (
                match eval vary_val fuel' t rho x with
                | Some p -> (
                    let o, tr = p in
                    match o with
                    | ONorm _ -> eval_seq rest (app acc_tr tr)
                    | ORet v -> Some (ORet v, app acc_tr tr))
                | None -> None)
          in
          eval_seq es []
      | ELet (x, e_val, body) -> (
          match eval vary_val fuel' t rho e_val with
          | Some p -> (
              let o, tr_v = p in
              match o with
              | ONorm v -> (
                  let rho' = venv_extend rho x v in
                  match eval vary_val fuel' t rho' body with
                  | Some p0 ->
                      let o0, tr_b = p0 in
                      Some (o0, app tr_v tr_b)
                  | None -> None)
              | ORet v -> Some (ORet v, tr_v))
          | None -> None)
      | ESuperstep (_, body, cont) -> (
          match eval vary_val fuel' t rho body with
          | Some p -> (
              let o, tr_b = p in
              match o with
              | ONorm _ -> (
                  match eval vary_val fuel' t rho cont with
                  | Some p0 ->
                      let o0, tr_c = p0 in
                      Some (o0, app tr_b (app (EvBarrier :: []) tr_c))
                  | None -> None)
              | ORet v -> Some (ORet v, app tr_b (EvBarrier :: [])))
          | None -> None)
      | EApp args ->
          let rec eval_args xs acc_tr last_v =
            match xs with
            | [] -> Some (ONorm last_v, acc_tr)
            | x :: rest -> (
                match eval vary_val fuel' t rho x with
                | Some p -> (
                    let o, tr = p in
                    match o with
                    | ONorm v -> eval_args rest (app acc_tr tr) v
                    | ORet v -> Some (ORet v, app acc_tr tr))
                | None -> None)
          in
          eval_args args [] 0
      | EReturn e_inner -> (
          match eval vary_val fuel' t rho e_inner with
          | Some p -> (
              let o, tr = p in
              match o with
              | ONorm v -> Some (ORet v, tr)
              | ORet v -> Some (ORet v, tr))
          | None -> None))
    fuel

(** val eval_concrete : int -> tid -> venv -> expr -> (outcome * trace) option **)

let eval_concrete fuel t rho e = eval (fun th -> th) fuel t rho e
