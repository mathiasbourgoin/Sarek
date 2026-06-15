(** val negb : bool -> bool **)

let negb = function true -> false | false -> true

(** val length : 'a1 list -> int **)

let rec length = function [] -> 0 | _ :: l' -> Stdlib.Int.succ (length l')

module Nat = struct end

(** val map : ('a1 -> 'a2) -> 'a1 list -> 'a2 list **)

let rec map f = function [] -> [] | a :: l0 -> f a :: map f l0

(** val existsb : ('a1 -> bool) -> 'a1 list -> bool **)

let rec existsb f = function [] -> false | a :: l0 -> f a || existsb f l0

type prim_type = TUnit | TBool | TInt32

type reg_type = RInt | RInt64 | RFloat32 | RFloat64 | RChar

(** val prim_type_beq : prim_type -> prim_type -> bool **)

let prim_type_beq x y =
  match x with
  | TUnit -> ( match y with TUnit -> true | _ -> false)
  | TBool -> ( match y with TBool -> true | _ -> false)
  | TInt32 -> ( match y with TInt32 -> true | _ -> false)

(** val reg_type_beq : reg_type -> reg_type -> bool **)

let reg_type_beq x y =
  match x with
  | RInt -> ( match y with RInt -> true | _ -> false)
  | RInt64 -> ( match y with RInt64 -> true | _ -> false)
  | RFloat32 -> ( match y with RFloat32 -> true | _ -> false)
  | RFloat64 -> ( match y with RFloat64 -> true | _ -> false)
  | RChar -> ( match y with RChar -> true | _ -> false)

type pre_type =
  | PVar of int
  | PPrim of prim_type
  | PReg of reg_type
  | PTuple of pre_type list

type pre_subst = (int * pre_type) list

(** val subst_lookup : pre_subst -> int -> pre_type option **)

let rec subst_lookup s id =
  match s with
  | [] -> None
  | p :: rest ->
      let id', t = p in
      if id = id' then Some t else subst_lookup rest id

(** val follow_pvar : int -> pre_subst -> int -> pre_type **)

let rec follow_pvar fuel s id =
  (fun fO fS n -> if n = 0 then fO () else fS (n - 1))
    (fun _ -> PVar id)
    (fun n ->
      match subst_lookup s id with
      | Some t -> ( match t with PVar id' -> follow_pvar n s id' | _ -> t)
      | None -> PVar id)
    fuel

(** val follow : int -> pre_subst -> pre_type -> pre_type **)

let follow fuel s t = match t with PVar id -> follow_pvar fuel s id | _ -> t

(** val occurs_in : int -> pre_subst -> int -> pre_type -> bool **)

let rec occurs_in fuel s id t =
  (fun fO fS n -> if n = 0 then fO () else fS (n - 1))
    (fun _ -> false)
    (fun n ->
      match follow n s t with
      | PVar id' -> id = id'
      | PTuple ts -> existsb (occurs_in n s id) ts
      | _ -> false)
    fuel

(** val unify_list_with : (pre_subst -> pre_type -> pre_type -> pre_subst
    option) -> pre_subst -> pre_type list -> pre_type list -> pre_subst option **)

let rec unify_list_with unify_f s ts1 ts2 =
  match ts1 with
  | [] -> ( match ts2 with [] -> Some s | _ :: _ -> None)
  | t1 :: rest1 -> (
      match ts2 with
      | [] -> None
      | t2 :: rest2 -> (
          match unify_f s t1 t2 with
          | Some s' -> unify_list_with unify_f s' rest1 rest2
          | None -> None))

(** val unify_fun : int -> pre_subst -> pre_type -> pre_type -> pre_subst option
    **)

let rec unify_fun fuel s t1 t2 =
  (fun fO fS n -> if n = 0 then fO () else fS (n - 1))
    (fun _ -> None)
    (fun n ->
      match follow n s t1 with
      | PVar id -> (
          match follow n s t2 with
          | PVar id2 -> if id = id2 then Some s else Some ((id, PVar id2) :: s)
          | x -> if occurs_in n s id x then None else Some ((id, x) :: s))
      | PPrim p1 -> (
          match follow n s t2 with
          | PVar id ->
              if occurs_in n s id (PPrim p1) then None
              else Some ((id, PPrim p1) :: s)
          | PPrim p2 -> if prim_type_beq p1 p2 then Some s else None
          | _ -> None)
      | PReg r1 -> (
          match follow n s t2 with
          | PVar id ->
              if occurs_in n s id (PReg r1) then None
              else Some ((id, PReg r1) :: s)
          | PReg r2 -> if reg_type_beq r1 r2 then Some s else None
          | _ -> None)
      | PTuple ts1 -> (
          match follow n s t2 with
          | PVar id ->
              if occurs_in n s id (PTuple ts1) then None
              else Some ((id, PTuple ts1) :: s)
          | PTuple ts2 ->
              if negb (length ts1 = length ts2) then None
              else unify_list_with (unify_fun n) s ts1 ts2
          | _ -> None))
    fuel

(** val apply_subst : int -> pre_subst -> pre_type -> pre_type **)

let rec apply_subst fuel s t =
  match t with
  | PVar id -> follow_pvar fuel s id
  | PTuple ts -> PTuple (map (apply_subst fuel s) ts)
  | _ -> t
