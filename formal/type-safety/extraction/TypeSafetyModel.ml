type ('a, 'b) sum = Inl of 'a | Inr of 'b

(** val eqb : bool -> bool -> bool **)

let eqb b1 b2 = if b1 then b2 else if b2 then false else true

type ascii = Ascii of bool * bool * bool * bool * bool * bool * bool * bool

(** val eqb0 : ascii -> ascii -> bool **)

let eqb0 a b =
  let (Ascii (a0, a1, a2, a3, a4, a5, a6, a7)) = a in
  let (Ascii (b0, b1, b2, b3, b4, b5, b6, b7)) = b in
  if
    if
      if
        if
          if if if eqb a0 b0 then eqb a1 b1 else false then eqb a2 b2 else false
          then eqb a3 b3
          else false
        then eqb a4 b4
        else false
      then eqb a5 b5
      else false
    then eqb a6 b6
    else false
  then eqb a7 b7
  else false

type string = EmptyString | String of ascii * string

(** val eqb1 : string -> string -> bool **)

let rec eqb1 s1 s2 =
  match s1 with
  | EmptyString -> (
      match s2 with EmptyString -> true | String (_, _) -> false)
  | String (c1, s1') -> (
      match s2 with
      | EmptyString -> false
      | String (c2, s2') -> if eqb0 c1 c2 then eqb1 s1' s2' else false)

type prim_type = TUnit | TBool | TInt32

type reg_type = RInt | RInt64 | RFloat32 | RFloat64 | RChar

type mem_space = Local | Shared | Global

type sarek_type =
  | TPrim of prim_type
  | TReg of reg_type
  | TVec of sarek_type
  | TArr of sarek_type * mem_space
  | TFun of sarek_type list * sarek_type
  | TTuple of sarek_type list

type lit = LInt of int | LFloat of int | LBool of bool | LUnit

type expr = ELit of lit | EVar of string | ELet of string * expr * expr

type type_env = (string * sarek_type) list

(** val lookup_env : type_env -> string -> sarek_type option **)

let rec lookup_env env x =
  match env with
  | [] -> None
  | p :: rest ->
      let y, t = p in
      if eqb1 x y then Some t else lookup_env rest x

type type_error =
  | UnboundVar of string
  | TypeMismatch of sarek_type * sarek_type

type infer_result = (sarek_type, type_error) sum

(** val infer_type : type_env -> expr -> infer_result **)

let rec infer_type env = function
  | ELit l -> (
      match l with
      | LInt _ -> Inl (TPrim TInt32)
      | LFloat _ -> Inl (TReg RFloat32)
      | LBool _ -> Inl (TPrim TBool)
      | LUnit -> Inl (TPrim TUnit))
  | EVar x -> (
      match lookup_env env x with Some t -> Inl t | None -> Inr (UnboundVar x))
  | ELet (x, e1, e2) -> (
      match infer_type env e1 with
      | Inl t1 -> infer_type ((x, t1) :: env) e2
      | Inr err -> Inr err)
