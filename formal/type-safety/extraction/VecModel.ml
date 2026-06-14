
type ('a, 'b) sum =
| Inl of 'a
| Inr of 'b



(** val eqb : bool -> bool -> bool **)

let eqb b1 b2 =
  if b1 then b2 else if b2 then false else true

type ascii =
| Ascii of bool * bool * bool * bool * bool * bool * bool * bool

(** val eqb0 : ascii -> ascii -> bool **)

let eqb0 a b =
  let Ascii (a0, a1, a2, a3, a4, a5, a6, a7) = a in
  let Ascii (b0, b1, b2, b3, b4, b5, b6, b7) = b in
  if if if if if if if eqb a0 b0 then eqb a1 b1 else false
                 then eqb a2 b2
                 else false
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

type string =
| EmptyString
| String of ascii * string

(** val eqb1 : string -> string -> bool **)

let rec eqb1 s1 s2 =
  match s1 with
  | EmptyString ->
    (match s2 with
     | EmptyString -> true
     | String (_, _) -> false)
  | String (c1, s1') ->
    (match s2 with
     | EmptyString -> false
     | String (c2, s2') -> if eqb0 c1 c2 then eqb1 s1' s2' else false)

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

type mem_space =
| Local
| Shared
| Global

type sarek_type =
| TPrim of prim_type
| TReg of reg_type
| TVec of sarek_type
| TArr of sarek_type * mem_space
| TFun of sarek_type list * sarek_type
| TTuple of sarek_type list

(** val prim_type_beq : prim_type -> prim_type -> bool **)

let prim_type_beq x y =
  match x with
  | TUnit -> (match y with
              | TUnit -> true
              | _ -> false)
  | TBool -> (match y with
              | TBool -> true
              | _ -> false)
  | TInt32 -> (match y with
               | TInt32 -> true
               | _ -> false)

(** val prim_type_eq_dec : prim_type -> prim_type -> bool **)

let prim_type_eq_dec x y =
  let b = prim_type_beq x y in if b then true else false

(** val reg_type_beq : reg_type -> reg_type -> bool **)

let reg_type_beq x y =
  match x with
  | RInt -> (match y with
             | RInt -> true
             | _ -> false)
  | RInt64 -> (match y with
               | RInt64 -> true
               | _ -> false)
  | RFloat32 -> (match y with
                 | RFloat32 -> true
                 | _ -> false)
  | RFloat64 -> (match y with
                 | RFloat64 -> true
                 | _ -> false)
  | RChar -> (match y with
              | RChar -> true
              | _ -> false)

(** val reg_type_eq_dec : reg_type -> reg_type -> bool **)

let reg_type_eq_dec x y =
  let b = reg_type_beq x y in if b then true else false

(** val mem_space_beq : mem_space -> mem_space -> bool **)

let mem_space_beq x y =
  match x with
  | Local -> (match y with
              | Local -> true
              | _ -> false)
  | Shared -> (match y with
               | Shared -> true
               | _ -> false)
  | Global -> (match y with
               | Global -> true
               | _ -> false)

(** val mem_space_eq_dec : mem_space -> mem_space -> bool **)

let mem_space_eq_dec x y =
  let b = mem_space_beq x y in if b then true else false

type lit =
| LInt of int
| LFloat of int
| LBool of bool
| LUnit

type expr =
| ELit of lit
| EVar of string
| ELet of string * expr * expr
| ETuple of expr list

type type_env = (string * sarek_type) list

(** val lookup_env : type_env -> string -> sarek_type option **)

let rec lookup_env env x =
  match env with
  | [] -> None
  | p :: rest ->
    let (y, t) = p in if eqb1 x y then Some t else lookup_env rest x

type type_error =
| UnboundVar of string
| TypeMismatch of sarek_type * sarek_type

type infer_result = (sarek_type, type_error) sum

(** val infer_list_with :
    (type_env -> expr -> infer_result) -> type_env -> expr list ->
    (sarek_type list, type_error) sum **)

let rec infer_list_with infer_f env = function
| [] -> Inl []
| e :: rest ->
  (match infer_f env e with
   | Inl t ->
     (match infer_list_with infer_f env rest with
      | Inl ts -> Inl (t :: ts)
      | Inr err -> Inr err)
   | Inr err -> Inr err)

(** val infer_type : type_env -> expr -> infer_result **)

let rec infer_type env = function
| ELit l ->
  (match l with
   | LInt _ -> Inl (TPrim TInt32)
   | LFloat _ -> Inl (TReg RFloat32)
   | LBool _ -> Inl (TPrim TBool)
   | LUnit -> Inl (TPrim TUnit))
| EVar x ->
  (match lookup_env env x with
   | Some t -> Inl t
   | None -> Inr (UnboundVar x))
| ELet (x, e1, e2) ->
  (match infer_type env e1 with
   | Inl t1 -> infer_type ((x, t1) :: env) e2
   | Inr err -> Inr err)
| ETuple es ->
  (match infer_list_with infer_type env es with
   | Inl ts -> Inl (TTuple ts)
   | Inr err -> Inr err)

(** val sarek_type_eq_dec : sarek_type -> sarek_type -> bool **)

let rec sarek_type_eq_dec t1 t2 =
  match t1 with
  | TPrim p -> (match t2 with
                | TPrim p0 -> prim_type_eq_dec p p0
                | _ -> false)
  | TReg r -> (match t2 with
               | TReg r0 -> reg_type_eq_dec r r0
               | _ -> false)
  | TVec s -> (match t2 with
               | TVec s0 -> sarek_type_eq_dec s s0
               | _ -> false)
  | TArr (s, m) ->
    (match t2 with
     | TArr (s0, m0) ->
       let s1 = sarek_type_eq_dec s s0 in
       if s1 then mem_space_eq_dec m m0 else false
     | _ -> false)
  | TFun (l, s) ->
    (match t2 with
     | TFun (l0, s0) ->
       let iHL =
         let rec iHL ts1 ts2 =
           match ts1 with
           | [] -> (match ts2 with
                    | [] -> true
                    | _ :: _ -> false)
           | s1 :: l1 ->
             (match ts2 with
              | [] -> false
              | s2 :: l2 ->
                let s3 = sarek_type_eq_dec s1 s2 in
                if s3 then iHL l1 l2 else false)
         in iHL
       in
       let s1 = iHL l l0 in if s1 then sarek_type_eq_dec s s0 else false
     | _ -> false)
  | TTuple l ->
    (match t2 with
     | TTuple l0 ->
       let rec iHL ts1 ts2 =
         match ts1 with
         | [] -> (match ts2 with
                  | [] -> true
                  | _ :: _ -> false)
         | s :: l1 ->
           (match ts2 with
            | [] -> false
            | s0 :: l2 ->
              let s1 = sarek_type_eq_dec s s0 in
              if s1 then iHL l1 l2 else false)
       in iHL l l0
     | _ -> false)

type vec_error =
| VCoreError of type_error
| NotAVector of sarek_type
| NotAnArray of sarek_type
| IndexNotInt of sarek_type
| ElemMismatch of sarek_type * sarek_type

type vec_result = (sarek_type, vec_error) sum

type mem_expr =
| MCore of expr
| EVecGet of mem_expr * mem_expr
| EVecSet of mem_expr * mem_expr * mem_expr
| EArrGet of mem_expr * mem_expr
| EArrSet of mem_expr * mem_expr * mem_expr

(** val infer_mem_type : type_env -> mem_expr -> vec_result **)

let rec infer_mem_type env = function
| MCore ce ->
  (match infer_type env ce with
   | Inl t -> Inl t
   | Inr err -> Inr (VCoreError err))
| EVecGet (vec, idx) ->
  (match infer_mem_type env vec with
   | Inl tv ->
     (match tv with
      | TVec elem_t ->
        (match infer_mem_type env idx with
         | Inl ti ->
           (match ti with
            | TPrim p ->
              (match p with
               | TInt32 -> Inl elem_t
               | _ -> Inr (IndexNotInt ti))
            | _ -> Inr (IndexNotInt ti))
         | Inr err -> Inr err)
      | _ -> Inr (NotAVector tv))
   | Inr err -> Inr err)
| EVecSet (vec, idx, value) ->
  (match infer_mem_type env vec with
   | Inl tv ->
     (match tv with
      | TVec elem_t ->
        (match infer_mem_type env idx with
         | Inl ti ->
           (match ti with
            | TPrim p ->
              (match p with
               | TInt32 ->
                 (match infer_mem_type env value with
                  | Inl vt ->
                    if sarek_type_eq_dec vt elem_t
                    then Inl (TPrim TUnit)
                    else Inr (ElemMismatch (vt, elem_t))
                  | Inr err -> Inr err)
               | _ -> Inr (IndexNotInt ti))
            | _ -> Inr (IndexNotInt ti))
         | Inr err -> Inr err)
      | _ -> Inr (NotAVector tv))
   | Inr err -> Inr err)
| EArrGet (arr, idx) ->
  (match infer_mem_type env arr with
   | Inl ta ->
     (match ta with
      | TArr (elem_t, _) ->
        (match infer_mem_type env idx with
         | Inl ti ->
           (match ti with
            | TPrim p ->
              (match p with
               | TInt32 -> Inl elem_t
               | _ -> Inr (IndexNotInt ti))
            | _ -> Inr (IndexNotInt ti))
         | Inr err -> Inr err)
      | _ -> Inr (NotAnArray ta))
   | Inr err -> Inr err)
| EArrSet (arr, idx, value) ->
  (match infer_mem_type env arr with
   | Inl ta ->
     (match ta with
      | TArr (elem_t, _) ->
        (match infer_mem_type env idx with
         | Inl ti ->
           (match ti with
            | TPrim p ->
              (match p with
               | TInt32 ->
                 (match infer_mem_type env value with
                  | Inl vt ->
                    if sarek_type_eq_dec vt elem_t
                    then Inl (TPrim TUnit)
                    else Inr (ElemMismatch (vt, elem_t))
                  | Inr err -> Inr err)
               | _ -> Inr (IndexNotInt ti))
            | _ -> Inr (IndexNotInt ti))
         | Inr err -> Inr err)
      | _ -> Inr (NotAnArray ta))
   | Inr err -> Inr err)
