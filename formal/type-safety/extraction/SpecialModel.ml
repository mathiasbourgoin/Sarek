
type ('a, 'b) sum =
| Inl of 'a
| Inr of 'b



(** val bool_dec : bool -> bool -> bool **)

let bool_dec b1 b2 =
  if b1 then if b2 then true else false else if b2 then false else true

(** val eqb : bool -> bool -> bool **)

let eqb b1 b2 =
  if b1 then b2 else if b2 then false else true

type ascii =
| Ascii of bool * bool * bool * bool * bool * bool * bool * bool

(** val ascii_dec : ascii -> ascii -> bool **)

let ascii_dec a b =
  let Ascii (b0, b1, b2, b3, b4, b5, b6, b7) = a in
  let Ascii (b8, b9, b10, b11, b12, b13, b14, b15) = b in
  if bool_dec b0 b8
  then if bool_dec b1 b9
       then if bool_dec b2 b10
            then if bool_dec b3 b11
                 then if bool_dec b4 b12
                      then if bool_dec b5 b13
                           then if bool_dec b6 b14
                                then bool_dec b7 b15
                                else false
                           else false
                      else false
                 else false
            else false
       else false
  else false

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

(** val string_dec : string -> string -> bool **)

let rec string_dec s x =
  match s with
  | EmptyString -> (match x with
                    | EmptyString -> true
                    | String (_, _) -> false)
  | String (a, s0) ->
    (match x with
     | EmptyString -> false
     | String (a0, s1) -> if ascii_dec a a0 then string_dec s0 s1 else false)

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
| TRecord of string * (string * sarek_type) list
| TVariant of string * (string * sarek_type option) list

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
  | TRecord (s, l) ->
    (match t2 with
     | TRecord (s0, l0) ->
       let iHLR =
         let rec iHLR fs1 fs2 =
           match fs1 with
           | [] -> (match fs2 with
                    | [] -> true
                    | _ :: _ -> false)
           | p :: l1 ->
             let (s1, s2) = p in
             (match fs2 with
              | [] -> false
              | p0 :: l2 ->
                let (s3, s4) = p0 in
                let s5 = string_dec s1 s3 in
                if s5
                then let s6 = sarek_type_eq_dec s2 s4 in
                     if s6 then iHLR l1 l2 else false
                else false)
         in iHLR
       in
       let s1 = string_dec s s0 in if s1 then iHLR l l0 else false
     | _ -> false)
  | TVariant (s, l) ->
    (match t2 with
     | TVariant (s0, l0) ->
       let iHLV =
         let rec iHLV cs1 cs2 =
           match cs1 with
           | [] -> (match cs2 with
                    | [] -> true
                    | _ :: _ -> false)
           | p :: l1 ->
             let (s1, o) = p in
             (match cs2 with
              | [] -> false
              | p0 :: l2 ->
                let (s2, o0) = p0 in
                let s3 = string_dec s1 s2 in
                if s3
                then (match o with
                      | Some s4 ->
                        (match o0 with
                         | Some s5 ->
                           let s6 = sarek_type_eq_dec s4 s5 in
                           if s6 then iHLV l1 l2 else false
                         | None -> false)
                      | None ->
                        (match o0 with
                         | Some _ -> false
                         | None -> iHLV l1 l2))
                else false)
         in iHLV
       in
       let s1 = string_dec s s0 in if s1 then iHLV l l0 else false
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

(** val field_lookup :
    string -> (string * sarek_type) list -> sarek_type option **)

let rec field_lookup field = function
| [] -> None
| p :: rest ->
  let (f, t) = p in if eqb1 f field then Some t else field_lookup field rest

type rec_error =
| RMemError of vec_error
| NotARecord of sarek_type
| FieldNotFound of string * sarek_type
| FieldMismatch of sarek_type * sarek_type

type rec_result = (sarek_type, rec_error) sum

type rec_expr =
| RMem of mem_expr
| EFieldGet of string * rec_expr
| EFieldSet of string * rec_expr * rec_expr

(** val infer_rec_type : type_env -> rec_expr -> rec_result **)

let rec infer_rec_type env = function
| RMem me ->
  (match infer_mem_type env me with
   | Inl t -> Inl t
   | Inr err -> Inr (RMemError err))
| EFieldGet (field, rec0) ->
  (match infer_rec_type env rec0 with
   | Inl t ->
     (match t with
      | TRecord (name, fields) ->
        (match field_lookup field fields with
         | Some t0 -> Inl t0
         | None -> Inr (FieldNotFound (field, (TRecord (name, fields)))))
      | _ -> Inr (NotARecord t))
   | Inr err -> Inr err)
| EFieldSet (field, rec0, value) ->
  (match infer_rec_type env rec0 with
   | Inl t ->
     (match t with
      | TRecord (name, fields) ->
        (match field_lookup field fields with
         | Some field_t ->
           (match infer_rec_type env value with
            | Inl vt ->
              if sarek_type_eq_dec vt field_t
              then Inl (TPrim TUnit)
              else Inr (FieldMismatch (vt, field_t))
            | Inr err -> Inr err)
         | None -> Inr (FieldNotFound (field, (TRecord (name, fields)))))
      | _ -> Inr (NotARecord t))
   | Inr err -> Inr err)

type cf_error =
| CRec of rec_error
| CondNotBool of sarek_type
| BranchMismatch of sarek_type * sarek_type
| BoundNotInt32 of sarek_type

type cf_expr =
| CFRec of rec_expr
| CFIfThen of cf_expr * cf_expr
| CFIfElse of cf_expr * cf_expr * cf_expr
| CFFor of string * cf_expr * cf_expr * cf_expr
| CFWhile of cf_expr * cf_expr
| CFSeq of cf_expr * cf_expr

(** val infer_cf_type : type_env -> cf_expr -> (sarek_type, cf_error) sum **)

let rec infer_cf_type env = function
| CFRec re ->
  (match infer_rec_type env re with
   | Inl t -> Inl t
   | Inr err -> Inr (CRec err))
| CFIfThen (cond, then_e) ->
  (match infer_cf_type env cond with
   | Inl cond_t ->
     if sarek_type_eq_dec cond_t (TPrim TBool)
     then (match infer_cf_type env then_e with
           | Inl then_t ->
             if sarek_type_eq_dec then_t (TPrim TUnit)
             then Inl (TPrim TUnit)
             else Inr (BranchMismatch (then_t, (TPrim TUnit)))
           | Inr err -> Inr err)
     else Inr (CondNotBool cond_t)
   | Inr err -> Inr err)
| CFIfElse (cond, then_e, else_e) ->
  (match infer_cf_type env cond with
   | Inl cond_t ->
     if sarek_type_eq_dec cond_t (TPrim TBool)
     then (match infer_cf_type env then_e with
           | Inl then_t ->
             (match infer_cf_type env else_e with
              | Inl else_t ->
                if sarek_type_eq_dec then_t else_t
                then Inl then_t
                else Inr (BranchMismatch (then_t, else_t))
              | Inr err -> Inr err)
           | Inr err -> Inr err)
     else Inr (CondNotBool cond_t)
   | Inr err -> Inr err)
| CFFor (var, lo, hi, body) ->
  (match infer_cf_type env lo with
   | Inl lo_t ->
     if sarek_type_eq_dec lo_t (TPrim TInt32)
     then (match infer_cf_type env hi with
           | Inl hi_t ->
             if sarek_type_eq_dec hi_t (TPrim TInt32)
             then (match infer_cf_type ((var, (TPrim TInt32)) :: env) body with
                   | Inl _ -> Inl (TPrim TUnit)
                   | Inr err -> Inr err)
             else Inr (BoundNotInt32 hi_t)
           | Inr err -> Inr err)
     else Inr (BoundNotInt32 lo_t)
   | Inr err -> Inr err)
| CFWhile (cond, body) ->
  (match infer_cf_type env cond with
   | Inl cond_t ->
     if sarek_type_eq_dec cond_t (TPrim TBool)
     then (match infer_cf_type env body with
           | Inl _ -> Inl (TPrim TUnit)
           | Inr err -> Inr err)
     else Inr (CondNotBool cond_t)
   | Inr err -> Inr err)
| CFSeq (e1, e2) ->
  (match infer_cf_type env e1 with
   | Inl _ -> infer_cf_type env e2
   | Inr err -> Inr err)

type binop =
| Add
| Sub
| Mul
| Div
| Mod
| And
| Or
| Eq
| Ne
| Lt
| Le
| Gt
| Ge
| Land
| Lor
| Lxor
| Lsl
| Lsr
| Asr

type unop =
| Neg
| Not
| Lnot

(** val is_numeric : sarek_type -> bool **)

let is_numeric = function
| TPrim p -> (match p with
              | TInt32 -> true
              | _ -> false)
| TReg r -> (match r with
             | RChar -> false
             | _ -> true)
| _ -> false

(** val is_integer : sarek_type -> bool **)

let is_integer = function
| TPrim p -> (match p with
              | TInt32 -> true
              | _ -> false)
| TReg r -> (match r with
             | RInt -> true
             | RInt64 -> true
             | _ -> false)
| _ -> false

type op_error =
| OCF of cf_error
| NotNumeric of sarek_type
| NotInteger of sarek_type
| NotBool of sarek_type
| OperandMismatch of sarek_type * sarek_type

type op_expr =
| OPCf of cf_expr
| OPBinop of binop * op_expr * op_expr
| OPUnop of unop * op_expr

(** val infer_op_type : type_env -> op_expr -> (sarek_type, op_error) sum **)

let rec infer_op_type env = function
| OPCf ce ->
  (match infer_cf_type env ce with
   | Inl t -> Inl t
   | Inr err -> Inr (OCF err))
| OPBinop (op, lhs, rhs) ->
  (match infer_op_type env lhs with
   | Inl t1 ->
     (match infer_op_type env rhs with
      | Inl t2 ->
        if sarek_type_eq_dec t1 t2
        then (match op with
              | Add -> if is_numeric t1 then Inl t1 else Inr (NotNumeric t1)
              | Sub -> if is_numeric t1 then Inl t1 else Inr (NotNumeric t1)
              | Mul -> if is_numeric t1 then Inl t1 else Inr (NotNumeric t1)
              | Div -> if is_numeric t1 then Inl t1 else Inr (NotNumeric t1)
              | And ->
                if sarek_type_eq_dec t1 (TPrim TBool)
                then Inl (TPrim TBool)
                else Inr (NotBool t1)
              | Or ->
                if sarek_type_eq_dec t1 (TPrim TBool)
                then Inl (TPrim TBool)
                else Inr (NotBool t1)
              | Eq -> Inl (TPrim TBool)
              | Ne -> Inl (TPrim TBool)
              | Lt ->
                if is_numeric t1
                then Inl (TPrim TBool)
                else Inr (NotNumeric t1)
              | Le ->
                if is_numeric t1
                then Inl (TPrim TBool)
                else Inr (NotNumeric t1)
              | Gt ->
                if is_numeric t1
                then Inl (TPrim TBool)
                else Inr (NotNumeric t1)
              | Ge ->
                if is_numeric t1
                then Inl (TPrim TBool)
                else Inr (NotNumeric t1)
              | _ -> if is_integer t1 then Inl t1 else Inr (NotInteger t1))
        else Inr (OperandMismatch (t1, t2))
      | Inr err -> Inr err)
   | Inr err -> Inr err)
| OPUnop (op, operand) ->
  (match infer_op_type env operand with
   | Inl t ->
     (match op with
      | Neg -> if is_numeric t then Inl t else Inr (NotNumeric t)
      | Not ->
        if sarek_type_eq_dec t (TPrim TBool)
        then Inl (TPrim TBool)
        else Inr (NotBool t)
      | Lnot -> if is_integer t then Inl t else Inr (NotInteger t))
   | Inr err -> Inr err)

type fun_error =
| FEOpErr of op_error
| NotAFunc of sarek_type
| ArgMismatch of sarek_type * sarek_type
| BodyMismatch of sarek_type * sarek_type

type fun_expr =
| FEOp of op_expr
| FEApp of fun_expr * fun_expr
| FELetRec of string * string * sarek_type * sarek_type * fun_expr * fun_expr

(** val infer_fun_type :
    type_env -> fun_expr -> (sarek_type, fun_error) sum **)

let rec infer_fun_type env = function
| FEOp oe ->
  (match infer_op_type env oe with
   | Inl t -> Inl t
   | Inr err -> Inr (FEOpErr err))
| FEApp (fn, arg) ->
  (match infer_fun_type env fn with
   | Inl tfn ->
     (match tfn with
      | TFun (l, ret_ty) ->
        (match l with
         | [] -> Inr (NotAFunc tfn)
         | p_ty :: l0 ->
           (match l0 with
            | [] ->
              (match infer_fun_type env arg with
               | Inl targ ->
                 if sarek_type_eq_dec targ p_ty
                 then Inl ret_ty
                 else Inr (ArgMismatch (p_ty, targ))
               | Inr err -> Inr err)
            | _ :: _ -> Inr (NotAFunc tfn)))
      | _ -> Inr (NotAFunc tfn))
   | Inr err -> Inr err)
| FELetRec (fn_name, p_name, p_ty, ret_ty, body, cont) ->
  let fn_ty = TFun ((p_ty :: []), ret_ty) in
  let body_env = (p_name, p_ty) :: ((fn_name, fn_ty) :: env) in
  (match infer_fun_type body_env body with
   | Inl tbody ->
     if sarek_type_eq_dec tbody ret_ty
     then infer_fun_type ((fn_name, fn_ty) :: env) cont
     else Inr (BodyMismatch (ret_ty, tbody))
   | Inr err -> Inr err)

type mut_env = string list

(** val is_mutable : mut_env -> string -> bool **)

let rec is_mutable mu x =
  match mu with
  | [] -> false
  | y :: rest -> if eqb1 x y then true else is_mutable rest x

type mut_error =
| MEFunErr of fun_error
| MEUnbound of string
| MEImmutable of string
| MEAssignMismatch of sarek_type * sarek_type

type mut_expr =
| MEFun of fun_expr
| MELetMut of string * mut_expr * mut_expr
| MEAssign of string * mut_expr

(** val infer_mut_type :
    type_env -> mut_env -> mut_expr -> (sarek_type, mut_error) sum **)

let rec infer_mut_type env mu = function
| MEFun fe ->
  (match infer_fun_type env fe with
   | Inl t -> Inl t
   | Inr err -> Inr (MEFunErr err))
| MELetMut (name, init, body) ->
  (match infer_mut_type env mu init with
   | Inl t -> infer_mut_type ((name, t) :: env) (name :: mu) body
   | Inr err -> Inr err)
| MEAssign (name, value) ->
  (match infer_mut_type env mu value with
   | Inl tv ->
     (match lookup_env env name with
      | Some tdecl ->
        if is_mutable mu name
        then if sarek_type_eq_dec tv tdecl
             then Inl (TPrim TUnit)
             else Inr (MEAssignMismatch (tdecl, tv))
        else Inr (MEImmutable name)
      | None -> Inr (MEUnbound name))
   | Inr err -> Inr err)

(** val lookup_constr :
    (string * sarek_type option) list -> string -> sarek_type option option **)

let rec lookup_constr constrs name =
  match constrs with
  | [] -> None
  | p :: rest ->
    let (cname, payload) = p in
    if eqb1 name cname then Some payload else lookup_constr rest name

type pat_error =
| PEMutErr of mut_error
| PENotVariant of sarek_type
| PEMismatch of string
| PEBranchType of sarek_type * sarek_type
| PEEmpty

type pat_expr =
| PEMut of mut_expr
| PEMatch of pat_expr * ((string * string option) * pat_expr) list

(** val branch_body_env :
    type_env -> sarek_type option -> string option -> type_env **)

let branch_body_env env payload bvar =
  match payload with
  | Some pty -> (match bvar with
                 | Some v -> (v, pty) :: env
                 | None -> env)
  | None -> env

(** val check_branches :
    (type_env -> mut_env -> pat_expr -> (sarek_type, pat_error) sum) ->
    type_env -> mut_env -> (string * sarek_type option) list -> sarek_type ->
    ((string * string option) * pat_expr) list -> (unit, pat_error) sum **)

let rec check_branches infer_f env mu constrs result_ty = function
| [] -> Inl ()
| p :: rest ->
  let (p0, body) = p in
  let (cname, bvar) = p0 in
  (match lookup_constr constrs cname with
   | Some payload ->
     (match infer_f (branch_body_env env payload bvar) mu body with
      | Inl tbody ->
        if sarek_type_eq_dec tbody result_ty
        then check_branches infer_f env mu constrs result_ty rest
        else Inr (PEBranchType (result_ty, tbody))
      | Inr err -> Inr err)
   | None -> Inr (PEMismatch cname))

(** val infer_pat_type :
    type_env -> mut_env -> pat_expr -> (sarek_type, pat_error) sum **)

let rec infer_pat_type env mu = function
| PEMut me ->
  (match infer_mut_type env mu me with
   | Inl t -> Inl t
   | Inr err -> Inr (PEMutErr err))
| PEMatch (scrut, branches) ->
  (match infer_pat_type env mu scrut with
   | Inl tscrut ->
     (match tscrut with
      | TVariant (_, constrs) ->
        (match branches with
         | [] -> Inr PEEmpty
         | p :: rest ->
           let (p0, body) = p in
           let (cname, bvar) = p0 in
           (match lookup_constr constrs cname with
            | Some payload ->
              (match infer_pat_type (branch_body_env env payload bvar) mu body with
               | Inl result_ty ->
                 (match check_branches infer_pat_type env mu constrs
                          result_ty rest with
                  | Inl _ -> Inl result_ty
                  | Inr err -> Inr err)
               | Inr err -> Inr err)
            | None -> Inr (PEMismatch cname)))
      | _ -> Inr (PENotVariant tscrut))
   | Inr err -> Inr err)

type constr_error =
| CPatternErr of pat_error
| FieldTypeMismatch of string * sarek_type * sarek_type
| UnknownField of string
| UnknownConstr of string
| ConstrArity of string

type constr_expr =
| CEPat of pat_expr
| CERecord of string * (string * sarek_type) list
   * (string * constr_expr) list
| CEConstr of string * (string * sarek_type option) list * string
   * constr_expr option

(** val check_fields :
    (type_env -> mut_env -> constr_expr -> (sarek_type, constr_error) sum) ->
    type_env -> mut_env -> (string * sarek_type) list ->
    (string * constr_expr) list -> (unit, constr_error) sum **)

let rec check_fields infer_f env mu declared = function
| [] -> Inl ()
| p :: rest ->
  let (fname, fexpr) = p in
  (match field_lookup fname declared with
   | Some declared_ty ->
     (match infer_f env mu fexpr with
      | Inl got_ty ->
        if sarek_type_eq_dec got_ty declared_ty
        then check_fields infer_f env mu declared rest
        else Inr (FieldTypeMismatch (fname, declared_ty, got_ty))
      | Inr err -> Inr err)
   | None -> Inr (UnknownField fname))

(** val infer_constr_type :
    type_env -> mut_env -> constr_expr -> (sarek_type, constr_error) sum **)

let rec infer_constr_type env mu = function
| CEPat pe ->
  (match infer_pat_type env mu pe with
   | Inl t -> Inl t
   | Inr err -> Inr (CPatternErr err))
| CERecord (rname, declared, provided) ->
  (match check_fields infer_constr_type env mu declared provided with
   | Inl _ -> Inl (TRecord (rname, declared))
   | Inr err -> Inr err)
| CEConstr (tyname, constrs, cname, arg) ->
  (match lookup_constr constrs cname with
   | Some payload ->
     (match payload with
      | Some pty ->
        (match arg with
         | Some a ->
           (match infer_constr_type env mu a with
            | Inl got ->
              if sarek_type_eq_dec got pty
              then Inl (TVariant (tyname, constrs))
              else Inr (FieldTypeMismatch (cname, pty, got))
            | Inr err -> Inr err)
         | None -> Inr (ConstrArity cname))
      | None ->
        (match arg with
         | Some _ -> Inr (ConstrArity cname)
         | None -> Inl (TVariant (tyname, constrs))))
   | None -> Inr (UnknownConstr cname))

type special_error =
| SConstrErr of constr_error
| EarlyReturnNotAllowed
| ArraySizeNotInt of sarek_type
| TypeAnnotMismatch of sarek_type * sarek_type

type special_expr =
| SEConstr of constr_expr
| SEReturn of bool * special_expr
| SECreateArray of special_expr * sarek_type * mem_space
| SETyped of special_expr * sarek_type

(** val infer_special_type :
    type_env -> mut_env -> special_expr -> (sarek_type, special_error) sum **)

let rec infer_special_type env mu = function
| SEConstr ce ->
  (match infer_constr_type env mu ce with
   | Inl t -> Inl t
   | Inr err -> Inr (SConstrErr err))
| SEReturn (allowed, body) ->
  if allowed
  then infer_special_type env mu body
  else Inr EarlyReturnNotAllowed
| SECreateArray (size, elt, mem) ->
  (match infer_special_type env mu size with
   | Inl sz_ty ->
     if sarek_type_eq_dec sz_ty (TPrim TInt32)
     then Inl (TArr (elt, mem))
     else Inr (ArraySizeNotInt sz_ty)
   | Inr err -> Inr err)
| SETyped (body, annot) ->
  (match infer_special_type env mu body with
   | Inl got ->
     if sarek_type_eq_dec got annot
     then Inl annot
     else Inr (TypeAnnotMismatch (annot, got))
   | Inr err -> Inr err)
