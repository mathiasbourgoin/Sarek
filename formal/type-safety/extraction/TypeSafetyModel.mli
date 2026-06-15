type ('a, 'b) sum = Inl of 'a | Inr of 'b

val eqb : bool -> bool -> bool

type ascii = Ascii of bool * bool * bool * bool * bool * bool * bool * bool

val eqb0 : ascii -> ascii -> bool

type string = EmptyString | String of ascii * string

val eqb1 : string -> string -> bool

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

type expr =
  | ELit of lit
  | EVar of string
  | ELet of string * expr * expr
  | ETuple of expr list

type type_env = (string * sarek_type) list

val lookup_env : type_env -> string -> sarek_type option

type type_error =
  | UnboundVar of string
  | TypeMismatch of sarek_type * sarek_type

type infer_result = (sarek_type, type_error) sum

val infer_list_with :
  (type_env -> expr -> infer_result) ->
  type_env ->
  expr list ->
  (sarek_type list, type_error) sum

val infer_type : type_env -> expr -> infer_result
