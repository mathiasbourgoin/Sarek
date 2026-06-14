type ('a, 'b) sum = Inl of 'a | Inr of 'b

val bool_dec : bool -> bool -> bool

val eqb : bool -> bool -> bool

type ascii = Ascii of bool * bool * bool * bool * bool * bool * bool * bool

val ascii_dec : ascii -> ascii -> bool

val eqb0 : ascii -> ascii -> bool

type string = EmptyString | String of ascii * string

val string_dec : string -> string -> bool

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
  | TRecord of string * (string * sarek_type) list
  | TVariant of string * (string * sarek_type option) list

val prim_type_beq : prim_type -> prim_type -> bool

val prim_type_eq_dec : prim_type -> prim_type -> bool

val reg_type_beq : reg_type -> reg_type -> bool

val reg_type_eq_dec : reg_type -> reg_type -> bool

val mem_space_beq : mem_space -> mem_space -> bool

val mem_space_eq_dec : mem_space -> mem_space -> bool

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

val sarek_type_eq_dec : sarek_type -> sarek_type -> bool

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

val infer_mem_type : type_env -> mem_expr -> vec_result

val field_lookup : string -> (string * sarek_type) list -> sarek_type option

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

val infer_rec_type : type_env -> rec_expr -> rec_result

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

val infer_cf_type : type_env -> cf_expr -> (sarek_type, cf_error) sum
