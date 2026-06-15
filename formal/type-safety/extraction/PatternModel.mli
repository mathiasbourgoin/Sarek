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

type unop = Neg | Not | Lnot

val is_numeric : sarek_type -> bool

val is_integer : sarek_type -> bool

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

val infer_op_type : type_env -> op_expr -> (sarek_type, op_error) sum

type fun_error =
  | FEOpErr of op_error
  | NotAFunc of sarek_type
  | ArgMismatch of sarek_type * sarek_type
  | BodyMismatch of sarek_type * sarek_type

type fun_expr =
  | FEOp of op_expr
  | FEApp of fun_expr * fun_expr
  | FELetRec of string * string * sarek_type * sarek_type * fun_expr * fun_expr

val infer_fun_type : type_env -> fun_expr -> (sarek_type, fun_error) sum

type mut_env = string list

val is_mutable : mut_env -> string -> bool

type mut_error =
  | MEFunErr of fun_error
  | MEUnbound of string
  | MEImmutable of string
  | MEAssignMismatch of sarek_type * sarek_type

type mut_expr =
  | MEFun of fun_expr
  | MELetMut of string * mut_expr * mut_expr
  | MEAssign of string * mut_expr

val infer_mut_type :
  type_env -> mut_env -> mut_expr -> (sarek_type, mut_error) sum

val lookup_constr :
  (string * sarek_type option) list -> string -> sarek_type option option

type pat_error =
  | PEMutErr of mut_error
  | PENotVariant of sarek_type
  | PEMismatch of string
  | PEBranchType of sarek_type * sarek_type
  | PEEmpty

type pat_expr =
  | PEMut of mut_expr
  | PEMatch of pat_expr * ((string * string option) * pat_expr) list

val branch_body_env : type_env -> sarek_type option -> string option -> type_env

val check_branches :
  (type_env -> mut_env -> pat_expr -> (sarek_type, pat_error) sum) ->
  type_env ->
  mut_env ->
  (string * sarek_type option) list ->
  sarek_type ->
  ((string * string option) * pat_expr) list ->
  (unit, pat_error) sum

val infer_pat_type :
  type_env -> mut_env -> pat_expr -> (sarek_type, pat_error) sum
