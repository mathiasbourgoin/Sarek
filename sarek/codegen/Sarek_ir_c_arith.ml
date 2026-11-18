(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Wrapping signed integer arithmetic for the C-family emitters; the rule and
   its scope are in the interface. *)

open Sarek_ir_types

type width = W32 | W64

type env = {
  arrays : (string, elttype) Hashtbl.t;
      (** element type of each array name; several bindings of one name (a
          helper parameter and a kernel array) are all kept, and a read is typed
          only when they agree *)
  records : (string * (string * elttype) list) list;
  helpers : (string * elttype) list;  (** helper name -> return type *)
}

let empty = {arrays = Hashtbl.create 1; records = []; helpers = []}

(* ---- environment ---- *)

let array_elt = function TVec elt | TArray (elt, _) -> Some elt | _ -> None

let add_array env name elt = Hashtbl.add env.arrays name elt

let add_var_array env (v : var) =
  Option.iter (add_array env v.var_name) (array_elt v.var_type)

let on_decl env = function
  | DParam (v, Some ai) -> add_array env v.var_name ai.arr_elttype
  | DParam (v, None) -> add_var_array env v
  | DLocal (v, Some (EArrayCreate (elt, _, _))) -> add_array env v.var_name elt
  | DLocal (v, _) -> add_var_array env v
  | DShared (name, elt, _) -> add_array env name elt

let scan_body env body =
  let on_stmt () = function
    | SLet (v, EArrayCreate (elt, _, _), _)
    | SLetMut (v, EArrayCreate (elt, _, _), _) ->
        add_array env v.var_name elt
    | SLet (v, _, _) | SLetMut (v, _, _) -> add_var_array env v
    | _ -> ()
  in
  let folder =
    {
      Sarek_ir_analysis.fe = (fun () _ -> ());
      ft = (fun () _ -> ());
      fs = on_stmt;
      fnative = (fun () -> ());
      visit_lvalue = false;
    }
  in
  Sarek_ir_analysis.stmt_fold folder () body

let env_of_kernel (k : kernel) =
  let env =
    {
      arrays = Hashtbl.create 16;
      records = k.kern_types;
      helpers = List.map (fun hf -> (hf.hf_name, hf.hf_ret_type)) k.kern_funcs;
    }
  in
  List.iter (on_decl env) k.kern_params ;
  List.iter (on_decl env) k.kern_locals ;
  scan_body env k.kern_body ;
  List.iter
    (fun hf ->
      List.iter (add_var_array env) hf.hf_params ;
      scan_body env hf.hf_body)
    k.kern_funcs ;
  env

(* ---- typing ---- *)

let const_type = function
  | CInt32 _ -> TInt32
  | CInt64 _ -> TInt64
  | CFloat32 _ -> TFloat32
  | CFloat64 _ -> TFloat64
  | CBool _ -> TBool
  | CUnit -> TUnit

(* The C type an operand of this Sarek type has after integer promotion. *)
let width_of_type = function
  | TInt32 | TBool | TUint8 | TInt8 -> Some W32
  | TInt64 -> Some W64
  | _ -> None

let is_float = function TFloat16 | TFloat32 | TFloat64 -> true | _ -> false

(* The usual arithmetic conversions of C on two operand types, when both are
   known: a float operand makes the result a float, an int64 one a long. *)
let arith a b =
  match (a, b) with
  | Some a, Some b when is_float a || is_float b ->
      Some (if a = TFloat64 || b = TFloat64 then TFloat64 else TFloat32)
  | Some a, Some b -> (
      match (width_of_type a, width_of_type b) with
      | Some W64, Some _ | Some _, Some W64 -> Some TInt64
      | Some W32, Some W32 -> Some TInt32
      | _ -> None)
  | _ -> None

let lookup_array env name =
  match Hashtbl.find_all env.arrays name with
  | [] -> None
  | t :: rest -> if List.for_all (( = ) t) rest then Some t else None

let field_type env ty field =
  match ty with
  | Some (TRecord (name, fields)) -> (
      match List.assoc_opt field fields with
      | Some t -> Some t
      | None ->
          Option.bind (List.assoc_opt name env.records) (List.assoc_opt field))
  | _ -> None

(* The Sarek type names of intrinsic signatures (Sarek_registry). *)
let type_of_name = function
  | "int32" | "int" -> Some TInt32
  | "int64" -> Some TInt64
  | "float32" | "float" -> Some TFloat32
  | "float64" -> Some TFloat64
  | "bool" -> Some TBool
  | _ -> None

let intrinsic_type path name =
  if List.mem name Sarek_ir_intrinsic_dispatch.thread_intrinsic_names then
    Some TInt32
  else
    match Sarek_registry.find_fun ~module_path:path name with
    | Some fi -> type_of_name fi.Sarek_registry.fi_ret_type
    | None -> None

let rec type_of env = function
  | EConst c -> Some (const_type c)
  | EVar v -> Some v.var_type
  | EBinop ((Eq | Ne | Lt | Le | Gt | Ge | And | Or), _, _) | EUnop (Not, _) ->
      Some TBool
  | EBinop (Lshr32, _, _) -> Some TInt32
  | EBinop (Lshr64, _, _) -> Some TInt64
  | EBinop ((Shl | Shr), a, _) -> arith (type_of env a) (type_of env a)
  | EBinop ((Add | Sub | Mul | Div | Mod | BitAnd | BitOr | BitXor), a, b) ->
      arith (type_of env a) (type_of env b)
  | EUnop ((Neg | BitNot), a) -> arith (type_of env a) (type_of env a)
  | EArrayRead (arr, _) -> lookup_array env arr
  | EArrayReadExpr (base, _) -> Option.bind (type_of env base) array_elt
  | ERecordField (e, field) -> field_type env (type_of env e) field
  | EIntrinsic (path, name, _) -> intrinsic_type path name
  | ECast (t, _) -> Some t
  | EApp (EVar f, _) -> List.assoc_opt f.var_name env.helpers
  | EArrayLen _ -> Some TInt32
  | EIf (_, a, b) ->
      let ta = type_of env a in
      if ta = type_of env b then ta else None
  | EMatch (_, (_, a) :: _) -> type_of env a
  | EApp _ | ETuple _ | ERecord _ | EVariant _ | EArrayCreate _ | EMatch _ ->
      None

let int_width env e = Option.bind (type_of env e) width_of_type

let binop_width env op a b =
  match op with
  | Add | Sub | Mul ->
      Option.bind (arith (type_of env a) (type_of env b)) width_of_type
  | Shl -> int_width env a
  | Div | Mod | Eq | Ne | Lt | Le | Gt | Ge | And | Or | Shr | Lshr32 | Lshr64
  | BitAnd | BitOr | BitXor ->
      None

(* ---- printing ---- *)

type spelling = {
  int32 : string;
  uint32 : string;
  int64 : string;
  uint64 : string;
  uint64_suffix : string;
}

let cuda =
  {
    int32 = "int";
    uint32 = "unsigned int";
    int64 = "long long";
    uint64 = "unsigned long long";
    uint64_suffix = "ull";
  }

let opencl =
  {
    int32 = "int";
    uint32 = "uint";
    int64 = "long";
    uint64 = "ulong";
    uint64_suffix = "ul";
  }

let metal = opencl

let types sp = function
  | W32 -> (sp.int32, sp.uint32)
  | W64 -> (sp.int64, sp.uint64)

let operator = function
  | Add -> " + "
  | Sub -> " - "
  | Mul -> " * "
  | Shl -> " << "
  | _ -> invalid_arg "Sarek_ir_c_arith.operator: not a wrapping operator"

(* Identifier, number and member-access characters: [r.px] and [1.5f] bind
   tighter than a cast. *)
let is_word_char = function
  | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' | '.' -> true
  | _ -> false

(* The bracket opened at [s.[start]] closes at the last character of [s]. *)
let closes_at_end s start =
  let n = String.length s in
  let rec go i depth =
    if i >= n then false
    else
      let depth =
        match s.[i] with
        | '(' | '[' -> depth + 1
        | ')' | ']' -> depth - 1
        | _ -> depth
      in
      if depth = 0 then i = n - 1 else go (i + 1) depth
  in
  go start 0

(* [s] is an identifier, a number or a field access, an array element [a[i]] or
   a call [f(x)] (postfix, so tighter than a cast), or one parenthesised whole:
   a cast or a binary operator applies to it without parentheses of its own. *)
let is_atomic s =
  let n = String.length s in
  let rec word_end i =
    if i < n && is_word_char s.[i] then word_end (i + 1) else i
  in
  let w = word_end 0 in
  n > 0
  && (w = n
     || (w > 0 && (s.[w] = '[' || s.[w] = '(') && closes_at_end s w)
     || (s.[0] = '(' && closes_at_end s 0))

let render ~gen_expr e =
  let b = Buffer.create 32 in
  gen_expr b e ;
  Buffer.contents b

let add_operand buf s =
  if is_atomic s then Buffer.add_string buf s
  else (
    Buffer.add_char buf '(' ;
    Buffer.add_string buf s ;
    Buffer.add_char buf ')')

(* [e] as a value of the unsigned type of width [w]. A wrapping operator of the
   same width is printed in the unsigned domain directly, so a chain such as
   [i * 6 + 1] converts back to the signed type once, at its root: unsigned
   arithmetic is modular, so the low bits are those of every intermediate
   conversion. A non-negative literal takes the unsigned suffix. *)
let rec emit_unsigned env sp w ~gen_expr buf e =
  let unsigned = emit_unsigned env sp w ~gen_expr buf in
  match e with
  | EBinop (op, a, b) when binop_width env op a b = Some w ->
      Buffer.add_char buf '(' ;
      unsigned a ;
      Buffer.add_string buf (operator op) ;
      (match op with
      | Shl -> add_operand buf (render ~gen_expr b)
      | _ -> unsigned b) ;
      Buffer.add_char buf ')'
  | EUnop (Neg, a) when int_width env a = Some w -> (
      match Sarek_ir_intrinsic_dispatch.int_negation_constant Neg a with
      | Some c -> unsigned (EConst c)
      | None ->
          Buffer.add_string buf "(-" ;
          unsigned a ;
          Buffer.add_char buf ')')
  | EConst (CInt32 n) when Int32.compare n 0l >= 0 ->
      Buffer.add_string buf (Int32.to_string n) ;
      Buffer.add_string
        buf
        (match w with W32 -> "u" | W64 -> sp.uint64_suffix)
  | EConst (CInt64 n) when w = W64 && Int64.compare n 0L >= 0 ->
      Buffer.add_string buf (Int64.to_string n) ;
      Buffer.add_string buf sp.uint64_suffix
  | _ ->
      let _, unsigned_type = types sp w in
      Buffer.add_string buf ("(" ^ unsigned_type ^ ")") ;
      add_operand buf (render ~gen_expr e)

let emit_signed sp w buf emit =
  let signed, _ = types sp w in
  Buffer.add_string buf ("((" ^ signed ^ ")") ;
  emit () ;
  Buffer.add_char buf ')'

(* [((S)((U)a op (U)b))]; for a shift the count keeps its own type. *)
let emit_binop env sp w ~gen_expr buf op a b =
  emit_signed sp w buf (fun () ->
      emit_unsigned env sp w ~gen_expr buf (EBinop (op, a, b)))

(* [((S)(-(U)a))]: negation of an unsigned value is defined modulo 2^n. *)
let emit_neg env sp w ~gen_expr buf a =
  emit_signed sp w buf (fun () ->
      emit_unsigned env sp w ~gen_expr buf (EUnop (Neg, a)))

(* Index arithmetic, printed as plain C: a chain of [Add]/[Sub]/[Mul]/[Shl] on
   signed integers at the root of a subscript, matrix offset or [for] bound.
   Anything else (a load, a call, [land], [/], ...) ends the chain and is
   printed by [gen_expr], wrapping included. *)
let rec emit_index env ~gen_expr buf e =
  match e with
  | EBinop (((Add | Sub | Mul | Shl) as op), a, b)
    when binop_width env op a b <> None ->
      Buffer.add_char buf '(' ;
      emit_index env ~gen_expr buf a ;
      Buffer.add_string buf (operator op) ;
      (match op with
      | Shl -> gen_expr buf b
      | _ -> emit_index env ~gen_expr buf b) ;
      Buffer.add_char buf ')'
  | _ -> gen_expr buf e
