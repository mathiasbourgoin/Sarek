(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Wrapping signed integer arithmetic for the C-family emitters; the rule and
   its scope are in the interface. *)

open Sarek_ir_types
module T = Sarek_ir_c_types

type width = T.width = W32 | W64

type env = T.env

let empty = T.empty

let int_width = T.int_width

let binop_width = T.binop_width

let typing_steps = T.steps

let index_steps = T.index_steps

let is_index_let (env : env) (v : var) = Hashtbl.mem env.T.index_lets v.var_name

(* ---- environments ---- *)

let decl_name = function
  | DParam (v, _) | DLocal (v, _) -> v.var_name
  | DShared (name, _, _) -> name

let helper_scope env (hf : helper_func) =
  let scope = T.with_scope env in
  List.iter (T.add_var_array scope) hf.hf_params ;
  T.add_body scope hf.hf_body ;
  scope

let env_of_kernel ?records (k : kernel) =
  let records =
    match records with Some r -> r @ k.kern_types | None -> k.kern_types
  in
  let env =
    T.create
      ~records
      ~variants:k.kern_variants
      ~helpers:(List.map (fun hf -> (hf.hf_name, hf.hf_ret_type)) k.kern_funcs)
  in
  List.iter (T.add_decl env) k.kern_params ;
  List.iter (T.add_decl env) k.kern_locals ;
  T.add_body env k.kern_body ;
  let scopes =
    (k.kern_body, env)
    :: List.map (fun hf -> (hf.hf_body, helper_scope env hf)) k.kern_funcs
  in
  let params =
    List.map decl_name (k.kern_params @ k.kern_locals)
    @ List.concat_map
        (fun hf -> List.map (fun (v : var) -> v.var_name) hf.hf_params)
        k.kern_funcs
  in
  let extra =
    List.filter_map
      (function DLocal (_, Some e) -> Some e | _ -> None)
      k.kern_locals
  in
  Sarek_ir_c_index.compute env ~scopes ~extra ~params ;
  env

let for_helper = helper_scope

(* Emit [e] with its typing on the pending stack, where its children's arms
   find it (Sarek_ir_c_types.annot_of): each emitted node is typed once,
   whatever non-arithmetic nodes (casts, fields, calls) lie between two
   arithmetic ones. *)
let visit env e f =
  let a = T.annot_of env e in
  T.with_pending env e a f

(* ---- printing ---- *)

type spelling = {
  int32 : string;
  uint32 : string;
  int64 : string;
  uint64 : string;
  uint64_suffix : string;
  to_int32 : string;  (** applied to the unsigned result: a cast or a bitcast *)
  to_int64 : string;
}

let cuda =
  {
    int32 = "int";
    uint32 = "unsigned int";
    int64 = "long long";
    uint64 = "unsigned long long";
    uint64_suffix = "ull";
    to_int32 = "(int)";
    to_int64 = "(long long)";
  }

let opencl =
  {
    int32 = "int";
    uint32 = "uint";
    int64 = "long";
    uint64 = "ulong";
    uint64_suffix = "ul";
    to_int32 = "as_int";
    to_int64 = "as_long";
  }

let metal = {opencl with to_int32 = "as_type<int>"; to_int64 = "as_type<long>"}

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

let add_operand buf s =
  if is_atomic s then Buffer.add_string buf s
  else (
    Buffer.add_char buf '(' ;
    Buffer.add_string buf s ;
    Buffer.add_char buf ')')

(* [e] (annotation [a]) printed by the emitter, which finds [a] again. *)
let print_child env ~gen_expr buf e a =
  T.with_pending env e a (fun () -> gen_expr buf e)

let render_child env ~gen_expr e a =
  let b = Buffer.create 32 in
  print_child env ~gen_expr b e a ;
  Buffer.contents b

(* [e] as a value of the unsigned type of width [w]. A wrapping operator of the
   same width is printed in the unsigned domain directly, so a chain such as
   [i * 6 + 1] converts back to the signed type once, at its root: unsigned
   arithmetic is modular, so the low bits are those of every intermediate
   conversion. A non-negative literal takes the unsigned suffix. *)
let rec emit_unsigned env sp w ~gen_expr buf e (a : T.annot) =
  let unsigned i x = emit_unsigned env sp w ~gen_expr buf x a.kids.(i) in
  match e with
  | EBinop (op, x, y) when T.binop_width_a op a.kids.(0) a.kids.(1) = Some w ->
      Buffer.add_char buf '(' ;
      unsigned 0 x ;
      Buffer.add_string buf (operator op) ;
      (match op with
      | Shl -> add_operand buf (render_child env ~gen_expr y a.kids.(1))
      | _ -> unsigned 1 y) ;
      Buffer.add_char buf ')'
  | EUnop (Neg, x) when T.width_a a.kids.(0) = Some w -> (
      match Sarek_ir_intrinsic_dispatch.int_negation_constant Neg x with
      | Some c -> emit_unsigned env sp w ~gen_expr buf (EConst c) a
      | None ->
          Buffer.add_string buf "(-" ;
          unsigned 0 x ;
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
      add_operand buf (render_child env ~gen_expr e a)

(* The unsigned result back in the signed type: [((int)u)] in CUDA C, where
   the conversion is modular on every compiler (normative since C++20);
   [as_int(u)] / [as_type<int>(u)] in OpenCL C and MSL, a bit reinterpretation
   that no implementation-defined conversion is involved in. *)
let emit_wrapping env sp w ~gen_expr buf e a =
  let conv = match w with W32 -> sp.to_int32 | W64 -> sp.to_int64 in
  let cast = conv.[0] = '(' in
  Buffer.add_string buf (if cast then "(" ^ conv else conv ^ "(") ;
  emit_unsigned env sp w ~gen_expr buf e a ;
  Buffer.add_char buf ')'

(* Index arithmetic, printed as plain C: a chain of [Add]/[Sub]/[Mul]/[Shl] on
   signed integers at the root of a subscript, matrix offset, [for] bound or
   index let. Anything else ends the chain and is printed by [gen_expr],
   wrapping included. *)
let rec emit_index_a env ~gen_expr buf e (a : T.annot) =
  match e with
  | EBinop (((Add | Sub | Mul | Shl) as op), x, y)
    when T.binop_width_a op a.kids.(0) a.kids.(1) <> None ->
      Buffer.add_char buf '(' ;
      emit_index_a env ~gen_expr buf x a.kids.(0) ;
      Buffer.add_string buf (operator op) ;
      (match op with
      | Shl -> print_child env ~gen_expr buf y a.kids.(1)
      | _ -> emit_index_a env ~gen_expr buf y a.kids.(1)) ;
      Buffer.add_char buf ')'
  | _ -> print_child env ~gen_expr buf e a

let emit_index env ~gen_expr buf e =
  emit_index_a env ~gen_expr buf e (T.annot_of env e)

(* ---- the emitters' arms ---- *)

let emit_binop_arm env sp ~gen_expr ~operator_text buf e =
  let a = T.annot_of env e in
  match e with
  | EBinop (op, x, y) -> (
      match T.binop_width_a op a.kids.(0) a.kids.(1) with
      | Some w -> emit_wrapping env sp w ~gen_expr buf e a
      | None ->
          Buffer.add_char buf '(' ;
          print_child env ~gen_expr buf x a.kids.(0) ;
          Buffer.add_string buf operator_text ;
          print_child env ~gen_expr buf y a.kids.(1) ;
          Buffer.add_char buf ')')
  | _ -> invalid_arg "Sarek_ir_c_arith.emit_binop_arm: not an EBinop"

let emit_unop_arm env sp ~gen_expr ~operator_text buf e =
  match e with
  | EUnop (op, x) -> (
      match Sarek_ir_intrinsic_dispatch.int_negation_constant op x with
      | Some c -> gen_expr buf (EConst c)
      | None -> (
          let a = T.annot_of env e in
          match (op, T.width_a a.kids.(0)) with
          | Neg, Some w -> emit_wrapping env sp w ~gen_expr buf e a
          | _ ->
              Sarek_ir_intrinsic_dispatch.emit_unop
                ~gen_expr:(fun b x' ->
                  print_child env ~gen_expr b x' a.kids.(0))
                buf
                operator_text
                x))
  | _ -> invalid_arg "Sarek_ir_c_arith.emit_unop_arm: not an EUnop"

let emit_let_init env ~gen_expr buf v e =
  if is_index_let env v then emit_index env ~gen_expr buf e else gen_expr buf e
