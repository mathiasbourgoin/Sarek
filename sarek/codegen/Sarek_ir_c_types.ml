(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Operand types for the C-family wrapping arithmetic (Sarek_ir_c_arith). The
   contract is in the interface. *)

open Sarek_ir_types

type width = W32 | W64

type annot = {ty : elttype option; kids : annot array}

type env = {
  arrays : (string, elttype) Hashtbl.t;
  records : (string * (string * elttype) list) list;
  variants : (string * (string * elttype list) list) list;
  helpers : (string * elttype) list;
  index_lets : (string, unit) Hashtbl.t;
  mutable pending : (expr * annot) list;
  steps : int ref;
}

let create ~records ~variants ~helpers =
  {
    arrays = Hashtbl.create 16;
    records;
    variants;
    helpers;
    index_lets = Hashtbl.create 8;
    pending = [];
    steps = ref 0;
  }

let empty () = create ~records:[] ~variants:[] ~helpers:[]

let steps env = !(env.steps)

(* ---- array scope ---- *)

let array_elt = function TVec elt | TArray (elt, _) -> Some elt | _ -> None

let add_array env name elt = Hashtbl.add env.arrays name elt

let add_var_array env (v : var) =
  Option.iter (add_array env v.var_name) (array_elt v.var_type)

let add_decl env = function
  | DParam (v, Some ai) -> add_array env v.var_name ai.arr_elttype
  | DParam (v, None) -> add_var_array env v
  | DLocal (v, Some (EArrayCreate (elt, _, _))) -> add_array env v.var_name elt
  | DLocal (v, _) -> add_var_array env v
  | DShared (name, elt, _) -> add_array env name elt

let add_body env body =
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

let with_scope env = {env with arrays = Hashtbl.create 16; pending = []}

(* ---- types of nodes ---- *)

let const_type = function
  | CInt32 _ -> TInt32
  | CInt64 _ -> TInt64
  | CFloat32 _ -> TFloat32
  | CFloat64 _ -> TFloat64
  | CBool _ -> TBool
  | CUnit -> TUnit

let width_of_type = function
  | TInt32 | TBool | TUint8 | TInt8 -> Some W32
  | TInt64 -> Some W64
  | _ -> None

let is_float = function TFloat16 | TFloat32 | TFloat64 -> true | _ -> false

(* C's usual arithmetic conversions on two known operand types. *)
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

(* A variant value [v] is read as [v.data.C_v] (one payload) or
   [v.data.C_v._i] (several), the C-family layout of
   Sarek_ir_codegen.payload_access. [data] keeps the variant type; [C_v] is the
   payload, or a record of the payloads [_0], [_1], ... *)
let variant_field env (name, constrs) field =
  let constrs =
    if constrs = [] then
      Option.value ~default:[] (List.assoc_opt name env.variants)
    else constrs
  in
  if field = "data" then Some (TVariant (name, constrs))
  else
    let n = String.length field in
    if n < 3 || String.sub field (n - 2) 2 <> "_v" then None
    else
      match List.assoc_opt (String.sub field 0 (n - 2)) constrs with
      | Some [t] -> Some t
      | Some ts ->
          Some
            (TRecord
               ( name ^ "." ^ field,
                 List.mapi (fun i t -> (Printf.sprintf "_%d" i, t)) ts ))
      | None -> None

let field_type env ty field =
  match ty with
  | Some (TRecord (name, fields)) -> (
      match List.assoc_opt field fields with
      | Some t -> Some t
      | None ->
          Option.bind (List.assoc_opt name env.records) (List.assoc_opt field))
  | Some (TVariant (name, constrs)) -> variant_field env (name, constrs) field
  | _ -> None

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

let children = function
  | EConst _ | EVar _ | EArrayLen _ -> []
  | EBinop (_, a, b) | EArrayReadExpr (a, b) -> [a; b]
  | EUnop (_, a) | ERecordField (a, _) | ECast (_, a) | EArrayRead (_, a) -> [a]
  | EIntrinsic (_, _, l) | ETuple l | EVariant (_, _, l) -> l
  | EApp (f, l) -> f :: l
  | ERecord (_, fields) -> List.map snd fields
  | EArrayCreate (_, size, _) -> [size]
  | EIf (c, t, e) -> [c; t; e]
  | EMatch (scrutinee, cases) -> scrutinee :: List.map snd cases

(* The type of [e] from the types [k] of its children, in [children] order. *)
let node_type env e (k : elttype option array) =
  match e with
  | EConst c -> Some (const_type c)
  | EVar v -> Some v.var_type
  | EBinop ((Eq | Ne | Lt | Le | Gt | Ge | And | Or), _, _) | EUnop (Not, _) ->
      Some TBool
  | EBinop (Lshr32, _, _) -> Some TInt32
  | EBinop (Lshr64, _, _) -> Some TInt64
  | EBinop ((Shl | Shr), _, _) | EUnop ((Neg | BitNot), _) -> arith k.(0) k.(0)
  | EBinop ((Add | Sub | Mul | Div | Mod | BitAnd | BitOr | BitXor), _, _) ->
      arith k.(0) k.(1)
  | EArrayRead (arr, _) -> lookup_array env arr
  | EArrayReadExpr _ -> Option.bind k.(0) array_elt
  | ERecordField (_, field) -> field_type env k.(0) field
  | EIntrinsic (path, name, _) -> intrinsic_type path name
  | ECast (t, _) -> Some t
  | EApp (EVar f, _) -> List.assoc_opt f.var_name env.helpers
  | EArrayLen _ -> Some TInt32
  | EIf _ -> if k.(1) = k.(2) then k.(1) else None
  | EMatch (_, _ :: _) -> k.(1)
  | EApp _ | ETuple _ | ERecord _ | EVariant _ | EArrayCreate _ | EMatch _ ->
      None

(* Bottom-up, once per node: [steps] counts the nodes typed. *)
let rec annotate env e =
  incr env.steps ;
  let kids = Array.of_list (List.map (annotate env) (children e)) in
  {ty = node_type env e (Array.map (fun a -> a.ty) kids); kids}

(* The annotation of [e]: the one a caller pushed for it, or for its parent
   (one level of non-arithmetic node in between, a cast or a read), else a
   fresh one. Emitting a subtree therefore types each of its nodes once. *)
let annot_of env e =
  match env.pending with
  | (p, a) :: _ when p == e -> a
  | (p, a) :: _ -> (
      let rec find i = function
        | [] -> None
        | c :: rest -> if c == e then Some a.kids.(i) else find (i + 1) rest
      in
      match find 0 (children p) with Some a -> a | None -> annotate env e)
  | [] -> annotate env e

let with_pending env e a f =
  env.pending <- (e, a) :: env.pending ;
  Fun.protect ~finally:(fun () -> env.pending <- List.tl env.pending) f

let width_a a = Option.bind a.ty width_of_type

let binop_width_a op ka kb =
  match op with
  | Add | Sub | Mul -> Option.bind (arith ka.ty kb.ty) width_of_type
  | Shl -> width_a ka
  | Div | Mod | Eq | Ne | Lt | Le | Gt | Ge | And | Or | Shr | Lshr32 | Lshr64
  | BitAnd | BitOr | BitXor ->
      None

let int_width env e = width_a (annot_of env e)

let binop_width env op a b =
  let n = EBinop (op, a, b) in
  let an = annotate env n in
  binop_width_a op an.kids.(0) an.kids.(1)
