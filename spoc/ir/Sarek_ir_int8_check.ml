(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Where the 8-bit storage types may appear, and the refusals for everywhere
    else. Re-exported by {!Sarek_ir_int8}, which documents the semantics; split
    out so that each file stays short. *)

open Sarek_ir_types

let is_8bit = function TUint8 | TInt8 -> true | _ -> false

let byte_elt_of_type = function
  | (TVec t | TArray (t, _)) when is_8bit t -> Some t
  | _ -> None

(** {1 Binders}

    One walk collects every name the kernel binds together with the type it
    binds it at, so that both {!byte_arrays} and {!refusal} read the same list.
    A [DShared] declares an array of its element type; everything else carries
    the binder's own [var_type]. *)

type binder =
  | Array_binder of string * elttype  (** an array of this element type *)
  | Scalar_binder of string * elttype
      (** a scalar parameter or loop variable of this type *)
  | Local_scalar of string * elttype
      (** a [let] (mutable or not) or [DLocal] scalar of this type *)
  | Local_array of string * elttype  (** [EArrayCreate (t, _, Local)] *)

let of_var (v : var) =
  match v.var_type with
  | TVec t | TArray (t, _) -> Array_binder (v.var_name, t)
  | t -> Scalar_binder (v.var_name, t)

let of_local (v : var) =
  match of_var v with Scalar_binder (n, t) -> Local_scalar (n, t) | b -> b

let rec stmt_binders acc = function
  | SLet (v, e, body) | SLetMut (v, e, body) ->
      let b =
        match e with
        | EArrayCreate (t, _, Local) -> Local_array (v.var_name, t)
        | _ -> of_local v
      in
      stmt_binders (b :: acc) body
  | SFor (v, _, _, _, body) -> stmt_binders (of_var v :: acc) body
  | SSeq ss -> List.fold_left stmt_binders acc ss
  | SIf (_, t, e) ->
      let acc = stmt_binders acc t in
      Option.fold ~none:acc ~some:(stmt_binders acc) e
  | SWhile (_, b) | SPragma (_, b) | SBlock b -> stmt_binders acc b
  | SMatch (_, cases) ->
      List.fold_left (fun a (_, s) -> stmt_binders a s) acc cases
  | SAssign _ | SReturn _ | SBarrier | SWarpBarrier | SExpr _ | SEmpty
  | SMemFence | SNative _ | SCoopmat _ ->
      acc

let decl_binder = function
  | DParam (v, _) -> of_var v
  | DLocal (({var_type = TArray (t, Local); _} as v), _) ->
      Local_array (v.var_name, t)
  | DLocal (v, _) -> of_local v
  | DShared (name, t, _) -> Array_binder (name, t)

let binders (k : kernel) : binder list =
  let acc = List.map decl_binder (k.kern_params @ k.kern_locals) in
  let acc = stmt_binders acc k.kern_body in
  List.fold_left
    (fun acc hf -> stmt_binders (List.map of_var hf.hf_params @ acc) hf.hf_body)
    acc
    k.kern_funcs

let byte_arrays (k : kernel) : (string * elttype) list =
  List.fold_left
    (fun acc -> function
      | Array_binder (name, t) when is_8bit t && not (List.mem_assoc name acc)
        ->
          (name, t) :: acc
      | Array_binder _ | Scalar_binder _ | Local_scalar _ | Local_array _ -> acc)
    []
    (binders k)
  |> List.rev

(** {1 Refusals}

    One function per kind of misuse, each returning the first offending site's
    sentence. {!refusal} asks them in a fixed order. *)

let type_name = function TInt8 -> "int8" | _ -> "uint8"

let refuse_scalar_binder bs =
  List.find_map
    (function
      | Scalar_binder (n, t) when is_8bit t ->
          Some
            (Printf.sprintf
               "'%s' is a %s parameter or loop variable. 8-bit types are \
                storage types: pass or iterate over an int32 instead"
               n
               (type_name t))
      | _ -> None)
    bs

let refuse_helper_result (k : kernel) =
  List.find_map
    (fun hf ->
      if is_8bit hf.hf_ret_type then
        Some
          (Printf.sprintf
             "helper '%s' returns %s. 8-bit types are storage types: return \
              the int32 value instead"
             hf.hf_name
             (type_name hf.hf_ret_type))
      else None)
    k.kern_funcs

let refuse_field (k : kernel) =
  let record =
    List.find_map
      (fun (rname, fields) ->
        List.find_map
          (fun (fname, t) ->
            if is_8bit t then
              Some
                (Printf.sprintf
                   "record '%s' has the %s field '%s'. 8-bit types are array \
                    element types only; use an int32 field"
                   rname
                   (type_name t)
                   fname)
            else None)
          fields)
      k.kern_types
  in
  let variant () =
    List.find_map
      (fun (vname, ctors) ->
        if List.exists (fun (_, args) -> List.exists is_8bit args) ctors then
          Some
            (Printf.sprintf
               "variant '%s' has an 8-bit payload. 8-bit types are array \
                element types only; use int32"
               vname)
        else None)
      k.kern_variants
  in
  match record with Some _ -> record | None -> variant ()

let refuse_local_array bs =
  List.find_map
    (function
      | Local_array (n, t) when is_8bit t ->
          Some
            (Printf.sprintf
               "'%s' is a %s array in private (Local) memory. 8-bit arrays are \
                supported in global and shared memory only"
               n
               (type_name t))
      | _ -> None)
    bs

(** Backends resolve an [EArrayRead] by NAME, so one name bound both to an 8-bit
    array and to any other array (or to both signednesses) would be read with
    the wrong extension somewhere. *)
let refuse_ambiguous bs bytes =
  let clash n t =
    match List.assoc_opt n bytes with
    | Some t' when t' <> t ->
        Some
          (Printf.sprintf
             "the array name '%s' is bound both to an 8-bit and to another \
              element type (%s and %s); rename one of them"
             n
             (type_name t')
             (Sarek_ir_pp.string_of_elttype t))
    | _ -> None
  in
  List.find_map
    (function
      | Array_binder (n, t) | Local_array (n, t) -> clash n t
      | Scalar_binder _ | Local_scalar _ -> None)
    bs

(** {2 Expression-level misuse} *)

let is_float = function TFloat16 | TFloat32 | TFloat64 -> true | _ -> false

(** Whether [e] is recognisably a float: the IR carries no expression types, so
    this follows constants, casts, variables and array reads. A float that
    reaches an 8-bit cast or store would be converted out of range differently
    by the interpreter (through int32) and by C/GLSL (undefined). *)
let rec float_shaped arrays e =
  let r = float_shaped arrays in
  match e with
  | EConst (CFloat32 _ | CFloat64 _) -> true
  | ECast (t, _) -> is_float t
  | EVar v -> is_float v.var_type
  | EArrayRead (a, _) -> (
      match List.assoc_opt a arrays with Some t -> is_float t | None -> false)
  | EBinop ((Add | Sub | Mul | Div | Mod), a, b) -> r a || r b
  | EUnop (Neg, a) -> r a
  | EIf (_, a, b) -> r a || r b
  | _ -> false

let base_refusal what =
  Some
    (Printf.sprintf
       "%s an array given by an expression rather than a name, in a kernel \
        with 8-bit arrays: the element type cannot be resolved, so the 8-bit \
        read or store semantics could not be applied; bind the array to a name"
       what)

let float_refusal =
  Some
    "a float value is cast or stored to an 8-bit integer type. 8-bit types \
     store the low byte of an integer, so convert the value to int32 first"

let expr_misuse arrays bytes = function
  | ECast (t, x) when is_8bit t && float_shaped arrays x -> float_refusal
  | EIntrinsic (_, name, args)
    when Sarek_ir_analysis.is_atomic_intrinsic_name name
         && List.exists
              (function
                | EVar v -> Option.is_some (byte_elt_of_type v.var_type)
                | EArrayRead (a, _) -> List.mem_assoc a bytes
                | _ -> false)
              args ->
      Some
        (Printf.sprintf
           "atomic '%s' on an 8-bit array: atomics are 32- or 64-bit only"
           name)
  | EArrayReadExpr (EVar _, _) -> None
  | EArrayReadExpr _ -> base_refusal "a read of"
  | _ -> None

let stmt_misuse arrays bytes = function
  | SAssign (LArrayElemExpr (EVar _, _), _) -> None
  | SAssign (LArrayElemExpr _, _) -> base_refusal "a store into"
  | SAssign (LArrayElem (a, _), x)
    when List.mem_assoc a bytes && float_shaped arrays x ->
      float_refusal
  | _ -> None

let refuse_expressions (k : kernel) bs bytes =
  let arrays =
    List.filter_map
      (function
        | Array_binder (n, t) | Local_array (n, t) -> Some (n, t)
        | Scalar_binder _ | Local_scalar _ -> None)
      bs
  in
  let found = ref None in
  let note r = if Option.is_none !found then found := r in
  let folder =
    Sarek_ir_analysis.exists_folder
      ~leaf:(fun e ->
        note (expr_misuse arrays bytes e) ;
        false)
      ~stmt_leaf:(fun s ->
        note (stmt_misuse arrays bytes s) ;
        false)
      ~native:false
      ~visit_lvalue:true
      ()
  in
  ignore (Sarek_ir_analysis.kernel_fold folder false k) ;
  !found

let refusal (k : kernel) : string option =
  let bs = binders k in
  let bytes = byte_arrays k in
  let checks =
    [
      (fun () -> refuse_scalar_binder bs);
      (fun () -> refuse_helper_result k);
      (fun () -> refuse_field k);
      (fun () -> refuse_local_array bs);
      (fun () -> refuse_ambiguous bs bytes);
      (fun () ->
        if
          bytes = []
          && not (Sarek_ir_analysis.kernel_uses Sarek_ir_analysis.Int8 k)
        then None
        else refuse_expressions k bs bytes);
    ]
  in
  List.find_map (fun c -> c ()) checks
