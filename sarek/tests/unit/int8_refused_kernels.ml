(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Kernels that use an 8-bit type where it is not supported, each with a
    substring of the refusal every backend and the interpreter must give
    ([Sarek_ir_int8.refusal]). Used by test_int8_storage. *)

open Sarek_ir_types
module K = Int8_kernels

(** A scalar 8-bit kernel PARAMETER (an 8-bit [let] is accepted, see
    {!K.scalar_kernel}). *)
let scalar_param_kernel =
  let x = K.var 0 "x" TUint8 and out = K.var 1 "out" (TVec TInt32) in
  {
    default_kernel with
    kern_name = "int8_param";
    kern_params = [DParam (x, None); K.vec_param out TInt32];
    kern_body = K.store "out" (K.i32 0) (EVar x);
  }

let one_u8_buffer name body =
  let b = K.var 0 "b" (TVec TUint8) and out = K.var 1 "out" (TVec TInt32) in
  {
    default_kernel with
    kern_name = name;
    kern_params = [K.vec_param b TUint8; K.vec_param out TInt32];
    kern_body = body;
  }

let float_cast_kernel =
  one_u8_buffer
    "int8_float_cast"
    (K.store "out" (K.i32 0) (ECast (TUint8, EConst (CFloat32 300.0))))

let float_store_kernel =
  one_u8_buffer
    "int8_float_store"
    (K.store "b" (K.i32 0) (ECast (TFloat32, K.read "out" (K.i32 0))))

let float_store_expr_kernel =
  one_u8_buffer
    "int8_float_store_expr"
    (SAssign
       ( LArrayElemExpr (EVar (K.var 0 "b" (TVec TUint8)), K.i32 0),
         EConst (CFloat32 300.0) ))

let atomic_kernel =
  one_u8_buffer
    "int8_atomic"
    (SExpr
       (EIntrinsic
          ( [],
            "atomic_add_global_int32",
            [EVar (K.var 0 "b" (TVec TUint8)); K.i32 0; K.i32 1] )))

let expr_base_kernel =
  one_u8_buffer
    "int8_expr_base"
    (K.store
       "out"
       (K.i32 0)
       (EArrayReadExpr
          ( EIf
              ( EConst (CBool true),
                EVar (K.var 0 "b" (TVec TUint8)),
                EVar (K.var 0 "b" (TVec TUint8)) ),
            K.i32 0 )))

let ambiguous_kernel =
  let b = K.var 0 "b" (TVec TUint8)
  and out = K.var 1 "out" (TVec TInt32)
  and l = K.var 2 "b" (TArray (TInt32, Shared)) in
  {
    default_kernel with
    kern_name = "int8_ambiguous";
    kern_params = [K.vec_param b TUint8; K.vec_param out TInt32];
    kern_body =
      SLet
        ( l,
          EArrayCreate (TInt32, K.i32 4, Shared),
          K.store "out" (K.i32 0) (K.read "b" (K.i32 0)) );
  }

let helper_result_kernel =
  let out = K.var 0 "out" (TVec TInt32) in
  {
    default_kernel with
    kern_name = "int8_helper";
    kern_params = [K.vec_param out TInt32];
    kern_funcs =
      [
        {
          hf_name = "h";
          hf_params = [];
          hf_ret_type = TInt8;
          hf_body = SReturn (K.i32 1);
        };
      ];
    kern_body = K.store "out" (K.i32 0) (K.i32 1);
  }

let variant_kernel =
  let out = K.var 0 "out" (TVec TInt32) in
  {
    default_kernel with
    kern_name = "int8_variant";
    kern_params = [K.vec_param out TInt32];
    kern_variants = [("opt", [("Some", [TUint8]); ("None", [])])];
    kern_body = K.store "out" (K.i32 0) (K.i32 1);
  }

let record_field_kernel =
  let out = K.var 0 "out" (TVec TInt32) in
  {
    default_kernel with
    kern_name = "int8_field";
    kern_params = [K.vec_param out TInt32];
    kern_types = [("px", [("a", TInt32); ("b", TInt8)])];
    kern_body = K.store "out" (K.i32 0) (K.i32 1);
  }

let local_array_kernel =
  let out = K.var 0 "out" (TVec TInt32)
  and l = K.var 1 "l" (TArray (TInt8, Local)) in
  {
    default_kernel with
    kern_name = "int8_local";
    kern_params = [K.vec_param out TInt32];
    kern_body =
      SLet
        ( l,
          EArrayCreate (TInt8, K.i32 4, Local),
          K.store "out" (K.i32 0) (K.read "l" (K.i32 0)) );
  }

let all =
  [
    (scalar_param_kernel, "is a uint8 parameter or loop variable");
    (float_cast_kernel, "a float value is cast or stored");
    (float_store_kernel, "a float value is cast or stored");
    (float_store_expr_kernel, "a float value is cast or stored");
    (atomic_kernel, "on an 8-bit array");
    (expr_base_kernel, "given by an expression rather than a name");
    (ambiguous_kernel, "is bound both to an 8-bit and to another");
    (helper_result_kernel, "helper 'h' returns int8");
    (variant_kernel, "has an 8-bit payload");
    (record_field_kernel, "has the int8 field");
    (local_array_kernel, "private (Local) memory");
  ]
