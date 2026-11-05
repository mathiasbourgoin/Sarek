(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** IR kernels exercising every position a float16 value can occupy in the PTX
    backend: vector loads/stores, widening and narrowing casts, locals, shared
    arrays, selects, helper parameters and returns, and integer conversions. f16
    is a storage type, so none of them contains f16 arithmetic: values are
    widened to f32, computed on, and narrowed back.

    Shared by the structural test (test_ptx_f16.ml) and the ptxas assembly gate
    (test_ptx_snapshot.ml), so what is asserted on the text is what is
    assembled. *)

open Sarek_ir_types

let var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let kernel ?(funcs = []) name params body =
  {
    default_kernel with
    kern_name = name;
    kern_params = params;
    kern_body = body;
    kern_funcs = funcs;
  }

let vec_param name elt =
  DParam (var name (TVec elt), Some {arr_elttype = elt; arr_memspace = Global})

let gid = EIntrinsic ([], "global_thread_id", [])

let widen e = ECast (TFloat32, e)

let narrow e = ECast (TFloat16, e)

let with_gid body =
  let idx = var "idx" TInt32 in
  SLet (idx, gid, body (EVar idx))

(** out[i] = f16(f32(inp[i]) * 1.1) *)
let scale () =
  kernel
    "f16_scale"
    [vec_param "out" TFloat16; vec_param "inp" TFloat16]
    (with_gid (fun i ->
         SAssign
           ( LArrayElem ("out", i),
             narrow
               (EBinop
                  (Mul, widen (EArrayRead ("inp", i)), EConst (CFloat32 1.1)))
           )))

(** out[i] = f16(f16(f32(inp[i]) * 1.1) + 1000): two narrowings in a chain. *)
let chain () =
  let inner =
   fun i ->
    narrow (EBinop (Mul, widen (EArrayRead ("inp", i)), EConst (CFloat32 1.1)))
  in
  kernel
    "f16_chain"
    [vec_param "out" TFloat16; vec_param "inp" TFloat16]
    (with_gid (fun i ->
         SAssign
           ( LArrayElem ("out", i),
             narrow (EBinop (Add, widen (inner i), EConst (CFloat32 1000.0))) )))

(** out[i] = inp[i]: a raw 16-bit load and store, no conversion at all. *)
let roundtrip () =
  kernel
    "f16_roundtrip"
    [vec_param "out" TFloat16; vec_param "inp" TFloat16]
    (with_gid (fun i -> SAssign (LArrayElem ("out", i), EArrayRead ("inp", i))))

(** A mutable f16 local, assigned twice, then widened: mov.b16 on a [%h]. *)
let local_var () =
  let h = {(var "h" TFloat16) with var_mutable = true} in
  kernel
    "f16_local"
    [vec_param "out" TFloat32; vec_param "inp" TFloat16]
    (with_gid (fun i ->
         SLetMut
           ( h,
             EArrayRead ("inp", i),
             SSeq
               [
                 SAssign (LVar h, narrow (EConst (CFloat32 2.5)));
                 SAssign
                   ( LArrayElem ("out", i),
                     EBinop (Add, widen (EVar h), widen (EArrayRead ("inp", i)))
                   );
               ] )))

(** A shared f16 array: ld/st.shared.b16 with a 2-byte stride. *)
let shared_array () =
  let sh = var "sh" (TArray (TFloat16, Shared)) in
  let tid = var "tid" TInt32 in
  kernel
    "f16_shared"
    [vec_param "out" TFloat32; vec_param "inp" TFloat16]
    (SLet
       ( tid,
         EIntrinsic ([], "thread_idx_x", []),
         SLet
           ( sh,
             EArrayCreate (TFloat16, EConst (CInt32 256l), Shared),
             SSeq
               [
                 SAssign
                   (LArrayElem ("sh", EVar tid), EArrayRead ("inp", EVar tid));
                 SBarrier;
                 SAssign
                   ( LArrayElem ("out", EVar tid),
                     widen
                       (EArrayRead
                          ("sh", EBinop (Sub, EConst (CInt32 255l), EVar tid)))
                   );
               ] ) ))

(** A per-thread local f16 array: ld/st.local.b16. *)
let local_array () =
  let la = var "la" (TArray (TFloat16, Local)) in
  kernel
    "f16_local_array"
    [vec_param "out" TFloat32; vec_param "inp" TFloat16]
    (with_gid (fun i ->
         SLet
           ( la,
             EArrayCreate (TFloat16, EConst (CInt32 4l), Local),
             SSeq
               [
                 SAssign
                   (LArrayElem ("la", EConst (CInt32 1l)), EArrayRead ("inp", i));
                 SAssign
                   ( LArrayElem ("out", i),
                     widen (EArrayRead ("la", EConst (CInt32 1l))) );
               ] )))

(** Integer and double conversions: f16 -> int32 (widen, truncate), int32 -> f16
    (convert to f32, narrow), f64 -> f16, f16 -> f64. *)
let int_and_double_casts () =
  kernel
    "f16_casts"
    [
      vec_param "oh" TFloat16;
      vec_param "oi" TInt32;
      vec_param "od" TFloat64;
      vec_param "oh2" TFloat16;
      vec_param "inp" TFloat16;
    ]
    (with_gid (fun i ->
         SSeq
           [
             SAssign
               (LArrayElem ("oi", i), ECast (TInt32, EArrayRead ("inp", i)));
             SAssign (LArrayElem ("oh", i), narrow i);
             SAssign
               (LArrayElem ("od", i), ECast (TFloat64, EArrayRead ("inp", i)));
             SAssign
               ( LArrayElem ("oh2", i),
                 narrow (ECast (TFloat64, EArrayRead ("inp", i))) );
           ]))

(** A select between two f16 values: [selp.b16] (no array read in a branch). *)
let select_values () =
  kernel
    "f16_select"
    [vec_param "out" TFloat16; vec_param "inp" TFloat16]
    (with_gid (fun i ->
         let a = var "a" TFloat16 and b = var "b" TFloat16 in
         SLet
           ( a,
             EArrayRead ("inp", i),
             SLet
               ( b,
                 narrow (EConst (CFloat32 0.0)),
                 SAssign
                   ( LArrayElem ("out", i),
                     EIf (EBinop (Lt, i, EConst (CInt32 8l)), EVar a, EVar b) )
               ) )))

(** A select whose branches load from memory: real control flow, f16 merged with
    [mov.b16]. *)
let select_loads () =
  kernel
    "f16_select_loads"
    [vec_param "out" TFloat16; vec_param "inp" TFloat16]
    (with_gid (fun i ->
         SAssign
           ( LArrayElem ("out", i),
             EIf
               ( EBinop (Lt, i, EConst (CInt32 8l)),
                 EArrayRead ("inp", i),
                 EArrayRead ("inp", EConst (CInt32 0l)) ) )))

(** An inlined helper taking and returning f16: parameter copy, result register
    and the return move are all [%h]. *)
let helper_roundtrip () =
  let x = var "x" TFloat16 in
  let helper =
    {
      hf_name = "bump";
      hf_params = [x];
      hf_ret_type = TFloat16;
      hf_body =
        SReturn (narrow (EBinop (Add, widen (EVar x), EConst (CFloat32 1.0))));
    }
  in
  kernel
    ~funcs:[helper]
    "f16_helper"
    [vec_param "out" TFloat16; vec_param "inp" TFloat16]
    (with_gid (fun i ->
         SAssign
           ( LArrayElem ("out", i),
             EApp (EVar (var "bump" TFloat16), [EArrayRead ("inp", i)]) )))

(** Every kernel above, by name. *)
let all () =
  [
    ("f16_scale", scale ());
    ("f16_chain", chain ());
    ("f16_roundtrip", roundtrip ());
    ("f16_local", local_var ());
    ("f16_shared", shared_array ());
    ("f16_local_array", local_array ());
    ("f16_casts", int_and_double_casts ());
    ("f16_select", select_values ());
    ("f16_select_loads", select_loads ());
    ("f16_helper", helper_roundtrip ());
  ]
