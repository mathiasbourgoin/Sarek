(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** IR kernels over the 8-bit storage types [TUint8]/[TInt8], with an
    independent host model of what they compute. Shared by the CPU unit test
    (interpreter and codegen) and the device e2e test, so both check the same
    kernels against the same model.

    The model is written from the semantics, not from [Sarek_ir_int8.narrow]: a
    sign extension spelled [(x land 0xff) lxor 0x80 - 0x80] here and as a pair
    of shifts there, so a defect in one is not copied into the other. *)

open Sarek_ir_types

let var id name ty =
  {var_name = name; var_id = id; var_type = ty; var_mutable = false}

let i32 n = EConst (CInt32 (Int32.of_int n))

let gid = EIntrinsic ([], "global_thread_id", [])

let tid = EIntrinsic ([], "thread_idx_x", [])

let vec_param v elt = DParam (v, Some {arr_elttype = elt; arr_memspace = Global})

let ( +: ) a b = EBinop (Add, a, b)

let ( *: ) a b = EBinop (Mul, a, b)

let ( -: ) a b = EBinop (Sub, a, b)

let xor a b = EBinop (BitXor, a, b)

let read a i = EArrayRead (a, i)

let store a i e = SAssign (LArrayElem (a, i), e)

(** {1 Host model} *)

let u8 x = x land 0xff

let s8 x = (x land 0xff lxor 0x80) - 0x80

(** {1 Global reads, stores and casts} *)

(** Slots per thread in [out] of {!rw_kernel}. *)
let rw_slots = 6

let rw_xor = 0x5a

(** One thread per element [i]:

    - [out.(6i)] the u8 element read (zero extension);
    - [out.(6i+1)] the s8 element read (sign extension);
    - [out.(6i+2)], [out.(6i+3)] the narrowing casts of an int32 to u8 and s8;
    - [out.(6i+4)] int32 arithmetic on two 8-bit reads and a widening cast;
    - [out.(6i+5)] u8 then s8: a cast of a cast;
    - [dst_u8.(i)], [dst_s8.(i)] plain stores of int32 values (low byte). *)
let rw_kernel () : kernel =
  let src_u8 = var 0 "src_u8" (TVec TUint8)
  and src_s8 = var 1 "src_s8" (TVec TInt8)
  and vals = var 2 "vals" (TVec TInt32)
  and out = var 3 "out" (TVec TInt32)
  and dst_u8 = var 4 "dst_u8" (TVec TUint8)
  and dst_s8 = var 5 "dst_s8" (TVec TInt8)
  and i = var 6 "i" TInt32 in
  let iv = EVar i in
  let slot k = (iv *: i32 rw_slots) +: i32 k in
  let v = read "vals" iv in
  let bu = read "src_u8" iv and bs = read "src_s8" iv in
  {
    default_kernel with
    kern_name = "int8_rw";
    kern_params =
      [
        vec_param src_u8 TUint8;
        vec_param src_s8 TInt8;
        vec_param vals TInt32;
        vec_param out TInt32;
        vec_param dst_u8 TUint8;
        vec_param dst_s8 TInt8;
      ];
    kern_body =
      SLet
        ( i,
          gid,
          SSeq
            [
              store "out" (slot 0) bu;
              store "out" (slot 1) bs;
              store "out" (slot 2) (ECast (TUint8, v));
              store "out" (slot 3) (ECast (TInt8, v));
              store "out" (slot 4) ((bs *: bu) +: ECast (TInt32, bs));
              store "out" (slot 5) (ECast (TInt8, ECast (TUint8, v)));
              store "dst_u8" iv v;
              store "dst_s8" iv (xor v (i32 rw_xor));
            ] );
  }

(** The model of {!rw_kernel}: [(out, dst_u8, dst_s8)] from the inputs. [?sext]
    replaces the sign extension, so a caller can build a deliberately wrong
    model (zero extension) and check that a comparison rejects it. *)
let rw_model ?(sext = s8) ~(src_u8 : int array) ~(src_s8 : int array)
    ~(vals : int32 array) () =
  let n = Array.length vals in
  let out = Array.make (rw_slots * n) 0l in
  let dst_u8 = Array.make n 0 and dst_s8 = Array.make n 0 in
  for i = 0 to n - 1 do
    let v = Int32.to_int vals.(i) and bu = src_u8.(i) and bs = src_s8.(i) in
    let set k x = out.((rw_slots * i) + k) <- Int32.of_int x in
    set 0 bu ;
    set 1 bs ;
    set 2 (u8 v) ;
    set 3 (sext v) ;
    set 4 ((bs * bu) + bs) ;
    set 5 (sext (u8 v)) ;
    dst_u8.(i) <- u8 v ;
    dst_s8.(i) <- sext (v lxor rw_xor)
  done ;
  (out, dst_u8, dst_s8)

(** {1 Shared arrays} *)

(** Threads per block of {!shared_kernel}, and the length of its two shared
    arrays. *)
let block = 64

let shared_xor = 0x33

(** Each thread stores its int32 into a u8 and (xor-ed) an s8 shared array, then
    after a barrier reads its MIRROR thread's elements back: the values cross
    threads, so a store that kept more than 8 bits in a register could not pass.
    [out.(2g)] is the mirrored u8 element, [out.(2g+1)] the mirrored s8 element
    plus the u8 element cast to s8. *)
let shared_kernel () : kernel =
  let vals = var 0 "vals" (TVec TInt32)
  and out = var 1 "out" (TVec TInt32)
  and g = var 2 "g" TInt32
  and t = var 3 "t" TInt32
  and su = var 4 "su" (TArray (TUint8, Shared))
  and ss = var 5 "ss" (TArray (TInt8, Shared)) in
  let gv = EVar g and tv = EVar t in
  let mirror = i32 (block - 1) -: tv in
  let v = read "vals" gv in
  {
    default_kernel with
    kern_name = "int8_shared";
    kern_params = [vec_param vals TInt32; vec_param out TInt32];
    kern_body =
      SLet
        ( g,
          gid,
          SLet
            ( t,
              tid,
              SLet
                ( su,
                  EArrayCreate (TUint8, i32 block, Shared),
                  SLet
                    ( ss,
                      EArrayCreate (TInt8, i32 block, Shared),
                      SSeq
                        [
                          store "su" tv v;
                          store "ss" tv (xor v (i32 shared_xor));
                          SBarrier;
                          store "out" (gv *: i32 2) (read "su" mirror);
                          store
                            "out"
                            ((gv *: i32 2) +: i32 1)
                            (read "ss" mirror +: ECast (TInt8, read "su" mirror));
                        ] ) ) ) );
  }

let shared_model ?(sext = s8) ~(vals : int32 array) () =
  let n = Array.length vals in
  let out = Array.make (2 * n) 0l in
  for g = 0 to n - 1 do
    let m = (g / block * block) + (block - 1 - (g mod block)) in
    let v = Int32.to_int vals.(m) in
    out.(2 * g) <- Int32.of_int (u8 v) ;
    out.((2 * g) + 1) <- Int32.of_int (sext (v lxor shared_xor) + sext (u8 v))
  done ;
  out

(** {1 Local scalars} *)

let scalar_slots = 3

let scalar_xor = 0x11

let scalar_add = 200

(** An 8-bit [let] holds the narrowed value, as a C [unsigned char] local does,
    and an assignment to a mutable one narrows again:

    - [x : uint8 = vals.(i)];
    - [y : int8 = vals.(i) lxor 0x11], then [y := y + 200];
    - [out.(3i)] = x, [out.(3i+1)] = y, [out.(3i+2)] = x + y. *)
let scalar_kernel () : kernel =
  let vals = var 0 "vals" (TVec TInt32)
  and out = var 1 "out" (TVec TInt32)
  and i = var 2 "i" TInt32
  and x = var 3 "x" TUint8
  and y = {(var 4 "y" TInt8) with var_mutable = true} in
  let iv = EVar i in
  let slot k = (iv *: i32 scalar_slots) +: i32 k in
  let v = read "vals" iv in
  {
    default_kernel with
    kern_name = "int8_scalar";
    kern_params = [vec_param vals TInt32; vec_param out TInt32];
    kern_body =
      SLet
        ( i,
          gid,
          SLet
            ( x,
              v,
              SLetMut
                ( y,
                  xor v (i32 scalar_xor),
                  SSeq
                    [
                      SAssign (LVar y, EVar y +: i32 scalar_add);
                      store "out" (slot 0) (EVar x);
                      store "out" (slot 1) (EVar y);
                      store "out" (slot 2) (EVar x +: EVar y);
                    ] ) ) );
  }

let scalar_model ?(sext = s8) ~(vals : int32 array) () =
  let n = Array.length vals in
  let out = Array.make (scalar_slots * n) 0l in
  for i = 0 to n - 1 do
    let v = Int32.to_int vals.(i) in
    let x = u8 v in
    let y = sext (sext (v lxor scalar_xor) + scalar_add) in
    let set k z = out.((scalar_slots * i) + k) <- Int32.of_int z in
    set 0 x ;
    set 1 y ;
    set 2 (x + y)
  done ;
  out

(** {1 Inputs} *)

(** Elements per kernel: four blocks of {!block}. *)
let n = 4 * block

(** The int32 inputs: every byte boundary and sign boundary first, then a
    deterministic spread over the whole int32 range. *)
let vals =
  let edges =
    [|
      0l;
      1l;
      127l;
      128l;
      255l;
      256l;
      257l;
      300l;
      -1l;
      -127l;
      -128l;
      -129l;
      -255l;
      -256l;
      -300l;
      Int32.max_int;
      Int32.min_int;
      0x12345678l;
      0x7fffff80l;
      0x800000ffl;
    |]
  in
  let state = ref 0x2545F491l in
  Array.init n (fun i ->
      if i < Array.length edges then edges.(i)
      else begin
        (* xorshift32 *)
        let x = !state in
        let x = Int32.logxor x (Int32.shift_left x 13) in
        let x = Int32.logxor x (Int32.shift_right_logical x 17) in
        let x = Int32.logxor x (Int32.shift_left x 5) in
        state := x ;
        x
      end)

(** Every u8 value, four times over. *)
let src_u8 = Array.init n (fun i -> i land 0xff)

(** Every s8 value, in a different order from {!src_u8} so the product in
    {!rw_kernel} pairs different bytes. *)
let src_s8 = Array.init n (fun i -> s8 ((i * 37) + 11))
