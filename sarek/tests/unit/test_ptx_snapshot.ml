(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** PTX snapshot test for the code generator.

    Constructs a minimal vector_add Sarek IR kernel and calls
    [Sarek_ir_ptx_kernel.generate] directly, asserting that the emitted PTX
    string contains canonical structural markers. This test is CPU-only: no CUDA
    device is required. *)

open Sarek_ir_types
open Sarek_codegen

(** Build a minimal vector_add kernel IR: for tid in 0..n: c[tid] = a[tid] +
    b[tid] Parameters: a, b, c (TVec TFloat32), n (TInt32). *)
let make_vector_add_kernel () : kernel =
  let make_var name ty =
    {var_name = name; var_id = 0; var_type = ty; var_mutable = false}
  in
  let a = make_var "a" (TVec TFloat32) in
  let b = make_var "b" (TVec TFloat32) in
  let c = make_var "c" (TVec TFloat32) in
  let n = make_var "n" TInt32 in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SIf
          ( EBinop (Lt, EVar tid, EVar n),
            SAssign
              ( LArrayElem ("c", EVar tid),
                EBinop
                  (Add, EArrayRead ("a", EVar tid), EArrayRead ("b", EVar tid))
              ),
            None ) )
  in
  {
    kern_name = "vector_add";
    kern_params =
      [
        DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (c, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (n, None);
      ];
    kern_locals = [];
    kern_body = body;
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

(** Check that [ptx] contains [marker]; fail with a readable message if not. *)
let assert_contains ptx marker =
  if not (String.length ptx >= String.length marker) then
    Alcotest.fail (Printf.sprintf "PTX too short to contain %S:\n%s" marker ptx) ;
  let found = ref false in
  let mlen = String.length marker in
  let plen = String.length ptx in
  for i = 0 to plen - mlen do
    if String.sub ptx i mlen = marker then found := true
  done ;
  if not !found then
    Alcotest.fail
      (Printf.sprintf
         "Expected PTX to contain %S but it did not.\nPTX:\n%s"
         marker
         ptx)

(** Substring check (no failure) — for asserting a marker's ABSENCE. *)
let contains ptx marker =
  let mlen = String.length marker in
  let plen = String.length ptx in
  let found = ref false in
  for i = 0 to plen - mlen do
    if String.sub ptx i mlen = marker then found := true
  done ;
  !found

let test_vector_add_markers () =
  let k = make_vector_add_kernel () in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx ".entry" ;
  assert_contains ptx ".param" ;
  assert_contains ptx "ld.global" ;
  assert_contains ptx "add.f32" ;
  assert_contains ptx "st.global" ;
  assert_contains ptx "ret"

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let base_kernel name params body funcs =
  {
    kern_name = name;
    kern_params = params;
    kern_locals = [];
    kern_body = body;
    kern_types = [];
    kern_variants = [];
    kern_funcs = funcs;
    kern_native_fn = None;
  }

(** Shared-array reduction shape: let%shared sdata = 256 lowers to SLet (sdata,
    EArrayCreate (f32, 256, Shared), ...). *)
let test_shared_array_markers () =
  let inp = make_var "inp" (TVec TFloat32) in
  let sdata = make_var "sdata" (TArray (TFloat32, Shared)) in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "thread_idx_x", []),
        SLet
          ( sdata,
            EArrayCreate (TFloat32, EConst (CInt32 256l), Shared),
            SSeq
              [
                SAssign
                  (LArrayElem ("sdata", EVar tid), EArrayRead ("inp", EVar tid));
                SBarrier;
                SAssign
                  (LArrayElem ("inp", EVar tid), EArrayRead ("sdata", EVar tid));
              ] ) )
  in
  let k =
    base_kernel
      "shared_red"
      [DParam (inp, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
      []
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx ".shared .align 4 .b32 sdata[256];" ;
  assert_contains ptx "mov.u32" ;
  assert_contains ptx "st.shared.f32" ;
  assert_contains ptx "bar.sync 0;" ;
  assert_contains ptx "ld.shared.f32"

(** Helper-function call is inlined: no .func, body appears in the entry. *)
let test_helper_inlining_markers () =
  let out = make_var "out" (TVec TFloat32) in
  let x = make_var "x" TFloat32 in
  let helper =
    {
      hf_name = "twice";
      hf_params = [x];
      hf_ret_type = TFloat32;
      hf_body = SReturn (EBinop (Add, EVar x, EVar x));
    }
  in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("out", EVar tid),
            EApp
              (EVar (make_var "twice" TFloat32), [EArrayRead ("out", EVar tid)])
          ) )
  in
  let k =
    base_kernel
      "use_helper"
      [DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
      [helper]
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "add.f32" ;
  (* inlined, not a PTX function call *)
  if
    String.length ptx >= 5
    &&
    let found = ref false in
    for i = 0 to String.length ptx - 5 do
      if String.sub ptx i 5 = ".func" then found := true
    done ;
    !found
  then Alcotest.fail "helper should be inlined, found .func directive"

(** Conversion intrinsic float (int -> f32) emits cvt. *)
let test_float_conversion_markers () =
  let out = make_var "out" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("out", EVar tid),
            EIntrinsic (["Sarek_stdlib"; "Gpu"], "float", [EVar tid]) ) )
  in
  let k =
    base_kernel
      "conv"
      [DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
      []
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "cvt.rn.f32.s32"

(** Recursive helper is rejected (falls back), not emitted as garbage. *)
let test_recursive_helper_rejected () =
  let out = make_var "out" (TVec TFloat32) in
  let x = make_var "x" TFloat32 in
  let helper =
    {
      hf_name = "loop_forever";
      hf_params = [x];
      hf_ret_type = TFloat32;
      hf_body =
        SReturn (EApp (EVar (make_var "loop_forever" TFloat32), [EVar x]));
    }
  in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("out", EVar tid),
            EApp
              ( EVar (make_var "loop_forever" TFloat32),
                [EArrayRead ("out", EVar tid)] ) ) )
  in
  let k =
    base_kernel
      "recursive"
      [DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
      [helper]
  in
  match Sarek_ir_ptx.generate k with
  | _ -> Alcotest.fail "recursive helper should raise Ptx_codegen_error"
  | exception Sarek_codegen.Sarek_ir_ptx_types.Ptx_codegen_error _ -> ()

(** Number of (possibly overlapping) occurrences of [sub] in [s]. *)
let count_substr s sub =
  let subl = String.length sub in
  let c = ref 0 in
  for i = 0 to String.length s - subl do
    if String.sub s i subl = sub then incr c
  done ;
  !c

(** Recursive int32 pow2-style helper whose body root carries
    [pragma ["sarek.inline 3"]]. *)
let make_pragma_pow2_helper n self =
  {
    hf_name = "pow2";
    hf_params = [n];
    hf_ret_type = TInt32;
    hf_body =
      SPragma
        ( ["sarek.inline 3"],
          SReturn
            (EIf
               ( EBinop (Le, EVar n, EConst (CInt32 0l)),
                 EConst (CInt32 1l),
                 EBinop
                   ( Mul,
                     EConst (CInt32 2l),
                     EApp (EVar self, [EBinop (Sub, EVar n, EConst (CInt32 1l))])
                   ) )) );
  }

(** Recursive helper WITH [pragma ["sarek.inline 3"]] is depth-unrolled inline:
    the body multiply appears once per unrolled level (>= 3 times) and no .func
    is emitted. *)
let test_recursive_helper_pragma_unrolled () =
  let out = make_var "out" (TVec TInt32) in
  let n = make_var "n" TInt32 in
  let self = make_var "pow2" TInt32 in
  let helper = make_pragma_pow2_helper n self in
  let body =
    SAssign (LArrayElem ("out", EConst (CInt32 0l)), EApp (EVar self, [EVar n]))
  in
  let k =
    base_kernel
      "pow2_unrolled"
      [
        DParam (out, Some {arr_elttype = TInt32; arr_memspace = Global});
        DParam (n, None);
      ]
      body
      [helper]
  in
  let ptx = Sarek_ir_ptx.generate k in
  let muls = count_substr ptx "mul.lo.u32" in
  if muls < 3 then
    Alcotest.fail
      (Printf.sprintf
         "expected >=3 unrolled multiplies, found %d:\n%s"
         muls
         ptx) ;
  if count_substr ptx ".func" > 0 then
    Alcotest.fail
      "pragma-unrolled helper must be inlined, found .func directive"

(** A guarded array read in an if-EXPRESSION must compile to real control flow
    (a predicated branch), never the eager evaluate-both + selp path, so the
    not-taken (possibly out-of-bounds) load is never executed. *)
let test_guarded_array_read_branches () =
  let out = make_var "out" (TVec TFloat32) in
  let a = make_var "a" (TVec TFloat32) in
  let n = make_var "n" TInt32 in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        (* out[tid] = if tid < n then a[tid] else 0.0 *)
        SAssign
          ( LArrayElem ("out", EVar tid),
            EIf
              ( EBinop (Lt, EVar tid, EVar n),
                EArrayRead ("a", EVar tid),
                EConst (CFloat32 0.0) ) ) )
  in
  let k =
    base_kernel
      "guarded_read"
      [
        DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (n, None);
      ]
      body
      []
  in
  let ptx = Sarek_ir_ptx.generate k in
  (* Branch-based lowering: the predicated bra guards the load, and the f32
     result is merged with mov.f32 — the eager path would instead select the
     result with selp.f32 (having already done the speculative load). Note a
     selp.u32 still appears for the comparison's 0/1 value; only the result
     selp.f32 signals the eager path. *)
  assert_contains ptx "@!" ;
  assert_contains ptx "bra " ;
  if
    let found = ref false in
    let m = "selp.f32" in
    for i = 0 to String.length ptx - String.length m do
      if String.sub ptx i (String.length m) = m then found := true
    done ;
    !found
  then
    Alcotest.fail
      "guarded array read must branch, not selp.f32 (speculative load risks \
       OOB)"

(** Native math: min/max/floor/ceil/rsqrt emit direct PTX ops. *)
let test_native_math_markers () =
  let out = make_var "out" (TVec TFloat32) in
  let a = make_var "a" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let f name args = EIntrinsic (["Sarek_stdlib"; "Gpu"], name, args) in
  let av = EArrayRead ("a", EVar tid) in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SSeq
          [
            SAssign (LArrayElem ("out", EVar tid), f "min" [av; av]);
            SAssign (LArrayElem ("out", EVar tid), f "max" [av; av]);
            SAssign (LArrayElem ("out", EVar tid), f "floor" [av]);
            SAssign (LArrayElem ("out", EVar tid), f "ceil" [av]);
            SAssign (LArrayElem ("out", EVar tid), f "rsqrt" [av]);
          ] )
  in
  let mk v = DParam (v, Some {arr_elttype = TFloat32; arr_memspace = Global}) in
  let k = base_kernel "native_math" [mk out; mk a] body [] in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "min.f32" ;
  assert_contains ptx "max.f32" ;
  assert_contains ptx "cvt.rmi.f32.f32" ;
  assert_contains ptx "cvt.rpi.f32.f32" ;
  assert_contains ptx "rsqrt.approx.f32"

(** Native f64 math: sqrt/abs_float/copysign/hypot/rsqrt on Float64 operands
    emit .f64-suffixed ops only (never an .f32 suffix on an %fd register). *)
let test_f64_native_math_markers () =
  let out = make_var "out" (TVec TFloat64) in
  let a = make_var "a" (TVec TFloat64) in
  let tid = make_var "tid" TInt32 in
  let f name args = EIntrinsic (["Float64"], name, args) in
  let av = EArrayRead ("a", EVar tid) in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SSeq
          [
            SAssign (LArrayElem ("out", EVar tid), f "sqrt" [av]);
            SAssign (LArrayElem ("out", EVar tid), f "abs_float" [av]);
            SAssign (LArrayElem ("out", EVar tid), f "copysign" [av; av]);
            SAssign (LArrayElem ("out", EVar tid), f "hypot" [av; av]);
            SAssign (LArrayElem ("out", EVar tid), f "rsqrt" [av]);
          ] )
  in
  let mk v = DParam (v, Some {arr_elttype = TFloat64; arr_memspace = Global}) in
  let k = base_kernel "f64_native_math" [mk out; mk a] body [] in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "sqrt.rn.f64" ;
  assert_contains ptx "abs.f64" ;
  assert_contains ptx "copysign.f64" ;
  assert_contains ptx "fma.rn.f64" ;
  assert_contains ptx "rcp.rn.f64"

(** f32 transcendental compositions: tan/pow/log10/tanh lower to sin/cos/lg2/ex2
    .approx building blocks plus div.approx.f32. *)
let test_f32_transcendental_markers () =
  let out = make_var "out" (TVec TFloat32) in
  let a = make_var "a" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let f name args = EIntrinsic (["Sarek_stdlib"; "Float32"], name, args) in
  let av = EArrayRead ("a", EVar tid) in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SSeq
          [
            SAssign (LArrayElem ("out", EVar tid), f "tan" [av]);
            SAssign (LArrayElem ("out", EVar tid), f "pow" [av; av]);
            SAssign (LArrayElem ("out", EVar tid), f "log10" [av]);
            SAssign (LArrayElem ("out", EVar tid), f "tanh" [av]);
          ] )
  in
  let mk v = DParam (v, Some {arr_elttype = TFloat32; arr_memspace = Global}) in
  let k = base_kernel "f32_transcendentals" [mk out; mk a] body [] in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "sin.approx.f32" ;
  assert_contains ptx "cos.approx.f32" ;
  assert_contains ptx "lg2.approx.f32" ;
  assert_contains ptx "ex2.approx.f32" ;
  assert_contains ptx "div.approx.f32"

(** Extended atomics emit atom.{shared,global}.<op>.<ty>; sub lowers to
    neg + add. *)
let test_atomic_family_markers () =
  let hist = make_var "hist" (TVec TInt32) in
  let facc = make_var "facc" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let a name args = EIntrinsic (["Sarek_stdlib"; "Gpu"], name, args) in
  let one = EConst (CInt32 1l) in
  let fone = EConst (CFloat32 1.0) in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SSeq
          [
            SExpr (a "atomic_min_int32" [EVar hist; EVar tid; one]);
            SExpr (a "atomic_max_int32" [EVar hist; EVar tid; one]);
            SExpr (a "atomic_and_int32" [EVar hist; EVar tid; one]);
            SExpr (a "atomic_or_int32" [EVar hist; EVar tid; one]);
            SExpr (a "atomic_xor_int32" [EVar hist; EVar tid; one]);
            SExpr (a "atomic_exch_int32" [EVar hist; EVar tid; one]);
            SExpr (a "atomic_sub_int32" [EVar hist; EVar tid; one]);
            SExpr (a "atomic_add_float32" [EVar facc; EVar tid; fone]);
          ] )
  in
  let k =
    base_kernel
      "atomics"
      [
        DParam (hist, Some {arr_elttype = TInt32; arr_memspace = Global});
        DParam (facc, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ]
      body
      []
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "atom.global.min.s32" ;
  assert_contains ptx "atom.global.max.s32" ;
  assert_contains ptx "atom.global.and.b32" ;
  assert_contains ptx "atom.global.or.b32" ;
  assert_contains ptx "atom.global.xor.b32" ;
  assert_contains ptx "atom.global.exch.b32" ;
  assert_contains ptx "atom.global.add.f32" ;
  (* sub has no PTX atom op → negate then atom.add *)
  assert_contains ptx "neg.s32" ;
  assert_contains ptx "atom.global.add.s32"

(** Extended atomics round 2: cas.b32/b64, wrapping inc/dec.u32 (limit
    0xffffffff = plain ±1 mod 2^32), add.u64/f64, exch.b64; 64-bit elements use
    an 8-byte stride (shl …, 3), and a shared cas addresses in 32-bit. *)
let test_atomic_cas_incdec_wide_markers () =
  let hist = make_var "hist" (TVec TInt32) in
  let lacc = make_var "lacc" (TVec TInt64) in
  let dacc = make_var "dacc" (TVec TFloat64) in
  let slock = make_var "slock" (TArray (TInt32, Shared)) in
  let tid = make_var "tid" TInt32 in
  let a name args = EIntrinsic (["Sarek_stdlib"; "Gpu"], name, args) in
  let one = EConst (CInt32 1l) in
  let zero = EConst (CInt32 0l) in
  let lone = EConst (CInt64 1L) in
  let lzero = EConst (CInt64 0L) in
  let done_ = EConst (CFloat64 1.0) in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( slock,
            EArrayCreate (TInt32, EConst (CInt32 32l), Shared),
            SSeq
              [
                SExpr (a "atomic_cas_int32" [EVar hist; EVar tid; zero; one]);
                SExpr (a "atomic_cas_int64" [EVar lacc; EVar tid; lzero; lone]);
                SExpr (a "atomic_cas_int32" [EVar slock; EVar tid; zero; one]);
                SExpr (a "atomic_inc_int32" [EVar hist; EVar tid]);
                SExpr (a "atomic_dec_int32" [EVar hist; EVar tid]);
                SExpr (a "atomic_add_int64" [EVar lacc; EVar tid; lone]);
                SExpr (a "atomic_add_float64" [EVar dacc; EVar tid; done_]);
                SExpr (a "atomic_exch_int64" [EVar lacc; EVar tid; lone]);
              ] ) )
  in
  let k =
    base_kernel
      "atomics_ext"
      [
        DParam (hist, Some {arr_elttype = TInt32; arr_memspace = Global});
        DParam (lacc, Some {arr_elttype = TInt64; arr_memspace = Global});
        DParam (dacc, Some {arr_elttype = TFloat64; arr_memspace = Global});
      ]
      body
      []
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "atom.global.cas.b32" ;
  assert_contains ptx "atom.global.cas.b64" ;
  assert_contains ptx "atom.shared.cas.b32" ;
  assert_contains ptx "atom.global.inc.u32" ;
  assert_contains ptx "atom.global.dec.u32" ;
  (* inc/dec wrap at the limit operand; 0xffffffff makes them plain ±1 *)
  assert_contains ptx "0xffffffff" ;
  assert_contains ptx "atom.global.add.u64" ;
  assert_contains ptx "atom.global.add.f64" ;
  assert_contains ptx "atom.global.exch.b64" ;
  (* 8-byte addressing stride for the 64-bit forms *)
  assert_contains ptx ", 3;"

(** Float Mod lowers to C fmod: x − trunc(x/y)·y via rn-rounded div, cvt.rzi
    trunc, neg and a single fma — for both f32 and f64. *)
let test_float_mod_fmod_markers () =
  let fa = make_var "fa" (TVec TFloat32) in
  let da = make_var "da" (TVec TFloat64) in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SSeq
          [
            SAssign
              ( LArrayElem ("fa", EVar tid),
                EBinop (Mod, EArrayRead ("fa", EVar tid), EConst (CFloat32 3.0))
              );
            SAssign
              ( LArrayElem ("da", EVar tid),
                EBinop (Mod, EArrayRead ("da", EVar tid), EConst (CFloat64 3.0))
              );
          ] )
  in
  let k =
    base_kernel
      "float_fmod"
      [
        DParam (fa, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (da, Some {arr_elttype = TFloat64; arr_memspace = Global});
      ]
      body
      []
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "div.rn.f32" ;
  assert_contains ptx "cvt.rzi.f32.f32" ;
  assert_contains ptx "neg.f32" ;
  assert_contains ptx "fma.rn.f32" ;
  assert_contains ptx "div.rn.f64" ;
  assert_contains ptx "cvt.rzi.f64.f64" ;
  assert_contains ptx "neg.f64" ;
  assert_contains ptx "fma.rn.f64"

(** Integer Div/Mod are SIGNED (audit finding H1): Sarek int32/int64 are
    signed everywhere (interpreter uses Int32.div/Int64.div, C backends emit
    / and % on signed types), so PTX must emit div.s32/s64 and rem.s32/s64.
    The old div.u32/u64, rem.u32/u64 silently returned garbage for negative
    operands ((-7)/2 = 2147483644). *)
let test_int_div_rem_signed_markers () =
  let ia = make_var "ia" (TVec TInt32) in
  let la = make_var "la" (TVec TInt64) in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SSeq
          [
            SAssign
              ( LArrayElem ("ia", EVar tid),
                EBinop (Div, EArrayRead ("ia", EVar tid), EConst (CInt32 (-2l)))
              );
            SAssign
              ( LArrayElem ("ia", EVar tid),
                EBinop (Mod, EArrayRead ("ia", EVar tid), EConst (CInt32 3l)) );
            SAssign
              ( LArrayElem ("la", EVar tid),
                EBinop
                  (Div, EArrayRead ("la", EVar tid), EConst (CInt64 (-2L))) );
            SAssign
              ( LArrayElem ("la", EVar tid),
                EBinop (Mod, EArrayRead ("la", EVar tid), EConst (CInt64 3L))
              );
          ] )
  in
  let k =
    base_kernel
      "int_div_rem"
      [
        DParam (ia, Some {arr_elttype = TInt32; arr_memspace = Global});
        DParam (la, Some {arr_elttype = TInt64; arr_memspace = Global});
      ]
      body
      []
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "div.s32" ;
  assert_contains ptx "rem.s32" ;
  assert_contains ptx "div.s64" ;
  assert_contains ptx "rem.s64" ;
  if contains ptx "div.u32" || contains ptx "div.u64" then
    Alcotest.fail "unsigned div emitted for signed Sarek int" ;
  if contains ptx "rem.u32" || contains ptx "rem.u64" then
    Alcotest.fail "unsigned rem emitted for signed Sarek int"

(** ECast scalar-matrix coverage: bool casts normalize to a u32 0/1 via
    setp+selp (float sources use unordered neu so NaN -> 1, matching C); i32 ->
    i64 sign-extends (cvt.s64.s32); i64 -> i32 truncates (cvt.u32.u64). *)
let test_cast_matrix_markers () =
  let out = make_var "out" (TVec TInt32) in
  let tid = make_var "tid" TInt32 in
  let store i e = SAssign (LArrayElem ("out", i), e) in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SSeq
          [
            store (EVar tid) (ECast (TBool, EVar tid));
            store (EVar tid) (ECast (TBool, EConst (CFloat32 1.0)));
            store (EVar tid) (ECast (TBool, EConst (CFloat64 1.0)));
            store (EVar tid) (ECast (TBool, EConst (CInt64 1L)));
            store (EVar tid) (ECast (TInt32, ECast (TInt64, EVar tid)));
          ] )
  in
  let k =
    base_kernel
      "cast_matrix"
      [DParam (out, Some {arr_elttype = TInt32; arr_memspace = Global})]
      body
      []
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "setp.ne.u32" ;
  assert_contains ptx "setp.neu.f32" ;
  assert_contains ptx "setp.neu.f64" ;
  assert_contains ptx "setp.ne.s64" ;
  assert_contains ptx "selp.u32" ;
  (* widen sign-extends, narrow truncates to the low 32 bits *)
  assert_contains ptx "cvt.s64.s32" ;
  assert_contains ptx "cvt.u32.u64"

(** ECast to unit is semantically meaningless and must be rejected. *)
let test_cast_to_unit_rejected () =
  let out = make_var "out" (TVec TInt32) in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SAssign (LArrayElem ("out", EVar tid), ECast (TUnit, EVar tid)) )
  in
  let k =
    base_kernel
      "cast_unit"
      [DParam (out, Some {arr_elttype = TInt32; arr_memspace = Global})]
      body
      []
  in
  match Sarek_ir_ptx.generate k with
  | _ -> Alcotest.fail "ECast to unit should be rejected"
  | exception Sarek_codegen.Sarek_ir_ptx_types.Ptx_codegen_error _ -> ()

(** A 32-bit value into a 64-bit atom form is invalid PTX; the emitter must
    reject it (width discipline), never emit the mismatched suffix. *)
let test_atomic_width_mismatch_rejected () =
  let lacc = make_var "lacc" (TVec TInt64) in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SExpr
          (EIntrinsic
             ( ["Sarek_stdlib"; "Gpu"],
               "atomic_add_int64",
               [EVar lacc; EVar tid; EConst (CInt32 1l)] )) )
  in
  let k =
    base_kernel
      "atomic_mismatch"
      [DParam (lacc, Some {arr_elttype = TInt64; arr_memspace = Global})]
      body
      []
  in
  match Sarek_ir_ptx.generate k with
  | _ -> Alcotest.fail "int32 value into atom.add.u64 should be rejected"
  | exception Sarek_codegen.Sarek_ir_ptx_types.Ptx_codegen_error _ -> ()

(** Check that [ptx] does NOT contain [marker]. *)
let assert_absent ptx marker ~why =
  let mlen = String.length marker in
  let found = ref false in
  for i = 0 to String.length ptx - mlen do
    if String.sub ptx i mlen = marker then found := true
  done ;
  if !found then
    Alcotest.fail
      (Printf.sprintf
         "Expected PTX to NOT contain %S (%s).\nPTX:\n%s"
         marker
         why
         ptx)

let point_ty = TRecord ("point", [("x", TFloat32); ("y", TFloat32)])

(** Local record construct + field reads live entirely in registers: the record
    itself must generate no global-memory traffic (SROA, FR-020). *)
let test_record_sroa_markers () =
  let dst = make_var "dst" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let p = make_var "p" point_ty in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( p,
            ERecord
              ( "point",
                [("x", EConst (CFloat32 1.5)); ("y", EConst (CFloat32 2.5))] ),
            SAssign
              ( LArrayElem ("dst", EVar tid),
                EBinop
                  (Add, ERecordField (EVar p, "x"), ERecordField (EVar p, "y"))
              ) ) )
  in
  let k =
    base_kernel
      "record_sroa"
      [DParam (dst, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
      []
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "add.f32" ;
  assert_contains ptx "st.global.f32" ;
  assert_absent
    ptx
    "ld.global"
    ~why:"local record must be SROA registers, never loaded from memory"

(** Nested record construct + two-level projection stays in registers. *)
let test_nested_record_sroa_markers () =
  let dst = make_var "dst" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let outer_ty = TRecord ("outer", [("inner", point_ty); ("c", TFloat32)]) in
  let q = make_var "q" outer_ty in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( q,
            ERecord
              ( "outer",
                [
                  ( "inner",
                    ERecord
                      ( "point",
                        [
                          ("x", EConst (CFloat32 1.0));
                          ("y", EConst (CFloat32 2.0));
                        ] ) );
                  ("c", EConst (CFloat32 3.0));
                ] ),
            SAssign
              ( LArrayElem ("dst", EVar tid),
                EBinop
                  ( Add,
                    ERecordField (ERecordField (EVar q, "inner"), "y"),
                    ERecordField (EVar q, "c") ) ) ) )
  in
  let k =
    base_kernel
      "nested_record_sroa"
      [DParam (dst, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
      []
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "add.f32" ;
  assert_contains ptx "st.global.f32" ;
  assert_absent
    ptx
    "ld.global"
    ~why:"nested local record must be SROA registers"

(** Mutable local record: field assignment is a register mov into the leaf. *)
let test_record_field_mutation_markers () =
  let dst = make_var "dst" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let p = {(make_var "p" point_ty) with var_mutable = true} in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLetMut
          ( p,
            ERecord
              ( "point",
                [("x", EConst (CFloat32 1.0)); ("y", EConst (CFloat32 2.0))] ),
            SSeq
              [
                SAssign (LRecordField (LVar p, "x"), EConst (CFloat32 4.0));
                SAssign
                  ( LArrayElem ("dst", EVar tid),
                    EBinop
                      ( Add,
                        ERecordField (EVar p, "x"),
                        ERecordField (EVar p, "y") ) );
              ] ) )
  in
  let k =
    base_kernel
      "record_field_mut"
      [DParam (dst, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
      []
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "mov.f32" ;
  assert_contains ptx "add.f32" ;
  assert_absent
    ptx
    "ld.global"
    ~why:"mutable local record field update must be a register mov"

(** Helper taking AND returning a record: aggregate args are leaf-wise copied
    into the callee, the aggregate return is pre-allocated and filled by SReturn
    — all inlined, all in registers (FR-023). *)
let test_helper_record_arg_ret_markers () =
  let dst = make_var "dst" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let p = make_var "p" point_ty in
  let kf = make_var "k" TFloat32 in
  let helper =
    {
      hf_name = "scale";
      hf_params = [p; kf];
      hf_ret_type = point_ty;
      hf_body =
        SReturn
          (ERecord
             ( "point",
               [
                 ("x", EBinop (Mul, ERecordField (EVar p, "x"), EVar kf));
                 ("y", EBinop (Mul, ERecordField (EVar p, "y"), EVar kf));
               ] ));
    }
  in
  let q = make_var "q" point_ty in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( q,
            EApp
              ( EVar (make_var "scale" point_ty),
                [
                  ERecord
                    ( "point",
                      [
                        ("x", EConst (CFloat32 1.5));
                        ("y", EConst (CFloat32 2.5));
                      ] );
                  EConst (CFloat32 2.0);
                ] ),
            SAssign
              ( LArrayElem ("dst", EVar tid),
                EBinop
                  (Add, ERecordField (EVar q, "x"), ERecordField (EVar q, "y"))
              ) ) )
  in
  let k =
    base_kernel
      "record_helper"
      [DParam (dst, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
      [helper]
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "mul.f32" ;
  assert_contains ptx "add.f32" ;
  assert_contains ptx "st.global.f32" ;
  assert_absent ptx ".func" ~why:"helper must be inlined, not a PTX function" ;
  assert_absent
    ptx
    "ld.global"
    ~why:"record argument and return must stay in SROA registers"

let color_decl = [("Red", []); ("Value", [TFloat32])]

let color_ty = TVariant ("color", color_decl)

(** [base_kernel] with variant declarations registered ([kern_variants]),
    required by EVariant construction. *)
let variant_kernel name params body variants =
  {(base_kernel name params body []) with kern_variants = variants}

(** Variant construct (nullary + 1-arg) and SMatch: tag is a mov of the
    declaration-index constant, dispatch is a setp.eq branch chain, and
    everything stays in registers. *)
let test_variant_construct_smatch_markers () =
  let dst = make_var "dst" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let c = make_var "c" color_ty in
  let v = make_var "v" TFloat32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( c,
            EVariant ("color", "Value", [EConst (CFloat32 3.5)]),
            SMatch
              ( EVar c,
                [
                  ( PConstr ("Red", []),
                    SAssign (LArrayElem ("dst", EVar tid), EConst (CFloat32 0.0))
                  );
                  ( PConstr ("Value", ["v"]),
                    SAssign
                      ( LArrayElem ("dst", EVar tid),
                        EBinop (Add, EVar v, EConst (CFloat32 1.0)) ) );
                ] ) ) )
  in
  let k =
    variant_kernel
      "variant_smatch"
      [DParam (dst, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
      [("color", color_decl)]
  in
  let ptx = Sarek_ir_ptx.generate k in
  (* Value's declaration index is 1. *)
  assert_contains ptx "mov.u32" ;
  assert_contains ptx "setp.eq.u32" ;
  assert_contains ptx "add.f32" ;
  assert_contains ptx "st.global.f32" ;
  assert_absent
    ptx
    "ld.global"
    ~why:"local variant must be SROA registers, never loaded from memory"

(** EMatch in value position is ALWAYS branch-based: branch labels present, no
    selp on the f32 match result (FR-022, AC-4). *)
let test_ematch_value_markers () =
  let dst = make_var "dst" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let c = make_var "c" color_ty in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( c,
            EVariant ("color", "Value", [EConst (CFloat32 2.0)]),
            SAssign
              ( LArrayElem ("dst", EVar tid),
                EMatch
                  ( EVar c,
                    [
                      (PConstr ("Red", []), EConst (CFloat32 0.0));
                      (PConstr ("Value", ["v"]), EVar (make_var "v" TFloat32));
                    ] ) ) ) )
  in
  let k =
    variant_kernel
      "ematch_value"
      [DParam (dst, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
      [("color", color_decl)]
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "setp.eq.u32" ;
  assert_contains ptx "bra " ;
  assert_contains ptx "st.global.f32" ;
  assert_absent
    ptx
    "selp.f32"
    ~why:"EMatch result must be merged by branch chain, never selp"

(** Three-constructor variant with mixed payload arities matches through a
    setp.eq chain (two tests + unconditional last arm). *)
let test_three_ctor_variant_markers () =
  let shape_decl =
    [("Circle", [TFloat32]); ("Square", [TFloat32]); ("Point", [])]
  in
  let dst = make_var "dst" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let s = make_var "s" (TVariant ("shape", shape_decl)) in
  let r = make_var "r" TFloat32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( s,
            EVariant ("shape", "Square", [EConst (CFloat32 4.0)]),
            SMatch
              ( EVar s,
                [
                  ( PConstr ("Circle", ["r"]),
                    SAssign
                      ( LArrayElem ("dst", EVar tid),
                        EBinop (Mul, EVar r, EVar r) ) );
                  ( PConstr ("Square", ["r"]),
                    SAssign
                      ( LArrayElem ("dst", EVar tid),
                        EBinop (Add, EVar r, EVar r) ) );
                  ( PConstr ("Point", []),
                    SAssign (LArrayElem ("dst", EVar tid), EConst (CFloat32 0.0))
                  );
                ] ) ) )
  in
  let k =
    variant_kernel
      "three_ctors"
      [DParam (dst, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
      [("shape", shape_decl)]
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "setp.eq.u32" ;
  assert_contains ptx "mul.f32" ;
  assert_contains ptx "add.f32" ;
  assert_absent ptx "ld.global" ~why:"3-ctor local variant stays in registers"

(** Multi-argument payload: construct with two args and bind both in the
    matching arm. *)
let test_multiarg_payload_markers () =
  let pair_decl = [("Pair", [TFloat32; TFloat32])] in
  let dst = make_var "dst" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let pv = make_var "pv" (TVariant ("pair_v", pair_decl)) in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( pv,
            EVariant
              ( "pair_v",
                "Pair",
                [EConst (CFloat32 1.25); EConst (CFloat32 2.75)] ),
            SMatch
              ( EVar pv,
                [
                  ( PConstr ("Pair", ["a"; "b"]),
                    SAssign
                      ( LArrayElem ("dst", EVar tid),
                        EBinop
                          ( Add,
                            EVar (make_var "a" TFloat32),
                            EVar (make_var "b" TFloat32) ) ) );
                ] ) ) )
  in
  let k =
    variant_kernel
      "multiarg_payload"
      [DParam (dst, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
      [("pair_v", pair_decl)]
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "add.f32" ;
  assert_contains ptx "st.global.f32" ;
  assert_absent
    ptx
    "ld.global"
    ~why:"multi-arg variant payload stays in registers"

(** Nullary-only variant (pure enum): tag register only, no payload slots. *)
let test_nullary_only_variant_markers () =
  let light_decl = [("Stop", []); ("Slow", []); ("Go", [])] in
  let dst = make_var "dst" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let l = make_var "l" (TVariant ("light", light_decl)) in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( l,
            EVariant ("light", "Go", []),
            SMatch
              ( EVar l,
                [
                  ( PConstr ("Stop", []),
                    SAssign (LArrayElem ("dst", EVar tid), EConst (CFloat32 0.0))
                  );
                  ( PConstr ("Slow", []),
                    SAssign (LArrayElem ("dst", EVar tid), EConst (CFloat32 1.0))
                  );
                  ( PConstr ("Go", []),
                    SAssign (LArrayElem ("dst", EVar tid), EConst (CFloat32 2.0))
                  );
                ] ) ) )
  in
  let k =
    variant_kernel
      "nullary_only"
      [DParam (dst, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
      [("light", light_decl)]
  in
  let ptx = Sarek_ir_ptx.generate k in
  (* Go's declaration index is 2. *)
  assert_contains ptx "mov.u32" ;
  assert_contains ptx "setp.eq.u32" ;
  assert_contains ptx "st.global.f32"

(** A variant match covering neither all constructors nor a wildcard is rejected
    with a precise error naming the type (C-9). *)
let test_nonexhaustive_match_rejected () =
  let dst = make_var "dst" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let c = make_var "c" color_ty in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( c,
            EVariant ("color", "Red", []),
            SMatch
              ( EVar c,
                [
                  ( PConstr ("Red", []),
                    SAssign (LArrayElem ("dst", EVar tid), EConst (CFloat32 0.0))
                  );
                ] ) ) )
  in
  let k =
    variant_kernel
      "nonexhaustive"
      [DParam (dst, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
      [("color", color_decl)]
  in
  match Sarek_ir_ptx.generate k with
  | _ -> Alcotest.fail "non-exhaustive match should raise Ptx_codegen_error"
  | exception Sarek_codegen.Sarek_ir_ptx_types.Ptx_codegen_error msg ->
      let expected = "non-exhaustive match on 'color'" in
      let found = ref false in
      let mlen = String.length expected in
      for i = 0 to String.length msg - mlen do
        if String.sub msg i mlen = expected then found := true
      done ;
      if not !found then
        Alcotest.fail
          (Printf.sprintf "error should contain %S, got: %s" expected msg)

(** Tuple construct + destructure: anonymous register aggregate with positional
    slots, no memory traffic (FR-024). *)
let test_tuple_construct_destructure_markers () =
  let dst = make_var "dst" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let t = make_var "t" TUnit in
  (* var_type is irrelevant to the ETuple binding shape *)
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( t,
            ETuple [EConst (CFloat32 1.5); EConst (CFloat32 2.5)],
            SMatch
              ( EVar t,
                [
                  ( PConstr ("tuple", ["u"; "v"]),
                    SAssign
                      ( LArrayElem ("dst", EVar tid),
                        EBinop
                          ( Add,
                            EVar (make_var "u" TFloat32),
                            EVar (make_var "v" TFloat32) ) ) );
                ] ) ) )
  in
  let k =
    base_kernel
      "tuple_destructure"
      [DParam (dst, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
      []
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "add.f32" ;
  assert_contains ptx "st.global.f32" ;
  assert_absent
    ptx
    "ld.global"
    ~why:"tuple components must stay in registers (anonymous SROA aggregate)"

(** 2-arg variant payload roundtrip through value-position EMatch: both payload
    registers flow construct -> match arm -> result without memory. *)
let test_variant_payload_roundtrip_markers () =
  let pair_decl = [("Pair", [TFloat32; TFloat32])] in
  let dst = make_var "dst" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let pv = make_var "pv" (TVariant ("pair_v", pair_decl)) in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( pv,
            EVariant
              ("pair_v", "Pair", [EConst (CFloat32 3.0); EConst (CFloat32 5.0)]),
            SAssign
              ( LArrayElem ("dst", EVar tid),
                EMatch
                  ( EVar pv,
                    [
                      ( PConstr ("Pair", ["a"; "b"]),
                        EBinop
                          ( Mul,
                            EVar (make_var "a" TFloat32),
                            EVar (make_var "b" TFloat32) ) );
                    ] ) ) ) )
  in
  let k =
    variant_kernel
      "payload_roundtrip"
      [DParam (dst, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
      [("pair_v", pair_decl)]
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "mul.f32" ;
  assert_contains ptx "st.global.f32" ;
  assert_absent
    ptx
    "ld.global"
    ~why:"2-arg payload roundtrip must stay in registers" ;
  assert_absent
    ptx
    "selp.f32"
    ~why:"EMatch result must be merged by branch chain, never selp"

(** Storing a tuple into a global vector element is rejected with an error
    naming the construct and a workaround (FR-024). *)
let test_tuple_into_vector_rejected () =
  let dst = make_var "dst" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("dst", EVar tid),
            ETuple [EConst (CFloat32 1.0); EConst (CFloat32 2.0)] ) )
  in
  let k =
    base_kernel
      "tuple_store"
      [DParam (dst, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
      []
  in
  match Sarek_ir_ptx.generate k with
  | _ -> Alcotest.fail "tuple store into vector should raise Ptx_codegen_error"
  | exception Sarek_codegen.Sarek_ir_ptx_types.Ptx_codegen_error msg ->
      let expected = "tuple value used in a scalar context" in
      let found = ref false in
      let mlen = String.length expected in
      for i = 0 to String.length msg - mlen do
        if String.sub msg i mlen = expected then found := true
      done ;
      if not !found then
        Alcotest.fail
          (Printf.sprintf "error should contain %S, got: %s" expected msg)

(** Index of the first occurrence of [marker] in [ptx], or [None]. *)
let find_first ptx marker =
  let mlen = String.length marker in
  let rec go i =
    if i > String.length ptx - mlen then None
    else if String.sub ptx i mlen = marker then Some i
    else go (i + 1)
  in
  go 0

(** Index of the last occurrence of [marker] in [ptx], or [None]. *)
let find_last ptx marker =
  let mlen = String.length marker in
  let rec go i best =
    if i > String.length ptx - mlen then best
    else if String.sub ptx i mlen = marker then go (i + 1) (Some i)
    else go (i + 1) best
  in
  go 0 None

let point_arr_info =
  Some {arr_elttype = point_ty; arr_memspace = Sarek_ir_types.Global}

(** Field-wise element access
    ([dst.(tid) <- {x = src.(tid).x + 1; y = src.(tid).y}]): stride via
    mul.wide.u32 (FR-010), field loads/stores at immediate offsets (+4 for y —
    FR-011), one typed ld/st per field. *)
let test_record_elem_field_rw_markers () =
  let src = make_var "src" (TVec point_ty) in
  let dst = make_var "dst" (TVec point_ty) in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("dst", EVar tid),
            ERecord
              ( "point",
                [
                  ( "x",
                    EBinop
                      ( Add,
                        ERecordField (EArrayRead ("src", EVar tid), "x"),
                        EConst (CFloat32 1.0) ) );
                  ("y", ERecordField (EArrayRead ("src", EVar tid), "y"));
                ] ) ) )
  in
  let k =
    base_kernel
      "record_elem_rw"
      [DParam (src, point_arr_info); DParam (dst, point_arr_info)]
      body
      []
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "mul.wide.u32" ;
  assert_contains ptx "ld.global.f32" ;
  assert_contains ptx "+4]" ;
  assert_contains ptx "st.global.f32" ;
  assert_absent
    ptx
    "shl.b64"
    ~why:"aggregate elements use byte-stride multiplication, not shifts"

(** Whole-element copy of a 12-byte point3d ([dst.(o) <- src.(i)]): stride-12
    mul.wide.u32 (AC-3, non-pow2), +8 field offset, and EVERY ld.global
    preceding the first st.global (EC-1 aliasing safety). *)
let test_point3d_whole_copy_markers () =
  let p3_ty =
    TRecord ("point3d", [("x", TFloat32); ("y", TFloat32); ("z", TFloat32)])
  in
  let info = Some {arr_elttype = p3_ty; arr_memspace = Sarek_ir_types.Global} in
  let src = make_var "src" (TVec p3_ty) in
  let dst = make_var "dst" (TVec p3_ty) in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SAssign (LArrayElem ("dst", EVar tid), EArrayRead ("src", EVar tid)) )
  in
  let k =
    base_kernel "p3d_copy" [DParam (src, info); DParam (dst, info)] body []
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx ", 12;" ;
  (* mul.wide.u32 …, 12 *)
  assert_contains ptx "+8]" ;
  match (find_last ptx "ld.global", find_first ptx "st.global") with
  | Some last_ld, Some first_st ->
      if last_ld > first_st then
        Alcotest.fail
          "whole-element copy must emit ALL loads before ANY store (EC-1)"
  | _ -> Alcotest.fail "expected both ld.global and st.global in copy kernel"

(** Single-field element write ([dst.(tid).y <- 4.0]) is ONE typed st at the
    field offset — no other global traffic. *)
let test_record_elem_field_store_markers () =
  let dst = make_var "dst" (TVec point_ty) in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LRecordField (LArrayElem ("dst", EVar tid), "y"),
            EConst (CFloat32 4.0) ) )
  in
  let k = base_kernel "field_store" [DParam (dst, point_arr_info)] body [] in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "mul.wide.u32" ;
  assert_contains ptx "st.global.f32 [" ;
  assert_contains ptx "+4]" ;
  assert_absent
    ptx
    "ld.global"
    ~why:"a single-field store must not load the element"

(** Variant vector element roundtrip: read loads the tag (ld.global.u32 at the
    element base) + payload slots (FR-013: all constructors' slots, never past
    the element); write stores the tag then only the active constructor's
    payload via a tag branch chain. *)
let test_variant_elem_roundtrip_markers () =
  let cinfo =
    Some {arr_elttype = color_ty; arr_memspace = Sarek_ir_types.Global}
  in
  let src = make_var "src" (TVec color_ty) in
  let vdst = make_var "vdst" (TVec color_ty) in
  let dst = make_var "dst" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let c = make_var "c" color_ty in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( c,
            EArrayRead ("src", EVar tid),
            SSeq
              [
                SAssign
                  ( LArrayElem ("dst", EVar tid),
                    EMatch
                      ( EVar c,
                        [
                          (PConstr ("Red", []), EConst (CFloat32 0.0));
                          ( PConstr ("Value", ["v"]),
                            EVar (make_var "v" TFloat32) );
                        ] ) );
                SAssign
                  ( LArrayElem ("vdst", EVar tid),
                    EVariant ("color", "Value", [EConst (CFloat32 3.5)]) );
              ] ) )
  in
  let k =
    variant_kernel
      "variant_elem"
      [
        DParam (src, cinfo);
        DParam (vdst, cinfo);
        DParam (dst, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ]
      body
      [("color", color_decl)]
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "ld.global.u32" ;
  (* tag load *)
  assert_contains ptx "st.global.u32" ;
  (* tag store *)
  assert_contains ptx "setp.eq.u32" ;
  (* match dispatch *)
  assert_contains ptx "setp.ne.u32" ;
  (* store branch chain guard *)
  assert_contains ptx "st.global.f32"

(** A bare record parameter (DParam with no arr_info) is rejected with the C-17
    message naming the param and both workarounds; a TVec of the same type is
    accepted (EC-11 discrimination — proven by the tests above). *)
let test_bare_record_param_rejected () =
  let p = make_var "p" point_ty in
  let dst = make_var "dst" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SAssign (LArrayElem ("dst", EVar tid), ERecordField (EVar p, "x")) )
  in
  let k =
    base_kernel
      "bare_record_param"
      [
        DParam (p, None);
        DParam (dst, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ]
      body
      []
  in
  match Sarek_ir_ptx.generate k with
  | _ -> Alcotest.fail "bare record param should raise Ptx_codegen_error"
  | exception Sarek_codegen.Sarek_ir_ptx_types.Ptx_codegen_error msg ->
      let check expected =
        match find_first msg expected with
        | Some _ -> ()
        | None ->
            Alcotest.fail
              (Printf.sprintf "error should contain %S, got: %s" expected msg)
      in
      check "parameter 'p'" ;
      check
        "pass fields as separate scalar params or use a 1-element 'point' \
         vector"

(** L8 (sanctioned rejection->acceptance conversion): a vector of
    mixed-alignment records {a:int32; b:float64} is now laid out with the
    aligned host ABI (b at the 8-aligned offset 8, element stride 16), so the
    kernel COMPILES and emits a natural [ld.global.f64] for the f64 field. This
    previously raised Ptx_codegen_error (AC-5 / FR-004, now superseded). *)
let test_mixed_align_record_param_accepted () =
  let mixed_ty = TRecord ("mixed", [("a", TInt32); ("b", TFloat64)]) in
  let src = make_var "src" (TVec mixed_ty) in
  let dst = make_var "dst" (TVec TFloat64) in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("dst", EVar tid),
            ERecordField (EArrayRead ("src", EVar tid), "b") ) )
  in
  let k =
    base_kernel
      "mixed_align_record"
      [
        DParam (src, Some {arr_elttype = mixed_ty; arr_memspace = Global});
        DParam (dst, Some {arr_elttype = TFloat64; arr_memspace = Global});
      ]
      body
      []
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx ".entry" ;
  (* Aligned ABI: 16-byte element stride and the f64 field read at its natural
     8-byte offset ([+8], not the packed [+4]) — proves placement, not just the
     opcode width. *)
  assert_contains ptx ", 16;" ;
  assert_contains ptx "ld.global.f64" ;
  assert_contains ptx "+8]" ;
  assert_contains ptx "st.global.f64"

(** L8: a vector of variants with an f64 payload now lays the payload region at
    the aligned offset 8 (element stride 16) and COMPILES, rather than raising
    Ptx_codegen_error. The variant param binds and the entry is emitted. *)
let test_f64_variant_param_accepted () =
  let vty = TVariant ("boxed", [("None_", []); ("Some_", [TFloat64])]) in
  let src = make_var "src" (TVec vty) in
  let tid = make_var "tid" TInt32 in
  (* dst is f64 so the matched Some_ payload is written straight back — this
     forces the variant element load to pull the f64 payload from its aligned
     offset 8 (ld.global.f64) and store it (st.global.f64), exercising the
     aligned layout rather than merely compiling. *)
  let dst = make_var "dst" (TVec TFloat64) in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SMatch
          ( EArrayRead ("src", EVar tid),
            [
              ( PConstr ("None_", []),
                SAssign (LArrayElem ("dst", EVar tid), EConst (CFloat64 0.0)) );
              ( PConstr ("Some_", ["v"]),
                SAssign
                  (LArrayElem ("dst", EVar tid), EVar (make_var "v" TFloat64))
              );
            ] ) )
  in
  let k =
    base_kernel
      "f64_variant"
      [
        DParam (src, Some {arr_elttype = vty; arr_memspace = Global});
        DParam (dst, Some {arr_elttype = TFloat64; arr_memspace = Global});
      ]
      body
      []
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx ".entry" ;
  (* Aligned ABI: 16-byte variant element stride and the f64 payload read at
     its aligned offset 8 ([+8], not the packed [+4]). *)
  assert_contains ptx ", 16;" ;
  assert_contains ptx "ld.global.f64" ;
  assert_contains ptx "+8]" ;
  assert_contains ptx "st.global.f64"

(** A one-intrinsic kernel over a float vector of [elt] element type. *)
let make_math_kernel elt path name =
  let out = make_var "out" (TVec elt) in
  let a = make_var "a" (TVec elt) in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("out", EVar tid),
            EIntrinsic (path, name, [EArrayRead ("a", EVar tid)]) ) )
  in
  let mk v = DParam (v, Some {arr_elttype = elt; arr_memspace = Global}) in
  base_kernel ("math_" ^ name) [mk out; mk a] body []

(** Float64 sin lowers to the software implementation (inlined
    Sarek_ir_ptx_softmath helper): Cody-Waite reduction + fma polynomial on f64
    registers, never the f32 [.approx] instruction. *)
let test_f64_sin_softmath () =
  let k = make_math_kernel TFloat64 ["Float64"] "sin" in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "fma.rn.f64" ;
  (* rint(x·2/π) via floor *)
  assert_contains ptx "cvt.rmi.f64.f64" ;
  (* quadrant select *)
  assert_contains ptx "selp.f64" ;
  if contains ptx "sin.approx.f32" then
    Alcotest.fail "f64 sin must not emit the f32 .approx instruction"

(** Float64 exp/log lower to softmath bodies built on the f64<->i64 bitcasts
    (mov.b64) and 64-bit exponent-field manipulation. *)
let test_f64_exp_log_softmath () =
  let k_exp = make_math_kernel TFloat64 ["Float64"] "exp" in
  let ptx_exp = Sarek_ir_ptx.generate k_exp in
  (* 2^n scaling: (n+1023) << 52 then bits_f64 *)
  assert_contains ptx_exp "shl.b64" ;
  assert_contains ptx_exp "mov.b64" ;
  assert_contains ptx_exp "fma.rn.f64" ;
  let k_log = make_math_kernel TFloat64 ["Float64"] "log" in
  let ptx_log = Sarek_ir_ptx.generate k_log in
  (* exponent extract: f64_bits then (bits >> 52) & 0x7ff, mantissa mask/or *)
  assert_contains ptx_log "shr.s64" ;
  assert_contains ptx_log "and.b64" ;
  assert_contains ptx_log "or.b64" ;
  assert_contains ptx_log "mov.b64" ;
  if contains ptx_log "lg2.approx.f32" then
    Alcotest.fail "f64 log must not emit the f32 .approx instruction"

(** f32 asin has no native PTX op and no accurate composition; it lowers via the
    f64 softmath helper: widen (cvt.rn.f64.f32), inline the fdlibm-style f64
    body (fma.rn.f64 + sqrt.rn.f64), round back (cvt.rn.f32.f64). *)
let test_f32_asin_via_f64 () =
  let k = make_math_kernel TFloat32 ["Sarek_stdlib"; "Float32"] "asin" in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx "cvt.rn.f64.f32" ;
  assert_contains ptx "fma.rn.f64" ;
  assert_contains ptx "sqrt.rn.f64" ;
  assert_contains ptx "cvt.rn.f32.f64"

(** Per-thread local array: create_array n Local lowers to SLet (arr,
    EArrayCreate (elt, n, Local), ...). Declaration in the .local state space,
    64-bit base address, typed ld.local/st.local accesses. *)
let test_local_array_markers () =
  let out = make_var "out" (TVec TFloat32) in
  let tmp = make_var "tmp" (TArray (TFloat32, Local)) in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( tmp,
            EArrayCreate (TFloat32, EConst (CInt32 16l), Local),
            SSeq
              [
                SAssign
                  ( LArrayElem ("tmp", EConst (CInt32 0l)),
                    ECast (TFloat32, EVar tid) );
                SAssign
                  ( LArrayElem ("out", EVar tid),
                    EArrayRead ("tmp", EConst (CInt32 0l)) );
              ] ) )
  in
  let k =
    base_kernel
      "local_arr"
      [DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global})]
      body
      []
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx ".local .align 4 .b32 tmp[16];" ;
  assert_contains ptx "mov.u64" ;
  assert_contains ptx "st.local.f32" ;
  assert_contains ptx "ld.local.f32"

(** 8-byte elements get .align 8 .b64 local declarations. *)
let test_local_array_int64_markers () =
  let out = make_var "out" (TVec TInt64) in
  let tmp = make_var "tmp" (TArray (TInt64, Local)) in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( tmp,
            EArrayCreate (TInt64, EConst (CInt32 8l), Local),
            SSeq
              [
                SAssign (LArrayElem ("tmp", EVar tid), EConst (CInt64 7L));
                SAssign
                  (LArrayElem ("out", EVar tid), EArrayRead ("tmp", EVar tid));
              ] ) )
  in
  let k =
    base_kernel
      "local_arr64"
      [DParam (out, Some {arr_elttype = TInt64; arr_memspace = Global})]
      body
      []
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx ".local .align 8 .b64 tmp[8];" ;
  assert_contains ptx "st.local.s64" ;
  assert_contains ptx "ld.local.s64"

(** A DLocal declaration of array type has no size and can never allocate
    storage: it must be rejected fail-closed (previously it fell through to a
    bare uninitialized u64 register — dangling-pointer PTX). *)
let test_dlocal_array_rejected () =
  let out = make_var "out" (TVec TInt32) in
  let arr = make_var "scratch" (TArray (TInt32, Local)) in
  let k =
    {
      (base_kernel
         "dlocal_arr"
         [DParam (out, Some {arr_elttype = TInt32; arr_memspace = Global})]
         (SAssign (LArrayElem ("out", EConst (CInt32 0l)), EConst (CInt32 1l)))
         [])
      with
      kern_locals = [DLocal (arr, None)];
    }
  in
  match Sarek_ir_ptx.generate k with
  | _ -> Alcotest.fail "DLocal of array type should raise Ptx_codegen_error"
  | exception Sarek_codegen.Sarek_ir_ptx_types.Ptx_codegen_error msg ->
      if not (contains msg "scratch" && contains msg "create_array") then
        Alcotest.fail
          (Printf.sprintf
             "DLocal array rejection must name the variable and the \
              workaround; got: %s"
             msg)

(** Non-literal local array sizes are rejected (per-thread stack allocations
    must be static). *)
let test_local_array_dynamic_size_rejected () =
  let out = make_var "out" (TVec TInt32) in
  let tmp = make_var "tmp" (TArray (TInt32, Local)) in
  let n = make_var "n" TInt32 in
  let body =
    SLet
      ( tmp,
        EArrayCreate (TInt32, EVar n, Local),
        SAssign (LArrayElem ("out", EConst (CInt32 0l)), EConst (CInt32 1l)) )
  in
  let k =
    base_kernel
      "local_dyn"
      [
        DParam (out, Some {arr_elttype = TInt32; arr_memspace = Global});
        DParam (n, None);
      ]
      body
      []
  in
  match Sarek_ir_ptx.generate k with
  | _ -> Alcotest.fail "non-literal local array size should be rejected"
  | exception Sarek_codegen.Sarek_ir_ptx_types.Ptx_codegen_error _ -> ()

(** PTX has no atom.local: atomics on a per-thread local array are rejected. *)
let test_atomic_on_local_array_rejected () =
  let out = make_var "out" (TVec TInt32) in
  let tmp = make_var "tmp" (TArray (TInt32, Local)) in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( tmp,
            EArrayCreate (TInt32, EConst (CInt32 4l), Local),
            SExpr
              (EIntrinsic
                 ( ["Sarek_stdlib"; "Gpu"],
                   "atomic_add_int32",
                   [EVar tmp; EVar tid; EConst (CInt32 1l)] )) ) )
  in
  let k =
    base_kernel
      "local_atomic"
      [DParam (out, Some {arr_elttype = TInt32; arr_memspace = Global})]
      body
      []
  in
  match Sarek_ir_ptx.generate k with
  | _ -> Alcotest.fail "atomic on local array should be rejected"
  | exception Sarek_codegen.Sarek_ir_ptx_types.Ptx_codegen_error _ -> ()

(** Dynamic shared memory: DShared (name, elt, None) declares one extern .shared
    region whose byte size is supplied at kernel launch (run_vectors
    ~shared_mem), like extern __shared__ in raw CUDA. Accesses go through the
    normal 32-bit shared path. *)
let test_dynamic_shared_markers () =
  let out = make_var "out" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "thread_idx_x", []),
        SSeq
          [
            SAssign
              (LArrayElem ("dynbuf", EVar tid), EArrayRead ("out", EVar tid));
            SBarrier;
            SAssign
              (LArrayElem ("out", EVar tid), EArrayRead ("dynbuf", EVar tid));
          ] )
  in
  let k =
    {
      (base_kernel
         "dyn_shared"
         [DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global})]
         body
         [])
      with
      kern_locals = [DShared ("dynbuf", TFloat32, None)];
    }
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx ".extern .shared .align 4 .b32 dynbuf[];" ;
  assert_contains ptx "st.shared.f32" ;
  assert_contains ptx "ld.shared.f32"

(** PTX allows one extern .shared region per kernel: a second dynamic shared
    array must be rejected with an error naming both arrays. *)
let test_two_dynamic_shared_rejected () =
  let out = make_var "out" (TVec TFloat32) in
  let k =
    {
      (base_kernel
         "dyn_shared2"
         [DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global})]
         (SAssign (LArrayElem ("out", EConst (CInt32 0l)), EConst (CFloat32 1.0)))
         [])
      with
      kern_locals =
        [DShared ("dyn_a", TFloat32, None); DShared ("dyn_b", TInt32, None)];
    }
  in
  match Sarek_ir_ptx.generate k with
  | _ -> Alcotest.fail "two dynamic shared arrays should be rejected"
  | exception Sarek_codegen.Sarek_ir_ptx_types.Ptx_codegen_error msg ->
      if not (contains msg "dyn_a" && contains msg "dyn_b") then
        Alcotest.fail
          (Printf.sprintf
             "two-dynamic-shared rejection must name both arrays; got: %s"
             msg)

(** Statically-sized DShared decls keep the non-extern declaration shape. *)
let test_static_dshared_markers () =
  let out = make_var "out" (TVec TInt32) in
  let tid = make_var "tid" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "thread_idx_x", []),
        SAssign (LArrayElem ("sbuf", EVar tid), EVar tid) )
  in
  let k =
    {
      (base_kernel
         "static_dshared"
         [DParam (out, Some {arr_elttype = TInt32; arr_memspace = Global})]
         body
         [])
      with
      kern_locals = [DShared ("sbuf", TInt32, Some (EConst (CInt32 128l)))];
    }
  in
  let ptx = Sarek_ir_ptx.generate k in
  assert_contains ptx ".shared .align 4 .b32 sbuf[128];" ;
  assert_absent
    ptx
    ".extern"
    ~why:"statically-sized shared array must not be extern" ;
  assert_contains ptx "st.shared.s32"

let () =
  Alcotest.run
    "ptx_snapshot"
    [
      ( "codegen",
        [
          Alcotest.test_case
            "vector_add PTX contains canonical markers"
            `Quick
            test_vector_add_markers;
          Alcotest.test_case
            "native math emits min/max/cvt.rmi/cvt.rpi/rsqrt"
            `Quick
            test_native_math_markers;
          Alcotest.test_case
            "f64 native math emits sqrt.rn/abs/copysign/fma/rcp .f64"
            `Quick
            test_f64_native_math_markers;
          Alcotest.test_case
            "f32 transcendentals compose sin/cos/lg2/ex2/div .approx"
            `Quick
            test_f32_transcendental_markers;
          Alcotest.test_case
            "f64 sin lowers to softmath (fma/floor/selp .f64, no .approx)"
            `Quick
            test_f64_sin_softmath;
          Alcotest.test_case
            "f64 exp/log lower to softmath (mov.b64 + 64-bit exponent ops)"
            `Quick
            test_f64_exp_log_softmath;
          Alcotest.test_case
            "f32 asin lowers via the f64 softmath helper (cvt round-trip)"
            `Quick
            test_f32_asin_via_f64;
          Alcotest.test_case
            "extended atomics emit atom.*.<op> (+ neg for sub)"
            `Quick
            test_atomic_family_markers;
          Alcotest.test_case
            "cas/inc/dec/64-bit atomics emit atom forms + stride 3"
            `Quick
            test_atomic_cas_incdec_wide_markers;
          Alcotest.test_case
            "width-mismatched atomic value is rejected"
            `Quick
            test_atomic_width_mismatch_rejected;
          Alcotest.test_case
            "float Mod lowers to fmod (div.rn + cvt.rzi + fma)"
            `Quick
            test_float_mod_fmod_markers;
          Alcotest.test_case
            "integer Div/Mod emit signed div.s32/s64 rem.s32/s64"
            `Quick
            test_int_div_rem_signed_markers;
          Alcotest.test_case
            "ECast matrix: bool setp/selp + i32<->i64 cvt pairs"
            `Quick
            test_cast_matrix_markers;
          Alcotest.test_case
            "ECast to unit is rejected"
            `Quick
            test_cast_to_unit_rejected;
          Alcotest.test_case
            "shared array SLet emits .shared decl + ld/st.shared"
            `Quick
            test_shared_array_markers;
          Alcotest.test_case
            "helper call is inlined without .func"
            `Quick
            test_helper_inlining_markers;
          Alcotest.test_case
            "float conversion emits cvt.rn.f32.s32"
            `Quick
            test_float_conversion_markers;
          Alcotest.test_case
            "recursive helper is rejected"
            `Quick
            test_recursive_helper_rejected;
          Alcotest.test_case
            "recursive helper with sarek.inline pragma is depth-unrolled"
            `Quick
            test_recursive_helper_pragma_unrolled;
          Alcotest.test_case
            "guarded array read in if-expr branches (no speculative load)"
            `Quick
            test_guarded_array_read_branches;
          Alcotest.test_case
            "local record is SROA registers (no memory traffic)"
            `Quick
            test_record_sroa_markers;
          Alcotest.test_case
            "nested local record stays in registers"
            `Quick
            test_nested_record_sroa_markers;
          Alcotest.test_case
            "mutable record field update is a register mov"
            `Quick
            test_record_field_mutation_markers;
          Alcotest.test_case
            "helper with record arg + record return is inlined SROA"
            `Quick
            test_helper_record_arg_ret_markers;
          Alcotest.test_case
            "variant construct + SMatch (nullary + 1-arg)"
            `Quick
            test_variant_construct_smatch_markers;
          Alcotest.test_case
            "EMatch value position branches (no selp on result)"
            `Quick
            test_ematch_value_markers;
          Alcotest.test_case
            "three-constructor variant matches via setp.eq chain"
            `Quick
            test_three_ctor_variant_markers;
          Alcotest.test_case
            "multi-arg payload construct + match binds both args"
            `Quick
            test_multiarg_payload_markers;
          Alcotest.test_case
            "nullary-only variant (enum) construct + match"
            `Quick
            test_nullary_only_variant_markers;
          Alcotest.test_case
            "non-exhaustive variant match is rejected"
            `Quick
            test_nonexhaustive_match_rejected;
          Alcotest.test_case
            "tuple construct + destructure stays in registers"
            `Quick
            test_tuple_construct_destructure_markers;
          Alcotest.test_case
            "2-arg variant payload roundtrip via EMatch value"
            `Quick
            test_variant_payload_roundtrip_markers;
          Alcotest.test_case
            "tuple store into a vector element is rejected"
            `Quick
            test_tuple_into_vector_rejected;
          Alcotest.test_case
            "record element field r/w uses mul.wide stride + typed ld/st"
            `Quick
            test_record_elem_field_rw_markers;
          Alcotest.test_case
            "point3d whole copy: stride 12, loads before stores"
            `Quick
            test_point3d_whole_copy_markers;
          Alcotest.test_case
            "single-field element store is one typed st"
            `Quick
            test_record_elem_field_store_markers;
          Alcotest.test_case
            "variant vector element roundtrip (tag ld/st + branch chain)"
            `Quick
            test_variant_elem_roundtrip_markers;
          Alcotest.test_case
            "bare record param rejected with C-17 message"
            `Quick
            test_bare_record_param_rejected;
          Alcotest.test_case
            "L8: mixed-alignment record vector param accepted (aligned f64)"
            `Quick
            test_mixed_align_record_param_accepted;
          Alcotest.test_case
            "L8: f64-payload variant vector param accepted (payload@8)"
            `Quick
            test_f64_variant_param_accepted;
          Alcotest.test_case
            "local array emits .local decl + ld/st.local"
            `Quick
            test_local_array_markers;
          Alcotest.test_case
            "int64 local array gets .align 8 .b64 decl"
            `Quick
            test_local_array_int64_markers;
          Alcotest.test_case
            "DLocal of array type is rejected fail-closed"
            `Quick
            test_dlocal_array_rejected;
          Alcotest.test_case
            "non-literal local array size rejected"
            `Quick
            test_local_array_dynamic_size_rejected;
          Alcotest.test_case
            "atomic on per-thread local array rejected"
            `Quick
            test_atomic_on_local_array_rejected;
          Alcotest.test_case
            "dynamic shared emits .extern .shared decl"
            `Quick
            test_dynamic_shared_markers;
          Alcotest.test_case
            "second dynamic shared array rejected"
            `Quick
            test_two_dynamic_shared_rejected;
          Alcotest.test_case
            "static DShared keeps non-extern decl"
            `Quick
            test_static_dshared_markers;
        ] );
    ]
