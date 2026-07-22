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
            "extended atomics emit atom.*.<op> (+ neg for sub)"
            `Quick
            test_atomic_family_markers;
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
        ] );
    ]
