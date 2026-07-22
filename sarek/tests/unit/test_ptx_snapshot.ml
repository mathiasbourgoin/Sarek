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
        ] );
    ]
