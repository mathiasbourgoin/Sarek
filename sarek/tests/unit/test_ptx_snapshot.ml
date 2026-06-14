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
  let ptx = Sarek_ir_ptx_kernel.generate k in
  assert_contains ptx ".entry" ;
  assert_contains ptx ".param" ;
  assert_contains ptx "ld.global" ;
  assert_contains ptx "add.u32" ;
  assert_contains ptx "st.global" ;
  assert_contains ptx "ret"

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
        ] );
    ]
