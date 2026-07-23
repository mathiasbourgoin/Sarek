(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_ir_ptx - PTX Code Generation from Sarek IR
 *
 * Thin re-export façade. The implementation is split across:
 *   Sarek_ir_ptx_types   - register allocator, environment, emit helpers
 *   Sarek_ir_ptx_mem     - array load/store emission
 *   Sarek_ir_ptx_expr    - expression emitter
 *   Sarek_ir_ptx_stmt    - statement emitter
 *   Sarek_ir_ptx_kernel  - kernel-level generation entry points
 *
 * Supported: records/variants (aligned C-ABI layout, see PtxLayout.v),
 * EMatch/SMatch, helper functions via EApp inlining, EArrayLen on parameter
 * arrays (local/shared arrays fall back to another backend), static and
 * dynamic .shared, per-thread .local arrays, atomics (incl. CAS and 64-bit
 * forms), f64 softmath.
 *
 * Known gaps:
 *   - No ptxas validation gate in CI; GPU validation is manual
 *   - Warp-level primitives modeled in the IR but not emitted
 ******************************************************************************)

open Sarek_ir_types

let generate = Sarek_ir_ptx_kernel.generate

let generate_with_types = Sarek_ir_ptx_kernel.generate_with_types

(** {1 Spike demo: vector_add}

    Constructs the vector_add IR and calls [generate], demonstrating that the
    emitter produces structurally correct PTX for the simplest Sarek kernel
    pattern. *)
let demo_vector_add_ptx () : string =
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
  let k =
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
  in
  generate k
