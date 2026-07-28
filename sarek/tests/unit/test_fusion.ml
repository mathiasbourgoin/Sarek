(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Sarek_fusion *)

[@@@warning "-32-34"]

open Sarek_ir_types
open Sarek.Sarek_fusion

(** Helper to create a simple variable *)
let mk_var name id typ =
  {var_name = name; var_id = id; var_type = typ; var_mutable = true}

(** Helper to create thread_idx_x intrinsic *)
let thread_idx_x = EIntrinsic (["Gpu"], "thread_idx_x", [])

(** Helper to create a minimal kernel record *)
let mk_kernel name body =
  {default_kernel with kern_name = name; kern_body = body}

let kernel_names kernels = List.map (fun k -> k.kern_name) kernels

(** Test: analyze simple kernel with OneToOne access pattern *)
let test_analyze_one_to_one () =
  (* output[thread_idx_x] = input[thread_idx_x] * 2 *)
  let body =
    SAssign
      ( LArrayElem ("output", thread_idx_x),
        EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l)) )
  in
  let kernel = {default_kernel with kern_name = "scale"; kern_body = body} in
  let info = analyze kernel in
  assert (List.length info.reads = 1) ;
  assert (List.length info.writes = 1) ;
  assert (List.mem_assoc "input" info.reads) ;
  assert (List.mem_assoc "output" info.writes) ;
  assert (not info.has_barriers) ;
  Printf.printf "test_analyze_one_to_one: PASSED\n"

(** Test: analyze kernel with barrier *)
let test_analyze_with_barrier () =
  let body =
    SSeq
      [
        SAssign
          ( LArrayElem ("shared", thread_idx_x),
            EArrayRead ("input", thread_idx_x) );
        SBarrier;
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EArrayRead ("shared", thread_idx_x) );
      ]
  in
  let kernel =
    {default_kernel with kern_name = "with_barrier"; kern_body = body}
  in
  let info = analyze kernel in
  assert info.has_barriers ;
  Printf.printf "test_analyze_with_barrier: PASSED\n"

(** Test: can_fuse returns true for compatible kernels *)
let test_can_fuse_compatible () =
  (* Producer: temp[i] = input[i] * 2 *)
  let producer =
    {
      default_kernel with
      kern_name = "producer";
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
    }
  in
  (* Consumer: output[i] = temp[i] + 1 *)
  let consumer =
    {
      default_kernel with
      kern_name = "consumer";
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop (Add, EArrayRead ("temp", thread_idx_x), EConst (CInt32 1l))
          );
    }
  in
  let result = can_fuse producer consumer "temp" in
  assert result ;
  Printf.printf "test_can_fuse_compatible: PASSED\n"

(** Test: can_fuse returns false when producer has barrier *)
let test_can_fuse_with_barrier () =
  let producer =
    {
      default_kernel with
      kern_name = "producer";
      kern_body =
        SSeq
          [
            SAssign
              ( LArrayElem ("temp", thread_idx_x),
                EArrayRead ("input", thread_idx_x) );
            SBarrier;
          ];
    }
  in
  let consumer =
    {
      default_kernel with
      kern_name = "consumer";
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EArrayRead ("temp", thread_idx_x) );
    }
  in
  let result = can_fuse producer consumer "temp" in
  assert (not result) ;
  Printf.printf "test_can_fuse_with_barrier: PASSED\n"

(** Test: can_fuse returns false when the producer contains a direct atomic
    operation. Pre-fix (Sarek_fusion.ml hardcoded [has_atomics = false]), this
    incorrectly returned [true]. *)
let test_can_fuse_with_direct_atomic () =
  let producer =
    {
      default_kernel with
      kern_name = "producer";
      kern_body =
        SSeq
          [
            SExpr
              (EIntrinsic
                 ([], "atomic_add_int32", [thread_idx_x; EConst (CInt32 1l)]));
            SAssign
              ( LArrayElem ("temp", thread_idx_x),
                EArrayRead ("input", thread_idx_x) );
          ];
    }
  in
  let consumer =
    {
      default_kernel with
      kern_name = "consumer";
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EArrayRead ("temp", thread_idx_x) );
    }
  in
  let result = can_fuse producer consumer "temp" in
  assert (not result) ;
  Printf.printf "test_can_fuse_with_direct_atomic: PASSED\n"

(** Test: can_fuse returns false when an atomic is only reachable through a
    helper function called from the kernel, not from kern_body directly. This is
    the load-bearing case: a walk that only inspects kern_body (rather than also
    walking kern_funcs) misses it and would wrongly permit fusion. *)
let test_can_fuse_with_atomic_in_helper () =
  let bump_param =
    {var_name = "x"; var_id = 0; var_type = TInt32; var_mutable = false}
  in
  let atomic_helper =
    {
      hf_name = "bump";
      hf_params = [bump_param];
      hf_ret_type = TInt32;
      hf_body =
        SReturn
          (EIntrinsic
             ([], "atomic_add_int32", [EVar bump_param; EConst (CInt32 1l)]));
    }
  in
  let producer =
    {
      default_kernel with
      kern_name = "producer";
      kern_body =
        SAssign
          (LArrayElem ("temp", thread_idx_x), EArrayRead ("input", thread_idx_x));
      kern_funcs = [atomic_helper];
    }
  in
  let consumer =
    {
      default_kernel with
      kern_name = "consumer";
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EArrayRead ("temp", thread_idx_x) );
    }
  in
  let result = can_fuse producer consumer "temp" in
  assert (not result) ;
  Printf.printf "test_can_fuse_with_atomic_in_helper: PASSED\n"

(** Test: can_fuse returns false when the only atomic in the producer is inside
    an SAssign's lvalue index expression (e.g.
    [arr.(atomic_add arr2 0 1) <- 5]), not in the RHS. This is the finding-3
    case: a walk that ignores the lvalue and only inspects the RHS expression
    would miss it and would wrongly permit fusion. *)
let test_can_fuse_with_atomic_in_assign_lvalue () =
  let atomic_idx =
    EIntrinsic ([], "atomic_add_int32", [EConst (CInt32 0l); EConst (CInt32 1l)])
  in
  let producer =
    {
      default_kernel with
      kern_name = "producer";
      kern_body =
        SSeq
          [
            SAssign (LArrayElem ("temp", atomic_idx), EConst (CInt32 5l));
            SAssign
              ( LArrayElem ("temp", thread_idx_x),
                EArrayRead ("input", thread_idx_x) );
          ];
    }
  in
  let consumer =
    {
      default_kernel with
      kern_name = "consumer";
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EArrayRead ("temp", thread_idx_x) );
    }
  in
  let result = can_fuse producer consumer "temp" in
  assert (not result) ;
  Printf.printf "test_can_fuse_with_atomic_in_assign_lvalue: PASSED\n"

(** Test: atomic-free kernels still fuse (no regression) *)
let test_can_fuse_no_atomics_regression () =
  let producer =
    {
      default_kernel with
      kern_name = "producer";
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
    }
  in
  let consumer =
    {
      default_kernel with
      kern_name = "consumer";
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop (Add, EArrayRead ("temp", thread_idx_x), EConst (CInt32 1l))
          );
    }
  in
  let result = can_fuse producer consumer "temp" in
  assert result ;
  Printf.printf "test_can_fuse_no_atomics_regression: PASSED\n"

(** Test: fuse inlines producer into consumer *)
let test_fuse_simple () =
  (* Producer: temp[i] = input[i] * 2 *)
  let producer =
    {
      default_kernel with
      kern_name = "producer";
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
    }
  in
  (* Consumer: output[i] = temp[i] + 1 *)
  let consumer =
    {
      default_kernel with
      kern_name = "consumer";
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop (Add, EArrayRead ("temp", thread_idx_x), EConst (CInt32 1l))
          );
    }
  in
  let fused = fuse producer consumer "temp" in
  (* Result should be: output[i] = (input[i] * 2) + 1 *)
  assert (fused.kern_name = "consumer_fused") ;
  (* Check that temp is no longer read *)
  let info = analyze fused in
  assert (not (List.mem_assoc "temp" info.reads)) ;
  assert (List.mem_assoc "input" info.reads) ;
  assert (List.mem_assoc "output" info.writes) ;
  Printf.printf "test_fuse_simple: PASSED\n"

(** Test: fuse_pipeline with multiple kernels *)
let test_fuse_pipeline () =
  (* K1: a[i] = input[i] * 2 *)
  let k1 =
    {
      default_kernel with
      kern_name = "k1";
      kern_body =
        SAssign
          ( LArrayElem ("a", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
    }
  in
  (* K2: b[i] = a[i] + 1 *)
  let k2 =
    {
      default_kernel with
      kern_name = "k2";
      kern_body =
        SAssign
          ( LArrayElem ("b", thread_idx_x),
            EBinop (Add, EArrayRead ("a", thread_idx_x), EConst (CInt32 1l)) );
    }
  in
  (* K3: output[i] = b[i] * 3 *)
  let k3 =
    {
      default_kernel with
      kern_name = "k3";
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop (Mul, EArrayRead ("b", thread_idx_x), EConst (CInt32 3l)) );
    }
  in
  let fused, eliminated = fuse_pipeline [k1; k2; k3] in
  (* Should eliminate both a and b *)
  assert (List.mem "a" eliminated) ;
  assert (List.mem "b" eliminated) ;
  (* Final kernel should read input and write output *)
  let info = analyze fused in
  assert (List.mem_assoc "input" info.reads) ;
  assert (List.mem_assoc "output" info.writes) ;
  assert (not (List.mem_assoc "a" info.reads)) ;
  assert (not (List.mem_assoc "b" info.reads)) ;
  Printf.printf
    "test_fuse_pipeline: PASSED (eliminated: %s)\n"
    (String.concat ", " eliminated)

(** Test: fuse_pipeline_list preserves a pipeline with no fusible pairs *)
let test_fuse_pipeline_list_preserves_unfused () =
  let k1 =
    mk_kernel
      "k1"
      (SAssign
         ( LArrayElem ("a", thread_idx_x),
           EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
         ))
  in
  let k2 =
    mk_kernel
      "k2"
      (SAssign
         ( LArrayElem ("b", thread_idx_x),
           EBinop (Add, EArrayRead ("other", thread_idx_x), EConst (CInt32 1l))
         ))
  in
  let k3 =
    mk_kernel
      "k3"
      (SAssign
         ( LArrayElem ("output", thread_idx_x),
           EBinop (Mul, EArrayRead ("third", thread_idx_x), EConst (CInt32 3l))
         ))
  in
  let fused, eliminated = fuse_pipeline_list [k1; k2; k3] in
  assert (kernel_names fused = ["k1"; "k2"; "k3"]) ;
  assert (eliminated = []) ;
  Printf.printf "test_fuse_pipeline_list_preserves_unfused: PASSED\n"

(** Test: fuse_pipeline_list preserves order around a fused pair *)
let test_fuse_pipeline_list_mixed_order () =
  let pre =
    mk_kernel
      "pre"
      (SAssign
         ( LArrayElem ("pre_out", thread_idx_x),
           EArrayRead ("pre_in", thread_idx_x) ))
  in
  let producer =
    mk_kernel
      "producer"
      (SAssign
         ( LArrayElem ("temp", thread_idx_x),
           EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
         ))
  in
  let consumer =
    mk_kernel
      "consumer"
      (SAssign
         ( LArrayElem ("mid", thread_idx_x),
           EBinop (Add, EArrayRead ("temp", thread_idx_x), EConst (CInt32 1l))
         ))
  in
  let post =
    mk_kernel
      "post"
      (SAssign
         ( LArrayElem ("output", thread_idx_x),
           EArrayRead ("post_in", thread_idx_x) ))
  in
  let fused, eliminated = fuse_pipeline_list [pre; producer; consumer; post] in
  assert (kernel_names fused = ["pre"; "consumer_fused"; "post"]) ;
  assert (eliminated = ["temp"]) ;
  let fused_pair = List.nth fused 1 in
  let info = analyze fused_pair in
  assert (List.mem_assoc "input" info.reads) ;
  assert (not (List.mem_assoc "temp" info.reads)) ;
  assert (List.mem_assoc "mid" info.writes) ;
  Printf.printf "test_fuse_pipeline_list_mixed_order: PASSED\n"

(** Test: fuse_pipeline_list preserves producer when indices differ *)
let test_fuse_pipeline_list_preserves_mismatched_indices () =
  let shifted_idx = EBinop (Add, thread_idx_x, EConst (CInt32 1l)) in
  let producer =
    mk_kernel
      "producer"
      (SAssign
         ( LArrayElem ("temp", thread_idx_x),
           EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
         ))
  in
  let consumer =
    mk_kernel
      "consumer"
      (SAssign
         ( LArrayElem ("output", thread_idx_x),
           EBinop (Add, EArrayRead ("temp", shifted_idx), EConst (CInt32 1l)) ))
  in
  let fused, eliminated = fuse_pipeline_list [producer; consumer] in
  assert (kernel_names fused = ["producer"; "consumer"]) ;
  assert (eliminated = []) ;
  Printf.printf "test_fuse_pipeline_list_preserves_mismatched_indices: PASSED\n"

(** Test: expr_equal *)
let test_expr_equal () =
  let e1 = EBinop (Add, EConst (CInt32 1l), EConst (CInt32 2l)) in
  let e2 = EBinop (Add, EConst (CInt32 1l), EConst (CInt32 2l)) in
  let e3 = EBinop (Add, EConst (CInt32 1l), EConst (CInt32 3l)) in
  assert (expr_equal e1 e2) ;
  assert (not (expr_equal e1 e3)) ;
  assert (expr_equal thread_idx_x thread_idx_x) ;
  Printf.printf "test_expr_equal: PASSED\n"

(** Test: subst_array_read *)
let test_subst_array_read () =
  (* temp[i] + 1  ->  (input[i] * 2) + 1 *)
  let original =
    EBinop (Add, EArrayRead ("temp", thread_idx_x), EConst (CInt32 1l))
  in
  let replacement =
    EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
  in
  let result = subst_array_read "temp" thread_idx_x replacement original in
  match result with
  | EBinop (Add, inner, EConst (CInt32 1l)) -> (
      match inner with
      | EBinop (Mul, EArrayRead ("input", _), EConst (CInt32 2l)) ->
          Printf.printf "test_subst_array_read: PASSED\n"
      | _ -> failwith "test_subst_array_read: wrong inner expression")
  | _ -> failwith "test_subst_array_read: wrong result structure"

(** Test: detect_reduction_pattern *)
let test_detect_reduction_pattern () =
  let loop_var =
    {var_name = "i"; var_id = 1; var_type = TInt32; var_mutable = true}
  in
  let acc =
    {var_name = "sum"; var_id = 2; var_type = TInt32; var_mutable = true}
  in
  (* for i = 0 to n: sum = sum + arr[i] *)
  let body =
    SAssign (LVar acc, EBinop (Add, EVar acc, EArrayRead ("arr", EVar loop_var)))
  in
  let stmt =
    SFor (loop_var, EConst (CInt32 0l), EConst (CInt32 100l), Upto, body)
  in
  let result = detect_reduction_pattern stmt in
  assert (Option.is_some result) ;
  let detected_acc, op, arr, _ = Option.get result in
  assert (detected_acc.var_name = "sum") ;
  assert (op = Add) ;
  assert (arr = "arr") ;
  Printf.printf "test_detect_reduction_pattern: PASSED\n"

(** Test: is_reduction_kernel *)
let test_is_reduction_kernel () =
  let loop_var =
    {var_name = "i"; var_id = 1; var_type = TInt32; var_mutable = true}
  in
  let acc =
    {var_name = "sum"; var_id = 2; var_type = TInt32; var_mutable = true}
  in
  let kernel =
    {
      default_kernel with
      kern_name = "reduce_sum";
      kern_body =
        SSeq
          [
            SAssign (LVar acc, EConst (CInt32 0l));
            SFor
              ( loop_var,
                EConst (CInt32 0l),
                EConst (CInt32 100l),
                Upto,
                SAssign
                  ( LVar acc,
                    EBinop (Add, EVar acc, EArrayRead ("temp", EVar loop_var))
                  ) );
          ];
    }
  in
  let result = is_reduction_kernel kernel "temp" in
  assert (result = Some Add) ;
  Printf.printf "test_is_reduction_kernel: PASSED\n"

(** Test: can_fuse_reduction *)
let test_can_fuse_reduction () =
  (* Map: temp[i] = input[i] * 2 *)
  let map_kernel =
    {
      default_kernel with
      kern_name = "map";
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
    }
  in
  (* Reduce: sum = fold(+, temp) *)
  let loop_var =
    {var_name = "i"; var_id = 1; var_type = TInt32; var_mutable = true}
  in
  let acc =
    {var_name = "sum"; var_id = 2; var_type = TInt32; var_mutable = true}
  in
  let reduce_kernel =
    {
      default_kernel with
      kern_name = "reduce";
      kern_body =
        SSeq
          [
            SAssign (LVar acc, EConst (CInt32 0l));
            SFor
              ( loop_var,
                EConst (CInt32 0l),
                EConst (CInt32 100l),
                Upto,
                SAssign
                  ( LVar acc,
                    EBinop (Add, EVar acc, EArrayRead ("temp", EVar loop_var))
                  ) );
          ];
    }
  in
  let result = can_fuse_reduction map_kernel reduce_kernel "temp" in
  assert result ;
  Printf.printf "test_can_fuse_reduction: PASSED\n"

(** Test: fuse_reduction *)
let test_fuse_reduction () =
  (* Map: temp[thread_idx_x] = input[thread_idx_x] * 2 *)
  let map_kernel =
    {
      default_kernel with
      kern_name = "map";
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
    }
  in
  (* Reduce: sum = fold(+, temp) with loop var i *)
  let loop_var =
    {var_name = "i"; var_id = 1; var_type = TInt32; var_mutable = true}
  in
  let acc =
    {var_name = "sum"; var_id = 2; var_type = TInt32; var_mutable = true}
  in
  let reduce_kernel =
    {
      default_kernel with
      kern_name = "reduce";
      kern_body =
        SSeq
          [
            SAssign (LVar acc, EConst (CInt32 0l));
            SFor
              ( loop_var,
                EConst (CInt32 0l),
                EConst (CInt32 100l),
                Upto,
                SAssign
                  ( LVar acc,
                    EBinop (Add, EVar acc, EArrayRead ("temp", EVar loop_var))
                  ) );
          ];
    }
  in
  let fused = fuse_reduction map_kernel reduce_kernel "temp" in
  assert (fused.kern_name = "reduce_fused") ;
  (* Fused should not read from temp anymore *)
  let info = analyze fused in
  assert (not (List.mem_assoc "temp" info.reads)) ;
  Printf.printf "test_fuse_reduction: PASSED\n"

(** Test: try_fuse with reduction *)
let test_try_fuse_reduction () =
  (* Map: temp[i] = input[i] * 2 *)
  let map_kernel =
    {
      default_kernel with
      kern_name = "map";
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
    }
  in
  let loop_var =
    {var_name = "i"; var_id = 1; var_type = TInt32; var_mutable = true}
  in
  let acc =
    {var_name = "sum"; var_id = 2; var_type = TInt32; var_mutable = true}
  in
  let reduce_kernel =
    {
      default_kernel with
      kern_name = "reduce";
      kern_body =
        SSeq
          [
            SAssign (LVar acc, EConst (CInt32 0l));
            SFor
              ( loop_var,
                EConst (CInt32 0l),
                EConst (CInt32 100l),
                Upto,
                SAssign
                  ( LVar acc,
                    EBinop (Add, EVar acc, EArrayRead ("temp", EVar loop_var))
                  ) );
          ];
    }
  in
  let result = try_fuse map_kernel reduce_kernel "temp" in
  assert (Option.is_some result) ;
  Printf.printf "test_try_fuse_reduction: PASSED\n"

(** Test: stencil pattern detection *)
let test_stencil_pattern () =
  (* Kernel: output[i] = (input[i-1] + input[i] + input[i+1]) / 3 *)
  let kernel =
    {
      default_kernel with
      kern_name = "blur";
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop
              ( Div,
                EBinop
                  ( Add,
                    EBinop
                      ( Add,
                        EArrayRead
                          ( "input",
                            EBinop (Sub, thread_idx_x, EConst (CInt32 1l)) ),
                        EArrayRead ("input", thread_idx_x) ),
                    EArrayRead
                      ("input", EBinop (Add, thread_idx_x, EConst (CInt32 1l)))
                  ),
                EConst (CInt32 3l) ) );
    }
  in
  let info = analyze kernel in
  match List.assoc_opt "input" info.reads with
  | Some (Stencil offsets) ->
      assert (List.mem (-1) offsets) ;
      assert (List.mem 0 offsets) ;
      assert (List.mem 1 offsets) ;
      Printf.printf
        "test_stencil_pattern: PASSED (offsets: %s)\n"
        (String.concat ", " (List.map string_of_int offsets))
  | _ -> failwith "test_stencil_pattern: expected Stencil pattern"

(** Test: stencil radius computation *)
let test_stencil_radius () =
  assert (stencil_radius [-1; 0; 1] = 1) ;
  assert (stencil_radius [-2; -1; 0; 1; 2] = 2) ;
  assert (stencil_radius [0] = 0) ;
  assert (stencil_radius [-3; 0; 1] = 3) ;
  Printf.printf "test_stencil_radius: PASSED\n"

(** Test: can_fuse_stencil *)
let test_can_fuse_stencil () =
  (* Producer: temp[i] = input[i-1] + input[i+1] *)
  let producer =
    {
      default_kernel with
      kern_name = "producer";
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop
              ( Add,
                EArrayRead
                  ("input", EBinop (Sub, thread_idx_x, EConst (CInt32 1l))),
                EArrayRead
                  ("input", EBinop (Add, thread_idx_x, EConst (CInt32 1l))) ) );
    }
  in
  (* Consumer: output[i] = temp[i-1] + temp[i] + temp[i+1] *)
  let consumer =
    {
      default_kernel with
      kern_name = "consumer";
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop
              ( Add,
                EBinop
                  ( Add,
                    EArrayRead
                      ("temp", EBinop (Sub, thread_idx_x, EConst (CInt32 1l))),
                    EArrayRead ("temp", thread_idx_x) ),
                EArrayRead
                  ("temp", EBinop (Add, thread_idx_x, EConst (CInt32 1l))) ) );
    }
  in
  let result = can_fuse_stencil producer consumer "temp" in
  assert result ;
  Printf.printf "test_can_fuse_stencil: PASSED\n"

(** Test: fuse_stencil *)
let test_fuse_stencil () =
  (* Producer: temp[i] = input[i] * 2 (simple case) *)
  let producer =
    {
      default_kernel with
      kern_name = "producer";
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
    }
  in
  (* Consumer: output[i] = temp[i-1] + temp[i+1] *)
  let consumer =
    {
      default_kernel with
      kern_name = "consumer";
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop
              ( Add,
                EArrayRead
                  ("temp", EBinop (Sub, thread_idx_x, EConst (CInt32 1l))),
                EArrayRead
                  ("temp", EBinop (Add, thread_idx_x, EConst (CInt32 1l))) ) );
    }
  in
  let fused = fuse_stencil producer consumer "temp" in
  assert (fused.kern_name = "consumer_stencil_fused") ;
  (* Fused should read input, not temp *)
  let info = analyze fused in
  assert (not (List.mem_assoc "temp" info.reads)) ;
  Printf.printf "test_fuse_stencil: PASSED\n"

(** Test: try_fuse_all *)
let test_try_fuse_all () =
  (* Simple OneToOne case should use vertical fusion *)
  let producer =
    {
      default_kernel with
      kern_name = "producer";
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
    }
  in
  let consumer =
    {
      default_kernel with
      kern_name = "consumer";
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop (Add, EArrayRead ("temp", thread_idx_x), EConst (CInt32 1l))
          );
    }
  in
  let result = try_fuse_all producer consumer "temp" in
  assert (Option.is_some result) ;
  Printf.printf "test_try_fuse_all: PASSED\n"

(** Test: should_fuse recommends Fuse for OneToOne -> OneToOne *)
let test_should_fuse_one_to_one () =
  let producer =
    {
      default_kernel with
      kern_name = "producer";
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
    }
  in
  let consumer =
    {
      default_kernel with
      kern_name = "consumer";
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop (Add, EArrayRead ("temp", thread_idx_x), EConst (CInt32 1l))
          );
    }
  in
  let hint = should_fuse producer consumer "temp" in
  assert (hint.decision = Fuse) ;
  Printf.printf "test_should_fuse_one_to_one: PASSED (%s)\n" hint.reason

(** Test: should_fuse returns DontFuse for barrier *)
let test_should_fuse_barrier () =
  let producer =
    {
      default_kernel with
      kern_name = "producer";
      kern_body =
        SSeq
          [
            SAssign
              ( LArrayElem ("temp", thread_idx_x),
                EArrayRead ("input", thread_idx_x) );
            SBarrier;
          ];
    }
  in
  let consumer =
    {
      default_kernel with
      kern_name = "consumer";
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EArrayRead ("temp", thread_idx_x) );
    }
  in
  let hint = should_fuse producer consumer "temp" in
  assert (hint.decision = DontFuse) ;
  Printf.printf "test_should_fuse_barrier: PASSED (%s)\n" hint.reason

(** Test: should_fuse returns MaybeFuse for small stencil *)
let test_should_fuse_small_stencil () =
  let producer =
    {
      default_kernel with
      kern_name = "producer";
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
    }
  in
  (* Consumer reads temp[i-1], temp[i], temp[i+1] *)
  let consumer =
    {
      default_kernel with
      kern_name = "consumer";
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop
              ( Add,
                EBinop
                  ( Add,
                    EArrayRead
                      ("temp", EBinop (Sub, thread_idx_x, EConst (CInt32 1l))),
                    EArrayRead ("temp", thread_idx_x) ),
                EArrayRead
                  ("temp", EBinop (Add, thread_idx_x, EConst (CInt32 1l))) ) );
    }
  in
  let hint = should_fuse producer consumer "temp" in
  assert (hint.decision = MaybeFuse) ;
  Printf.printf "test_should_fuse_small_stencil: PASSED (%s)\n" hint.reason

(** Test: auto_fuse_pipeline with OneToOne kernels *)
let test_auto_fuse_pipeline () =
  (* K1: a[i] = input[i] * 2 *)
  let k1 =
    {
      default_kernel with
      kern_name = "k1";
      kern_body =
        SAssign
          ( LArrayElem ("a", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
    }
  in
  (* K2: b[i] = a[i] + 1 *)
  let k2 =
    {
      default_kernel with
      kern_name = "k2";
      kern_body =
        SAssign
          ( LArrayElem ("b", thread_idx_x),
            EBinop (Add, EArrayRead ("a", thread_idx_x), EConst (CInt32 1l)) );
    }
  in
  (* K3: output[i] = b[i] * 3 *)
  let k3 =
    {
      default_kernel with
      kern_name = "k3";
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop (Mul, EArrayRead ("b", thread_idx_x), EConst (CInt32 3l)) );
    }
  in
  let fused, eliminated, skipped = auto_fuse_pipeline [k1; k2; k3] in
  assert (List.mem "a" eliminated) ;
  assert (List.mem "b" eliminated) ;
  assert (List.length skipped = 0) ;
  let info = analyze fused in
  assert (List.mem_assoc "input" info.reads) ;
  assert (List.mem_assoc "output" info.writes) ;
  Printf.printf
    "test_auto_fuse_pipeline: PASSED (eliminated: %s)\n"
    (String.concat ", " eliminated)

(** Test: auto_fuse_pipeline skips stencil *)
let test_auto_fuse_pipeline_skip_stencil () =
  (* K1: temp[i] = input[i] * 2 *)
  let k1 =
    {
      default_kernel with
      kern_name = "k1";
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
    }
  in
  (* K2: output[i] = temp[i-1] + temp[i+1] (stencil) *)
  let k2 =
    {
      default_kernel with
      kern_name = "k2";
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop
              ( Add,
                EArrayRead
                  ("temp", EBinop (Sub, thread_idx_x, EConst (CInt32 1l))),
                EArrayRead
                  ("temp", EBinop (Add, thread_idx_x, EConst (CInt32 1l))) ) );
    }
  in
  let fused, eliminated, skipped = auto_fuse_pipeline_list [k1; k2] in
  (* Should skip because stencil is MaybeFuse *)
  assert (kernel_names fused = ["k1"; "k2"]) ;
  assert (List.length eliminated = 0) ;
  assert (List.length skipped = 1) ;
  Printf.printf
    "test_auto_fuse_pipeline_skip_stencil: PASSED (skipped: %s)\n"
    (String.concat ", " skipped)

(** Test: auto_fuse_pipeline_list preserves kernels when heuristics skip fusion
*)
let test_auto_fuse_pipeline_list_preserves_dont_fuse () =
  let producer =
    mk_kernel
      "barrier_producer"
      (SSeq
         [
           SAssign
             ( LArrayElem ("temp", thread_idx_x),
               EArrayRead ("input", thread_idx_x) );
           SBarrier;
         ])
  in
  let consumer =
    mk_kernel
      "consumer"
      (SAssign
         ( LArrayElem ("output", thread_idx_x),
           EBinop (Add, EArrayRead ("temp", thread_idx_x), EConst (CInt32 1l))
         ))
  in
  let fused, eliminated, skipped =
    auto_fuse_pipeline_list [producer; consumer]
  in
  assert (kernel_names fused = ["barrier_producer"; "consumer"]) ;
  assert (eliminated = []) ;
  assert (skipped = ["Barrier prevents fusion"]) ;
  Printf.printf "test_auto_fuse_pipeline_list_preserves_dont_fuse: PASSED\n"

(** Test: auto_fuse_pipeline_list preserves order around a fused pair *)
let test_auto_fuse_pipeline_list_mixed_order () =
  let pre =
    mk_kernel
      "pre"
      (SAssign
         ( LArrayElem ("pre_out", thread_idx_x),
           EArrayRead ("pre_in", thread_idx_x) ))
  in
  let producer =
    mk_kernel
      "producer"
      (SAssign
         ( LArrayElem ("temp", thread_idx_x),
           EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
         ))
  in
  let consumer =
    mk_kernel
      "consumer"
      (SAssign
         ( LArrayElem ("mid", thread_idx_x),
           EBinop (Add, EArrayRead ("temp", thread_idx_x), EConst (CInt32 1l))
         ))
  in
  let post =
    mk_kernel
      "post"
      (SAssign
         ( LArrayElem ("output", thread_idx_x),
           EArrayRead ("post_in", thread_idx_x) ))
  in
  let fused, eliminated, skipped =
    auto_fuse_pipeline_list [pre; producer; consumer; post]
  in
  assert (kernel_names fused = ["pre"; "consumer_fused"; "post"]) ;
  assert (eliminated = ["temp"]) ;
  assert (skipped = []) ;
  Printf.printf "test_auto_fuse_pipeline_list_mixed_order: PASSED\n"

(** Test: auto_fuse_pipeline_list preserves producer when indices differ *)
let test_auto_fuse_pipeline_list_preserves_mismatched_indices () =
  let shifted_idx = EBinop (Add, thread_idx_x, EConst (CInt32 1l)) in
  let producer =
    mk_kernel
      "producer"
      (SAssign
         ( LArrayElem ("temp", thread_idx_x),
           EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
         ))
  in
  let consumer =
    mk_kernel
      "consumer"
      (SAssign
         ( LArrayElem ("output", thread_idx_x),
           EBinop (Add, EArrayRead ("temp", shifted_idx), EConst (CInt32 1l)) ))
  in
  let fused, eliminated, skipped =
    auto_fuse_pipeline_list [producer; consumer]
  in
  assert (kernel_names fused = ["producer"; "consumer"]) ;
  assert (eliminated = []) ;
  assert (skipped = ["Producer and consumer use different element indices"]) ;
  Printf.printf
    "test_auto_fuse_pipeline_list_preserves_mismatched_indices: PASSED\n"

(** {1 backlog-153: eliminating an array whose LENGTH is still used}

    [len(arr)] does not live in the IR. It lowers to the companion
    [sarek_<arr>_length] launch argument, which every backend emits from the
    PARAMETER list, so deleting the parameter deletes the length with it and
    there is no expression to substitute in its place.

    Before the fix, [analyze] could not see a length use at all ([EArrayLen]
    contributed to neither [reads] nor anything else), so a consumer body of the
    form [output[tid] = temp[tid] + len(temp)] still looked purely [OneToOne] in
    [temp]. [can_fuse] said yes, [fuse] dropped the [temp] parameter,
    [subst_array_read] rewrote the element read and walked straight past the
    [EArrayLen], and the fused kernel named a parameter that no longer existed.

    The consequence was measured on this shape, on every backend:

    - CUDA / OpenCL / Metal emitted the bare identifier [sarek_temp_length].
      clang's OpenCL front end:
      [error: use of undeclared identifier 'sarek_temp_length'].
    - GLSL emitted it too; glslangValidator:
      ['sarek_temp_length' : undeclared identifier].
    - WGSL emitted [params.sarek_temp_length] against a [Params] struct with no
      such field; naga: [invalid field accessor `sarek_temp_length`].
    - PTX raised its own [unsupported construct: EArrayLen] refusal.
    - The interpreter raised [Unbound_variable] from [get_array].

    So the failure was LOUD on every path — a broken build, not a wrong answer,
    and specifically not the silent wrong-width family. It is still worth
    refusing at the fusion pass rather than at the vendor compiler: the name in
    every one of those diagnostics is one the author never wrote, in generated
    source they never saw, for an array the optimiser deleted. *)

let len_producer =
  mk_kernel
    "producer"
    (SAssign
       ( LArrayElem ("temp", thread_idx_x),
         EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l)) ))

(* output[tid] = temp[tid] + len(temp) — an element read AND a length use of
   the same array. The element read is what makes this fusable-looking; the
   length use is what makes it unfusable. *)
let len_consumer =
  mk_kernel
    "consumer"
    (SAssign
       ( LArrayElem ("output", thread_idx_x),
         EBinop (Add, EArrayRead ("temp", thread_idx_x), EArrayLen "temp") ))

(** [expr_uses_array] is the predicate that answers "may I delete this array?".
    It returned [false] for [EArrayLen] — the root of backlog-153. *)
let test_expr_uses_array_sees_length () =
  assert (expr_uses_array "temp" (EArrayLen "temp")) ;
  (* Still discriminating: a length of a DIFFERENT array is not a use. *)
  assert (not (expr_uses_array "temp" (EArrayLen "other"))) ;
  (* And it reaches a length nested in ordinary expression structure. *)
  assert (
    expr_uses_array "temp" (EBinop (Add, EConst (CInt32 1l), EArrayLen "temp"))) ;
  assert (stmt_uses_array "temp" len_consumer.kern_body) ;
  Printf.printf "test_expr_uses_array_sees_length: PASSED\n"

(** [analyze] reports the length use, and does NOT let it corrupt the access
    pattern — [temp] must still read as [OneToOne], because the reason to refuse
    is the length, not a pretended gather. *)
let test_analyze_reports_length_uses () =
  let info = analyze len_consumer in
  assert (info.length_uses = ["temp"]) ;
  assert (
    match List.assoc_opt "temp" info.reads with
    | Some (OneToOne _) -> true
    | _ -> false) ;
  (* A kernel that takes no length reports none. *)
  assert ((analyze len_producer).length_uses = []) ;
  Printf.printf "test_analyze_reports_length_uses: PASSED\n"

(** The three [can_fuse*] guards all refuse the shape. *)
let test_can_fuse_refuses_length_use () =
  assert (not (can_fuse len_producer len_consumer "temp")) ;
  assert (not (can_fuse_stencil len_producer len_consumer "temp")) ;
  (* And the producer side counts too: its write expression is spliced into
     the consumer body verbatim, so a length there lands in the fused kernel
     just the same. *)
  let producer_takes_len =
    mk_kernel
      "producer_len"
      (SAssign (LArrayElem ("temp", thread_idx_x), EArrayLen "temp"))
  in
  let plain_consumer =
    mk_kernel
      "consumer"
      (SAssign
         (LArrayElem ("output", thread_idx_x), EArrayRead ("temp", thread_idx_x)))
  in
  assert (not (can_fuse producer_takes_len plain_consumer "temp")) ;
  (* Positive control: the same pair WITHOUT the length use still fuses, so
     the guard is refusing the length and not the shape. *)
  assert (can_fuse len_producer plain_consumer "temp") ;
  Printf.printf "test_can_fuse_refuses_length_use: PASSED\n"

(** [auto_fuse_pipeline_list] preserves both stages and names the real reason.
*)
let test_auto_fuse_preserves_length_user () =
  let hint = should_fuse len_producer len_consumer "temp" in
  assert (hint.decision = DontFuse) ;
  assert (
    hint.reason
    = "len(temp) is used, and the length of an eliminated array cannot be \
       recovered") ;
  let pipeline, eliminated, skipped =
    auto_fuse_pipeline_list [len_producer; len_consumer]
  in
  assert (kernel_names pipeline = ["producer"; "consumer"]) ;
  assert (eliminated = []) ;
  assert (skipped = [hint.reason]) ;
  (* fuse_pipeline_list, which does not consult should_fuse, must refuse via
     can_fuse alone. *)
  let pipeline', eliminated' =
    fuse_pipeline_list [len_producer; len_consumer]
  in
  assert (kernel_names pipeline' = ["producer"; "consumer"]) ;
  assert (eliminated' = []) ;
  Printf.printf "test_auto_fuse_preserves_length_user: PASSED\n"

(** The backstop. [fuse] is public and documented as callable directly, so the
    guards are not the only way in. Reaching it means an invariant the guards
    claim to enforce did not hold, and the only alternative to raising is
    handing back the kernel that produced those five vendor diagnostics. *)
let test_fuse_rejects_surviving_intermediate () =
  match fuse len_producer len_consumer "temp" with
  | exception
      Sarek.Fusion_error.Fusion_error
        (Sarek.Fusion_error.Invalid_fusion {kernel; reason}) ->
      assert (kernel = "consumer_fused") ;
      (* The message must name the parameter, or it cannot be acted on. *)
      assert (
        let re = Str.regexp_string "'temp'" in
        try
          ignore (Str.search_forward re reason 0) ;
          true
        with Not_found -> false) ;
      Printf.printf "test_fuse_rejects_surviving_intermediate: PASSED\n"
  | _ ->
      Printf.printf
        "test_fuse_rejects_surviving_intermediate: FAILED (fuse returned a \
         kernel that still references 'temp')\n" ;
      assert false

(** End-to-end, on the artefact that actually broke: no backend source may
    mention [sarek_temp_length]. This is the assertion that pins the observable
    defect rather than the internal predicate — it goes red on the unfixed pass
    whatever route the regression takes back in. *)
let test_no_backend_names_eliminated_length () =
  let arr name id =
    DParam
      ( {
          var_name = name;
          var_id = id;
          var_type = TVec TInt32;
          var_mutable = true;
        },
        Some {arr_elttype = TInt32; arr_memspace = Global} )
  in
  let with_params k params = {k with kern_params = params} in
  let producer = with_params len_producer [arr "input" 1; arr "temp" 2] in
  let consumer = with_params len_consumer [arr "temp" 2; arr "output" 3] in
  let pipeline, _, _ = auto_fuse_pipeline_list [producer; consumer] in
  let mentions_dead_length src =
    let re = Str.regexp_string "sarek_temp_length" in
    try
      ignore (Str.search_forward re src 0) ;
      true
    with Not_found -> false
  in
  List.iter
    (fun k ->
      let declares_temp =
        List.exists
          (function
            | DParam (v, _) -> v.var_name = "temp"
            | DShared (n, _, _) -> n = "temp"
            | _ -> false)
          k.kern_params
      in
      List.iter
        (fun (label, src) ->
          (* A kernel that still DECLARES temp may of course name its length;
             a kernel that does not declare it must never do so. *)
          if (not declares_temp) && mentions_dead_length src then begin
            Printf.printf
              "test_no_backend_names_eliminated_length: FAILED (%s source for \
               %s names sarek_temp_length with no temp parameter)\n"
              label
              k.kern_name ;
            assert false
          end)
        [
          ("CUDA", Sarek_codegen.Sarek_ir_cuda.generate k);
          ("OpenCL", Sarek_codegen.Sarek_ir_opencl.generate k);
          ("Metal", Sarek_codegen.Sarek_ir_metal.generate k);
        ])
    pipeline ;
  Printf.printf "test_no_backend_names_eliminated_length: PASSED\n"

let () =
  Printf.printf "=== Fusion Unit Tests ===\n" ;
  test_expr_equal () ;
  test_subst_array_read () ;
  test_analyze_one_to_one () ;
  test_analyze_with_barrier () ;
  test_can_fuse_compatible () ;
  test_can_fuse_with_barrier () ;
  test_can_fuse_with_direct_atomic () ;
  test_can_fuse_with_atomic_in_helper () ;
  test_can_fuse_with_atomic_in_assign_lvalue () ;
  test_can_fuse_no_atomics_regression () ;
  test_fuse_simple () ;
  test_fuse_pipeline () ;
  test_fuse_pipeline_list_preserves_unfused () ;
  test_fuse_pipeline_list_mixed_order () ;
  test_fuse_pipeline_list_preserves_mismatched_indices () ;
  Printf.printf "\n=== Reduction Fusion Tests ===\n" ;
  test_detect_reduction_pattern () ;
  test_is_reduction_kernel () ;
  test_can_fuse_reduction () ;
  test_fuse_reduction () ;
  test_try_fuse_reduction () ;
  Printf.printf "\n=== Stencil Fusion Tests ===\n" ;
  test_stencil_pattern () ;
  test_stencil_radius () ;
  test_can_fuse_stencil () ;
  test_fuse_stencil () ;
  test_try_fuse_all () ;
  Printf.printf "\n=== Auto-Fusion Heuristics Tests ===\n" ;
  test_should_fuse_one_to_one () ;
  test_should_fuse_barrier () ;
  test_should_fuse_small_stencil () ;
  test_auto_fuse_pipeline () ;
  test_auto_fuse_pipeline_skip_stencil () ;
  test_auto_fuse_pipeline_list_preserves_dont_fuse () ;
  test_auto_fuse_pipeline_list_mixed_order () ;
  test_auto_fuse_pipeline_list_preserves_mismatched_indices () ;
  Printf.printf "\n=== backlog-153: length of an eliminated array ===\n" ;
  test_expr_uses_array_sees_length () ;
  test_analyze_reports_length_uses () ;
  test_can_fuse_refuses_length_use () ;
  test_auto_fuse_preserves_length_user () ;
  test_fuse_rejects_surviving_intermediate () ;
  test_no_backend_names_eliminated_length () ;
  Printf.printf "=== All tests passed! ===\n"
