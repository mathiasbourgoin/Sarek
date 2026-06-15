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
  {
    kern_name = name;
    kern_params = [];
    kern_locals = [];
    kern_body = body;
    kern_types = [];
    kern_funcs = [];
    kern_native_fn = None;
    kern_variants = [];
  }

let kernel_names kernels = List.map (fun k -> k.kern_name) kernels

(** Test: analyze simple kernel with OneToOne access pattern *)
let test_analyze_one_to_one () =
  (* output[thread_idx_x] = input[thread_idx_x] * 2 *)
  let body =
    SAssign
      ( LArrayElem ("output", thread_idx_x),
        EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l)) )
  in
  let kernel =
    {
      kern_name = "scale";
      kern_params = [];
      kern_locals = [];
      kern_body = body;
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
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
    {
      kern_name = "with_barrier";
      kern_params = [];
      kern_locals = [];
      kern_body = body;
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  let info = analyze kernel in
  assert info.has_barriers ;
  Printf.printf "test_analyze_with_barrier: PASSED\n"

(** Test: can_fuse returns true for compatible kernels *)
let test_can_fuse_compatible () =
  (* Producer: temp[i] = input[i] * 2 *)
  let producer =
    {
      kern_name = "producer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  (* Consumer: output[i] = temp[i] + 1 *)
  let consumer =
    {
      kern_name = "consumer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop (Add, EArrayRead ("temp", thread_idx_x), EConst (CInt32 1l))
          );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  let result = can_fuse producer consumer "temp" in
  assert result ;
  Printf.printf "test_can_fuse_compatible: PASSED\n"

(** Test: can_fuse returns false when producer has barrier *)
let test_can_fuse_with_barrier () =
  let producer =
    {
      kern_name = "producer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SSeq
          [
            SAssign
              ( LArrayElem ("temp", thread_idx_x),
                EArrayRead ("input", thread_idx_x) );
            SBarrier;
          ];
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  let consumer =
    {
      kern_name = "consumer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EArrayRead ("temp", thread_idx_x) );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  let result = can_fuse producer consumer "temp" in
  assert (not result) ;
  Printf.printf "test_can_fuse_with_barrier: PASSED\n"

(** Test: fuse inlines producer into consumer *)
let test_fuse_simple () =
  (* Producer: temp[i] = input[i] * 2 *)
  let producer =
    {
      kern_name = "producer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  (* Consumer: output[i] = temp[i] + 1 *)
  let consumer =
    {
      kern_name = "consumer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop (Add, EArrayRead ("temp", thread_idx_x), EConst (CInt32 1l))
          );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
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
      kern_name = "k1";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("a", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  (* K2: b[i] = a[i] + 1 *)
  let k2 =
    {
      kern_name = "k2";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("b", thread_idx_x),
            EBinop (Add, EArrayRead ("a", thread_idx_x), EConst (CInt32 1l)) );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  (* K3: output[i] = b[i] * 3 *)
  let k3 =
    {
      kern_name = "k3";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop (Mul, EArrayRead ("b", thread_idx_x), EConst (CInt32 3l)) );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
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
      kern_name = "reduce_sum";
      kern_params = [];
      kern_locals = [];
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
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
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
      kern_name = "map";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
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
      kern_name = "reduce";
      kern_params = [];
      kern_locals = [];
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
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
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
      kern_name = "map";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
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
      kern_name = "reduce";
      kern_params = [];
      kern_locals = [];
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
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
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
      kern_name = "map";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
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
      kern_name = "reduce";
      kern_params = [];
      kern_locals = [];
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
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
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
      kern_name = "blur";
      kern_params = [];
      kern_locals = [];
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
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
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
      kern_name = "producer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop
              ( Add,
                EArrayRead
                  ("input", EBinop (Sub, thread_idx_x, EConst (CInt32 1l))),
                EArrayRead
                  ("input", EBinop (Add, thread_idx_x, EConst (CInt32 1l))) ) );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  (* Consumer: output[i] = temp[i-1] + temp[i] + temp[i+1] *)
  let consumer =
    {
      kern_name = "consumer";
      kern_params = [];
      kern_locals = [];
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
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
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
      kern_name = "producer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  (* Consumer: output[i] = temp[i-1] + temp[i+1] *)
  let consumer =
    {
      kern_name = "consumer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop
              ( Add,
                EArrayRead
                  ("temp", EBinop (Sub, thread_idx_x, EConst (CInt32 1l))),
                EArrayRead
                  ("temp", EBinop (Add, thread_idx_x, EConst (CInt32 1l))) ) );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
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
      kern_name = "producer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  let consumer =
    {
      kern_name = "consumer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop (Add, EArrayRead ("temp", thread_idx_x), EConst (CInt32 1l))
          );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  let result = try_fuse_all producer consumer "temp" in
  assert (Option.is_some result) ;
  Printf.printf "test_try_fuse_all: PASSED\n"

(** Test: should_fuse recommends Fuse for OneToOne -> OneToOne *)
let test_should_fuse_one_to_one () =
  let producer =
    {
      kern_name = "producer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  let consumer =
    {
      kern_name = "consumer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop (Add, EArrayRead ("temp", thread_idx_x), EConst (CInt32 1l))
          );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  let hint = should_fuse producer consumer "temp" in
  assert (hint.decision = Fuse) ;
  Printf.printf "test_should_fuse_one_to_one: PASSED (%s)\n" hint.reason

(** Test: should_fuse returns DontFuse for barrier *)
let test_should_fuse_barrier () =
  let producer =
    {
      kern_name = "producer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SSeq
          [
            SAssign
              ( LArrayElem ("temp", thread_idx_x),
                EArrayRead ("input", thread_idx_x) );
            SBarrier;
          ];
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  let consumer =
    {
      kern_name = "consumer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EArrayRead ("temp", thread_idx_x) );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  let hint = should_fuse producer consumer "temp" in
  assert (hint.decision = DontFuse) ;
  Printf.printf "test_should_fuse_barrier: PASSED (%s)\n" hint.reason

(** Test: should_fuse returns MaybeFuse for small stencil *)
let test_should_fuse_small_stencil () =
  let producer =
    {
      kern_name = "producer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  (* Consumer reads temp[i-1], temp[i], temp[i+1] *)
  let consumer =
    {
      kern_name = "consumer";
      kern_params = [];
      kern_locals = [];
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
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
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
      kern_name = "k1";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("a", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  (* K2: b[i] = a[i] + 1 *)
  let k2 =
    {
      kern_name = "k2";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("b", thread_idx_x),
            EBinop (Add, EArrayRead ("a", thread_idx_x), EConst (CInt32 1l)) );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  (* K3: output[i] = b[i] * 3 *)
  let k3 =
    {
      kern_name = "k3";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop (Mul, EArrayRead ("b", thread_idx_x), EConst (CInt32 3l)) );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
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
      kern_name = "k1";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
    }
  in
  (* K2: output[i] = temp[i-1] + temp[i+1] (stencil) *)
  let k2 =
    {
      kern_name = "k2";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop
              ( Add,
                EArrayRead
                  ("temp", EBinop (Sub, thread_idx_x, EConst (CInt32 1l))),
                EArrayRead
                  ("temp", EBinop (Add, thread_idx_x, EConst (CInt32 1l))) ) );
      kern_types = [];
      kern_funcs = [];
      kern_native_fn = None;
      kern_variants = [];
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

let () =
  Printf.printf "=== Fusion Unit Tests ===\n" ;
  test_expr_equal () ;
  test_subst_array_read () ;
  test_analyze_one_to_one () ;
  test_analyze_with_barrier () ;
  test_can_fuse_compatible () ;
  test_can_fuse_with_barrier () ;
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
  Printf.printf "=== All tests passed! ===\n"
