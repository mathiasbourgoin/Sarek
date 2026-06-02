(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_convergence module
 *
 * Tests convergence analysis for barrier safety and dimension usage analysis
 ******************************************************************************)

(* Force stdlib initialization *)
let () = Sarek_stdlib.force_init ()

open Sarek_typed_ast

let dummy_loc =
  Sarek_ast.
    {
      loc_file = "test.ml";
      loc_line = 1;
      loc_col = 0;
      loc_end_line = 1;
      loc_end_col = 10;
    }

(* Helper to create texpr with type *)
let mk_texpr te typ = {te; ty = typ; te_loc = dummy_loc}

(* Helper types *)
let typ_int32 = Sarek_types.(TPrim TInt32)

let typ_bool = Sarek_types.(TPrim TBool)

let typ_unit = Sarek_types.(TPrim TUnit)

(** {1 Thread-Varying Detection Tests} *)

let test_is_thread_varying_literal () =
  let e = mk_texpr (TEInt 42) typ_int32 in
  let varying = Sarek_convergence.is_thread_varying e in
  Alcotest.(check bool) "literals are uniform" false varying

let test_is_thread_varying_thread_idx_x () =
  let e = mk_texpr (TEVar ("thread_idx_x", 0)) typ_int32 in
  let varying = Sarek_convergence.is_thread_varying e in
  Alcotest.(check bool) "thread_idx_x is varying" true varying

let test_is_thread_varying_thread_idx_y () =
  let e = mk_texpr (TEVar ("thread_idx_y", 0)) typ_int32 in
  let varying = Sarek_convergence.is_thread_varying e in
  Alcotest.(check bool) "thread_idx_y is varying" true varying

let test_is_thread_varying_global_idx () =
  let e = mk_texpr (TEVar ("global_idx_x", 0)) typ_int32 in
  let varying = Sarek_convergence.is_thread_varying e in
  Alcotest.(check bool) "global_idx_x is varying" true varying

let test_is_thread_varying_block_idx () =
  let e = mk_texpr (TEVar ("block_idx_x", 0)) typ_int32 in
  let varying = Sarek_convergence.is_thread_varying e in
  Alcotest.(check bool) "block_idx_x is uniform" false varying

let test_is_thread_varying_block_dim () =
  let e = mk_texpr (TEVar ("block_dim_x", 0)) typ_int32 in
  let varying = Sarek_convergence.is_thread_varying e in
  Alcotest.(check bool) "block_dim_x is uniform" false varying

let test_is_thread_varying_binop () =
  let tid_x = mk_texpr (TEVar ("thread_idx_x", 0)) typ_int32 in
  let const = mk_texpr (TEInt 16) typ_int32 in
  let e = mk_texpr (TEBinop (Sarek_ast.Gt, tid_x, const)) typ_bool in
  let varying = Sarek_convergence.is_thread_varying e in
  Alcotest.(check bool) "thread_idx_x > 16 is varying" true varying

let test_is_thread_varying_uniform_binop () =
  let block_x = mk_texpr (TEVar ("block_idx_x", 0)) typ_int32 in
  let const = mk_texpr (TEInt 2) typ_int32 in
  let e = mk_texpr (TEBinop (Sarek_ast.Mul, block_x, const)) typ_int32 in
  let varying = Sarek_convergence.is_thread_varying e in
  Alcotest.(check bool) "block_idx_x * 2 is uniform" false varying

let test_is_thread_varying_intrinsic_const () =
  let ref = Sarek_env.CorePrimitiveRef "thread_idx_x" in
  let e = mk_texpr (TEIntrinsicConst ref) typ_int32 in
  let varying = Sarek_convergence.is_thread_varying e in
  Alcotest.(check bool) "intrinsic thread_idx_x is varying" true varying

(** {1 Convergence Checking Tests} *)

let test_check_expr_simple_uniform () =
  let e = mk_texpr (TEInt 42) typ_int32 in
  let errors = Sarek_convergence.check_expr Sarek_convergence.init_ctx e in
  Alcotest.(check int) "no errors" 0 (List.length errors)

let test_check_expr_barrier_converged () =
  (* Barrier in converged mode should be allowed *)
  let ref = Sarek_env.CorePrimitiveRef "block_barrier" in
  let e =
    mk_texpr
      (TEIntrinsicFun (ref, Some Sarek_core_primitives.ConvergencePoint, []))
      typ_unit
  in
  let errors = Sarek_convergence.check_expr Sarek_convergence.init_ctx e in
  Alcotest.(check int)
    "no errors for barrier in converged"
    0
    (List.length errors)

let test_check_expr_barrier_diverged () =
  (* Barrier in diverged mode should fail *)
  let ref = Sarek_env.CorePrimitiveRef "block_barrier" in
  let e =
    mk_texpr
      (TEIntrinsicFun (ref, Some Sarek_core_primitives.ConvergencePoint, []))
      typ_unit
  in
  let ctx = Sarek_convergence.{mode = Diverged} in
  let errors = Sarek_convergence.check_expr ctx e in
  Alcotest.(check bool)
    "error for barrier in diverged"
    true
    (List.length errors > 0)

let test_check_expr_if_uniform_condition () =
  (* if block_idx_x > 0 then barrier() - should be allowed *)
  let block_x = mk_texpr (TEVar ("block_idx_x", 0)) typ_int32 in
  let zero = mk_texpr (TEInt 0) typ_int32 in
  let cond = mk_texpr (TEBinop (Sarek_ast.Gt, block_x, zero)) typ_bool in
  let ref = Sarek_env.CorePrimitiveRef "block_barrier" in
  let barrier =
    mk_texpr
      (TEIntrinsicFun (ref, Some Sarek_core_primitives.ConvergencePoint, []))
      typ_unit
  in
  let e = mk_texpr (TEIf (cond, barrier, None)) typ_unit in
  let errors = Sarek_convergence.check_expr Sarek_convergence.init_ctx e in
  Alcotest.(check int) "no errors for uniform if" 0 (List.length errors)

let test_check_expr_if_varying_condition_with_barrier () =
  (* if thread_idx_x > 16 then barrier() - should fail *)
  let tid_x = mk_texpr (TEVar ("thread_idx_x", 0)) typ_int32 in
  let const = mk_texpr (TEInt 16) typ_int32 in
  let cond = mk_texpr (TEBinop (Sarek_ast.Gt, tid_x, const)) typ_bool in
  let ref = Sarek_env.CorePrimitiveRef "block_barrier" in
  let barrier =
    mk_texpr
      (TEIntrinsicFun (ref, Some Sarek_core_primitives.ConvergencePoint, []))
      typ_unit
  in
  let e = mk_texpr (TEIf (cond, barrier, None)) typ_unit in
  let errors = Sarek_convergence.check_expr Sarek_convergence.init_ctx e in
  Alcotest.(check bool)
    "error for varying if with barrier"
    true
    (List.length errors > 0)

let test_check_expr_while_varying_with_barrier () =
  (* while thread_idx_x < N do barrier() - should fail *)
  let tid_x = mk_texpr (TEVar ("thread_idx_x", 0)) typ_int32 in
  let n = mk_texpr (TEInt 100) typ_int32 in
  let cond = mk_texpr (TEBinop (Sarek_ast.Lt, tid_x, n)) typ_bool in
  let ref = Sarek_env.CorePrimitiveRef "block_barrier" in
  let barrier =
    mk_texpr
      (TEIntrinsicFun (ref, Some Sarek_core_primitives.ConvergencePoint, []))
      typ_unit
  in
  let e = mk_texpr (TEWhile (cond, barrier)) typ_unit in
  let errors = Sarek_convergence.check_expr Sarek_convergence.init_ctx e in
  Alcotest.(check bool)
    "error for varying while with barrier"
    true
    (List.length errors > 0)

let test_check_expr_for_varying_bounds_with_barrier () =
  (* for i = thread_idx_x to N do barrier() - should fail *)
  let tid_x = mk_texpr (TEVar ("thread_idx_x", 0)) typ_int32 in
  let n = mk_texpr (TEInt 100) typ_int32 in
  let ref = Sarek_env.CorePrimitiveRef "block_barrier" in
  let barrier =
    mk_texpr
      (TEIntrinsicFun (ref, Some Sarek_core_primitives.ConvergencePoint, []))
      typ_unit
  in
  let e =
    mk_texpr (TEFor ("i", 0, tid_x, n, Sarek_ast.Upto, barrier)) typ_unit
  in
  let errors = Sarek_convergence.check_expr Sarek_convergence.init_ctx e in
  Alcotest.(check bool)
    "error for varying for with barrier"
    true
    (List.length errors > 0)

let test_check_expr_warp_collective_diverged () =
  (* Warp-level collective in diverged flow should fail *)
  let ref = Sarek_env.CorePrimitiveRef "warp_shuffle" in
  let arg = mk_texpr (TEInt 0) typ_int32 in
  let e =
    mk_texpr
      (TEIntrinsicFun (ref, Some Sarek_core_primitives.WarpConvergence, [arg]))
      typ_int32
  in
  let ctx = Sarek_convergence.{mode = Diverged} in
  let errors = Sarek_convergence.check_expr ctx e in
  Alcotest.(check bool)
    "error for warp collective in diverged"
    true
    (List.length errors > 0)

(** {1 Barrier Detection Tests} *)

let test_expr_uses_barriers_simple () =
  let e = mk_texpr (TEInt 42) typ_int32 in
  let uses = Sarek_convergence.expr_uses_barriers e in
  Alcotest.(check bool) "simple expr has no barriers" false uses

let test_expr_uses_barriers_explicit () =
  let ref = Sarek_env.CorePrimitiveRef "block_barrier" in
  let e =
    mk_texpr
      (TEIntrinsicFun (ref, Some Sarek_core_primitives.ConvergencePoint, []))
      typ_unit
  in
  let uses = Sarek_convergence.expr_uses_barriers e in
  Alcotest.(check bool) "explicit barrier detected" true uses

let test_expr_uses_barriers_superstep () =
  let step = mk_texpr (TEInt 1) typ_int32 in
  let cont = mk_texpr (TEInt 2) typ_int32 in
  let e = mk_texpr (TESuperstep ("test", false, step, cont)) typ_int32 in
  let uses = Sarek_convergence.expr_uses_barriers e in
  Alcotest.(check bool) "superstep has implicit barrier" true uses

let test_expr_uses_barriers_nested () =
  (* if cond then barrier() *)
  let cond = mk_texpr (TEBool true) typ_bool in
  let ref = Sarek_env.CorePrimitiveRef "block_barrier" in
  let barrier =
    mk_texpr
      (TEIntrinsicFun (ref, Some Sarek_core_primitives.ConvergencePoint, []))
      typ_unit
  in
  let e = mk_texpr (TEIf (cond, barrier, None)) typ_unit in
  let uses = Sarek_convergence.expr_uses_barriers e in
  Alcotest.(check bool) "nested barrier detected" true uses

(** {1 Kernel Analysis Tests} *)

let test_check_kernel_no_barriers () =
  let body = mk_texpr (TEInt 42) typ_int32 in
  let kernel =
    {
      tkern_name = Some "test";
      tkern_params = [];
      tkern_type_decls = [];
      tkern_module_items = [];
      tkern_external_item_count = 0;
      tkern_body = body;
      tkern_return_type = typ_int32;
      tkern_loc = dummy_loc;
    }
  in
  let result = Sarek_convergence.check_kernel kernel in
  match result with
  | Ok () -> ()
  | Error _ -> Alcotest.fail "kernel without barriers should pass"

let test_check_kernel_with_safe_barrier () =
  (* Barrier in converged flow should be safe *)
  let ref = Sarek_env.CorePrimitiveRef "block_barrier" in
  let body =
    mk_texpr
      (TEIntrinsicFun (ref, Some Sarek_core_primitives.ConvergencePoint, []))
      typ_unit
  in
  let kernel =
    {
      tkern_name = Some "test";
      tkern_params = [];
      tkern_type_decls = [];
      tkern_module_items = [];
      tkern_external_item_count = 0;
      tkern_body = body;
      tkern_return_type = typ_unit;
      tkern_loc = dummy_loc;
    }
  in
  let result = Sarek_convergence.check_kernel kernel in
  match result with
  | Ok () -> ()
  | Error _ -> Alcotest.fail "kernel with safe barrier should pass"

let test_check_kernel_with_diverged_barrier () =
  (* if thread_idx_x > 16 then barrier() should fail *)
  let tid_x = mk_texpr (TEVar ("thread_idx_x", 0)) typ_int32 in
  let const = mk_texpr (TEInt 16) typ_int32 in
  let cond = mk_texpr (TEBinop (Sarek_ast.Gt, tid_x, const)) typ_bool in
  let ref = Sarek_env.CorePrimitiveRef "block_barrier" in
  let barrier =
    mk_texpr
      (TEIntrinsicFun (ref, Some Sarek_core_primitives.ConvergencePoint, []))
      typ_unit
  in
  let body = mk_texpr (TEIf (cond, barrier, None)) typ_unit in
  let kernel =
    {
      tkern_name = Some "test";
      tkern_params = [];
      tkern_type_decls = [];
      tkern_module_items = [];
      tkern_external_item_count = 0;
      tkern_body = body;
      tkern_return_type = typ_unit;
      tkern_loc = dummy_loc;
    }
  in
  let result = Sarek_convergence.check_kernel kernel in
  match result with
  | Ok () -> Alcotest.fail "kernel with diverged barrier should fail"
  | Error errs -> Alcotest.(check bool) "has errors" true (List.length errs > 0)

let test_kernel_uses_barriers_true () =
  let ref = Sarek_env.CorePrimitiveRef "block_barrier" in
  let body =
    mk_texpr
      (TEIntrinsicFun (ref, Some Sarek_core_primitives.ConvergencePoint, []))
      typ_unit
  in
  let kernel =
    {
      tkern_name = Some "test";
      tkern_params = [];
      tkern_type_decls = [];
      tkern_module_items = [];
      tkern_external_item_count = 0;
      tkern_body = body;
      tkern_return_type = typ_unit;
      tkern_loc = dummy_loc;
    }
  in
  let uses = Sarek_convergence.kernel_uses_barriers kernel in
  Alcotest.(check bool) "kernel uses barriers" true uses

let test_kernel_uses_barriers_false () =
  let body = mk_texpr (TEInt 42) typ_int32 in
  let kernel =
    {
      tkern_name = Some "test";
      tkern_params = [];
      tkern_type_decls = [];
      tkern_module_items = [];
      tkern_external_item_count = 0;
      tkern_body = body;
      tkern_return_type = typ_int32;
      tkern_loc = dummy_loc;
    }
  in
  let uses = Sarek_convergence.kernel_uses_barriers kernel in
  Alcotest.(check bool) "kernel doesn't use barriers" false uses

(** {1 Dimension Usage Tests} *)

let test_dim_usage_empty () =
  let e = mk_texpr (TEInt 42) typ_int32 in
  let usage = Sarek_convergence.expr_dim_usage e in
  Alcotest.(check bool) "no x usage" false usage.Sarek_convergence.uses_x ;
  Alcotest.(check bool) "no y usage" false usage.uses_y ;
  Alcotest.(check bool) "no z usage" false usage.uses_z

let test_dim_usage_global_idx_x () =
  let e = mk_texpr (TEVar ("global_idx_x", 0)) typ_int32 in
  let usage = Sarek_convergence.expr_dim_usage e in
  Alcotest.(check bool) "uses x" true usage.Sarek_convergence.uses_x ;
  Alcotest.(check bool) "no y usage" false usage.uses_y ;
  Alcotest.(check bool) "no z usage" false usage.uses_z

let test_dim_usage_global_idx_y () =
  let e = mk_texpr (TEVar ("global_idx_y", 0)) typ_int32 in
  let usage = Sarek_convergence.expr_dim_usage e in
  Alcotest.(check bool) "uses x" true usage.Sarek_convergence.uses_x ;
  Alcotest.(check bool) "uses y" true usage.uses_y ;
  Alcotest.(check bool) "no z usage" false usage.uses_z

let test_dim_usage_global_idx_z () =
  let e = mk_texpr (TEVar ("global_idx_z", 0)) typ_int32 in
  let usage = Sarek_convergence.expr_dim_usage e in
  Alcotest.(check bool) "uses x" true usage.Sarek_convergence.uses_x ;
  Alcotest.(check bool) "uses y" true usage.uses_y ;
  Alcotest.(check bool) "uses z" true usage.uses_z

let test_dim_usage_thread_idx () =
  let e = mk_texpr (TEVar ("thread_idx_x", 0)) typ_int32 in
  let usage = Sarek_convergence.expr_dim_usage e in
  Alcotest.(check bool) "uses x" true usage.Sarek_convergence.uses_x ;
  Alcotest.(check bool) "uses thread_idx" true usage.uses_thread_idx

let test_dim_usage_block_idx () =
  let e = mk_texpr (TEVar ("block_idx_x", 0)) typ_int32 in
  let usage = Sarek_convergence.expr_dim_usage e in
  Alcotest.(check bool) "uses x" true usage.Sarek_convergence.uses_x ;
  Alcotest.(check bool) "uses block_idx" true usage.uses_block_idx

let test_dim_usage_block_dim () =
  let e = mk_texpr (TEVar ("block_dim_x", 0)) typ_int32 in
  let usage = Sarek_convergence.expr_dim_usage e in
  Alcotest.(check bool) "uses x" true usage.Sarek_convergence.uses_x ;
  Alcotest.(check bool) "uses block_dim" true usage.uses_block_dim

let test_dim_usage_grid_dim () =
  let e = mk_texpr (TEVar ("grid_dim_x", 0)) typ_int32 in
  let usage = Sarek_convergence.expr_dim_usage e in
  Alcotest.(check bool) "uses x" true usage.Sarek_convergence.uses_x ;
  Alcotest.(check bool) "uses grid_dim" true usage.uses_grid_dim

let test_dim_usage_shared_mem () =
  let size = mk_texpr (TEInt 256) typ_int32 in
  let body = mk_texpr TEUnit typ_unit in
  let e =
    mk_texpr (TELetShared ("smem", 0, typ_int32, Some size, body)) typ_unit
  in
  let usage = Sarek_convergence.expr_dim_usage e in
  Alcotest.(check bool)
    "uses shared mem"
    true
    usage.Sarek_convergence.uses_shared_mem

let test_dim_usage_merge () =
  (* global_idx_x + global_idx_y *)
  let x = mk_texpr (TEVar ("global_idx_x", 0)) typ_int32 in
  let y = mk_texpr (TEVar ("global_idx_y", 0)) typ_int32 in
  let e = mk_texpr (TEBinop (Sarek_ast.Add, x, y)) typ_int32 in
  let usage = Sarek_convergence.expr_dim_usage e in
  Alcotest.(check bool) "uses x" true usage.Sarek_convergence.uses_x ;
  Alcotest.(check bool) "uses y" true usage.uses_y ;
  Alcotest.(check bool) "no z usage" false usage.uses_z

(** {1 Execution Strategy Tests} *)

let test_exec_strategy_simple_1d () =
  (* Kernel only uses global_idx_x *)
  let e = mk_texpr (TEVar ("global_idx_x", 0)) typ_int32 in
  let kernel =
    {
      tkern_name = Some "test";
      tkern_params = [];
      tkern_type_decls = [];
      tkern_module_items = [];
      tkern_external_item_count = 0;
      tkern_body = e;
      tkern_return_type = typ_int32;
      tkern_loc = dummy_loc;
    }
  in
  let strategy = Sarek_convergence.kernel_exec_strategy kernel in
  match strategy with
  | Sarek_convergence.Simple1D -> ()
  | _ -> Alcotest.fail "expected Simple1D strategy"

let test_exec_strategy_simple_2d () =
  (* Kernel uses global_idx_x and global_idx_y *)
  let x = mk_texpr (TEVar ("global_idx_x", 0)) typ_int32 in
  let y = mk_texpr (TEVar ("global_idx_y", 0)) typ_int32 in
  let e = mk_texpr (TEBinop (Sarek_ast.Add, x, y)) typ_int32 in
  let kernel =
    {
      tkern_name = Some "test";
      tkern_params = [];
      tkern_type_decls = [];
      tkern_module_items = [];
      tkern_external_item_count = 0;
      tkern_body = e;
      tkern_return_type = typ_int32;
      tkern_loc = dummy_loc;
    }
  in
  let strategy = Sarek_convergence.kernel_exec_strategy kernel in
  match strategy with
  | Sarek_convergence.Simple2D -> ()
  | _ -> Alcotest.fail "expected Simple2D strategy"

let test_exec_strategy_simple_3d () =
  (* Kernel uses all three dimensions *)
  let x = mk_texpr (TEVar ("global_idx_x", 0)) typ_int32 in
  let y = mk_texpr (TEVar ("global_idx_y", 0)) typ_int32 in
  let z = mk_texpr (TEVar ("global_idx_z", 0)) typ_int32 in
  let xy = mk_texpr (TEBinop (Sarek_ast.Add, x, y)) typ_int32 in
  let e = mk_texpr (TEBinop (Sarek_ast.Add, xy, z)) typ_int32 in
  let kernel =
    {
      tkern_name = Some "test";
      tkern_params = [];
      tkern_type_decls = [];
      tkern_module_items = [];
      tkern_external_item_count = 0;
      tkern_body = e;
      tkern_return_type = typ_int32;
      tkern_loc = dummy_loc;
    }
  in
  let strategy = Sarek_convergence.kernel_exec_strategy kernel in
  match strategy with
  | Sarek_convergence.Simple3D -> ()
  | _ -> Alcotest.fail "expected Simple3D strategy"

let test_exec_strategy_full_state_thread_idx () =
  (* Kernel uses thread_idx_x - needs full state *)
  let e = mk_texpr (TEVar ("thread_idx_x", 0)) typ_int32 in
  let kernel =
    {
      tkern_name = Some "test";
      tkern_params = [];
      tkern_type_decls = [];
      tkern_module_items = [];
      tkern_external_item_count = 0;
      tkern_body = e;
      tkern_return_type = typ_int32;
      tkern_loc = dummy_loc;
    }
  in
  let strategy = Sarek_convergence.kernel_exec_strategy kernel in
  match strategy with
  | Sarek_convergence.FullState -> ()
  | _ -> Alcotest.fail "expected FullState strategy"

let test_exec_strategy_full_state_block_idx () =
  (* Kernel uses block_idx_x - needs full state *)
  let e = mk_texpr (TEVar ("block_idx_x", 0)) typ_int32 in
  let kernel =
    {
      tkern_name = Some "test";
      tkern_params = [];
      tkern_type_decls = [];
      tkern_module_items = [];
      tkern_external_item_count = 0;
      tkern_body = e;
      tkern_return_type = typ_int32;
      tkern_loc = dummy_loc;
    }
  in
  let strategy = Sarek_convergence.kernel_exec_strategy kernel in
  match strategy with
  | Sarek_convergence.FullState -> ()
  | _ -> Alcotest.fail "expected FullState strategy"

let test_exec_strategy_full_state_shared_mem () =
  (* Kernel uses shared memory - needs full state *)
  let size = mk_texpr (TEInt 256) typ_int32 in
  let body = mk_texpr TEUnit typ_unit in
  let e =
    mk_texpr (TELetShared ("smem", 0, typ_int32, Some size, body)) typ_unit
  in
  let kernel =
    {
      tkern_name = Some "test";
      tkern_params = [];
      tkern_type_decls = [];
      tkern_module_items = [];
      tkern_external_item_count = 0;
      tkern_body = e;
      tkern_return_type = typ_unit;
      tkern_loc = dummy_loc;
    }
  in
  let strategy = Sarek_convergence.kernel_exec_strategy kernel in
  match strategy with
  | Sarek_convergence.FullState -> ()
  | _ -> Alcotest.fail "expected FullState strategy"

let test_exec_strategy_full_state_barriers () =
  (* Kernel uses barriers - needs full state *)
  let ref = Sarek_env.CorePrimitiveRef "block_barrier" in
  let e =
    mk_texpr
      (TEIntrinsicFun (ref, Some Sarek_core_primitives.ConvergencePoint, []))
      typ_unit
  in
  let kernel =
    {
      tkern_name = Some "test";
      tkern_params = [];
      tkern_type_decls = [];
      tkern_module_items = [];
      tkern_external_item_count = 0;
      tkern_body = e;
      tkern_return_type = typ_unit;
      tkern_loc = dummy_loc;
    }
  in
  let strategy = Sarek_convergence.kernel_exec_strategy kernel in
  match strategy with
  | Sarek_convergence.FullState -> ()
  | _ -> Alcotest.fail "expected FullState strategy for kernel with barriers"

(** {1 Superstep Analysis Tests} *)

let test_superstep_non_divergent_safe () =
  (* Non-divergent superstep with uniform code should be safe *)
  let step = mk_texpr (TEInt 1) typ_int32 in
  let cont = mk_texpr (TEInt 2) typ_int32 in
  let e = mk_texpr (TESuperstep ("test", false, step, cont)) typ_int32 in
  let errors = Sarek_convergence.check_expr Sarek_convergence.init_ctx e in
  Alcotest.(check int) "no errors for uniform superstep" 0 (List.length errors)

let test_superstep_divergent_with_control_flow () =
  (* Divergent superstep explicitly marked - should allow varying control flow *)
  let tid_x = mk_texpr (TEVar ("thread_idx_x", 0)) typ_int32 in
  let const = mk_texpr (TEInt 16) typ_int32 in
  let cond = mk_texpr (TEBinop (Sarek_ast.Gt, tid_x, const)) typ_bool in
  let then_e = mk_texpr (TEInt 1) typ_int32 in
  let step = mk_texpr (TEIf (cond, then_e, None)) typ_int32 in
  let cont = mk_texpr (TEInt 2) typ_int32 in
  let e = mk_texpr (TESuperstep ("test", true, step, cont)) typ_int32 in
  let errors = Sarek_convergence.check_expr Sarek_convergence.init_ctx e in
  Alcotest.(check int)
    "no errors for marked divergent superstep"
    0
    (List.length errors)

let test_superstep_non_divergent_with_varying_control () =
  (* Non-divergent superstep with varying control flow should fail *)
  let tid_x = mk_texpr (TEVar ("thread_idx_x", 0)) typ_int32 in
  let const = mk_texpr (TEInt 16) typ_int32 in
  let cond = mk_texpr (TEBinop (Sarek_ast.Gt, tid_x, const)) typ_bool in
  let then_e = mk_texpr (TEInt 1) typ_int32 in
  let step = mk_texpr (TEIf (cond, then_e, None)) typ_int32 in
  let cont = mk_texpr (TEInt 2) typ_int32 in
  let e = mk_texpr (TESuperstep ("test", false, step, cont)) typ_int32 in
  let errors = Sarek_convergence.check_expr Sarek_convergence.init_ctx e in
  Alcotest.(check bool)
    "error for non-divergent superstep with varying flow"
    true
    (List.length errors > 0)

(** {1 Test Suite} *)

let () =
  let open Alcotest in
  run
    "Sarek_convergence"
    [
      ( "is_thread_varying",
        [
          test_case "literal" `Quick test_is_thread_varying_literal;
          test_case "thread_idx_x" `Quick test_is_thread_varying_thread_idx_x;
          test_case "thread_idx_y" `Quick test_is_thread_varying_thread_idx_y;
          test_case "global_idx" `Quick test_is_thread_varying_global_idx;
          test_case "block_idx" `Quick test_is_thread_varying_block_idx;
          test_case "block_dim" `Quick test_is_thread_varying_block_dim;
          test_case "binop" `Quick test_is_thread_varying_binop;
          test_case "uniform binop" `Quick test_is_thread_varying_uniform_binop;
          test_case
            "intrinsic const"
            `Quick
            test_is_thread_varying_intrinsic_const;
        ] );
      ( "check_expr",
        [
          test_case "simple uniform" `Quick test_check_expr_simple_uniform;
          test_case "barrier converged" `Quick test_check_expr_barrier_converged;
          test_case "barrier diverged" `Quick test_check_expr_barrier_diverged;
          test_case
            "if uniform with barrier"
            `Quick
            test_check_expr_if_uniform_condition;
          test_case
            "if varying with barrier"
            `Quick
            test_check_expr_if_varying_condition_with_barrier;
          test_case
            "while varying with barrier"
            `Quick
            test_check_expr_while_varying_with_barrier;
          test_case
            "for varying with barrier"
            `Quick
            test_check_expr_for_varying_bounds_with_barrier;
          test_case
            "warp collective diverged"
            `Quick
            test_check_expr_warp_collective_diverged;
        ] );
      ( "barrier_detection",
        [
          test_case "simple expr" `Quick test_expr_uses_barriers_simple;
          test_case "explicit barrier" `Quick test_expr_uses_barriers_explicit;
          test_case "superstep" `Quick test_expr_uses_barriers_superstep;
          test_case "nested barrier" `Quick test_expr_uses_barriers_nested;
        ] );
      ( "kernel_analysis",
        [
          test_case "no barriers" `Quick test_check_kernel_no_barriers;
          test_case "safe barrier" `Quick test_check_kernel_with_safe_barrier;
          test_case
            "diverged barrier"
            `Quick
            test_check_kernel_with_diverged_barrier;
          test_case "uses barriers true" `Quick test_kernel_uses_barriers_true;
          test_case "uses barriers false" `Quick test_kernel_uses_barriers_false;
        ] );
      ( "dim_usage",
        [
          test_case "empty" `Quick test_dim_usage_empty;
          test_case "global_idx_x" `Quick test_dim_usage_global_idx_x;
          test_case "global_idx_y" `Quick test_dim_usage_global_idx_y;
          test_case "global_idx_z" `Quick test_dim_usage_global_idx_z;
          test_case "thread_idx" `Quick test_dim_usage_thread_idx;
          test_case "block_idx" `Quick test_dim_usage_block_idx;
          test_case "block_dim" `Quick test_dim_usage_block_dim;
          test_case "grid_dim" `Quick test_dim_usage_grid_dim;
          test_case "shared_mem" `Quick test_dim_usage_shared_mem;
          test_case "merge" `Quick test_dim_usage_merge;
        ] );
      ( "exec_strategy",
        [
          test_case "Simple1D" `Quick test_exec_strategy_simple_1d;
          test_case "Simple2D" `Quick test_exec_strategy_simple_2d;
          test_case "Simple3D" `Quick test_exec_strategy_simple_3d;
          test_case
            "FullState thread_idx"
            `Quick
            test_exec_strategy_full_state_thread_idx;
          test_case
            "FullState block_idx"
            `Quick
            test_exec_strategy_full_state_block_idx;
          test_case
            "FullState shared_mem"
            `Quick
            test_exec_strategy_full_state_shared_mem;
          test_case
            "FullState barriers"
            `Quick
            test_exec_strategy_full_state_barriers;
        ] );
      ( "superstep",
        [
          test_case "non-divergent safe" `Quick test_superstep_non_divergent_safe;
          test_case
            "divergent with control flow"
            `Quick
            test_superstep_divergent_with_control_flow;
          test_case
            "non-divergent with varying"
            `Quick
            test_superstep_non_divergent_with_varying_control;
        ] );
    ]
