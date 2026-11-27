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
      loc_bol = 0;
      loc_end_bol = 0;
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
  let ctx = {Sarek_convergence.init_ctx with mode = Diverged} in
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
  let ctx = {Sarek_convergence.init_ctx with mode = Diverged} in
  let errors = Sarek_convergence.check_expr ctx e in
  Alcotest.(check bool)
    "error for warp collective in diverged"
    true
    (List.length errors > 0)

(* A warp shuffle as the typer builds it: intrinsic call with the
   WarpConvergence class. *)
let shuffle_call arg =
  mk_texpr
    (TEIntrinsicFun
       ( Sarek_env.CorePrimitiveRef "warp_shuffle_xor",
         Some Sarek_core_primitives.WarpConvergence,
         [arg; mk_texpr (TEInt 1) typ_int32] ))
    typ_int32

let varying_cond =
  let tid_x = mk_texpr (TEVar ("thread_idx_x", 0)) typ_int32 in
  mk_texpr
    (TEBinop (Sarek_ast.Gt, tid_x, mk_texpr (TEInt 16) typ_int32))
    typ_bool

let check_shuffle_refused msg e =
  match Sarek_convergence.check_expr Sarek_convergence.init_ctx e with
  | [Sarek_error.Warp_collective_in_diverged_flow ("warp_shuffle_xor", _)] -> ()
  | errors ->
      Alcotest.failf
        "%s: expected one Warp_collective_in_diverged_flow, got [%s]"
        msg
        (String.concat "; " (List.map Sarek_error.error_to_string errors))

let test_shuffle_after_uniform_if () =
  (* if block_idx_x > 0 then shuffle: a block-uniform condition keeps the flow
     converged *)
  let block_x = mk_texpr (TEVar ("block_idx_x", 0)) typ_int32 in
  let cond =
    mk_texpr
      (TEBinop (Sarek_ast.Gt, block_x, mk_texpr (TEInt 0) typ_int32))
      typ_bool
  in
  let e =
    mk_texpr
      (TEIf (cond, shuffle_call (mk_texpr (TEInt 7) typ_int32), None))
      typ_int32
  in
  Alcotest.(check int)
    "accepted"
    0
    (List.length (Sarek_convergence.check_expr Sarek_convergence.init_ctx e))

let test_shuffle_in_varying_if () =
  check_shuffle_refused
    "varying if"
    (mk_texpr
       (TEIf (varying_cond, shuffle_call (mk_texpr (TEInt 7) typ_int32), None))
       typ_int32)

let test_shuffle_in_varying_loop () =
  (* for i = thread_idx_x to 100 do shuffle: the trip count differs per lane *)
  let tid_x = mk_texpr (TEVar ("thread_idx_x", 0)) typ_int32 in
  let n = mk_texpr (TEInt 100) typ_int32 in
  check_shuffle_refused
    "varying for bound"
    (mk_texpr
       (TEFor
          ( "i",
            0,
            tid_x,
            n,
            Sarek_ast.Upto,
            shuffle_call (mk_texpr (TEInt 7) typ_int32) ))
       typ_int32) ;
  check_shuffle_refused
    "varying while"
    (mk_texpr
       (TEWhile (varying_cond, shuffle_call (mk_texpr (TEInt 7) typ_int32)))
       typ_int32)

let test_shuffle_after_varying_return () =
  (* F-04: a varying early return makes the rest of the sequence diverged *)
  let early_return =
    mk_texpr
      (TEIf
         ( varying_cond,
           mk_texpr (TEReturn (mk_texpr TEUnit typ_unit)) typ_unit,
           None ))
      typ_unit
  in
  check_shuffle_refused
    "after a varying return"
    (mk_texpr
       (TESeq [early_return; shuffle_call (mk_texpr (TEInt 7) typ_int32)])
       typ_int32)

let test_shuffle_in_helper_called_from_diverged () =
  (* let rec h x = shuffle x in if varying then h 7: rule H-2 *)
  let x = mk_texpr (TEVar ("x", 1)) typ_int32 in
  let helper_ty = Sarek_types.TFun ([typ_int32], typ_int32) in
  let call =
    mk_texpr
      (TEApp
         (mk_texpr (TEVar ("h", 2)) helper_ty, [mk_texpr (TEInt 7) typ_int32]))
      typ_int32
  in
  let param =
    {
      tparam_name = "x";
      tparam_type = typ_int32;
      tparam_index = 0;
      tparam_is_vec = false;
      tparam_id = 1;
    }
  in
  let body_in_varying = mk_texpr (TEIf (varying_cond, call, None)) typ_int32 in
  let e =
    mk_texpr
      (TELetRec ("h", 2, [param], shuffle_call x, body_in_varying))
      typ_int32
  in
  check_shuffle_refused "helper called from diverged flow" e ;
  (* The same helper called from converged flow is accepted. *)
  let ok =
    mk_texpr (TELetRec ("h", 2, [param], shuffle_call x, call)) typ_int32
  in
  Alcotest.(check int)
    "helper called from converged flow"
    0
    (List.length (Sarek_convergence.check_expr Sarek_convergence.init_ctx ok))

(** {2 Strict analysis of warp collectives}

    check_toplevel_expr adds the strict analysis to the legacy one for warp
    collectives: thread-varying helper parameters, thread-dependent helper
    results, mutables assigned in divergence, the value of an if. *)

let var name = mk_texpr (TEVar (name, 9)) typ_int32

let int n = mk_texpr (TEInt n) typ_int32

let tid_x = mk_texpr (TEVar ("thread_idx_x", 0)) typ_int32

let gt a b = mk_texpr (TEBinop (Sarek_ast.Gt, a, b)) typ_bool

let tparam name id =
  {
    tparam_name = name;
    tparam_type = typ_int32;
    tparam_index = 0;
    tparam_is_vec = false;
    tparam_id = id;
  }

let helper_ty = Sarek_types.TFun ([typ_int32], typ_int32)

let call name args =
  mk_texpr (TEApp (mk_texpr (TEVar (name, 2)) helper_ty, args)) typ_int32

let if_ c t = mk_texpr (TEIf (c, t, None)) typ_unit

let seq es = mk_texpr (TESeq es) typ_unit

let let_rec name params fn_body cont =
  mk_texpr (TELetRec (name, 2, params, fn_body, cont)) typ_unit

let check_all e = Sarek_convergence.check_toplevel_expr e

let refused msg e =
  match check_all e with
  | [Sarek_error.Warp_collective_in_diverged_flow ("warp_shuffle_xor", _)] -> ()
  | errors ->
      Alcotest.failf
        "%s: expected one Warp_collective_in_diverged_flow, got [%s]"
        msg
        (String.concat "; " (List.map Sarek_error.error_to_string errors))

let accepted msg e = Alcotest.(check int) msg 0 (List.length (check_all e))

let test_strict_helper_param () =
  (* let gate n x = if n > 16 then shuffle x in gate thread_idx_x 7 *)
  let body = if_ (gt (var "n") (int 16)) (shuffle_call (var "x")) in
  let params = [tparam "n" 3; tparam "x" 4] in
  refused
    "varying argument"
    (let_rec "gate" params body (call "gate" [tid_x; int 7])) ;
  accepted
    "uniform argument"
    (let_rec "gate" params body (call "gate" [int 20; int 7])) ;
  (* a varying argument that only feeds the shuffled value *)
  accepted
    "varying value argument"
    (let_rec "red" [tparam "x" 4] (shuffle_call (var "x")) (call "red" [tid_x]))

let test_strict_helper_result () =
  (* let me k = thread_idx_x + k in if me 0 > 16 then shuffle *)
  let me_body = mk_texpr (TEBinop (Sarek_ast.Add, tid_x, var "k")) typ_int32 in
  let uniform_body =
    mk_texpr (TEBinop (Sarek_ast.Add, int 1, var "k")) typ_int32
  in
  let use = if_ (gt (call "me" [int 0]) (int 16)) (shuffle_call (int 7)) in
  refused "thread-dependent result" (let_rec "me" [tparam "k" 3] me_body use) ;
  accepted "uniform result" (let_rec "me" [tparam "k" 3] uniform_body use)

let test_strict_mutable () =
  (* let m = mut 0 in if thread_idx_x > 16 then m := 1; if m > 0 then shuffle *)
  let assign v = mk_texpr (TEAssign ("m", 5, v)) typ_unit in
  let guarded_shuffle = if_ (gt (var "m") (int 0)) (shuffle_call (int 7)) in
  let let_m body = mk_texpr (TELetMut ("m", 5, int 0, body)) typ_unit in
  refused
    "assigned under a varying condition"
    (let_m (seq [if_ (gt tid_x (int 16)) (assign (int 1)); guarded_shuffle])) ;
  refused
    "assigned a varying value"
    (let_m (seq [assign tid_x; guarded_shuffle])) ;
  accepted
    "assigned a uniform value in uniform flow"
    (let_m (seq [assign (int 3); guarded_shuffle]))

let test_strict_if_value () =
  (* let c = if thread_idx_x > 16 then 1 else 0 in if c > 0 then shuffle *)
  let value =
    mk_texpr (TEIf (gt tid_x (int 16), int 1, Some (int 0))) typ_int32
  in
  refused
    "value of a varying if"
    (mk_texpr
       (TELet ("c", 6, value, if_ (gt (var "c") (int 0)) (shuffle_call (int 7))))
       typ_unit)

(** {2 Strict analysis of barriers}

    The same four cases as for warp collectives: a deadlock instead of an
    undefined result, but the same missed divergence. Each refused case has an
    accepted twin so the rule does not reject uniform code. *)

let barrier_call =
  mk_texpr
    (TEIntrinsicFun
       ( Sarek_env.CorePrimitiveRef "block_barrier",
         Some Sarek_core_primitives.ConvergencePoint,
         [] ))
    typ_unit

let barrier_refused msg e =
  match check_all e with
  | [Sarek_error.Barrier_in_diverged_flow _] -> ()
  | errors ->
      Alcotest.failf
        "%s: expected one Barrier_in_diverged_flow, got [%s]"
        msg
        (String.concat "; " (List.map Sarek_error.error_to_string errors))

let test_strict_barrier_helper_param () =
  (* let gate n = if n > 16 then barrier in gate thread_idx_x *)
  let body = if_ (gt (var "n") (int 16)) barrier_call in
  let params = [tparam "n" 3] in
  barrier_refused
    "varying argument"
    (let_rec "gate" params body (call "gate" [tid_x])) ;
  accepted
    "uniform argument"
    (let_rec "gate" params body (call "gate" [int 20])) ;
  (* a varying argument that does not steer the barrier *)
  accepted
    "varying argument not steering the barrier"
    (let_rec "sync" params (seq [barrier_call; var "n"]) (call "sync" [tid_x]))

let test_strict_barrier_helper_result () =
  (* let me k = thread_idx_x + k in if me 0 > 16 then barrier *)
  let me_body = mk_texpr (TEBinop (Sarek_ast.Add, tid_x, var "k")) typ_int32 in
  let uniform_body =
    mk_texpr (TEBinop (Sarek_ast.Add, int 1, var "k")) typ_int32
  in
  let use = if_ (gt (call "me" [int 0]) (int 16)) barrier_call in
  barrier_refused
    "thread-dependent result"
    (let_rec "me" [tparam "k" 3] me_body use) ;
  accepted "uniform result" (let_rec "me" [tparam "k" 3] uniform_body use)

let test_strict_barrier_mutable () =
  (* let m = mut 0 in if thread_idx_x > 16 then m := 1; if m > 0 then barrier *)
  let assign v = mk_texpr (TEAssign ("m", 5, v)) typ_unit in
  let guarded = if_ (gt (var "m") (int 0)) barrier_call in
  let let_m body = mk_texpr (TELetMut ("m", 5, int 0, body)) typ_unit in
  barrier_refused
    "assigned under a varying condition"
    (let_m (seq [if_ (gt tid_x (int 16)) (assign (int 1)); guarded])) ;
  barrier_refused
    "assigned a varying value"
    (let_m (seq [assign tid_x; guarded])) ;
  accepted
    "assigned a uniform value in uniform flow"
    (let_m (seq [assign (int 3); guarded])) ;
  accepted
    "varying mutable not steering the barrier"
    (let_m
       (seq
          [
            if_ (gt tid_x (int 16)) (assign (int 1));
            barrier_call;
            if_ (gt (var "m") (int 0)) (assign (int 2));
          ]))

(** {2 Nested mutables (#558)}

    The strict analysis of a mutable was repeated for every subset of the
    mutables enclosing it, so the check took time exponential in their number
    (about 12 s for 16 on the machine that measured it, minutes for 18). [count]
    constant-initialised mutables [s0 .. s(count-1)], each assigned in the
    innermost body together with a barrier. *)
let nested_name k = Printf.sprintf "s%d" k

let nested_id k = 200 + k

let add_expr a b = mk_texpr (TEBinop (Sarek_ast.Add, a, b)) typ_int32

(** [assign_rhs k] is what [s_k] is incremented by; [prelude] runs first in the
    innermost body; [outer] wraps the whole nest (a mutable or a let). *)
let nested_mutables ~count ~assign_rhs ~prelude ~last ~outer =
  let assign k =
    mk_texpr
      (TEAssign
         ( nested_name k,
           nested_id k,
           add_expr (var (nested_name k)) (assign_rhs k) ))
      typ_unit
  in
  let innermost =
    seq ((barrier_call :: prelude) @ List.init count assign @ last)
  in
  let rec wrap k body =
    if k < 0 then body
    else
      wrap
        (k - 1)
        (mk_texpr (TELetMut (nested_name k, nested_id k, int 0, body)) typ_unit)
  in
  outer (wrap (count - 1) innermost)

(** The shape of the issue: the varying value comes from
    [let tx = thread_idx_x]. *)
let nested_through_let ~count ~last =
  nested_mutables
    ~count
    ~assign_rhs:(fun _ -> var "tx")
    ~prelude:[]
    ~last
    ~outer:(fun body -> mk_texpr (TELet ("tx", 100, tid_x, body)) typ_unit)

(** The varying value reaches the accumulators through an enclosing mutable that
    the same body assigns. *)
let nested_through_mutable ~count ~last =
  let w = mk_texpr (TEAssign ("w", 99, tid_x)) typ_unit in
  nested_mutables
    ~count
    ~assign_rhs:(fun _ -> var "w")
    ~prelude:[w]
    ~last
    ~outer:(fun body -> mk_texpr (TELetMut ("w", 99, int 0, body)) typ_unit)

let guarded_by_last count =
  [if_ (gt (var (nested_name (count - 1))) (int 0)) barrier_call]

let check_linear msg count e =
  let start = Sys.time () in
  let errors = check_all e in
  let elapsed = Sys.time () -. start in
  Alcotest.(check int) (msg ^ ": no error") 0 (List.length errors) ;
  (* Linear time takes microseconds; the exponential one took seconds. *)
  Alcotest.(check bool)
    (Printf.sprintf "%s: %d nested mutables in %.3fs (< 2s)" msg count elapsed)
    true
    (elapsed < 2.0)

let test_nested_mutables_linear () =
  check_linear "through a let" 16 (nested_through_let ~count:16 ~last:[]) ;
  check_linear
    "through a mutable"
    16
    (nested_through_mutable ~count:16 ~last:[])

let test_nested_mutables_still_detected () =
  (* The restriction of the memo key must not hide a real divergence: the last
     mutable is varying, so a barrier guarded by it is refused. *)
  let count = 12 in
  barrier_refused
    "through a let"
    (nested_through_let ~count ~last:(guarded_by_last count)) ;
  barrier_refused
    "through a mutable"
    (nested_through_mutable ~count ~last:(guarded_by_last count)) ;
  (* and the same nest with a uniform increment is accepted *)
  accepted
    "uniform increment"
    (nested_mutables
       ~count
       ~assign_rhs:(fun _ -> int 1)
       ~prelude:[]
       ~last:(guarded_by_last count)
       ~outer:(fun body -> body))

(** Each way [relevant_reads] learns that a name is read: a mutable that only
    becomes varying through a name bound outside it ([y] below, standing for
    [let y = thread_idx_x]) must still be refused when it guards a barrier. Each
    case needs a different part of [relevant_reads]: the existing tests branch
    on [thread_idx_x] itself, which is not an environment name, so none of them
    would notice a part missing. *)
let test_nested_mutables_each_read_position () =
  let y = var "y" in
  let assign_s v = mk_texpr (TEAssign ("s", 7, v)) typ_unit in
  let guarded = if_ (gt (var "s") (int 0)) barrier_call in
  let case name body =
    (* [y] varying: refused. [y] uniform (a constant): accepted. *)
    let wrap yv =
      mk_texpr
        (TELet
           ( "y",
             8,
             yv,
             mk_texpr (TELetMut ("s", 7, int 0, seq [body; guarded])) typ_unit
           ))
        typ_unit
    in
    barrier_refused (name ^ ": varying") (wrap tid_x) ;
    accepted (name ^ ": uniform") (wrap (int 0))
  in
  case
    "let value"
    (mk_texpr
       (TELet ("n", 9, y, if_ (gt (var "n") (int 0)) (assign_s (int 1))))
       typ_unit) ;
  case "if condition" (if_ (gt y (int 0)) (assign_s (int 1))) ;
  case
    "while condition"
    (mk_texpr (TEWhile (gt y (int 0), assign_s (int 1))) typ_unit) ;
  case
    "for bound"
    (mk_texpr
       (TEFor ("i", 10, int 0, y, Sarek_ast.Upto, assign_s (int 1)))
       typ_unit) ;
  case
    "inner mutable"
    (mk_texpr
       (TELetMut
          ( "q",
            11,
            int 0,
            seq
              [
                mk_texpr (TEAssign ("q", 11, y)) typ_unit;
                if_ (gt (var "q") (int 0)) (assign_s (int 1));
              ] ))
       typ_unit)

let test_strict_barrier_if_value () =
  (* let c = if thread_idx_x > 16 then 1 else 0 in if c > 0 then barrier *)
  let value =
    mk_texpr (TEIf (gt tid_x (int 16), int 1, Some (int 0))) typ_int32
  in
  barrier_refused
    "value of a varying if"
    (mk_texpr
       (TELet ("c", 6, value, if_ (gt (var "c") (int 0)) barrier_call))
       typ_unit)

let test_strict_barrier_while_condition () =
  (* while (barrier; thread_idx_x > 0) do () : the condition is re-evaluated by
     the threads still looping only. *)
  let loop cond = mk_texpr (TEWhile (cond, int 0)) typ_unit in
  let cond test = seq [barrier_call; test] in
  barrier_refused
    "barrier in a varying while condition"
    (loop (cond (gt tid_x (int 0)))) ;
  accepted
    "barrier in a uniform while condition"
    (loop (cond (gt (int 1) (int 0))))

let test_strict_superstep_helper_param () =
  (* let gate n = if n > 16 then superstep in gate thread_idx_x *)
  let unit_e = mk_texpr TEUnit typ_unit in
  let superstep =
    mk_texpr (TESuperstep ("s", false, unit_e, unit_e)) typ_unit
  in
  let params = [tparam "n" 3] in
  let body = if_ (gt (var "n") (int 16)) superstep in
  barrier_refused
    "superstep under a varying argument"
    (let_rec "gate" params body (call "gate" [tid_x])) ;
  accepted
    "superstep under a uniform argument"
    (let_rec "gate" params body (call "gate" [int 20]))

let test_strict_barrier_helper_uniform_result () =
  (* let load base = (let t = thread_idx_x in ...; base + 32) in
     if load 0 > 16 then barrier: the result does not depend on the thread *)
  let add_to a b = mk_texpr (TEBinop (Sarek_ast.Add, a, b)) typ_int32 in
  let body =
    mk_texpr (TELet ("t", 7, tid_x, add_to (var "base") (int 32))) typ_int32
  in
  accepted
    "helper binding a thread index but returning a uniform value"
    (let_rec
       "load"
       [tparam "base" 3]
       body
       (if_ (gt (call "load" [int 0]) (int 16)) barrier_call))

let test_strict_barrier_shadowed_helper () =
  (* let f x = (if x > 16 then barrier; x) in let g y = f y in
     let f x = x + 1 in g thread_idx_x : g calls the first f *)
  let old_f = seq [if_ (gt (var "n") (int 16)) barrier_call; var "n"] in
  let new_f = mk_texpr (TEBinop (Sarek_ast.Add, var "n", int 1)) typ_int32 in
  let p = [tparam "n" 3] in
  barrier_refused
    "barrier hidden by a redefinition of the helper"
    (let_rec
       "f"
       p
       old_f
       (let_rec
          "g"
          [tparam "y" 4]
          (call "f" [var "y"])
          (let_rec "f" p new_f (call "g" [tid_x]))))

let test_shuffle_makes_barrier_kernel () =
  (* A shuffle kernel is a barrier kernel for the CPU runtimes: the Native
     simulation exchanges through shared memory between two barriers. *)
  Alcotest.(check bool)
    "expr_uses_barriers sees the warp collective"
    true
    (Sarek_convergence.expr_uses_barriers
       (shuffle_call (mk_texpr (TEInt 7) typ_int32)))

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
  (* [@divergent] superstep with varying control flow and no synchronisation
     point in the branch: accepted, as it is without the attribute (SS-2,
     SS-4: the attribute is inert). *)
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
  (* Non-divergent superstep with varying-branched body but NO barrier inside.
     The implicit end-of-superstep barrier is safe: all threads complete the body
     (each taking their own branch) and then all reach the implicit barrier.
     The old contains_diverging_control_flow check incorrectly flagged this — the
     correct behaviour is: no error. *)
  let tid_x = mk_texpr (TEVar ("thread_idx_x", 0)) typ_int32 in
  let const = mk_texpr (TEInt 16) typ_int32 in
  let cond = mk_texpr (TEBinop (Sarek_ast.Gt, tid_x, const)) typ_bool in
  let then_e = mk_texpr (TEInt 1) typ_int32 in
  let step = mk_texpr (TEIf (cond, then_e, None)) typ_int32 in
  let cont = mk_texpr (TEInt 2) typ_int32 in
  let e = mk_texpr (TESuperstep ("test", false, step, cont)) typ_int32 in
  let errors = Sarek_convergence.check_expr Sarek_convergence.init_ctx e in
  Alcotest.(check bool)
    "no error: varying branching inside superstep body without a nested \
     barrier is safe"
    false
    (List.length errors > 0)

(* specs/convergence-semantics.md section 2 (superstep settlement). *)

let varying_cond () =
  let tid_x = mk_texpr (TEVar ("thread_idx_x", 0)) typ_int32 in
  let const = mk_texpr (TEInt 16) typ_int32 in
  mk_texpr (TEBinop (Sarek_ast.Gt, tid_x, const)) typ_bool

let barrier () =
  mk_texpr
    (TEIntrinsicFun
       ( Sarek_env.CorePrimitiveRef "block_barrier",
         Some Sarek_core_primitives.ConvergencePoint,
         [] ))
    typ_unit

let count_barrier_errors errors =
  List.length
    (List.filter
       (function Sarek_error.Barrier_in_diverged_flow _ -> true | _ -> false)
       errors)

(* SS-1 (G-1, G-2): a superstep under a thread-varying if is rejected, with
   and without [@divergent]. *)
let test_superstep_under_varying_if divergent () =
  let unit_e = mk_texpr TEUnit typ_unit in
  let step = mk_texpr (TESuperstep ("s", divergent, unit_e, unit_e)) typ_unit in
  let e = mk_texpr (TEIf (varying_cond (), step, None)) typ_unit in
  let errors = Sarek_convergence.check_expr Sarek_convergence.init_ctx e in
  Alcotest.(check int) "entry error (SS-1)" 1 (count_barrier_errors errors)

(* SS-2 (G-4): a barrier under a varying if inside the body is rejected,
   with either flag. *)
let test_superstep_body_varying_barrier divergent () =
  let unit_e = mk_texpr TEUnit typ_unit in
  let body = mk_texpr (TEIf (varying_cond (), barrier (), None)) typ_unit in
  let e = mk_texpr (TESuperstep ("s", divergent, body, unit_e)) typ_unit in
  let errors = Sarek_convergence.check_expr Sarek_convergence.init_ctx e in
  Alcotest.(check int)
    "barrier in varying branch of the body (SS-2)"
    1
    (count_barrier_errors errors)

(* SS-3: the continuation keeps the outer mode. Under a diverged outer
   context, a barrier in the continuation is flagged too (before the
   settlement the continuation was reset to Converged). *)
let test_superstep_cont_keeps_outer_mode () =
  let unit_e = mk_texpr TEUnit typ_unit in
  let step = mk_texpr (TESuperstep ("s", false, unit_e, barrier ())) typ_unit in
  let ctx =
    {Sarek_convergence.init_ctx with mode = Sarek_convergence.Diverged}
  in
  let errors = Sarek_convergence.check_expr ctx step in
  Alcotest.(check int)
    "entry error and continuation barrier (SS-1, SS-3)"
    2
    (count_barrier_errors errors)

(* SS-3/SS-5: from a converged start, a barrier in the continuation is fine. *)
let test_superstep_cont_converged_clean () =
  let unit_e = mk_texpr TEUnit typ_unit in
  let body = mk_texpr (TEIf (varying_cond (), unit_e, None)) typ_unit in
  let step = mk_texpr (TESuperstep ("s", false, body, barrier ())) typ_unit in
  let errors = Sarek_convergence.check_expr Sarek_convergence.init_ctx step in
  Alcotest.(check int) "clean" 0 (List.length errors)

(** {1 Barrier recognition through name resolution (#447)}

    The kernels are typed from source here, so the barrier reaches the check
    through the environment's name resolution, as it does in the PPX. The same
    kernel must be refused in the base environment and under every spelling of
    [let open], because the check keys on the resolved intrinsic's convergence
    class. *)

let src_expr e = Sarek_ast.{e; expr_loc = dummy_loc}

let src_var name = src_expr (Sarek_ast.EVar name)

let src_if cond body = src_expr (Sarek_ast.EIf (cond, body, None))

let src_gt name =
  src_expr
    (Sarek_ast.EBinop
       (Sarek_ast.Gt, src_var name, src_expr (Sarek_ast.EInt32 16l)))

let src_barrier name =
  src_expr (Sarek_ast.EApp (src_var name, [src_expr Sarek_ast.EUnit]))

(* The barrier-error count of [if <cond_name> > 16l then <barrier> ()] typed
   from source, and whether the barrier is seen by [expr_uses_barriers] (the
   native backend's barrier detection). *)
let barrier_analysis_of ~opens ~barrier cond_name =
  Sarek_types.reset_tvar_counter () ;
  reset_var_id_counter () ;
  let kernel_body = src_if (src_gt cond_name) (src_barrier barrier) in
  let wrapped =
    List.fold_right
      (fun path body -> src_expr (Sarek_ast.EOpen (path, body)))
      opens
      kernel_body
  in
  match Sarek_typer.infer (Sarek_env.with_stdlib Sarek_env.empty) wrapped with
  | Error errors ->
      Alcotest.failf
        "typing failed: %s"
        (String.concat ", " (List.map Sarek_error.error_to_string errors))
  | Ok (te, _) ->
      ( count_barrier_errors
          (Sarek_convergence.check_expr Sarek_convergence.init_ctx te),
        Sarek_convergence.expr_uses_barriers te )

(* (label, opens, spelling of the barrier) *)
let open_spellings =
  [
    ("no open", [], "block_barrier");
    ("open Std", [["Std"]], "block_barrier");
    ("open Sarek_stdlib.Std", [["Sarek_stdlib"; "Std"]], "block_barrier");
    ("open Gpu", [["Gpu"]], "block_barrier");
    ("open Sarek_stdlib.Gpu", [["Sarek_stdlib"; "Gpu"]], "block_barrier");
    ("open Float32 then Std", [["Float32"]; ["Std"]], "block_barrier");
    ("open Std twice", [["Std"]; ["Std"]], "block_barrier");
    ("qualified Gpu.block_barrier", [], "Gpu.block_barrier");
    ("open Std, qualified", [["Std"]], "Gpu.block_barrier");
  ]

let test_divergent_barrier_refused_under_opens () =
  List.iter
    (fun (label, opens, barrier) ->
      let errors, _ = barrier_analysis_of ~opens ~barrier "thread_idx_x" in
      Alcotest.(check int)
        (label ^ ": barrier under a thread_idx_x branch is refused")
        1
        errors)
    open_spellings

let test_uniform_barrier_accepted_under_opens () =
  List.iter
    (fun (label, opens, barrier) ->
      let errors, uses = barrier_analysis_of ~opens ~barrier "block_idx_x" in
      Alcotest.(check int)
        (label ^ ": barrier under a block_idx_x branch is accepted")
        0
        errors ;
      Alcotest.(check bool)
        (label ^ ": the native backend sees the barrier")
        true
        uses)
    open_spellings

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
      ( "warp_shuffle",
        [
          test_case "after a uniform if" `Quick test_shuffle_after_uniform_if;
          test_case "in a varying if" `Quick test_shuffle_in_varying_if;
          test_case "in a varying loop" `Quick test_shuffle_in_varying_loop;
          test_case
            "after a varying return"
            `Quick
            test_shuffle_after_varying_return;
          test_case
            "in a helper called from diverged flow"
            `Quick
            test_shuffle_in_helper_called_from_diverged;
          test_case "strict: helper parameter" `Quick test_strict_helper_param;
          test_case "strict: helper result" `Quick test_strict_helper_result;
          test_case "strict: mutable" `Quick test_strict_mutable;
          test_case "strict: value of an if" `Quick test_strict_if_value;
          test_case
            "strict barrier: helper parameter"
            `Quick
            test_strict_barrier_helper_param;
          test_case
            "strict barrier: helper result"
            `Quick
            test_strict_barrier_helper_result;
          test_case "strict barrier: mutable" `Quick test_strict_barrier_mutable;
          test_case
            "nested mutables: linear time"
            `Quick
            test_nested_mutables_linear;
          test_case
            "nested mutables: divergence still detected"
            `Quick
            test_nested_mutables_still_detected;
          test_case
            "nested mutables: each read position"
            `Quick
            test_nested_mutables_each_read_position;
          test_case
            "strict barrier: value of an if"
            `Quick
            test_strict_barrier_if_value;
          test_case
            "strict barrier: while condition"
            `Quick
            test_strict_barrier_while_condition;
          test_case
            "strict barrier: superstep"
            `Quick
            test_strict_superstep_helper_param;
          test_case
            "strict barrier: uniform helper result"
            `Quick
            test_strict_barrier_helper_uniform_result;
          test_case
            "strict barrier: shadowed helper"
            `Quick
            test_strict_barrier_shadowed_helper;
          test_case
            "a shuffle kernel is a barrier kernel"
            `Quick
            test_shuffle_makes_barrier_kernel;
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
      ( "name resolution",
        [
          test_case
            "divergent barrier refused under every let open"
            `Quick
            test_divergent_barrier_refused_under_opens;
          test_case
            "uniform barrier accepted under every let open"
            `Quick
            test_uniform_barrier_accepted_under_opens;
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
          test_case
            "SS-1 under varying if"
            `Quick
            (test_superstep_under_varying_if false);
          test_case
            "SS-1 under varying if, [@divergent]"
            `Quick
            (test_superstep_under_varying_if true);
          test_case
            "SS-2 barrier in varying body branch"
            `Quick
            (test_superstep_body_varying_barrier false);
          test_case
            "SS-2 barrier in varying body branch, [@divergent]"
            `Quick
            (test_superstep_body_varying_barrier true);
          test_case
            "SS-3 continuation keeps diverged outer mode"
            `Quick
            test_superstep_cont_keeps_outer_mode;
          test_case
            "SS-3 continuation converged clean"
            `Quick
            test_superstep_cont_converged_clean;
        ] );
    ]
