(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Sarek_ir_types
open Sarek_ir_interp_value
open Sarek_ir_interp_intrinsics

(** Main intrinsic dispatcher - tries each category in order *)
let rec eval_intrinsic state path name args =
  (* Try GPU path intrinsics *)
  if is_gpu_path path then
    match eval_gpu_index_intrinsic state name with
    | Some v -> v
    | None -> (
        match eval_barrier_intrinsic name with
        | Some v -> v
        | None -> (
            match eval_type_conversion_intrinsic name args with
            | Some v -> v
            | None ->
                (* Not a GPU intrinsic, fall through to type-specific *)
                eval_intrinsic_by_type path name args))
  else eval_intrinsic_by_type path name args

(** Try type-specific intrinsics based on path *)
and eval_intrinsic_by_type path name args =
  if is_float32_path path then
    match eval_float32_math_intrinsic name args with
    | Some v -> v
    | None ->
        let full = String.concat "." (path @ [name]) in
        Interp_error.raise_error (Unknown_intrinsic {name = full})
  else if is_float64_path path then
    match eval_float64_math_intrinsic name args with
    | Some v -> v
    | None ->
        let full = String.concat "." (path @ [name]) in
        Interp_error.raise_error (Unknown_intrinsic {name = full})
  else if is_int32_path path then
    match eval_int32_math_intrinsic name args with
    | Some v -> v
    | None ->
        let full = String.concat "." (path @ [name]) in
        Interp_error.raise_error (Unknown_intrinsic {name = full})
  else
    let full = String.concat "." (path @ [name]) in
    Interp_error.raise_error (Unknown_intrinsic {name = full})

(** {1 Expression Evaluation} *)

(** Array expression evaluation *)
and eval_array_expr state env = function
  | EArrayRead (arr, idx) ->
      let a = get_array env arr in
      let i = to_int (eval_expr state env idx) in
      if i < 0 || i >= Array.length a then
        Interp_error.raise_error
          (Array_bounds_error
             {array_name = arr; index = i; length = Array.length a})
      else a.(i)
  | EArrayReadExpr (base, idx) ->
      let a =
        match eval_expr state env base with
        | VArray arr -> arr
        | _ ->
            Interp_error.raise_error
              (Not_an_array {expr = "EArrayReadExpr base"})
      in
      let i = to_int (eval_expr state env idx) in
      a.(i)
  | EArrayLen arr ->
      let a = get_array env arr in
      VInt32 (Int32.of_int (Array.length a))
  | EArrayCreate (ty, size_expr, _memspace) ->
      let size = to_int (eval_expr state env size_expr) in
      let init =
        match ty with
        | TInt32 -> VInt32 0l
        | TInt64 -> VInt64 0L
        | TFloat32 -> VFloat32 0.0
        | TFloat64 -> VFloat64 0.0
        | TBool -> VBool false
        | _ -> VUnit
      in
      VArray (Array.make size init)
  | _ ->
      Interp_error.raise_error
        (Pattern_match_failure
           {context = Printf.sprintf "eval_array_expr: unexpected expression"})

(** Record and variant expression evaluation *)
and eval_composite_expr state env = function
  | ERecordField (e, field) -> (
      match eval_expr state env e with
      | VRecord (type_name, fields) as vrec -> (
          match Sarek_type_helpers.lookup type_name with
          | Some h -> h.get_field vrec field
          | None ->
              let field_infos = Sarek_registry.record_fields type_name in
              let rec find_idx i = function
                | [] ->
                    Interp_error.raise_error
                      (Pattern_match_failure
                         {
                           context =
                             Printf.sprintf
                               "Record field %s not found in %s"
                               field
                               type_name;
                         })
                | info :: rest ->
                    if info.Sarek_registry.field_name = field then i
                    else find_idx (i + 1) rest
              in
              let idx = find_idx 0 field_infos in
              fields.(idx))
      | _ -> Interp_error.raise_error (Not_a_record {expr = "ERecordField"}))
  | ERecord (name, fields) ->
      VRecord
        ( name,
          Array.of_list (List.map (fun (_, e) -> eval_expr state env e) fields)
        )
  | EVariant (ty, ctor, args) ->
      VVariant
        (ty, Hashtbl.hash ctor mod 256, List.map (eval_expr state env) args)
  | _ ->
      Interp_error.raise_error
        (Pattern_match_failure
           {context = "eval_composite_expr: unexpected expression"})

(** Control flow expression evaluation *)
and eval_control_flow state env = function
  | EIf (cond, then_, else_) ->
      if to_bool (eval_expr state env cond) then eval_expr state env then_
      else eval_expr state env else_
  | EMatch (e, cases) ->
      let v = eval_expr state env e in
      let tag =
        match v with
        | VVariant (_, t, _) -> t
        | VInt32 n -> Int32.to_int n
        | _ -> 0
      in
      let rec find_case = function
        | [] ->
            Interp_error.raise_error
              (Pattern_match_failure {context = "EMatch"})
        | (PConstr (name, _), body) :: rest ->
            if Hashtbl.hash name mod 256 = tag then body else find_case rest
        | (PWild, body) :: _ -> body
      in
      eval_expr state env (find_case cases)
  | _ ->
      Interp_error.raise_error
        (Pattern_match_failure
           {context = "eval_control_flow: unexpected expression"})

(** Cast and intrinsic expression evaluation *)
and eval_special_expr state env = function
  | EIntrinsic (path, name, args) ->
      let arg_vals = List.map (eval_expr state env) args in
      eval_intrinsic state path name arg_vals
  | ECast (ty, e) -> (
      let v = eval_expr state env e in
      match ty with
      | TInt32 -> VInt32 (to_int32 v)
      | TInt64 -> VInt64 (to_int64 v)
      | TFloat32 -> VFloat32 (to_float32 v)
      | TFloat64 -> VFloat64 (to_float64 v)
      | TBool -> VBool (to_bool v)
      | _ -> v)
  | _ ->
      Interp_error.raise_error
        (Pattern_match_failure
           {context = "eval_special_expr: unexpected expression"})

(** Main expression evaluator - dispatches to specialized handlers *)
and eval_expr state env expr =
  match expr with
  (* Simple cases *)
  | EConst (CInt32 n) -> VInt32 n
  | EConst (CInt64 n) -> VInt64 n
  | EConst (CFloat32 f) -> VFloat32 f
  | EConst (CFloat64 f) -> VFloat64 f
  | EConst (CBool b) -> VBool b
  | EConst CUnit -> VUnit
  | EVar v -> lookup_var env v
  | ETuple exprs ->
      VArray (Array.of_list (List.map (eval_expr state env) exprs))
  (* Operators *)
  | EBinop (op, e1, e2) ->
      eval_binop op (eval_expr state env e1) (eval_expr state env e2)
  | EUnop (op, e) -> eval_unop op (eval_expr state env e)
  (* Array operations *)
  | (EArrayRead _ | EArrayReadExpr _ | EArrayLen _ | EArrayCreate _) as e ->
      eval_array_expr state env e
  (* Record/Variant operations *)
  | (ERecordField _ | ERecord _ | EVariant _) as e ->
      eval_composite_expr state env e
  (* Control flow *)
  | (EIf _ | EMatch _) as e -> eval_control_flow state env e
  (* Special operations *)
  | (EIntrinsic _ | ECast _) as e -> eval_special_expr state env e
  (* Function application *)
  | EApp (fn_expr, args) -> eval_app state env fn_expr args

and get_array env name =
  try Hashtbl.find env.arrays name
  with Not_found -> (
    try Hashtbl.find env.shared name
    with Not_found ->
      Interp_error.raise_error (Unbound_variable {name; context = "get_array"}))

and eval_app state env fn_expr args =
  match fn_expr with
  | EIntrinsic (path, name, []) ->
      let arg_vals = List.map (eval_expr state env) args in
      eval_intrinsic state path name arg_vals
  | EVar v -> (
      match Hashtbl.find_opt env.funcs v.var_name with
      | Some hf ->
          (* Call helper function *)
          let arg_vals = List.map (eval_expr state env) args in
          let local_env = copy_env env in
          List.iter2
            (fun param arg -> bind_var local_env param arg)
            hf.hf_params
            arg_vals ;
          (* Execute function body and get return value *)
          exec_stmt_for_return state local_env hf.hf_body
      | None -> Interp_error.raise_error (Unknown_function {name = v.var_name}))
  | _ ->
      Interp_error.raise_error
        (Unsupported_operation
           {
             operation = "function call";
             reason = "unsupported function expression";
           })

(** {1 Statement Execution} *)

and exec_stmt state env stmt =
  match stmt with
  | SEmpty -> ()
  | SSeq stmts -> List.iter (exec_stmt state env) stmts
  | SAssign (lv, e) ->
      let v = eval_expr state env e in
      assign_lvalue state env lv v
  | SIf (cond, then_s, else_s) ->
      if to_bool (eval_expr state env cond) then exec_stmt state env then_s
      else Option.iter (exec_stmt state env) else_s
  | SWhile (cond, body) ->
      while to_bool (eval_expr state env cond) do
        exec_stmt state env body
      done
  | SFor (v, start, stop, dir, body) ->
      let start_val = to_int32 (eval_expr state env start) in
      let stop_val = to_int32 (eval_expr state env stop) in
      (* OCaml for loops are inclusive: "for i = 0 to n" runs i=0,1,...,n *)
      let incr, cmp =
        match dir with
        | Upto -> ((fun i -> Int32.add i 1l), fun i s -> i <= s)
        | Downto -> ((fun i -> Int32.sub i 1l), fun i s -> i >= s)
      in
      let i = ref start_val in
      while cmp !i stop_val do
        bind_var env v (VInt32 !i) ;
        exec_stmt state env body ;
        i := incr !i
      done
  | SMatch (e, cases) ->
      let v = eval_expr state env e in
      let tag =
        match v with
        | VVariant (_, t, _) -> t
        | VInt32 n -> Int32.to_int n
        | _ -> 0
      in
      let rec find_case = function
        | [] ->
            Interp_error.raise_error
              (Pattern_match_failure {context = "SMatch"})
        | (PConstr (name, vars), body) :: rest ->
            if Hashtbl.hash name mod 256 = tag then begin
              (* Bind pattern variables by name *)
              (match v with
              | VVariant (_, _, args) ->
                  List.iter2
                    (fun vname arg ->
                      Hashtbl.replace env.vars_by_name vname arg)
                    vars
                    args
              | _ -> ()) ;
              body
            end
            else find_case rest
        | (PWild, body) :: _ -> body
      in
      exec_stmt state env (find_case cases)
  | SReturn _ -> () (* Return handled by exec_stmt_for_return *)
  | SBarrier -> Effect.perform Barrier
  | SWarpBarrier -> Effect.perform Barrier
  | SExpr e ->
      let _ = eval_expr state env e in
      ()
  | SLet (v, e, body) -> (
      (* Special handling for shared memory arrays *)
      match e with
      | EArrayCreate (ty, size_expr, Shared) ->
          (* Shared memory: reuse if exists, else create and store in env.shared *)
          let name = v.var_name in
          (match Hashtbl.find_opt env.shared name with
          | Some arr -> bind_var env v (VArray arr)
          | None ->
              let size = to_int (eval_expr state env size_expr) in
              let init =
                match ty with
                | TInt32 -> VInt32 0l
                | TInt64 -> VInt64 0L
                | TFloat32 -> VFloat32 0.0
                | TFloat64 -> VFloat64 0.0
                | TBool -> VBool false
                | _ -> VUnit
              in
              let arr = Array.make size init in
              Hashtbl.add env.shared name arr ;
              bind_var env v (VArray arr)) ;
          exec_stmt state env body
      | _ ->
          let value = eval_expr state env e in
          bind_var env v value ;
          exec_stmt state env body)
  | SLetMut (v, e, body) ->
      let value = eval_expr state env e in
      bind_var env v value ;
      exec_stmt state env body
  | SPragma (_, body) -> exec_stmt state env body
  | SMemFence -> ()
  | SBlock body -> exec_stmt state env body
  | SNative {ocaml; _} ->
      (* Call the typed OCaml fallback *)
      ocaml.run ~block:state.block_dim ~grid:state.grid_dim [||]

and assign_lvalue state env lv value =
  (* Store values directly - VRecord is handled by ERecordField *)
  match lv with
  | LVar v -> bind_var env v value
  | LArrayElem (arr, idx_expr) ->
      let a = get_array env arr in
      let i = to_int (eval_expr state env idx_expr) in
      a.(i) <- value
  | LArrayElemExpr (base_expr, idx_expr) ->
      let a =
        match eval_expr state env base_expr with
        | VArray arr -> arr
        | _ ->
            Interp_error.raise_error
              (Not_an_array {expr = "LArrayElemExpr base"})
      in
      let i = to_int (eval_expr state env idx_expr) in
      a.(i) <- value
  | LRecordField (base_lv, _field) ->
      (* Record field assignment is complex - simplified here *)
      ignore base_lv ;
      Interp_error.raise_error
        (Unsupported_operation
           {
             operation = "record field assignment";
             reason = "not fully supported";
           })

and exec_stmt_for_return state env stmt =
  match stmt with
  | SReturn e -> eval_expr state env e
  | SSeq stmts ->
      let rec exec = function
        | [] -> VUnit
        | [s] -> exec_stmt_for_return state env s
        | s :: rest ->
            exec_stmt state env s ;
            exec rest
      in
      exec stmts
  | SIf (cond, then_s, else_s) -> (
      if to_bool (eval_expr state env cond) then
        exec_stmt_for_return state env then_s
      else
        match else_s with
        | Some s -> exec_stmt_for_return state env s
        | None -> VUnit)
  | SLet (v, e, body) -> (
      (* Special handling for shared memory arrays *)
      match e with
      | EArrayCreate (ty, size_expr, Shared) ->
          let name = v.var_name in
          (match Hashtbl.find_opt env.shared name with
          | Some arr -> bind_var env v (VArray arr)
          | None ->
              let size = to_int (eval_expr state env size_expr) in
              let init =
                match ty with
                | TInt32 -> VInt32 0l
                | TInt64 -> VInt64 0L
                | TFloat32 -> VFloat32 0.0
                | TFloat64 -> VFloat64 0.0
                | TBool -> VBool false
                | _ -> VUnit
              in
              let arr = Array.make size init in
              Hashtbl.add env.shared name arr ;
              bind_var env v (VArray arr)) ;
          exec_stmt_for_return state env body
      | _ ->
          let value = eval_expr state env e in
          bind_var env v value ;
          exec_stmt_for_return state env body)
  | SLetMut (v, e, body) ->
      let value = eval_expr state env e in
      bind_var env v value ;
      exec_stmt_for_return state env body
  | _ ->
      exec_stmt state env stmt ;
      VUnit

(** {1 Kernel Execution} *)

(** Run all threads in a block with BSP barrier synchronization *)
let run_block env body block_idx block_dim grid_dim =
  let bx, by, bz = block_dim in
  let num_threads = bx * by * bz in
  let waiting : (unit, unit) Effect.Deep.continuation option array =
    Array.make num_threads None
  in
  let num_waiting = ref 0 in
  let num_completed = ref 0 in

  let run_thread_with_barrier tid =
    let tx = tid mod bx in
    let ty = tid / bx mod by in
    let tz = tid / (bx * by) in
    let state = {thread_idx = (tx, ty, tz); block_idx; block_dim; grid_dim} in
    let thread_env = copy_env env in
    Effect.Deep.match_with
      (fun () -> exec_stmt state thread_env body)
      ()
      {
        retc = (fun () -> incr num_completed);
        exnc = raise;
        effc =
          (fun (type a) (eff : a Effect.t) ->
            match eff with
            | Barrier ->
                Some
                  (fun (k : (a, unit) Effect.Deep.continuation) ->
                    waiting.(tid) <- Some k ;
                    incr num_waiting)
            | _ -> None);
      }
  in

  let resume_thread tid =
    match waiting.(tid) with
    | Some k ->
        waiting.(tid) <- None ;
        Effect.Deep.match_with
          (fun () -> Effect.Deep.continue k ())
          ()
          {
            retc = (fun () -> incr num_completed);
            exnc = raise;
            effc =
              (fun (type a) (eff : a Effect.t) ->
                match eff with
                | Barrier ->
                    Some
                      (fun (k : (a, unit) Effect.Deep.continuation) ->
                        waiting.(tid) <- Some k ;
                        incr num_waiting)
                | _ -> None);
          }
    | None -> ()
  in

  (* Start all threads *)
  for tid = 0 to num_threads - 1 do
    run_thread_with_barrier tid
  done ;

  (* Superstep loop *)
  while !num_waiting > 0 do
    let to_resume = !num_waiting in
    num_waiting := 0 ;
    for tid = 0 to num_threads - 1 do
      if Option.is_some waiting.(tid) then resume_thread tid
    done ;
    if !num_waiting = to_resume && !num_completed < num_threads then
      Interp_error.raise_error
        (BSP_deadlock {message = "no progress made in interpreter"})
  done

(** Run all blocks in a grid (sequential) *)
let run_grid_sequential env body block_dim grid_dim =
  let gx, gy, gz = grid_dim in
  for bz = 0 to gz - 1 do
    for by = 0 to gy - 1 do
      for bx = 0 to gx - 1 do
        Hashtbl.clear env.shared ;
        run_block env body (bx, by, bz) block_dim grid_dim
      done
    done
  done
