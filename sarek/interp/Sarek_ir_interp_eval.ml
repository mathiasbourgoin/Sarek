(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Sarek_ir_types
open Sarek_ir_interp_value
open Sarek_ir_interp_intrinsics

(** Positional tuple-field index for a synthesized [_tup_*] record: ["_0"] ->
    [Some 0], ["_12"] -> [Some 12], any other name -> [None]. Used to resolve
    tuple-element field access without a record-registry entry. *)
let positional_field_index (field : string) : int option =
  let n = String.length field in
  if n >= 2 && field.[0] = '_' then begin
    let rec all_digits i =
      i >= n || (field.[i] >= '0' && field.[i] <= '9' && all_digits (i + 1))
    in
    if all_digits 1 then int_of_string_opt (String.sub field 1 (n - 1))
    else None
  end
  else None

(** Main intrinsic dispatcher - tries each category in order *)
let rec eval_intrinsic state path name args =
  (* Global-memory atomics and fences are path-independent (they may be opened
     from Gpu or referenced unqualified), so try them first. *)
  match eval_atomic_intrinsic name args with
  | Some v -> v
  | None ->
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
      if i < 0 || i >= Array.length a then
        Interp_error.raise_error
          (Array_bounds_error
             {array_name = "EArrayReadExpr"; index = i; length = Array.length a})
      else a.(i)
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
          | None when positional_field_index field <> None ->
              (* Synthesized tuple records (L13, [_tup_*]) use positional field
                 names [_0], [_1], ...; resolve the index directly rather than
                 through the record registry, which never holds these. *)
              let idx = Option.get (positional_field_index field) in
              if idx < Array.length fields then fields.(idx)
              else
                Interp_error.raise_error
                  (Pattern_match_failure
                     {
                       context =
                         Printf.sprintf
                           "Positional field %s out of range in %s"
                           field
                           type_name;
                     })
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
        | (PConstr (name, vars), body) :: rest ->
            if Hashtbl.hash name mod 256 = tag then begin
              (* Bind the constructor's payload variables positionally, exactly
                 as SMatch does, so an expression-position match on a variant
                 (e.g. [let s = match c.kind with Shade f -> f]) can use the
                 payload. Previously EMatch selected the arm but left payload
                 vars unbound. *)
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
      | TFloat16 ->
          (* An f16 value is carried as VFloat32 (there is no VFloat16 -- f16 is
             a storage width, not a compute type), but it MUST be narrowed here.
             Without this arm the old catch-all returned [v] unchanged, so
             `float32_of_float16 (float16_of_float32 3.14159)` would yield
             3.14159 on the interpreter and 3.14062 on the GPU. Rounding here is
             what keeps the interpreter a faithful oracle for f16 kernels. *)
          VFloat32 (Sarek_float16.to_float16 (to_float32 v))
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

(* KNOWN GAP — a vector passed as a HELPER FUNCTION PARAMETER is not reachable
   here. [eval_app] binds arguments with [bind_var], which writes
   [vars]/[vars_by_name], while this looks only in [arrays]/[shared] — the
   KERNEL's own parameters. Indexing a vector parameter inside a helper
   therefore raises "Unbound variable '<param>' in get_array", so a
   tail-recursive fold over a vector runs on every backend except the
   interpreter.

   A [vars_by_name] fallback here does make those folds run, and was written and
   then deliberately REVERTED, because on its own it trades a loud failure for a
   silent wrong answer. Helper parameter ids come from the typer's [tparam_id]
   space while body locals come from the kernel-wide [fresh_id] counter
   (Sarek_lower_ir.ml), the two spaces overlap, and [lookup_var] resolves by id
   before name — so a tail-recursion temporary can carry the same id as a
   parameter and clobber it. Observed with the fallback in place: a
   single-helper [vsum acc v k n] fold over four 1.0s returns 0 on the
   interpreter where Vulkan and Native both return 4, with no error. Which
   kernels are affected depends only on id numbering, so a passing test proves
   nothing about the next kernel.

   The repair is to make helper ids unique, and it lands with the fallback and
   with the lookup-precedence question it raises (a helper's vector formal that
   shares a name with a kernel array must shadow it, not lose to it). *)
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
  | SCoopmat op -> exec_coopmat state env op

(** {1 Cooperative matrix — backlog-62 slice 3}

    The interpreter is the ORACLE every backend is checked against, so this is
    not a courtesy implementation: [test_vulkan_coopmat_ir_e2e] compares the
    GLSL backend's output against these numbers bit for bit, and a skip here
    would let a coopmat kernel "agree" with an interpreter that never computed
    the product.

    Every invocation holds the whole matrix and performs the whole operation;
    see {!Sarek_ir_interp_value.env}.[coopmats] for why that is exact rather
    than approximate.

    {b Only the INTEGER configurations are evaluated.} Float accumulation is
    refused, and refused HERE rather than left to produce a plausible number,
    because the interpreter cannot honestly model it: SPV_KHR_cooperative_matrix
    leaves the ORDER of the k+1 additions to the implementation, so there is no
    single value a strict oracle could compare against (design document §5.1).
    Integer accumulation has no such freedom — the specification states it is
    exact at the precision of the result type — which is the entire reason the
    integer path lands under Sarek's existing strict contract. *)

and coopmat_zero (c : Sarek_coopmat_types.component_type) =
  match c with
  | Sarek_coopmat_types.Uint8 | Sarek_coopmat_types.Sint8
  | Sarek_coopmat_types.Uint32 | Sarek_coopmat_types.Sint32 ->
      VInt32 0l
  | Sarek_coopmat_types.Float16 | Sarek_coopmat_types.Float32 ->
      coopmat_refuse_float c

and coopmat_refuse_float : 'a. Sarek_coopmat_types.component_type -> 'a =
 fun c ->
  Interp_error.raise_error
    (Unsupported_operation
       {
         operation =
           "cooperative matrix with "
           ^ Sarek_coopmat_types.component_name c
           ^ " components";
         reason =
           "float cooperative-matrix accumulation has no single correct value \
            to be an oracle for: SPV_KHR_cooperative_matrix leaves the order \
            of the additions to the implementation. The INTEGER configurations \
            are evaluated, and are exact.";
       })

(** Narrow an accumulated Int32 to the declared component type.

    This is what makes the interpreter a faithful oracle for the 8-bit operand
    types rather than merely a plausible one: a value loaded into a [u8]
    fragment from a buffer holding 300 is 44 on the device, and an oracle that
    kept 300 would disagree with correct hardware. *)
and coopmat_narrow (c : Sarek_coopmat_types.component_type) (n : int32) =
  match c with
  | Sarek_coopmat_types.Uint8 -> Int32.logand n 0xffl
  | Sarek_coopmat_types.Sint8 ->
      let b = Int32.logand n 0xffl in
      if Int32.compare b 0x80l >= 0 then Int32.sub b 0x100l else b
  | Sarek_coopmat_types.Uint32 | Sarek_coopmat_types.Sint32 -> n
  | Sarek_coopmat_types.Float16 | Sarek_coopmat_types.Float32 ->
      coopmat_refuse_float c

and get_coopmat env name =
  match Hashtbl.find_opt env.coopmats name with
  | Some m -> m
  | None ->
      Interp_error.raise_error
        (Unbound_variable {name; context = "cooperative-matrix fragment"})

and exec_coopmat state env op =
  let open Sarek_coopmat_types in
  match op with
  | CM_decl {name; frag} ->
      Hashtbl.replace
        env.coopmats
        name
        (Array.make
           (fragment_components frag)
           (coopmat_zero frag.frag_component))
  | CM_load {dst; frag; src; index; stride} ->
      let buf = get_array env src in
      let base = to_int (eval_expr state env index) in
      let stride = to_int (eval_expr state env stride) in
      let rows, cols = fragment_dims frag in
      let m = Array.make (rows * cols) (coopmat_zero frag.frag_component) in
      for r = 0 to rows - 1 do
        for c = 0 to cols - 1 do
          let i = base + (r * stride) + c in
          if i < 0 || i >= Array.length buf then
            Interp_error.raise_error
              (Array_bounds_error
                 {array_name = src; index = i; length = Array.length buf})
          else
            m.((r * cols) + c) <-
              VInt32 (coopmat_narrow frag.frag_component (to_int32 buf.(i)))
        done
      done ;
      Hashtbl.replace env.coopmats dst m
  | CM_store {src; frag; dst; index; stride} ->
      let m = get_coopmat env src in
      let buf = get_array env dst in
      let base = to_int (eval_expr state env index) in
      let stride = to_int (eval_expr state env stride) in
      let rows, cols = fragment_dims frag in
      for r = 0 to rows - 1 do
        for c = 0 to cols - 1 do
          let i = base + (r * stride) + c in
          if i < 0 || i >= Array.length buf then
            Interp_error.raise_error
              (Array_bounds_error
                 {array_name = dst; index = i; length = Array.length buf})
          else buf.(i) <- m.((r * cols) + c)
        done
      done
  | CM_muladd {dst; a; b; c; cfg} ->
      if cfg.cfg_saturating then
        Interp_error.raise_error
          (Unsupported_operation
             {
               operation = "saturating cooperative-matrix accumulation";
               reason =
                 "the saturating variant computes a different function from \
                  the plain one and nothing has executed it, so there is \
                  nothing for an oracle to be faithful to";
             }) ;
      let ma = get_coopmat env a
      and mb = get_coopmat env b
      and mc = get_coopmat env c in
      let sh = cfg.cfg_shape in
      let out = Array.make (sh.m * sh.n) (coopmat_zero cfg.cfg_result) in
      for i = 0 to sh.m - 1 do
        for j = 0 to sh.n - 1 do
          (* Int32 throughout, so that a result exceeding 32 bits WRAPS exactly
             as the device wraps, rather than being accidentally right on a
             63-bit host int. This is the claim the whole integer path rests
             on. *)
          let acc = ref (to_int32 mc.((i * sh.n) + j)) in
          for kk = 0 to sh.k - 1 do
            let av = to_int32 ma.((i * sh.k) + kk) in
            let bv = to_int32 mb.((kk * sh.n) + j) in
            acc := Int32.add !acc (Int32.mul av bv)
          done ;
          out.((i * sh.n) + j) <- VInt32 (coopmat_narrow cfg.cfg_result !acc)
        done
      done ;
      Hashtbl.replace env.coopmats dst out

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
      if i < 0 || i >= Array.length a then
        Interp_error.raise_error
          (Array_bounds_error
             {array_name = "LArrayElemExpr"; index = i; length = Array.length a})
      else a.(i) <- value
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
