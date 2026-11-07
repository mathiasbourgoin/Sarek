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

(** Lanes per cooperative-matrix subgroup. 32, the CUDA/Metal width and the tile
    width of the kernels in this repository. A device with a wider subgroup
    (wave64) corresponds to two interpreter subgroups; that is only visible to a
    kernel in which two interpreter subgroups touch the same tile, which is a
    data race on a 32-wide device and not a supported program. *)
let coopmat_subgroup_size = 32

(** Linear invocation index within the block (x fastest). *)
let linear_thread_id state =
  let tx, ty, tz = state.thread_idx and bx, by, _ = state.block_dim in
  tx + (bx * (ty + (by * tz)))

(** The lowest lane of its subgroup: the one that executes a cooperative-matrix
    operation on the subgroup's behalf. *)
let coopmat_is_leader state =
  linear_thread_id state mod coopmat_subgroup_size = 0

(** The fragment-table key of [name] as seen by the subgroup of [state]. *)
let coopmat_key state name =
  (linear_thread_id state / coopmat_subgroup_size, name)

(** The buffer position of element ([r], [c]) of a fragment loaded from or
    stored to [base] with [stride] in [layout] (Sarek_ir_types.CM_load). *)
let coopmat_element ~layout ~base ~stride r c =
  match (layout : Sarek_coopmat_types.layout) with
  | Row_major -> base + (r * stride) + c
  | Col_major -> base + (c * stride) + r

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

(** {!eval_intrinsic} for a call that may need the block's shared table: the
    warp operations exchange through it. Everything else is path-dispatched as
    before. *)
and eval_intrinsic_in_env env state path name args =
  match Sarek_warp.classify name with
  | Some call -> eval_warp_call env.shared state ~name call args
  | None -> eval_intrinsic state path name args

(** Try type-specific intrinsics based on path *)
and eval_intrinsic_by_type path name args =
  let builtin =
    if is_float32_path path then eval_float32_math_intrinsic name args
    else if is_float64_path path then eval_float64_math_intrinsic name args
    else if is_int32_path path then eval_int32_math_intrinsic name args
    else None
  in
  match builtin with
  | Some v -> v
  | None -> eval_registered_intrinsic path name args

(** An intrinsic with no built-in entry: run the host implementation that
    [let%sarek_intrinsic] registered for it (any library's, sarek's stdlib or a
    downstream one), converting each argument to its DECLARED type. Raises
    [Unknown_intrinsic] when nothing is registered, which is the case for a
    signature with a non-scalar type. *)
and eval_registered_intrinsic path name args =
  let unknown () =
    Interp_error.raise_error
      (Unknown_intrinsic {name = String.concat "." (path @ [name])})
  in
  match
    ( Sarek_registry.find_host_impl ~module_path:path name,
      Sarek_registry.find_fun ~module_path:path name )
  with
  | Some impl, Some fi when List.length fi.fi_arg_types = List.length args -> (
      let to_host ty v : Sarek_registry.host_value =
        match ty with
        | "int32" -> HInt32 (to_int32 v)
        | "int64" -> HInt64 (to_int64 v)
        | "float32" -> HFloat32 (to_float32 v)
        | "float" | "float64" -> HFloat64 (to_float64 v)
        | "bool" -> HBool (to_bool v)
        | _ -> HUnit
      in
      match impl (List.map2 to_host fi.fi_arg_types args) with
      | HInt32 n -> VInt32 n
      | HInt64 n -> VInt64 n
      | HFloat32 f -> VFloat32 (Sarek_float32.to_float32 f)
      | HFloat64 f -> VFloat64 f
      | HBool b -> VBool b
      | HUnit -> VUnit)
  | _ -> unknown ()

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
      VArray (alloc_kernel_array ty size)
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
              (* THIRD tier, and the only resolver that has one — neither
                 [assign_lvalue] nor [read_lvalue] consults [Sarek_registry].
                 Is it reachable? Both tables are written by the SAME
                 [@@sarek.type] expansion under the same key
                 ([Sarek.Sarek_type_helpers.register] and
                 [Sarek_registry.register_record], both keyed on the type's full
                 name), so a type in one is in the other. The only way in is
                 [Sarek_registry.record_fields]' extra SHORT-NAME fallback, which
                 [Sarek_type_helpers.lookup] does not have: a [VRecord] carrying
                 the bare [point] for a type registered as [Geometry_lib.point]
                 would miss the helpers and hit here.
                 Measured: instrumented with an eprintf and run over the whole
                 test suite (dune runtest), this arm was entered ZERO times. So
                 it is not demonstrably reachable, and it is kept rather than
                 turned into an error because it is also the only arm that would
                 catch that short-name shape if the IR ever produced it. Note
                 that [record_fields] raises [Failure] (not an [Interp_error]) for
                 an unknown type, so an unregistered record reaching here reports
                 differently from every other refusal in this file. *)
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
        (ty, variant_tag_of_ctor ctor, List.map (eval_expr state env) args)
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
            if variant_tag_of_ctor name = tag then begin
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
      eval_intrinsic_in_env env state path name arg_vals
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
      | (TUint8 | TInt8) as t ->
          (* 8-bit storage types (Sarek_ir_int8): the int32 a store of [v]
             followed by a read gives back: the low byte, zero-extended for
             uint8 and sign-extended for int8. Reads and stores of 8-bit arrays
             reach this arm too, through [Sarek_ir_int8.normalise]. *)
          VInt32 (Sarek_ir_int8.narrow t (to_int32 v))
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

(* Vector lookup, innermost scope first.

   A vector reaches a helper as a PARAMETER, bound by [eval_app] through
   [bind_var] — so it lives in [vars_by_name], not in [arrays], which holds the
   KERNEL's own parameters. That table is therefore consulted FIRST: a helper's
   formal shadows a kernel array of the same name, which is what lexical scoping
   means and what every GPU backend does. Checking [arrays] first made the
   shadowing case silently read the wrong buffer — measured: a helper folding
   over its formal [src] while the kernel also had a vector [src] folded the
   KERNEL's array, returning 400 instead of 4 on the interpreter while Vulkan and
   Native both returned 4.

   The lookup is read-only and must stay so: [arrays] is aliased rather than
   copied into a callee scope because kernel buffers are shared across threads by
   design, so registering a binding there would be a cross-thread write.

   This fallback existed briefly on its own and was reverted, because alone it
   traded a loud [Unbound_variable] for a silently wrong number: the
   tail-recursion transform allocated its loop scaffolding from a counter
   starting at 0, colliding with the typer's parameter ids. That is fixed in
   Sarek_tailrec_elim.ml, and a helper call now gets its own scope
   ([callee_env]) rather than a copy of the caller's, so the fallback can be
   reinstated. *)
and get_array env name =
  match Hashtbl.find_opt env.vars_by_name name with
  | Some (VArray data) -> data
  | _ -> (
      try Hashtbl.find env.arrays name
      with Not_found -> (
        try Hashtbl.find env.shared name
        with Not_found ->
          Interp_error.raise_error
            (Unbound_variable {name; context = "get_array"})))

and eval_app state env fn_expr args =
  match fn_expr with
  | EIntrinsic (path, name, []) ->
      let arg_vals = List.map (eval_expr state env) args in
      eval_intrinsic_in_env env state path name arg_vals
  | EVar v -> (
      match Hashtbl.find_opt env.funcs v.var_name with
      | Some hf ->
          (* Call helper function *)
          let arg_vals = List.map (eval_expr state env) args in
          let local_env = callee_env env in
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
            if variant_tag_of_ctor name = tag then begin
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
  | SWarpBarrier -> Effect.perform Warp_barrier
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
              let arr = alloc_kernel_array ty size in
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
    not a courtesy implementation: [test_coopmat_integer_e2e] (GLSL) and
    [test_coopmat_hip_e2e] (HIP) compare the backends' output against these
    numbers bit for bit, and a skip here would let a coopmat kernel "agree" with
    an interpreter that never computed the product.

    Each operation is executed ONCE PER SUBGROUP, by the subgroup's lowest lane
    ({!coopmat_is_leader}), on a whole-matrix fragment owned by the subgroup
    ({!coopmat_key}); the other lanes' statements are no-ops. Running it in
    every invocation would not be equivalent to the device: in-place
    [D = A*B + D] would accumulate once per lane. See
    {!Sarek_ir_interp_value.env}.[coopmats] for the model and its assumptions
    (dynamically uniform arguments, every lane reaching the operation).

    {b Integer configurations are a strict oracle}: the specification states
    integer accumulation is exact at the precision of the result type.

    {b The one float configuration with a derived bound} —
    [f16 * f16 + f32 -> f32] — is evaluated only after
    {!Sarek_coopmat.accept_relaxed_accumulation}, and in ONE FIXED ORDER:
    [acc := C[m][n]], then for [k = 0 .. K-1],
    [acc := fl32 (acc + A[m][k] * B[k][n])]. The product of two binary16 values
    is exact in binary32, so the only rounding is the addition's. The hardware
    chooses its own order, so this is one admissible member of the bounded set
    (docs/design/f16-relaxed-accuracy.md §5.2), not a value a device must equal;
    tests compare both against the exact result under the bound. Every other
    float configuration is refused ({!Sarek_coopmat.policy_verdict}). *)

and coopmat_zero (c : Sarek_coopmat_types.component_type) =
  match c with
  | Sarek_coopmat_types.Uint8 | Sarek_coopmat_types.Sint8
  | Sarek_coopmat_types.Uint32 | Sarek_coopmat_types.Sint32 ->
      VInt32 0l
  | Sarek_coopmat_types.Float16 | Sarek_coopmat_types.BFloat16
  | Sarek_coopmat_types.Float32 ->
      VFloat32 0.0

(** Round a binary32 value to bfloat16 (round to nearest, ties to even), the
    result kept as a binary32 value. NaN stays NaN. *)
and to_bfloat16 (x : float) =
  if Float.is_nan x then x
  else
    let bits = Int32.bits_of_float x in
    let lsb = Int32.logand (Int32.shift_right_logical bits 16) 1l in
    let rounded = Int32.add bits (Int32.add 0x7fffl lsb) in
    Int32.float_of_bits (Int32.logand rounded 0xffff0000l)

(** Convert a buffer element to the declared component type.

    For the 8-bit operand types this is what makes the interpreter a faithful
    oracle rather than merely a plausible one: a value loaded into a [u8]
    fragment from a buffer holding 300 is 44 on the device, and an oracle that
    kept 300 would disagree with correct hardware. For [f16] it is the
    round-to-nearest-even narrowing the device performs (the identity on a
    [float16] buffer). *)
and coopmat_component (c : Sarek_coopmat_types.component_type) (v : value) =
  match c with
  | Sarek_coopmat_types.Uint8 -> VInt32 (Int32.logand (to_int32 v) 0xffl)
  | Sarek_coopmat_types.Sint8 ->
      let b = Int32.logand (to_int32 v) 0xffl in
      VInt32 (if Int32.compare b 0x80l >= 0 then Int32.sub b 0x100l else b)
  | Sarek_coopmat_types.Uint32 | Sarek_coopmat_types.Sint32 ->
      VInt32 (to_int32 v)
  | Sarek_coopmat_types.Float16 ->
      VFloat32 (Sarek_float16.to_float16 (to_float32 v))
  | Sarek_coopmat_types.BFloat16 -> VFloat32 (to_bfloat16 (to_float32 v))
  | Sarek_coopmat_types.Float32 -> VFloat32 (to_float32 v)

and get_coopmat state env name =
  match Hashtbl.find_opt env.coopmats (coopmat_key state name) with
  | Some m -> m
  | None ->
      Interp_error.raise_error
        (Unbound_variable {name; context = "cooperative-matrix fragment"})

and coopmat_refuse_policy env (cfg : Sarek_coopmat_types.config) =
  match Sarek_coopmat.policy_verdict_for env.policy cfg with
  | Sarek_capability.Available -> ()
  | Sarek_capability.Unavailable cap ->
      Interp_error.raise_error
        (Unsupported_operation
           {
             operation =
               "cooperative matrix " ^ Sarek_coopmat_types.config_name cfg;
             reason = cap.Sarek_capability.cap_why;
           })
  | Sarek_capability.Unknown why ->
      Interp_error.raise_error
        (Unsupported_operation
           {
             operation =
               "cooperative matrix " ^ Sarek_coopmat_types.config_name cfg;
             reason = why;
           })

and coopmat_muladd_int (cfg : Sarek_coopmat_types.config) ma mb mc =
  let open Sarek_coopmat_types in
  let sh = cfg.cfg_shape in
  Array.init (sh.m * sh.n) (fun idx ->
      let i = idx / sh.n and j = idx mod sh.n in
      (* Int32 throughout, so that a result exceeding 32 bits WRAPS exactly as
         the device wraps, rather than being accidentally right on a 63-bit
         host int. This is the claim the whole integer path rests on. *)
      let acc = ref (to_int32 mc.(idx)) in
      for kk = 0 to sh.k - 1 do
        let av = to_int32 ma.((i * sh.k) + kk) in
        let bv = to_int32 mb.((kk * sh.n) + j) in
        acc := Int32.add !acc (Int32.mul av bv)
      done ;
      coopmat_component cfg.cfg_result (VInt32 !acc))

and coopmat_muladd_float (cfg : Sarek_coopmat_types.config) ma mb mc =
  let open Sarek_coopmat_types in
  let sh = cfg.cfg_shape in
  Array.init (sh.m * sh.n) (fun idx ->
      let i = idx / sh.n and j = idx mod sh.n in
      let acc = ref (to_float32 mc.(idx)) in
      for kk = 0 to sh.k - 1 do
        (* Exact: two binary16 significands (11 bits) multiply into 22 bits,
           inside binary32's 24, and the exponent range fits; two bfloat16
           significands (8 bits) into 16 bits, the exponent range being
           binary32's own (barring underflow of the product). *)
        let p =
          to_float32 ma.((i * sh.k) + kk) *. to_float32 mb.((kk * sh.n) + j)
        in
        acc := Sarek_float32.add !acc p
      done ;
      VFloat32 !acc)

and exec_coopmat state env op =
  if coopmat_is_leader state then exec_coopmat_once state env op

and exec_coopmat_once state env op =
  let open Sarek_coopmat_types in
  match op with
  | CM_decl {name; frag} ->
      Hashtbl.replace
        env.coopmats
        (coopmat_key state name)
        (Array.make
           (fragment_components frag)
           (coopmat_zero frag.frag_component))
  | CM_load {dst; frag; src; index; stride; layout} ->
      let buf = get_array env src in
      let base = to_int (eval_expr state env index) in
      let stride = to_int (eval_expr state env stride) in
      let rows, cols = fragment_dims frag in
      let m = Array.make (rows * cols) (coopmat_zero frag.frag_component) in
      for r = 0 to rows - 1 do
        for c = 0 to cols - 1 do
          let i = coopmat_element ~layout ~base ~stride r c in
          if i < 0 || i >= Array.length buf then
            Interp_error.raise_error
              (Array_bounds_error
                 {array_name = src; index = i; length = Array.length buf})
          else
            m.((r * cols) + c) <- coopmat_component frag.frag_component buf.(i)
        done
      done ;
      Hashtbl.replace env.coopmats (coopmat_key state dst) m
  | CM_store {src; frag; dst; index; stride; layout} ->
      let m = get_coopmat state env src in
      let buf = get_array env dst in
      let base = to_int (eval_expr state env index) in
      let stride = to_int (eval_expr state env stride) in
      let rows, cols = fragment_dims frag in
      for r = 0 to rows - 1 do
        for c = 0 to cols - 1 do
          let i = coopmat_element ~layout ~base ~stride r c in
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
      coopmat_refuse_policy env cfg ;
      let ma = get_coopmat state env a
      and mb = get_coopmat state env b
      and mc = get_coopmat state env c in
      let out =
        if accumulation_is_exact cfg then coopmat_muladd_int cfg ma mb mc
        else coopmat_muladd_float cfg ma mb mc
      in
      Hashtbl.replace env.coopmats (coopmat_key state dst) out

(* Read the value currently AT an lvalue, sharing rather than copying.

   Distinct from [eval_expr] on an equivalent expression, and the distinction is
   the whole point: a [VRecord]'s [value array] is mutable, so a shared read lets
   [assign_lvalue]'s [LRecordField] arm store into the element in place, whereas
   anything that reconstructs the record (the [to_values]/[from_value] pair, which
   goes through the OCaml record) yields a temporary and the store is lost.

   No [LVar] mutation concern: binding is by [bind_var], and this only ever reads. *)
and read_lvalue state env lv =
  match lv with
  | LVar v -> lookup_var env v
  | LArrayElem (arr, idx_expr) ->
      let a = get_array env arr in
      let i = to_int (eval_expr state env idx_expr) in
      if i < 0 || i >= Array.length a then
        Interp_error.raise_error
          (Array_bounds_error
             {array_name = arr; index = i; length = Array.length a})
      else a.(i)
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
      else a.(i)
  | LRecordField (inner, field) -> (
      (* Nested: [r.a.b <- v] reads [r.a] here, then the caller stores into [b].
         Sharing all the way down is what makes the nested store land.

         So this MUST index [fields] directly. It used to call the registry's
         [get_field], which is [H.get_field (H.from_value v)] — a round trip
         through the OCaml record, i.e. a COPY. That made the depth-1 store work
         (its base is read by another path) while silently dropping the nested
         one: [v.(i).f.g <- e] stored into a temporary that was then discarded.
         The bug survived because it only bit REGISTERED types — an unregistered
         record already took the positional branch below, which shares — so the
         synthesized/tuple cases looked correct and hid it (backlog-172).

         [field_index] comes from [HELPERS.field_names], generated from the same
         [labels] list as [to_values] and [get_field], so the position it
         returns is the position in this [value array] by construction. *)
      match read_lvalue state env inner with
      | VRecord (type_name, fields) -> (
          let idx =
            match Sarek_type_helpers.lookup type_name with
            | Some h -> h.Sarek_type_helpers.field_index field
            | None -> positional_field_index field
          in
          match idx with
          | Some i when i >= 0 && i < Array.length fields -> fields.(i)
          | _ ->
              Interp_error.raise_error
                (Not_a_record
                   {
                     expr =
                       Printf.sprintf
                         "record %s has no field %s"
                         type_name
                         field;
                   }))
      | v ->
          Interp_error.raise_error
            (Not_a_record
               {
                 expr =
                   Printf.sprintf
                     "base of .%s (got %s)"
                     field
                     (Sarek_value.value_type_name v);
               }))

and assign_lvalue state env lv value =
  (* Store values directly - VRecord is handled by ERecordField *)
  match lv with
  | LVar v -> bind_var env v value
  | LArrayElem (arr, idx_expr) ->
      let a = get_array env arr in
      let i = to_int (eval_expr state env idx_expr) in
      (* Same check as the LArrayElemExpr arm below, which had it while this one
         did not — adjacent arms of one function disagreeing about whether an
         out-of-range store is an interpreter error or an OCaml one.

         NOT a memory-safety fix, and the backlog entry that claimed it was is
         wrong: OCaml bounds-checks [a.(i) <- v] and raises
         Invalid_argument "index out of bounds", so nothing was ever corrupted.
         What was wrong is the DIAGNOSTIC. An interpreter that is the
         cross-backend oracle should say which array, which index and what
         length, the way error_to_string does — an opaque Invalid_argument names
         none of the three, and on a GPU backend the same store is genuine
         undefined behaviour, so this is the one place able to describe it. *)
      if i < 0 || i >= Array.length a then
        Interp_error.raise_error
          (Array_bounds_error
             {array_name = arr; index = i; length = Array.length a})
      else a.(i) <- value
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
  | LRecordField (base_lv, field) -> (
      (* backlog-172. [VRecord (name, fields)] holds a MUTABLE [value array], and
         reading the record out of its container hands back that same array
         rather than a copy — so mutating a slot mutates the element in place,
         which is what makes this expressible at all.

         It must be [read_lvalue] and not [eval_expr] on a reconstructed
         expression: the [to_values]/[from_value] pair round-trips through the
         OCaml record and therefore COPIES, so a store through it would land in a
         temporary and vanish. That is the shape the Native backend had, and it
         is fixed in this same change (see the [TEVecGet] arm of [TEFieldSet] in
         sarek/ppx/Sarek_native_gen_expr.ml) — by a read-modify-write back into
         vector storage rather than by an in-place mutation, because on that path
         nothing hands back the storage to mutate. *)
      match read_lvalue state env base_lv with
      | VRecord (type_name, fields) -> (
          let index =
            (* Registered types resolve through their helper and STOP. A
               registered record that has no field of this name is an error here,
               exactly as it is when the same name is read.

               THERE USED TO BE A POSITIONAL FALLBACK UNDER [Some h], and it was
               a silent wrong-field WRITE waiting to happen. [read_lvalue]'s
               [LRecordField] arm below resolves a registered type with
               [h.field_index] alone and raises when it answers [None]; this arm
               went on to try [positional_field_index]. So for a registered
               record and a field name that merely LOOKS positional — [_0] — the
               read of that same name raised "record T has no field _0" while the
               write landed in slot 0 and reported success. A store that hits the
               wrong field is strictly worse than the dropped store this arm
               exists to fix, and "the name cannot occur in practice" is an
               argument about the front end, not a property of this function.

               Removing it makes write resolution agree with read resolution, and
               that is the whole change: no currently-passing case relied on the
               fallback, because reaching it required the helper to lack the field
               (see test_interp_field_store_resolution.ml, which pins both
               polarities on a registered type).

               One divergence REMAINS and is deliberately untouched:
               [ERecordField] (reading, above) has a third tier under [None],
               [Sarek_registry.record_fields], which neither this arm nor
               [read_lvalue] has. Adding it here would change which stores
               succeed, not just which fail, and that is outside this fix. *)
            match Sarek_type_helpers.lookup type_name with
            | Some h -> h.Sarek_type_helpers.field_index field
            | None ->
                (* Synthesized tuple records (L13, [_tup_*]) use positional field
                   names [_0], [_1], … and are never in the record registry. *)
                positional_field_index field
          in
          match index with
          | Some idx when idx < Array.length fields -> fields.(idx) <- value
          | Some idx ->
              (* Registry and value disagree on the arity of the same type. Loud,
                 because the alternative is writing outside the record or
                 silently dropping the store — and a dropped store is exactly the
                 failure mode this arm replaces. *)
              Interp_error.raise_error
                (Unsupported_operation
                   {
                     operation =
                       Printf.sprintf
                         "record field assignment %s.%s"
                         type_name
                         field;
                     reason =
                       Printf.sprintf
                         "field index %d is out of range for a %d-field value"
                         idx
                         (Array.length fields);
                   })
          | None ->
              Interp_error.raise_error
                (Unsupported_operation
                   {
                     operation =
                       Printf.sprintf
                         "record field assignment %s.%s"
                         type_name
                         field;
                     reason =
                       "no such field (the type is not registered as a record, \
                        or has no field of that name)";
                   }))
      | v ->
          Interp_error.raise_error
            (Not_a_record
               {
                 expr =
                   Printf.sprintf
                     "assignment target of .%s (got %s)"
                     field
                     (Sarek_value.value_type_name v);
               }))

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
              let arr = alloc_kernel_array ty size in
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

(** A parked thread: its continuation, and whether it waits on a block-wide
    barrier (as opposed to a warp barrier). *)
type parked = (unit, unit) Effect.Deep.continuation * bool

(** The effect handler of thread [tid]: a barrier parks the thread in [waiting];
    finishing counts it in [num_completed]. *)
let thread_handler (waiting : parked option array) num_completed tid :
    (unit, unit) Effect.Deep.handler =
  {
    retc = (fun () -> incr num_completed);
    exnc = raise;
    effc =
      (fun (type a) (eff : a Effect.t) ->
        match eff with
        | Barrier ->
            Some
              (fun (k : (a, unit) Effect.Deep.continuation) ->
                waiting.(tid) <- Some (k, true))
        | Warp_barrier ->
            Some
              (fun (k : (a, unit) Effect.Deep.continuation) ->
                waiting.(tid) <- Some (k, false))
        | _ -> None);
  }

(** A block-wide barrier releases only when EVERY thread of the block has
    arrived. After a round of execution each thread is either parked or
    finished, so a block barrier with parked threads AND finished ones can never
    release: the finished threads will not arrive. Threads parked on a warp
    barrier prove nothing about the rest of the block. *)
let check_barrier_can_release (waiting : parked option array) ~num_completed =
  let at_block_barrier =
    Array.fold_left
      (fun n w -> match w with Some (_, true) -> n + 1 | _ -> n)
      0
      waiting
  in
  if at_block_barrier > 0 && num_completed > 0 then
    Interp_error.raise_error
      (BSP_deadlock
         {
           message =
             Printf.sprintf
               "%d of %d threads wait at a barrier that the %d finished \
                threads never reach"
               at_block_barrier
               (Array.length waiting)
               num_completed;
         })

(** Run all threads in a block with BSP barrier synchronization *)
let run_block env body block_idx block_dim grid_dim =
  let bx, by, bz = block_dim in
  let num_threads = bx * by * bz in
  let waiting : parked option array = Array.make num_threads None in
  let num_completed = ref 0 in

  let start_thread tid =
    let tx = tid mod bx in
    let ty = tid / bx mod by in
    let tz = tid / (bx * by) in
    let state = {thread_idx = (tx, ty, tz); block_idx; block_dim; grid_dim} in
    let thread_env = copy_env env in
    Effect.Deep.match_with
      (fun () -> exec_stmt state thread_env body)
      ()
      (thread_handler waiting num_completed tid)
  in

  (* The deep handler installed by [start_thread] persists across [continue], so
     a resumed thread that reaches another barrier performs the effect to the
     SAME handler and a thread that finishes reaches the original [retc].
     Wrapping [continue] in a second [match_with] counted every re-suspended
     thread as completed. *)
  let resume_thread tid =
    match waiting.(tid) with
    | Some (k, _) ->
        waiting.(tid) <- None ;
        Effect.Deep.continue k ()
    | None -> ()
  in

  for tid = 0 to num_threads - 1 do
    start_thread tid
  done ;
  check_barrier_can_release waiting ~num_completed:!num_completed ;
  while Array.exists Option.is_some waiting do
    for tid = 0 to num_threads - 1 do
      resume_thread tid
    done ;
    check_barrier_can_release waiting ~num_completed:!num_completed
  done

(** Run all blocks in a grid (sequential) *)
let run_grid_sequential env body block_dim grid_dim =
  let gx, gy, gz = grid_dim in
  for bz = 0 to gz - 1 do
    for by = 0 to gy - 1 do
      for bx = 0 to gx - 1 do
        Hashtbl.clear env.shared ;
        Hashtbl.clear env.coopmats ;
        run_block env body (bx, by, bz) block_dim grid_dim
      done
    done
  done
