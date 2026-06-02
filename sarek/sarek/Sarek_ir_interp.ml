(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_ir_interp - CPU interpreter for Sarek V2 IR kernels
 *
 * Executes Sarek_ir.kernel on CPU for debugging and testing without GPU.
 * Supports BSP-style barrier synchronization using OCaml 5 effects.
 *
 * This is the V2 counterpart to Sarek_interp (which works on Kirc_Ast).
 ******************************************************************************)

open Sarek_ir_types
open Spoc_framework.Typed_value
open Sarek_ir_interp_eval

(** Re-export value type and constructors for external callers *)
type value = Sarek_ir_interp_value.value =
  | VInt32 of int32
  | VInt64 of int64
  | VFloat32 of float
  | VFloat64 of float
  | VBool of bool
  | VUnit
  | VArray of value array
  | VRecord of string * value array
  | VVariant of string * int * value list

let to_int = Sarek_ir_interp_value.to_int

let to_int32 = Sarek_ir_interp_value.to_int32

let to_int64 = Sarek_ir_interp_value.to_int64

let to_float32 = Sarek_ir_interp_value.to_float32

let to_float64 = Sarek_ir_interp_value.to_float64

let to_bool = Sarek_ir_interp_value.to_bool

(** Re-export thread_state type for external callers *)
type thread_state = Sarek_ir_interp_value.thread_state = {
  thread_idx : int * int * int;
  block_idx : int * int * int;
  block_dim : int * int * int;
  grid_dim : int * int * int;
}

(** Re-export env type for external callers *)
type env = Sarek_ir_interp_value.env = {
  vars : (int, value) Hashtbl.t;
  vars_by_name : (string, value) Hashtbl.t;
  arrays : (string, value array) Hashtbl.t;
  shared : (string, value array) Hashtbl.t;
  funcs : (string, Sarek_ir_types.helper_func) Hashtbl.t;
}

let create_env = Sarek_ir_interp_value.create_env

let copy_env = Sarek_ir_interp_value.copy_env

let bind_var = Sarek_ir_interp_value.bind_var

let lookup_var = Sarek_ir_interp_value.lookup_var

let eval_binop = Sarek_ir_interp_value.eval_binop

let eval_unop = Sarek_ir_interp_value.eval_unop

let eval_expr = Sarek_ir_interp_eval.eval_expr

let exec_stmt = Sarek_ir_interp_eval.exec_stmt

(** {1 Domain Pool for Parallel Execution} *)

module DomainPool = struct
  type task = unit -> unit

  type t = {
    num_domains : int;
    task_queue : task Queue.t;
    mutex : Mutex.t;
    cond : Condition.t;
    mutable shutdown : bool;
    domains : unit Domain.t array;
    mutable active_tasks : int;
    done_cond : Condition.t;
  }

  let worker pool =
    let rec loop () =
      Mutex.lock pool.mutex ;
      while Queue.is_empty pool.task_queue && not pool.shutdown do
        Condition.wait pool.cond pool.mutex
      done ;
      if pool.shutdown && Queue.is_empty pool.task_queue then begin
        Mutex.unlock pool.mutex
      end
      else begin
        let task = Queue.pop pool.task_queue in
        pool.active_tasks <- pool.active_tasks + 1 ;
        Mutex.unlock pool.mutex ;
        (try task () with _ -> ()) ;
        Mutex.lock pool.mutex ;
        pool.active_tasks <- pool.active_tasks - 1 ;
        if pool.active_tasks = 0 && Queue.is_empty pool.task_queue then
          Condition.broadcast pool.done_cond ;
        Mutex.unlock pool.mutex ;
        loop ()
      end
    in
    loop ()

  let create num_domains =
    let pool =
      {
        num_domains;
        task_queue = Queue.create ();
        mutex = Mutex.create ();
        cond = Condition.create ();
        shutdown = false;
        domains = [||];
        active_tasks = 0;
        done_cond = Condition.create ();
      }
    in
    let domains =
      Array.init num_domains (fun _ -> Domain.spawn (fun () -> worker pool))
    in
    {pool with domains}

  let submit pool task =
    Mutex.lock pool.mutex ;
    Queue.add task pool.task_queue ;
    Condition.signal pool.cond ;
    Mutex.unlock pool.mutex

  let wait_all pool =
    Mutex.lock pool.mutex ;
    while pool.active_tasks > 0 || not (Queue.is_empty pool.task_queue) do
      Condition.wait pool.done_cond pool.mutex
    done ;
    Mutex.unlock pool.mutex
end

(** Global pool - lazily initialized *)
let global_pool : DomainPool.t option ref = ref None

let get_pool () =
  match !global_pool with
  | Some pool -> pool
  | None ->
      let num_cores = try Domain.recommended_domain_count () with _ -> 4 in
      let pool = DomainPool.create num_cores in
      global_pool := Some pool ;
      pool

(** Run all blocks in a grid (parallel - distributes blocks across domain pool)
*)
let run_grid_parallel env body block_dim grid_dim =
  let pool = get_pool () in
  let gx, gy, gz = grid_dim in
  for bz = 0 to gz - 1 do
    for by = 0 to gy - 1 do
      for bx = 0 to gx - 1 do
        (* Shadow loop vars with local bindings to capture values, not refs *)
        let bx = bx and by = by and bz = bz in
        DomainPool.submit pool (fun () ->
            (* Each block gets its own NEW shared memory hashtable.
               Don't use copy_env.shared because it's shared by reference. *)
            let block_env = {(copy_env env) with shared = Hashtbl.create 8} in
            run_block block_env body (bx, by, bz) block_dim grid_dim)
      done
    done
  done ;
  DomainPool.wait_all pool

(** Parallel execution mode flag *)
let parallel_mode = ref true

(** Run all blocks in a grid (uses parallel or sequential based on flag) *)
let run_grid env body block_dim grid_dim =
  if !parallel_mode then run_grid_parallel env body block_dim grid_dim
  else run_grid_sequential env body block_dim grid_dim

(** {1 Public API} *)

(** Argument for kernel execution *)
type arg = ArgArray of value array | ArgScalar of value

(** Run a kernel on CPU *)
let run_kernel (k : kernel) ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    (args : (string * arg) list) =
  let env = create_env () in

  (* Register helper functions *)
  List.iter (fun hf -> Hashtbl.add env.funcs hf.hf_name hf) k.kern_funcs ;

  (* Bind parameters *)
  List.iter2
    (fun decl (name, arg) ->
      match (decl, arg) with
      | DParam (v, Some _), ArgArray arr ->
          Hashtbl.add env.arrays name arr ;
          bind_var env v (VArray arr)
      | DParam (v, None), ArgScalar value -> bind_var env v value
      | DShared (name, ty, Some size_expr), _ ->
          let dummy_state =
            {
              thread_idx = (0, 0, 0);
              block_idx = (0, 0, 0);
              block_dim = (bx, by, bz);
              grid_dim = (gx, gy, gz);
            }
          in
          let size = to_int (eval_expr dummy_state env size_expr) in
          let init =
            match ty with
            | TInt32 -> VInt32 0l
            | TFloat32 -> VFloat32 0.0
            | _ -> VUnit
          in
          Hashtbl.add env.shared name (Array.make size init)
      | _ -> ())
    k.kern_params
    args ;

  run_grid env k.kern_body (bx, by, bz) (gx, gy, gz)

(** {1 V2 Vector Support}

    These functions work with typed Kernel_arg.t values. This is the preferred
    interface for Native/Interpreter backends. *)

(** Convert V2 Vector to interpreter value array. Uses the vector's element type
    to create properly typed values. *)
let vector_to_array : type a b. (a, b) Spoc_core.Vector.t -> value array =
 fun vec ->
  let len = Spoc_core.Vector.length vec in
  match Spoc_core.Vector.kind vec with
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Int32 ->
      Array.init len (fun i -> VInt32 (Spoc_core.Vector.get vec i))
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Int64 ->
      Array.init len (fun i -> VInt64 (Spoc_core.Vector.get vec i))
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Float32 ->
      Array.init len (fun i -> VFloat32 (Spoc_core.Vector.get vec i))
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Float64 ->
      Array.init len (fun i -> VFloat64 (Spoc_core.Vector.get vec i))
  | Spoc_core.Vector.Custom custom -> (
      (* Custom types: use helpers to convert to VRecord *)
      let type_name = custom.Spoc_core.Vector.name in
      match Sarek_type_helpers.lookup_typed custom.Spoc_core.Vector.type_id with
      | Some (module H) ->
          Array.init len (fun i ->
              let native_record = Spoc_core.Vector.get vec i in
              H.to_value native_record)
      | None ->
          (* Fallback: wrap in VRecord with empty fields - shouldn't happen *)
          Array.init len (fun _i -> VRecord (type_name, [||])))
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Char ->
      Array.init len (fun i ->
          VInt32 (Int32.of_int (Char.code (Spoc_core.Vector.get vec i))))
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Complex32 ->
      Interp_error.raise_error
        (Unsupported_operation
           {
             operation = "vector_to_array";
             reason = "Complex32 vectors are not supported by the interpreter";
           })

(** Write interpreter value array back to V2 Vector *)
let array_to_vector : type a b. value array -> (a, b) Spoc_core.Vector.t -> unit
    =
 fun arr vec ->
  let len = min (Array.length arr) (Spoc_core.Vector.length vec) in
  match Spoc_core.Vector.kind vec with
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Int32 ->
      for i = 0 to len - 1 do
        Spoc_core.Vector.set vec i (to_int32 arr.(i))
      done
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Int64 ->
      for i = 0 to len - 1 do
        Spoc_core.Vector.set vec i (to_int64 arr.(i))
      done
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Float32 ->
      for i = 0 to len - 1 do
        Spoc_core.Vector.set vec i (to_float32 arr.(i))
      done
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Float64 ->
      for i = 0 to len - 1 do
        Spoc_core.Vector.set vec i (to_float64 arr.(i))
      done
  | Spoc_core.Vector.Custom _ ->
      (* Custom types: convert VRecord to native OCaml values using helpers *)
      for i = 0 to len - 1 do
        match arr.(i) with
        | VRecord (type_name, _fields) as vrec -> (
            (* All [@@sarek.type] records have helpers - this should always succeed *)
            match
              Sarek_type_helpers.lookup_typed
                (Spoc_core.Vector.type_id (Spoc_core.Vector.kind vec))
            with
            | Some (module H) ->
                (* Use generated helper for type-safe conversion *)
                let native_record = H.from_value vrec in
                Spoc_core.Vector.set vec i native_record
            | None ->
                Interp_error.raise_error
                  (Unsupported_operation
                     {
                       operation = "vector_to_array";
                       reason =
                         Printf.sprintf
                           "No helper found for type '%s'. Did you forget \
                            [@@sarek.type]?"
                           type_name;
                     }))
        | _ -> () (* Skip other values *)
      done
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Char ->
      for i = 0 to len - 1 do
        Spoc_core.Vector.set vec i (Char.chr (Int32.to_int (to_int32 arr.(i))))
      done
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Complex32 ->
      Interp_error.raise_error
        (Unsupported_operation
           {
             operation = "array_to_vector";
             reason = "Complex32 vectors are not supported by the interpreter";
           })

(** Existential wrapper to track V2 Vector + its interpreter array for writeback
*)
type writeback =
  | Writeback : (('a, 'b) Spoc_core.Vector.t * value array) -> writeback

type exec_writeback =
  | Exec_writeback :
      (module Spoc_framework.Typed_value.EXEC_VECTOR) * value array
      -> exec_writeback

let value_of_typed_value (tv : Spoc_framework.Typed_value.typed_value) : value =
  match tv with
  | TV_Scalar (SV ((module S), x)) -> (
      match S.to_primitive x with
      | PInt32 n -> VInt32 n
      | PInt64 n -> VInt64 n
      | PFloat f -> if S.name = "float64" then VFloat64 f else VFloat32 f
      | PBool b -> VBool b
      | PBytes _ ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "value_of_typed_value"; reason = "PBytes scalar"}))
  | TV_Composite (CV ((module C), _x)) -> VRecord (C.name, [||])

let typed_value_of_value (type a)
    (module V : Spoc_framework.Typed_value.EXEC_VECTOR with type elt = a)
    (v : value) : Spoc_framework.Typed_value.typed_value option =
  match v with
  | VInt32 n ->
      Some (TV_Scalar (SV ((module Spoc_framework.Typed_value.Int32_type), n)))
  | VInt64 n ->
      Some (TV_Scalar (SV ((module Spoc_framework.Typed_value.Int64_type), n)))
  | VFloat32 f ->
      Some
        (TV_Scalar (SV ((module Spoc_framework.Typed_value.Float32_type), f)))
  | VFloat64 f ->
      Some
        (TV_Scalar (SV ((module Spoc_framework.Typed_value.Float64_type), f)))
  | VBool b ->
      Some (TV_Scalar (SV ((module Spoc_framework.Typed_value.Bool_type), b)))
  | VUnit | VArray _ | VRecord _ | VVariant _ -> None

let exec_vector_to_array (module V : Spoc_framework.Typed_value.EXEC_VECTOR) :
    value array =
  Array.init V.length (fun i ->
      match Sarek_type_helpers.lookup_typed V.type_id with
      | Some (module H) -> H.to_value (V.get_typed i)
      | None -> value_of_typed_value (V.get i))

let array_to_exec_vector (module V : Spoc_framework.Typed_value.EXEC_VECTOR)
    (arr : value array) : unit =
  let len = min (Array.length arr) V.length in
  for i = 0 to len - 1 do
    match arr.(i) with
    | VRecord _ as vrec -> (
        match Sarek_type_helpers.lookup_typed V.type_id with
        | Some (module H) -> V.set_typed i (H.from_value vrec)
        | None -> ())
    | scalar -> (
        match typed_value_of_value (module V) scalar with
        | Some tv -> V.set i tv
        | None -> ())
  done

let args_from_exec_args (k : kernel)
    (exec_args : Spoc_framework.Framework_sig.exec_arg list) :
    (string * arg) list * exec_writeback list =
  let writebacks = ref [] in
  let idx = ref 0 in
  let arg_at i =
    if i >= List.length exec_args then None else Some (List.nth exec_args i)
  in
  let args =
    List.filter_map
      (fun decl ->
        match decl with
        | DParam (v, Some _arr_info) -> (
            match arg_at !idx with
            | None -> None
            | Some karg -> (
                incr idx ;
                match karg with
                | Spoc_framework.Framework_sig.EA_Vec vec ->
                    let arr = exec_vector_to_array vec in
                    writebacks := Exec_writeback (vec, arr) :: !writebacks ;
                    Some (v.var_name, ArgArray arr)
                | _ ->
                    Interp_error.raise_error
                      (Type_conversion_error
                         {
                           from_type = "scalar";
                           to_type = "Vec";
                           context = "param " ^ v.var_name;
                         })))
        | DParam (v, None) -> (
            match arg_at !idx with
            | None -> None
            | Some karg ->
                incr idx ;
                let value =
                  match karg with
                  | Spoc_framework.Framework_sig.EA_Int32 n -> VInt32 n
                  | Spoc_framework.Framework_sig.EA_Int64 n -> VInt64 n
                  | Spoc_framework.Framework_sig.EA_Float32 f -> VFloat32 f
                  | Spoc_framework.Framework_sig.EA_Float64 f -> VFloat64 f
                  | Spoc_framework.Framework_sig.EA_Scalar ((module S), x) ->
                      value_of_typed_value (TV_Scalar (SV ((module S), x)))
                  | Spoc_framework.Framework_sig.EA_Composite ((module C), x) ->
                      value_of_typed_value (TV_Composite (CV ((module C), x)))
                  | Spoc_framework.Framework_sig.EA_Vec _ ->
                      Interp_error.raise_error
                        (Type_conversion_error
                           {
                             from_type = "non-scalar";
                             to_type = "scalar";
                             context = "param " ^ v.var_name;
                           })
                in
                Some (v.var_name, ArgScalar value))
        | DShared _ | DLocal _ -> None)
      k.kern_params
  in
  (args, List.rev !writebacks)

let run_kernel_with_exec_args (k : kernel) ~(block : int * int * int)
    ~(grid : int * int * int)
    (exec_args : Spoc_framework.Framework_sig.exec_arg list) : unit =
  let args, writebacks = args_from_exec_args k exec_args in
  run_kernel k ~block ~grid args ;
  List.iter
    (fun (Exec_writeback (vec, arr)) -> array_to_exec_vector vec arr)
    writebacks

(** Convert Kernel_arg.t list to interpreter args, tracking vectors for
    writeback *)
let args_from_kernel_args (k : kernel) (kargs : Spoc_core.Kernel_arg.t list) :
    (string * arg) list * writeback list =
  let writebacks = ref [] in
  let idx = ref 0 in
  let args =
    List.filter_map
      (fun decl ->
        match decl with
        | DParam (v, Some _arr_info) ->
            (* Vector parameter: expects a Vec in Kernel_arg *)
            if !idx >= List.length kargs then None
            else begin
              let karg = List.nth kargs !idx in
              incr idx ;
              match karg with
              | Spoc_core.Kernel_arg.Vec vec ->
                  let arr = vector_to_array vec in
                  writebacks := Writeback (vec, arr) :: !writebacks ;
                  Some (v.var_name, ArgArray arr)
              | _ ->
                  Interp_error.raise_error
                    (Type_conversion_error
                       {
                         from_type = "scalar";
                         to_type = "Vec";
                         context = "param " ^ v.var_name;
                       })
            end
        | DParam (v, None) ->
            (* Scalar parameter *)
            if !idx >= List.length kargs then None
            else begin
              let karg = List.nth kargs !idx in
              incr idx ;
              let value =
                match karg with
                | Spoc_core.Kernel_arg.Int n -> VInt32 (Int32.of_int n)
                | Spoc_core.Kernel_arg.Int32 n -> VInt32 n
                | Spoc_core.Kernel_arg.Int64 n -> VInt64 n
                | Spoc_core.Kernel_arg.Float32 f -> VFloat32 f
                | Spoc_core.Kernel_arg.Float64 f -> VFloat64 f
                | Spoc_core.Kernel_arg.Vec _ ->
                    Interp_error.raise_error
                      (Type_conversion_error
                         {
                           from_type = "non-scalar";
                           to_type = "scalar";
                           context = "param " ^ v.var_name;
                         })
              in
              Some (v.var_name, ArgScalar value)
            end
        | DShared _ -> None
        | DLocal _ -> None)
      k.kern_params
  in
  (args, List.rev !writebacks)

(** Run kernel with V2 Vector arguments (Kernel_arg.t list). This is the
    preferred entry point for Native/Interpreter backends. Handles conversion
    to/from interpreter format with proper writeback. *)
let run_kernel_with_args (k : kernel) ~(block : int * int * int)
    ~(grid : int * int * int) (kargs : Spoc_core.Kernel_arg.t list) : unit =
  let args, writebacks = args_from_kernel_args k kargs in
  run_kernel k ~block ~grid args ;
  (* Write modified arrays back to V2 Vectors *)
  List.iter (fun (Writeback (vec, arr)) -> array_to_vector arr vec) writebacks
