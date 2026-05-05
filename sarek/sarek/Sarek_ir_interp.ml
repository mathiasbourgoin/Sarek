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
module F32 = Sarek_float32

(** Re-export value type from Sarek_value for convenience *)
type value = Sarek_value.value =
  | VInt32 of int32
  | VInt64 of int64
  | VFloat32 of float
  | VFloat64 of float
  | VBool of bool
  | VUnit
  | VArray of value array
  | VRecord of string * value array
  | VVariant of string * int * value list

(** {1 BSP Barrier Effect}
    Used for synchronizing threads at barriers. Each thread is suspended when it
    hits a barrier, and all threads are resumed together. *)

type _ Effect.t += Barrier : unit Effect.t

(** {1 Thread State} *)

type thread_state = {
  thread_idx : int * int * int;
  block_idx : int * int * int;
  block_dim : int * int * int;
  grid_dim : int * int * int;
}

(** {1 Environment} *)

type env = {
  vars : (int, value) Hashtbl.t;  (** var_id -> value *)
  vars_by_name : (string, value) Hashtbl.t;  (** var_name -> value (fallback) *)
  arrays : (string, value array) Hashtbl.t;  (** array_name -> data *)
  shared : (string, value array) Hashtbl.t;  (** shared arrays for block *)
  funcs : (string, helper_func) Hashtbl.t;  (** helper functions *)
}

let create_env () =
  {
    vars = Hashtbl.create 32;
    vars_by_name = Hashtbl.create 32;
    arrays = Hashtbl.create 16;
    shared = Hashtbl.create 8;
    funcs = Hashtbl.create 8;
  }

let copy_env env =
  {
    vars = Hashtbl.copy env.vars;
    vars_by_name = Hashtbl.copy env.vars_by_name;
    arrays = env.arrays;
    (* shared across threads *)
    shared = env.shared;
    (* shared within block *)
    funcs = env.funcs;
    (* shared *)
  }

(** Bind a variable in the environment (both by id and name) *)
let bind_var env (v : var) value =
  Hashtbl.replace env.vars v.var_id value ;
  Hashtbl.replace env.vars_by_name v.var_name value

(** Look up a variable (try id first, then name as fallback) *)
let lookup_var env (v : var) =
  match Hashtbl.find_opt env.vars v.var_id with
  | Some value -> value
  | None -> (
      match Hashtbl.find_opt env.vars_by_name v.var_name with
      | Some value -> value
      | None ->
          Interp_error.raise_error
            (Unbound_variable {name = v.var_name; context = "eval_expr"}))

(** {1 Value Operations} *)

let to_int32 = function
  | VInt32 n -> n
  | VInt64 n -> Int64.to_int32 n
  | VFloat32 f -> Int32.of_float f
  | VFloat64 f -> Int32.of_float f
  | VBool b -> if b then 1l else 0l
  | v ->
      Interp_error.raise_error
        (Type_conversion_error
           {
             from_type = Sarek_value.value_type_name v;
             to_type = "int32";
             context = "to_int32";
           })

let to_int64 = function
  | VInt64 n -> n
  | VInt32 n -> Int64.of_int32 n
  | VFloat32 f -> Int64.of_float f
  | VFloat64 f -> Int64.of_float f
  | VBool b -> if b then 1L else 0L
  | v ->
      Interp_error.raise_error
        (Type_conversion_error
           {
             from_type = Sarek_value.value_type_name v;
             to_type = "int64";
             context = "to_int64";
           })

let to_int v = Int32.to_int (to_int32 v)

let to_float32 = function
  | VFloat32 f -> f
  | VFloat64 f -> F32.to_float32 f
  | VInt32 n -> F32.to_float32 (Int32.to_float n)
  | VInt64 n -> F32.to_float32 (Int64.to_float n)
  | v ->
      Interp_error.raise_error
        (Type_conversion_error
           {
             from_type = Sarek_value.value_type_name v;
             to_type = "float32";
             context = "to_float32";
           })

let to_float64 = function
  | VFloat64 f -> f
  | VFloat32 f -> f
  | VInt32 n -> Int32.to_float n
  | VInt64 n -> Int64.to_float n
  | v ->
      Interp_error.raise_error
        (Type_conversion_error
           {
             from_type = Sarek_value.value_type_name v;
             to_type = "float64";
             context = "to_float64";
           })

let to_bool = function
  | VBool b -> b
  | VInt32 n -> n <> 0l
  | VInt64 n -> n <> 0L
  | VFloat32 f -> f <> 0.0
  | VFloat64 f -> f <> 0.0
  | v ->
      Interp_error.raise_error
        (Type_conversion_error
           {
             from_type = Sarek_value.value_type_name v;
             to_type = "bool";
             context = "to_bool";
           })

(** {1 Binary Operations} *)

let eval_binop op v1 v2 =
  match op with
  | Add -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VFloat32 (F32.add a b)
      | VFloat64 a, VFloat64 b -> VFloat64 (a +. b)
      | VFloat32 _, _ | _, VFloat32 _ ->
          VFloat32 (F32.add (to_float32 v1) (to_float32 v2))
      | VFloat64 _, _ | _, VFloat64 _ ->
          VFloat64 (to_float64 v1 +. to_float64 v2)
      | VInt64 _, _ | _, VInt64 _ ->
          VInt64 (Int64.add (to_int64 v1) (to_int64 v2))
      | _ -> VInt32 (Int32.add (to_int32 v1) (to_int32 v2)))
  | Sub -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VFloat32 (F32.sub a b)
      | VFloat64 a, VFloat64 b -> VFloat64 (a -. b)
      | VFloat32 _, _ | _, VFloat32 _ ->
          VFloat32 (F32.sub (to_float32 v1) (to_float32 v2))
      | VFloat64 _, _ | _, VFloat64 _ ->
          VFloat64 (to_float64 v1 -. to_float64 v2)
      | VInt64 _, _ | _, VInt64 _ ->
          VInt64 (Int64.sub (to_int64 v1) (to_int64 v2))
      | _ -> VInt32 (Int32.sub (to_int32 v1) (to_int32 v2)))
  | Mul -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VFloat32 (F32.mul a b)
      | VFloat64 a, VFloat64 b -> VFloat64 (a *. b)
      | VFloat32 _, _ | _, VFloat32 _ ->
          VFloat32 (F32.mul (to_float32 v1) (to_float32 v2))
      | VFloat64 _, _ | _, VFloat64 _ ->
          VFloat64 (to_float64 v1 *. to_float64 v2)
      | VInt64 _, _ | _, VInt64 _ ->
          VInt64 (Int64.mul (to_int64 v1) (to_int64 v2))
      | _ -> VInt32 (Int32.mul (to_int32 v1) (to_int32 v2)))
  | Div -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VFloat32 (F32.div a b)
      | VFloat64 a, VFloat64 b -> VFloat64 (a /. b)
      | VFloat32 _, _ | _, VFloat32 _ ->
          VFloat32 (F32.div (to_float32 v1) (to_float32 v2))
      | VFloat64 _, _ | _, VFloat64 _ ->
          VFloat64 (to_float64 v1 /. to_float64 v2)
      | VInt64 _, _ | _, VInt64 _ ->
          VInt64 (Int64.div (to_int64 v1) (to_int64 v2))
      | _ -> VInt32 (Int32.div (to_int32 v1) (to_int32 v2)))
  | Mod -> (
      match (v1, v2) with
      | VInt64 _, _ | _, VInt64 _ ->
          VInt64 (Int64.rem (to_int64 v1) (to_int64 v2))
      | _ -> VInt32 (Int32.rem (to_int32 v1) (to_int32 v2)))
  | Eq -> VBool (v1 = v2)
  | Ne -> VBool (v1 <> v2)
  | Lt -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VBool (a < b)
      | VFloat64 a, VFloat64 b -> VBool (a < b)
      | VFloat32 _, _ | _, VFloat32 _ -> VBool (to_float32 v1 < to_float32 v2)
      | VFloat64 _, _ | _, VFloat64 _ -> VBool (to_float64 v1 < to_float64 v2)
      | _ -> VBool (to_int32 v1 < to_int32 v2))
  | Le -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VBool (a <= b)
      | VFloat64 a, VFloat64 b -> VBool (a <= b)
      | VFloat32 _, _ | _, VFloat32 _ -> VBool (to_float32 v1 <= to_float32 v2)
      | VFloat64 _, _ | _, VFloat64 _ -> VBool (to_float64 v1 <= to_float64 v2)
      | _ -> VBool (to_int32 v1 <= to_int32 v2))
  | Gt -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VBool (a > b)
      | VFloat64 a, VFloat64 b -> VBool (a > b)
      | VFloat32 _, _ | _, VFloat32 _ -> VBool (to_float32 v1 > to_float32 v2)
      | VFloat64 _, _ | _, VFloat64 _ -> VBool (to_float64 v1 > to_float64 v2)
      | _ -> VBool (to_int32 v1 > to_int32 v2))
  | Ge -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VBool (a >= b)
      | VFloat64 a, VFloat64 b -> VBool (a >= b)
      | VFloat32 _, _ | _, VFloat32 _ -> VBool (to_float32 v1 >= to_float32 v2)
      | VFloat64 _, _ | _, VFloat64 _ -> VBool (to_float64 v1 >= to_float64 v2)
      | _ -> VBool (to_int32 v1 >= to_int32 v2))
  | And -> VBool (to_bool v1 && to_bool v2)
  | Or -> VBool (to_bool v1 || to_bool v2)
  | Shl -> VInt32 (Int32.shift_left (to_int32 v1) (to_int v2))
  | Shr -> VInt32 (Int32.shift_right_logical (to_int32 v1) (to_int v2))
  | BitAnd -> VInt32 (Int32.logand (to_int32 v1) (to_int32 v2))
  | BitOr -> VInt32 (Int32.logor (to_int32 v1) (to_int32 v2))
  | BitXor -> VInt32 (Int32.logxor (to_int32 v1) (to_int32 v2))

let eval_unop op v =
  match op with
  | Neg -> (
      match v with
      | VFloat32 f -> VFloat32 (-.f)
      | VFloat64 f -> VFloat64 (-.f)
      | VInt64 n -> VInt64 (Int64.neg n)
      | _ -> VInt32 (Int32.neg (to_int32 v)))
  | Not -> VBool (not (to_bool v))
  | BitNot -> VInt32 (Int32.lognot (to_int32 v))

(** {1 Intrinsics} *)

let is_gpu_path = function
  | ["Gpu"] | [] | ["Std"] | ["Sarek_stdlib"; "Gpu"] | ["Sarek_stdlib"; "Std"]
    ->
      true
  | _ -> false

let is_float32_path = function
  | ["Float32"] | ["Sarek_stdlib"; "Float32"] -> true
  | _ -> false

let is_float64_path = function
  | ["Float64"] | ["Sarek_stdlib"; "Float64"] -> true
  | _ -> false

let is_int32_path = function
  | ["Int32"] | ["Sarek_stdlib"; "Int32"] -> true
  | _ -> false

(** {1 Intrinsic Evaluation Helpers}

    These helper functions handle specific categories of GPU intrinsics. Split
    from a monolithic eval_intrinsic for better maintainability and testability.
    Each helper returns Option to enable clean dispatch logic. *)

(** Evaluate GPU thread/block/grid index and dimension intrinsics.

    Handles the complete GPU execution model intrinsics:
    - Thread indices: Position of thread within its block (0 to block_dim-1)
    - Block indices: Position of block within grid (0 to grid_dim-1)
    - Dimensions: Size of blocks and grid
    - Global indices: Thread's global position across entire grid
    - Global sizes: Total number of threads in each dimension

    @param state Thread execution state containing all index/dimension values
    @param name Intrinsic name (e.g. "thread_idx_x", "global_size_y")
    @return Some value if intrinsic matches, None otherwise

    Example: eval_gpu_index_intrinsic state "global_idx_x" (* Returns: VInt32
    (block_idx_x * block_dim_x + thread_idx_x) *) *)

(** GPU thread/block/grid indices and dimensions *)
let eval_gpu_index_intrinsic state name =
  match name with
  (* Thread indices *)
  | "thread_idx_x" ->
      let x, _, _ = state.thread_idx in
      Some (VInt32 (Int32.of_int x))
  | "thread_idx_y" ->
      let _, y, _ = state.thread_idx in
      Some (VInt32 (Int32.of_int y))
  | "thread_idx_z" ->
      let _, _, z = state.thread_idx in
      Some (VInt32 (Int32.of_int z))
  (* Block indices *)
  | "block_idx_x" ->
      let x, _, _ = state.block_idx in
      Some (VInt32 (Int32.of_int x))
  | "block_idx_y" ->
      let _, y, _ = state.block_idx in
      Some (VInt32 (Int32.of_int y))
  | "block_idx_z" ->
      let _, _, z = state.block_idx in
      Some (VInt32 (Int32.of_int z))
  (* Block dimensions *)
  | "block_dim_x" ->
      let x, _, _ = state.block_dim in
      Some (VInt32 (Int32.of_int x))
  | "block_dim_y" ->
      let _, y, _ = state.block_dim in
      Some (VInt32 (Int32.of_int y))
  | "block_dim_z" ->
      let _, _, z = state.block_dim in
      Some (VInt32 (Int32.of_int z))
  (* Grid dimensions *)
  | "grid_dim_x" ->
      let x, _, _ = state.grid_dim in
      Some (VInt32 (Int32.of_int x))
  | "grid_dim_y" ->
      let _, y, _ = state.grid_dim in
      Some (VInt32 (Int32.of_int y))
  | "grid_dim_z" ->
      let _, _, z = state.grid_dim in
      Some (VInt32 (Int32.of_int z))
  (* Global index helpers *)
  | "global_idx" | "global_idx_x" | "global_thread_id" ->
      let tx, _, _ = state.thread_idx in
      let bx, _, _ = state.block_idx in
      let bdx, _, _ = state.block_dim in
      Some (VInt32 (Int32.of_int ((bx * bdx) + tx)))
  | "global_idx_y" ->
      let _, ty, _ = state.thread_idx in
      let _, by, _ = state.block_idx in
      let _, bdy, _ = state.block_dim in
      Some (VInt32 (Int32.of_int ((by * bdy) + ty)))
  | "global_idx_z" ->
      let _, _, tz = state.thread_idx in
      let _, _, bz = state.block_idx in
      let _, _, bdz = state.block_dim in
      Some (VInt32 (Int32.of_int ((bz * bdz) + tz)))
  (* Global size helpers *)
  | "global_size" | "global_size_x" ->
      let bdx, _, _ = state.block_dim in
      let gdx, _, _ = state.grid_dim in
      Some (VInt32 (Int32.of_int (bdx * gdx)))
  | "global_size_y" ->
      let _, bdy, _ = state.block_dim in
      let _, gdy, _ = state.grid_dim in
      Some (VInt32 (Int32.of_int (bdy * gdy)))
  | "global_size_z" ->
      let _, _, bdz = state.block_dim in
      let _, _, gdz = state.grid_dim in
      Some (VInt32 (Int32.of_int (bdz * gdz)))
  | _ -> None

(** Barrier synchronization intrinsics *)
let eval_barrier_intrinsic name =
  match name with
  | "block_barrier" | "warp_barrier" ->
      Effect.perform Barrier ;
      Some VUnit
  | _ -> None

(** Float32 math intrinsics *)
let eval_float32_math_intrinsic name args =
  match name with
  | "sin" -> (
      match args with
      | arg :: _ -> Some (VFloat32 (F32.sin (to_float32 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "sin"; reason = "requires 1 argument"}))
  | "cos" -> (
      match args with
      | arg :: _ -> Some (VFloat32 (F32.cos (to_float32 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "cos"; reason = "requires 1 argument"}))
  | "tan" -> (
      match args with
      | arg :: _ -> Some (VFloat32 (F32.tan (to_float32 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "tan"; reason = "requires 1 argument"}))
  | "sqrt" -> (
      match args with
      | arg :: _ -> Some (VFloat32 (F32.sqrt (to_float32 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "sqrt"; reason = "requires 1 argument"}))
  | "exp" -> (
      match args with
      | arg :: _ -> Some (VFloat32 (F32.exp (to_float32 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "exp"; reason = "requires 1 argument"}))
  | "log" -> (
      match args with
      | arg :: _ -> Some (VFloat32 (F32.log (to_float32 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "log"; reason = "requires 1 argument"}))
  | "abs" -> (
      match args with
      | arg :: _ -> Some (VFloat32 (F32.abs (to_float32 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "abs"; reason = "requires 1 argument"}))
  | "floor" -> (
      match args with
      | arg :: _ -> Some (VFloat32 (F32.floor (to_float32 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "floor"; reason = "requires 1 argument"}))
  | "ceil" -> (
      match args with
      | arg :: _ -> Some (VFloat32 (F32.ceil (to_float32 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "ceil"; reason = "requires 1 argument"}))
  | "pow" -> (
      match args with
      | arg1 :: arg2 :: _ ->
          Some (VFloat32 (F32.pow (to_float32 arg1) (to_float32 arg2)))
      | _ ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "pow"; reason = "requires 2 arguments"}))
  | "min" -> (
      match args with
      | arg1 :: arg2 :: _ ->
          Some (VFloat32 (F32.min (to_float32 arg1) (to_float32 arg2)))
      | _ ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "min"; reason = "requires 2 arguments"}))
  | "max" -> (
      match args with
      | arg1 :: arg2 :: _ ->
          Some (VFloat32 (F32.max (to_float32 arg1) (to_float32 arg2)))
      | _ ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "max"; reason = "requires 2 arguments"}))
  | "of_int" -> (
      match args with
      | arg :: _ -> Some (VFloat32 (F32.of_int (to_int arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "of_int"; reason = "requires 1 argument"}))
  | _ -> None

(** Float64 math intrinsics *)
let eval_float64_math_intrinsic name args =
  match name with
  | "sin" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (sin (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "sin (float64)"; reason = "requires 1 argument"}))
  | "cos" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (cos (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "cos (float64)"; reason = "requires 1 argument"}))
  | "sqrt" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (sqrt (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "sqrt (float64)"; reason = "requires 1 argument"}))
  | "exp" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (exp (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "exp (float64)"; reason = "requires 1 argument"}))
  | "log" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (log (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "log (float64)"; reason = "requires 1 argument"}))
  | "abs" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (Float.abs (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "abs (float64)"; reason = "requires 1 argument"}))
  | "of_int" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (Float.of_int (to_int arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "of_int (float64)"; reason = "requires 1 argument"})
      )
  | _ -> None

(** Int32 math intrinsics *)
let eval_int32_math_intrinsic name args =
  match name with
  | "abs" -> (
      match args with
      | arg :: _ -> Some (VInt32 (Int32.abs (to_int32 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "abs (int32)"; reason = "requires 1 argument"}))
  | "min" -> (
      match args with
      | arg1 :: arg2 :: _ -> Some (VInt32 (min (to_int32 arg1) (to_int32 arg2)))
      | _ ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "min (int32)"; reason = "requires 2 arguments"}))
  | "max" -> (
      match args with
      | arg1 :: arg2 :: _ -> Some (VInt32 (max (to_int32 arg1) (to_int32 arg2)))
      | _ ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "max (int32)"; reason = "requires 2 arguments"}))
  | _ -> None

(** Type conversion intrinsics *)
let eval_type_conversion_intrinsic name args =
  match name with
  | "float" -> (
      match args with
      | arg :: _ -> Some (VFloat32 (F32.of_int (to_int arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "float"; reason = "requires 1 argument"}))
  | "float64" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (Float.of_int (to_int arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "float64"; reason = "requires 1 argument"}))
  | "int_of_float" -> (
      match args with
      | arg :: _ -> Some (VInt32 (Int32.of_float (to_float32 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "int_of_float"; reason = "requires 1 argument"}))
  | "int_of_float64" -> (
      match args with
      | arg :: _ -> Some (VInt32 (Int32.of_float (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "int_of_float64"; reason = "requires 1 argument"}))
  | _ -> None

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
          | Some h ->
              h.get_field vrec field
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

    These functions work with typed Kernel_arg.t values. This
    is the preferred interface for Native/Interpreter backends. *)

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
        Spoc_core.Vector.set
          vec
          i
          (Char.chr (Int32.to_int (to_int32 arr.(i))))
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
  | Exec_writeback : (module Spoc_framework.Typed_value.EXEC_VECTOR) * value array -> exec_writeback

let value_of_typed_value (tv : Spoc_framework.Typed_value.typed_value) : value =
  match tv with
  | TV_Scalar (SV ((module S), x)) -> (
      match S.to_primitive x with
      | PInt32 n -> VInt32 n
      | PInt64 n -> VInt64 n
      | PFloat f ->
          if S.name = "float64" then VFloat64 f else VFloat32 f
      | PBool b -> VBool b
      | PBytes _ ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "value_of_typed_value"; reason = "PBytes scalar"}))
  | TV_Composite (CV ((module C), _x)) ->
      VRecord (C.name, [||])

let typed_value_of_value (type a)
    (module V : Spoc_framework.Typed_value.EXEC_VECTOR with type elt = a)
    (v : value) : Spoc_framework.Typed_value.typed_value option =
  match v with
  | VInt32 n ->
      Some
        (TV_Scalar (SV ((module Spoc_framework.Typed_value.Int32_type), n)))
  | VInt64 n ->
      Some
        (TV_Scalar (SV ((module Spoc_framework.Typed_value.Int64_type), n)))
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

let array_to_exec_vector
    (module V : Spoc_framework.Typed_value.EXEC_VECTOR)
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
  let arg_at i = if i >= List.length exec_args then None else Some (List.nth exec_args i) in
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
