(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Sarek_ir_interp_value

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
