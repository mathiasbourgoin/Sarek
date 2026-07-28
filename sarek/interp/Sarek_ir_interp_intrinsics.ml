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

(* Serialises global-memory atomics. The sequential interpreter needs no lock,
   but the parallel interpreter distributes blocks across a domain pool sharing
   the same global VArray, so the read-modify-write must be atomic. *)
let atomic_global_mutex = Mutex.create ()

let with_atomic_lock f =
  Mutex.lock atomic_global_mutex ;
  Fun.protect ~finally:(fun () -> Mutex.unlock atomic_global_mutex) f

(* Bounds-checked index into an atomic's target array, mirroring
   eval_array_expr's Array_bounds_error path so the interpreter oracle rejects
   out-of-range atomics the same way an ordinary array access would. *)

(** Global-memory atomics and memory fences.

    Atomics on global vectors return the OLD value and update in place. The
    interpreter models global memory as a shared mutable [VArray]; fences are
    no-ops because a single logical memory is already sequentially consistent
    here. Sufficient for the Sarek_worklist queue pattern (atomic HEAD/TAIL
    counters + ring slots). *)
let atomic_index name a idx =
  let i = to_int idx in
  if i < 0 || i >= Array.length a then
    Interp_error.raise_error
      (Array_bounds_error
         {array_name = name; index = i; length = Array.length a}) ;
  i

let atomic_arity name n _args =
  Interp_error.raise_error
    (Unsupported_operation
       {operation = name; reason = Printf.sprintf "requires %d arguments" n})

let eval_atomic_intrinsic name args =
  match (name, args) with
  | "atomic_add_global_int32", [VArray a; idx; delta] ->
      with_atomic_lock (fun () ->
          let i = atomic_index "atomic_add_global_int32" a idx in
          let old = to_int32 a.(i) in
          a.(i) <- VInt32 (Int32.add old (to_int32 delta)) ;
          Some (VInt32 old))
  | "atomic_add_global_int32", _ -> atomic_arity name 3 args
  | "atomic_inc_global_int32", [VArray a; idx] ->
      with_atomic_lock (fun () ->
          let i = atomic_index "atomic_inc_global_int32" a idx in
          let old = to_int32 a.(i) in
          a.(i) <- VInt32 (Int32.add old 1l) ;
          Some (VInt32 old))
  | "atomic_inc_global_int32", _ -> atomic_arity name 2 args
  | ("memory_fence_block" | "memory_fence_device"), _ -> Some VUnit
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
  | "asin" -> (
      match args with
      | arg :: _ -> Some (VFloat32 (F32.asin (to_float32 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "asin"; reason = "requires 1 argument"}))
  | "acos" -> (
      match args with
      | arg :: _ -> Some (VFloat32 (F32.acos (to_float32 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "acos"; reason = "requires 1 argument"}))
  | "atan" -> (
      match args with
      | arg :: _ -> Some (VFloat32 (F32.atan (to_float32 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "atan"; reason = "requires 1 argument"}))
  | "atan2" -> (
      match args with
      | arg1 :: arg2 :: _ ->
          Some (VFloat32 (F32.atan2 (to_float32 arg1) (to_float32 arg2)))
      | _ ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "atan2"; reason = "requires 2 arguments"}))
  | "sinh" -> (
      match args with
      | arg :: _ -> Some (VFloat32 (F32.sinh (to_float32 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "sinh"; reason = "requires 1 argument"}))
  | "cosh" -> (
      match args with
      | arg :: _ -> Some (VFloat32 (F32.cosh (to_float32 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "cosh"; reason = "requires 1 argument"}))
  | "tanh" -> (
      match args with
      | arg :: _ -> Some (VFloat32 (F32.tanh (to_float32 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "tanh"; reason = "requires 1 argument"}))
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
  | "fma" -> (
      match args with
      | arg1 :: arg2 :: arg3 :: _ ->
          Some
            (VFloat32
               (F32.fma (to_float32 arg1) (to_float32 arg2) (to_float32 arg3)))
      | _ ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "fma"; reason = "requires 3 arguments"}))
  | "fmod" -> (
      (* C fmod = OCaml Float.rem (sign of dividend, magnitude < |divisor|),
         rounded back to float32. Matches Sarek_ir_interp_value's float Mod. *)
      match args with
      | arg1 :: arg2 :: _ ->
          Some
            (VFloat32
               (F32.to_float32 (Float.rem (to_float32 arg1) (to_float32 arg2))))
      | _ ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "fmod"; reason = "requires 2 arguments"}))
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
  | "tan" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (tan (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "tan (float64)"; reason = "requires 1 argument"}))
  | "asin" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (asin (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "asin (float64)"; reason = "requires 1 argument"}))
  | "acos" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (acos (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "acos (float64)"; reason = "requires 1 argument"}))
  | "atan" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (atan (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "atan (float64)"; reason = "requires 1 argument"}))
  | "atan2" -> (
      match args with
      | arg1 :: arg2 :: _ ->
          Some (VFloat64 (atan2 (to_float64 arg1) (to_float64 arg2)))
      | _ ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "atan2 (float64)"; reason = "requires 2 arguments"})
      )
  | "sinh" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (sinh (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "sinh (float64)"; reason = "requires 1 argument"}))
  | "cosh" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (cosh (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "cosh (float64)"; reason = "requires 1 argument"}))
  | "tanh" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (tanh (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "tanh (float64)"; reason = "requires 1 argument"}))
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
  | "floor" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (floor (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "floor (float64)"; reason = "requires 1 argument"}))
  | "ceil" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (ceil (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "ceil (float64)"; reason = "requires 1 argument"}))
  | "pow" -> (
      match args with
      | arg1 :: arg2 :: _ ->
          Some (VFloat64 (Float.pow (to_float64 arg1) (to_float64 arg2)))
      | _ ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "pow (float64)"; reason = "requires 2 arguments"}))
  | "min" -> (
      match args with
      | arg1 :: arg2 :: _ ->
          Some (VFloat64 (min (to_float64 arg1) (to_float64 arg2)))
      | _ ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "min (float64)"; reason = "requires 2 arguments"}))
  | "max" -> (
      match args with
      | arg1 :: arg2 :: _ ->
          Some (VFloat64 (max (to_float64 arg1) (to_float64 arg2)))
      | _ ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "max (float64)"; reason = "requires 2 arguments"}))
  | "of_int" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (Float.of_int (to_int arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "of_int (float64)"; reason = "requires 1 argument"})
      )
  (* The Float64 conversions that every implementation agrees on. All three are
     declared in Sarek_float64/Float64.ml and type-check in the DSL, so the
     interpreter - the cross-backend ORACLE - must evaluate them or it cannot
     run kernels the frontend accepts. int32 is the DSL's integer type (thread
     ids, loop counters), so [of_int32] is the one user code reaches for first.

     [of_int32] and [of_float32] are exact (widening). [to_int32] agrees with
     the device template [(int)(x)], with the registry's [ocaml] field and with
     Sarek_float64_native.ml, all four truncating toward zero - FOR VALUES
     REPRESENTABLE IN INT32. Outside that range and for NaN, OCaml's
     [Int32.of_float] and GLSL's [int(double)] are each unspecified, and they
     are unspecified independently: this is agreement on the defined domain, not
     an oracle guarantee everywhere.

     [to_int] and [to_float32] are deliberately ABSENT. Their device templates
     round/truncate while their [ocaml] field - which Sarek_float64_native.ml
     mirrors and executes - is [Stdlib.int_of_float] (63-bit) and the IDENTITY
     respectively. Three implementations already disagree; implementing them
     here would make that disagreement reachable on one more backend rather
     than settle it. Tracked separately. *)
  | "of_int32" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (Int32.to_float (to_int32 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {
                 operation = "of_int32 (float64)";
                 reason = "requires 1 argument";
               }))
  | "to_int32" -> (
      match args with
      | arg :: _ -> Some (VInt32 (Int32.of_float (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {
                 operation = "to_int32 (float64)";
                 reason = "requires 1 argument";
               }))
  | "of_float32" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (to_float32 arg))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {
                 operation = "of_float32 (float64)";
                 reason = "requires 1 argument";
               }))
  | "expm1" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (Float.expm1 (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "expm1 (float64)"; reason = "requires 1 argument"}))
  | "log10" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (Float.log10 (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "log10 (float64)"; reason = "requires 1 argument"}))
  | "log1p" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (Float.log1p (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "log1p (float64)"; reason = "requires 1 argument"}))
  | "rsqrt" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (1.0 /. Float.sqrt (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "rsqrt (float64)"; reason = "requires 1 argument"}))
  | "abs_float" -> (
      match args with
      | arg :: _ -> Some (VFloat64 (Float.abs (to_float64 arg)))
      | [] ->
          Interp_error.raise_error
            (Unsupported_operation
               {
                 operation = "abs_float (float64)";
                 reason = "requires 1 argument";
               }))
  | "hypot" -> (
      match args with
      | arg1 :: arg2 :: _ ->
          Some (VFloat64 (Float.hypot (to_float64 arg1) (to_float64 arg2)))
      | _ ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "hypot (float64)"; reason = "requires 2 arguments"})
      )
  | "copysign" -> (
      match args with
      | arg1 :: arg2 :: _ ->
          Some (VFloat64 (Float.copy_sign (to_float64 arg1) (to_float64 arg2)))
      | _ ->
          Interp_error.raise_error
            (Unsupported_operation
               {
                 operation = "copysign (float64)";
                 reason = "requires 2 arguments";
               }))
  | "fmod" -> (
      (* C fmod = OCaml Float.rem (sign of dividend, magnitude < |divisor|).
         Matches Sarek_ir_interp_value's float64 Mod arm. *)
      match args with
      | arg1 :: arg2 :: _ ->
          Some (VFloat64 (Float.rem (to_float64 arg1) (to_float64 arg2)))
      | _ ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "fmod (float64)"; reason = "requires 2 arguments"}))
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
