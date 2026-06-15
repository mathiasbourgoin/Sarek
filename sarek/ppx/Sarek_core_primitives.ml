(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - Core Primitives
 *
 * This module defines semantic properties for GPU primitives that are known
 * at compile time by the PPX. This separates semantic analysis (variance,
 * convergence, purity) from device implementations (resolved at JIT time).
 ******************************************************************************)

open Sarek_types

(** Variance levels in the GPU execution model. Forms a lattice: Uniform ≤
    BlockVarying ≤ WarpVarying ≤ ThreadVarying *)
type variance =
  | Uniform  (** Same value for all threads in grid *)
  | BlockVarying  (** Uniform within block, varies between blocks *)
  | WarpVarying  (** Uniform within warp, varies between warps *)
  | ThreadVarying  (** Varies per thread *)

(** Convergence requirements *)
type convergence =
  | NoEffect  (** Does not affect convergence *)
  | ConvergencePoint  (** All workgroup threads must reach together *)
  | WarpConvergence  (** All warp threads must reach together *)

(** Purity classification *)
type purity =
  | Pure  (** No side effects *)
  | Impure  (** Has side effects *)
  | Atomic  (** Atomic memory operation *)

(** A core primitive definition *)
type primitive = {
  name : string;
  typ : typ;
  variance : variance;
  convergence : convergence;
  purity : purity;
  category : string;
}

(* All core primitives *)
let primitives =
  [
    (* === Thread Indices (ThreadVarying) === *)
    {
      name = "thread_idx_x";
      typ = t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Pure;
      category = "thread_id";
    };
    {
      name = "thread_idx_y";
      typ = t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Pure;
      category = "thread_id";
    };
    {
      name = "thread_idx_z";
      typ = t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Pure;
      category = "thread_id";
    };
    {
      name = "global_thread_id";
      typ = t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Pure;
      category = "thread_id";
    };
    (* === Global Indices (ThreadVarying) - for Simple1D/2D/3D optimization === *)
    {
      name = "global_idx_x";
      typ = t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Pure;
      category = "thread_id";
    };
    {
      name = "global_idx_y";
      typ = t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Pure;
      category = "thread_id";
    };
    {
      name = "global_idx_z";
      typ = t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Pure;
      category = "thread_id";
    };
    (* === Block Indices (BlockVarying) === *)
    {
      name = "block_idx_x";
      typ = t_int32;
      variance = BlockVarying;
      convergence = NoEffect;
      purity = Pure;
      category = "block_id";
    };
    {
      name = "block_idx_y";
      typ = t_int32;
      variance = BlockVarying;
      convergence = NoEffect;
      purity = Pure;
      category = "block_id";
    };
    {
      name = "block_idx_z";
      typ = t_int32;
      variance = BlockVarying;
      convergence = NoEffect;
      purity = Pure;
      category = "block_id";
    };
    (* === Dimensions (Uniform) === *)
    {
      name = "block_dim_x";
      typ = t_int32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "dimension";
    };
    {
      name = "block_dim_y";
      typ = t_int32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "dimension";
    };
    {
      name = "block_dim_z";
      typ = t_int32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "dimension";
    };
    {
      name = "grid_dim_x";
      typ = t_int32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "dimension";
    };
    {
      name = "grid_dim_y";
      typ = t_int32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "dimension";
    };
    {
      name = "grid_dim_z";
      typ = t_int32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "dimension";
    };
    (* === Synchronization === *)
    {
      name = "block_barrier";
      typ = t_fun [t_unit] t_unit;
      variance = Uniform;
      convergence = ConvergencePoint;
      purity = Impure;
      category = "sync";
    };
    (* === Atomic Operations (Atomic purity, thread-varying) ===
       All atomics return the OLD value at the memory location.
       Variance is ThreadVarying because return value depends on execution order. *)
    (* Local/shared memory atomics - use array[Local] + index *)
    {
      name = "atomic_add_int32";
      typ = t_fun [t_arr t_int32 Local; t_int32; t_int32] t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Atomic;
      category = "atomic";
    };
    {
      name = "atomic_sub_int32";
      typ = t_fun [t_arr t_int32 Local; t_int32; t_int32] t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Atomic;
      category = "atomic";
    };
    {
      name = "atomic_exch_int32";
      typ = t_fun [t_arr t_int32 Local; t_int32; t_int32] t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Atomic;
      category = "atomic";
    };
    {
      name = "atomic_min_int32";
      typ = t_fun [t_arr t_int32 Local; t_int32; t_int32] t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Atomic;
      category = "atomic";
    };
    {
      name = "atomic_max_int32";
      typ = t_fun [t_arr t_int32 Local; t_int32; t_int32] t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Atomic;
      category = "atomic";
    };
    {
      name = "atomic_cas_int32";
      (* array, index, compare, value -> old *)
      typ = t_fun [t_arr t_int32 Local; t_int32; t_int32; t_int32] t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Atomic;
      category = "atomic";
    };
    {
      name = "atomic_and_int32";
      typ = t_fun [t_arr t_int32 Local; t_int32; t_int32] t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Atomic;
      category = "atomic";
    };
    {
      name = "atomic_or_int32";
      typ = t_fun [t_arr t_int32 Local; t_int32; t_int32] t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Atomic;
      category = "atomic";
    };
    {
      name = "atomic_xor_int32";
      typ = t_fun [t_arr t_int32 Local; t_int32; t_int32] t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Atomic;
      category = "atomic";
    };
    {
      name = "atomic_inc_int32";
      (* array, index -> old (increments by 1) *)
      typ = t_fun [t_arr t_int32 Local; t_int32] t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Atomic;
      category = "atomic";
    };
    {
      name = "atomic_dec_int32";
      (* array, index -> old (decrements by 1) *)
      typ = t_fun [t_arr t_int32 Local; t_int32] t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Atomic;
      category = "atomic";
    };
    (* 64-bit atomics *)
    {
      name = "atomic_add_int64";
      typ = t_fun [t_arr t_int64 Local; t_int32; t_int64] t_int64;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Atomic;
      category = "atomic";
    };
    (* Float atomics *)
    {
      name = "atomic_add_float32";
      typ = t_fun [t_arr t_float32 Local; t_int32; t_float32] t_float32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Atomic;
      category = "atomic";
    };
    {
      name = "atomic_add_float64";
      typ = t_fun [t_arr t_float64 Local; t_int32; t_float64] t_float64;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Atomic;
      category = "atomic";
    };
    (* Global memory atomics (for vectors) *)
    {
      name = "atomic_add_global_int32";
      typ = t_fun [t_vec t_int32; t_int32; t_int32] t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Atomic;
      category = "atomic";
    };
    {
      name = "atomic_inc_global_int32";
      typ = t_fun [t_vec t_int32; t_int32] t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Atomic;
      category = "atomic";
    };
    (* === Float32 Math (Pure, Uniform inherent variance) === *)
    {
      name = "sin";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "cos";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "tan";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "sqrt";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "rsqrt";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "exp";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "log";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "log2";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "log10";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "fabs";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "floor";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "ceil";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "pow";
      typ = t_fun [t_float32; t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "fma";
      typ = t_fun [t_float32; t_float32; t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "asin";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "acos";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "atan";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "atan2";
      typ = t_fun [t_float32; t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    (* === Float64 Math (Pure) === *)
    {
      name = "sin_f64";
      typ = t_fun [t_float64] t_float64;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f64";
    };
    {
      name = "cos_f64";
      typ = t_fun [t_float64] t_float64;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f64";
    };
    {
      name = "tan_f64";
      typ = t_fun [t_float64] t_float64;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f64";
    };
    {
      name = "sqrt_f64";
      typ = t_fun [t_float64] t_float64;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f64";
    };
    {
      name = "exp_f64";
      typ = t_fun [t_float64] t_float64;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f64";
    };
    {
      name = "log_f64";
      typ = t_fun [t_float64] t_float64;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f64";
    };
    {
      name = "log2_f64";
      typ = t_fun [t_float64] t_float64;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f64";
    };
    {
      name = "log10_f64";
      typ = t_fun [t_float64] t_float64;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f64";
    };
    {
      name = "pow_f64";
      typ = t_fun [t_float64; t_float64] t_float64;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f64";
    };
    {
      name = "fabs_f64";
      typ = t_fun [t_float64] t_float64;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f64";
    };
    {
      name = "floor_f64";
      typ = t_fun [t_float64] t_float64;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f64";
    };
    {
      name = "ceil_f64";
      typ = t_fun [t_float64] t_float64;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f64";
    };
    {
      name = "fma_f64";
      typ = t_fun [t_float64; t_float64; t_float64] t_float64;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f64";
    };
    (* === Integer Primitives (Pure) === *)
    {
      name = "abs_int32";
      typ = t_fun [t_int32] t_int32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "int";
    };
    {
      name = "min_int32";
      typ = t_fun [t_int32; t_int32] t_int32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "int";
    };
    {
      name = "max_int32";
      typ = t_fun [t_int32; t_int32] t_int32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "int";
    };
    {
      name = "clz_int32";
      typ = t_fun [t_int32] t_int32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "int";
    };
    {
      name = "popcount_int32";
      typ = t_fun [t_int32] t_int32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "int";
    };
    (* === Memory Fences (Impure, no convergence) === *)
    {
      name = "memory_fence_block";
      typ = t_fun [t_unit] t_unit;
      variance = Uniform;
      convergence = NoEffect;
      purity = Impure;
      category = "fence";
    };
    {
      name = "memory_fence_device";
      typ = t_fun [t_unit] t_unit;
      variance = Uniform;
      convergence = NoEffect;
      purity = Impure;
      category = "fence";
    };
    (* === Warp-Level Primitives === *)
    (* Warp shuffle: exchange values between lanes *)
    {
      name = "warp_shuffle";
      typ = t_fun [t_int32; t_int32] t_int32;
      variance = ThreadVarying;
      (* Output varies per thread *)
      convergence = WarpConvergence;
      (* All warp threads must participate *)
      purity = Pure;
      category = "warp";
    };
    {
      name = "warp_shuffle_down";
      typ = t_fun [t_int32; t_int32] t_int32;
      variance = ThreadVarying;
      convergence = WarpConvergence;
      purity = Pure;
      category = "warp";
    };
    {
      name = "warp_shuffle_up";
      typ = t_fun [t_int32; t_int32] t_int32;
      variance = ThreadVarying;
      convergence = WarpConvergence;
      purity = Pure;
      category = "warp";
    };
    {
      name = "warp_shuffle_xor";
      typ = t_fun [t_int32; t_int32] t_int32;
      variance = ThreadVarying;
      convergence = WarpConvergence;
      purity = Pure;
      category = "warp";
    };
    (* Warp vote: collective predicates *)
    {
      name = "warp_vote_all";
      typ = t_fun [t_bool] t_bool;
      variance = WarpVarying;
      (* Same result for all threads in warp *)
      convergence = WarpConvergence;
      purity = Pure;
      category = "warp";
    };
    {
      name = "warp_vote_any";
      typ = t_fun [t_bool] t_bool;
      variance = WarpVarying;
      convergence = WarpConvergence;
      purity = Pure;
      category = "warp";
    };
    {
      name = "warp_ballot";
      typ = t_fun [t_bool] t_int32;
      variance = WarpVarying;
      (* Bitmask is same for all threads in warp *)
      convergence = WarpConvergence;
      purity = Pure;
      category = "warp";
    };
    (* Warp introspection *)
    {
      name = "warp_active_mask";
      typ = t_fun [t_unit] t_int32;
      variance = WarpVarying;
      convergence = NoEffect;
      (* Doesn't require convergence *)
      purity = Pure;
      category = "warp";
    };
    {
      name = "warp_size";
      typ = t_int32;
      variance = Uniform;
      (* Same for all threads *)
      convergence = NoEffect;
      purity = Pure;
      category = "warp";
    };
  ]

(* Build lookup table for O(1) access *)
let primitive_table : (string, primitive) Hashtbl.t =
  let tbl = Hashtbl.create (List.length primitives) in
  List.iter (fun p -> Hashtbl.add tbl p.name p) primitives ;
  tbl

let find name = Hashtbl.find_opt primitive_table name

let find_exn name = Hashtbl.find primitive_table name

let is_core_primitive name = Hashtbl.mem primitive_table name

let is_thread_varying name =
  match find name with Some p -> p.variance = ThreadVarying | None -> false

(** Check if a primitive has warp-level or finer granularity variance. Returns
    true for WarpVarying and ThreadVarying primitives. *)
let is_warp_varying name =
  match find name with
  | Some p -> p.variance = WarpVarying || p.variance = ThreadVarying
  | None -> false

let is_convergence_point name =
  match find name with
  | Some p -> p.convergence = ConvergencePoint
  | None -> false

(** Check if a primitive requires warp-level convergence *)
let is_warp_convergence_point name =
  match find name with
  | Some p -> p.convergence = WarpConvergence
  | None -> false

(** Check if a primitive requires any kind of convergence (block or warp) *)
let requires_convergence name =
  match find name with
  | Some p ->
      p.convergence = ConvergencePoint || p.convergence = WarpConvergence
  | None -> false

let is_pure name =
  match find name with Some p -> p.purity = Pure | None -> false

(** Check if a primitive is an atomic memory operation *)
let is_atomic name =
  match find name with Some p -> p.purity = Atomic | None -> false

let variance_of name = Option.map (fun p -> p.variance) (find name)

(* Variance lattice join (least upper bound)
   Lattice: Uniform ≤ BlockVarying ≤ WarpVarying ≤ ThreadVarying *)
let join_variance v1 v2 =
  match (v1, v2) with
  | Uniform, v | v, Uniform -> v
  | BlockVarying, BlockVarying -> BlockVarying
  | BlockVarying, WarpVarying | WarpVarying, BlockVarying -> WarpVarying
  | WarpVarying, WarpVarying -> WarpVarying
  | ThreadVarying, _ | _, ThreadVarying -> ThreadVarying

(* Variance ordering: v1 ≤ v2 means v1 is less varying than v2 *)
let variance_leq v1 v2 =
  match (v1, v2) with
  | Uniform, _ -> true
  | BlockVarying, (BlockVarying | WarpVarying | ThreadVarying) -> true
  | WarpVarying, (WarpVarying | ThreadVarying) -> true
  | ThreadVarying, ThreadVarying -> true
  | _ -> false

let primitives_in_category cat =
  List.filter (fun p -> p.category = cat) primitives

(* Pretty printing *)
let pp_variance fmt = function
  | Uniform -> Format.fprintf fmt "Uniform"
  | BlockVarying -> Format.fprintf fmt "BlockVarying"
  | WarpVarying -> Format.fprintf fmt "WarpVarying"
  | ThreadVarying -> Format.fprintf fmt "ThreadVarying"

let pp_convergence fmt = function
  | NoEffect -> Format.fprintf fmt "NoEffect"
  | ConvergencePoint -> Format.fprintf fmt "ConvergencePoint"
  | WarpConvergence -> Format.fprintf fmt "WarpConvergence"

let pp_purity fmt = function
  | Pure -> Format.fprintf fmt "Pure"
  | Impure -> Format.fprintf fmt "Impure"
  | Atomic -> Format.fprintf fmt "Atomic"

let pp_primitive fmt p =
  Format.fprintf
    fmt
    "@[<v 2>%s:@ type: %a@ variance: %a@ convergence: %a@ purity: %a@ \
     category: %s@]"
    p.name
    Sarek_types.pp_typ
    p.typ
    pp_variance
    p.variance
    pp_convergence
    p.convergence
    pp_purity
    p.purity
    p.category
