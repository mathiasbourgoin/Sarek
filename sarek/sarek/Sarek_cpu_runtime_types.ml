(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Re-export Float32 module for use in generated kernels *)
module Float32 = Sarek_float32

(** Execution mode for native kernels *)
type exec_mode =
  | Sequential  (** Single-threaded, barriers are no-ops *)
  | Parallel  (** Spawn domains per kernel launch *)
  | Threadpool  (** Use persistent thread pool (fission mode) *)

(** {1 Thread State}

    Thread state is passed to each generated kernel function. The kernel reads
    thread/block/grid indices from this record.

    All indices are int32 to match GPU semantics (Sarek_stdlib.Gpu uses int32).
*)

type thread_state = {
  thread_idx_x : int32;
  thread_idx_y : int32;
  thread_idx_z : int32;
  block_idx_x : int32;
  block_idx_y : int32;
  block_idx_z : int32;
  block_dim_x : int32;
  block_dim_y : int32;
  block_dim_z : int32;
  grid_dim_x : int32;
  grid_dim_y : int32;
  grid_dim_z : int32;
  barrier : unit -> unit;
      (** Barrier function - no-op in sequential, effect in parallel *)
}

(** {1 Global Index Helpers} *)

let global_idx_x st =
  Int32.add (Int32.mul st.block_idx_x st.block_dim_x) st.thread_idx_x

let global_idx_y st =
  Int32.add (Int32.mul st.block_idx_y st.block_dim_y) st.thread_idx_y

let global_idx_z st =
  Int32.add (Int32.mul st.block_idx_z st.block_dim_z) st.thread_idx_z

let global_size_x st = Int32.mul st.grid_dim_x st.block_dim_x

let global_size_y st = Int32.mul st.grid_dim_y st.block_dim_y

let global_size_z st = Int32.mul st.grid_dim_z st.block_dim_z

(** {1 Shared Memory}

    Shared memory is allocated per-block and accessible by all threads in the
    block. Uses regular OCaml arrays to support custom types.

    Implementation: We use separate hashtables for each primitive type to avoid
    boxing, and an existential wrapper for custom types. *)

(** Existential wrapper for custom type arrays with a runtime type witness. *)
type any_array =
  | AnyArray : 'a Sarek_ir_types.Type_id.t * 'a array -> any_array

type shared_mem = {
  int_arrays : (string, int array) Hashtbl.t;
  float_arrays : (string, float array) Hashtbl.t;
  int32_arrays : (string, int32 array) Hashtbl.t;
  int64_arrays : (string, int64 array) Hashtbl.t;
  custom_arrays : (string, any_array) Hashtbl.t;
}

let create_shared () =
  {
    int_arrays = Hashtbl.create 2;
    float_arrays = Hashtbl.create 2;
    int32_arrays = Hashtbl.create 2;
    int64_arrays = Hashtbl.create 2;
    custom_arrays = Hashtbl.create 2;
  }

(** Typed allocators for common array types - completely type-safe *)

let alloc_shared_int (shared : shared_mem) name size (default : int) : int array
    =
  match Hashtbl.find_opt shared.int_arrays name with
  | Some arr -> arr
  | None ->
      let arr = Array.make size default in
      Hashtbl.add shared.int_arrays name arr ;
      arr

let alloc_shared_float (shared : shared_mem) name size (default : float) :
    float array =
  match Hashtbl.find_opt shared.float_arrays name with
  | Some arr -> arr
  | None ->
      let arr = Array.make size default in
      Hashtbl.add shared.float_arrays name arr ;
      arr

let alloc_shared_int32 (shared : shared_mem) name size (default : int32) :
    int32 array =
  match Hashtbl.find_opt shared.int32_arrays name with
  | Some arr -> arr
  | None ->
      let arr = Array.make size default in
      Hashtbl.add shared.int32_arrays name arr ;
      arr

let alloc_shared_int64 (shared : shared_mem) name size (default : int64) :
    int64 array =
  match Hashtbl.find_opt shared.int64_arrays name with
  | Some arr -> arr
  | None ->
      let arr = Array.make size default in
      Hashtbl.add shared.int64_arrays name arr ;
      arr

(** Generic allocator for custom types. The caller must ensure they use
    consistent types for each name. *)
let alloc_shared_with_key (type a) (shared : shared_mem)
    (key : a Sarek_ir_types.Type_id.t) name size (default : a) : a array =
  match Hashtbl.find_opt shared.custom_arrays name with
  | Some (AnyArray (stored_key, arr)) -> (
      match Sarek_ir_types.Type_id.equal key stored_key with
      | Some Sarek_ir_types.Type_id.Refl -> arr
      | None -> invalid_arg ("alloc_shared: type mismatch for " ^ name))
  | None ->
      let arr = Array.make size default in
      Hashtbl.add shared.custom_arrays name (AnyArray (key, arr)) ;
      arr

(** Effect for yielding control at barrier - declared here for sequential use *)
type _ Effect.t += Barrier : unit Effect.t
