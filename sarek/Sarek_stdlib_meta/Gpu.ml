(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek GPU Stdlib Meta - pure metadata registration (FFI-free)
 *
 * Registers GPU thread/block index and synchronization intrinsic signatures
 * into Sarek_ppx_registry without any Ctypes or Spoc_core dependency.
 *
 * The global-memory atomic operations (atomic_add_global_int32,
 * atomic_inc_global_int32) use stub OCaml implementations here; the FFI
 * execution path in sarek_stdlib.Gpu provides the real Spoc_core.Vector
 * implementations.
 ******************************************************************************)

let dev cuda opencl (framework : string) =
  match framework with "CUDA" -> cuda | _ -> opencl

(******************************************************************************
 * Thread indices within the block
 ******************************************************************************)

let%sarek_intrinsic (thread_idx_x : int32) =
  {device = dev "threadIdx.x" "get_local_id(0)"; ocaml = 0l}

let%sarek_intrinsic (thread_idx_y : int32) =
  {device = dev "threadIdx.y" "get_local_id(1)"; ocaml = 0l}

let%sarek_intrinsic (thread_idx_z : int32) =
  {device = dev "threadIdx.z" "get_local_id(2)"; ocaml = 0l}

(******************************************************************************
 * Block indices within the grid
 ******************************************************************************)

let%sarek_intrinsic (block_idx_x : int32) =
  {device = dev "blockIdx.x" "get_group_id(0)"; ocaml = 0l}

let%sarek_intrinsic (block_idx_y : int32) =
  {device = dev "blockIdx.y" "get_group_id(1)"; ocaml = 0l}

let%sarek_intrinsic (block_idx_z : int32) =
  {device = dev "blockIdx.z" "get_group_id(2)"; ocaml = 0l}

(******************************************************************************
 * Block dimensions
 ******************************************************************************)

let%sarek_intrinsic (block_dim_x : int32) =
  {device = dev "blockDim.x" "get_local_size(0)"; ocaml = 0l}

let%sarek_intrinsic (block_dim_y : int32) =
  {device = dev "blockDim.y" "get_local_size(1)"; ocaml = 0l}

let%sarek_intrinsic (block_dim_z : int32) =
  {device = dev "blockDim.z" "get_local_size(2)"; ocaml = 0l}

(******************************************************************************
 * Grid dimensions
 ******************************************************************************)

let%sarek_intrinsic (grid_dim_x : int32) =
  {device = dev "gridDim.x" "get_num_groups(0)"; ocaml = 0l}

let%sarek_intrinsic (grid_dim_y : int32) =
  {device = dev "gridDim.y" "get_num_groups(1)"; ocaml = 0l}

let%sarek_intrinsic (grid_dim_z : int32) =
  {device = dev "gridDim.z" "get_num_groups(2)"; ocaml = 0l}

(******************************************************************************
 * Global thread ID and global indices
 ******************************************************************************)

let%sarek_intrinsic (global_thread_id : int32) =
  {
    device = dev "(threadIdx.x + blockIdx.x * blockDim.x)" "get_global_id(0)";
    ocaml = 0l;
  }

let%sarek_intrinsic (global_idx_x : int32) =
  {
    device = dev "(threadIdx.x + blockIdx.x * blockDim.x)" "get_global_id(0)";
    ocaml = 0l;
  }

let%sarek_intrinsic (global_idx_y : int32) =
  {
    device = dev "(threadIdx.y + blockIdx.y * blockDim.y)" "get_global_id(1)";
    ocaml = 0l;
  }

let%sarek_intrinsic (global_idx_z : int32) =
  {
    device = dev "(threadIdx.z + blockIdx.z * blockDim.z)" "get_global_id(2)";
    ocaml = 0l;
  }

(******************************************************************************
 * Synchronization
 ******************************************************************************)

let%sarek_intrinsic (block_barrier : unit -> unit) =
  {
    device = dev "__syncthreads();%s" "barrier(CLK_LOCAL_MEM_FENCE);%s";
    ocaml = (fun () -> ());
  }

let%sarek_intrinsic (return_unit : unit -> unit) =
  {device = (fun _ -> "return"); ocaml = (fun () -> ())}

(******************************************************************************
 * Atomic Operations
 *
 * Shared-memory atomics have pure OCaml implementations (arrays).
 * Global-memory atomics stub with failwith — the FFI sarek_stdlib provides
 * the real Spoc_core.Vector implementations for host execution.
 ******************************************************************************)

let%sarek_intrinsic (atomic_add_int32 : int32 array -> int32 -> int32 -> int32)
    =
  {
    device = dev "atomicAdd(%s + %s, %s)" "atomic_add(%s + %s, %s)";
    ocaml =
      (fun arr idx value ->
        let i = Stdlib.Int32.to_int idx in
        let old = arr.(i) in
        arr.(i) <- Stdlib.Int32.add old value ;
        old);
  }

let%sarek_intrinsic (atomic_inc_int32 : int32 array -> int32 -> int32) =
  {
    device = dev "atomicAdd(%s + %s, 1)" "atomic_inc(%s + %s)";
    ocaml =
      (fun arr idx ->
        let i = Stdlib.Int32.to_int idx in
        let old = arr.(i) in
        arr.(i) <- Stdlib.Int32.add old 1l ;
        old);
  }

(* Global-memory atomics: metadata only; host execution requires sarek_stdlib *)
let%sarek_intrinsic
    (atomic_add_global_int32 : int32 vector -> int32 -> int32 -> int32) =
  {
    device = dev "atomicAdd(%s + %s, %s)" "atomic_add(%s + %s, %s)";
    ocaml =
      (fun _vec _idx _value ->
        failwith "atomic_add_global_int32: host execution requires sarek_stdlib");
  }

let%sarek_intrinsic (atomic_inc_global_int32 : int32 vector -> int32 -> int32) =
  {
    device = dev "atomicAdd(%s + %s, 1)" "atomic_inc(%s + %s)";
    ocaml =
      (fun _vec _idx ->
        failwith "atomic_inc_global_int32: host execution requires sarek_stdlib");
  }

(******************************************************************************
 * Type conversions
 ******************************************************************************)

let%sarek_intrinsic (float_of_int : int32 -> float) =
  {
    device = (fun _ -> "(float)");
    ocaml = (fun i -> Stdlib.float_of_int (Stdlib.Int32.to_int i));
  }

let%sarek_intrinsic (float : int32 -> float32) =
  {
    device = (fun _ -> "(float)");
    ocaml = (fun i -> Stdlib.float_of_int (Stdlib.Int32.to_int i));
  }

let%sarek_intrinsic (float64_of_int : int32 -> float) =
  {
    device = (fun _ -> "(double)");
    ocaml = (fun i -> Stdlib.float_of_int (Stdlib.Int32.to_int i));
  }

let%sarek_intrinsic (int_of_float : float -> int32) =
  {
    device = (fun _ -> "(int)");
    ocaml = (fun f -> Stdlib.Int32.of_int (Stdlib.int_of_float f));
  }

let%sarek_intrinsic (int_of_float64 : float -> int32) =
  {
    device = (fun _ -> "(int)");
    ocaml = (fun f -> Stdlib.Int32.of_int (Stdlib.int_of_float f));
  }

(******************************************************************************
 * Integer power
 ******************************************************************************)

let%sarek_intrinsic (spoc_powint : int32 -> int32 -> int32) =
  {
    device = (fun _ -> "spoc_powint");
    ocaml =
      (fun base exp ->
        let rec pow b e acc =
          if e = 0l then acc
          else pow b (Stdlib.Int32.sub e 1l) (Stdlib.Int32.mul acc b)
        in
        pow base exp 1l);
  }
