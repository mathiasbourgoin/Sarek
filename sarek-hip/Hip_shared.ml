(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Spoc_framework

(** Backend-specific kargs witness for HIP. *)
type Framework_sig.kargs += Hip_kargs of Hip_plugin_base.Hip.Kernel.args

(** HIP-specific intrinsic implementation. HIP C++ uses the identical device
    tokens as CUDA C (threadIdx.x, __syncthreads, ...), so the codegen emitted
    by Sarek_ir_cuda is valid verbatim. *)
type hip_intrinsic = {
  intr_name : string;
  intr_codegen : string;
  intr_convergence : Framework_sig.convergence;
}

module Hip_intrinsics : Framework_sig.INTRINSIC_REGISTRY = struct
  type intrinsic_impl = hip_intrinsic

  let table : (string, intrinsic_impl) Hashtbl.t = Hashtbl.create 64

  let register name impl = Hashtbl.replace table name impl

  let find name = Hashtbl.find_opt table name

  let list_all () =
    Hashtbl.fold (fun name _ acc -> name :: acc) table [] |> List.sort compare

  let () =
    let reg intr_name intr_codegen intr_convergence =
      register intr_name {intr_name; intr_codegen; intr_convergence}
    in
    reg "thread_id_x" "threadIdx.x" Divergent ;
    reg "thread_id_y" "threadIdx.y" Divergent ;
    reg "thread_id_z" "threadIdx.z" Divergent ;
    reg "block_id_x" "blockIdx.x" Uniform ;
    reg "block_id_y" "blockIdx.y" Uniform ;
    reg "block_id_z" "blockIdx.z" Uniform ;
    reg "block_dim_x" "blockDim.x" Uniform ;
    reg "block_dim_y" "blockDim.y" Uniform ;
    reg "block_dim_z" "blockDim.z" Uniform ;
    reg "global_thread_id" "(threadIdx.x + blockIdx.x * blockDim.x)" Divergent ;
    reg "block_barrier" "__syncthreads()" Sync ;
    reg "warp_barrier" "__syncwarp()" Sync ;
    reg "memory_fence" "__threadfence()" Uniform
end

let is_disabled () =
  Sys.getenv_opt "SPOC_DISABLE_GPU" = Some "1"
  || Sys.getenv_opt "SPOC_DISABLE_HIP" = Some "1"

let bind_args wrapped_kargs kargs (args : Framework_sig.run_source_arg list) =
  List.iteri
    (fun i arg ->
      match arg with
      | Framework_sig.RSA_Buffer {binder; _} -> binder wrapped_kargs i
      | Framework_sig.RSA_Vector_Length n | Framework_sig.RSA_Int32 n ->
          Hip_plugin_base.Hip.Kernel.set_arg_int32 kargs i n
      | Framework_sig.RSA_Int64 n ->
          Hip_plugin_base.Hip.Kernel.set_arg_int64 kargs i n
      | Framework_sig.RSA_Float32 f ->
          Hip_plugin_base.Hip.Kernel.set_arg_float32 kargs i f
      | Framework_sig.RSA_Float64 f ->
          Hip_plugin_base.Hip.Kernel.set_arg_float64 kargs i f)
    args
