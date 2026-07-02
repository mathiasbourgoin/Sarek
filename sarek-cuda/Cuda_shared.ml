(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Spoc_framework

(** Shared kargs extension for all CUDA backends (PTX and C). Defined once so
    wrap/unwrap interoperates across both plugins. *)
type Framework_sig.kargs += Cuda_kargs of Cuda_plugin_base.Cuda.Kernel.args

(** CUDA-specific intrinsic implementation *)
type cuda_intrinsic = {
  intr_name : string;
  intr_codegen : string;
  intr_convergence : Framework_sig.convergence;
}

(** Shared intrinsic registry — both CUDA/PTX and CUDA/C expose the same set *)
module Cuda_intrinsics : Framework_sig.INTRINSIC_REGISTRY = struct
  type intrinsic_impl = cuda_intrinsic

  let table : (string, intrinsic_impl) Hashtbl.t = Hashtbl.create 64

  let register name impl = Hashtbl.replace table name impl

  let find name = Hashtbl.find_opt table name

  let list_all () =
    Hashtbl.fold (fun name _ acc -> name :: acc) table [] |> List.sort compare

  let () =
    register
      "thread_id_x"
      {
        intr_name = "thread_id_x";
        intr_codegen = "threadIdx.x";
        intr_convergence = Divergent;
      } ;
    register
      "thread_id_y"
      {
        intr_name = "thread_id_y";
        intr_codegen = "threadIdx.y";
        intr_convergence = Divergent;
      } ;
    register
      "thread_id_z"
      {
        intr_name = "thread_id_z";
        intr_codegen = "threadIdx.z";
        intr_convergence = Divergent;
      } ;
    register
      "block_id_x"
      {
        intr_name = "block_id_x";
        intr_codegen = "blockIdx.x";
        intr_convergence = Uniform;
      } ;
    register
      "block_id_y"
      {
        intr_name = "block_id_y";
        intr_codegen = "blockIdx.y";
        intr_convergence = Uniform;
      } ;
    register
      "block_id_z"
      {
        intr_name = "block_id_z";
        intr_codegen = "blockIdx.z";
        intr_convergence = Uniform;
      } ;
    register
      "block_dim_x"
      {
        intr_name = "block_dim_x";
        intr_codegen = "blockDim.x";
        intr_convergence = Uniform;
      } ;
    register
      "block_dim_y"
      {
        intr_name = "block_dim_y";
        intr_codegen = "blockDim.y";
        intr_convergence = Uniform;
      } ;
    register
      "block_dim_z"
      {
        intr_name = "block_dim_z";
        intr_codegen = "blockDim.z";
        intr_convergence = Uniform;
      } ;
    register
      "global_thread_id"
      {
        intr_name = "global_thread_id";
        intr_codegen = "(threadIdx.x + blockIdx.x * blockDim.x)";
        intr_convergence = Divergent;
      } ;
    register
      "block_barrier"
      {
        intr_name = "block_barrier";
        intr_codegen = "__syncthreads()";
        intr_convergence = Sync;
      } ;
    register
      "warp_barrier"
      {
        intr_name = "warp_barrier";
        intr_codegen = "__syncwarp()";
        intr_convergence = Sync;
      } ;
    register
      "memory_fence"
      {
        intr_name = "memory_fence";
        intr_codegen = "__threadfence()";
        intr_convergence = Uniform;
      }
end

let is_disabled () =
  Sys.getenv_opt "SPOC_DISABLE_GPU" = Some "1"
  || Sys.getenv_opt "SPOC_DISABLE_CUDA" = Some "1"

let bind_args wrapped_kargs kargs (args : Framework_sig.run_source_arg list) =
  List.iteri
    (fun i arg ->
      match arg with
      | Framework_sig.RSA_Buffer {binder; _} -> binder wrapped_kargs i
      | Framework_sig.RSA_Vector_Length n | Framework_sig.RSA_Int32 n ->
          (* CUDA kernels expect the vector length as an ordinary (ptr, len)
             scalar argument, so RSA_Vector_Length is bound the same way as a
             real RSA_Int32 here. *)
          Cuda_plugin_base.Cuda.Kernel.set_arg_int32 kargs i n
      | Framework_sig.RSA_Int64 n ->
          Cuda_plugin_base.Cuda.Kernel.set_arg_int64 kargs i n
      | Framework_sig.RSA_Float32 f ->
          Cuda_plugin_base.Cuda.Kernel.set_arg_float32 kargs i f
      | Framework_sig.RSA_Float64 f ->
          Cuda_plugin_base.Cuda.Kernel.set_arg_float64 kargs i f)
    args
