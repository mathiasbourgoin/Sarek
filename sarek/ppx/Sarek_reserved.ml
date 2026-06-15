(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - Reserved Keywords Validation
 *
 * Checks identifiers against C/CUDA/OpenCL reserved keywords to prevent
 * code generation errors.
 ******************************************************************************)

(** C reserved keywords *)
let c_keywords =
  [
    "auto";
    "break";
    "case";
    "char";
    "const";
    "continue";
    "default";
    "do";
    "double";
    "else";
    "enum";
    "extern";
    "float";
    "for";
    "goto";
    "if";
    "inline";
    "int";
    "long";
    "register";
    "restrict";
    "return";
    "short";
    "signed";
    "sizeof";
    "static";
    "struct";
    "switch";
    "typedef";
    "union";
    "unsigned";
    "void";
    "volatile";
    "while";
    "_Bool";
    "_Complex";
    "_Imaginary";
  ]

(** OpenCL additional reserved keywords *)
let opencl_keywords =
  [
    (* OpenCL C keywords *)
    "__kernel";
    "kernel";
    "__global";
    "global";
    "__local";
    "local";
    "__constant";
    "constant";
    "__private";
    "private";
    "__read_only";
    "read_only";
    "__write_only";
    "write_only";
    "__read_write";
    "read_write";
    (* OpenCL vector types *)
    "char2";
    "char3";
    "char4";
    "char8";
    "char16";
    "uchar";
    "uchar2";
    "uchar3";
    "uchar4";
    "uchar8";
    "uchar16";
    "short2";
    "short3";
    "short4";
    "short8";
    "short16";
    "ushort";
    "ushort2";
    "ushort3";
    "ushort4";
    "ushort8";
    "ushort16";
    "int2";
    "int3";
    "int4";
    "int8";
    "int16";
    "uint";
    "uint2";
    "uint3";
    "uint4";
    "uint8";
    "uint16";
    "long2";
    "long3";
    "long4";
    "long8";
    "long16";
    "ulong";
    "ulong2";
    "ulong3";
    "ulong4";
    "ulong8";
    "ulong16";
    "float2";
    "float3";
    "float4";
    "float8";
    "float16";
    "double2";
    "double3";
    "double4";
    "double8";
    "double16";
    "half";
    "half2";
    "half3";
    "half4";
    "half8";
    "half16";
    (* OpenCL image types *)
    "image2d_t";
    "image3d_t";
    "sampler_t";
    "event_t";
    (* OpenCL built-in functions that shouldn't be shadowed *)
    "barrier";
    "mem_fence";
    "get_global_id";
    "get_local_id";
    "get_group_id";
    "get_global_size";
    "get_local_size";
    "get_num_groups";
    "get_work_dim";
  ]

(** CUDA additional reserved keywords *)
let cuda_keywords =
  [
    "__device__";
    "__global__";
    "__host__";
    "__shared__";
    "__constant__";
    "__managed__";
    "__restrict__";
    "__noinline__";
    "__forceinline__";
    (* CUDA vector types *)
    "dim3";
    "int1";
    "int2";
    "int3";
    "int4";
    "uint1";
    "uint2";
    "uint3";
    "uint4";
    "float1";
    "float2";
    "float3";
    "float4";
    "double1";
    "double2";
    "double3";
    "double4";
    (* CUDA built-in variables *)
    "threadIdx";
    "blockIdx";
    "blockDim";
    "gridDim";
    "warpSize";
    (* CUDA synchronization *)
    "__syncthreads";
    "__threadfence";
    "__threadfence_block";
  ]

(** All reserved keywords combined *)
let all_reserved =
  let tbl = Hashtbl.create 256 in
  List.iter (fun kw -> Hashtbl.replace tbl kw ()) c_keywords ;
  List.iter (fun kw -> Hashtbl.replace tbl kw ()) opencl_keywords ;
  List.iter (fun kw -> Hashtbl.replace tbl kw ()) cuda_keywords ;
  tbl

(** Check if an identifier is a reserved keyword *)
let is_reserved (name : string) : bool = Hashtbl.mem all_reserved name
