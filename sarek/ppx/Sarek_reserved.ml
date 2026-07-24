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

(** The prefix the Sarek code generator reserves for its own emitted device-code
    identifiers.

    Generated helpers and parameter aliases all start with this prefix in the
    emitted C/CUDA/OpenCL/GLSL/WGSL — e.g. the per-array length parameters
    [sarek_<arr>_length], the integer-remainder helper [sarek_smod] /
    [sarek_smod_N] (PR #255), and the sign-copy helper [sarek_copysign] (PR
    #256). Reserving the whole prefix at elaboration time closes the
    user/generated name-collision class at its root, rather than defending each
    generated name individually.

    The prefix is matched {b case-sensitively}: GLSL and C are case-sensitive
    languages, and the generator only ever emits the lowercase [sarek_] form.
    The OCaml module path [Sarek_.*] (capital [S]) is a compile-time name that
    is never emitted into device code, so it is deliberately {e not} matched. *)
let generated_prefix = "sarek_"

(** [has_reserved_prefix name] is [true] iff [name] begins with
    {!generated_prefix} ([sarek_]), matched case-sensitively.

    This is the single predicate behind the reserved-prefix policy enforced on
    user-written binders (kernel params, local [let]/[let mut] bindings,
    [[@sarek.module]] helper names, and [[@@sarek.type]] type / field /
    constructor names). It only inspects the leading characters, so boundary
    cases such as [sarekX_foo], [_sarek_foo], and [mysarek_] are accepted. *)
let has_reserved_prefix (name : string) : bool =
  let plen = String.length generated_prefix in
  String.length name >= plen && String.sub name 0 plen = generated_prefix
