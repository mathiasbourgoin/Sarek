(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 mathias <mathias@ladon.local> *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Metal API - Ctypes Type Definitions
 *
 * Pure OCaml bindings to Metal API using ctypes.
 * Metal uses Objective-C but we bind to the C-compatible parts.
 * No C stubs required - all FFI via ctypes-foreign.
 ******************************************************************************)

open Ctypes

(** {1 Basic Types} *)

(** Metal uses NS types from Foundation/CoreFoundation *)
type ns_uint = Unsigned.uint64

let ns_uint : ns_uint typ = uint64_t

type ns_uinteger = Unsigned.size_t

let ns_uinteger : ns_uinteger typ = size_t

(** {1 Opaque Handle Types} *)

(** Device - MTLDevice protocol (id<MTLDevice>) *)
type mtl_device = unit ptr

let mtl_device : mtl_device typ = ptr void

(** Command Queue - MTLCommandQueue protocol *)
type mtl_command_queue = unit ptr

let mtl_command_queue : mtl_command_queue typ = ptr void

(** Command Buffer - MTLCommandBuffer protocol *)
type mtl_command_buffer = unit ptr

let mtl_command_buffer : mtl_command_buffer typ = ptr void

(** Compute Command Encoder - MTLComputeCommandEncoder protocol *)
type mtl_compute_command_encoder = unit ptr

let mtl_compute_command_encoder : mtl_compute_command_encoder typ = ptr void

(** Buffer - MTLBuffer protocol *)
type mtl_buffer = unit ptr

let mtl_buffer : mtl_buffer typ = ptr void

(** Library - MTLLibrary protocol *)
type mtl_library = unit ptr

let mtl_library : mtl_library typ = ptr void

(** Function - MTLFunction protocol *)
type mtl_function = unit ptr

let mtl_function : mtl_function typ = ptr void

(** Compute Pipeline State - MTLComputePipelineState protocol *)
type mtl_compute_pipeline_state = unit ptr

let mtl_compute_pipeline_state : mtl_compute_pipeline_state typ = ptr void

(** NSError - for error reporting *)
type ns_error = unit ptr

let ns_error : ns_error typ = ptr void

(** NSString - for string passing *)
type ns_string = unit ptr

let ns_string : ns_string typ = ptr void

(** {1 Resource Options} *)

(** MTLResourceOptions bitfield *)
type mtl_resource_options = Unsigned.uint64

let mtl_resource_options : mtl_resource_options typ = uint64_t

(** Storage modes *)
let mtl_storage_mode_shared =
  Unsigned.UInt64.of_int 0 (* MTLStorageModeShared *)

let mtl_storage_mode_managed = Unsigned.UInt64.of_int 1
(* MTLStorageModeManaged *)

let mtl_storage_mode_private = Unsigned.UInt64.of_int 2
(* MTLStorageModePrivate *)

let mtl_storage_mode_memoryless = Unsigned.UInt64.of_int 3
(* MTLStorageModeMemoryless *)

(** Resource options (storage mode is bits 0-3) *)
let mtl_resource_storage_mode_shared = Unsigned.UInt64.of_int 0

let mtl_resource_storage_mode_managed =
  Unsigned.UInt64.shift_left (Unsigned.UInt64.of_int 1) 4

let mtl_resource_storage_mode_private =
  Unsigned.UInt64.shift_left (Unsigned.UInt64.of_int 2) 4

let mtl_resource_cpu_cache_mode_default_cache = Unsigned.UInt64.of_int 0

let mtl_resource_cpu_cache_mode_write_combined =
  Unsigned.UInt64.shift_left (Unsigned.UInt64.of_int 1) 8

(** {1 Size and Origin Types} *)

(** MTLSize - 3D dimensions for compute grid *)
type mtl_size

let mtl_size : mtl_size structure typ = structure "MTLSize"

let mtl_size_width = field mtl_size "width" ns_uinteger

let mtl_size_height = field mtl_size "height" ns_uinteger

let mtl_size_depth = field mtl_size "depth" ns_uinteger

let () = seal mtl_size

(** MTLOrigin - 3D origin *)
type mtl_origin

let mtl_origin : mtl_origin structure typ = structure "MTLOrigin"

let mtl_origin_x = field mtl_origin "x" ns_uinteger

let mtl_origin_y = field mtl_origin "y" ns_uinteger

let mtl_origin_z = field mtl_origin "z" ns_uinteger

let () = seal mtl_origin

(** {1 Helper Functions} *)

(** Create MTLSize from dimensions *)
let make_mtl_size ~width ~height ~depth =
  let s = make mtl_size in
  setf s mtl_size_width (Unsigned.Size_t.of_int width) ;
  setf s mtl_size_height (Unsigned.Size_t.of_int height) ;
  setf s mtl_size_depth (Unsigned.Size_t.of_int depth) ;
  s

(** Create MTLOrigin at (0,0,0) *)
let make_mtl_origin_zero () =
  let o = make mtl_origin in
  setf o mtl_origin_x Unsigned.Size_t.zero ;
  setf o mtl_origin_y Unsigned.Size_t.zero ;
  setf o mtl_origin_z Unsigned.Size_t.zero ;
  o

(** {1 Error Codes} *)

(** Metal doesn't have error codes like OpenCL/CUDA - it uses NSError *)
type mtl_error = NS_ERROR of string | MTL_SUCCESS

let string_of_mtl_error = function
  | MTL_SUCCESS -> "MTL_SUCCESS"
  | NS_ERROR s -> Printf.sprintf "NS_ERROR: %s" s
