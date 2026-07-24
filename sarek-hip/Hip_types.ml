(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * HIP Runtime/Driver API - Ctypes Type Definitions
 *
 * Pure OCaml bindings to the AMD ROCm/HIP API using ctypes.
 * No C stubs required - all FFI via ctypes-foreign.
 *
 * This is the HIP analog of Cuda_types. HIP is source- and largely
 * ABI-shaped like the CUDA driver API, so the structure mirrors the CUDA
 * bindings closely; the enum integers below were extracted from the ROCm
 * 7.2.4 headers (/opt/rocm/include/hip/hip_runtime_api.h) via a throwaway C
 * probe, since hipDeviceAttribute_t / hipError_t are auto-numbered.
 ******************************************************************************)

open Ctypes

(** {1 Basic Types} *)

(** HIP device - integer handle (typedef int hipDevice_t). *)
type hip_device = int

let hip_device : hip_device typ = int

(** Device pointer - hipDeviceptr_t is [void*] (8 bytes on this platform). We
    model it as an opaque [void] pointer, mirroring the C typedef. *)
type hip_deviceptr = unit ptr

let hip_deviceptr : hip_deviceptr typ = ptr void

(** {1 Opaque Handle Types}

    HIP handles are pointer typedefs ([struct ihipXxx_t*]); we model each as a
    pointer to an opaque named structure, exactly as Cuda_types does for the
    CUDA driver handles. *)

type hip_module

let hip_module : hip_module structure typ = structure "ihipModule_t"

let hip_module_ptr : hip_module structure ptr typ = ptr hip_module

type hip_function

let hip_function : hip_function structure typ = structure "ihipModuleSymbol_t"

let hip_function_ptr : hip_function structure ptr typ = ptr hip_function

type hip_stream

let hip_stream : hip_stream structure typ = structure "ihipStream_t"

let hip_stream_ptr : hip_stream structure ptr typ = ptr hip_stream

type hip_event

let hip_event : hip_event structure typ = structure "ihipEvent_t"

let hip_event_ptr : hip_event structure ptr typ = ptr hip_event

(** {1 Result Codes}

    hipError_t. The common subset is enumerated; anything else is preserved via
    [HIP_ERROR_UNKNOWN]. Integers verified against ROCm 7.2.4. *)
type hip_result =
  | HIP_SUCCESS
  | HIP_ERROR_INVALID_VALUE
  | HIP_ERROR_OUT_OF_MEMORY
  | HIP_ERROR_NOT_INITIALIZED
  | HIP_ERROR_DEINITIALIZED
  | HIP_ERROR_NO_DEVICE
  | HIP_ERROR_INVALID_DEVICE
  | HIP_ERROR_INVALID_IMAGE
  | HIP_ERROR_INVALID_CONTEXT
  | HIP_ERROR_INVALID_HANDLE
  | HIP_ERROR_NOT_FOUND
  | HIP_ERROR_NOT_READY
  | HIP_ERROR_NO_BINARY_FOR_GPU
  | HIP_ERROR_SHARED_OBJECT_INIT_FAILED
  | HIP_ERROR_NOT_SUPPORTED
  | HIP_ERROR_LAUNCH_FAILURE
  | HIP_ERROR_UNKNOWN of int

let hip_result_of_int = function
  | 0 -> HIP_SUCCESS
  | 1 -> HIP_ERROR_INVALID_VALUE
  | 2 -> HIP_ERROR_OUT_OF_MEMORY
  | 3 -> HIP_ERROR_NOT_INITIALIZED
  | 4 -> HIP_ERROR_DEINITIALIZED
  | 100 -> HIP_ERROR_NO_DEVICE
  | 101 -> HIP_ERROR_INVALID_DEVICE
  | 200 -> HIP_ERROR_INVALID_IMAGE
  | 201 -> HIP_ERROR_INVALID_CONTEXT
  | 209 -> HIP_ERROR_NO_BINARY_FOR_GPU
  | 303 -> HIP_ERROR_SHARED_OBJECT_INIT_FAILED
  | 400 -> HIP_ERROR_INVALID_HANDLE
  | 500 -> HIP_ERROR_NOT_FOUND
  | 600 -> HIP_ERROR_NOT_READY
  | 719 -> HIP_ERROR_LAUNCH_FAILURE
  | 801 -> HIP_ERROR_NOT_SUPPORTED
  | n -> HIP_ERROR_UNKNOWN n

let int_of_hip_result = function
  | HIP_SUCCESS -> 0
  | HIP_ERROR_INVALID_VALUE -> 1
  | HIP_ERROR_OUT_OF_MEMORY -> 2
  | HIP_ERROR_NOT_INITIALIZED -> 3
  | HIP_ERROR_DEINITIALIZED -> 4
  | HIP_ERROR_NO_DEVICE -> 100
  | HIP_ERROR_INVALID_DEVICE -> 101
  | HIP_ERROR_INVALID_IMAGE -> 200
  | HIP_ERROR_INVALID_CONTEXT -> 201
  | HIP_ERROR_NO_BINARY_FOR_GPU -> 209
  | HIP_ERROR_SHARED_OBJECT_INIT_FAILED -> 303
  | HIP_ERROR_INVALID_HANDLE -> 400
  | HIP_ERROR_NOT_FOUND -> 500
  | HIP_ERROR_NOT_READY -> 600
  | HIP_ERROR_LAUNCH_FAILURE -> 719
  | HIP_ERROR_NOT_SUPPORTED -> 801
  | HIP_ERROR_UNKNOWN n -> n

let hip_result : hip_result typ =
  view ~read:hip_result_of_int ~write:int_of_hip_result int

let string_of_hip_result = function
  | HIP_SUCCESS -> "hipSuccess"
  | HIP_ERROR_INVALID_VALUE -> "hipErrorInvalidValue"
  | HIP_ERROR_OUT_OF_MEMORY -> "hipErrorOutOfMemory"
  | HIP_ERROR_NOT_INITIALIZED -> "hipErrorNotInitialized"
  | HIP_ERROR_DEINITIALIZED -> "hipErrorDeinitialized"
  | HIP_ERROR_NO_DEVICE -> "hipErrorNoDevice"
  | HIP_ERROR_INVALID_DEVICE -> "hipErrorInvalidDevice"
  | HIP_ERROR_INVALID_IMAGE -> "hipErrorInvalidImage"
  | HIP_ERROR_INVALID_CONTEXT -> "hipErrorInvalidContext"
  | HIP_ERROR_INVALID_HANDLE -> "hipErrorInvalidHandle"
  | HIP_ERROR_NOT_FOUND -> "hipErrorNotFound"
  | HIP_ERROR_NOT_READY -> "hipErrorNotReady"
  | HIP_ERROR_NO_BINARY_FOR_GPU -> "hipErrorNoBinaryForGpu"
  | HIP_ERROR_SHARED_OBJECT_INIT_FAILED -> "hipErrorSharedObjectInitFailed"
  | HIP_ERROR_NOT_SUPPORTED -> "hipErrorNotSupported"
  | HIP_ERROR_LAUNCH_FAILURE -> "hipErrorLaunchFailure"
  | HIP_ERROR_UNKNOWN n -> Printf.sprintf "hipErrorUnknown(%d)" n

(** {1 Device Attributes}

    hipDeviceAttribute_t ordinals from ROCm 7.2.4 (see module header). Only the
    handful SPOC needs are bound. *)
type hip_device_attribute =
  | HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
  | HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
  | HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y
  | HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
  | HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X
  | HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y
  | HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z
  | HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
  | HIP_DEVICE_ATTRIBUTE_WARP_SIZE
  | HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
  | HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
  | HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR

let int_of_device_attribute = function
  | HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK -> 56
  | HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X -> 26
  | HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y -> 27
  | HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z -> 28
  | HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X -> 29
  | HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y -> 30
  | HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z -> 31
  | HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK -> 74
  | HIP_DEVICE_ATTRIBUTE_WARP_SIZE -> 87
  | HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT -> 63
  | HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR -> 23
  | HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR -> 61
