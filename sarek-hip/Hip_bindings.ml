(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * HIP Runtime/Driver API - Ctypes Bindings
 *
 * Direct FFI bindings to the AMD ROCm/HIP API (libamdhip64) via
 * ctypes-foreign. All bindings are lazy - they only dlopen the library when
 * first used, so this module links fine on machines without ROCm.
 *
 * This is the HIP analog of Cuda_bindings. We use HIP's runtime device model
 * (hipSetDevice) rather than the deprecated hipCtx* API, which keeps the
 * surface small and robust; module load/launch use the driver-style
 * hipModule* entry points, which are shape-identical to their CUDA cousins.
 ******************************************************************************)

open Ctypes
open Foreign
open Hip_types

(** {1 Library Loading} *)

let hip_lib : Dl.library option Lazy.t =
  lazy
    (try
       Some
         (Dl.dlopen
            ~filename:"libamdhip64.so"
            ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL])
     with _ -> (
       try
         Some
           (Dl.dlopen
              ~filename:"libamdhip64.so.7"
              ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL])
       with _ -> (
         try
           Some
             (Dl.dlopen
                ~filename:"libamdhip64.so.6"
                ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL])
         with _ -> None)))

let is_available () =
  match Lazy.force hip_lib with Some _ -> true | None -> false

let get_hip_lib () =
  match Lazy.force hip_lib with
  | Some lib -> lib
  | None ->
      Hip_error.raise_error
        (Hip_error.library_not_found
           "libamdhip64"
           ["libamdhip64.so"; "libamdhip64.so.7"; "libamdhip64.so.6"])

let foreign_hip_lazy name typ = lazy (foreign ~from:(get_hip_lib ()) name typ)

(** {1 Initialization} *)

let hipInit_lazy = foreign_hip_lazy "hipInit" (uint @-> returning hip_result)

let hipInit flags = Lazy.force hipInit_lazy flags

(** {1 Device Management} *)

let hipGetDeviceCount_lazy =
  foreign_hip_lazy "hipGetDeviceCount" (ptr int @-> returning hip_result)

let hipGetDeviceCount p = Lazy.force hipGetDeviceCount_lazy p

let hipDeviceGet_lazy =
  foreign_hip_lazy
    "hipDeviceGet"
    (ptr hip_device @-> int @-> returning hip_result)

let hipDeviceGet p i = Lazy.force hipDeviceGet_lazy p i

let hipSetDevice_lazy =
  foreign_hip_lazy "hipSetDevice" (int @-> returning hip_result)

let hipSetDevice d = Lazy.force hipSetDevice_lazy d

let hipDeviceGetName_lazy =
  foreign_hip_lazy
    "hipDeviceGetName"
    (ptr char @-> int @-> hip_device @-> returning hip_result)

let hipDeviceGetName p len d = Lazy.force hipDeviceGetName_lazy p len d

let hipDeviceTotalMem_lazy =
  foreign_hip_lazy
    "hipDeviceTotalMem"
    (ptr size_t @-> hip_device @-> returning hip_result)

let hipDeviceTotalMem p d = Lazy.force hipDeviceTotalMem_lazy p d

let hipDeviceGetAttribute_lazy =
  foreign_hip_lazy
    "hipDeviceGetAttribute"
    (ptr int @-> int @-> int @-> returning hip_result)

let hipDeviceGetAttribute p attr d =
  Lazy.force hipDeviceGetAttribute_lazy p attr d

let hipDeviceComputeCapability_lazy =
  foreign_hip_lazy
    "hipDeviceComputeCapability"
    (ptr int @-> ptr int @-> hip_device @-> returning hip_result)

let hipDeviceComputeCapability major minor d =
  Lazy.force hipDeviceComputeCapability_lazy major minor d

let hipDeviceSynchronize_lazy =
  foreign_hip_lazy "hipDeviceSynchronize" (void @-> returning hip_result)

let hipDeviceSynchronize () = Lazy.force hipDeviceSynchronize_lazy ()

(** {1 Memory Management} *)

let hipMalloc_lazy =
  foreign_hip_lazy
    "hipMalloc"
    (ptr (ptr void) @-> size_t @-> returning hip_result)

let hipMalloc p size = Lazy.force hipMalloc_lazy p size

let hipFree_lazy = foreign_hip_lazy "hipFree" (ptr void @-> returning hip_result)

let hipFree ptr = Lazy.force hipFree_lazy ptr

let hipMemcpyHtoD_lazy =
  foreign_hip_lazy
    "hipMemcpyHtoD"
    (hip_deviceptr @-> ptr void @-> size_t @-> returning hip_result)

let hipMemcpyHtoD dst src size = Lazy.force hipMemcpyHtoD_lazy dst src size

let hipMemcpyDtoH_lazy =
  foreign_hip_lazy
    "hipMemcpyDtoH"
    (ptr void @-> hip_deviceptr @-> size_t @-> returning hip_result)

let hipMemcpyDtoH dst src size = Lazy.force hipMemcpyDtoH_lazy dst src size

let hipMemcpyDtoD_lazy =
  foreign_hip_lazy
    "hipMemcpyDtoD"
    (hip_deviceptr @-> hip_deviceptr @-> size_t @-> returning hip_result)

let hipMemcpyDtoD dst src size = Lazy.force hipMemcpyDtoD_lazy dst src size

let hipMemsetD8_lazy =
  foreign_hip_lazy
    "hipMemsetD8"
    (hip_deviceptr @-> uchar @-> size_t @-> returning hip_result)

let hipMemsetD8 ptr value count = Lazy.force hipMemsetD8_lazy ptr value count

let hipMemGetInfo_lazy =
  foreign_hip_lazy
    "hipMemGetInfo"
    (ptr size_t @-> ptr size_t @-> returning hip_result)

let hipMemGetInfo free total = Lazy.force hipMemGetInfo_lazy free total

(** {1 Module Management} *)

let hipModuleLoadData_lazy =
  foreign_hip_lazy
    "hipModuleLoadData"
    (ptr hip_module_ptr @-> ptr void @-> returning hip_result)

let hipModuleLoadData p data = Lazy.force hipModuleLoadData_lazy p data

let hipModuleLoad_lazy =
  foreign_hip_lazy
    "hipModuleLoad"
    (ptr hip_module_ptr @-> string @-> returning hip_result)

let hipModuleLoad p fname = Lazy.force hipModuleLoad_lazy p fname

let hipModuleUnload_lazy =
  foreign_hip_lazy "hipModuleUnload" (hip_module_ptr @-> returning hip_result)

let hipModuleUnload m = Lazy.force hipModuleUnload_lazy m

let hipModuleGetFunction_lazy =
  foreign_hip_lazy
    "hipModuleGetFunction"
    (ptr hip_function_ptr @-> hip_module_ptr @-> string @-> returning hip_result)

let hipModuleGetFunction p m name =
  Lazy.force hipModuleGetFunction_lazy p m name

(** {1 Kernel Execution} *)

let hipModuleLaunchKernel_lazy =
  foreign_hip_lazy
    "hipModuleLaunchKernel"
    (hip_function_ptr @-> uint @-> uint @-> uint @-> uint @-> uint @-> uint
   @-> uint @-> hip_stream_ptr
    @-> ptr (ptr void)
    @-> ptr (ptr void)
    @-> returning hip_result)

let hipModuleLaunchKernel f gx gy gz bx by bz shm stream params extra =
  Lazy.force
    hipModuleLaunchKernel_lazy
    f
    gx
    gy
    gz
    bx
    by
    bz
    shm
    stream
    params
    extra

(** {1 Stream Management} *)

let hipStreamCreate_lazy =
  foreign_hip_lazy
    "hipStreamCreate"
    (ptr hip_stream_ptr @-> returning hip_result)

let hipStreamCreate p = Lazy.force hipStreamCreate_lazy p

let hipStreamDestroy_lazy =
  foreign_hip_lazy "hipStreamDestroy" (hip_stream_ptr @-> returning hip_result)

let hipStreamDestroy s = Lazy.force hipStreamDestroy_lazy s

let hipStreamSynchronize_lazy =
  foreign_hip_lazy
    "hipStreamSynchronize"
    (hip_stream_ptr @-> returning hip_result)

let hipStreamSynchronize s = Lazy.force hipStreamSynchronize_lazy s

(** {1 Event Management} *)

let hipEventCreate_lazy =
  foreign_hip_lazy "hipEventCreate" (ptr hip_event_ptr @-> returning hip_result)

let hipEventCreate p = Lazy.force hipEventCreate_lazy p

let hipEventDestroy_lazy =
  foreign_hip_lazy "hipEventDestroy" (hip_event_ptr @-> returning hip_result)

let hipEventDestroy e = Lazy.force hipEventDestroy_lazy e

let hipEventRecord_lazy =
  foreign_hip_lazy
    "hipEventRecord"
    (hip_event_ptr @-> hip_stream_ptr @-> returning hip_result)

let hipEventRecord e s = Lazy.force hipEventRecord_lazy e s

let hipEventSynchronize_lazy =
  foreign_hip_lazy "hipEventSynchronize" (hip_event_ptr @-> returning hip_result)

let hipEventSynchronize e = Lazy.force hipEventSynchronize_lazy e

let hipEventElapsedTime_lazy =
  foreign_hip_lazy
    "hipEventElapsedTime"
    (ptr float @-> hip_event_ptr @-> hip_event_ptr @-> returning hip_result)

let hipEventElapsedTime t start stop =
  Lazy.force hipEventElapsedTime_lazy t start stop

(** {1 Version} *)

let hipRuntimeGetVersion_lazy =
  foreign_hip_lazy "hipRuntimeGetVersion" (ptr int @-> returning hip_result)

let hipRuntimeGetVersion p = Lazy.force hipRuntimeGetVersion_lazy p
