(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * CUDA Driver API - Ctypes Bindings
 *
 * Direct FFI bindings to CUDA driver API via ctypes-foreign.
 * All bindings are lazy - they only dlopen the library when first used.
 * This allows the module to be linked even on systems without CUDA.
 *
 * TODO: Performance optimization opportunity
 * ==========================================
 * Currently each binding uses Lazy.force which adds ~5-10ns overhead per call
 * (atomic read + branch after first evaluation). For hot paths like
 * cuLaunchKernel called in tight loops, this may be noticeable.
 *
 * Optimization approach:
 * 1. Use mutable refs: let cuInit_fn : (int -> cu_result) option ref = ref None
 * 2. Add init_bindings() that populates all refs at once via dlsym
 * 3. Call init_bindings() once from is_available() or Device.init()
 * 4. Direct function calls: match !cuInit_fn with Some f -> f x | None -> ...
 *
 * This eliminates per-call overhead after initialization while preserving
 * the lazy loading benefit (no dlopen on non-CUDA systems).
 *
 * Current approach is acceptable because kernel launch time (~100μs-10ms)
 * dwarfs the lazy overhead (~5ns), making it <0.01% of total execution.
 ******************************************************************************)

open Ctypes
open Foreign
open Cuda_types

(** {1 Library Loading} *)

(** Load CUDA driver library dynamically (lazy) *)
let cuda_lib : Dl.library option Lazy.t =
  lazy
    (try
       Some
         (Dl.dlopen
            ~filename:"libcuda.so.1"
            ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL])
     with _ -> (
       try
         Some
           (Dl.dlopen
              ~filename:"libcuda.so"
              ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL])
       with _ -> (
         try
           Some
             (Dl.dlopen
                ~filename:"libcuda.dylib"
                ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL])
         with _ -> (
           try
             Some
               (Dl.dlopen
                  ~filename:"nvcuda.dll"
                  ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL])
           with _ -> None))))

(** Check if CUDA library is available *)
let is_available () =
  match Lazy.force cuda_lib with Some _ -> true | None -> false

(** Get CUDA library, raising if not available *)
let get_cuda_lib () =
  match Lazy.force cuda_lib with
  | Some lib -> lib
  | None ->
      Cuda_error.raise_error
        (Cuda_error.library_not_found
           "libcuda"
           ["libcuda.so.1"; "libcuda.so"; "libcuda.dylib"; "nvcuda.dll"])

(** Create a lazy foreign binding to CUDA driver API *)
let foreign_cuda_lazy name typ = lazy (foreign ~from:(get_cuda_lib ()) name typ)

(** {1 Initialization} *)

let cuInit_lazy = foreign_cuda_lazy "cuInit" (int @-> returning cu_result)

let cuInit flags = Lazy.force cuInit_lazy flags

(** {1 Device Management} *)

let cuDeviceGetCount_lazy =
  foreign_cuda_lazy "cuDeviceGetCount" (ptr int @-> returning cu_result)

let cuDeviceGetCount p = Lazy.force cuDeviceGetCount_lazy p

let cuDeviceGet_lazy =
  foreign_cuda_lazy "cuDeviceGet" (ptr cu_device @-> int @-> returning cu_result)

let cuDeviceGet p i = Lazy.force cuDeviceGet_lazy p i

let cuDeviceGetName_lazy =
  foreign_cuda_lazy
    "cuDeviceGetName"
    (ptr char @-> int @-> cu_device @-> returning cu_result)

let cuDeviceGetName p len d = Lazy.force cuDeviceGetName_lazy p len d

let cuDeviceTotalMem_lazy =
  foreign_cuda_lazy
    "cuDeviceTotalMem_v2"
    (ptr size_t @-> cu_device @-> returning cu_result)

let cuDeviceTotalMem p d = Lazy.force cuDeviceTotalMem_lazy p d

let cuDeviceGetAttribute_lazy =
  foreign_cuda_lazy
    "cuDeviceGetAttribute"
    (ptr int @-> int @-> cu_device @-> returning cu_result)

let cuDeviceGetAttribute p attr d =
  Lazy.force cuDeviceGetAttribute_lazy p attr d

let cuDeviceComputeCapability_lazy =
  foreign_cuda_lazy
    "cuDeviceComputeCapability"
    (ptr int @-> ptr int @-> cu_device @-> returning cu_result)

let cuDeviceComputeCapability major minor d =
  Lazy.force cuDeviceComputeCapability_lazy major minor d

(** {1 Context Management} *)

let cuCtxCreate_lazy =
  foreign_cuda_lazy
    "cuCtxCreate_v2"
    (ptr cu_context_ptr @-> uint @-> cu_device @-> returning cu_result)

let cuCtxCreate p flags d = Lazy.force cuCtxCreate_lazy p flags d

let cuCtxDestroy_lazy =
  foreign_cuda_lazy "cuCtxDestroy_v2" (cu_context_ptr @-> returning cu_result)

let cuCtxDestroy ctx = Lazy.force cuCtxDestroy_lazy ctx

let cuCtxPushCurrent_lazy =
  foreign_cuda_lazy
    "cuCtxPushCurrent_v2"
    (cu_context_ptr @-> returning cu_result)

let cuCtxPushCurrent ctx = Lazy.force cuCtxPushCurrent_lazy ctx

let cuCtxPopCurrent_lazy =
  foreign_cuda_lazy
    "cuCtxPopCurrent_v2"
    (ptr cu_context_ptr @-> returning cu_result)

let cuCtxPopCurrent p = Lazy.force cuCtxPopCurrent_lazy p

let cuCtxSetCurrent_lazy =
  foreign_cuda_lazy "cuCtxSetCurrent" (cu_context_ptr @-> returning cu_result)

let cuCtxSetCurrent ctx = Lazy.force cuCtxSetCurrent_lazy ctx

let cuCtxGetCurrent_lazy =
  foreign_cuda_lazy
    "cuCtxGetCurrent"
    (ptr cu_context_ptr @-> returning cu_result)

let cuCtxGetCurrent p = Lazy.force cuCtxGetCurrent_lazy p

let cuCtxSynchronize_lazy =
  foreign_cuda_lazy "cuCtxSynchronize" (void @-> returning cu_result)

let cuCtxSynchronize () = Lazy.force cuCtxSynchronize_lazy ()

let cuCtxGetDevice_lazy =
  foreign_cuda_lazy "cuCtxGetDevice" (ptr cu_device @-> returning cu_result)

let cuCtxGetDevice p = Lazy.force cuCtxGetDevice_lazy p

(** {1 Memory Management} *)

let cuMemAlloc_lazy =
  foreign_cuda_lazy
    "cuMemAlloc_v2"
    (ptr cu_deviceptr @-> size_t @-> returning cu_result)

let cuMemAlloc p size = Lazy.force cuMemAlloc_lazy p size

let cuMemFree_lazy =
  foreign_cuda_lazy "cuMemFree_v2" (cu_deviceptr @-> returning cu_result)

let cuMemFree ptr = Lazy.force cuMemFree_lazy ptr

let cuMemcpyHtoD_lazy =
  foreign_cuda_lazy
    "cuMemcpyHtoD_v2"
    (cu_deviceptr @-> ptr void @-> size_t @-> returning cu_result)

let cuMemcpyHtoD dst src size = Lazy.force cuMemcpyHtoD_lazy dst src size

let cuMemcpyDtoH_lazy =
  foreign_cuda_lazy
    "cuMemcpyDtoH_v2"
    (ptr void @-> cu_deviceptr @-> size_t @-> returning cu_result)

let cuMemcpyDtoH dst src size = Lazy.force cuMemcpyDtoH_lazy dst src size

let cuMemcpyDtoD_lazy =
  foreign_cuda_lazy
    "cuMemcpyDtoD_v2"
    (cu_deviceptr @-> cu_deviceptr @-> size_t @-> returning cu_result)

let cuMemcpyDtoD dst src size = Lazy.force cuMemcpyDtoD_lazy dst src size

let cuMemcpyHtoDAsync_lazy =
  foreign_cuda_lazy
    "cuMemcpyHtoDAsync_v2"
    (cu_deviceptr @-> ptr void @-> size_t @-> cu_stream_ptr
   @-> returning cu_result)

let cuMemcpyHtoDAsync dst src size stream =
  Lazy.force cuMemcpyHtoDAsync_lazy dst src size stream

let cuMemcpyDtoHAsync_lazy =
  foreign_cuda_lazy
    "cuMemcpyDtoHAsync_v2"
    (ptr void @-> cu_deviceptr @-> size_t @-> cu_stream_ptr
   @-> returning cu_result)

let cuMemcpyDtoHAsync dst src size stream =
  Lazy.force cuMemcpyDtoHAsync_lazy dst src size stream

let cuMemsetD8_lazy =
  foreign_cuda_lazy
    "cuMemsetD8_v2"
    (cu_deviceptr @-> uchar @-> size_t @-> returning cu_result)

let cuMemsetD8 ptr value count = Lazy.force cuMemsetD8_lazy ptr value count

let cuMemsetD32_lazy =
  foreign_cuda_lazy
    "cuMemsetD32_v2"
    (cu_deviceptr @-> uint32_t @-> size_t @-> returning cu_result)

let cuMemsetD32 ptr value count = Lazy.force cuMemsetD32_lazy ptr value count

let cuMemAllocHost_lazy =
  foreign_cuda_lazy
    "cuMemAllocHost_v2"
    (ptr (ptr void) @-> size_t @-> returning cu_result)

let cuMemAllocHost p size = Lazy.force cuMemAllocHost_lazy p size

let cuMemFreeHost_lazy =
  foreign_cuda_lazy "cuMemFreeHost" (ptr void @-> returning cu_result)

let cuMemFreeHost ptr = Lazy.force cuMemFreeHost_lazy ptr

let cuMemGetInfo_lazy =
  foreign_cuda_lazy
    "cuMemGetInfo_v2"
    (ptr size_t @-> ptr size_t @-> returning cu_result)

let cuMemGetInfo free total = Lazy.force cuMemGetInfo_lazy free total

(** {1 Module Management} *)

let cuModuleLoadData_lazy =
  foreign_cuda_lazy
    "cuModuleLoadData"
    (ptr cu_module_ptr @-> ptr void @-> returning cu_result)

let cuModuleLoadData p data = Lazy.force cuModuleLoadData_lazy p data

let cuModuleLoad_lazy =
  foreign_cuda_lazy
    "cuModuleLoad"
    (ptr cu_module_ptr @-> string @-> returning cu_result)

let cuModuleLoad p fname = Lazy.force cuModuleLoad_lazy p fname

let cuModuleLoadDataEx_lazy =
  foreign_cuda_lazy
    "cuModuleLoadDataEx"
    (ptr cu_module_ptr @-> ptr void @-> uint @-> ptr int
    @-> ptr (ptr void)
    @-> returning cu_result)

let cuModuleLoadDataEx p data num opts vals =
  Lazy.force cuModuleLoadDataEx_lazy p data num opts vals

let cuModuleUnload_lazy =
  foreign_cuda_lazy "cuModuleUnload" (cu_module_ptr @-> returning cu_result)

let cuModuleUnload m = Lazy.force cuModuleUnload_lazy m

let cuModuleGetFunction_lazy =
  foreign_cuda_lazy
    "cuModuleGetFunction"
    (ptr cu_function_ptr @-> cu_module_ptr @-> string @-> returning cu_result)

let cuModuleGetFunction p m name = Lazy.force cuModuleGetFunction_lazy p m name

let cuModuleGetGlobal_lazy =
  foreign_cuda_lazy
    "cuModuleGetGlobal_v2"
    (ptr cu_deviceptr @-> ptr size_t @-> cu_module_ptr @-> string
   @-> returning cu_result)

let cuModuleGetGlobal ptr size m name =
  Lazy.force cuModuleGetGlobal_lazy ptr size m name

(** {1 Kernel Execution} *)

let cuLaunchKernel_lazy =
  foreign_cuda_lazy
    "cuLaunchKernel"
    (cu_function_ptr @-> uint @-> uint @-> uint @-> uint @-> uint @-> uint
   @-> uint @-> cu_stream_ptr
    @-> ptr (ptr void)
    @-> ptr (ptr void)
    @-> returning cu_result)

let cuLaunchKernel f gx gy gz bx by bz shm stream params extra =
  Lazy.force cuLaunchKernel_lazy f gx gy gz bx by bz shm stream params extra

let cuFuncGetAttribute_lazy =
  foreign_cuda_lazy
    "cuFuncGetAttribute"
    (ptr int @-> int @-> cu_function_ptr @-> returning cu_result)

let cuFuncGetAttribute p attr f = Lazy.force cuFuncGetAttribute_lazy p attr f

let cuFuncSetCacheConfig_lazy =
  foreign_cuda_lazy
    "cuFuncSetCacheConfig"
    (cu_function_ptr @-> int @-> returning cu_result)

let cuFuncSetCacheConfig f config =
  Lazy.force cuFuncSetCacheConfig_lazy f config

let cuFuncSetSharedMemConfig_lazy =
  foreign_cuda_lazy
    "cuFuncSetSharedMemConfig"
    (cu_function_ptr @-> int @-> returning cu_result)

let cuFuncSetSharedMemConfig f config =
  Lazy.force cuFuncSetSharedMemConfig_lazy f config

(** {1 Stream Management} *)

let cuStreamCreate_lazy =
  foreign_cuda_lazy
    "cuStreamCreate"
    (ptr cu_stream_ptr @-> uint @-> returning cu_result)

let cuStreamCreate p flags = Lazy.force cuStreamCreate_lazy p flags

let cuStreamCreateWithPriority_lazy =
  foreign_cuda_lazy
    "cuStreamCreateWithPriority"
    (ptr cu_stream_ptr @-> uint @-> int @-> returning cu_result)

let cuStreamCreateWithPriority p flags prio =
  Lazy.force cuStreamCreateWithPriority_lazy p flags prio

let cuStreamDestroy_lazy =
  foreign_cuda_lazy "cuStreamDestroy_v2" (cu_stream_ptr @-> returning cu_result)

let cuStreamDestroy s = Lazy.force cuStreamDestroy_lazy s

let cuStreamSynchronize_lazy =
  foreign_cuda_lazy "cuStreamSynchronize" (cu_stream_ptr @-> returning cu_result)

let cuStreamSynchronize s = Lazy.force cuStreamSynchronize_lazy s

let cuStreamQuery_lazy =
  foreign_cuda_lazy "cuStreamQuery" (cu_stream_ptr @-> returning cu_result)

let cuStreamQuery s = Lazy.force cuStreamQuery_lazy s

let cuStreamWaitEvent_lazy =
  foreign_cuda_lazy
    "cuStreamWaitEvent"
    (cu_stream_ptr @-> cu_event_ptr @-> uint @-> returning cu_result)

let cuStreamWaitEvent s e flags = Lazy.force cuStreamWaitEvent_lazy s e flags

let cuCtxGetStreamPriorityRange_lazy =
  foreign_cuda_lazy
    "cuCtxGetStreamPriorityRange"
    (ptr int @-> ptr int @-> returning cu_result)

let cuCtxGetStreamPriorityRange lo hi =
  Lazy.force cuCtxGetStreamPriorityRange_lazy lo hi

(** {1 Event Management} *)

let cuEventCreate_lazy =
  foreign_cuda_lazy
    "cuEventCreate"
    (ptr cu_event_ptr @-> uint @-> returning cu_result)

let cuEventCreate p flags = Lazy.force cuEventCreate_lazy p flags

let cuEventDestroy_lazy =
  foreign_cuda_lazy "cuEventDestroy_v2" (cu_event_ptr @-> returning cu_result)

let cuEventDestroy e = Lazy.force cuEventDestroy_lazy e

let cuEventRecord_lazy =
  foreign_cuda_lazy
    "cuEventRecord"
    (cu_event_ptr @-> cu_stream_ptr @-> returning cu_result)

let cuEventRecord e s = Lazy.force cuEventRecord_lazy e s

let cuEventSynchronize_lazy =
  foreign_cuda_lazy "cuEventSynchronize" (cu_event_ptr @-> returning cu_result)

let cuEventSynchronize e = Lazy.force cuEventSynchronize_lazy e

let cuEventQuery_lazy =
  foreign_cuda_lazy "cuEventQuery" (cu_event_ptr @-> returning cu_result)

let cuEventQuery e = Lazy.force cuEventQuery_lazy e

let cuEventElapsedTime_lazy =
  foreign_cuda_lazy
    "cuEventElapsedTime"
    (ptr float @-> cu_event_ptr @-> cu_event_ptr @-> returning cu_result)

let cuEventElapsedTime t start stop =
  Lazy.force cuEventElapsedTime_lazy t start stop

(** {1 Error Handling} *)

let cuGetErrorName_lazy =
  foreign_cuda_lazy
    "cuGetErrorName"
    (cu_result @-> ptr string @-> returning cu_result)

let cuGetErrorName err p = Lazy.force cuGetErrorName_lazy err p

let cuGetErrorString_lazy =
  foreign_cuda_lazy
    "cuGetErrorString"
    (cu_result @-> ptr string @-> returning cu_result)

let cuGetErrorString err p = Lazy.force cuGetErrorString_lazy err p

(** {1 Version} *)

let cuDriverGetVersion_lazy =
  foreign_cuda_lazy "cuDriverGetVersion" (ptr int @-> returning cu_result)

let cuDriverGetVersion p = Lazy.force cuDriverGetVersion_lazy p

(** {1 Occupancy} *)

let cuOccupancyMaxActiveBlocksPerMultiprocessor_lazy =
  foreign_cuda_lazy
    "cuOccupancyMaxActiveBlocksPerMultiprocessor"
    (ptr int @-> cu_function_ptr @-> int @-> size_t @-> returning cu_result)

let cuOccupancyMaxActiveBlocksPerMultiprocessor p f bs dsmem =
  Lazy.force cuOccupancyMaxActiveBlocksPerMultiprocessor_lazy p f bs dsmem

(** {1 Peer Access} *)

let cuDeviceCanAccessPeer_lazy =
  foreign_cuda_lazy
    "cuDeviceCanAccessPeer"
    (ptr int @-> cu_device @-> cu_device @-> returning cu_result)

let cuDeviceCanAccessPeer p d1 d2 =
  Lazy.force cuDeviceCanAccessPeer_lazy p d1 d2

let cuCtxEnablePeerAccess_lazy =
  foreign_cuda_lazy
    "cuCtxEnablePeerAccess"
    (cu_context_ptr @-> uint @-> returning cu_result)

let cuCtxEnablePeerAccess ctx flags =
  Lazy.force cuCtxEnablePeerAccess_lazy ctx flags

let cuCtxDisablePeerAccess_lazy =
  foreign_cuda_lazy
    "cuCtxDisablePeerAccess"
    (cu_context_ptr @-> returning cu_result)

let cuCtxDisablePeerAccess ctx = Lazy.force cuCtxDisablePeerAccess_lazy ctx

(** {1 Profiler Control} *)

let cuProfilerStart_lazy =
  lazy
    (try
       Some
         (foreign
            ~from:(get_cuda_lib ())
            "cuProfilerStart"
            (void @-> returning cu_result))
     with _ -> None)

let cuProfilerStart () =
  match Lazy.force cuProfilerStart_lazy with
  | Some f -> f ()
  | None -> CUDA_SUCCESS

let cuProfilerStop_lazy =
  lazy
    (try
       Some
         (foreign
            ~from:(get_cuda_lib ())
            "cuProfilerStop"
            (void @-> returning cu_result))
     with _ -> None)

let cuProfilerStop () =
  match Lazy.force cuProfilerStop_lazy with
  | Some f -> f ()
  | None -> CUDA_SUCCESS
