(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * OpenCL API - Ctypes Bindings
 *
 * Direct FFI bindings to OpenCL API via ctypes-foreign.
 * All bindings are lazy - they only dlopen the library when first used.
 * This allows the module to be linked even on systems without OpenCL.
 *
 * TODO: Performance optimization opportunity
 * ==========================================
 * Currently each binding uses Lazy.force which adds ~5-10ns overhead per call.
 * See Cuda_bindings.ml for the optimization approach using mutable refs.
 * Current approach is acceptable because kernel launch time dwarfs the overhead.
 ******************************************************************************)

open Ctypes
open Foreign
open Opencl_types

(** {1 Library Loading} *)

(** Load OpenCL library dynamically (lazy) *)
let opencl_lib : Dl.library option Lazy.t =
  lazy
    (try Some (Dl.dlopen ~filename:"libOpenCL.so.1" ~flags:[Dl.RTLD_LAZY])
     with _ -> (
       try Some (Dl.dlopen ~filename:"libOpenCL.so" ~flags:[Dl.RTLD_LAZY])
       with _ -> (
         try Some (Dl.dlopen ~filename:"libOpenCL.dylib" ~flags:[Dl.RTLD_LAZY])
         with _ -> (
           try
             Some
               (Dl.dlopen
                  ~filename:"/System/Library/Frameworks/OpenCL.framework/OpenCL"
                  ~flags:[Dl.RTLD_LAZY])
           with _ -> (
             try Some (Dl.dlopen ~filename:"OpenCL.dll" ~flags:[Dl.RTLD_LAZY])
             with _ -> None)))))

(** Check if OpenCL library is available *)
let is_available () =
  match Lazy.force opencl_lib with Some _ -> true | None -> false

(** Get OpenCL library, raising if not available *)
let get_opencl_lib () =
  match Lazy.force opencl_lib with
  | Some lib -> lib
  | None ->
      Opencl_error.raise_error
        (Opencl_error.library_not_found
           "libOpenCL.so"
           ["/usr/lib"; "/usr/local/lib"; "/opt/lib"])

(** Create a lazy foreign binding to OpenCL *)
let foreign_cl_lazy name typ = lazy (foreign ~from:(get_opencl_lib ()) name typ)

(** {1 Platform API} *)

let clGetPlatformIDs_lazy =
  foreign_cl_lazy
    "clGetPlatformIDs"
    (cl_uint @-> ptr cl_platform_id @-> ptr cl_uint @-> returning cl_int)

let clGetPlatformIDs n p np = Lazy.force clGetPlatformIDs_lazy n p np

let clGetPlatformInfo_lazy =
  foreign_cl_lazy
    "clGetPlatformInfo"
    (cl_platform_id @-> cl_uint @-> size_t @-> ptr void @-> ptr size_t
   @-> returning cl_int)

let clGetPlatformInfo p pn sz v szr =
  Lazy.force clGetPlatformInfo_lazy p pn sz v szr

(** {1 Device API} *)

let clGetDeviceIDs_lazy =
  foreign_cl_lazy
    "clGetDeviceIDs"
    (cl_platform_id @-> cl_bitfield @-> cl_uint @-> ptr cl_device_id
   @-> ptr cl_uint @-> returning cl_int)

let clGetDeviceIDs p dt n d nd = Lazy.force clGetDeviceIDs_lazy p dt n d nd

let clGetDeviceInfo_lazy =
  foreign_cl_lazy
    "clGetDeviceInfo"
    (cl_device_id @-> cl_uint @-> size_t @-> ptr void @-> ptr size_t
   @-> returning cl_int)

let clGetDeviceInfo d pn sz v szr =
  Lazy.force clGetDeviceInfo_lazy d pn sz v szr

(** {1 Context API} *)

let clCreateContext_lazy =
  foreign_cl_lazy
    "clCreateContext"
    (ptr cl_ulong @-> cl_uint @-> ptr cl_device_id @-> ptr void @-> ptr void
   @-> ptr cl_int @-> returning cl_context)

let clCreateContext props nd devs cb ud err =
  Lazy.force clCreateContext_lazy props nd devs cb ud err

let clCreateContextFromType_lazy =
  foreign_cl_lazy
    "clCreateContextFromType"
    (ptr cl_ulong @-> cl_bitfield @-> ptr void @-> ptr void @-> ptr cl_int
   @-> returning cl_context)

let clCreateContextFromType props dt cb ud err =
  Lazy.force clCreateContextFromType_lazy props dt cb ud err

let clRetainContext_lazy =
  foreign_cl_lazy "clRetainContext" (cl_context @-> returning cl_int)

let clRetainContext ctx = Lazy.force clRetainContext_lazy ctx

let clReleaseContext_lazy =
  foreign_cl_lazy "clReleaseContext" (cl_context @-> returning cl_int)

let clReleaseContext ctx = Lazy.force clReleaseContext_lazy ctx

let clGetContextInfo_lazy =
  foreign_cl_lazy
    "clGetContextInfo"
    (cl_context @-> cl_uint @-> size_t @-> ptr void @-> ptr size_t
   @-> returning cl_int)

let clGetContextInfo ctx pn sz v szr =
  Lazy.force clGetContextInfo_lazy ctx pn sz v szr

(** {1 Command Queue API} *)

let clCreateCommandQueueWithProperties_lazy =
  lazy
    (try
       Some
         (foreign
            ~from:(get_opencl_lib ())
            "clCreateCommandQueueWithProperties"
            (cl_context @-> cl_device_id @-> ptr cl_ulong @-> ptr cl_int
           @-> returning cl_command_queue))
     with _ -> None)

let clCreateCommandQueueWithProperties ctx d props err =
  match Lazy.force clCreateCommandQueueWithProperties_lazy with
  | Some f -> Some (f ctx d props err)
  | None -> None

let clCreateCommandQueue_lazy =
  lazy
    (try
       Some
         (foreign
            ~from:(get_opencl_lib ())
            "clCreateCommandQueue"
            (cl_context @-> cl_device_id @-> cl_bitfield @-> ptr cl_int
           @-> returning cl_command_queue))
     with _ -> None)

let clCreateCommandQueue ctx d props err =
  match Lazy.force clCreateCommandQueue_lazy with
  | Some f -> f ctx d props err
  | None ->
      err <-@ -34l ;
      from_voidp void null

let clRetainCommandQueue_lazy =
  foreign_cl_lazy "clRetainCommandQueue" (cl_command_queue @-> returning cl_int)

let clRetainCommandQueue q = Lazy.force clRetainCommandQueue_lazy q

let clReleaseCommandQueue_lazy =
  foreign_cl_lazy "clReleaseCommandQueue" (cl_command_queue @-> returning cl_int)

let clReleaseCommandQueue q = Lazy.force clReleaseCommandQueue_lazy q

let clFlush_lazy =
  foreign_cl_lazy "clFlush" (cl_command_queue @-> returning cl_int)

let clFlush q = Lazy.force clFlush_lazy q

let clFinish_lazy =
  foreign_cl_lazy "clFinish" (cl_command_queue @-> returning cl_int)

let clFinish q = Lazy.force clFinish_lazy q

(** {1 Memory Object API} *)

let clCreateBuffer_lazy =
  foreign_cl_lazy
    "clCreateBuffer"
    (cl_context @-> cl_bitfield @-> size_t @-> ptr void @-> ptr cl_int
   @-> returning cl_mem)

let clCreateBuffer ctx flags sz hp err =
  Lazy.force clCreateBuffer_lazy ctx flags sz hp err

let clRetainMemObject_lazy =
  foreign_cl_lazy "clRetainMemObject" (cl_mem @-> returning cl_int)

let clRetainMemObject m = Lazy.force clRetainMemObject_lazy m

let clReleaseMemObject_lazy =
  foreign_cl_lazy "clReleaseMemObject" (cl_mem @-> returning cl_int)

let clReleaseMemObject m = Lazy.force clReleaseMemObject_lazy m

let clGetMemObjectInfo_lazy =
  foreign_cl_lazy
    "clGetMemObjectInfo"
    (cl_mem @-> cl_uint @-> size_t @-> ptr void @-> ptr size_t
   @-> returning cl_int)

let clGetMemObjectInfo m pn sz v szr =
  Lazy.force clGetMemObjectInfo_lazy m pn sz v szr

(** {1 Enqueue Operations} *)

let clEnqueueReadBuffer_lazy =
  foreign_cl_lazy
    "clEnqueueReadBuffer"
    (cl_command_queue @-> cl_mem @-> cl_bool @-> size_t @-> size_t @-> ptr void
   @-> cl_uint @-> ptr cl_event @-> ptr cl_event @-> returning cl_int)

let clEnqueueReadBuffer q b bl off sz p ne el e =
  Lazy.force clEnqueueReadBuffer_lazy q b bl off sz p ne el e

let clEnqueueWriteBuffer_lazy =
  foreign_cl_lazy
    "clEnqueueWriteBuffer"
    (cl_command_queue @-> cl_mem @-> cl_bool @-> size_t @-> size_t @-> ptr void
   @-> cl_uint @-> ptr cl_event @-> ptr cl_event @-> returning cl_int)

let clEnqueueWriteBuffer q b bl off sz p ne el e =
  Lazy.force clEnqueueWriteBuffer_lazy q b bl off sz p ne el e

let clEnqueueCopyBuffer_lazy =
  foreign_cl_lazy
    "clEnqueueCopyBuffer"
    (cl_command_queue @-> cl_mem @-> cl_mem @-> size_t @-> size_t @-> size_t
   @-> cl_uint @-> ptr cl_event @-> ptr cl_event @-> returning cl_int)

let clEnqueueCopyBuffer q sb db so do_ sz ne el e =
  Lazy.force clEnqueueCopyBuffer_lazy q sb db so do_ sz ne el e

let clEnqueueFillBuffer_lazy =
  lazy
    (try
       Some
         (foreign
            ~from:(get_opencl_lib ())
            "clEnqueueFillBuffer"
            (cl_command_queue @-> cl_mem @-> ptr void @-> size_t @-> size_t
           @-> size_t @-> cl_uint @-> ptr cl_event @-> ptr cl_event
           @-> returning cl_int))
     with _ -> None)

let clEnqueueFillBuffer q b pat psz off sz ne el e =
  match Lazy.force clEnqueueFillBuffer_lazy with
  | Some f -> f q b pat psz off sz ne el e
  | None -> -30l

(** {1 Program API} *)

let clCreateProgramWithSource_lazy =
  foreign_cl_lazy
    "clCreateProgramWithSource"
    (cl_context @-> cl_uint @-> ptr string @-> ptr size_t @-> ptr cl_int
   @-> returning cl_program)

let clCreateProgramWithSource ctx cnt strs lens err =
  Lazy.force clCreateProgramWithSource_lazy ctx cnt strs lens err

let clCreateProgramWithBinary_lazy =
  foreign_cl_lazy
    "clCreateProgramWithBinary"
    (cl_context @-> cl_uint @-> ptr cl_device_id @-> ptr size_t
    @-> ptr (ptr uchar)
    @-> ptr cl_int @-> ptr cl_int @-> returning cl_program)

let clCreateProgramWithBinary ctx nd dl lens bins bs err =
  Lazy.force clCreateProgramWithBinary_lazy ctx nd dl lens bins bs err

let clBuildProgram_lazy =
  foreign_cl_lazy
    "clBuildProgram"
    (cl_program @-> cl_uint @-> ptr cl_device_id @-> string @-> ptr void
   @-> ptr void @-> returning cl_int)

let clBuildProgram p nd dl opts cb ud =
  Lazy.force clBuildProgram_lazy p nd dl opts cb ud

let clGetProgramBuildInfo_lazy =
  foreign_cl_lazy
    "clGetProgramBuildInfo"
    (cl_program @-> cl_device_id @-> cl_uint @-> size_t @-> ptr void
   @-> ptr size_t @-> returning cl_int)

let clGetProgramBuildInfo p d pn sz v szr =
  Lazy.force clGetProgramBuildInfo_lazy p d pn sz v szr

let clRetainProgram_lazy =
  foreign_cl_lazy "clRetainProgram" (cl_program @-> returning cl_int)

let clRetainProgram p = Lazy.force clRetainProgram_lazy p

let clReleaseProgram_lazy =
  foreign_cl_lazy "clReleaseProgram" (cl_program @-> returning cl_int)

let clReleaseProgram p = Lazy.force clReleaseProgram_lazy p

(** {1 Kernel API} *)

let clCreateKernel_lazy =
  foreign_cl_lazy
    "clCreateKernel"
    (cl_program @-> string @-> ptr cl_int @-> returning cl_kernel)

let clCreateKernel p name err = Lazy.force clCreateKernel_lazy p name err

let clSetKernelArg_lazy =
  foreign_cl_lazy
    "clSetKernelArg"
    (cl_kernel @-> cl_uint @-> size_t @-> ptr void @-> returning cl_int)

let clSetKernelArg k idx sz v = Lazy.force clSetKernelArg_lazy k idx sz v

let clRetainKernel_lazy =
  foreign_cl_lazy "clRetainKernel" (cl_kernel @-> returning cl_int)

let clRetainKernel k = Lazy.force clRetainKernel_lazy k

let clReleaseKernel_lazy =
  foreign_cl_lazy "clReleaseKernel" (cl_kernel @-> returning cl_int)

let clReleaseKernel k = Lazy.force clReleaseKernel_lazy k

let clGetKernelInfo_lazy =
  foreign_cl_lazy
    "clGetKernelInfo"
    (cl_kernel @-> cl_uint @-> size_t @-> ptr void @-> ptr size_t
   @-> returning cl_int)

let clGetKernelInfo k pn sz v szr =
  Lazy.force clGetKernelInfo_lazy k pn sz v szr

let clGetKernelWorkGroupInfo_lazy =
  foreign_cl_lazy
    "clGetKernelWorkGroupInfo"
    (cl_kernel @-> cl_device_id @-> cl_uint @-> size_t @-> ptr void
   @-> ptr size_t @-> returning cl_int)

let clGetKernelWorkGroupInfo k d pn sz v szr =
  Lazy.force clGetKernelWorkGroupInfo_lazy k d pn sz v szr

let clEnqueueNDRangeKernel_lazy =
  foreign_cl_lazy
    "clEnqueueNDRangeKernel"
    (cl_command_queue @-> cl_kernel @-> cl_uint @-> ptr size_t @-> ptr size_t
   @-> ptr size_t @-> cl_uint @-> ptr cl_event @-> ptr cl_event
   @-> returning cl_int)

let clEnqueueNDRangeKernel q k dim gwo gws lws ne el e =
  Lazy.force clEnqueueNDRangeKernel_lazy q k dim gwo gws lws ne el e

(** {1 Event API} *)

let clWaitForEvents_lazy =
  foreign_cl_lazy
    "clWaitForEvents"
    (cl_uint @-> ptr cl_event @-> returning cl_int)

let clWaitForEvents n el = Lazy.force clWaitForEvents_lazy n el

let clRetainEvent_lazy =
  foreign_cl_lazy "clRetainEvent" (cl_event @-> returning cl_int)

let clRetainEvent e = Lazy.force clRetainEvent_lazy e

let clReleaseEvent_lazy =
  foreign_cl_lazy "clReleaseEvent" (cl_event @-> returning cl_int)

let clReleaseEvent e = Lazy.force clReleaseEvent_lazy e

let clGetEventInfo_lazy =
  foreign_cl_lazy
    "clGetEventInfo"
    (cl_event @-> cl_uint @-> size_t @-> ptr void @-> ptr size_t
   @-> returning cl_int)

let clGetEventInfo e pn sz v szr = Lazy.force clGetEventInfo_lazy e pn sz v szr

let clGetEventProfilingInfo_lazy =
  foreign_cl_lazy
    "clGetEventProfilingInfo"
    (cl_event @-> cl_uint @-> size_t @-> ptr void @-> ptr size_t
   @-> returning cl_int)

let clGetEventProfilingInfo e pn sz v szr =
  Lazy.force clGetEventProfilingInfo_lazy e pn sz v szr

(** {1 Synchronization} *)

let clEnqueueMarkerWithWaitList_lazy =
  lazy
    (try
       Some
         (foreign
            ~from:(get_opencl_lib ())
            "clEnqueueMarkerWithWaitList"
            (cl_command_queue @-> cl_uint @-> ptr cl_event @-> ptr cl_event
           @-> returning cl_int))
     with _ -> None)

let clEnqueueMarkerWithWaitList q ne el e =
  match Lazy.force clEnqueueMarkerWithWaitList_lazy with
  | Some f -> f q ne el e
  | None -> -30l

let clEnqueueBarrierWithWaitList_lazy =
  lazy
    (try
       Some
         (foreign
            ~from:(get_opencl_lib ())
            "clEnqueueBarrierWithWaitList"
            (cl_command_queue @-> cl_uint @-> ptr cl_event @-> ptr cl_event
           @-> returning cl_int))
     with _ -> None)

let clEnqueueBarrierWithWaitList q ne el e =
  match Lazy.force clEnqueueBarrierWithWaitList_lazy with
  | Some f -> f q ne el e
  | None -> -30l
