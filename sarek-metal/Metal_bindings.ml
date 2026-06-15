(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Metal API - Ctypes Bindings
 *
 * Direct FFI bindings to Metal API via ctypes-foreign.
 * Metal uses Objective-C, so we need C helper shims for most operations.
 * All bindings are lazy - they only load when first used.
 *
 * Note: Unlike OpenCL/CUDA, Metal is macOS/iOS only and always available
 * on these platforms via the Metal framework.
 ******************************************************************************)

open Ctypes
open Foreign
open Metal_types

(** {1 Library Loading} *)

(** Load Metal framework dynamically (lazy) *)
let metal_lib : Dl.library option Lazy.t =
  lazy
    (try
       let lib =
         Dl.dlopen
           ~filename:"/System/Library/Frameworks/Metal.framework/Metal"
           ~flags:[Dl.RTLD_LAZY]
       in
       Some lib
     with _ -> None)

(** Check if Metal library is available *)
let is_available () =
  match Lazy.force metal_lib with Some _ -> true | None -> false

(** Get Metal library, raising if not available *)
let get_metal_lib () =
  match Lazy.force metal_lib with
  | Some lib -> lib
  | None -> Metal_error.raise_error (Metal_error.library_not_found "Metal" [])

(** {1 Objective-C Runtime Helpers} *)

(** We need libobjc for calling Objective-C methods *)
let objc_lib : Dl.library option Lazy.t =
  lazy
    (try Some (Dl.dlopen ~filename:"libobjc.dylib" ~flags:[Dl.RTLD_LAZY])
     with _ -> None)

let get_objc_lib () =
  match Lazy.force objc_lib with
  | Some lib -> lib
  | None -> Metal_error.raise_error (Metal_error.library_not_found "libobjc" [])

(** objc_msgSend - the core Objective-C message dispatch *)
let objc_msgSend_lazy =
  lazy
    (foreign
       ~from:(get_objc_lib ())
       "objc_msgSend"
       (ptr void @-> ptr void @-> returning (ptr void)))

let objc_msgSend obj sel = Lazy.force objc_msgSend_lazy obj sel

(** sel_registerName - register a selector *)
let sel_registerName_lazy =
  lazy
    (foreign
       ~from:(get_objc_lib ())
       "sel_registerName"
       (string @-> returning (ptr void)))

let sel_registerName name = Lazy.force sel_registerName_lazy name

(** objc_getClass - get a class by name *)
let objc_getClass_lazy =
  lazy
    (foreign
       ~from:(get_objc_lib ())
       "objc_getClass"
       (string @-> returning (ptr void)))

let objc_getClass name = Lazy.force objc_getClass_lazy name

(** {1 Foundation Helpers} *)

(** Load Foundation framework for NSString, NSArray, etc *)
let foundation_lib : Dl.library option Lazy.t =
  lazy
    (try
       Some
         (Dl.dlopen
            ~filename:
              "/System/Library/Frameworks/Foundation.framework/Foundation"
            ~flags:[Dl.RTLD_LAZY])
     with _ -> None)

let get_foundation_lib () =
  match Lazy.force foundation_lib with
  | Some lib -> lib
  | None ->
      Metal_error.raise_error (Metal_error.library_not_found "Foundation" [])

(** Helper: Create NSString from C string *)
let nsstring_from_cstring str =
  let nsstring_class = objc_getClass "NSString" in
  let alloc_sel = sel_registerName "alloc" in
  let init_sel = sel_registerName "initWithUTF8String:" in
  let obj = objc_msgSend nsstring_class alloc_sel in
  (* For initWithUTF8String:, we need a different signature *)
  let init_fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend"
      (ptr void @-> ptr void @-> string @-> returning (ptr void))
  in
  init_fn obj init_sel str

(** Helper: Get C string from NSString *)
let cstring_from_nsstring nsstr =
  let sel = sel_registerName "UTF8String" in
  let fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend"
      (ptr void @-> ptr void @-> returning string)
  in
  fn nsstr sel

(** Helper: Get NSError description *)
let nserror_description err =
  if is_null err then "No error"
  else
    let sel = sel_registerName "localizedDescription" in
    let desc = objc_msgSend err sel in
    if is_null desc then "Unknown error" else cstring_from_nsstring desc

(** {1 Metal Device API} *)

(** MTLCreateSystemDefaultDevice - get default GPU *)
let mtl_create_system_default_device_lazy =
  lazy
    (foreign
       ~from:(get_metal_lib ())
       "MTLCreateSystemDefaultDevice"
       (void @-> returning mtl_device))

let mtl_create_system_default_device () =
  Lazy.force mtl_create_system_default_device_lazy ()

(** MTLCopyAllDevices - get all Metal devices *)
let mtl_copy_all_devices_lazy =
  lazy
    (foreign
       ~from:(get_metal_lib ())
       "MTLCopyAllDevices"
       (void @-> returning (ptr void)))
(* Returns NSArray *)

let mtl_copy_all_devices () = Lazy.force mtl_copy_all_devices_lazy ()

(** Device property getters via objc_msgSend *)
let mtl_device_name dev =
  let sel = sel_registerName "name" in
  let nsstr = objc_msgSend dev sel in
  if is_null nsstr then "Unknown Device" else cstring_from_nsstring nsstr

let mtl_device_max_threads_per_threadgroup dev =
  let sel = sel_registerName "maxThreadsPerThreadgroup" in
  let fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend_stret"
      (ptr void @-> ptr void @-> ptr void @-> returning void)
  in
  let result = allocate_n mtl_size ~count:1 in
  fn (to_voidp result) dev (to_voidp sel) ;
  !@result

let mtl_device_max_threadgroup_memory_length dev =
  let sel = sel_registerName "maxThreadgroupMemoryLength" in
  let fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend"
      (ptr void @-> ptr void @-> returning uint64_t)
  in
  Unsigned.UInt64.to_int (fn dev sel)

(** {1 Command Queue API} *)

let mtl_device_new_command_queue dev =
  let sel = sel_registerName "newCommandQueue" in
  objc_msgSend dev sel

let mtl_command_queue_command_buffer queue =
  let sel = sel_registerName "commandBuffer" in
  objc_msgSend queue sel

(** {1 Command Buffer API} *)

let mtl_command_buffer_compute_command_encoder cmdbuf =
  let sel = sel_registerName "computeCommandEncoder" in
  objc_msgSend cmdbuf sel

let mtl_command_buffer_commit cmdbuf =
  let sel = sel_registerName "commit" in
  let fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend"
      (ptr void @-> ptr void @-> returning void)
  in
  fn cmdbuf sel

let mtl_command_buffer_wait_until_completed cmdbuf =
  let sel = sel_registerName "waitUntilCompleted" in
  let fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend"
      (ptr void @-> ptr void @-> returning void)
  in
  fn cmdbuf sel

(** {1 Buffer API} *)

let mtl_device_new_buffer_with_length dev length options =
  let sel = sel_registerName "newBufferWithLength:options:" in
  let fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend"
      (ptr void @-> ptr void @-> uint64_t @-> uint64_t @-> returning mtl_buffer)
  in
  fn dev sel (Unsigned.UInt64.of_int length) options

let mtl_buffer_contents buf =
  let sel = sel_registerName "contents" in
  let fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend"
      (ptr void @-> ptr void @-> returning (ptr void))
  in
  fn buf sel

let mtl_buffer_length buf =
  let sel = sel_registerName "length" in
  let fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend"
      (ptr void @-> ptr void @-> returning uint64_t)
  in
  Unsigned.UInt64.to_int (fn buf sel)

(** {1 Library API} *)

let mtl_device_new_library_with_source dev source _options =
  let sel = sel_registerName "newLibraryWithSource:options:error:" in
  let source_ns = nsstring_from_cstring source in
  let error_ptr = allocate (ptr void) null in
  let fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend"
      (ptr void @-> ptr void @-> ns_string @-> ptr void
      @-> ptr (ptr void)
      @-> returning mtl_library)
  in
  let lib = fn dev sel source_ns null error_ptr in
  if is_null lib then begin
    let err = !@error_ptr in
    Error (nserror_description err)
  end
  else Ok lib

let mtl_library_new_function_with_name lib fname =
  let sel = sel_registerName "newFunctionWithName:" in
  let fname_ns = nsstring_from_cstring fname in
  let fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend"
      (ptr void @-> ptr void @-> ns_string @-> returning mtl_function)
  in
  fn lib sel fname_ns

(** {1 Compute Pipeline API} *)

let mtl_device_new_compute_pipeline_state dev func =
  let sel = sel_registerName "newComputePipelineStateWithFunction:error:" in
  let error_ptr = allocate (ptr void) null in
  let fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend"
      (ptr void @-> ptr void @-> mtl_function
      @-> ptr (ptr void)
      @-> returning mtl_compute_pipeline_state)
  in
  let pso = fn dev sel func error_ptr in
  if is_null pso then begin
    let err = !@error_ptr in
    Error (nserror_description err)
  end
  else Ok pso

let mtl_compute_pipeline_state_max_total_threads_per_threadgroup pso =
  let sel = sel_registerName "maxTotalThreadsPerThreadgroup" in
  let fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend"
      (ptr void @-> ptr void @-> returning uint64_t)
  in
  Unsigned.UInt64.to_int (fn pso sel)

let mtl_compute_pipeline_state_threadgroup_memory_length pso =
  let sel = sel_registerName "threadExecutionWidth" in
  let fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend"
      (ptr void @-> ptr void @-> returning uint64_t)
  in
  Unsigned.UInt64.to_int (fn pso sel)

(** {1 Compute Command Encoder API} *)

let mtl_compute_command_encoder_set_compute_pipeline_state encoder pso =
  let sel = sel_registerName "setComputePipelineState:" in
  let fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend"
      (ptr void @-> ptr void @-> mtl_compute_pipeline_state @-> returning void)
  in
  fn encoder sel pso

let mtl_compute_command_encoder_set_buffer encoder buffer offset index =
  let sel = sel_registerName "setBuffer:offset:atIndex:" in
  let fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend"
      (ptr void @-> ptr void @-> mtl_buffer @-> uint64_t @-> uint64_t
     @-> returning void)
  in
  fn
    encoder
    sel
    buffer
    (Unsigned.UInt64.of_int offset)
    (Unsigned.UInt64.of_int index)

let mtl_compute_command_encoder_set_bytes encoder bytes length index =
  let sel = sel_registerName "setBytes:length:atIndex:" in
  let fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend"
      (ptr void @-> ptr void @-> ptr void @-> uint64_t @-> uint64_t
     @-> returning void)
  in
  fn
    encoder
    sel
    bytes
    (Unsigned.UInt64.of_int length)
    (Unsigned.UInt64.of_int index)

let mtl_compute_command_encoder_dispatch_threads encoder threads
    threads_per_threadgroup =
  let sel = sel_registerName "dispatchThreads:threadsPerThreadgroup:" in
  let fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend"
      (ptr void @-> ptr void @-> mtl_size @-> mtl_size @-> returning void)
  in
  fn encoder sel threads threads_per_threadgroup

let mtl_compute_command_encoder_end_encoding encoder =
  let sel = sel_registerName "endEncoding" in
  let fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend"
      (ptr void @-> ptr void @-> returning void)
  in
  fn encoder sel

(** {1 Memory Management} *)

(** Release any NSObject/Metal object *)
let release obj =
  if not (is_null obj) then begin
    let sel = sel_registerName "release" in
    let fn =
      foreign
        ~from:(get_objc_lib ())
        "objc_msgSend"
        (ptr void @-> ptr void @-> returning void)
    in
    fn obj sel
  end

(** Retain any NSObject/Metal object *)
let retain obj =
  let sel = sel_registerName "retain" in
  objc_msgSend obj sel
