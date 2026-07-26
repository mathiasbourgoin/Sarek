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

(** {2 MTLMathMode / MTLMathFloatingPointFunctions}

    READ FROM THE SDK, not guessed: [MTLLibrary.h:241-246] and [258-262] on
    macOS 15.6.1 (Command Line Tools SDK). The previous revision of this file
    declined to use [setMathMode:] precisely because these values could not be
    checked; they can now. *)

let mtl_math_mode_safe = 0L

let mtl_math_floating_point_functions_precise = 1L

(** Build an [MTLCompileOptions] configured for Sarek's float semantics, or
    [None] if that cannot be done on this host.

    WHY (backlog #125). Metal's defaults trade accuracy for speed, and until
    this change {!mtl_device_new_library_with_source} took an [_options]
    argument and IGNORED it while [Metal_api] passed null anyway. So every Sarek
    Metal kernel took those defaults and there was no route to change them — not
    "no policy", but an UNSETTABLE wrong default.

    MEASURED on Apple M4 / macOS 15.6.1 (24G90) / Apple clang 17.0.0. A freshly
    constructed [MTLCompileOptions] reports [mathMode = 2] ([MTLMathModeFast])
    and [mathFloatingPointFunctions = 0] ([...Fast]). BOTH defaults are the fast
    one, which is why both are set here — setting [mathMode] alone still leaves
    single-precision math functions resolving to [metal::fast].

    THE OPTIONS ARE HONOURED, and that had to be established rather than
    assumed: a compile that succeeds proves plumbing, not semantics. Over 65536
    inputs on [sqrt(a) + 1/a], against the default:

    - [mathMode=Safe] alone changes 16017 results;
    - [mathMode=Safe] + [fpFunctions=Precise] changes 22135.

    So Metal is NOT in the class of rusticl's OpenCL FP options, which are
    accepted and discarded (docs/fp-contraction-policy.md §10.2).

    WHAT THESE OPTIONS DO NOT BUY: CONTRACTION. Measured on the same device,
    [a*b+c] is contracted into an fma on all 8773 observable elements under
    EVERY setting tried, including [mathMode=Safe]. Contraction is defeated in
    the generated source instead, by [#pragma METAL fp contract(off)]
    ([Sarek_ir_metal.metal_fp_contract_pragma], measured 0/8773). Do not read
    this function as a contraction defence; it is not one.

    WHY BOTH SPELLINGS. [fastMathEnabled] is deprecated since macOS 15.0 in
    favour of [mathMode], but [mathMode] does not exist before macOS 15.0 / iOS
    18.0, so the deprecated property is the only route on older systems. The
    modern pair is preferred when present and the boolean is the fallback. They
    are EQUIVALENT, measured: [fastMathEnabled = NO] and
    [mathMode=Safe + fpFunctions=Precise] are BIT-IDENTICAL over 65536 elements
    of [sqrt + reciprocal + sin + log + exp]. The fallback is therefore not a
    degraded path, and that is measured rather than believed.

    FAIL-SOFT BY CONSTRUCTION. Every failure path returns [None], and [None]
    makes the caller pass null, which is EXACTLY the behaviour that shipped
    before this change: a missing libobjc, a missing class, an OS responding to
    neither selector, or a null allocation all degrade to the old behaviour
    rather than propagating. *)
let mtl_compile_options_conformant () : mtl_compile_options option =
  try
    let cls = objc_getClass "MTLCompileOptions" in
    if is_null cls then None
    else
      let allocated = objc_msgSend cls (sel_registerName "alloc") in
      if is_null allocated then None
      else
        let obj = objc_msgSend allocated (sel_registerName "init") in
        if is_null obj then None
        else begin
          (* From here on [obj] is owned by us (+1 from [alloc]), so every exit
             that does not hand it to the caller must release it. *)
          let responds_to sel =
            let f =
              foreign
                ~from:(get_objc_lib ())
                "objc_msgSend"
                (ptr void @-> ptr void @-> ptr void @-> returning uchar)
            in
            Unsigned.UChar.to_int
              (f obj (sel_registerName "respondsToSelector:") sel)
            <> 0
          in
          (* [MTLMathMode] and [MTLMathFloatingPointFunctions] are [NS_ENUM(
             NSInteger, ...)], i.e. [long] on every Apple 64-bit platform. *)
          let set_long sel v =
            let f =
              foreign
                ~from:(get_objc_lib ())
                "objc_msgSend"
                (ptr void @-> ptr void @-> long @-> returning void)
            in
            f obj sel (Signed.Long.of_int64 v)
          in
          let sel_math_mode = sel_registerName "setMathMode:" in
          let sel_fp_funcs =
            sel_registerName "setMathFloatingPointFunctions:"
          in
          let sel_fast_math = sel_registerName "setFastMathEnabled:" in
          if responds_to sel_math_mode then begin
            set_long sel_math_mode mtl_math_mode_safe ;
            (* Independent of mathMode: without it, single-precision math
               functions still resolve to [metal::fast]. Measured: 16017 vs
               22135 changed results out of 65536. *)
            if responds_to sel_fp_funcs then
              set_long sel_fp_funcs mtl_math_floating_point_functions_precise ;
            Some obj
          end
          else if responds_to sel_fast_math then begin
            (* Pre-macOS-15 fallback. BOOL is [signed char] on x86_64 macOS and
               [_Bool] on arm64; both are one byte passed in the low byte of the
               argument register, so [uchar] is correct on both and ctypes'
               [bool] would not be. *)
            let set_bool =
              foreign
                ~from:(get_objc_lib ())
                "objc_msgSend"
                (ptr void @-> ptr void @-> uchar @-> returning void)
            in
            set_bool obj sel_fast_math (Unsigned.UChar.of_int 0) ;
            Some obj
          end
          else begin
            (* Neither spelling available: release rather than leak, and fall
               back to null options (the pre-change behaviour). Spelled out
               rather than calling {!release}, which is defined further down
               this file. *)
            let f =
              foreign
                ~from:(get_objc_lib ())
                "objc_msgSend"
                (ptr void @-> ptr void @-> returning void)
            in
            f obj (sel_registerName "release") ;
            None
          end
        end
  with _ -> None

(** Compile MSL source into an [MTLLibrary].

    [options] is now HONOURED. Before backlog #125 this function's parameter was
    named [_options] and dropped on the floor, so the only thing it could ever
    pass was null — see {!mtl_compile_options_conformant} for what that meant.
*)
let mtl_device_new_library_with_source dev source
    (options : mtl_compile_options option) =
  let sel = sel_registerName "newLibraryWithSource:options:error:" in
  let source_ns = nsstring_from_cstring source in
  let error_ptr = allocate (ptr void) null in
  let fn =
    foreign
      ~from:(get_objc_lib ())
      "objc_msgSend"
      (ptr void @-> ptr void @-> ns_string @-> mtl_compile_options
      @-> ptr (ptr void)
      @-> returning mtl_library)
  in
  let opts = match options with Some o -> o | None -> null in
  let lib = fn dev sel source_ns opts error_ptr in
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
