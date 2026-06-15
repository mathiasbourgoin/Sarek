(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Execute - Unified Kernel Execution Dispatcher
 *
 * Provides a unified interface for executing Sarek kernels across different
 * backends. Dispatches based on the backend's execution model:
 *
 * - JIT (CUDA, OpenCL): Generate source → compile → launch
 * - Direct (Native): Call pre-compiled OCaml function
 * - Custom: Delegate to backend's custom pipeline
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry
open Spoc_core

(** Re-export structured error types *)
open Execute_error

(** {1 V2 Vector Argument Type} *)

(** V2 Vector argument type - supports automatic transfers and length expansion.
    This is the main type-safe way to pass arguments to kernels. *)
type vector_arg =
  | Vec : ('a, 'b) Vector.t -> vector_arg
      (** V2 Vector - expands to (buffer, length) for JIT *)
  | Int : int -> vector_arg  (** Integer scalar *)
  | Int32 : int32 -> vector_arg  (** 32-bit integer scalar *)
  | Int64 : int64 -> vector_arg  (** 64-bit integer scalar *)
  | Float32 : float -> vector_arg  (** 32-bit float scalar *)
  | Float64 : float -> vector_arg  (** 64-bit float scalar *)

let custom_value_to_bytes : type a. a Vector.custom_type -> a -> bytes =
  Vector.custom_to_bytes

let custom_value_of_bytes : type a. a Vector.custom_type -> bytes -> a =
 fun custom bytes ->
  let size = custom.Vector.elem_size in
  if Bytes.length bytes < size then
    Execute_error.raise_error
      (Type_mismatch
         {
           expected = Printf.sprintf "%s bytes" custom.Vector.name;
           actual = Printf.sprintf "%d bytes" (Bytes.length bytes);
           context = "custom vector element assignment";
         }) ;
  Vector.custom_of_bytes custom bytes

(** Convert vector_arg list to exec_arg array (new typed interface). Creates
    EXEC_VECTOR wrappers for vectors. *)
let exec_arg_of_vector : type a b. (a, b) Vector.t -> Framework_sig.exec_arg =
 fun v ->
  let module EV :
    Typed_value.EXEC_VECTOR
      with type elt = a
       and type underlying = (a, b) Vector.t = struct
    type elt = a

    type underlying = (a, b) Vector.t

    let length = Vector.length v

    let type_name = Vector.kind_name (Vector.kind v)

    let elem_size = Vector.elem_size (Vector.kind v)

    let type_id = Vector.type_id (Vector.kind v)

    let underlying_type_id = Vector.vector_type_id (Vector.kind v)

    let underlying = v

    let device_ptr () =
      (* Get device pointer from location-based buffer *)
      match Vector.location v with
      | Vector.GPU dev | Vector.Both dev | Vector.Stale_CPU dev -> (
          match Vector.get_buffer v dev with
          | Some (module B : Vector.DEVICE_BUFFER) -> B.device_ptr
          | None ->
              Execute_error.raise_error
                (Transfer_failed
                   {vector = "unknown"; reason = "Vector has no device buffer"})
          )
      | Vector.CPU | Vector.Stale_GPU _ ->
          Execute_error.raise_error
            (Transfer_failed
               {vector = "unknown"; reason = "Vector not on device"})

    let get i =
      (* Convert element to typed_value based on vector kind *)
      match Vector.kind v with
      | Vector.Scalar Vector.Int32 ->
          Typed_value.TV_Scalar
            (Typed_value.SV ((module Typed_value.Int32_type), Vector.get v i))
      | Vector.Scalar Vector.Int64 ->
          Typed_value.TV_Scalar
            (Typed_value.SV ((module Typed_value.Int64_type), Vector.get v i))
      | Vector.Scalar Vector.Float32 ->
          Typed_value.TV_Scalar
            (Typed_value.SV ((module Typed_value.Float32_type), Vector.get v i))
      | Vector.Scalar Vector.Float64 ->
          Typed_value.TV_Scalar
            (Typed_value.SV ((module Typed_value.Float64_type), Vector.get v i))
      | Vector.Custom custom ->
          let module C : Typed_value.COMPOSITE_TYPE with type t = a = struct
            type t = a

            let name = custom.Vector.name

            let size = custom.Vector.elem_size

            let fields = []

            let to_bytes = custom_value_to_bytes custom

            let of_bytes = custom_value_of_bytes custom
          end in
          Typed_value.TV_Composite (Typed_value.CV ((module C), Vector.get v i))
      | Vector.Scalar (Vector.Char | Vector.Complex32) ->
          Execute_error.raise_error
            (Unsupported_argument
               {arg_type = type_name; context = "vector element access"})

    let set i tv =
      let type_error expected actual =
        Execute_error.raise_error
          (Type_mismatch
             {expected; actual; context = "vector element assignment"})
      in
      match (tv, Vector.kind v) with
      | ( Typed_value.TV_Scalar (Typed_value.SV ((module S), x)),
          Vector.Scalar Vector.Int32 ) -> (
          match S.to_primitive x with
          | Typed_value.PInt32 n -> Vector.set v i n
          | Typed_value.PInt64 _ -> type_error "int32" "int64"
          | Typed_value.PFloat _ -> type_error "int32" "float"
          | Typed_value.PBool _ -> type_error "int32" "bool"
          | Typed_value.PBytes _ -> type_error "int32" "bytes")
      | ( Typed_value.TV_Scalar (Typed_value.SV ((module S), x)),
          Vector.Scalar Vector.Int64 ) -> (
          match S.to_primitive x with
          | Typed_value.PInt64 n -> Vector.set v i n
          | Typed_value.PInt32 _ -> type_error "int64" "int32"
          | Typed_value.PFloat _ -> type_error "int64" "float"
          | Typed_value.PBool _ -> type_error "int64" "bool"
          | Typed_value.PBytes _ -> type_error "int64" "bytes")
      | ( Typed_value.TV_Scalar (Typed_value.SV ((module S), x)),
          Vector.Scalar Vector.Float32 ) -> (
          match S.to_primitive x with
          | Typed_value.PFloat f -> Vector.set v i f
          | Typed_value.PInt32 _ -> type_error "float" "int32"
          | Typed_value.PInt64 _ -> type_error "float" "int64"
          | Typed_value.PBool _ -> type_error "float" "bool"
          | Typed_value.PBytes _ -> type_error "float" "bytes")
      | ( Typed_value.TV_Scalar (Typed_value.SV ((module S), x)),
          Vector.Scalar Vector.Float64 ) -> (
          match S.to_primitive x with
          | Typed_value.PFloat f -> Vector.set v i f
          | Typed_value.PInt32 _ -> type_error "float" "int32"
          | Typed_value.PInt64 _ -> type_error "float" "int64"
          | Typed_value.PBool _ -> type_error "float" "bool"
          | Typed_value.PBytes _ -> type_error "float" "bytes")
      | ( Typed_value.TV_Composite (Typed_value.CV ((module C), x)),
          Vector.Custom custom ) ->
          if C.name <> custom.Vector.name || C.size <> custom.Vector.elem_size
          then type_error custom.Vector.name C.name
          else Vector.set v i (custom_value_of_bytes custom (C.to_bytes x))
      | _ ->
          Execute_error.raise_error
            (Unsupported_argument
               {
                 arg_type = "unknown combination";
                 context = "vector element assignment";
               })

    let get_typed i = Vector.get v i

    let set_typed i x = Vector.kernel_set v i x
  end in
  Framework_sig.EA_Vec (module EV)

let vector_args_to_exec_array (args : vector_arg list) :
    Framework_sig.exec_arg array =
  Array.of_list
    (List.map
       (function
         | Vec v -> exec_arg_of_vector v
         | Int n -> Framework_sig.EA_Int32 (Int32.of_int n)
         | Int32 n -> Framework_sig.EA_Int32 n
         | Int64 n -> Framework_sig.EA_Int64 n
         | Float32 f -> Framework_sig.EA_Float32 f
         | Float64 f -> Framework_sig.EA_Float64 f)
       args)

(** Retrieve device buffer for a vector on a specific device.

    Returns a first-class module containing the device buffer's pointer, size,
    and binding function. The buffer must exist (typically created by a prior
    transfer).

    @param v Vector to get buffer from
    @param dev Device the buffer should be allocated on
    @return Device buffer module
    @raise Transfer_failed if vector has no buffer on this device *)
let get_device_buffer (type a b) (v : (a, b) Vector.t) (dev : Device.t) :
    (module Vector.DEVICE_BUFFER) =
  Log.debugf Log.Execute "get_device_buffer for dev=%d" dev.Device.id ;
  match Vector.get_buffer v dev with
  | Some buf ->
      let (module B : Vector.DEVICE_BUFFER) = buf in
      Log.debugf
        Log.Execute
        "got buffer: ptr=%Ld size=%d"
        (Int64.of_nativeint B.device_ptr)
        B.size ;
      buf
  | None ->
      Execute_error.raise_error
        (Transfer_failed
           {vector = "unknown"; reason = "Vector has no device buffer"})

(** Transfer all V2 Vector args to device *)
let transfer_vectors_to_device (args : vector_arg list) (dev : Device.t) : unit
    =
  List.iter (function Vec v -> Transfer.to_device v dev | _ -> ()) args

(** Expand vector args to run_source_arg format.
    @param inject_lengths
      If true (default), auto-inject vector length as Int32 after each buffer.
      This matches Sarek-generated kernels which expect (ptr, len) pairs. Set to
      false for external kernels with different signatures. *)
let expand_to_run_source_args ?(inject_lengths = true) (args : vector_arg list)
    (dev : Device.t) : Framework_sig.run_source_arg list =
  List.concat_map
    (function
      | Vec v ->
          let buf = get_device_buffer v dev in
          let (module B : Vector.DEVICE_BUFFER) = buf in
          let len = Vector.length v in
          let buf_arg =
            Framework_sig.RSA_Buffer {binder = B.bind_to_kargs; length = len}
          in
          if inject_lengths then
            [buf_arg; Framework_sig.RSA_Int32 (Int32.of_int len)]
          else [buf_arg]
      | Int n -> [Framework_sig.RSA_Int32 (Int32.of_int n)]
      | Int32 n -> [Framework_sig.RSA_Int32 n]
      | Int64 n -> [Framework_sig.RSA_Int64 n]
      | Float32 f -> [Framework_sig.RSA_Float32 f]
      | Float64 f -> [Framework_sig.RSA_Float64 f])
    args

(** {1 Execution Dispatch} *)

(** Execute a kernel on a device using the unified dispatch mechanism.

    @param device Target device
    @param name Kernel name
    @param ir Sarek IR kernel (lazy, only forced for JIT backends)
    @param native_fn Pre-compiled native function (for Direct backends)
    @param block Block dimensions
    @param grid Grid dimensions
    @param shared_mem Shared memory size in bytes (default 0)
    @param args Kernel arguments as vector_arg list
    @raise Execution_error if execution fails *)
let run ~(device : Device.t) ~(name : string)
    ~(ir : Sarek_ir_types.kernel Lazy.t option)
    ~(native_fn :
       (block:Framework_sig.dims ->
       grid:Framework_sig.dims ->
       Framework_sig.exec_arg array ->
       unit)
       option) ~(block : Framework_sig.dims) ~(grid : Framework_sig.dims)
    ?(shared_mem : int = 0) (args : vector_arg list) : unit =
  match Framework_registry.find_backend device.framework with
  | None ->
      Execute_error.raise_error
        (Backend_error
           {
             backend = device.framework;
             message = "Backend not found in registry";
           })
  | Some (module B : Framework_sig.BACKEND) -> (
      match B.execution_model with
      | Framework_sig.JIT -> (
          (* JIT path: generate source, compile, use B.run_source *)
          match ir with
          | None -> Execute_error.raise_error (Missing_ir {kernel = name})
          | Some ir_lazy -> (
              let ir = Lazy.force ir_lazy in
              match B.generate_source ~block ir with
              | None ->
                  Execute_error.raise_error
                    (Compilation_failed
                       {
                         kernel = name;
                         reason = "Backend failed to generate source";
                       })
              | Some source ->
                  (* Convert vector args to run_source_arg format (auto-injects lengths) *)
                  let rs_args = expand_to_run_source_args args device in
                  (* Set current device *)
                  let dev = B.Device.get device.backend_id in
                  B.Device.set_current dev ;
                  (* Determine source language from backend's supported langs *)
                  let lang =
                    match B.supported_source_langs with
                    | [] ->
                        Execute_error.raise_error
                          (Backend_error
                             {
                               backend = device.framework;
                               message = "No supported source languages";
                             })
                    | lang :: _ -> lang
                  in
                  (* Use backend's run_source - handles compilation and launch *)
                  B.run_source
                    ~source
                    ~lang
                    ~kernel_name:name
                    ~block
                    ~grid
                    ~shared_mem
                    rs_args))
      | Framework_sig.Direct ->
          (* Direct path: call native function or interpret IR *)
          let dev = B.Device.get device.backend_id in
          B.Device.set_current dev ;
          let exec_args = vector_args_to_exec_array args in
          let ir_val = Option.map Lazy.force ir in
          B.execute_direct ~native_fn ~ir:ir_val ~block ~grid exec_args
      | Framework_sig.Custom ->
          (* Custom path: delegate to backend with IR *)
          let dev = B.Device.get device.backend_id in
          B.Device.set_current dev ;
          let exec_args = vector_args_to_exec_array args in
          let ir_val = Option.map Lazy.force ir in
          B.execute_direct ~native_fn ~ir:ir_val ~block ~grid exec_args)

(** {1 V2 Vector Execution Helpers} *)

(** Mark vectors as stale on CPU after kernel execution.

    After a kernel modifies vector data on a device, we need to track that the
    CPU-side data is now stale. This ensures future CPU reads will trigger a
    device→CPU transfer.

    Special cases:
    - Native backend: No-op (uses zero-copy shared memory, no staleness)
    - JIT backends: Always mark stale (Transfer module handles zero-copy checks)
    - OpenCL CPU: Mark stale for custom types (scalar types use zero-copy)

    @param args Arguments that may contain vectors
    @param dev Device that just executed the kernel *)
let mark_vectors_stale (args : vector_arg list) (dev : Device.t) : unit =
  (* Only Native uses true zero-copy for all vector types.
     OpenCL CPU uses zero-copy for scalar types but NOT for custom types.
     Always mark stale for JIT backends - Transfer will check zero_copy. *)
  if dev.Device.framework = "Native" then ()
  else
    List.iter
      (function
        | Vec v -> (
            (* Mark as Stale_CPU: device has authoritative data, CPU is stale *)
            match v.Vector.location with
            | Vector.Both _ -> v.Vector.location <- Vector.Stale_CPU dev
            | _ -> ())
        | _ -> ())
      args

(** Convert a V2 Vector to interpreter value array.

    Converts vectors of primitive types (int32, float32, etc.) to the
    interpreter's runtime value representation. Custom types are converted using
    registered type helpers from Sarek_type_helpers.

    This enables the interpreter backend to execute kernels on CPU without
    requiring GPU infrastructure.

    @param vec Input vector of any type
    @return Array of interpreter values matching vector contents *)
let vector_to_interp_array : type a b.
    (a, b) Vector.t -> Sarek_ir_interp.value array =
 fun vec ->
  let len = Vector.length vec in
  match Vector.kind vec with
  | Vector.Scalar Vector.Int32 ->
      Array.init len (fun i -> Sarek_ir_interp.VInt32 (Vector.get vec i))
  | Vector.Scalar Vector.Int64 ->
      Array.init len (fun i -> Sarek_ir_interp.VInt64 (Vector.get vec i))
  | Vector.Scalar Vector.Float32 ->
      Array.init len (fun i -> Sarek_ir_interp.VFloat32 (Vector.get vec i))
  | Vector.Scalar Vector.Float64 ->
      Array.init len (fun i -> Sarek_ir_interp.VFloat64 (Vector.get vec i))
  | Vector.Custom custom -> (
      (* Custom types: use helpers to convert to VRecord *)
      let type_name = custom.Vector.name in
      match Sarek_type_helpers.lookup_typed custom.Vector.type_id with
      | Some (module H) ->
          Array.init len (fun i ->
              let native_record = Vector.get vec i in
              H.to_value native_record)
      | None ->
          (* Fallback: wrap in VRecord with empty fields *)
          Array.init len (fun _i -> Sarek_ir_interp.VRecord (type_name, [||])))
  | Vector.Scalar Vector.Char ->
      (* Char type: convert to int32 *)
      Array.init len (fun i ->
          Sarek_ir_interp.VInt32 (Int32.of_int (Char.code (Vector.get vec i))))
  | Vector.Scalar Vector.Complex32 ->
      (* Complex32: not directly supported, skip for now *)
      Array.init len (fun _i -> Sarek_ir_interp.VUnit)

(** Copy interpreter value array back to V2 Vector.

    After interpreter execution, this function copies the runtime values back
    into the typed vector representation. Performs type checking and conversion
    for each element.

    @param arr Array of interpreter runtime values
    @param vec Destination vector (must match type of values) *)
let interp_array_to_vector : type a b.
    Sarek_ir_interp.value array -> (a, b) Vector.t -> unit =
 fun arr vec ->
  let len = min (Array.length arr) (Vector.length vec) in
  match Vector.kind vec with
  | Vector.Scalar Vector.Int32 ->
      for i = 0 to len - 1 do
        Vector.set vec i (Sarek_ir_interp.to_int32 arr.(i))
      done
  | Vector.Scalar Vector.Int64 ->
      for i = 0 to len - 1 do
        Vector.set vec i (Sarek_ir_interp.to_int64 arr.(i))
      done
  | Vector.Scalar Vector.Float32 ->
      for i = 0 to len - 1 do
        Vector.set vec i (Sarek_ir_interp.to_float32 arr.(i))
      done
  | Vector.Scalar Vector.Float64 ->
      for i = 0 to len - 1 do
        Vector.set vec i (Sarek_ir_interp.to_float64 arr.(i))
      done
  | Vector.Custom custom ->
      (* Custom types: convert VRecord to native OCaml values using helpers *)
      for i = 0 to len - 1 do
        match arr.(i) with
        | Sarek_ir_interp.VRecord (type_name, _fields) as vrec -> (
            match Sarek_type_helpers.lookup_typed custom.Vector.type_id with
            | Some (module H) ->
                let native_record = H.from_value vrec in
                Vector.set vec i native_record
            | None ->
                Execute_error.raise_error
                  (Execute_error.Type_helper_not_found
                     {
                       type_name;
                       context =
                         "sync_vector_back (record conversion from interpreter)";
                     }))
        | _ -> () (* Skip other values *)
      done
  | Vector.Scalar Vector.Char ->
      (* Char type: convert from int32 *)
      for i = 0 to len - 1 do
        Vector.set
          vec
          i
          (Char.chr (Int32.to_int (Sarek_ir_interp.to_int32 arr.(i))))
      done
  | Vector.Scalar Vector.Complex32 ->
      (* Complex32: not directly supported, skip for now *)
      ()

(** Run kernel via interpreter with V2 Vectors. Note: Interpreter works with IR
    params directly - one arg per param. Vectors map to ArgArray (length is
    intrinsic to array). *)
let run_interpreter_vectors ~(ir : Sarek_ir_types.kernel)
    ~(args : vector_arg list) ~(block : Framework_sig.dims)
    ~(grid : Framework_sig.dims) ~(parallel : bool) : unit =
  (* Set interpreter parallel mode *)
  Sarek_ir_interp.parallel_mode := parallel ;
  (* Convert vector args to interpreter format, tracking arrays for writeback *)
  let writebacks : Sarek_ir_interp.writeback list ref = ref [] in

  (* Extract param names from kernel IR (only DParam entries) *)
  let param_names =
    List.filter_map
      (function
        | Sarek_ir_types.DParam (v, _) -> Some v.Sarek_ir_types.var_name
        | _ -> None)
      ir.Sarek_ir_types.kern_params
  in

  (* Build args matching kernel params 1:1, using actual param names *)
  let interp_args =
    List.mapi
      (fun i arg ->
        let name =
          if i < List.length param_names then List.nth param_names i
          else Printf.sprintf "param%d" i
        in
        match arg with
        | Vec v ->
            let arr = vector_to_interp_array v in
            writebacks := Sarek_ir_interp.Writeback (v, arr) :: !writebacks ;
            (name, Sarek_ir_interp.ArgArray arr)
        | Int n ->
            ( name,
              Sarek_ir_interp.ArgScalar
                (Sarek_ir_interp.VInt32 (Int32.of_int n)) )
        | Int32 n -> (name, Sarek_ir_interp.ArgScalar (Sarek_ir_interp.VInt32 n))
        | Int64 n -> (name, Sarek_ir_interp.ArgScalar (Sarek_ir_interp.VInt64 n))
        | Float32 f ->
            (name, Sarek_ir_interp.ArgScalar (Sarek_ir_interp.VFloat32 f))
        | Float64 f ->
            (name, Sarek_ir_interp.ArgScalar (Sarek_ir_interp.VFloat64 f)))
      args
  in

  (* Run interpreter *)
  Sarek_ir_interp.run_kernel
    ir
    ~block:(block.x, block.y, block.z)
    ~grid:(grid.x, grid.y, grid.z)
    interp_args ;

  (* Copy results back to vectors *)
  List.iter
    (fun (Sarek_ir_interp.Writeback (vec, arr)) ->
      interp_array_to_vector arr vec)
    !writebacks

(** Execute a kernel with V2 Vectors. Auto-transfers, dispatches to backend.

    This is the main execution entry point for Sarek-generated kernels. It
    performs the complete execution pipeline:

    1. **Transfer**: Move vectors to device (no-op for CPU backends) 2.
    **Dispatch**: Call appropriate backend's execution method 3. **Mark stale**:
    Update vector location tracking

    The function automatically handles differences between execution models:
    - JIT backends: Generate source, compile, launch
    - Direct (Native): Call pre-compiled OCaml function
    - Custom (Interpreter): Walk IR and evaluate expressions

    @param device Target device (determines backend)
    @param ir Sarek IR kernel definition
    @param args Kernel arguments (vectors and scalars)
    @param block Thread block dimensions (e.g., (256, 1, 1))
    @param grid Grid dimensions (e.g., (4, 1, 1))
    @param shared_mem Optional shared memory size in bytes (default: 0)
    @raise Execute_error on validation or execution failure *)
let run_vectors ~(device : Device.t) ~(ir : Sarek_ir_types.kernel)
    ~(args : vector_arg list) ~(block : Framework_sig.dims)
    ~(grid : Framework_sig.dims) ?(shared_mem : int = 0) () : unit =
  (* Unified path for all backends:
     1. Transfer to device (CPU backends: zero-copy, no actual transfer)
     2. Pass vector_args to run (it expands for JIT, passes direct for Direct)
     3. Mark stale (CPU backends: no-op due to zero-copy) *)

  (* 1. Transfer all vectors to device *)
  transfer_vectors_to_device args device ;

  (* 2. Dispatch via run - it handles expansion per backend *)
  run
    ~device
    ~name:ir.kern_name
    ~ir:(Some (lazy ir))
    ~native_fn:None
    ~block
    ~grid
    ~shared_mem
    args ;

  (* 3. Mark vectors as stale (no-op for CPU backends due to zero-copy) *)
  mark_vectors_stale args device

(** Sync all V2 Vector outputs back to CPU *)
let sync_vectors_to_cpu (args : vector_arg list) : unit =
  List.iter (function Vec v -> Transfer.to_cpu v | _ -> ()) args

(** {1 Convenience Functions} *)

(** Create 1D grid and block dimensions *)
let dims1d size = Framework_sig.dims_1d size

(** Create 2D grid and block dimensions *)
let dims2d x y = Framework_sig.dims_2d x y

(** Create 3D grid and block dimensions *)
let dims3d x y z = Framework_sig.dims_3d x y z

(** Calculate grid size for a given problem size and block size *)
let grid_for_size ~problem_size ~block_size =
  (problem_size + block_size - 1) / block_size

(** Calculate 1D grid dimensions for a problem size *)
let grid_for ~problem_size ~block_size =
  dims1d (grid_for_size ~problem_size ~block_size)

(** {1 External Kernel Execution} *)

(** Re-export source language type *)
type source_lang = Framework_sig.source_lang =
  | CUDA_Source
  | OpenCL_Source
  | PTX
  | SPIR_V
  | GLSL_Source

(** Check if a device supports a given source language.

    Different backends support different source languages:
    - CUDA: CUDA source (.cu), PTX
    - OpenCL: OpenCL source (.cl)
    - Vulkan: SPIR-V, GLSL source (.comp, .glsl)
    - Native/Interpreter: None (not JIT backends)

    @param dev Device to check
    @param lang Source language to query
    @return true if device can compile and execute this language *)
let supports_lang (dev : Device.t) (lang : source_lang) : bool =
  match Framework_registry.find_backend dev.framework with
  | Some (module B : Framework_sig.BACKEND) ->
      List.mem lang B.supported_source_langs
  | None -> false

(** Execute an external kernel from source code.

    This function allows running pre-written GPU kernels (CUDA, OpenCL, PTX)
    directly without going through the Sarek DSL.

    @param device Target device
    @param source Kernel source code as string
    @param lang Source language (CUDA_Source, OpenCL_Source, PTX)
    @param kernel_name Name of the kernel function in the source
    @param block Block dimensions
    @param grid Grid dimensions
    @param shared_mem Shared memory size in bytes (default 0)
    @param inject_lengths
      If true (default), auto-inject vector length as Int32 after each buffer
      argument. Sarek-generated kernels expect (ptr, len) pairs. Set to false
      for external kernels that don't follow this convention.
    @param args Kernel arguments as vector_arg list
    @raise Execution_error if device doesn't support the source language *)
let run_source ~(device : Device.t) ~(source : string) ~(lang : source_lang)
    ~(kernel_name : string) ~(block : Framework_sig.dims)
    ~(grid : Framework_sig.dims) ?(shared_mem : int = 0)
    ?(inject_lengths : bool = true) (args : vector_arg list) : unit =
  (* Transfer vectors to device first *)
  transfer_vectors_to_device args device ;

  match Framework_registry.find_backend device.framework with
  | Some (module B : Framework_sig.BACKEND) ->
      (if not (List.mem lang B.supported_source_langs) then
         let lang_name =
           match lang with
           | CUDA_Source -> "CUDA source"
           | OpenCL_Source -> "OpenCL source"
           | PTX -> "PTX"
           | SPIR_V -> "SPIR-V"
           | GLSL_Source -> "GLSL source"
         in
         Execute_error.raise_error
           (Unsupported_argument
              {arg_type = lang_name; context = device.framework ^ " backend"})) ;

      (* Expand vector args to run_source_arg format for external kernels *)
      let rs_args = expand_to_run_source_args ~inject_lengths args device in

      (* Set current device and run *)
      let dev = B.Device.get device.backend_id in
      B.Device.set_current dev ;
      B.run_source ~source ~lang ~kernel_name ~block ~grid ~shared_mem rs_args ;

      (* Mark vectors as stale *)
      mark_vectors_stale args device
  | None ->
      Execute_error.raise_error
        (Backend_error
           {
             backend = device.framework;
             message = "Backend not found in registry";
           })

(** Load kernel source from a file *)
let load_source (path : string) : string =
  In_channel.with_open_text path In_channel.input_all

(** Detect source language from file extension *)
let detect_lang (path : string) : source_lang =
  if String.ends_with ~suffix:".cu" path then CUDA_Source
  else if String.ends_with ~suffix:".cl" path then OpenCL_Source
  else if String.ends_with ~suffix:".ptx" path then PTX
  else if String.ends_with ~suffix:".spv" path then SPIR_V
  else if String.ends_with ~suffix:".comp" path then GLSL_Source
  else if String.ends_with ~suffix:".glsl" path then GLSL_Source
  else
    Execute_error.raise_error
      (Invalid_file
         {
           path;
           reason =
             "Unknown source file extension (expected .cu, .cl, .ptx, .spv, \
              .comp, or .glsl)";
         })

(** Execute an external kernel from a file.

    Loads pre-written GPU kernel source from a file and executes it. Useful for
    integrating hand-optimized kernels or using features not yet supported by
    the Sarek PPX.

    Source language is auto-detected from file extension:
    - .cu → CUDA source
    - .cl → OpenCL source
    - .ptx → PTX assembly
    - .spv → SPIR-V binary
    - .comp / .glsl → GLSL compute shader

    Example:
    {[
      (* Execute hand-written CUDA kernel *)
      Execute.run_source_file
        ~device:(Device.get_default ())
        ~path:"kernels/optimized_matmul.cu"
        ~kernel_name:"matmul_kernel"
        ~block:(16, 16, 1)
        ~grid:(64, 64, 1)
        [Vec a; Vec b; Vec c; Int32 1024l]
    ]}

    @param device Target device
    @param path Path to kernel source file
    @param kernel_name Name of kernel function in source
    @param block Block dimensions
    @param grid Grid dimensions
    @param shared_mem Optional shared memory in bytes
    @param inject_lengths
      If true (default), inject vector lengths after buffer args
    @param args Kernel arguments
    @raise Invalid_file if file extension is not recognized
    @raise Execute_error if compilation or execution fails *)
let run_source_file ~(device : Device.t) ~(path : string)
    ~(kernel_name : string) ~(block : Framework_sig.dims)
    ~(grid : Framework_sig.dims) ?(shared_mem : int = 0)
    ?(inject_lengths : bool = true) (args : vector_arg list) : unit =
  let source = load_source path in
  let lang = detect_lang path in
  run_source
    ~device
    ~source
    ~lang
    ~kernel_name
    ~block
    ~grid
    ~shared_mem
    ~inject_lengths
    args
