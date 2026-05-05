(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Native Plugin - CPU Execution Backend
 *
 * Implements the Framework_sig.BACKEND interface for CPU execution.
 * This is a Direct backend - no JIT compilation needed.
 *
 * Kernels are pre-compiled OCaml functions registered by the PPX at module
 * load time. The "compile" step is a no-op - we look up registered functions.
 ******************************************************************************)

open Spoc_framework

[@@@warning "-21-69"]

(** Registry for native kernel functions.

    Kernels are registered by name. The function signature uses exec_arg array
    for type-safe argument passing:
    - args: Framework_sig.exec_arg array (typed kernel arguments)
    - grid: (gx, gy, gz) grid dimensions
    - block: (bx, by, bz) block dimensions

    Defined at module level so it can be accessed by both the Native module and
    the external registration functions. *)
let native_kernels :
    ( string,
      Framework_sig.exec_arg array -> int * int * int -> int * int * int -> unit
    )
    Hashtbl.t =
  Hashtbl.create 16

module Native : sig
  val name : string

  val version : int * int * int

  module Device : sig
    type t

    type id = int

    val init : unit -> unit

    val count : unit -> int

    val get : int -> t

    val id : t -> id

    val name : t -> string

    val capabilities : t -> Framework_sig.capabilities

    val set_current : t -> unit

    val synchronize : t -> unit
  end

  module Memory : sig
    type 'a buffer

    val alloc : Device.t -> int -> ('a, 'b) Bigarray.kind -> 'a buffer

    val alloc_custom : Device.t -> size:int -> elem_size:int -> 'a buffer

    val alloc_zero_copy :
      Device.t ->
      ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t ->
      ('a, 'b) Bigarray.kind ->
      'a buffer option

    val is_zero_copy : 'a buffer -> bool

    val free : 'a buffer -> unit

    val host_to_device :
      src:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> dst:'a buffer -> unit

    val device_to_host :
      src:'a buffer -> dst:('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> unit

    val host_ptr_to_device :
      src_ptr:unit Ctypes.ptr -> byte_size:int -> dst:'a buffer -> unit

    val device_to_host_ptr :
      src:'a buffer -> dst_ptr:unit Ctypes.ptr -> byte_size:int -> unit

    val device_to_device : src:'a buffer -> dst:'a buffer -> unit

    val size : 'a buffer -> int

    val device_ptr : 'a buffer -> nativeint
  end

  module Stream : sig
    type t

    val create : Device.t -> t

    val destroy : t -> unit

    val synchronize : t -> unit

    val default : Device.t -> t
  end

  module Event : sig
    type t

    val create : unit -> t

    val destroy : t -> unit

    val record : t -> Stream.t -> unit

    val synchronize : t -> unit

    val elapsed : start:t -> stop:t -> float
  end

  module Kernel : sig
    type t

    type args

    val compile : Device.t -> name:string -> source:string -> t

    val compile_cached : Device.t -> name:string -> source:string -> t

    val create_args : unit -> args

    val set_arg_buffer : args -> int -> _ Memory.buffer -> unit

    val set_arg_int32 : args -> int -> int32 -> unit

    val set_arg_int64 : args -> int -> int64 -> unit

    val set_arg_float32 : args -> int -> float -> unit

    val set_arg_float64 : args -> int -> float -> unit

    val set_arg_ptr : args -> int -> nativeint -> unit

    val launch :
      t ->
      args:args ->
      grid:Framework_sig.dims ->
      block:Framework_sig.dims ->
      shared_mem:int ->
      stream:Stream.t option ->
      unit

    val clear_cache : unit -> unit
  end

  val enable_profiling : unit -> unit

  val disable_profiling : unit -> unit

  val is_available : unit -> bool
end = struct
  let name = "Native"

  let version = (1, 0, 0)

  module Device = struct
    type t = {id : int; name : string; num_cores : int; parallel : bool}

    type id = int

    let devices : t array ref = ref [||]

    let current : t option ref = ref None

    let init () =
      if Array.length !devices = 0 then begin
        let num_cores = try Domain.recommended_domain_count () with _ -> 1 in
        devices :=
          [|
            {
              id = 0;
              name = Printf.sprintf "CPU Native (Parallel, %d cores)" num_cores;
              num_cores;
              parallel = true;
            };
          |]
      end

    let count () = Array.length !devices

    let get idx =
      if idx < 0 || idx >= Array.length !devices then
        Native_error.(
          raise_error (device_not_found idx (Array.length !devices)))
      else !devices.(idx)

    let id d = d.id

    let name d = d.name

    let capabilities d : Framework_sig.capabilities =
      (* Get approximate system memory - use 16GB as reasonable default *)
      let total_mem =
        try
          let ic = open_in "/proc/meminfo" in
          let line = input_line ic in
          close_in ic ;
          (* Parse "MemTotal:       32657436 kB" *)
          Scanf.sscanf line "MemTotal: %Ld kB" (fun kb -> Int64.mul kb 1024L)
        with _ -> Int64.of_int (16 * 1024 * 1024 * 1024)
        (* 16 GB default *)
      in
      {
        max_threads_per_block = d.num_cores;
        max_block_dims = (d.num_cores, 1, 1);
        max_grid_dims = (max_int, max_int, max_int);
        shared_mem_per_block = 1024 * 1024 * 1024;
        total_global_mem = total_mem;
        compute_capability = (0, 0);
        (* Native has no compute capability *)
        supports_fp64 = true;
        supports_atomics = true;
        warp_size = 1;
        max_registers_per_block = 0;
        clock_rate_khz = 0;
        multiprocessor_count = d.num_cores;
        is_cpu = true;
      }

    let set_current d = current := Some d

    let synchronize _d = () (* CPU is always synchronized *)
  end

  module Memory = struct
    (* For CPU, buffers are just bigarrays or raw ctypes memory.
       No actual device transfer - we just keep pointers. *)

    (** Element kind - carries type information for buffers. Single type
        parameter 'a = actual OCaml element type. The Bigarray elt type is
        erased (wildcard) since it's always derivable. *)
    type 'a element_kind =
      | Scalar_kind :
          ('a, 'b) Spoc_core.Vector_types.scalar_kind
          -> 'a element_kind
      | Custom_kind : 'a Spoc_core.Vector_types.custom_type -> 'a element_kind

    (** Buffer storage - GADT with typed parameter. For Bigarray: uses wildcard
        for elt type (always derivable from element_kind). For Ctypes: raw
        pointer for custom types. *)
    type 'a buffer_storage =
      | Bigarray_storage :
          ('a, _, Bigarray.c_layout) Bigarray.Array1.t
          -> 'a buffer_storage
      | Ctypes_storage : unit Ctypes.ptr -> 'a buffer_storage

    (** Typed buffer record - the 'a parameter is real, not phantom. *)
    type 'a buffer = {
      storage : 'a buffer_storage;
      kind : 'a element_kind;
      size : int;
      device : Device.t;
    }

    (** Get element size from kind *)
    let elem_size : type a. a element_kind -> int = function
      | Scalar_kind k -> Spoc_core.Vector_types.scalar_elem_size k
      | Custom_kind c -> c.elem_size

    let buffer_byte_size buf = buf.size * elem_size buf.kind

    let alloc : type a b. Device.t -> int -> (a, b) Bigarray.kind -> a buffer =
     fun device size kind ->
      let arr = Bigarray.Array1.create kind Bigarray.c_layout size in
      (* Pattern match on kind to determine element_kind - each branch has matching types *)
      match kind with
      | Bigarray.Float32 ->
          {
            storage = Bigarray_storage arr;
            kind = Scalar_kind Spoc_core.Vector_types.Float32;
            size;
            device;
          }
      | Bigarray.Float64 ->
          {
            storage = Bigarray_storage arr;
            kind = Scalar_kind Spoc_core.Vector_types.Float64;
            size;
            device;
          }
      | Bigarray.Int32 ->
          {
            storage = Bigarray_storage arr;
            kind = Scalar_kind Spoc_core.Vector_types.Int32;
            size;
            device;
          }
      | Bigarray.Int64 ->
          {
            storage = Bigarray_storage arr;
            kind = Scalar_kind Spoc_core.Vector_types.Int64;
            size;
            device;
          }
      | Bigarray.Char ->
          {
            storage = Bigarray_storage arr;
            kind = Scalar_kind Spoc_core.Vector_types.Char;
            size;
            device;
          }
      | Bigarray.Complex32 ->
          {
            storage = Bigarray_storage arr;
            kind = Scalar_kind Spoc_core.Vector_types.Complex32;
            size;
            device;
          }
      | _ ->
          Native_error.(
            raise_error (unsupported_construct "bigarray kind" "unknown kind"))

    (** Allocate buffer for custom types with explicit element size in bytes.
        For native, we allocate raw ctypes memory. *)
    let alloc_custom : type a. Device.t -> size:int -> elem_size:int -> a buffer
        =
     fun device ~size ~elem_size ->
      let bytes = size * elem_size in
      let ptr = Ctypes.allocate_n Ctypes.char ~count:bytes in
      let unit_ptr = Ctypes.to_voidp ptr in
      (* Convert to unit ptr *)
      (* Create a dummy custom_type for the kind - we don't have full type info *)
      let custom =
        {
          Spoc_core.Vector_types.elem_size;
          type_id = Sarek_ir_types.Type_id.create ();
          vector_type_id = Sarek_ir_types.Type_id.create ();
          get =
            (fun _ _ ->
              Native_error.(
                raise_error (feature_not_supported "custom type get accessor")));
          set =
            (fun _ _ _ ->
              Native_error.(
                raise_error (feature_not_supported "custom type set accessor")));
          name = "custom";
        }
      in
      {
        storage = Ctypes_storage unit_ptr;
        kind = Custom_kind custom;
        size;
        device;
      }

    (** For Native CPU, zero-copy is the default - we just wrap the bigarray *)
    let alloc_zero_copy : type a b.
        Device.t ->
        (a, b, Bigarray.c_layout) Bigarray.Array1.t ->
        (a, b) Bigarray.kind ->
        a buffer option =
     fun device ba kind ->
      let size = Bigarray.Array1.dim ba in
      match kind with
      | Bigarray.Float32 ->
          Some
            {
              storage = Bigarray_storage ba;
              kind = Scalar_kind Spoc_core.Vector_types.Float32;
              size;
              device;
            }
      | Bigarray.Float64 ->
          Some
            {
              storage = Bigarray_storage ba;
              kind = Scalar_kind Spoc_core.Vector_types.Float64;
              size;
              device;
            }
      | Bigarray.Int32 ->
          Some
            {
              storage = Bigarray_storage ba;
              kind = Scalar_kind Spoc_core.Vector_types.Int32;
              size;
              device;
            }
      | Bigarray.Int64 ->
          Some
            {
              storage = Bigarray_storage ba;
              kind = Scalar_kind Spoc_core.Vector_types.Int64;
              size;
              device;
            }
      | Bigarray.Char ->
          Some
            {
              storage = Bigarray_storage ba;
              kind = Scalar_kind Spoc_core.Vector_types.Char;
              size;
              device;
            }
      | Bigarray.Complex32 ->
          Some
            {
              storage = Bigarray_storage ba;
              kind = Scalar_kind Spoc_core.Vector_types.Complex32;
              size;
              device;
            }
      | _ ->
          Native_error.(
            raise_error (unsupported_construct "bigarray kind" "unknown kind"))

    let is_zero_copy _buf = true (* Native is always zero-copy *)

    let free _buf = ()

    let host_to_device : type a b.
        src:(a, b, Bigarray.c_layout) Bigarray.Array1.t -> dst:a buffer -> unit
        =
     fun ~src ~dst ->
	      match dst.storage with
	      | Bigarray_storage dst_arr ->
	          let len = min (Bigarray.Array1.dim src) dst.size in
	          for i = 0 to len - 1 do
	            Bigarray.Array1.set dst_arr i (Bigarray.Array1.get src i)
	          done
      | Ctypes_storage _ ->
          invalid_arg
            "host_to_device: destination is ctypes buffer (use \
             host_ptr_to_device)"

    let device_to_host : type a b.
        src:a buffer -> dst:(a, b, Bigarray.c_layout) Bigarray.Array1.t -> unit
        =
     fun ~src ~dst ->
	      match src.storage with
	      | Bigarray_storage src_arr ->
	          let len = min src.size (Bigarray.Array1.dim dst) in
	          for i = 0 to len - 1 do
	            Bigarray.Array1.set dst i (Bigarray.Array1.get src_arr i)
	          done
      | Ctypes_storage _ ->
          invalid_arg
            "device_to_host: source is ctypes buffer (use device_to_host_ptr)"

    (** Transfer from raw pointer to native buffer (for custom types). For
        native, this is a memcpy using ctypes. *)
    let host_ptr_to_device ~src_ptr ~byte_size ~dst =
      let open Ctypes in
      let byte_size = min byte_size (buffer_byte_size dst) in
      match dst.storage with
      | Ctypes_storage dst_ptr ->
          let dst_char_ptr = from_voidp char dst_ptr in
          let src_char_ptr = from_voidp char src_ptr in
          for i = 0 to byte_size - 1 do
            dst_char_ptr +@ i <-@ !@(src_char_ptr +@ i)
          done
      | Bigarray_storage dst_arr ->
          let dst_ptr = bigarray_start array1 dst_arr |> to_voidp in
          let dst_char_ptr = from_voidp char dst_ptr in
          let src_char_ptr = from_voidp char src_ptr in
          for i = 0 to byte_size - 1 do
            dst_char_ptr +@ i <-@ !@(src_char_ptr +@ i)
          done

    (** Transfer from native buffer to raw pointer (for custom types). *)
    let device_to_host_ptr ~src ~dst_ptr ~byte_size =
      let open Ctypes in
      let byte_size = min byte_size (buffer_byte_size src) in
      match src.storage with
      | Ctypes_storage src_ptr ->
          let src_char_ptr = from_voidp char src_ptr in
          let dst_char_ptr = from_voidp char dst_ptr in
          for i = 0 to byte_size - 1 do
            dst_char_ptr +@ i <-@ !@(src_char_ptr +@ i)
          done
      | Bigarray_storage src_arr ->
          let src_ptr = bigarray_start array1 src_arr |> to_voidp in
          let src_char_ptr = from_voidp char src_ptr in
          let dst_char_ptr = from_voidp char dst_ptr in
          for i = 0 to byte_size - 1 do
            dst_char_ptr +@ i <-@ !@(src_char_ptr +@ i)
          done

    let device_to_device : type a. src:a buffer -> dst:a buffer -> unit =
     fun ~src ~dst ->
	      match (src.storage, dst.storage) with
	      | Bigarray_storage src_arr, Bigarray_storage dst_arr ->
	          let len = min src.size dst.size in
	          for i = 0 to len - 1 do
	            Bigarray.Array1.set dst_arr i (Bigarray.Array1.get src_arr i)
	          done
      | Ctypes_storage src_ptr, Ctypes_storage dst_ptr ->
          let src_char_ptr = Ctypes.from_voidp Ctypes.char src_ptr in
          let dst_char_ptr = Ctypes.from_voidp Ctypes.char dst_ptr in
          let bytes =
            min (src.size * elem_size src.kind) (dst.size * elem_size dst.kind)
          in
          for i = 0 to bytes - 1 do
            Ctypes.(dst_char_ptr +@ i <-@ !@(src_char_ptr +@ i))
          done
      | _ -> invalid_arg "device_to_device: storage type mismatch"

    let size : type a. a buffer -> int = fun buf -> buf.size

    let device_ptr : type a. a buffer -> nativeint =
     fun buf ->
      match buf.storage with
      | Bigarray_storage arr ->
          (* Bigarray storage - get pointer from bigarray *)
          let ptr = Ctypes.bigarray_start Ctypes.array1 arr in
          Ctypes.to_voidp ptr |> Ctypes.raw_address_of_ptr
      | Ctypes_storage ptr ->
          (* Ctypes storage - data is already a pointer *)
          Ctypes.raw_address_of_ptr ptr
  end

  module Stream = struct
    type t = unit

    let create _dev = ()

    let destroy _s = ()

    let synchronize _s = ()

    let default _dev = ()
  end

  module Event = struct
    type t = {mutable time : float}

    let create () = {time = 0.0}

    let destroy _e = ()

    let record e _stream = e.time <- Unix.gettimeofday ()

    let synchronize _e = ()

    let elapsed ~start ~stop = (stop.time -. start.time) *. 1000.0
  end

  module Kernel = struct
    (* Native kernels are pre-compiled OCaml functions registered by the PPX.
       The "compile" here is a no-op - we look up registered functions. *)
    type t = {name : string}

    (** Use exec_arg directly - no intermediate type needed! *)
    type args = {mutable list : Framework_sig.exec_arg list}

    let compile _device ~name ~source:_ = {name}

    let compile_cached = compile

    let create_args () = {list = []}

    let set_arg_buffer : type a. args -> int -> a Memory.buffer -> unit =
     fun args _idx buf ->
      (match (buf.Memory.kind, buf.Memory.storage) with
      | Memory.Scalar_kind Spoc_core.Vector_types.Int32, Bigarray_storage _
      | Memory.Scalar_kind Spoc_core.Vector_types.Int64, Bigarray_storage _
      | Memory.Scalar_kind Spoc_core.Vector_types.Float32, Bigarray_storage _
      | Memory.Scalar_kind Spoc_core.Vector_types.Float64, Bigarray_storage _
      | Memory.Custom_kind _, Ctypes_storage _ ->
          ()
      | _ ->
          Native_error.(
            raise_error
              (feature_not_supported
                 "unsupported buffer kind/storage for argument"))) ;
      (* Wrap buffer in EXEC_VECTOR for exec_arg *)
      let module EV : Typed_value.EXEC_VECTOR
        with type elt = a
         and type underlying = a Memory.buffer = struct
        type elt = a

        type underlying = a Memory.buffer

        let length = Memory.size buf

        let type_name = "buffer"

        let elem_size = Memory.elem_size buf.Memory.kind

        let type_id =
          match buf.Memory.kind with
          | Memory.Scalar_kind k -> Spoc_core.Vector_types.scalar_type_id k
          | Memory.Custom_kind c -> c.type_id

        let underlying_type_id = Sarek_ir_types.Type_id.create ()

        let underlying = buf

        let device_ptr () = Memory.device_ptr buf

        let get i =
          match (buf.Memory.kind, buf.Memory.storage) with
          | Memory.Scalar_kind Spoc_core.Vector_types.Int32, Bigarray_storage ba ->
              Typed_value.TV_Scalar
                (Typed_value.SV ((module Typed_value.Int32_type), Bigarray.Array1.get ba i))
          | Memory.Scalar_kind Spoc_core.Vector_types.Int64, Bigarray_storage ba ->
              Typed_value.TV_Scalar
                (Typed_value.SV ((module Typed_value.Int64_type), Bigarray.Array1.get ba i))
          | Memory.Scalar_kind Spoc_core.Vector_types.Float32, Bigarray_storage ba ->
              Typed_value.TV_Scalar
                (Typed_value.SV ((module Typed_value.Float32_type), Bigarray.Array1.get ba i))
          | Memory.Scalar_kind Spoc_core.Vector_types.Float64, Bigarray_storage ba ->
              Typed_value.TV_Scalar
                (Typed_value.SV ((module Typed_value.Float64_type), Bigarray.Array1.get ba i))
          | _ ->
              Native_error.(
                raise_error (feature_not_supported "buffer element get accessor"))

        let set i tv =
          match (tv, buf.Memory.kind, buf.Memory.storage) with
          | ( Typed_value.TV_Scalar (Typed_value.SV ((module S), x)),
              Memory.Scalar_kind Spoc_core.Vector_types.Int32,
              Bigarray_storage ba ) -> (
              match S.to_primitive x with
              | Typed_value.PInt32 n -> Bigarray.Array1.set ba i n
              | _ ->
                  Native_error.(
                    raise_error (feature_not_supported "int32 buffer set conversion")))
          | ( Typed_value.TV_Scalar (Typed_value.SV ((module S), x)),
              Memory.Scalar_kind Spoc_core.Vector_types.Int64,
              Bigarray_storage ba ) -> (
              match S.to_primitive x with
              | Typed_value.PInt64 n -> Bigarray.Array1.set ba i n
              | _ ->
                  Native_error.(
                    raise_error (feature_not_supported "int64 buffer set conversion")))
          | ( Typed_value.TV_Scalar (Typed_value.SV ((module S), x)),
              Memory.Scalar_kind Spoc_core.Vector_types.Float32,
              Bigarray_storage ba ) -> (
              match S.to_primitive x with
              | Typed_value.PFloat f -> Bigarray.Array1.set ba i f
              | _ ->
                  Native_error.(
                    raise_error (feature_not_supported "float32 buffer set conversion")))
          | ( Typed_value.TV_Scalar (Typed_value.SV ((module S), x)),
              Memory.Scalar_kind Spoc_core.Vector_types.Float64,
              Bigarray_storage ba ) -> (
              match S.to_primitive x with
              | Typed_value.PFloat f -> Bigarray.Array1.set ba i f
              | _ ->
                  Native_error.(
                    raise_error (feature_not_supported "float64 buffer set conversion")))
          | _ ->
              Native_error.(
                raise_error (feature_not_supported "buffer element set accessor"))

        let get_typed _i =
          match (buf.Memory.kind, buf.Memory.storage) with
          | Memory.Scalar_kind Spoc_core.Vector_types.Int32, Bigarray_storage ba
            ->
              Bigarray.Array1.get ba _i
          | Memory.Scalar_kind Spoc_core.Vector_types.Int64, Bigarray_storage ba
            ->
              Bigarray.Array1.get ba _i
          | ( Memory.Scalar_kind Spoc_core.Vector_types.Float32,
              Bigarray_storage ba ) ->
              Bigarray.Array1.get ba _i
          | ( Memory.Scalar_kind Spoc_core.Vector_types.Float64,
              Bigarray_storage ba ) ->
              Bigarray.Array1.get ba _i
          | Memory.Custom_kind c, Ctypes_storage ptr -> c.get ptr _i
          | _ ->
              Native_error.(
                raise_error (feature_not_supported "buffer typed get accessor"))

        let set_typed _i _v =
          match (buf.Memory.kind, buf.Memory.storage) with
          | Memory.Scalar_kind Spoc_core.Vector_types.Int32, Bigarray_storage ba
            ->
              Bigarray.Array1.set ba _i _v
          | Memory.Scalar_kind Spoc_core.Vector_types.Int64, Bigarray_storage ba
            ->
              Bigarray.Array1.set ba _i _v
          | ( Memory.Scalar_kind Spoc_core.Vector_types.Float32,
              Bigarray_storage ba ) ->
              Bigarray.Array1.set ba _i _v
          | ( Memory.Scalar_kind Spoc_core.Vector_types.Float64,
              Bigarray_storage ba ) ->
              Bigarray.Array1.set ba _i _v
          | Memory.Custom_kind c, Ctypes_storage ptr -> c.set ptr _i _v
          | _ ->
              Native_error.(
                raise_error (feature_not_supported "buffer typed set accessor"))
      end in
      args.list <- Framework_sig.EA_Vec (module EV) :: args.list

    let set_arg_int32 args _idx v =
      args.list <- Framework_sig.EA_Int32 v :: args.list

    let set_arg_int64 args _idx v =
      args.list <- Framework_sig.EA_Int64 v :: args.list

    let set_arg_float32 args _idx v =
      args.list <- Framework_sig.EA_Float32 v :: args.list

    let set_arg_float64 args _idx v =
      args.list <- Framework_sig.EA_Float64 v :: args.list

    let set_arg_ptr _args _idx _ptr =
      Native_error.(raise_error (feature_not_supported "raw pointer arguments"))

    (** Set a raw OCaml value argument (for SPOC Vector/customarray). Note: Not
        yet implemented with exec_arg. *)
    let[@warning "-32"] set_arg_raw _args _idx _v =
      Native_error.(raise_error (feature_not_supported "set_arg_raw"))

    let launch kernel ~args ~(grid : Framework_sig.dims)
        ~(block : Framework_sig.dims) ~shared_mem:_ ~stream:_ =
      match Hashtbl.find_opt native_kernels kernel.name with
      | Some fn ->
          (* Convert args list to array for native function call *)
          let arg_array = args.list |> List.rev |> Array.of_list in
          fn arg_array (grid.x, grid.y, grid.z) (block.x, block.y, block.z)
      | None ->
          Native_error.(
            raise_error
              (compilation_failed
                 kernel.name
                 (Printf.sprintf "kernel '%s' not registered" kernel.name)))

    let clear_cache () = Hashtbl.clear native_kernels
  end

  let profiling_enabled = ref false

  let enable_profiling () = profiling_enabled := true

  let disable_profiling () = profiling_enabled := false

  let is_available () = true
end

(* Legacy init retained for compatibility; backend registration now handled by
   Native_plugin. *)
let init () = ()

(** Register a native kernel function by name.

    This is called by PPX-generated code at module load time to register the
    pre-compiled OCaml kernel function. The function signature matches what
    Sarek_cpu_runtime expects:
    - args: Framework_sig.exec_arg array of kernel arguments
    - grid: (gx, gy, gz) grid dimensions
    - block: (bx, by, bz) block dimensions

    Example:
    {[
      let () =
        Sarek_native.Native_plugin.register_kernel
          "my_kernel"
          (fun args grid block ->
            Sarek_cpu_runtime.run_threadpool
              ~has_barriers:false
              ~block
              ~grid
              my_kernel_fn
              args)
    ]} *)
let register_kernel name fn = Hashtbl.replace native_kernels name fn

(** Check if a kernel is registered *)
let kernel_registered name = Hashtbl.mem native_kernels name

(** List all registered kernels (for debugging) *)
let list_kernels () =
  Hashtbl.fold (fun name _ acc -> name :: acc) native_kernels []

(** Run a kernel directly with typed arguments. This bypasses the Runtime API.

    @param name The registered kernel name
    @param args Array of exec_arg (typed kernel arguments)
    @param grid (gx, gy, gz) grid dimensions
    @param block (bx, by, bz) block dimensions *)
let run_kernel_raw ~name ~args ~grid ~block =
  match Hashtbl.find_opt native_kernels name with
  | Some fn -> fn args grid block
  | None ->
      Native_error.(
        raise_error
          (compilation_failed
             name
             (Printf.sprintf "kernel '%s' not registered" name)))
