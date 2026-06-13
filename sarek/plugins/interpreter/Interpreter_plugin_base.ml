(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Interpreter Plugin - CPU Interpreter Backend
 *
 * Implements the Framework_sig.BACKEND interface for CPU interpretation.
 * This is a Direct backend that interprets Sarek V2 IR without compilation.
 *
 * Key differences from Native:
 * - Native uses pre-compiled OCaml functions (from PPX)
 * - Interpreter walks the IR tree at runtime (slower, but no compilation)
 ******************************************************************************)

[@@@warning "-21-69"]

open Spoc_framework

(** Registry for interpreter kernels. Maps kernel name to IR for interpretation.
*)
let interpreter_kernels : (string, Sarek_ir_types.kernel) Hashtbl.t =
  Hashtbl.create 16

module Interpreter : sig
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

    val is_parallel : t -> bool

    val capabilities : t -> Framework_sig.capabilities

    val set_current : t -> unit

    val current : t option ref

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
      src_ptr:nativeint -> byte_size:int -> dst:'a buffer -> unit

    val device_to_host_ptr :
      src:'a buffer -> dst_ptr:nativeint -> byte_size:int -> unit

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

    val load_from_ptx : name:string -> ptx:string -> t

    val clear_cache : unit -> unit
  end

  val enable_profiling : unit -> unit

  val disable_profiling : unit -> unit

  val is_available : unit -> bool
end = struct
  let name = "Interpreter"

  let version = (1, 0, 0)

  module Device = struct
    type t = {id : int; name : string; parallel : bool; num_cores : int}

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
              name = "CPU Interpreter (Sequential)";
              parallel = false;
              num_cores = 1;
            };
            {
              id = 1;
              name =
                Printf.sprintf "CPU Interpreter (Parallel, %d cores)" num_cores;
              parallel = true;
              num_cores;
            };
          |]
      end

    let count () = Array.length !devices

    let get idx =
      if idx < 0 || idx >= Array.length !devices then
        Interpreter_error.(
          raise_error (device_not_found idx (Array.length !devices)))
      else !devices.(idx)

    let id d = d.id

    let name d = d.name

    let is_parallel d = d.parallel

    let capabilities d : Framework_sig.capabilities =
      {
        max_threads_per_block = 256;
        (* Reasonable limit for interpretation *)
        max_block_dims = (256, 256, 64);
        max_grid_dims = (max_int, max_int, max_int);
        shared_mem_per_block = 1024 * 1024;
        total_global_mem = Int64.of_int (16 * 1024 * 1024 * 1024);
        compute_capability = (0, 0);
        supports_fp64 = true;
        supports_atomics = true;
        warp_size = (if d.parallel then 32 else 1);
        max_registers_per_block = 0;
        clock_rate_khz = 0;
        multiprocessor_count = d.num_cores;
        is_cpu = true;
      }

    let set_current d = current := Some d

    let synchronize _d = ()
  end

  module Memory = struct
    (* Interpreter uses same memory model as Native - just wraps host memory *)

    (** Element kind - carries type information for buffers *)
    type 'a element_kind =
      | Scalar_kind :
          ('a, 'b) Spoc_core.Vector_types.scalar_kind
          -> 'a element_kind
      | Custom_kind : 'a Spoc_core.Vector_types.custom_type -> 'a element_kind

    (** Buffer storage - GADT with typed parameter *)
    type 'a buffer_storage =
      | Bigarray_storage :
          ('a, _, Bigarray.c_layout) Bigarray.Array1.t
          -> 'a buffer_storage
      | Ctypes_storage : unit Ctypes.ptr -> 'a buffer_storage

    (** Typed buffer record *)
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
          Interpreter_error.(
            raise_error (unsupported_construct "bigarray kind" "unknown kind"))

    let alloc_custom : type a. Device.t -> size:int -> elem_size:int -> a buffer
        =
     fun device ~size ~elem_size ->
      let bytes = size * elem_size in
      let ptr = Ctypes.allocate_n Ctypes.char ~count:bytes in
      let unit_ptr = Ctypes.to_voidp ptr in
      let custom =
        {
          Spoc_core.Vector_types.elem_size;
          type_id = Sarek_ir_types.Type_id.create ();
          vector_type_id = Sarek_ir_types.Type_id.create ();
          get =
            (fun _ _ ->
              Interpreter_error.(
                raise_error (feature_not_supported "custom type get accessor")));
          set =
            (fun _ _ _ ->
              Interpreter_error.(
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
          Interpreter_error.(
            raise_error (unsupported_construct "bigarray kind" "unknown kind"))

    let is_zero_copy : type a. a buffer -> bool =
     fun buf ->
      match buf.storage with
      | Bigarray_storage _ -> true
      | Ctypes_storage _ -> false

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
          invalid_arg "host_to_device: destination is ctypes buffer"

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
          invalid_arg "device_to_host: source is ctypes buffer"

    let host_ptr_to_device ~src_ptr ~byte_size ~dst =
      let src_ptr = Ctypes.ptr_of_raw_address src_ptr in
      let open Ctypes in
      let byte_size = min byte_size (buffer_byte_size dst) in
      match dst.storage with
      | Ctypes_storage dst_ptr ->
          let dst_char_ptr = from_voidp char dst_ptr in
          let src_char_ptr = from_voidp char src_ptr in
          for i = 0 to byte_size - 1 do
            dst_char_ptr +@ i <-@ !@(src_char_ptr +@ i)
          done
      | Bigarray_storage _ ->
          invalid_arg "host_ptr_to_device: destination is bigarray"

    let device_to_host_ptr ~src ~dst_ptr ~byte_size =
      let dst_ptr = Ctypes.ptr_of_raw_address dst_ptr in
      let open Ctypes in
      let byte_size = min byte_size (buffer_byte_size src) in
      match src.storage with
      | Ctypes_storage src_ptr ->
          let src_char_ptr = from_voidp char src_ptr in
          let dst_char_ptr = from_voidp char dst_ptr in
          for i = 0 to byte_size - 1 do
            dst_char_ptr +@ i <-@ !@(src_char_ptr +@ i)
          done
      | Bigarray_storage _ ->
          invalid_arg "device_to_host_ptr: source is bigarray"

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
          let ptr = Ctypes.bigarray_start Ctypes.array1 arr in
          Ctypes.to_voidp ptr |> Ctypes.raw_address_of_ptr
      | Ctypes_storage ptr -> Ctypes.raw_address_of_ptr ptr
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
      | Memory.Scalar_kind Spoc_core.Vector_types.Float64, Bigarray_storage _ ->
          ()
      | _ ->
          Interpreter_error.(
            raise_error
              (feature_not_supported
                 "unsupported buffer kind/storage for argument"))) ;
      (* Wrap buffer in EXEC_VECTOR for exec_arg *)
      let module EV :
        Typed_value.EXEC_VECTOR
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
          | Memory.Scalar_kind Spoc_core.Vector_types.Int32, Bigarray_storage ba
            ->
              Typed_value.TV_Scalar
                (Typed_value.SV
                   ((module Typed_value.Int32_type), Bigarray.Array1.get ba i))
          | Memory.Scalar_kind Spoc_core.Vector_types.Int64, Bigarray_storage ba
            ->
              Typed_value.TV_Scalar
                (Typed_value.SV
                   ((module Typed_value.Int64_type), Bigarray.Array1.get ba i))
          | ( Memory.Scalar_kind Spoc_core.Vector_types.Float32,
              Bigarray_storage ba ) ->
              Typed_value.TV_Scalar
                (Typed_value.SV
                   ((module Typed_value.Float32_type), Bigarray.Array1.get ba i))
          | ( Memory.Scalar_kind Spoc_core.Vector_types.Float64,
              Bigarray_storage ba ) ->
              Typed_value.TV_Scalar
                (Typed_value.SV
                   ((module Typed_value.Float64_type), Bigarray.Array1.get ba i))
          | _ ->
              Interpreter_error.(
                raise_error
                  (feature_not_supported "buffer element get accessor"))

        let set i tv =
          match (tv, buf.Memory.kind, buf.Memory.storage) with
          | ( Typed_value.TV_Scalar (Typed_value.SV ((module S), x)),
              Memory.Scalar_kind Spoc_core.Vector_types.Int32,
              Bigarray_storage ba ) -> (
              match S.to_primitive x with
              | Typed_value.PInt32 n -> Bigarray.Array1.set ba i n
              | _ ->
                  Interpreter_error.(
                    raise_error
                      (feature_not_supported "int32 buffer set conversion")))
          | ( Typed_value.TV_Scalar (Typed_value.SV ((module S), x)),
              Memory.Scalar_kind Spoc_core.Vector_types.Int64,
              Bigarray_storage ba ) -> (
              match S.to_primitive x with
              | Typed_value.PInt64 n -> Bigarray.Array1.set ba i n
              | _ ->
                  Interpreter_error.(
                    raise_error
                      (feature_not_supported "int64 buffer set conversion")))
          | ( Typed_value.TV_Scalar (Typed_value.SV ((module S), x)),
              Memory.Scalar_kind Spoc_core.Vector_types.Float32,
              Bigarray_storage ba ) -> (
              match S.to_primitive x with
              | Typed_value.PFloat f -> Bigarray.Array1.set ba i f
              | _ ->
                  Interpreter_error.(
                    raise_error
                      (feature_not_supported "float32 buffer set conversion")))
          | ( Typed_value.TV_Scalar (Typed_value.SV ((module S), x)),
              Memory.Scalar_kind Spoc_core.Vector_types.Float64,
              Bigarray_storage ba ) -> (
              match S.to_primitive x with
              | Typed_value.PFloat f -> Bigarray.Array1.set ba i f
              | _ ->
                  Interpreter_error.(
                    raise_error
                      (feature_not_supported "float64 buffer set conversion")))
          | _ ->
              Interpreter_error.(
                raise_error
                  (feature_not_supported "buffer element set accessor"))

        let get_typed _i =
          Interpreter_error.(
            raise_error (feature_not_supported "buffer typed get accessor"))

        let set_typed _i _v =
          Interpreter_error.(
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
      Interpreter_error.(
        raise_error (feature_not_supported "raw pointer arguments"))

    let launch kernel ~args ~(grid : Framework_sig.dims)
        ~(block : Framework_sig.dims) ~shared_mem:_ ~stream:_ =
      match Hashtbl.find_opt interpreter_kernels kernel.name with
      | Some ir ->
          (* Convert exec_arg list to interpreter format *)
          let arg_list = List.rev args.list in
          let param_args =
            List.mapi
              (fun i arg ->
                let name = Printf.sprintf "param%d" i in
                match arg with
                | Framework_sig.EA_Vec (module V) ->
                    let arr =
                      Sarek.Sarek_ir_interp.exec_vector_to_array (module V)
                    in
                    (name, Sarek.Sarek_ir_interp.ArgArray arr)
                | Framework_sig.EA_Int32 n ->
                    ( name,
                      Sarek.Sarek_ir_interp.ArgScalar
                        (Sarek.Sarek_ir_interp.VInt32 n) )
                | Framework_sig.EA_Int64 n ->
                    ( name,
                      Sarek.Sarek_ir_interp.ArgScalar
                        (Sarek.Sarek_ir_interp.VInt64 n) )
                | Framework_sig.EA_Float32 f ->
                    ( name,
                      Sarek.Sarek_ir_interp.ArgScalar
                        (Sarek.Sarek_ir_interp.VFloat32 f) )
                | Framework_sig.EA_Float64 f ->
                    ( name,
                      Sarek.Sarek_ir_interp.ArgScalar
                        (Sarek.Sarek_ir_interp.VFloat64 f) )
                | _ ->
                    Interpreter_error.(
                      raise_error
                        (unsupported_construct "exec_arg" "unsupported type")))
              arg_list
          in
          Sarek.Sarek_ir_interp.run_kernel
            ir
            ~block:(block.x, block.y, block.z)
            ~grid:(grid.x, grid.y, grid.z)
            param_args
      | None ->
          Interpreter_error.(
            raise_error
              (compilation_failed
                 kernel.name
                 (Printf.sprintf "kernel '%s' not registered" kernel.name)))

    let clear_cache () = Hashtbl.clear interpreter_kernels

    let load_from_ptx ~name:_ ~ptx:_ =
      failwith "PTX kernels not supported by Interpreter backend"
  end

  let profiling_enabled = ref false

  let enable_profiling () = profiling_enabled := true

  let disable_profiling () = profiling_enabled := false

  let is_available () = true
end

(* Legacy init retained for compatibility; backend registration now handled by
   Interpreter_plugin. *)
let init () = ()

(** Register an IR kernel for interpretation *)
let register_kernel name ir = Hashtbl.replace interpreter_kernels name ir

(** Check if a kernel is registered *)
let kernel_registered name = Hashtbl.mem interpreter_kernels name

(** List all registered kernels *)
let list_kernels () =
  Hashtbl.fold (fun name _ acc -> name :: acc) interpreter_kernels []
