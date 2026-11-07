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
      | Vector.Scalar Vector.Float16 ->
          (* f16 has no Typed_value module of its own: it is a storage width,
             and [Vector.get] already returns the binary16-rounded value as an
             OCaml float. Surfacing it as Float32_type is what makes "compute in
             f32" automatic on the native path. *)
          Typed_value.TV_Scalar
            (Typed_value.SV ((module Typed_value.Float32_type), Vector.get v i))
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
      | Vector.Scalar Vector.Char ->
          (* An 8-bit element reads as the int32 it promotes to (zero
             extension), which is what the IR's [TUint8] read yields. *)
          Typed_value.TV_Scalar
            (Typed_value.SV
               ( (module Typed_value.Int32_type),
                 Int32.of_int (Char.code (Vector.get v i)) ))
      | Vector.Scalar Vector.Int8 ->
          (* Sign extension: the IR's [TInt8] read. *)
          Typed_value.TV_Scalar
            (Typed_value.SV
               ((module Typed_value.Int32_type), Int32.of_int (Vector.get v i)))
      | Vector.Scalar Vector.Complex32 ->
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
          Vector.Scalar Vector.Float16 ) -> (
          (* Mirrors the [get] arm above: f16 has no Typed_value module of its
             own (it is a storage width, not a compute type), so the value
             arrives as a Float32_type float and [Vector.set] narrows it to
             binary16 on store. Without this arm, writing an f16 element through
             the framework's exec-arg interface fell into the catch-all below
             with "unknown combination" — so an f16 kernel could not run on the
             Interpreter DEVICE at all, even though test_hip_f16 passes: that
             test uses [run_interpreter_vectors], which bypasses this path. *)
          match S.to_primitive x with
          | Typed_value.PFloat f -> Vector.set v i f
          | Typed_value.PInt32 _ -> type_error "float16" "int32"
          | Typed_value.PInt64 _ -> type_error "float16" "int64"
          | Typed_value.PBool _ -> type_error "float16" "bool"
          | Typed_value.PBytes _ -> type_error "float16" "bytes")
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
      | ( Typed_value.TV_Scalar (Typed_value.SV ((module S), x)),
          Vector.Scalar Vector.Char ) -> (
          (* Mirrors the [get] arm: an int32 stored into an 8-bit element keeps
             its low byte. *)
          match S.to_primitive x with
          | Typed_value.PInt32 n ->
              Vector.set v i (Char.chr (Int32.to_int n land 0xff))
          | Typed_value.PInt64 _ -> type_error "uint8" "int64"
          | Typed_value.PFloat _ -> type_error "uint8" "float"
          | Typed_value.PBool _ -> type_error "uint8" "bool"
          | Typed_value.PBytes _ -> type_error "uint8" "bytes")
      | ( Typed_value.TV_Scalar (Typed_value.SV ((module S), x)),
          Vector.Scalar Vector.Int8 ) -> (
          match S.to_primitive x with
          | Typed_value.PInt32 n -> Vector.set v i (Int32.to_int n)
          | Typed_value.PInt64 _ -> type_error "int8" "int64"
          | Typed_value.PFloat _ -> type_error "int8" "float"
          | Typed_value.PBool _ -> type_error "int8" "bool"
          | Typed_value.PBytes _ -> type_error "int8" "bytes")
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
  let logging = Log.would_log Log.Debug Log.Execute in
  if logging then
    Log.debugf Log.Execute "get_device_buffer for dev=%d" dev.Device.id ;
  match Vector.get_buffer v dev with
  | Some buf ->
      (if logging then
         let (module B : Vector.DEVICE_BUFFER) = buf in
         Log.debugf
           Log.Execute
           "got buffer: ptr=%Ld size=%d"
           (Int64.of_nativeint B.device_ptr)
           B.size) ;
      buf
  | None ->
      Execute_error.raise_error
        (Transfer_failed
           {vector = "unknown"; reason = "Vector has no device buffer"})

(* An SoA-bound vector transfers its N LEAVES, not its packed AoS buffer
   (backlog-54). soa_to_device scatters first, so a host [set] made before the
   launch is always reflected on the device — the same coherence contract
   Soa_launch documents.

   [soa_dispatch] is the ONE predicate that decides SoA vs AoS, so the transfer
   and the argument expansion below cannot disagree about which ABI a launch
   uses. Two sites answering that question separately is how you get N leaf
   buffers transferred and a single packed pointer bound, or the reverse.

   The CUDA/PTX restriction is the documented "never wrong data" fallback: only
   that backend's emitter lowers a record param to N base pointers, so on every
   other backend the vector stays exactly what it also is — an ordinary packed
   AoS custom vector — and takes the untouched path.

   [soa_abi] is the caller's statement that the kernel about to run was emitted
   by Sarek with [~soa_params] — i.e. that its signature really is N leaf
   pointers plus one shared length. [run]'s [expand_to_run_source_args] and
   [run_vectors]'s [transfer_vectors_to_device] pass [true] explicitly (and
   [soa_param_names] hardcodes it, per its own comment); everything else
   defaults to [false], because the dangerous direction is an EXTERNAL kernel:
   [run_source] hands the backend a source string Sarek did not emit, whose
   parameter block at the vector's position is, under [run_source]'s
   [~inject_lengths] default, the packed [(ptr, len)] pair the kernel author
   wrote there (a [~inject_lengths:false] caller declares a bare pointer
   instead), not the N-pointer layout this module can produce. [run_source]
   does not expose [~soa_abi] and always calls this with the default, so that
   mismatch cannot occur through it today. The design point is what a future
   caller passing [~soa_abi:true] for an externally-authored kernel would hit:
   [soa_dispatch] only ever answers [Some] on CUDA/PTX, which binds through
   the same bare pointer array as CUDA/C and HIP, with nothing checking the
   bound arguments against the compiled signature — so the outcome is the
   same shape-dependent shift {!Backend_error.reject_soa_params} analyses in
   detail, where a trap is among the possibilities but so is silent
   corruption (a record's Nth leaf landing in a valid pointer slot of the
   wrong buffer) or, with the vector declared last, no shift into any
   FOLLOWING pointer slot at all — though the vector's own (ptr, len) slots
   still receive leaf pointers — never uniformly "a crash" or uniformly
   "silently wrong data." A default of [true] would make that hazard the
   behaviour any new caller inherits by omission. *)
let soa_dispatch (type a b) ~(soa_abi : bool) (v : (a, b) Vector.t)
    (dev : Device.t) : Vector.soa_binding option =
  match v.Vector.soa with
  | Some b when soa_abi && dev.Device.framework = "CUDA/PTX" -> Some b
  | Some _ | None -> None

(* Kernel param names at the positions holding an SoA-dispatched vector.
   Positional, because that is the only correspondence there is: [vector_arg] is
   existential, so nothing relates an argument to a parameter but its index.

   [~soa_abi:true] is fixed here rather than a parameter: this function's whole
   purpose is to compute the [~soa_params] argument to [generate_source], so it
   is only ever reached on the generated-JIT path, and it is what MAKES that
   kernel's signature the N-pointer one. *)
let soa_param_names (ir : Sarek_ir_types.kernel) (args : vector_arg list)
    (dev : Device.t) : string list =
  let param_names =
    List.filter_map
      (function
        | Sarek_ir_types.DParam (v, _) -> Some v.Sarek_ir_types.var_name
        | _ -> None)
      ir.Sarek_ir_types.kern_params
  in
  List.filteri
    (fun i _ ->
      match List.nth_opt args i with
      | Some (Vec v) -> soa_dispatch ~soa_abi:true v dev <> None
      | _ -> false)
    param_names

(** Transfer all V2 Vector args to device *)
let transfer_vectors_to_device ?(soa_abi = false) ?(settle = true)
    (args : vector_arg list) (dev : Device.t) : unit =
  List.iter
    (function
      | Vec v -> (
          (* A default-stream launch cannot rely on the legacy-stream hidden
             synchronisation for work on a non-blocking stream; a launch on an
             explicit stream orders itself through events instead. *)
          if settle then Transfer.settle_vector v ;
          match soa_dispatch ~soa_abi v dev with
          | Some b -> b.Vector.soa_to_device dev
          | None ->
              (* This launch takes the PACKED AoS ABI. If an EARLIER launch on
                 this vector took the SoA one, its results are still sitting in
                 the leaves and [soa_leaves_live] still says so — and nothing
                 else ever clears it, so read-back would keep following leaves
                 for the rest of the vector's life and silently discard whatever
                 the packed kernel writes. The reachable sequence is a
                 transparent CUDA/PTX launch followed by [run_source] on the same
                 vector.

                 The flag is therefore normalised on BOTH outcomes of the one
                 [soa_dispatch] predicate, so it states the ABI of the MOST
                 RECENT launch rather than of some launch.

                 Clearing it is not enough on its own, and the two extra lines
                 are not bookkeeping: after an SoA launch the AUTHORITATIVE copy
                 is the leaves, so the gather has to happen before this launch
                 can upload anything meaningful, and the location then has to be
                 one [to_device] does not short-circuit. It records [Stale_CPU d]
                 on this very device, which [to_device] skips ("skip
                 (Stale_CPU)") — leaving the packed launch to run against a
                 buffer holding pre-SoA-launch bytes, or against no buffer at all
                 (the transparent path never allocates one, so the argument
                 expansion then raises [Transfer_failed]). *)
              (* Gather where the LEAVES are the newer copy, skip where the HOST
                 is, clear the flag either way. Two separate reasons, and the
                 asymmetry between them is deliberate:

                 - the SKIP arms are a correctness requirement, and getting this
                   wrong is how this round's first attempt lost data. [CPU] and
                   [Stale_GPU] assert the host copy is the newer one, and on a
                   leaf-live vector [Stale_GPU] is reachable through the public
                   API: a transparent launch leaves [Stale_CPU dev], and a host
                   write then records [Stale_GPU dev] (Vector.ml) — [Vector.set]
                   and [unsafe_set] gather the leaves first through
                   [ensure_cpu_sync], while [Vector.fill] deliberately does not
                   and moves [Stale_CPU dev] straight to [Stale_GPU dev]; the
                   location this arm reads is the same either way. An
                   unconditional gather replays the leaves OVER that write and it
                   vanishes with no diagnostic — the backlog-181/190 class on the
                   packed-after-SoA path. Pinned by
                   [check_host_write_survives_packed].

                   The case analysis is over LOCATIONS, so it inherits whatever
                   the location bookkeeping is worth: with auto-sync disabled
                   ([Vector.set_auto_sync] or [Transfer.disable_auto]) a
                   [Vector.set]/[unsafe_set] after a transparent launch leaves the
                   vector at [Stale_CPU dev] with the write pending, and the
                   gather arm below then replays the leaves over it. ([Vector.fill]
                   is unaffected: it never gathers and records [Stale_GPU dev]
                   either way, which this arm skips.) Stated here rather than
                   cited: no doc comment in [Vector.ml] describes that
                   consequence of disabling auto-sync, so there is nothing to
                   point at, and it is not something this condition can decide.
                 - deciding it HERE rather than delegating to [Transfer.to_cpu]'s
                   [needs_transfer] is not a bug fix, and no test distinguishes the
                   two. An unforced [to_cpu] also skips on [Both]. Whether that is
                   safe depends on the sites that RECORD [Both] on a leaf-live
                   vector — [to_cpu] and [sync] after a real read-back, and
                   [free_buffer] after its drain — so the host copy there already
                   holds the gathered result and the gather this arm performs is
                   redundant today. ([Transfer.to_device] also records [Both] —
                   twice: after its migration read-back and again after the
                   upload — and that path clears this flag itself before it
                   returns.) The point is that clearing the flag
                   on the next line makes the SoA arm of
                   [Transfer.read_back_to_host] unreachable until the next
                   transparent launch sets the flag again, so what may be skipped
                   is a property this site must state rather than inherit from
                   another module.

                 [Transfer.to_cpu ~force:true] is not the shape either way: it
                 would gather on [Stale_GPU] (the data-loss arm above), and the
                 jsoo build compiles this same file against [Transfer_jsx], whose
                 [to_cpu] takes no [~force]. *)
              (match v.Vector.soa with
              | Some b when !(b.Vector.soa_leaves_live) ->
                  (match v.Vector.location with
                  | Vector.GPU _ | Vector.Stale_CPU _ | Vector.Both _ ->
                      b.Vector.soa_from_device dev
                  | Vector.CPU | Vector.Stale_GPU _ -> ()) ;
                  b.Vector.soa_leaves_live := false ;
                  v.Vector.location <- Vector.Stale_GPU dev
              | Some _ | None -> ()) ;
              Transfer.to_device v dev)
      | _ -> ())
    args

(** The auto-injected vector length. Kernels index with int32, and
    [Int32.of_int] WRAPS silently above [Int32.max_int] — a 2^31-element vector
    would reach the kernel with a negative length. Refused instead (FR-018). *)
let vector_length_arg (len : int) : Framework_sig.run_source_arg =
  if len > Int32.to_int Int32.max_int then
    Execute_error.raise_error
      (Unsupported_argument
         {
           arg_type = Printf.sprintf "vector of %d elements" len;
           context =
             "kernel launch: lengths are passed as int32; split the vector (at \
              most 2147483647 elements)";
         }) ;
  Framework_sig.RSA_Vector_Length (Int32.of_int len)

(** Expand vector args to run_source_arg format.
    @param inject_lengths
      If true (default), auto-inject vector length as [RSA_Vector_Length] after
      each buffer. This matches Sarek-generated kernels which expect (ptr, len)
      pairs. Set to false for external kernels with different signatures.

    The injected length is tagged [RSA_Vector_Length], not [RSA_Int32], so
    backends can tell it apart from a genuine caller-supplied scalar that
    happens to immediately follow a buffer (which is exactly what a
    [~inject_lengths:false] caller can pass) - see
    {!Framework_sig.run_source_arg}. *)
let expand_to_run_source_args ?(inject_lengths = true) ?(soa_abi = false)
    (args : vector_arg list) (dev : Device.t) :
    Framework_sig.run_source_arg list =
  List.concat_map
    (function
      | Vec v when Option.is_some (soa_dispatch ~soa_abi v dev) ->
          (* N leaf base pointers in leaf (record declaration) order, then ONE
             shared length — exactly the param block Sarek_ir_ptx.emit_params
             produces for a ~soa_params vector. The length is emitted even under
             ~inject_lengths:false: it is not an injected convenience here but a
             declared parameter of that ABI. *)
          (* [Option.get] on a predicate the guard just evaluated: [soa_dispatch]
             is pure (it reads [v.Vector.soa] and compares [dev]'s framework
             string), so the second call cannot disagree with the first. Kept as
             two calls rather than restructured into one [match] because the
             packed arm below is shared with the non-dispatching [Vec] case and
             merging them would duplicate it. *)
          let b = Option.get (soa_dispatch ~soa_abi v dev) in
          let len = Vector.length v in
          let bufs = b.Vector.soa_leaf_bufs dev in
          (* The binding declares a leaf count ([soa_num_leaves]) and separately
             produces the buffers; this is the only place those two meet, so it is
             the only place that can compare them. A disagreement is not a wrong
             result but a malformed kernel-argument array:
             [Cuda_api.Kernel.launch] sizes it with [List.length args] and passes
             [cuLaunchKernel] a bare pointer with no count (the same mechanism
             [check_launch_args] documents at the ARITY level below), so a SHORT
             list makes the driver read past the end and dereference whatever
             follows as a leaf pointer, and a LONG one shifts every parameter
             after this vector.

             Not reachable through [Soa_vector.create_transparent] today: its
             [soa_leaf_bufs] maps over the leaf array, so it returns exactly
             [soa_num_leaves] buffers or raises [invalid_arg] for a leaf with no
             buffer on [dev] (Soa_vector.ml:148-160). The binding is a record of
             closures crossing a module boundary that exists precisely so
             [Execute] cannot see [Soa_vector] — the count agreement is therefore
             a cross-module convention, and this makes it a checked one.

             WHAT IT DOES NOT CHECK, because both sides of the comparison come
             from the VECTOR: that the vector's leaf count is the one the KERNEL
             declares. The emitter derives the parameter block from the [TRecord]
             on the kernel's own [DParam], so a transparent vector bound to a
             parameter of a DIFFERENT record type disagrees with the generated
             signature while satisfying this check. [Soa_launch.run_soa] has that
             comparison ([check_soa_layout], on the whole plan rather than the
             count); the generic transparent path does not, and did not before
             this check either — [check_launch_args] skips the element comparison
             for [Custom] elements on the grounds that both sides derive from one
             registered layout, which is the assumption the SoA ABI makes
             load-bearing. Not closed here: [Execute] cannot call [Soa_launch]
             (the dependency runs the other way) and this function does not have
             the IR. *)
          let got = List.length bufs in
          if got <> b.Vector.soa_num_leaves then
            Execute_error.raise_error
              (Transfer_failed
                 {
                   vector = "SoA-dispatched vector";
                   reason =
                     Printf.sprintf
                       "the SoA binding produced %d leaf buffer(s) on device \
                        %d but declares %d leaves; binding a list of the wrong \
                        length to the generated N-pointer parameter block \
                        hands the driver a malformed kernel-argument array"
                       got
                       dev.Device.id
                       b.Vector.soa_num_leaves;
                 }) ;
          let leaf_args =
            List.map
              (fun buf ->
                let (module B : Vector.DEVICE_BUFFER) = buf in
                Framework_sig.RSA_Buffer
                  {binder = B.bind_to_kargs; length = len})
              bufs
          in
          leaf_args @ [vector_length_arg len]
      | Vec v ->
          let buf = get_device_buffer v dev in
          let (module B : Vector.DEVICE_BUFFER) = buf in
          let len = Vector.length v in
          let buf_arg =
            Framework_sig.RSA_Buffer {binder = B.bind_to_kargs; length = len}
          in
          if inject_lengths then [buf_arg; vector_length_arg len] else [buf_arg]
      | Int n -> [Framework_sig.RSA_Int32 (Int32.of_int n)]
      | Int32 n -> [Framework_sig.RSA_Int32 n]
      | Int64 n -> [Framework_sig.RSA_Int64 n]
      | Float32 f -> [Framework_sig.RSA_Float32 f]
      | Float64 f -> [Framework_sig.RSA_Float64 f])
    args

(** {1 Execution Dispatch} *)

(** {1 Launch-time argument check}

    The [Vec] constructor of {!vector_arg} is existential
    ([Vec : ('a, 'b) Vector.t -> vector_arg]), so a vector's element type is
    ERASED the moment it enters an [~args] list. No OCaml type constraint on the
    generated kernel closure can therefore catch passing a [float32 vector]
    where the kernel declared a [float16 vector] — the mismatch happens on a
    path where the types are already gone. Executed on gfx1100: such a launch
    compiled clean and read/wrote 2N bytes of a 4N-byte buffer, producing
    [1 2 0 0] for input [1 2 3 4], with the Native path catching it only by
    accident.

    The IR is the one place where the DECLARED parameters and the SUPPLIED
    arguments meet, so the check lives here, and it covers every element type.

    ARITY IS CHECKED FIRST, and is an error rather than a precondition for the
    rest. An earlier version ran the per-argument checks only [if] the counts
    matched, which made a wrong count silently disable every other check — the
    conservatism was also a bypass. Worse, a SHORT argument list is a
    memory-safety problem in its own right and independent of f16: both
    [Cuda_api.Kernel.launch] and [Hip_api.Kernel.launch] size the
    kernel-argument array with [CArray.make (ptr void) (List.length args)] and
    hand [cuLaunchKernel] / [hipModuleLaunchKernel] a bare [CArray.start params]
    with NO count (the trailing [extra] pointer is NULL). The driver then reads
    as many entries as the COMPILED signature declares, so a short list makes it
    read past the end of that array and dereference whatever it finds as a
    parameter — for a pointer parameter, an arbitrary device address. Rejecting
    the arity here is what keeps that unreachable.

    Element types are compared exactly where the runtime kind has an IR
    counterpart. Where it does not, the check does NOT silently pass: it falls
    back to comparing PHYSICAL ELEMENT WIDTHS, which is the property that
    actually matters for buffer striding. That closes the wildcard:
    [Vector.Char] holds 1-byte elements while source [char] lowers to [TInt32],
    so a Char vector is accessed through a 4-byte [int*]. (That lowering is
    PRE-EXISTING — [Sarek_lower_ir.elttype_of_typ] mapped
    [TReg Char -> Ir.TInt32] before this branch — and is NOT fixed here; the
    check simply refuses to be the thing that hides it.)

    Still conservative where it must be: a [Custom] (record/variant) element is
    nominal and both sides derive it from the same registered layout, so its
    element comparison is skipped deliberately rather than by omission. *)
let ir_elttype_of_vector_kind : type a b.
    (a, b) Vector.kind -> Sarek_ir_types.elttype option = function
  | Vector.Scalar Vector.Float16 -> Some Sarek_ir_types.TFloat16
  | Vector.Scalar Vector.Float32 -> Some Sarek_ir_types.TFloat32
  | Vector.Scalar Vector.Float64 -> Some Sarek_ir_types.TFloat64
  | Vector.Scalar Vector.Int32 -> Some Sarek_ir_types.TInt32
  | Vector.Scalar Vector.Int64 -> Some Sarek_ir_types.TInt64
  (* The two 8-bit kinds have exact counterparts, so a [char] vector cannot be
     passed for an int8 parameter (its bytes would be read back as unsigned on
     the CPU backends and signed on the device). *)
  | Vector.Scalar Vector.Char -> Some Sarek_ir_types.TUint8
  | Vector.Scalar Vector.Int8 -> Some Sarek_ir_types.TInt8
  (* No IR constructor: falls through to the byte-width comparison, not to an
     implicit pass. *)
  | Vector.Scalar Vector.Complex32 -> None
  | Vector.Custom _ -> None

let elttype_label (t : Sarek_ir_types.elttype) : string =
  Sarek_ir_pp.string_of_elttype t

(** Byte width of an IR element type, when it is a scalar. [None] for aggregates
    (whose width comes from the registered layout, not from this table). *)
let ir_scalar_width (t : Sarek_ir_types.elttype) : int option =
  match Sarek_ir_layout.scalar_size t with
  | n -> Some n
  | exception Invalid_argument _ -> None

let is_custom_kind : type a b. (a, b) Vector.kind -> bool = function
  | Vector.Custom _ -> true
  | Vector.Scalar _ -> false

(** IR element type a SCALAR launch argument is tagged with on the host. [Int]
    and [Int32] denote the same 32-bit slot. [None] for [Vec], which is handled
    by the vector arms.

    This tag is exactly what decides how many bytes the launch writes into the
    argument slot, so it is the host side of the same width contract the vector
    check enforces — see {!check_launch_args}. *)
let ir_elttype_of_scalar_arg = function
  | Int _ | Int32 _ -> Some Sarek_ir_types.TInt32
  | Int64 _ -> Some Sarek_ir_types.TInt64
  | Float32 _ -> Some Sarek_ir_types.TFloat32
  | Float64 _ -> Some Sarek_ir_types.TFloat64
  | Vec _ -> None

let arg_label = function
  | Vec _ -> "a vector"
  | Int _ | Int32 _ -> "an int32 scalar"
  | Int64 _ -> "an int64 scalar"
  | Float32 _ -> "a float32 scalar"
  | Float64 _ -> "a float64 scalar"

(** Portable half of [pragma ["sarek.max_threads N"]]: refuse, before any
    backend sees the launch, a block whose thread count exceeds the declared
    bound. Applies to every backend, Native and Interpreter included, so the
    mistake is the same Sarek error everywhere rather than a driver failure on
    some of them. A kernel without a bound is never refused here. *)
let check_launch_bounds ~(kernel : string) (ir : Sarek_ir_types.kernel)
    ~(block : Framework_sig.dims) : unit =
  match ir.Sarek_ir_types.kern_launch_bounds with
  | None -> ()
  | Some {Sarek_ir_types.max_threads} ->
      let threads = block.Framework_sig.x * block.y * block.z in
      if threads > max_threads then
        Execute_error.raise_error
          (Invalid_dimensions
             {
               grid = "-";
               block =
                 Printf.sprintf
                   "(%d, %d, %d)"
                   block.Framework_sig.x
                   block.y
                   block.z;
               reason =
                 Printf.sprintf
                   "kernel %S declares pragma [\"sarek.max_threads %d\"] but \
                    the block has %d threads"
                   kernel
                   max_threads
                   threads;
             })

(** The warp width a kernel will run on, on [device]. A hardware device reports
    its own; Native and the interpreter simulate one, whose width is read at
    launch time ({!Sarek_warp.simulated_width}) because it is the test hook of
    the parity tests and is not frozen into the device's capabilities. *)
let warp_width_of_device (device : Device.t) : int =
  match device.Device.framework with
  | "Native" | "Interpreter" -> Sarek_warp.simulated_width ()
  | _ -> device.Device.capabilities.Framework_sig.warp_size

(* [Sarek_ir_analysis.kernel_uses_shuffle] folds the whole IR, helpers included:
   cost proportional to the kernel, paid on every launch (a downstream LLM decode token
   launches 281 to 497 kernels). The verdict is a pure function of the IR, and
   an IR is immutable, so it is recorded once in the IR's {!Ir_memo} record
   (physical identity: a different IR value, even a structurally derived one,
   is analysed afresh). Nothing here refers to the IR, so the entry dies with
   it, and the launch-cache generation is irrelevant (no registry is read). *)
let shuffle_key : bool Ir_memo.key = Ir_memo.new_key ()

let kernel_uses_shuffle_memo (ir : Sarek_ir_types.kernel) : bool =
  let facts = Ir_memo.facts ir in
  match Ir_memo.find facts shuffle_key with
  | Some verdict -> verdict
  | None ->
      let verdict = Sarek_ir_analysis.kernel_uses_shuffle ir in
      Ir_memo.set facts shuffle_key verdict ;
      verdict

(** Refuse a launch whose kernel shuffles and whose block is not a whole number
    of warps (docs/plans/warp-shuffle-intake-2026-10-17.md, decision D3). The
    lanes of a partial warp would read sources that do not exist, which is
    undefined on CUDA, HIP and Metal; the rule is the same on every backend
    (Native and the interpreter included) so the mistake is one Sarek error
    everywhere. Kernels without a shuffle are never refused here. The error is
    {!Execute_error.Invalid_dimensions}, the one [check_launch_bounds] raises
    for the sibling [sarek.max_threads] rule. *)
let check_launch_warp_width ~(kernel : string) ~(width : int)
    (ir : Sarek_ir_types.kernel) ~(block : Framework_sig.dims) : unit =
  if kernel_uses_shuffle_memo ir then begin
    let threads = block.Framework_sig.x * block.y * block.z in
    if width <= 0 || threads mod width <> 0 then
      Execute_error.raise_error
        (Invalid_dimensions
           {
             grid = "-";
             block =
               Printf.sprintf
                 "(%d, %d, %d)"
                 block.Framework_sig.x
                 block.y
                 block.z;
             reason =
               Printf.sprintf
                 "kernel %S uses a warp shuffle, so its block must be a \
                  multiple of the warp width %d; the block has %d threads"
                 kernel
                 width
                 threads;
           })
  end

let check_launch_warp ~(device : Device.t) ~(kernel : string)
    (ir : Sarek_ir_types.kernel) ~(block : Framework_sig.dims) : unit =
  check_launch_warp_width ~kernel ~width:(warp_width_of_device device) ir ~block

let check_launch_args ~(kernel : string) (ir : Sarek_ir_types.kernel)
    (args : vector_arg list) : unit =
  let params = Array.of_list ir.Sarek_ir_types.kern_params in
  let n_params = Array.length params and n_args = List.length args in
  (* 1. Arity, unconditionally and first. *)
  if n_params <> n_args then
    Execute_error.raise_error
      (Type_mismatch
         {
           expected = Printf.sprintf "%d argument(s)" n_params;
           actual = Printf.sprintf "%d" n_args;
           context =
             Printf.sprintf
               "kernel %S: the launch builds its device argument array from \
                the SUPPLIED count while the driver reads the COMPILED \
                parameter count, so a mismatch is unsafe, not merely wrong"
               kernel;
         }) ;
  (* 2. Per position: shape, then element type or physical width. *)
  List.iteri
    (fun i arg ->
      let mismatch ~expected ~actual ~why =
        let pname =
          match params.(i) with
          | Sarek_ir_types.DParam (pv, _) -> pv.Sarek_ir_types.var_name
          | _ -> "?"
        in
        Execute_error.raise_error
          (Type_mismatch
             {
               expected;
               actual;
               context =
                 Printf.sprintf
                   "kernel %S argument %d (parameter %S): %s"
                   kernel
                   i
                   pname
                   why;
             })
      in
      match (arg, params.(i)) with
      | Vec v, Sarek_ir_types.DParam (_, Some info) -> (
          let want = info.Sarek_ir_types.arr_elttype in
          match ir_elttype_of_vector_kind (Vector.kind v) with
          | Some got ->
              if got <> want then
                mismatch
                  ~expected:(elttype_label want ^ " vector")
                  ~actual:(elttype_label got ^ " vector")
                  ~why:
                    "the element types differ, so the device would read the \
                     buffer at the wrong stride or interpret its bits as the \
                     wrong type"
          | None -> (
              if
                (* No IR constructor for this kind. Compare the physical element
                 widths instead of passing silently. *)
                not (is_custom_kind (Vector.kind v))
              then
                let got_w = Vector.elem_size (Vector.kind v) in
                match ir_scalar_width want with
                | None ->
                    (* [want] has no scalar width, i.e. it is an AGGREGATE
                       (record/variant/array). We are already inside the
                       non-[Custom] branch, so the supplied vector holds plain
                       scalars: there is no layout under which a Char or
                       Complex32 buffer is a valid record/variant buffer. This
                       used to fall through to [()], which made the width
                       fallback silently inapplicable exactly where the shapes
                       are most different. *)
                    mismatch
                      ~expected:(elttype_label want ^ " vector")
                      ~actual:
                        (Printf.sprintf
                           "a scalar vector (%d-byte elements)"
                           got_w)
                      ~why:
                        "the kernel declares an aggregate element type, which \
                         a scalar vector cannot supply at any width"
                | Some want_w ->
                    if got_w <> want_w then
                      mismatch
                        ~expected:
                          (Printf.sprintf
                             "%s vector (%d-byte elements)"
                             (elttype_label want)
                             want_w)
                        ~actual:(Printf.sprintf "%d-byte elements" got_w)
                        ~why:
                          "the host element width does not match the width the \
                           kernel accesses the buffer with"))
      | Vec _, Sarek_ir_types.DParam (_, None) ->
          mismatch
            ~expected:"a scalar"
            ~actual:"a vector"
            ~why:"the kernel declares a scalar parameter here"
      | ( ((Int _ | Int32 _ | Int64 _ | Float32 _ | Float64 _) as a),
          Sarek_ir_types.DParam (_, Some _) ) ->
          mismatch
            ~expected:"a vector"
            ~actual:(arg_label a)
            ~why:"the kernel declares a vector parameter here"
      | ( ((Int _ | Int32 _ | Int64 _ | Float32 _ | Float64 _) as a),
          Sarek_ir_types.DParam (pv, None) ) -> (
          (* Scalar against scalar. This is the SAME hazard as the vector arm,
             not a lesser one: the host tag fixes how wide a slot the launch
             writes, the driver reads the COMPILED parameter's width, and a
             narrower host tag against a wider compiled parameter makes the
             driver read past the value it was given. Leaving this to the
             catch-all meant scalars had no type check at all. *)
          let want = pv.Sarek_ir_types.var_type in
          match ir_elttype_of_scalar_arg a with
          | None -> ()
          | Some got -> (
              match want with
              | Sarek_ir_types.TInt32 | Sarek_ir_types.TInt64
              | Sarek_ir_types.TFloat32 | Sarek_ir_types.TFloat64 ->
                  (* The four types the host can name exactly: compare
                     exactly, so int-vs-float confusion at equal width is
                     caught too, matching the vector arm's discipline. *)
                  if got <> want then
                    mismatch
                      ~expected:(elttype_label want ^ " scalar")
                      ~actual:(elttype_label got ^ " scalar")
                      ~why:
                        "the host tags the argument slot with a different type \
                         than the kernel declares, so the driver reads the \
                         slot at the wrong width or interprets its bits as the \
                         wrong type"
              | _ -> (
                  (* Not one of the four: [TBool]/[TUnit] share the 32-bit slot
                     with [TInt32] and are legitimately reached through [Int],
                     so an exact comparison would reject correct launches.
                     Fall back to the physical width, which is the property the
                     driver actually depends on. Aggregates have no scalar
                     width; they are left alone here rather than guessed at,
                     the same deliberate conservatism the [Custom] vector case
                     gets. *)
                  match (ir_scalar_width got, ir_scalar_width want) with
                  | Some got_w, Some want_w when got_w <> want_w ->
                      mismatch
                        ~expected:
                          (Printf.sprintf
                             "%s scalar (%d-byte slot)"
                             (elttype_label want)
                             want_w)
                        ~actual:
                          (Printf.sprintf
                             "%s scalar (%d-byte slot)"
                             (elttype_label got)
                             got_w)
                        ~why:
                          "the host writes a differently-sized argument slot \
                           than the driver reads"
                  | _ -> ())))
      | Vec _, _
      | Int _, _
      | Int32 _, _
      | Int64 _, _
      | Float32 _, _
      | Float64 _, _ ->
          (* Remaining shapes are non-[DParam] declarations (locals/shared),
             which are not launch arguments. *)
          ())
    args

(** Render a non-permitting verdict and raise it as a [Backend_error].

    Shared by {!check_device_capabilities} and {!check_interpreter_capabilities}
    so the two launch gates cannot drift into describing the same refusal
    differently — which is a real hazard here, since backlog-154 was two gates
    disagreeing about the same kernel. *)
let raise_capability_refusal ~(backend : string) ~(target : string)
    (verdict : Sarek_capability.verdict) : 'a =
  let message =
    match verdict with
    | Sarek_capability.Unavailable cap -> Sarek_capability.explain ~target cap
    | Sarek_capability.Unknown why -> Printf.sprintf "%s: %s" target why
    | Sarek_capability.Available ->
        (* [first_refusal] returns only non-permitting verdicts. *)
        assert false
  in
  Execute_error.raise_error (Backend_error {backend; message})

(** Refuse a launch whose kernel needs a wide element type the target device
    does not provide (#142).

    WHY THIS IS A LAUNCH GATE AND NOT A CODEGEN REFUSAL. These are
    {!Sarek_capability.Device_optional} capabilities:
    [kind_needs_device Device_optional = true], and
    [Framework_sig.generate_source] takes no device, so codegen is structurally
    the wrong place to ask. The device is first in scope here, next to
    {!check_launch_args}.

    WHAT IT REPLACES. Nothing — that is the defect. Before #142 an int64 kernel
    on Vulkan reached [vkCreateShaderModule] with SPIR-V declaring
    [OpCapability Int64] and no [shaderInt64] enabled on the logical device. On
    an RX 7900 XTX (RADV, Mesa 26.1.4-arch3.1) that is not a crash and not a
    wrong answer: results are correct and the violation is visible only under
    VK_LAYER_KHRONOS_validation (VUID-VkShaderModuleCreateInfo-pCode-08740).
    Silent undefined behaviour on the driver that happens to cope is exactly the
    failure mode a capability model exists to convert into a diagnostic.

    Routed through {!Sarek_capability.permits} rather than a membership test so
    an {!Sarek_capability.Unknown} verdict refuses instead of falling through to
    permitted. *)
let check_device_capabilities ~(device : Device.t) (ir : Sarek_ir_types.kernel)
    : unit =
  let provided =
    Some device.Device.capabilities.Framework_sig.device_features
  in
  (* Deliberately NOT [all_features]. [Float16] is governed by the existing
     codegen-time refusals (Toolchain_semantic + Policy — the measured ACO
     narrowing), and nothing probes shaderFloat16 or CUDA sm_53 yet, so no
     backend can honestly claim f16 in [device_features]. Gating on it here
     would refuse every working f16 kernel on Native, Interpreter, CUDA and HIP
     on the strength of a probe that was never written — a refusal with no
     evidence behind it, which is the mirror image of the #142 defect rather
     than a fix for it. Widening this list is the work that must come WITH the
     f16 probe, not before it. *)
  let gated =
    [
      Sarek_ir_analysis.Float64;
      Sarek_ir_analysis.Int64;
      (* 8-bit storage (uint8/int8 arrays): every backend but Vulkan provides
         it by construction; a Vulkan device must have enabled shaderInt8 and
         storageBuffer8BitAccess. *)
      Sarek_ir_analysis.Int8;
      (* Warp shuffles, warp_size and lane_id: a device lists the feature in
         [device_features] only when it probed (or, for CUDA, HIP, Metal,
         Native and the interpreter, by construction) that it can run them. *)
      Sarek_ir_analysis.Warp_shuffle;
    ]
  in
  let required =
    List.filter (fun f -> Sarek_ir_analysis.kernel_uses f ir) gated
  in
  (* backlog-62 slice 3. Cooperative matrices are gated HERE and not in
     [gated] above, because they are not a [device_features] flag: they are
     decided per CONFIGURATION. The same RX 7900 XTX that permits
     [u8 x u8 + s32 -> s32] refuses [f16 x f16 -> f16 saturating], so a boolean
     "this device has coopmat" would be [Available] for a request the device
     cannot run — which is the #142 defect in a new place.

     WHY IT IS WIDENED NOW AND WAS NOT AT SLICE 2. Gating at slice 2 would have
     been a refusal with nothing to refuse: no kernel could reach a
     cooperative-matrix instruction, so the branch was unreachable and an
     unreachable gate is not a gate. After this slice a kernel CAN require one.

     [capabilities.coopmat = None] means the backend never probed, which
     {!Sarek_coopmat.verdict} turns into [Unknown] and
     {!Sarek_capability.permits} refuses. That is the safe direction and it is
     load-bearing: OpenCL, CUDA/PTX and the CPU backends leave it
     [None] (Vulkan, HIP, Metal and CUDA/C probe), so a coopmat kernel aimed at one
     of them is refused here by the capability model before any source is
     compiled — and the diagnostic names the device rather than the emitter.
     CUDA/C (the hidden, explicitly registered NVRTC backend) is admitted on
     sm_70+ (float16) / sm_72+ (8-bit) with compiler-output evidence only;
     ZLUDA devices are refused (no wmma). *)
  (* The device verdict first, then the accuracy policy (§6.1 of
     docs/design/f16-relaxed-accuracy.md): a device that cannot run the
     configuration is the more useful refusal, and a float configuration the
     device CAN run still needs the program's explicit opt-in. *)
  let coopmat_verdicts =
    List.concat_map
      (fun cfg ->
        [
          Sarek_coopmat.verdict
            ~support:device.Device.capabilities.Framework_sig.coopmat
            cfg;
          Sarek_coopmat.policy_verdict_for
            (Sarek_policy.budget ~kernels:[ir.Sarek_ir_types.kern_name])
            cfg;
        ])
      (Sarek_ir_analysis.kernel_coopmat_configs ir)
  in
  (* Every fragment too, multiply-add or not: a kernel that only loads and
     stores fragments still needs the device's layout for them, and without
     this a CUDA device (coopmat = None) or a non-gfx11 HIP device would
     compile and run such a kernel unprobed. *)
  let fragment_verdicts =
    List.map
      (Sarek_coopmat.fragment_verdict
         ~support:device.Device.capabilities.Framework_sig.coopmat)
      (List.sort_uniq
         compare
         (List.concat_map
            (function
              | Sarek_ir_types.CM_decl {frag; _}
              | Sarek_ir_types.CM_load {frag; _}
              | Sarek_ir_types.CM_store {frag; _} ->
                  [frag]
              | Sarek_ir_types.CM_muladd {cfg; _} ->
                  let _, _, _, r = Sarek_coopmat.fragments_of_config cfg in
                  [r]
              | Sarek_ir_types.CM_scale_add {dst_frag; src_frag; _} ->
                  [dst_frag; src_frag])
            (Sarek_ir_analysis.kernel_coopmat_ops ir)))
  in
  (* docs/design/coopmat-scaled-accumulate.md §3: CM_scale_add is checked on
     its own, not through a producer (a CM_decl or CM_load can create its
     source with no multiply-add in the kernel): its operands, then whether
     the device's backend lowers it at all. This runs once per (IR, device)
     through [check_device_capabilities_memo], not per launch. *)
  let scale_add_verdicts =
    List.filter_map
      (function
        | Sarek_ir_types.CM_scale_add {dst_frag; src_frag; _} ->
            Some
              (Sarek_coopmat.scale_add_verdict
                 ~support:device.Device.capabilities.Framework_sig.coopmat
                 ~dst_frag
                 ~src_frag)
        | Sarek_ir_types.CM_decl _ | Sarek_ir_types.CM_load _
        | Sarek_ir_types.CM_store _ | Sarek_ir_types.CM_muladd _ ->
            None)
      (Sarek_ir_analysis.kernel_coopmat_ops ir)
  in
  Launch_policy.coopmat_gate ~device ir ;
  match
    Sarek_capability.first_refusal
      (List.map (Sarek_capability.device_verdict ~provided) required
      @ coopmat_verdicts @ scale_add_verdicts @ fragment_verdicts)
  with
  | None -> ()
  | Some verdict ->
      raise_capability_refusal
        ~backend:device.Device.framework
        ~target:device.Device.name
        verdict

(** The launch gate for {!run_interpreter_vectors} (backlog-154).

    [run_interpreter_vectors] applied {!check_launch_args} and NOT
    {!check_device_capabilities}, so the interpreter — the cross-backend numeric
    oracle — was the one execution path with no capability gate on it. The
    asymmetry was observable: in one run of [test_coopmat_integer_e2e] on this
    workstation, [check_device_capabilities] printed
    [launch gate refuses on CPU Interpreter (Sequential)] for the same kernel
    [run_interpreter_vectors] then evaluated to 65536 bit-exact results.

    Note what could NOT be done about that: calling {!check_device_capabilities}
    here. It needs a [Device.t], this entry point has none by design, and — the
    substantive half — the interpreter's [Framework_sig.capabilities] says
    [coopmat = None], which means "not probed", which refuses. Wiring the device
    gate in would have taken the oracle offline rather than closed a hole. The
    bypass was load-bearing, which is why the fix is a capability answer for the
    interpreter and not a call to someone else's.

    {!Sarek_interp_capability} is that answer, and it lives beside the evaluator
    that justifies it. See its interface for the argument about what an
    interpreter should advertise. *)
let check_interpreter_capabilities (ir : Sarek_ir_types.kernel) : unit =
  match Sarek_interp_capability.first_refusal ir with
  | None -> ()
  | Some verdict ->
      raise_capability_refusal
        ~backend:"Interpreter"
        ~target:"CPU Interpreter"
        verdict

(** {1 Launch-preparation memo}

    Everything [run] does to a kernel before binding its arguments depends on
    the IR and not on the arguments: the device capability analysis (IR folds)
    and, on JIT backends, generating the kernel source. Measured on HIP (RX 7900
    XTX), that work was 60-70 % of the host cost of a launch — 5.5 of 18 us for
    a 10-argument kernel, ~25 of ~40 us for a 2 KB matrix-vector kernel — and it
    was repeated identically on every launch of the same kernel. This memo
    performs it once.

    KEYS. The IR by PHYSICAL identity: [Sarek_ir_types.kernel] is immutable and
    a PPX kernel hands the same [body_ir] record to every launch, so [==] is a
    sound and free equality. A structurally equal IR rebuilt per launch is
    simply a miss and runs exactly as before. Then, per IR:
    - generated source by (backend, [block], [soa_params]) — the inputs of
      [generate_source] besides the IR ([block] is read by Vulkan; [soa_params]
      is derived from the arguments and is non-empty only on CUDA/PTX);
    - a passed capability check by (device id, the device's [capabilities]
      record compared PHYSICALLY). The verdict depends on the capabilities, and
      device ids are reused ([Device.reset], hand-built devices), so the id
      alone is not enough: a device carrying another capabilities record is
      re-checked.

    WHAT IS NEVER MEMOISED. Failures: a refused capability check or a
    [generate_source] that raises or returns [None] stores nothing, so the next
    identical launch fails the same way. And everything that depends on the
    arguments — [check_launch_args], transfers, argument expansion — still runs
    on every launch.

    THE ONE THING IT DOES NOT TRACK. Codegen reads the global intrinsic and type
    registries. They are filled at module initialisation and only grow, and a
    kernel whose intrinsic is missing fails codegen (not memoised). Replacing an
    intrinsic's implementation AFTER a kernel using it has launched is not seen
    by that kernel until {!clear_launch_cache} is called, nor is backend debug
    logging (first launch only). The POLICY is tracked ({!Policy_stamp}): a
    later grant or restriction is honoured at once.

    LIFETIME AND CONCURRENCY. The entry is a slot of the IR's {!Ir_memo} record,
    which is weak in the IR, and its data never refers to the IR, so an entry
    dies with its kernel. One mutex guards reading and installing entries and
    their lists; the analysis and codegen run outside it, so two domains missing
    at once may both generate — the first installed source wins and both get it.
    An entry belongs to the generation it was made in; {!clear_launch_cache}
    bumps the generation, which makes every entry stale (replaced on the IR's
    next launch), and a result computed across a clear is returned to its caller
    but not installed. The memo is cleared by [Cache_hooks.notify_clear_all]
    (every backend [Kernel.clear_cache], [Device.reset]) and by every
    device-destroy notification. *)

(* The backend is identified by its [generate_source] closure, compared
   PHYSICALLY, not by its name: [Framework_registry.register_backend] replaces
   a backend registered under the same name, and the replacement must not be
   handed its predecessor's source. [codegen] must stay exactly
   [generate_source]'s type: a type that drops or reorders its optional
   arguments would wrap it in a fresh closure that never compares equal. *)
type source_key = {
  codegen :
    ?block:Framework_sig.dims ->
    ?soa_params:string list ->
    Sarek_ir_types.kernel ->
    string option;
  block : Framework_sig.dims;
  soa_params : string list;
}

let same_source_key a b =
  a.codegen == b.codegen && a.block = b.block && a.soa_params = b.soa_params

let find_source ~kernel key sources =
  Policy_stamp.find_current ~kernel ~matches:(same_source_key key) sources

type launch_entry = {
  generation : int;  (** the {!launch_memo_generation} it belongs to *)
  mutable sources : (source_key * (string * Policy_stamp.t)) list;
      (** source and policy stamp per {!source_key} *)
  mutable verified : (int * Framework_sig.capabilities) list;
      (** (device id, physical capabilities record) that passed the check *)
}

let launch_key : launch_entry Ir_memo.key = Ir_memo.new_key ()

let launch_memo_lock = Mutex.create ()

let launch_memo_generation = ref 0

(** Drop every memoised launch preparation: the next launch of every kernel
    re-runs its device capability check and regenerates its source. Needed only
    after replacing an intrinsic or type implementation that already-launched
    kernels use; backend cache clears and device teardown call it already. *)
let clear_launch_cache () =
  Mutex.protect launch_memo_lock (fun () -> incr launch_memo_generation)

let () =
  Cache_hooks.on_clear_all clear_launch_cache ;
  Cache_hooks.on_device_destroy (fun ~backend:_ _ -> clear_launch_cache ())

(* Under [launch_memo_lock]: the entry of the current generation, if any. *)
let current_entry facts =
  match Ir_memo.find facts launch_key with
  | Some e when e.generation = !launch_memo_generation -> Some e
  | Some _ | None -> None

(* Under [launch_memo_lock]. *)
let memo_entry facts =
  match current_entry facts with
  | Some e -> e
  | None ->
      let e =
        {generation = !launch_memo_generation; sources = []; verified = []}
      in
      Ir_memo.set facts launch_key e ;
      e

(** [check_device_capabilities], performed once per (IR, device id, physical
    capabilities record). A refusal raises every time. *)
let check_device_capabilities_memo ~(device : Device.t)
    (ir : Sarek_ir_types.kernel) : unit =
  let caps = device.Device.capabilities in
  let id = device.Device.id in
  let passed (i, c) = i = id && c == caps in
  let facts = Ir_memo.facts ir in
  let known, generation =
    Mutex.protect launch_memo_lock (fun () ->
        let hit =
          match current_entry facts with
          | Some e -> List.exists passed e.verified
          | None -> false
        in
        (hit, !launch_memo_generation))
  in
  if not known then begin
    check_device_capabilities ~device ir ;
    Mutex.protect launch_memo_lock (fun () ->
        if !launch_memo_generation = generation then
          let e = memo_entry facts in
          if not (List.exists passed e.verified) then
            e.verified <- (id, caps) :: e.verified)
  end

(** [B.generate_source], performed once per (IR, backend, block, soa_params).
    Every hit returns the same physical string, which also lets the backend's
    {!Compile_cache.make_key} memo skip its digests. [None] and exceptions are
    not memoised. *)
let generate_source_memo (module B : Framework_sig.BACKEND) ~block ~soa_params
    (ir : Sarek_ir_types.kernel) : string option =
  let key = {codegen = B.generate_source; block; soa_params} in
  let kernel = ir.Sarek_ir_types.kern_name in
  let facts = Ir_memo.facts ir in
  let cached, generation =
    Mutex.protect launch_memo_lock (fun () ->
        let hit =
          match current_entry facts with
          | Some e -> find_source ~kernel key e.sources
          | None -> None
        in
        (hit, !launch_memo_generation))
  in
  match cached with
  | Some _ -> cached
  | None -> (
      let stamp = Policy_stamp.make ~kernel in
      match B.generate_source ~block ~soa_params ir with
      | None -> None
      | Some source ->
          Mutex.protect launch_memo_lock (fun () ->
              if !launch_memo_generation <> generation then Some source
              else
                let e = memo_entry facts in
                match find_source ~kernel key e.sources with
                | Some first -> Some first
                | None ->
                    e.sources <-
                      Policy_stamp.install
                        ~matches:(same_source_key key)
                        e.sources
                        key
                        source
                        stamp ;
                    Some source))

(** Validate a launch on [stream] for [device] BEFORE anything is transferred:
    the stream must belong to [device], and a JIT backend must implement stream
    launches — a launch the caller ordered on a stream must never run on the
    default queue instead. Direct/Custom backends (Native, Interpreter) execute
    synchronously, which satisfies any stream ordering. *)
let check_stream_launch ~(device : Device.t) (stream : Transfer.stream option) :
    unit =
  match stream with
  | None -> ()
  | Some s -> (
      let sdev, _ = Transfer.stream_handle s in
      if sdev.Device.id <> device.Device.id then
        invalid_arg
          (Printf.sprintf
             "Execute: the stream belongs to device %d (%s), the launch \
              targets device %d (%s)"
             sdev.Device.id
             sdev.Device.name
             device.Device.id
             device.Device.name) ;
      match Framework_registry.find_backend device.framework with
      | None -> ()
      | Some (module B : Framework_sig.BACKEND) -> (
          match B.execution_model with
          | Framework_sig.JIT ->
              if not B.Async.support.Framework_sig.stream_launch then
                Framework_sig.refuse_stream_launch ~backend:B.name
          | Framework_sig.Direct | Framework_sig.Custom -> ()))

(** {1 Launch window}

    JIT launches with no [~stream] on a backend whose window is enabled are
    deferred in {!window} and issued at the next synchronisation point, possibly
    fused with independent launches ({!Launch_window}, specs/launch-window.md).
    Argument checks, the capability check and source generation still happen in
    the launching call; device compilation and the driver launch happen when the
    launch is issued, so an error there is raised by the call that flushed the
    window. *)

let window : vector_arg Launch_window.t =
  Launch_window.create
    ~int32_arg:(fun n -> Int32 n)
    ~arg_buf:(function Vec v -> Some v.Vector.id | _ -> None)
    ~arg_int32:(function
      | Int n -> Some (Int32.of_int n)
      | Int32 n -> Some n
      | Vec _ | Int64 _ | Float32 _ | Float64 _ -> None)
    ~set_pending:(fun n -> Atomic.set Launch_hooks.pending n)
    ~guard:Launch_hooks.while_flushing

let () =
  Launch_epilogue.install window ;
  Launch_hooks.set_flusher ~touches:(Launch_window.touches window) (fun () ->
      Launch_window.flush window) ;
  (* Pending launches must reach the device before a device is destroyed or the
     backends' caches are cleared. *)
  Cache_hooks.on_clear_all Launch_hooks.flush_pending ;
  Cache_hooks.on_device_destroy (fun ~backend:_ _ ->
      Launch_hooks.flush_pending ()) ;
  at_exit (fun () ->
      try Launch_hooks.flush_pending ()
      with e ->
        (* No caller is left to raise to: report instead of losing it. *)
        prerr_endline
          ("Sarek: a deferred kernel launch failed at exit: "
         ^ Printexc.to_string e))

(** Control and statistics of the launch window. *)
module Window = struct
  type stats = Launch_window.stats = {
    submitted : int;  (** JIT launches made through [run]/[run_vectors] *)
    issued : int;  (** backend launches actually made *)
    fused_groups : int;  (** groups of several launches issued as one kernel *)
    fused_built : int;  (** distinct fused kernels built *)
    fused_refused : int;  (** launch sequences whose fusion failed *)
  }

  (** Issue every deferred launch now. Synchronisation points do this
      themselves; this is for callers that want the device to start early. *)
  let flush () = Launch_hooks.flush_pending ()

  let stats () = Launch_window.stats window

  let reset_stats () = Launch_window.reset_stats window

  (** [true] when JIT launches on this backend name are deferred. *)
  let defers backend = Launch_window.defers backend

  (** [true] when JIT launches on this device are deferred: its backend's window
      is on and it is not a CPU device (whose host-pointer buffers alias host
      memory, so they can alias each other). *)
  let defers_device (device : Device.t) =
    Window_gate.defers_device
      ~framework:device.framework
      ~is_cpu:device.capabilities.Framework_sig.is_cpu

  (** [false] when deferral is off for every backend. *)
  let enabled () = Atomic.get Launch_window.enabled

  (** Turn deferral on or off for every backend (flushes first).
      [SAREK_OPT_DISABLE=window] in the environment turns it off at start-up. *)
  let set_enabled b = Launch_window.set_enabled window b

  (** Turn deferral on or off for one backend (flushes first). On by default for
      HIP, OpenCL, Vulkan, CUDA and Metal. *)
  let set_backend_enabled ~backend b =
    Launch_window.set_backend_enabled window ~backend b

  (** [true] when epilogue fusion may run on this backend under the process
      grants (HIP, Metal; the others with [Cross_kernel_contraction]). *)
  let epilogue_on backend = Launch_epilogue.available backend

  (** Switch epilogue fusion off or back on for one backend (flushes first);
      never a grant. *)
  let set_epilogue_backend ~backend b =
    Launch_window.flush window ;
    Launch_epilogue.set_backend_enabled ~backend b

  (** Forget every fused kernel built so far. *)
  let clear_fused_cache () =
    Launch_window.clear_cache window ~also:Launch_epilogue.clear
end

(* Prepare the hand-over of a generated kernel to the backend: everything that
   can fail on the arguments (buffer lookup, length range, the backend's
   source language) happens NOW, in the launching call, and the returned
   closure only selects the device and calls the backend. A deferred launch
   therefore reports argument errors where it is made. *)
let prepare_jit (module B : Framework_sig.BACKEND) ?stream ~(device : Device.t)
    ~name ~source ~block ~grid ~shared_mem (args : vector_arg list) :
    unit -> unit =
  (* ~soa_abi:true: this is the generated-JIT path, and [source] was emitted
     with the matching ~soa_params. *)
  let rs_args = expand_to_run_source_args ~soa_abi:true args device in
  let dev = B.Device.get device.backend_id in
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
  fun () ->
    B.Device.set_current dev ;
    match stream with
    | None ->
        B.run_source
          ~source
          ~lang
          ~kernel_name:name
          ~block
          ~grid
          ~shared_mem
          rs_args
    | Some s ->
        let _, sref = Transfer.stream_handle s in
        let bs =
          match B.unwrap_stream sref with
          | Some bs -> bs
          | None ->
              invalid_arg "Execute.run: the stream belongs to another backend"
        in
        B.run_source_on_stream
          ~stream:bs
          ~source
          ~lang
          ~kernel_name:name
          ~block
          ~grid
          ~shared_mem
          rs_args ;
        (* The kernel may still be running: keep the vectors whose buffers it
         touches alive until the stream (or a later event) completes — the GC
         finaliser would otherwise free them under it. *)
        Transfer.retain_on_stream
          s
          ~keep:(Transfer.Keep args)
          ~vec_ids:
            (List.filter_map
               (function Vec v -> Some v.Vector.id | _ -> None)
               args)

(* Hand a generated kernel to the backend, now. *)
let issue_jit (module B : Framework_sig.BACKEND) ?stream ~device ~name ~source
    ~block ~grid ~shared_mem args : unit =
  prepare_jit
    (module B)
    ?stream
    ~device
    ~name
    ~source
    ~block
    ~grid
    ~shared_mem
    args
    ()

let jit_source (module B : Framework_sig.BACKEND) ~(device : Device.t) ~name
    ~block ~soa_params ir : string =
  match generate_source_memo (module B) ~block ~soa_params ir with
  | Some source -> source
  | None ->
      Execute_error.raise_error
        (Compilation_failed
           {
             kernel = name;
             reason =
               device.framework
               ^ ": generate_source returned None (kernel may use unsupported \
                  IR nodes)";
           })

(* The buffers the kernel touches, with how it uses each. *)
let launch_accesses (info : Launch_window.ir_info) (ir : Sarek_ir_types.kernel)
    (args : vector_arg list) : Launch_window.access list =
  List.concat
    (List.map2
       (fun decl arg ->
         match (decl, arg) with
         | Sarek_ir_types.DParam (v, Some _), Vec vec -> (
             match
               List.assoc_opt v.Sarek_ir_types.var_name info.param_accesses
             with
             | Some {Kernel_access.read; write} when read || write ->
                 [{Launch_window.buf = vec.Vector.id; read; write}]
             | _ -> [])
         | _ -> [])
       ir.Sarek_ir_types.kern_params
       args)

let submit_deferred (module B : Framework_sig.BACKEND) ~(device : Device.t) ~ir
    ~block ~grid ~shared_mem ~issue (args : vector_arg list) : unit =
  (* The fused kernel's parameters are its members' (already checked by
     their own launches) followed by int32 positions, by construction
     (Fusion_horizontal.fuse, Launch_window.fused_args). *)
  let issue_fused (fused : Sarek_ir_types.kernel) fused_args fused_grid =
    let source =
      jit_source
        (module B)
        ~device
        ~name:fused.kern_name
        ~block
        ~soa_params:[]
        fused
    in
    issue_jit
      (module B)
      ~device
      ~name:fused.kern_name
      ~source
      ~block
      ~grid:fused_grid
      ~shared_mem:0
      fused_args
  in
  let info = Launch_window.ir_info window ir in
  Launch_window.submit
    window
    {
      Launch_window.device_id = device.Device.id;
      backend = device.framework;
      ir;
      ir_id = info.ir_id;
      shared_bytes = Result.to_option info.fusable;
      max_grid_x =
        (let x, _, _ = device.capabilities.Framework_sig.max_grid_dims in
         x);
      max_shared = device.capabilities.Framework_sig.shared_mem_per_block;
      max_arg_slots = Launch_window.arg_slot_limit device.framework;
      args;
      block;
      grid;
      shared_mem;
      accesses = launch_accesses info ir args;
      issue;
      issue_fused;
    }

(* The JIT arm of {!run}. *)
let run_jit (module B : Framework_sig.BACKEND) ?stream ~(device : Device.t)
    ~name ~ir ~block ~grid ~shared_mem (args : vector_arg list) : unit =
  match ir with
  | None -> Execute_error.raise_error (Missing_ir {kernel = name})
  | Some ir_lazy ->
      let ir = Lazy.force ir_lazy in
      check_launch_bounds ~kernel:name ir ~block ;
      check_launch_args ~kernel:name ir args ;
      (* Before generate_source, so the diagnostic names the missing device
         capability rather than letting the backend emit a shader the device
         cannot legally load. *)
      check_device_capabilities_memo ~device ir ;
      check_launch_warp ~device ~kernel:name ir ~block ;
      (* SoA param names, derived from the ARGUMENTS by the same
                 predicate the transfer and expansion use (backlog-54). The
                 emitted signature and the bound argument list are then two
                 consequences of one decision rather than two decisions that
                 could disagree — and because both derive from [soa_dispatch]
                 applied to the same [args]/[device], they structurally cannot
                 disagree today. This list is only ever non-empty on CUDA/PTX
                 (the one framework [soa_dispatch] answers [Some] for), which
                 binds through the same bare pointer array as CUDA/C and HIP,
                 with nothing checking the bound arguments against the
                 compiled signature. So WERE the emitted signature and the
                 bound arguments to disagree here, the outcome would be the
                 same shape-dependent shift {!Backend_error.reject_soa_params}
                 analyses for the external-kernel case: a trap is among the
                 possible outcomes, but so is silent corruption or, when the
                 vector is declared last, no shift into any following
                 pointer slot — the vector's own slots still mis-bind —
                 depending on where the SoA vector sits in the parameter
                 list — never uniformly "a crash" or uniformly
                 "silently wrong data." Empty list on every non-CUDA/PTX
                 backend, because soa_dispatch is what decides — and since
                 backlog-214 the source-generating ones among them REFUSE a
                 non-empty list rather than ignore it, so this gate is the
                 fast path and no longer the only guarantee. Native,
                 Interpreter and WebGPU return None unconditionally and carry
                 no refusal; WebGPU is the one to watch, being a JIT backend
                 whose WGSL codegen is not yet wired. *)
      let soa_params = soa_param_names ir args device in
      let source = jit_source (module B) ~device ~name ~block ~soa_params ir in
      (* Arguments are expanded now, deferred or not (see [prepare_jit]). *)
      let issue =
        prepare_jit
          (module B)
          ?stream
          ~device
          ~name
          ~source
          ~block
          ~grid
          ~shared_mem
          args
      in
      if Launch_policy.deferred ir ~device ~stream ~soa_params then
        submit_deferred
          (module B)
          ~device
          ~ir
          ~block
          ~grid
          ~shared_mem
          ~issue
          args
      else begin
        (* Not deferred: everything deferred before it is issued first. *)
        Launch_hooks.flush_pending () ;
        Launch_window.count_immediate window ;
        issue ()
      end

(** Execute a kernel on a device using the unified dispatch mechanism.

    @param stream
      Enqueue the launch on this stream instead of the default one (HIP, CUDA);
      the call may return before the kernel has run, and the argument vectors
      stay rooted until the stream or a later event completes. On Metal the
      launch is enqueued on the stream's own command queue (host access to its
      buffers waits for it); on Native/Interpreter it is synchronous. Refused
      with [Feature_not_supported] on other JIT backends, and with
      [Invalid_argument] when the stream belongs to another device. The
      capability check and, on JIT backends, source generation are done once per
      physical IR and reused by later launches (see the launch-preparation memo
      above and {!clear_launch_cache}); argument checks run every time. Without
      [~stream], a JIT launch on a backend whose launch window is enabled (HIP,
      OpenCL, Vulkan, CUDA and Metal by default) is deferred until the next
      synchronisation point and may be fused with independent launches
      ({!Window}, specs/launch-window.md). Its checks and source generation
      still run in this call, but a device compilation or driver launch failure
      is raised by the call that flushes the window.

    @param device Target device
    @param name Kernel name
    @param ir Sarek IR kernel (lazy, only forced for JIT backends)
    @param native_fn Pre-compiled native function (for Direct backends)
    @param block Block dimensions
    @param grid Grid dimensions
    @param shared_mem Shared memory size in bytes (default 0)
    @param args Kernel arguments as vector_arg list
    @raise Execution_error if execution fails *)
let run ?(stream : Transfer.stream option) ~(device : Device.t) ~(name : string)
    ~(ir : Sarek_ir_types.kernel Lazy.t option)
    ~(native_fn :
       (block:Framework_sig.dims ->
       grid:Framework_sig.dims ->
       Framework_sig.exec_arg array ->
       unit)
       option) ~(block : Framework_sig.dims) ~(grid : Framework_sig.dims)
    ?(shared_mem : int = 0) (args : vector_arg list) : unit =
  check_stream_launch ~device stream ;
  (* A default-stream launch settles non-blocking streams still using its
     vectors; a launch on an explicit stream orders through events. *)
  if Option.is_none stream then
    List.iter (function Vec v -> Transfer.settle_vector v | _ -> ()) args ;
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
      | Framework_sig.JIT ->
          run_jit
            (module B)
            ?stream
            ~device
            ~name
            ~ir
            ~block
            ~grid
            ~shared_mem
            args
      | Framework_sig.Direct ->
          (* Direct path: call native function or interpret IR *)
          let dev = B.Device.get device.backend_id in
          B.Device.set_current dev ;
          let ir_val = Option.map Lazy.force ir in
          Option.iter
            (fun ir ->
              check_launch_bounds ~kernel:name ir ~block ;
              check_launch_args ~kernel:name ir args)
            ir_val ;
          Option.iter
            (fun ir ->
              check_device_capabilities_memo ~device ir ;
              check_launch_warp ~device ~kernel:name ir ~block)
            ir_val ;
          let exec_args = vector_args_to_exec_array args in
          B.execute_direct ~native_fn ~ir:ir_val ~block ~grid exec_args
      | Framework_sig.Custom ->
          (* Custom path: delegate to backend with IR *)
          let dev = B.Device.get device.backend_id in
          B.Device.set_current dev ;
          let ir_val = Option.map Lazy.force ir in
          Option.iter
            (fun ir ->
              check_launch_bounds ~kernel:name ir ~block ;
              check_launch_args ~kernel:name ir args)
            ir_val ;
          Option.iter
            (fun ir ->
              check_device_capabilities_memo ~device ir ;
              check_launch_warp ~device ~kernel:name ir ~block)
            ir_val ;
          let exec_args = vector_args_to_exec_array args in
          B.execute_direct ~native_fn ~ir:ir_val ~block ~grid exec_args)

(** {1 V2 Vector Execution Helpers} *)

(** Mark vectors as stale on CPU after kernel execution.

    After a kernel modifies vector data on a device, we need to track that the
    CPU-side data is now stale. This ensures future CPU reads will trigger a
    device→CPU transfer.

    Special cases:
    - Native backend: No-op (uses zero-copy shared memory, no staleness)
    - JIT backends: Always mark stale (Transfer module handles zero-copy checks)
    - OpenCL CPU: Mark stale, like the other JIT backends (a host-pointer buffer
      from [Host_ptr_buffer] is read back through map/unmap)

    @param args Arguments that may contain vectors
    @param dev Device that just executed the kernel *)
let mark_vectors_stale (args : vector_arg list) (dev : Device.t) : unit =
  (* Native and Interpreter execute directly on host storage (the interpreter
     reads/writes vector elements through EXEC_VECTOR get/set, never through
     the device buffer), so the CPU copy is authoritative after the run -
     marking it Stale_CPU would make the next host read clobber kernel
     results with the stale device-buffer copy (observed with custom record
     vectors, which have no zero-copy path).
     Always mark stale for JIT backends, OpenCL CPU devices included: their
     buffers are copies unless created by [Host_ptr_buffer], whose read-back
     is a map. *)
  if dev.Device.framework = "Native" || dev.Device.framework = "Interpreter"
  then ()
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
  | Vector.Scalar Vector.Float16 ->
      (* See the Float16 arm of [get] above: f16 reads as an f32 value. *)
      Array.init len (fun i -> Sarek_ir_interp.VFloat32 (Vector.get vec i))
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
  | Vector.Scalar Vector.Int8 ->
      (* Signed 8-bit: the sign-extended int32 an 8-bit read yields. *)
      Array.init len (fun i ->
          Sarek_ir_interp.VInt32 (Int32.of_int (Vector.get vec i)))
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
  | Vector.Scalar Vector.Float16 ->
      (* Round-on-store: the Bigarray.Float16 cell narrows to binary16. *)
      for i = 0 to len - 1 do
        Vector.set vec i (Sarek_ir_interp.to_float32 arr.(i))
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
        | ( Sarek_ir_interp.VRecord (type_name, _)
          | Sarek_ir_interp.VVariant (type_name, _, _) ) as v -> (
            (* Both a record element (VRecord) and a standalone variant element
               (VVariant) are decoded back to their native OCaml value through
               the registered [@@sarek.type] helper. *)
            match Sarek_type_helpers.lookup_typed custom.Vector.type_id with
            | Some (module H) -> Vector.set vec i (H.from_value v)
            | None ->
                Execute_error.raise_error
                  (Execute_error.Type_helper_not_found
                     {
                       type_name;
                       context =
                         "sync_vector_back (custom conversion from interpreter)";
                     }))
        | _ -> () (* Skip other values *)
      done
  | Vector.Scalar Vector.Char ->
      (* Char type: convert from int32, keeping the low byte as a device store
         does ([Char.chr] would raise on anything else rather than wrap). *)
      for i = 0 to len - 1 do
        Vector.set
          vec
          i
          (Char.chr (Int32.to_int (Sarek_ir_interp.to_int32 arr.(i)) land 0xff))
      done
  | Vector.Scalar Vector.Int8 ->
      (* [Bigarray.int8_signed] keeps the low 8 bits of the [int] it is given. *)
      for i = 0 to len - 1 do
        Vector.set vec i (Int32.to_int (Sarek_ir_interp.to_int32 arr.(i)))
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
  (* The SAME launch-time check the three {!run} dispatch paths apply. This
     entry point used to skip it, which was the worst place to skip it: the
     interpreter is the ORACLE the GPU backends are checked against, so an
     unchecked mismatch here does not just produce a wrong answer, it produces a
     wrong answer that the f16 agreement gates then compare the GPU to.

     Concretely, without this an f32 vector passed to an f16-declared kernel
     ran to completion: [vector_to_interp_array] yields [VFloat32] from the f32
     host vector and [interp_array_to_vector] dispatches on the HOST vector's
     kind, so the writeback never narrows and the interpreter silently returns
     f32-precision results. The positional zip below would also have quietly
     renamed or dropped arguments on an arity mismatch (it falls back to
     "param%d") instead of reporting it. *)
  check_launch_bounds ~kernel:ir.Sarek_ir_types.kern_name ir ~block ;
  check_launch_warp_width
    ~kernel:ir.Sarek_ir_types.kern_name
    ~width:(Sarek_warp.simulated_width ())
    ir
    ~block ;
  check_launch_args ~kernel:ir.Sarek_ir_types.kern_name ir args ;
  (* The capability half of the same discipline (backlog-154). Placed next to
     check_launch_args because the two together are what the three [run]
     dispatch paths apply, and this entry point had only the first. See
     {!check_interpreter_capabilities}. *)
  check_interpreter_capabilities ir ;
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
            (* A vector passed twice is ONE value array (R-63). *)
            let arr =
              Sarek_ir_interp.bind_vector
                writebacks
                v
                ~to_array:vector_to_interp_array
            in
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

    @param stream
      Launch on this stream (see {!run}). Auto-transfers of the arguments still
      happen synchronously before the launch; vectors already resident (e.g.
      device-only ones) cost nothing.
    @param device Target device (determines backend)
    @param ir Sarek IR kernel definition
    @param args Kernel arguments (vectors and scalars)
    @param block Thread block dimensions (e.g., (256, 1, 1))
    @param grid Grid dimensions (e.g., (4, 1, 1))
    @param shared_mem Optional shared memory size in bytes (default: 0)
    @raise Execute_error on validation or execution failure *)
let run_vectors_plain ?(stream : Transfer.stream option) ~(device : Device.t)
    ~(ir : Sarek_ir_types.kernel) ~(args : vector_arg list)
    ~(block : Framework_sig.dims) ~(grid : Framework_sig.dims)
    ?(shared_mem : int = 0) () : unit =
  (* Refuse an unsupported stream launch before any transfer happens. *)
  check_stream_launch ~device stream ;

  (* Unified path for all backends:
     1. Transfer to device (Native, Interpreter: host memory, no transfer)
     2. Pass vector_args to run (it expands for JIT, passes direct for Direct)
     3. Mark stale (Native, Interpreter: no-op, the host copy is the result) *)

  (* 1. Transfer all vectors to device. ~soa_abi:true — this entry point runs a
     Sarek-generated kernel, so an SoA-bound vector's leaves are what the launch
     will read. [Execute.run_source] deliberately does NOT pass it. *)
  transfer_vectors_to_device
    ~soa_abi:true
    ~settle:(Option.is_none stream)
    args
    device ;

  (* 2. Dispatch via run - it handles expansion per backend *)
  run
    ?stream
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

(* ---- Matrix-product specialisation (specs/matrix-pattern-dynamic.md) ---- *)

let matrix_scalar = function
  | Int i -> Some (Int32.of_int i)
  | Int32 i -> Some i
  | Int64 i ->
      (* Bounds are int32 in the kernel: a value that does not fit is not a
         bound the rewrite can use. *)
      if Int64.of_int32 (Int64.to_int32 i) = i then Some (Int64.to_int32 i)
      else None
  | Vec _ | Float32 _ | Float64 _ -> None

let matrix_launch (ir : Sarek_ir_types.kernel) args
    ~(block : Framework_sig.dims) ~(grid : Framework_sig.dims) :
    Sarek_ir_eval_int.launch option =
  if List.length args <> List.length ir.kern_params then None
  else
    let values =
      List.filter_map
        (fun (d, a) ->
          match d with
          | Sarek_ir_types.DParam (v, None) ->
              Option.map (fun x -> (v.var_id, x)) (matrix_scalar a)
          | _ -> None)
        (List.combine ir.kern_params args)
    in
    Some
      {
        Sarek_ir_eval_int.grid = (grid.x, grid.y, grid.z);
        block = (block.x, block.y, block.z);
        scalar = (fun v -> List.assoc_opt v.var_id values);
      }

let interpreter_is_parallel (device : Device.t) =
  let n = device.name and p = "Parallel" in
  let rec has i =
    i + String.length p <= String.length n
    && (String.sub n i (String.length p) = p || has (i + 1))
  in
  has 0

let run_specialised ?stream ~device ~args (s : Matrix_specialise.specialised) =
  let extra = List.map (fun x -> Int32 (Int32.of_int x)) s.extra in
  let d3 (x, y, z) = Framework_sig.dims_3d x y z in
  if s.via_interpreter then
    run_interpreter_vectors
      ~ir:s.ir
      ~args:(args @ extra)
      ~block:(d3 s.block)
      ~grid:(d3 s.grid)
      ~parallel:(interpreter_is_parallel device)
  else
    run_vectors_plain
      ?stream
      ~device
      ~ir:s.ir
      ~args:(args @ extra)
      ~block:(d3 s.block)
      ~grid:(d3 s.grid)
      ()

(** [run_vectors] that also returns what ran: the kernel as given, or a
    matrix-unit rewrite of it ({!Matrix_specialise}). [precision] chooses the
    operand rounding of a float32 product (it must have been accepted).
    [~explicit:true] is an explicit request, like [Sarek_matrix.run]: float
    products need the accuracy opt-ins only (not
    [Sarek_coopmat.accept_automatic_float_rewrite]), and the interpreter is
    rewritten too. *)
let run_vectors_outcome ?(stream : Transfer.stream option) ?precision ?explicit
    ~(device : Device.t) ~(ir : Sarek_ir_types.kernel) ~(args : vector_arg list)
    ~(block : Framework_sig.dims) ~(grid : Framework_sig.dims)
    ?(shared_mem : int = 0) () : Matrix_specialise.outcome =
  (* Against the caller's kernel, before a matrix rewrite (which chooses its
     own geometry and carries no bound) can replace it. *)
  check_launch_bounds ~kernel:ir.Sarek_ir_types.kern_name ir ~block ;
  check_launch_warp ~device ~kernel:ir.Sarek_ir_types.kern_name ir ~block ;
  let outcome =
    Matrix_specialise.choose
      ?precision
      ?explicit
      ~stream:(Option.is_some stream)
      ~device
      ~ir
      ~launch:(fun () -> matrix_launch ir args ~block ~grid)
      ()
  in
  (match outcome with
  | Matrix_specialise.Specialised s -> run_specialised ?stream ~device ~args s
  | Matrix_specialise.Original _ ->
      run_vectors_plain ?stream ~device ~ir ~args ~block ~grid ~shared_mem ()) ;
  outcome

(** Launch [ir] on [device]. When [ir] is a matrix product that matrix units can
    compute under the current restrictions and accuracy opt-ins, the rewritten
    kernel runs instead, writing exactly the same outputs
    (specs/matrix-pattern-dynamic.md). *)
let run_vectors ?stream ~device ~ir ~args ~block ~grid ?shared_mem () : unit =
  ignore
    (run_vectors_outcome ?stream ~device ~ir ~args ~block ~grid ?shared_mem ()
      : Matrix_specialise.outcome)

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
  (* Not deferred (external source): issue every deferred launch first. *)
  Launch_hooks.flush_pending () ;
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
