(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Structure-of-Arrays (SoA) kernel launch — Tier 1c.
 *
 * Launches a Sarek kernel with one or more custom (flat-record) vector
 * arguments lowered as Structure-of-Arrays: each such argument binds N per-leaf
 * base pointers + one shared length (the #260 PTX emitter ABI) instead of a
 * single packed AoS (pointer, length) pair.
 *
 * SoA lowering is CUDA/PTX-only (the #260 emitter path). {!run_soa} raises a
 * located error on any non-PTX backend rather than binding the SoA N-pointer
 * ABI to an AoS kernel signature (which would read wrong data), so it can never
 * produce incorrect results. For a non-PTX device, drive the AoS host vector
 * ({!Spoc_core.Soa_vector.aos_vector}) through {!Execute.run_vectors} instead.
 *
 * This lives in a separate module (not Execute.ml) because Execute.ml is copied
 * verbatim into the ctypes-free jsoo smoke target, whose Spoc_core shim has no
 * Soa/Soa_vector. Keeping the SoA launch here confines the Soa_vector
 * dependency to the native library.
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry
open Spoc_core
open Execute_error

(** An argument to {!run_soa}: either a regular {!Execute.vector_arg}, or a SoA
    custom vector (lowered in place to its N leaf base pointers + shared
    length). *)
type soa_arg =
  | SA_Soa : 'a Soa_vector.t -> soa_arg
  | SA_Reg of Execute.vector_arg

(** Names of the kernel's [DParam]s, in declaration order. *)
let kernel_param_names (ir : Sarek_ir_types.kernel) : string list =
  List.filter_map
    (function
      | Sarek_ir_types.DParam (v, _) -> Some v.Sarek_ir_types.var_name
      | _ -> None)
    ir.Sarek_ir_types.kern_params

(** [run_source_arg]s for one regular vector: (buffer, length). *)
let rs_args_of_reg_vector (type a b) (v : (a, b) Vector.t) (dev : Device.t) :
    Framework_sig.run_source_arg list =
  let (module B : Vector.DEVICE_BUFFER) = Execute.get_device_buffer v dev in
  let len = Vector.length v in
  [
    Framework_sig.RSA_Buffer {binder = B.bind_to_kargs; length = len};
    Framework_sig.RSA_Vector_Length (Int32.of_int len);
  ]

(** [run_source_arg]s for a SoA vector: N leaf base pointers (leaf order)
    followed by one shared [RSA_Vector_Length]. *)
let rs_args_of_soa_vector (sv : 'a Soa_vector.t) (dev : Device.t) :
    Framework_sig.run_source_arg list =
  let len = Soa_vector.length sv in
  let leaf_args =
    Array.to_list
      (Array.map
         (fun (Soa_vector.Leaf v) ->
           let (module B : Vector.DEVICE_BUFFER) =
             Execute.get_device_buffer v dev
           in
           Framework_sig.RSA_Buffer {binder = B.bind_to_kargs; length = len})
         (Soa_vector.leaves sv))
  in
  leaf_args @ [Framework_sig.RSA_Vector_Length (Int32.of_int len)]

let rs_args_of_reg (a : Execute.vector_arg) (dev : Device.t) :
    Framework_sig.run_source_arg list =
  match a with
  | Execute.Vec v -> rs_args_of_reg_vector v dev
  | Execute.Int n -> [Framework_sig.RSA_Int32 (Int32.of_int n)]
  | Execute.Int32 n -> [Framework_sig.RSA_Int32 n]
  | Execute.Int64 n -> [Framework_sig.RSA_Int64 n]
  | Execute.Float32 f -> [Framework_sig.RSA_Float32 f]
  | Execute.Float64 f -> [Framework_sig.RSA_Float64 f]

(** Execute a kernel with SoA-lowered custom-vector arguments.

    {b Host coherence contract.} The AoS host buffer of each SoA vector is the
    source of truth. [run_soa] calls {!Spoc_core.Soa_vector.scatter} internally
    (AoS host -> per-leaf host -> device) immediately before launch, so there is
    no user-visible window between scatter and launch and host [set]s made
    before the call are always reflected on the device. For a kernel that only
    {e reads} SoA leaves nothing more is needed. For a kernel that {e writes} an
    SoA leaf, the device-side leaf buffers become authoritative; to observe the
    result through the AoS vector the caller must round-trip explicitly:
    transfer each leaf back to the host (e.g. {!Spoc_core.Transfer.to_cpu} on
    every {!Spoc_core.Soa_vector.leaves} entry — [run_soa] marks them stale so
    this triggers a D2H copy) and then call {!Spoc_core.Soa_vector.gather}
    (per-leaf host -> AoS host). [run_soa] never gathers automatically.

    @param device Target device (must be a CUDA/PTX backend)
    @param ir Sarek IR kernel definition
    @param args Kernel arguments (SoA vectors + regular args) in param order
    @param block Thread block dimensions
    @param grid Grid dimensions
    @param shared_mem Optional shared memory size in bytes (default: 0)
    @raise Execute_error on a non-PTX device or a codegen/launch failure *)
let run_soa ~(device : Device.t) ~(ir : Sarek_ir_types.kernel)
    ~(args : soa_arg list) ~(block : Framework_sig.dims)
    ~(grid : Framework_sig.dims) ?(shared_mem : int = 0) () : unit =
  match Framework_registry.find_backend device.framework with
  | None ->
      raise_error
        (Backend_error
           {
             backend = device.framework;
             message = "Backend not found in registry";
           })
  | Some (module B : Framework_sig.BACKEND) ->
      (* Gate: SoA lowering is PTX-only. Never emit the SoA N-pointer ABI for a
         backend that generates AoS code. *)
      if not (List.mem Framework_sig.PTX B.supported_source_langs) then
        raise_error
          (Unsupported_argument
             {
               arg_type = "SoA custom vector";
               context =
                 device.framework
                 ^ " backend: SoA lowering requires a CUDA/PTX device (drive \
                    the AoS host vector through run_vectors for other \
                    backends)";
             }) ;
      (* SoA parameter names = kernel param name at each SA_Soa position. *)
      let param_names = kernel_param_names ir in
      let soa_params =
        List.filteri
          (fun i _ ->
            match List.nth_opt args i with
            | Some (SA_Soa _) -> true
            | _ -> false)
          param_names
      in
      (* Generate SoA-lowered source through the backend (keeps Execute
         codegen-free; only the CUDA/PTX backend honours ~soa_params). *)
      let source =
        match B.generate_source ~block ~soa_params ir with
        | Some s -> s
        | None ->
            raise_error
              (Compilation_failed
                 {
                   kernel = ir.Sarek_ir_types.kern_name;
                   reason = device.framework ^ ": generate_source returned None";
                 })
      in
      (* Host->device: scatter each SoA vector into its leaves and transfer
         every leaf; transfer regular vectors. *)
      List.iter
        (function
          | SA_Soa sv ->
              Soa_vector.scatter sv ;
              Array.iter
                (fun (Soa_vector.Leaf v) -> Transfer.to_device v device)
                (Soa_vector.leaves sv)
          | SA_Reg (Execute.Vec v) -> Transfer.to_device v device
          | SA_Reg _ -> ())
        args ;
      (* Build the run_source args in parameter order. *)
      let rs_args =
        List.concat_map
          (function
            | SA_Soa sv -> rs_args_of_soa_vector sv device
            | SA_Reg a -> rs_args_of_reg a device)
          args
      in
      let dev = B.Device.get device.backend_id in
      B.Device.set_current dev ;
      B.run_source
        ~source
        ~lang:Framework_sig.PTX
        ~kernel_name:ir.Sarek_ir_types.kern_name
        ~block
        ~grid
        ~shared_mem
        rs_args ;
      (* Mark device-authoritative vectors stale on CPU (regular outputs and any
         SoA leaves the kernel may have written). *)
      List.iter
        (function
          | SA_Reg a -> Execute.mark_vectors_stale [a] device
          | SA_Soa sv ->
              Array.iter
                (fun (Soa_vector.Leaf v) ->
                  Execute.mark_vectors_stale [Execute.Vec v] device)
                (Soa_vector.leaves sv))
        args
