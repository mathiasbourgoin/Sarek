(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Ctypes
open Vulkan_types
open Vulkan_bindings
open Vulkan_api_base
open Spoc_framework_registry
module Device = Vulkan_api_device
module Memory = Vulkan_api_memory
module Stream = Vulkan_api_stream

type t = {
  shader_module : vk_shader_module;
  pipeline : vk_pipeline;
  pipeline_layout : vk_pipeline_layout;
  descriptor_set_layout : vk_descriptor_set_layout;
  descriptor_pool : vk_descriptor_pool;
  descriptor_set : vk_descriptor_set;
  name : string;
  num_bindings : int;
  device : Device.t;
}

type arg =
  | ArgBuffer : _ Memory.buffer -> arg
  | ArgInt32 : int32 -> arg
  | ArgInt64 : int64 -> arg
  | ArgFloat32 : float -> arg
  | ArgFloat64 : float -> arg
  | ArgPtr : nativeint -> arg

(** Existential wrapper to hide buffer type parameter *)
type any_buffer = AnyBuf : 'a Memory.buffer -> any_buffer

(** Scalar kernel argument, tagged with its byte width/value so the
    push-constant block can be materialized in one pass at launch time (see
    {!build_push_constants}) instead of being appended byte-by-byte in whatever
    order [set_arg_*] happens to be called. *)
type scalar_arg =
  | SInt32 of int32
  | SInt64 of int64
  | SFloat32 of float
  | SFloat64 of float

type args = {
  (* Buffer arguments, indexed by the caller-supplied idx (last-set-wins),
     not by call order: see Spoc_framework.Kernel_args. idx here shares its
     numbering space with scalar args (push constants below), so it is
     generally sparse/non-contiguous from this store's point of view --
     resolve_bindings below derives the actual Vulkan descriptor binding
     numbers from it. *)
  buffer_store : any_buffer Spoc_framework.Kernel_args.t;
  (* Scalar (push-constant) arguments, indexed by the caller-supplied idx,
     same last-set-wins semantics as buffer_store. Kept separate from the
     packed byte layout until launch: see build_push_constants. *)
  scalar_store : scalar_arg Spoc_framework.Kernel_args.t;
  mutable descriptor_set : vk_descriptor_set;
}

(** Vulkan descriptor binding numbers are assigned to buffer-typed kernel
    parameters only, in the order those parameters appear in the kernel
    signature (see the GLSL codegen, which numbers `layout(binding = N)` this
    way, skipping scalar parameters entirely). Since kernel-arg idx is shared
    across buffers and scalars, idx itself is not usable as a binding number
    directly; instead the buffer with the Nth-smallest idx among buffer args
    gets binding N. Because buffers cannot be reordered by the caller relative
    to each other without also changing which kernel parameter they represent,
    this rank-by-idx is stable and order-of-call-independent, unlike the
    previous per-call sequential counter it replaces. *)
let resolve_bindings args : (int * any_buffer) list =
  Spoc_framework.Kernel_args.to_sorted_list args.buffer_store
  |> List.mapi (fun binding (_idx, buf) -> (binding, buf))

(** Validate the set of buffer-argument indices actually bound in [store]
    against the number of buffer bindings the compiled kernel expects
    ([expected_count], derived from the GLSL source's `binding = N` declarations
    \- see [num_bindings] in {!compile}).

    [resolve_bindings] above silently compresses whatever indices are present
    into dense, ascending descriptor bindings; that rank-mapping is correct for
    a valid index *set*, but gives no signal at all if the caller passed a
    nonsensical one (negative indices, or simply the wrong number of buffers -
    e.g. a caller-side idx typo that drops one buffer and duplicates another
    slot). This checks the set's sanity before [resolve_bindings]'s compression
    can paper over it. Exposed for direct unit testing (no hardware required).
*)
let validate_buffer_indices ~expected_count
    (store : 'a Spoc_framework.Kernel_args.t) : (unit, string) result =
  let entries = Spoc_framework.Kernel_args.to_sorted_list store in
  let negative_idxs =
    List.filter_map (fun (idx, _) -> if idx < 0 then Some idx else None) entries
  in
  if negative_idxs <> [] then
    Error
      (Printf.sprintf
         "negative buffer index%s: [%s]"
         (if List.length negative_idxs = 1 then "" else "es")
         (String.concat ", " (List.map string_of_int negative_idxs)))
  else
    let actual_count = List.length entries in
    if actual_count <> expected_count then
      Error
        (Printf.sprintf
           "expected %d buffer argument%s but got %d (indices: [%s])"
           expected_count
           (if expected_count = 1 then "" else "s")
           actual_count
           (String.concat
              ", "
              (List.map (fun (idx, _) -> string_of_int idx) entries)))
    else Ok ()

(* Compilation cache *)
let cache : (string, t) Hashtbl.t = Hashtbl.create 16

(** Create shader module from SPIR-V *)
let create_shader_module device spirv =
  let code_size = String.length spirv in
  (* SPIR-V must be 4-byte aligned *)
  if code_size mod 4 <> 0 then
    Vulkan_error.raise_error
      (Vulkan_error.module_load_failed
         code_size
         "SPIR-V size must be multiple of 4") ;

  (* Convert string to uint32 array *)
  let num_words = code_size / 4 in
  let code = CArray.make uint32_t num_words in
  for i = 0 to num_words - 1 do
    let b0 = Char.code spirv.[i * 4] in
    let b1 = Char.code spirv.[(i * 4) + 1] in
    let b2 = Char.code spirv.[(i * 4) + 2] in
    let b3 = Char.code spirv.[(i * 4) + 3] in
    let word = b0 lor (b1 lsl 8) lor (b2 lsl 16) lor (b3 lsl 24) in
    CArray.set code i (Unsigned.UInt32.of_int word)
  done ;

  let create_info = make vk_shader_module_create_info in
  setf
    create_info
    shader_create_sType
    (u32 vk_structure_type_shader_module_create_info) ;
  setf create_info shader_create_pNext null ;
  setf create_info shader_create_flags (Unsigned.UInt32.of_int 0) ;
  setf create_info shader_create_codeSize (Unsigned.Size_t.of_int code_size) ;
  setf create_info shader_create_pCode (CArray.start code) ;

  let shader_module = allocate vk_shader_module vk_null_handle in
  check
    "vkCreateShaderModule"
    (vkCreateShaderModule
       device.Device.device
       (addr create_info)
       null
       shader_module) ;
  !@shader_module

(** Compile GLSL source to compute pipeline *)
let compile device ~name ~source =
  (* 1. Check cache for SPIR-V *)
  let driver_version =
    let maj, min, patch = device.Device.api_version in
    Printf.sprintf "%d.%d.%d" maj min patch
  in
  let cache_key =
    Framework_cache.compute_key
      ~dev_name:device.Device.name
      ~driver_version
      ~name
      ~source
  in

  let spirv =
    match Framework_cache.get ~key:cache_key with
    | Some data ->
        Spoc_core.Log.debugf
          Spoc_core.Log.Device
          "[Vulkan] Cache hit for kernel %s"
          name ;
        data
    | None ->
        Spoc_core.Log.debugf
          Spoc_core.Log.Device
          "[Vulkan] Cache miss for kernel %s, compiling..."
          name ;
        let data = compile_glsl_to_spirv ~entry_point:name source in
        Framework_cache.put ~key:cache_key ~data ;
        data
  in

  (* Create shader module *)
  let shader_module = create_shader_module device spirv in

  (* Count buffer bindings from source (look for "binding = N") *)
  let num_bindings =
    let count = ref 0 in
    let binding_re = Str.regexp "binding *= *[0-9]+" in
    let _ =
      try
        let pos = ref 0 in
        while true do
          let _ = Str.search_forward binding_re source !pos in
          incr count ;
          pos := Str.match_end ()
        done
      with Not_found -> ()
    in
    max 1 !count
  in

  (* Create descriptor set layout *)
  let bindings = CArray.make vk_descriptor_set_layout_binding num_bindings in
  for i = 0 to num_bindings - 1 do
    let binding = make vk_descriptor_set_layout_binding in
    setf binding dsl_binding_binding (Unsigned.UInt32.of_int i) ;
    setf
      binding
      dsl_binding_descriptorType
      (u32 vk_descriptor_type_storage_buffer) ;
    setf binding dsl_binding_descriptorCount (Unsigned.UInt32.of_int 1) ;
    setf
      binding
      dsl_binding_stageFlags
      (Unsigned.UInt32.of_int vk_shader_stage_compute_bit) ;
    setf binding dsl_binding_pImmutableSamplers null ;
    CArray.set bindings i binding
  done ;

  let dsl_create_info = make vk_descriptor_set_layout_create_info in
  setf
    dsl_create_info
    dsl_create_sType
    (u32 vk_structure_type_descriptor_set_layout_create_info) ;
  setf dsl_create_info dsl_create_pNext null ;
  setf dsl_create_info dsl_create_flags (Unsigned.UInt32.of_int 0) ;
  setf
    dsl_create_info
    dsl_create_bindingCount
    (Unsigned.UInt32.of_int num_bindings) ;
  setf dsl_create_info dsl_create_pBindings (CArray.start bindings) ;

  let dsl = allocate vk_descriptor_set_layout vk_null_handle in
  check
    "vkCreateDescriptorSetLayout"
    (vkCreateDescriptorSetLayout
       device.Device.device
       (addr dsl_create_info)
       null
       dsl) ;

  (* Create pipeline layout *)
  let pl_create_info = make vk_pipeline_layout_create_info in
  setf
    pl_create_info
    pl_create_sType
    (u32 vk_structure_type_pipeline_layout_create_info) ;
  setf pl_create_info pl_create_pNext null ;
  setf pl_create_info pl_create_flags (Unsigned.UInt32.of_int 0) ;
  setf pl_create_info pl_create_setLayoutCount (Unsigned.UInt32.of_int 1) ;
  setf pl_create_info pl_create_pSetLayouts dsl ;

  (* Add push constant range for scalar parameters *)
  let push_constant_range = make vk_push_constant_range in
  setf
    push_constant_range
    push_const_stageFlags
    (Unsigned.UInt32.of_int vk_shader_stage_compute_bit) ;
  setf push_constant_range push_const_offset (Unsigned.UInt32.of_int 0) ;
  setf push_constant_range push_const_size (Unsigned.UInt32.of_int 128) ;

  (* 128 bytes max for push constants *)
  setf
    pl_create_info
    pl_create_pushConstantRangeCount
    (Unsigned.UInt32.of_int 1) ;
  setf pl_create_info pl_create_pPushConstantRanges (addr push_constant_range) ;

  let pipeline_layout = allocate vk_pipeline_layout vk_null_handle in
  check
    "vkCreatePipelineLayout"
    (vkCreatePipelineLayout
       device.Device.device
       (addr pl_create_info)
       null
       pipeline_layout) ;

  (* Create compute pipeline *)
  let stage_info = make vk_pipeline_shader_stage_create_info in
  setf
    stage_info
    shader_stage_sType
    (u32 vk_structure_type_pipeline_shader_stage_create_info) ;
  setf stage_info shader_stage_pNext null ;
  setf stage_info shader_stage_flags (Unsigned.UInt32.of_int 0) ;
  setf
    stage_info
    shader_stage_stage
    (Unsigned.UInt32.of_int vk_shader_stage_compute_bit) ;
  setf stage_info shader_stage_module shader_module ;
  setf stage_info shader_stage_pName "main" ;
  setf stage_info shader_stage_pSpecializationInfo null ;

  let pipeline_info = make vk_compute_pipeline_create_info in
  setf
    pipeline_info
    compute_pipe_sType
    (u32 vk_structure_type_compute_pipeline_create_info) ;
  setf pipeline_info compute_pipe_pNext null ;
  setf pipeline_info compute_pipe_flags (Unsigned.UInt32.of_int 0) ;
  setf pipeline_info compute_pipe_stage stage_info ;
  setf pipeline_info compute_pipe_layout !@pipeline_layout ;
  setf pipeline_info compute_pipe_basePipelineHandle vk_null_handle ;
  setf pipeline_info compute_pipe_basePipelineIndex (Int32.of_int (-1)) ;

  let pipeline = allocate vk_pipeline vk_null_handle in
  let result =
    vkCreateComputePipelines
      device.Device.device
      vk_null_handle
      (Unsigned.UInt32.of_int 1)
      (addr pipeline_info)
      null
      pipeline
  in
  check "vkCreateComputePipelines" result ;

  (* Create descriptor pool *)
  let pool_size = make vk_descriptor_pool_size in
  setf pool_size pool_size_type (u32 vk_descriptor_type_storage_buffer) ;
  setf
    pool_size
    pool_size_descriptorCount
    (Unsigned.UInt32.of_int (num_bindings * 10)) ;

  let pool_info = make vk_descriptor_pool_create_info in
  setf
    pool_info
    desc_pool_sType
    (u32 vk_structure_type_descriptor_pool_create_info) ;
  setf pool_info desc_pool_pNext null ;
  setf pool_info desc_pool_flags (Unsigned.UInt32.of_int 0) ;
  setf pool_info desc_pool_maxSets (Unsigned.UInt32.of_int 10) ;
  setf pool_info desc_pool_poolSizeCount (Unsigned.UInt32.of_int 1) ;
  setf pool_info desc_pool_pPoolSizes (addr pool_size) ;

  let pool = allocate vk_descriptor_pool vk_null_handle in
  check
    "vkCreateDescriptorPool"
    (vkCreateDescriptorPool device.Device.device (addr pool_info) null pool) ;

  (* Allocate persistent descriptor set *)
  let ds_ai = make vk_descriptor_set_allocate_info in
  setf
    ds_ai
    desc_set_alloc_sType
    (u32 vk_structure_type_descriptor_set_allocate_info) ;
  setf ds_ai desc_set_alloc_pNext null ;
  setf ds_ai desc_set_alloc_descriptorPool !@pool ;
  setf ds_ai desc_set_alloc_descriptorSetCount (u32 1) ;
  (* Keep this allocation alive - it's passed by pointer to Vulkan *)
  let dsl_ptr = allocate vk_descriptor_set_layout !@dsl in
  setf ds_ai desc_set_alloc_pSetLayouts dsl_ptr ;

  let desc_set = allocate vk_descriptor_set vk_null_handle in
  check
    "vkAllocateDescriptorSets"
    (vkAllocateDescriptorSets device.Device.device (addr ds_ai) desc_set) ;
  ignore dsl_ptr ;
  {
    shader_module;
    pipeline = !@pipeline;
    pipeline_layout = !@pipeline_layout;
    descriptor_set_layout = !@dsl;
    descriptor_pool = !@pool;
    descriptor_set = !@desc_set;
    name;
    num_bindings;
    device;
  }

let compile_cached device ~name ~source =
  (* Cache key must include the kernel/entry name - a source string may
     define more than one kernel, and without the name in the key a second
     kernel compiled from the same source would resolve to the first
     kernel's cached pipeline (see Compile_cache.mli). *)
  let key =
    Spoc_framework.Compile_cache.make_key
      ~device:(string_of_int device.Device.id)
      ~name
      ~source
      ()
  in
  match Hashtbl.find_opt cache key with
  | Some k -> k
  | None ->
      let k = compile device ~name ~source in
      Hashtbl.add cache key k ;
      k

let clear_cache () =
  Hashtbl.iter
    (fun _ k ->
      vkDestroyPipeline k.device.Device.device k.pipeline null ;
      vkDestroyPipelineLayout k.device.Device.device k.pipeline_layout null ;
      vkDestroyDescriptorPool k.device.Device.device k.descriptor_pool null ;
      vkDestroyDescriptorSetLayout
        k.device.Device.device
        k.descriptor_set_layout
        null ;
      vkDestroyShaderModule k.device.Device.device k.shader_module null)
    cache ;
  Hashtbl.clear cache

let create_args () =
  {
    buffer_store = Spoc_framework.Kernel_args.create ();
    scalar_store = Spoc_framework.Kernel_args.create ();
    descriptor_set = vk_null_handle;
  }

let set_arg_buffer args idx buf =
  Spoc_framework.Kernel_args.set args.buffer_store idx (AnyBuf buf)

let set_arg_int32 args idx n =
  Spoc_framework.Kernel_args.set args.scalar_store idx (SInt32 n)

let set_arg_int64 args idx n =
  Spoc_framework.Kernel_args.set args.scalar_store idx (SInt64 n)

let set_arg_float32 args idx f =
  Spoc_framework.Kernel_args.set args.scalar_store idx (SFloat32 f)

let set_arg_float64 args idx f =
  Spoc_framework.Kernel_args.set args.scalar_store idx (SFloat64 f)

let set_arg_ptr _args _idx _p =
  Vulkan_error.raise_error
    (Vulkan_error.feature_not_supported "raw pointer kernel arguments")

(* Vulkan guarantees at least 128 bytes of push constants. *)
let push_constant_limit = 128

(** Materialize the push-constant byte block for [args] at launch time, from the
    full ordered argument set (buffers + scalars), matching EXACTLY the GLSL
    block layout emitted by [Sarek_ir_glsl.gen_push_constants]
    (sarek/codegen/Sarek_ir_glsl.ml:889-919): all vector lengths first, in
    vector-declaration order, followed by all user scalar parameters, in
    declaration order.

    - "Vector-declaration order" here is the same order used for descriptor
      bindings: the buffer with the Nth-smallest caller-supplied idx gets the
      Nth length slot (see resolve_bindings above / gen_push_constants's
      [vectors] list, which is built by iterating kernel params in order).
    - "Scalar declaration order" is the ascending order of the scalar arguments'
      own caller-supplied idx (see gen_push_constants's [scalars] list, same
      iteration).

    This must not be done incrementally inside each [set_arg_*] call: the caller
    may invoke those in any call order (e.g. a vector, then a scalar, then
    another vector), which does not match the GLSL grouping, so the block can
    only be assembled correctly once every argument is known. *)
let build_push_constants args : bytes option =
  let vector_lengths =
    resolve_bindings args
    |> List.map (fun (_binding, AnyBuf buf) -> Int32.of_int buf.Memory.size)
  in
  let scalars =
    let entries = Spoc_framework.Kernel_args.to_sorted_list args.scalar_store in
    let negative_idxs =
      List.filter_map
        (fun (idx, _) -> if idx < 0 then Some idx else None)
        entries
    in
    if negative_idxs <> [] then
      Vulkan_error.raise_error
        (Vulkan_error.context_error
           "push constant"
           (Printf.sprintf
              "negative scalar index%s: [%s]"
              (if List.length negative_idxs = 1 then "" else "es")
              (String.concat ", " (List.map string_of_int negative_idxs)))) ;
    List.map snd entries
  in
  if vector_lengths = [] && scalars = [] then None
  else begin
    let pc = Bytes.create push_constant_limit in
    let offset = ref 0 in
    (* GLSL's std430-like push_constant layout base-aligns each scalar member
       to its own size (4 bytes for int/float, 8 bytes for int64_t/double), so
       an 8-byte field following one or more 4-byte fields needs 4 bytes of
       padding inserted by the compiler before it. [write_at] must reproduce
       that padding on the host side or a mixed 32/64-bit push-constant block
       reads shifted bytes for every field after the first misalignment. *)
    let align_to width =
      let misalignment = !offset mod width in
      if misalignment <> 0 then offset := !offset + (width - misalignment)
    in
    let write_at width write =
      align_to width ;
      if !offset + width > push_constant_limit then
        Vulkan_error.raise_error
          (Vulkan_error.context_error
             "push constant"
             (Printf.sprintf
                "push constant block overflow: exceeded %d-byte limit"
                push_constant_limit)) ;
      write pc !offset ;
      offset := !offset + width
    in
    List.iter
      (fun len -> write_at 4 (fun buf off -> Bytes.set_int32_le buf off len))
      vector_lengths ;
    List.iter
      (function
        | SInt32 n -> write_at 4 (fun buf off -> Bytes.set_int32_le buf off n)
        | SInt64 n -> write_at 8 (fun buf off -> Bytes.set_int64_le buf off n)
        | SFloat32 f ->
            write_at 4 (fun buf off ->
                Bytes.set_int32_le buf off (Int32.bits_of_float f))
        | SFloat64 f ->
            write_at 8 (fun buf off ->
                Bytes.set_int64_le buf off (Int64.bits_of_float f)))
      scalars ;
    Some (Bytes.sub pc 0 !offset)
  end

let launch kernel ~args ~(grid : Spoc_framework.Framework_sig.dims)
    ~(block : Spoc_framework.Framework_sig.dims) ~shared_mem:_ ~stream =
  ignore block ;
  (* Vulkan doesn't use block size in dispatch, only grid *)
  let device = kernel.device in
  let u32 = Unsigned.UInt32.of_int in
  let u64 = Unsigned.UInt64.of_int in

  (* Helper to prevent GC from collecting Ctypes allocations.
     Unlike OpenCL which copies values, Vulkan reads from pointers during calls,
     so we must keep allocations alive through the entire function scope. *)
  let keep = Sys.opaque_identity in

  try
    (* 1. Get Stream (Command Buffer + Fence) *)
    let s = match stream with Some s -> s | None -> Stream.default device in
    let cmd_buf = s.Stream.command_buffer in
    let fence = s.Stream.fence in

    (* 2. Update Descriptor Set (reuse persistent set) *)
    (match
       validate_buffer_indices
         ~expected_count:kernel.num_bindings
         args.buffer_store
     with
    | Ok () -> ()
    | Error reason ->
        Vulkan_error.raise_error
          (Vulkan_error.kernel_launch_failed kernel.name reason)) ;
    let desc_set = kernel.descriptor_set in
    let bindings = resolve_bindings args in
    let num_bindings = List.length bindings in
    let writes = CArray.make vk_write_descriptor_set num_bindings in
    let buf_infos = CArray.make vk_descriptor_buffer_info num_bindings in

    List.iteri
      (fun i (binding_idx, any_buf) ->
        let buf_handle, buf_size, buf_elem_size =
          match any_buf with
          | AnyBuf buf -> (buf.buffer, buf.size, buf.elem_size)
        in

        let buf_info = CArray.get buf_infos i in
        setf buf_info desc_buf_buffer buf_handle ;
        setf buf_info desc_buf_offset (u64 0) ;
        setf buf_info desc_buf_range (u64 (buf_size * buf_elem_size)) ;

        let write = CArray.get writes i in
        setf write write_desc_sType (u32 vk_structure_type_write_descriptor_set) ;
        setf write write_desc_pNext null ;
        setf write write_desc_dstSet desc_set ;
        setf write write_desc_dstBinding (u32 binding_idx) ;
        setf write write_desc_dstArrayElement (u32 0) ;
        setf write write_desc_descriptorCount (u32 1) ;
        setf
          write
          write_desc_descriptorType
          (u32 vk_descriptor_type_storage_buffer) ;
        setf write write_desc_pImageInfo null ;
        setf write write_desc_pBufferInfo (addr buf_info) ;
        setf write write_desc_pTexelBufferView null)
      bindings ;

    if num_bindings > 0 then
      vkUpdateDescriptorSets
        device.Device.device
        (u32 num_bindings)
        (CArray.start writes)
        (u32 0)
        null ;
    ignore (keep writes) ;
    ignore (keep buf_infos) ;

    (* 3. Wait for any previous work to complete before reusing command buffer.
          This is critical: vkBeginCommandBuffer on an in-flight buffer is UB. *)
    let fence_ptr = allocate vk_fence fence in
    check
      "vkWaitForFences (pre-record)"
      (vkWaitForFences
         device.Device.device
         (u32 1)
         fence_ptr
         vk_true
         (Unsigned.UInt64.of_int64 Int64.max_int)) ;

    (* Reset fence after waiting, before recording new commands *)
    check
      "vkResetFences (pre-record)"
      (vkResetFences device.Device.device (u32 1) fence_ptr) ;
    ignore (keep fence_ptr) ;

    (* 4. Record Command Buffer *)
    let begin_info = make vk_command_buffer_begin_info in
    setf
      begin_info
      cmd_buf_begin_sType
      (u32 vk_structure_type_command_buffer_begin_info) ;
    setf begin_info cmd_buf_begin_pNext null ;
    setf
      begin_info
      cmd_buf_begin_flags
      (u32 vk_command_buffer_usage_one_time_submit_bit) ;
    setf begin_info cmd_buf_begin_pInheritanceInfo null ;

    check
      "vkBeginCommandBuffer"
      (vkBeginCommandBuffer cmd_buf (addr begin_info)) ;
    ignore (keep begin_info) ;

    vkCmdBindPipeline
      cmd_buf
      (u32 vk_pipeline_bind_point_compute)
      kernel.pipeline ;
    let desc_set_ptr = allocate vk_descriptor_set desc_set in
    vkCmdBindDescriptorSets
      cmd_buf
      (u32 vk_pipeline_bind_point_compute)
      kernel.pipeline_layout
      (u32 0)
      (u32 1)
      desc_set_ptr
      (u32 0)
      (from_voidp uint32_t null) ;
    ignore (keep desc_set_ptr) ;

    (* Push constants *)
    (match build_push_constants args with
    | Some pc ->
        let len = Bytes.length pc in
        let pc_ptr = Ctypes.allocate_n Ctypes.char ~count:len in
        for i = 0 to len - 1 do
          pc_ptr +@ i <-@ Bytes.get pc i
        done ;
        vkCmdPushConstants
          cmd_buf
          kernel.pipeline_layout
          (u32 vk_shader_stage_compute_bit)
          (u32 0)
          (u32 len)
          (Ctypes.to_voidp pc_ptr) ;
        ignore (keep pc_ptr)
    | None -> ()) ;

    vkCmdDispatch cmd_buf (u32 grid.x) (u32 grid.y) (u32 grid.z) ;

    check "vkEndCommandBuffer" (vkEndCommandBuffer cmd_buf) ;

    (* 5. Submit *)
    let cmd_buf_ptr = allocate vk_command_buffer_ptr cmd_buf in
    let submit_info = make vk_submit_info in
    setf submit_info submit_sType (u32 vk_structure_type_submit_info) ;
    setf submit_info submit_pNext null ;
    setf submit_info submit_waitSemaphoreCount (u32 0) ;
    setf submit_info submit_pWaitSemaphores (from_voidp vk_semaphore null) ;
    setf submit_info submit_pWaitDstStageMask (from_voidp vk_flags null) ;
    setf submit_info submit_commandBufferCount (u32 1) ;
    setf submit_info submit_pCommandBuffers cmd_buf_ptr ;
    setf submit_info submit_signalSemaphoreCount (u32 0) ;
    setf submit_info submit_pSignalSemaphores (from_voidp vk_semaphore null) ;

    check
      "vkQueueSubmit"
      (vkQueueSubmit
         device.Device.compute_queue
         (u32 1)
         (addr submit_info)
         fence) ;
    ignore (keep cmd_buf_ptr) ;
    ignore (keep submit_info) ;

    (* 6. Wait for completion *)
    check
      "vkWaitForFences"
      (vkWaitForFences
         device.Device.device
         (u32 1)
         fence_ptr
         vk_true
         (Unsigned.UInt64.of_int64 Int64.max_int))
  with e ->
    Spoc_core.Log.errorf
      Spoc_core.Log.Device
      "[Vulkan] launch() EXCEPTION: %s"
      (Printexc.to_string e) ;
    (* Printexc.print_backtrace stderr ; *)
    raise e
