(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** OCaml/js_of_ocaml WebGPU compute runtime for Sarek-generated WGSL kernels.

    [Webgpu_runtime] executes a transpiled WGSL kernel on the GPU using the
    browser WebGPU API via [Js_of_ocaml.Js.Unsafe]. The module is jsoo-only: it
    must not be linked into native executables.

    Ownership and lifetime:
    - All GPU buffers are allocated per-call and destroyed after read-back.
    - The adapter and device are acquired fresh each call.
    - Buffer data passed via [inputs] is uploaded; the OCaml arrays are not
      mutated by this module.
    - [on_done] fires exactly once after all output buffers are read back and
      all transient GPU resources are destroyed (unmap + destroy).
    - [on_error] fires instead of [on_done] on non-recoverable failures (no
      WebGPU adapter, WGSL compile error, unsupported workgroup shape, missing
      input buffer).

    Explicit layout: A [GPUBindGroupLayout] is always built from the ABI rather
    than using [layout:'auto'], ensuring the params uniform binding is present
    even when the WGSL shader does not statically reference it. *)

(** [run ~wgsl ~abi_json ~inputs ~scalars ~outputs_wanted ~on_done ~on_error ()]
    dispatches the WGSL compute kernel and delivers results via [on_done].

    - [wgsl]: WGSL source string from {!Sarek_transpile.of_source_with_abi}.
    - [abi_json]: ABI JSON string from the same call. Schema:
      [{workgroupSize:[x,y,z],
       buffers:[{name,binding,elementType:"f32"|"i32"|"u32",access}],
       params:{binding,byteSize,fields:[{name,type,offset,kind:"length"|"scalar",of?}]}|null}].
    - [inputs]: [(buffer_name, float_array)] for every buffer in
      [abi_json.buffers].
    - [scalars]: [(field_name, float)] for scalar params fields.
    - [outputs_wanted]: Names of storage buffers to include in the result.
    - [on_done]: Receives [(name * float_array) list] on success.
    - [on_error]: Receives an error message on failure.

    Only 1D workgroups ([workgroupSize[1]=1, workgroupSize[2]=1]) are supported;
    [on_error] fires immediately for multi-dimensional workgroups. *)
val run :
  wgsl:string ->
  abi_json:string ->
  inputs:(string * float array) list ->
  scalars:(string * float) list ->
  outputs_wanted:string list ->
  on_done:((string * float array) list -> unit) ->
  on_error:(string -> unit) ->
  unit ->
  unit
