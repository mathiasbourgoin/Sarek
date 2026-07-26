(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Shared, table-driven [EIntrinsic] dispatcher for the source backends
    (GLSL/Vulkan, WGSL/WebGPU, Metal, CUDA, OpenCL).

    Before this module each backend hand-rolled a near-identical [gen_intrinsic]
    (~200-265 lines apiece): the same 25-entry thread-intrinsic list, the same
    "emit [callee(arg0, arg1, ...)]" argument loop, the same atomic add/min/max
    arg-count arms copy-pasted three times per backend, the same pure-registry
    query skeleton, and the same FFI-registry template expansion. Unifying them
    (audit #49) also closes audit #48: the unknown-intrinsic fall-through no
    longer emits a raw [full_name(args)] string (which the pipeline returned
    [Ok], yielding invalid device code) - it now raises the same located
    [unknown_intrinsic] error on every backend, exactly as GLSL already did
    after #259.

    PTX is intentionally NOT a client: it lowers intrinsics natively through
    [Sarek_ir_ptx_expr.emit_intrinsic_native] and is owned by a separate task.
*)

type 'e spec = {
  framework : unit -> string;
      (** Framework tag passed to [Sarek_pure_registry.fun_device_template]. A
          thunk because most backends read a mutable [current_framework] ref set
          per kernel. GLSL always yields the constant ["GLSL"]. *)
  gen_expr : Buffer.t -> 'e -> unit;
      (** The backend's (recursive) expression generator, for call arguments. *)
  thread_intrinsic : string -> string;
      (** Maps a thread-intrinsic name to its backend spelling. Only invoked for
          names in {!thread_intrinsic_names}. *)
  arm : string -> (Buffer.t -> 'e list -> unit) option;
      (** Backend-specific math / atomic / cast lowering. [None] means "not
          handled here" - the dispatcher then tries [post_hook] and finally
          raises via [on_unknown]. *)
  pre_hook :
    Buffer.t -> full_name:string -> string list -> string -> 'e list -> bool;
      (** Arms tried BEFORE the pure registry. Returns [true] iff it emitted. *)
  post_hook : Buffer.t -> string list -> string -> 'e list -> bool;
      (** Arms tried AFTER [arm], before raising. Returns [true] iff it emitted
          (the FFI-registry template path for Metal/CUDA/OpenCL). *)
  on_unknown : string -> unit;
      (** Raise the backend's located [unknown_intrinsic] error. Never returns.
      *)
}

(** The thread/grid position intrinsics, identical across all five backends. *)
let thread_intrinsic_names =
  [
    "thread_id_x";
    "thread_idx_x";
    "thread_id_y";
    "thread_idx_y";
    "thread_id_z";
    "thread_idx_z";
    "block_id_x";
    "block_idx_x";
    "block_id_y";
    "block_idx_y";
    "block_id_z";
    "block_idx_z";
    "block_dim_x";
    "block_dim_y";
    "block_dim_z";
    "grid_dim_x";
    "grid_dim_y";
    "grid_dim_z";
    "global_thread_id";
    "global_idx";
    "global_idx_x";
    "global_idx_y";
    "global_idx_z";
    "global_size";
  ]

(** The dotted OCaml source name of an intrinsic ([Float64.log10], [sin], ...).
*)
let full_name path name =
  match path with [] -> name | _ -> String.concat "." path ^ "." ^ name

(** Emit [arg0, arg1, ...] (comma-space separated) using the backend generator.
*)
let emit_args ~gen_expr buf args =
  List.iteri
    (fun i e ->
      if i > 0 then Buffer.add_string buf ", " ;
      gen_expr buf e)
    args

(** Emit a plain call [callee(arg0, arg1, ...)]. *)
let emit_call ~gen_expr buf callee args =
  Buffer.add_string buf callee ;
  Buffer.add_char buf '(' ;
  emit_args ~gen_expr buf args ;
  Buffer.add_char buf ')'

(** Emit a binary atomic, factoring the add/min/max arg-count arms that were
    copy-pasted three times inside every backend's [gen_intrinsic]. Output is
    [callee(<prefix>addr, value)<suffix>] for the two-argument form, or
    [callee(<prefix>arr[idx], value)<suffix>] for the optional three-argument
    array form. *)
let emit_atomic ~gen_expr ~invalid_arg_count buf ~callee ~prefix ~suffix ~opname
    ~expected ~allow_array args =
  Buffer.add_string buf callee ;
  Buffer.add_char buf '(' ;
  (match args with
  | [addr; value] ->
      Buffer.add_string buf prefix ;
      gen_expr buf addr ;
      Buffer.add_string buf ", " ;
      gen_expr buf value
  | [arr; idx; value] when allow_array ->
      Buffer.add_string buf prefix ;
      gen_expr buf arr ;
      Buffer.add_char buf '[' ;
      gen_expr buf idx ;
      Buffer.add_string buf "], " ;
      gen_expr buf value
  | _ -> invalid_arg_count opname expected (List.length args)) ;
  Buffer.add_string buf suffix

(* Count [%s] placeholders in an FFI-registry device template. *)
let count_placeholders s =
  let rec count i acc =
    if i >= String.length s - 1 then acc
    else if s.[i] = '%' && s.[i + 1] = 's' then count (i + 2) (acc + 1)
    else count (i + 1) acc
  in
  count 0 0

(** Expand a Metal/CUDA/OpenCL FFI-registry ([Sarek_registry]) device template
    for [path.name]. Returns [false] when the registry has no template (caller
    then raises the unknown-intrinsic error); [true] once emitted. *)
let emit_registry_template ~gen_expr ~framework buf path name args =
  match
    Sarek_registry.fun_device_template ~module_path:path ~framework name
  with
  | None -> false
  | Some template ->
      let arg_strs =
        List.map
          (fun e ->
            let b = Buffer.create 64 in
            gen_expr b e ;
            Buffer.contents b)
          args
      in
      let num_placeholders = count_placeholders template in
      let result =
        if num_placeholders = 0 then
          template ^ "(" ^ String.concat ", " arg_strs ^ ")"
        else
          match (num_placeholders, arg_strs) with
          | 1, [arg1] ->
              Printf.sprintf (Scanf.format_from_string template "%s") arg1
          | 2, [arg1; arg2] ->
              Printf.sprintf
                (Scanf.format_from_string template "%s%s")
                arg1
                arg2
          | 3, [arg1; arg2; arg3] ->
              Printf.sprintf
                (Scanf.format_from_string template "%s%s%s")
                arg1
                arg2
                arg3
          | _ -> template ^ "(" ^ String.concat ", " arg_strs ^ ")"
      in
      Buffer.add_string buf result ;
      true

(** The shared dispatch pipeline every backend routes through. Order matches the
    former hand-rolled bodies: pre_hook, pure registry (path-qualified), shared
    thread list, arm, post_hook, else raise via on_unknown (audit #48). *)
let gen_intrinsic (spec : 'e spec) buf path name (args : 'e list) =
  let full = full_name path name in
  if spec.pre_hook buf ~full_name:full path name args then ()
  else
    let pure_registry_hit =
      match path with
      | [] -> None
      | _ -> (
          match
            Sarek_pure_registry.fun_device_template ~module_path:path name
          with
          | Some f -> Some (f ~framework:(spec.framework ()))
          | None -> None)
    in
    match pure_registry_hit with
    | Some device_name -> emit_call ~gen_expr:spec.gen_expr buf device_name args
    | None -> (
        if List.mem name thread_intrinsic_names then
          Buffer.add_string buf (spec.thread_intrinsic name)
        else
          match spec.arm name with
          | Some emit -> emit buf args
          | None ->
              if spec.post_hook buf path name args then ()
              else spec.on_unknown full)
