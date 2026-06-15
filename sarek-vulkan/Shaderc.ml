(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Shaderc - Ctypes Bindings for libshaderc
 *
 * Provides in-memory GLSL to SPIR-V compilation.
 ******************************************************************************)

open Ctypes
open Foreign

(** {1 Library Loading} *)

let shaderc_lib : Dl.library option Lazy.t =
  lazy
    (try
       Some
         (Dl.dlopen
            ~filename:"libshaderc_shared.so"
            ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL])
     with _ -> (
       try
         Some
           (Dl.dlopen
              ~filename:"libshaderc_shared.so.1"
              ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL])
       with _ -> (
         try
           Some
             (Dl.dlopen
                ~filename:"libshaderc_shared.dylib"
                ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL])
         with _ -> (
           try
             Some
               (Dl.dlopen
                  ~filename:"shaderc_shared.dll"
                  ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL])
           with _ -> None))))

let is_available () =
  match Lazy.force shaderc_lib with Some _ -> true | None -> false

let get_lib () =
  match Lazy.force shaderc_lib with
  | Some lib -> lib
  | None ->
      Vulkan_error.raise_error (Vulkan_error.library_not_found "shaderc" [])

let foreign_lazy name typ = lazy (foreign ~from:(get_lib ()) name typ)

(** {1 Types} *)

type shaderc_compiler = unit ptr

let shaderc_compiler : shaderc_compiler typ = ptr void

type shaderc_compile_options = unit ptr

let shaderc_compile_options : shaderc_compile_options typ = ptr void

type shaderc_compilation_result = unit ptr

let shaderc_compilation_result : shaderc_compilation_result typ = ptr void

(* shaderc_shader_kind *)
let shaderc_glsl_vertex_shader = 0

let shaderc_glsl_fragment_shader = 1

let shaderc_glsl_compute_shader = 2

let shaderc_glsl_geometry_shader = 3

let shaderc_glsl_tess_control_shader = 4

let shaderc_glsl_tess_evaluation_shader = 5

(** {1 Functions} *)

let shaderc_compiler_initialize_lazy =
  foreign_lazy
    "shaderc_compiler_initialize"
    (void @-> returning shaderc_compiler)

let shaderc_compiler_initialize () =
  Lazy.force shaderc_compiler_initialize_lazy ()

let shaderc_compiler_release_lazy =
  foreign_lazy "shaderc_compiler_release" (shaderc_compiler @-> returning void)

let shaderc_compiler_release compiler =
  Lazy.force shaderc_compiler_release_lazy compiler

let shaderc_compile_into_spv_lazy =
  foreign_lazy
    "shaderc_compile_into_spv"
    (shaderc_compiler @-> string @-> size_t @-> int @-> string @-> string
   @-> shaderc_compile_options
    @-> returning shaderc_compilation_result)

let shaderc_compile_into_spv compiler source source_len kind input_file_name
    entry_point_name options =
  Lazy.force
    shaderc_compile_into_spv_lazy
    compiler
    source
    source_len
    kind
    input_file_name
    entry_point_name
    options

let shaderc_result_get_compilation_status_lazy =
  foreign_lazy
    "shaderc_result_get_compilation_status"
    (shaderc_compilation_result @-> returning int)

let shaderc_result_get_compilation_status result =
  Lazy.force shaderc_result_get_compilation_status_lazy result

let shaderc_result_get_bytes_lazy =
  foreign_lazy
    "shaderc_result_get_bytes"
    (shaderc_compilation_result @-> returning (ptr char))

let shaderc_result_get_bytes result =
  Lazy.force shaderc_result_get_bytes_lazy result

let shaderc_result_get_length_lazy =
  foreign_lazy
    "shaderc_result_get_length"
    (shaderc_compilation_result @-> returning size_t)

let shaderc_result_get_length result =
  Lazy.force shaderc_result_get_length_lazy result

let shaderc_result_get_error_message_lazy =
  foreign_lazy
    "shaderc_result_get_error_message"
    (shaderc_compilation_result @-> returning string)

let shaderc_result_get_error_message result =
  Lazy.force shaderc_result_get_error_message_lazy result

let shaderc_result_release_lazy =
  foreign_lazy
    "shaderc_result_release"
    (shaderc_compilation_result @-> returning void)

let shaderc_result_release result =
  Lazy.force shaderc_result_release_lazy result

(** {1 High-Level API} *)

let global_compiler = ref None

let get_compiler () =
  match !global_compiler with
  | Some c -> c
  | None ->
      let c = shaderc_compiler_initialize () in
      global_compiler := Some c ;
      c

let compile_glsl_to_spirv ~entry_point source =
  let compiler = get_compiler () in
  let src_len = String.length source in
  let len = Unsigned.Size_t.of_int src_len in

  (* Force fresh copies of dynamic strings to prevent GC from moving them
     during the potentially long-running shaderc compilation. *)
  let source = String.init src_len (String.get source) in
  let entry_point =
    String.init (String.length entry_point) (String.get entry_point)
  in

  (* Compile *)
  let result =
    shaderc_compile_into_spv
      compiler
      source
      len
      shaderc_glsl_compute_shader
      "shader.glsl"
      entry_point
      null
  in
  ignore (Sys.opaque_identity (source, entry_point)) ;

  let status = shaderc_result_get_compilation_status result in

  let output =
    if status = 0 then (* shaderc_compilation_status_success *)
      let bytes_ptr = shaderc_result_get_bytes result in
      let bytes_len =
        shaderc_result_get_length result |> Unsigned.Size_t.to_int
      in
      string_from_ptr bytes_ptr ~length:bytes_len
    else
      let err = shaderc_result_get_error_message result in
      shaderc_result_release result ;
      (* Don't release global compiler *)
      Vulkan_error.raise_error (Vulkan_error.compilation_failed "" err)
  in

  shaderc_result_release result ;
  output
