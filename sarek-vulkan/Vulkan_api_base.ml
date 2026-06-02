(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Ctypes
open Vulkan_types

(** memcpy from libc for memory transfers *)
let memcpy dst src size =
  let memcpy_c =
    Foreign.foreign
      "memcpy"
      (ptr void @-> ptr void @-> size_t @-> returning (ptr void))
  in
  memcpy_c dst src size

(** Helper to convert OCaml int to Unsigned.UInt32.t *)
let u32 = Unsigned.UInt32.of_int

(** {1 Exceptions} *)

exception Vk_result_error of vk_result * string

(** Check Vulkan result and raise exception on error *)
let check (ctx : string) (result : vk_result) : unit =
  match result with
  | VK_SUCCESS -> ()
  | err ->
      Spoc_core.Log.errorf
        Spoc_core.Log.Device
        "[Vulkan] %s failed with %s"
        ctx
        (string_of_vk_result err) ;
      raise (Vk_result_error (err, ctx))

(** {1 SPIR-V Compilation} *)

(** Compile GLSL to SPIR-V using glslangValidator. Requires glslangValidator in
    PATH. *)
let compile_glsl_to_spirv_cli ~(entry_point : string) (glsl_source : string) :
    string =
  (* Write GLSL to temp file *)
  let glsl_file = Filename.temp_file "sarek_" ".comp" in
  let spirv_file = Filename.temp_file "sarek_" ".spv" in
  let oc = open_out glsl_file in
  output_string oc glsl_source ;
  close_out oc ;

  (* Compile with glslangValidator *)
  (* NOTE: Don't use --target-env vulkan1.1 - it changes storage classes from
     Uniform to StorageBuffer which may cause issues with some drivers *)
  let _entry_point = entry_point in
  let cmd =
    Printf.sprintf
      "glslangValidator -V -S comp -o %s %s 2>&1"
      spirv_file
      glsl_file
  in
  Spoc_core.Log.debugf
    Spoc_core.Log.Device
    "[Vulkan] Compiling GLSL to SPIR-V: %s"
    cmd ;
  let ic = Unix.open_process_in cmd in
  let output = Buffer.create 256 in
  (try
     while true do
       Buffer.add_string output (input_line ic ^ "\n")
     done
   with End_of_file -> ()) ;
  let status = Unix.close_process_in ic in

  (* Check result *)
  (match status with
  | Unix.WEXITED 0 -> (
      Spoc_core.Log.debugf
        Spoc_core.Log.Device
        "[Vulkan] SPIR-V compilation succeeded, file size: %d bytes"
        Unix.((stat spirv_file).st_size) ;
      (* Save both SPIR-V and GLSL for debugging if logging is enabled *)
      if Spoc_core.Log.is_enabled Spoc_core.Log.Device then
        let debug_spirv = "/tmp/sarek_debug.spv" in
        let debug_glsl = "/tmp/sarek_debug.comp" in
        try
          let cmd =
            Printf.sprintf
              "cp %s %s && cp %s %s"
              spirv_file
              debug_spirv
              glsl_file
              debug_glsl
          in
          ignore (Sys.command cmd) ;
          Spoc_core.Log.debugf
            Spoc_core.Log.Device
            "[Vulkan] Saved SPIR-V to %s and GLSL to %s for debugging"
            debug_spirv
            debug_glsl
        with _ -> ())
  | _ ->
      (try Unix.unlink spirv_file with _ -> ()) ;
      Vulkan_error.raise_error
        (Vulkan_error.compilation_failed
           ""
           (Printf.sprintf
              "glslangValidator failed:\n%s"
              (Buffer.contents output)))) ;

  (* Clean up GLSL file *)
  (try Unix.unlink glsl_file with _ -> ()) ;

  (* Read SPIR-V binary *)
  let ic = open_in_bin spirv_file in
  let size = in_channel_length ic in
  let spirv = really_input_string ic size in
  close_in ic ;

  (* Clean up SPIR-V file *)
  (try Unix.unlink spirv_file with _ -> ()) ;

  spirv

(** Compile GLSL to SPIR-V using Shaderc if available, otherwise fallback to CLI
*)
let compile_glsl_to_spirv ~(entry_point : string) (glsl_source : string) :
    string =
  if Shaderc.is_available () then begin
    Spoc_core.Log.debug
      Spoc_core.Log.Device
      "[Vulkan] Compiling with libshaderc" ;
    try Shaderc.compile_glsl_to_spirv ~entry_point glsl_source
    with e ->
      Spoc_core.Log.errorf
        Spoc_core.Log.Device
        "[Vulkan] libshaderc failed: %s"
        (Printexc.to_string e) ;
      compile_glsl_to_spirv_cli ~entry_point glsl_source
  end
  else begin
    Spoc_core.Log.debug
      Spoc_core.Log.Device
      "[Vulkan] libshaderc not found, using glslangValidator" ;
    compile_glsl_to_spirv_cli ~entry_point glsl_source
  end

(** Check if glslangValidator is available *)
let glslang_available () : bool =
  try
    let ic = Unix.open_process_in "glslangValidator --version 2>&1" in
    let _ = input_line ic in
    let status = Unix.close_process_in ic in
    match status with Unix.WEXITED 0 -> true | _ -> false
  with _ -> false
