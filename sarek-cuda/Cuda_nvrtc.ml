(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * CUDA NVRTC - Runtime Compilation Bindings
 *
 * Ctypes bindings to NVIDIA Runtime Compilation library.
 * All bindings are lazy - they only dlopen the library when first used.
 * This allows the module to be linked even on systems without CUDA.
 ******************************************************************************)

open Ctypes
open Foreign

(** {1 Types} *)

(** NVRTC program handle *)
type nvrtc_program

let nvrtc_program : nvrtc_program structure typ = structure "nvrtcProgram_st"

let nvrtc_program_ptr : nvrtc_program structure ptr typ = ptr nvrtc_program

(** NVRTC result codes *)
type nvrtc_result =
  | NVRTC_SUCCESS
  | NVRTC_ERROR_OUT_OF_MEMORY
  | NVRTC_ERROR_PROGRAM_CREATION_FAILURE
  | NVRTC_ERROR_INVALID_INPUT
  | NVRTC_ERROR_INVALID_PROGRAM
  | NVRTC_ERROR_INVALID_OPTION
  | NVRTC_ERROR_COMPILATION
  | NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
  | NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
  | NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
  | NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID
  | NVRTC_ERROR_INTERNAL_ERROR
  | NVRTC_ERROR_UNKNOWN of int

let nvrtc_result_of_int = function
  | 0 -> NVRTC_SUCCESS
  | 1 -> NVRTC_ERROR_OUT_OF_MEMORY
  | 2 -> NVRTC_ERROR_PROGRAM_CREATION_FAILURE
  | 3 -> NVRTC_ERROR_INVALID_INPUT
  | 4 -> NVRTC_ERROR_INVALID_PROGRAM
  | 5 -> NVRTC_ERROR_INVALID_OPTION
  | 6 -> NVRTC_ERROR_COMPILATION
  | 7 -> NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
  | 8 -> NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
  | 9 -> NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
  | 10 -> NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID
  | 11 -> NVRTC_ERROR_INTERNAL_ERROR
  | n -> NVRTC_ERROR_UNKNOWN n

let int_of_nvrtc_result = function
  | NVRTC_SUCCESS -> 0
  | NVRTC_ERROR_OUT_OF_MEMORY -> 1
  | NVRTC_ERROR_PROGRAM_CREATION_FAILURE -> 2
  | NVRTC_ERROR_INVALID_INPUT -> 3
  | NVRTC_ERROR_INVALID_PROGRAM -> 4
  | NVRTC_ERROR_INVALID_OPTION -> 5
  | NVRTC_ERROR_COMPILATION -> 6
  | NVRTC_ERROR_BUILTIN_OPERATION_FAILURE -> 7
  | NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION -> 8
  | NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION -> 9
  | NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID -> 10
  | NVRTC_ERROR_INTERNAL_ERROR -> 11
  | NVRTC_ERROR_UNKNOWN n -> n

let nvrtc_result : nvrtc_result typ =
  view ~read:nvrtc_result_of_int ~write:int_of_nvrtc_result int

let string_of_nvrtc_result = function
  | NVRTC_SUCCESS -> "NVRTC_SUCCESS"
  | NVRTC_ERROR_OUT_OF_MEMORY -> "NVRTC_ERROR_OUT_OF_MEMORY"
  | NVRTC_ERROR_PROGRAM_CREATION_FAILURE ->
      "NVRTC_ERROR_PROGRAM_CREATION_FAILURE"
  | NVRTC_ERROR_INVALID_INPUT -> "NVRTC_ERROR_INVALID_INPUT"
  | NVRTC_ERROR_INVALID_PROGRAM -> "NVRTC_ERROR_INVALID_PROGRAM"
  | NVRTC_ERROR_INVALID_OPTION -> "NVRTC_ERROR_INVALID_OPTION"
  | NVRTC_ERROR_COMPILATION -> "NVRTC_ERROR_COMPILATION"
  | NVRTC_ERROR_BUILTIN_OPERATION_FAILURE ->
      "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE"
  | NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION ->
      "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION"
  | NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION ->
      "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION"
  | NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID ->
      "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID"
  | NVRTC_ERROR_INTERNAL_ERROR -> "NVRTC_ERROR_INTERNAL_ERROR"
  | NVRTC_ERROR_UNKNOWN n -> Printf.sprintf "NVRTC_ERROR_UNKNOWN(%d)" n

(** {1 Library Loading} *)

(** Load NVRTC library dynamically (lazy). Prefer unversioned to get system
    default that matches driver. *)
let nvrtc_lib : Dl.library option Lazy.t =
  lazy
    (* Try unversioned first - should match driver *)
    (try Some (Dl.dlopen ~filename:"libnvrtc.so" ~flags:[Dl.RTLD_LAZY])
     with _ -> (
       try Some (Dl.dlopen ~filename:"libnvrtc.so.12" ~flags:[Dl.RTLD_LAZY])
       with _ -> (
         try Some (Dl.dlopen ~filename:"libnvrtc.so.11" ~flags:[Dl.RTLD_LAZY])
         with _ -> (
           try Some (Dl.dlopen ~filename:"libnvrtc.dylib" ~flags:[Dl.RTLD_LAZY])
           with _ -> (
             try
               Some
                 (Dl.dlopen ~filename:"nvrtc64_120_0.dll" ~flags:[Dl.RTLD_LAZY])
             with _ -> None)))))

(** Check if NVRTC library is available *)
let is_available () =
  match Lazy.force nvrtc_lib with Some _ -> true | None -> false

(** Get NVRTC library, raising if not available *)
let get_nvrtc_lib () =
  match Lazy.force nvrtc_lib with
  | Some lib -> lib
  | None ->
      Cuda_error.raise_error
        (Cuda_error.library_not_found
           "libnvrtc"
           [
             "libnvrtc.so";
             "libnvrtc.so.12";
             "libnvrtc.so.11";
             "libnvrtc.dylib";
             "nvrtc64_120_0.dll";
           ])

(** Create a lazy foreign binding to NVRTC *)
let foreign_nvrtc_lazy name typ =
  lazy (foreign ~from:(get_nvrtc_lib ()) name typ)

(** {1 Bindings} *)

let nvrtcVersion_lazy =
  foreign_nvrtc_lazy
    "nvrtcVersion"
    (ptr int @-> ptr int @-> returning nvrtc_result)

let nvrtcVersion major minor = Lazy.force nvrtcVersion_lazy major minor

let nvrtcCreateProgram_lazy =
  foreign_nvrtc_lazy
    "nvrtcCreateProgram"
    (ptr nvrtc_program_ptr @-> string @-> string_opt @-> int @-> ptr string_opt
   @-> ptr string_opt @-> returning nvrtc_result)

let nvrtcCreateProgram prog src name numh headers includes =
  Lazy.force nvrtcCreateProgram_lazy prog src name numh headers includes

let nvrtcDestroyProgram_lazy =
  foreign_nvrtc_lazy
    "nvrtcDestroyProgram"
    (ptr nvrtc_program_ptr @-> returning nvrtc_result)

let nvrtcDestroyProgram prog = Lazy.force nvrtcDestroyProgram_lazy prog

let nvrtcCompileProgram_lazy =
  foreign_nvrtc_lazy
    "nvrtcCompileProgram"
    (nvrtc_program_ptr @-> int @-> ptr string @-> returning nvrtc_result)

let nvrtcCompileProgram prog numopts opts =
  Lazy.force nvrtcCompileProgram_lazy prog numopts opts

let nvrtcGetPTXSize_lazy =
  foreign_nvrtc_lazy
    "nvrtcGetPTXSize"
    (nvrtc_program_ptr @-> ptr size_t @-> returning nvrtc_result)

let nvrtcGetPTXSize prog size = Lazy.force nvrtcGetPTXSize_lazy prog size

let nvrtcGetPTX_lazy =
  foreign_nvrtc_lazy
    "nvrtcGetPTX"
    (nvrtc_program_ptr @-> ptr char @-> returning nvrtc_result)

let nvrtcGetPTX prog buf = Lazy.force nvrtcGetPTX_lazy prog buf

let nvrtcGetCUBINSize_lazy =
  lazy
    (try
       Some
         (foreign
            ~from:(get_nvrtc_lib ())
            "nvrtcGetCUBINSize"
            (nvrtc_program_ptr @-> ptr size_t @-> returning nvrtc_result))
     with _ -> None)

let nvrtcGetCUBINSize prog size =
  match Lazy.force nvrtcGetCUBINSize_lazy with
  | Some f -> f prog size
  | None -> NVRTC_ERROR_INVALID_PROGRAM

let nvrtcGetCUBIN_lazy =
  lazy
    (try
       Some
         (foreign
            ~from:(get_nvrtc_lib ())
            "nvrtcGetCUBIN"
            (nvrtc_program_ptr @-> ptr char @-> returning nvrtc_result))
     with _ -> None)

let nvrtcGetCUBIN prog buf =
  match Lazy.force nvrtcGetCUBIN_lazy with
  | Some f -> f prog buf
  | None -> NVRTC_ERROR_INVALID_PROGRAM

let nvrtcGetProgramLogSize_lazy =
  foreign_nvrtc_lazy
    "nvrtcGetProgramLogSize"
    (nvrtc_program_ptr @-> ptr size_t @-> returning nvrtc_result)

let nvrtcGetProgramLogSize prog size =
  Lazy.force nvrtcGetProgramLogSize_lazy prog size

let nvrtcGetProgramLog_lazy =
  foreign_nvrtc_lazy
    "nvrtcGetProgramLog"
    (nvrtc_program_ptr @-> ptr char @-> returning nvrtc_result)

let nvrtcGetProgramLog prog buf = Lazy.force nvrtcGetProgramLog_lazy prog buf

let nvrtcAddNameExpression_lazy =
  foreign_nvrtc_lazy
    "nvrtcAddNameExpression"
    (nvrtc_program_ptr @-> string @-> returning nvrtc_result)

let nvrtcAddNameExpression prog name =
  Lazy.force nvrtcAddNameExpression_lazy prog name

let nvrtcGetLoweredName_lazy =
  foreign_nvrtc_lazy
    "nvrtcGetLoweredName"
    (nvrtc_program_ptr @-> string @-> ptr string @-> returning nvrtc_result)

let nvrtcGetLoweredName prog name lowered =
  Lazy.force nvrtcGetLoweredName_lazy prog name lowered

(** {1 High-Level Helpers} *)

(** Exception for NVRTC errors *)
exception Nvrtc_error of nvrtc_result * string

(** Check result and raise if error *)
let check ctx result =
  match result with
  | NVRTC_SUCCESS -> ()
  | err -> raise (Nvrtc_error (err, ctx))

(** {2 CUDA header search path}

    NVRTC has NO default include path. It is a library, not a driver, so it does
    not inherit the built-in [-I] that nvcc adds, and [__half] is not an nvrtc
    builtin. A generated kernel that says [#include <cuda_fp16.h>] — which is
    exactly what the f16 codegen emits, see
    [Sarek_codegen.Sarek_ir_cuda.cuda_fp16_include] — therefore fails with
    [NVRTC_ERROR_COMPILATION] and

    {v
could not open source file "cuda_fp16.h" (no directories in search list)
    v}

    unless the toolkit's include directory is passed explicitly. Verified
    against libnvrtc 13.3: byte-identical source compiles to PTX containing
    [cvt.rn.f16.f32] with the flag and fails without it.

    Discovery order (every surviving candidate is passed, so a partial toolkit
    layout still resolves):

    - [SAREK_CUDA_INCLUDE] — explicit ':'-separated override, kept if the
      directory exists;
    - [CUDA_PATH] / [CUDA_HOME] / [CUDA_ROOT] derived [include] and
      [targets/<triple>/include];
    - a short list of conventional install roots.

    Derived and conventional candidates must additionally CONTAIN [cuda_fp16.h].
    That marker is what keeps the discovery honest: a stale [CUDA_PATH] or a
    bare [/usr/include] never becomes an [-I] that could shadow a real header.
    On a machine with no CUDA headers the list is empty and the option array is
    byte-identical to before, so non-CUDA hosts are unaffected.

    Passed unconditionally rather than gated on the f16 detector: this module
    never sees the IR, an unused include directory cannot change a kernel that
    includes nothing, and gating here would leave any future header-using
    codegen broken in the same way. *)
let cuda_target_triples =
  ["x86_64-linux"; "sbsa-linux"; "aarch64-linux"; "ppc64le-linux"]

let cuda_conventional_roots = ["/opt/cuda"; "/usr/local/cuda"; "/usr/lib/cuda"]

let cuda_fp16_header = "cuda_fp16.h"

let is_dir d = try Sys.is_directory d with Sys_error _ -> false

let has_fp16_header d = Sys.file_exists (Filename.concat d cuda_fp16_header)

let env_nonempty v =
  match Sys.getenv_opt v with Some s when s <> "" -> Some s | _ -> None

(** Existing CUDA include directories, most-specific first, de-duplicated. *)
let cuda_include_paths : string list Lazy.t =
  lazy
    (let override =
       match env_nonempty "SAREK_CUDA_INCLUDE" with
       | None -> []
       | Some s -> String.split_on_char ':' s |> List.filter (fun d -> d <> "")
     in
     let roots =
       List.filter_map env_nonempty ["CUDA_PATH"; "CUDA_HOME"; "CUDA_ROOT"]
       @ cuda_conventional_roots
     in
     let derived =
       List.concat_map
         (fun root ->
           Filename.concat root "include"
           :: List.map
                (fun t -> Filename.concat root ("targets/" ^ t ^ "/include"))
                cuda_target_triples)
         roots
     in
     let keep = List.filter is_dir override in
     let keep =
       keep @ List.filter (fun d -> is_dir d && has_fp16_header d) derived
     in
     let seen = Hashtbl.create 8 in
     List.filter
       (fun d ->
         if Hashtbl.mem seen d then false
         else (
           Hashtbl.replace seen d () ;
           true))
       keep)

(** The [--include-path=] flags derived from {!cuda_include_paths}. *)
let cuda_include_flags : string list Lazy.t =
  lazy
    (List.map (fun d -> "--include-path=" ^ d) (Lazy.force cuda_include_paths))

(** Compile CUDA source to PTX.
    @param source CUDA C source code
    @param name Optional program name
    @param arch Target architecture (e.g., "compute_75")
    @return PTX code as string *)
let compile_to_ptx ?(name = "kernel") ~arch (source : string) : string =
  (* Create program *)
  let prog = allocate nvrtc_program_ptr (from_voidp nvrtc_program null) in
  check
    "nvrtcCreateProgram"
    (nvrtcCreateProgram
       prog
       source
       (Some name)
       0
       (from_voidp string_opt null)
       (from_voidp string_opt null)) ;

  let prog_handle = !@prog in

  (* Try compiling with an explicit architecture; fall back to no options if
     the NVRTC version rejects the flag. 
     For newer architectures (>= compute_90), prioritize compute_90 since 
     older targets may not be compatible with the device. *)
  let arch_num (a : string) : int option =
    match String.split_on_char '_' a with
    | [_prefix; n] -> ( try Some (int_of_string n) with _ -> None)
    | _ -> None
  in

  let arch_candidates =
    if arch_num arch |> Option.value ~default:0 >= 90 then
      [arch; "compute_90"; "compute_89"; "compute_86"; "compute_80"]
    else [arch; "compute_80"; "compute_75"; "compute_70"]
  in

  (* The CUDA header search path. Prepended to every attempt (including the
     no-arch last resort) so a generated `#include <cuda_fp16.h>` resolves —
     nvrtc supplies no default include path of its own. Empty on a host with no
     CUDA headers, in which case the option array is exactly as before. *)
  let include_opts = Lazy.force cuda_include_flags in

  let compile_with_string_opts (opts : string list) =
    match opts with
    | [] -> nvrtcCompileProgram prog_handle 0 (from_voidp string null)
    | _ ->
        let opt_array = CArray.of_list string opts in
        let res =
          nvrtcCompileProgram
            prog_handle
            (CArray.length opt_array)
            (CArray.start opt_array)
        in
        ignore (Sys.opaque_identity (opts, opt_array)) ;
        res
  in

  let rec try_arch = function
    | [] ->
        (* Last resort: no arch flag (the include path is still supplied) *)
        Spoc_core.Log.warn
          Spoc_core.Log.Kernel
          "NVRTC: falling back to no arch option" ;
        (compile_with_string_opts include_opts, None)
    | a :: rest -> (
        let opt_arch = "--gpu-architecture=" ^ a in
        let res = compile_with_string_opts (opt_arch :: include_opts) in
        match res with
        | NVRTC_SUCCESS -> (res, Some a)
        | NVRTC_ERROR_INVALID_OPTION | NVRTC_ERROR_INVALID_INPUT ->
            Spoc_core.Log.warnf
              Spoc_core.Log.Kernel
              "NVRTC rejected arch option %s, trying next fallback"
              opt_arch ;
            try_arch rest
        | other -> (other, Some a))
  in

  let compile_result, used_arch = try_arch arch_candidates in

  (match used_arch with
  | Some a ->
      Spoc_core.Log.debugf Spoc_core.Log.Kernel "NVRTC compiling (arch=%s)" a
  | None ->
      Spoc_core.Log.debug
        Spoc_core.Log.Kernel
        "NVRTC compiling (no arch option)") ;

  Spoc_core.Log.debugf
    Spoc_core.Log.Kernel
    "NVRTC compile result: %s"
    (string_of_nvrtc_result compile_result) ;

  (* Get log regardless of result *)
  let log =
    let log_size = allocate size_t Unsigned.Size_t.zero in
    if nvrtcGetProgramLogSize prog_handle log_size = NVRTC_SUCCESS then
      let size = Unsigned.Size_t.to_int !@log_size in
      if size > 1 then begin
        let log_buf = allocate_n char ~count:size in
        if nvrtcGetProgramLog prog_handle log_buf = NVRTC_SUCCESS then
          Some (string_from_ptr log_buf ~length:(size - 1))
        else None
      end
      else None
    else None
  in

  (* Log the compile log if available *)
  (match log with
  | Some l when String.length l > 0 ->
      Spoc_core.Log.debugf Spoc_core.Log.Kernel "NVRTC log:\n%s" l
  | _ -> ()) ;

  (* Check compilation result *)
  (match compile_result with
  | NVRTC_SUCCESS ->
      Spoc_core.Log.debug Spoc_core.Log.Kernel "NVRTC compilation successful"
  | NVRTC_ERROR_COMPILATION ->
      let msg = match log with Some l -> l | None -> "no log available" in
      Spoc_core.Log.error
        Spoc_core.Log.Kernel
        (Printf.sprintf "NVRTC compilation failed:\n%s" msg) ;
      let _ = nvrtcDestroyProgram prog in
      Cuda_error.raise_error (Cuda_error.compilation_failed source msg)
  | err ->
      Spoc_core.Log.errorf
        Spoc_core.Log.Kernel
        "NVRTC error: %s"
        (string_of_nvrtc_result err) ;
      let _ = nvrtcDestroyProgram prog in
      raise (Nvrtc_error (err, "nvrtcCompileProgram"))) ;

  (* Get PTX *)
  let ptx_size = allocate size_t Unsigned.Size_t.zero in
  check "nvrtcGetPTXSize" (nvrtcGetPTXSize prog_handle ptx_size) ;

  let size = Unsigned.Size_t.to_int !@ptx_size in
  let ptx_buf = allocate_n char ~count:size in
  check "nvrtcGetPTX" (nvrtcGetPTX prog_handle ptx_buf) ;

  (* Cleanup *)
  let _ = nvrtcDestroyProgram prog in

  string_from_ptr ptx_buf ~length:(size - 1)

(** Get NVRTC version as (major, minor) *)
let get_version () : int * int =
  let major = allocate int 0 in
  let minor = allocate int 0 in
  check "nvrtcVersion" (nvrtcVersion major minor) ;
  (!@major, !@minor)
