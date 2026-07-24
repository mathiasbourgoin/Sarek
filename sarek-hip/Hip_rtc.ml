(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * HIPRTC - Runtime Compilation Bindings (libhiprtc)
 *
 * Ctypes bindings to the HIP runtime compilation library, the AMD analog of
 * NVRTC. All bindings are lazy - they only dlopen the library on first use.
 *
 * Unlike NVRTC (which returns editable PTX text that then needs a separate
 * ptxas / cuModuleLoadDataEx JIT stage), hiprtcGetCode returns a *finalized*
 * code object for the target gfx arch, which feeds straight into
 * hipModuleLoadData - one fewer stage than the CUDA path.
 ******************************************************************************)

open Ctypes
open Foreign

(** {1 Types} *)

(** hiprtcProgram is a pointer typedef ([struct _hiprtcProgram*]); we model the
    pointee as an opaque named structure, so a [hiprtc_program_ptr] mirrors
    NVRTC's [nvrtc_program_ptr]. *)
type hiprtc_program

let hiprtc_program : hiprtc_program structure typ = structure "_hiprtcProgram"

let hiprtc_program_ptr : hiprtc_program structure ptr typ = ptr hiprtc_program

type hiprtc_result =
  | HIPRTC_SUCCESS
  | HIPRTC_ERROR_OUT_OF_MEMORY
  | HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
  | HIPRTC_ERROR_INVALID_INPUT
  | HIPRTC_ERROR_INVALID_PROGRAM
  | HIPRTC_ERROR_INVALID_OPTION
  | HIPRTC_ERROR_COMPILATION
  | HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE
  | HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
  | HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
  | HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID
  | HIPRTC_ERROR_INTERNAL_ERROR
  | HIPRTC_ERROR_LINKING
  | HIPRTC_ERROR_UNKNOWN of int

let hiprtc_result_of_int = function
  | 0 -> HIPRTC_SUCCESS
  | 1 -> HIPRTC_ERROR_OUT_OF_MEMORY
  | 2 -> HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
  | 3 -> HIPRTC_ERROR_INVALID_INPUT
  | 4 -> HIPRTC_ERROR_INVALID_PROGRAM
  | 5 -> HIPRTC_ERROR_INVALID_OPTION
  | 6 -> HIPRTC_ERROR_COMPILATION
  | 7 -> HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE
  | 8 -> HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
  | 9 -> HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
  | 10 -> HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID
  | 11 -> HIPRTC_ERROR_INTERNAL_ERROR
  | 100 -> HIPRTC_ERROR_LINKING
  | n -> HIPRTC_ERROR_UNKNOWN n

let int_of_hiprtc_result = function
  | HIPRTC_SUCCESS -> 0
  | HIPRTC_ERROR_OUT_OF_MEMORY -> 1
  | HIPRTC_ERROR_PROGRAM_CREATION_FAILURE -> 2
  | HIPRTC_ERROR_INVALID_INPUT -> 3
  | HIPRTC_ERROR_INVALID_PROGRAM -> 4
  | HIPRTC_ERROR_INVALID_OPTION -> 5
  | HIPRTC_ERROR_COMPILATION -> 6
  | HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE -> 7
  | HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION -> 8
  | HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION -> 9
  | HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID -> 10
  | HIPRTC_ERROR_INTERNAL_ERROR -> 11
  | HIPRTC_ERROR_LINKING -> 100
  | HIPRTC_ERROR_UNKNOWN n -> n

let hiprtc_result : hiprtc_result typ =
  view ~read:hiprtc_result_of_int ~write:int_of_hiprtc_result int

let string_of_hiprtc_result = function
  | HIPRTC_SUCCESS -> "HIPRTC_SUCCESS"
  | HIPRTC_ERROR_OUT_OF_MEMORY -> "HIPRTC_ERROR_OUT_OF_MEMORY"
  | HIPRTC_ERROR_PROGRAM_CREATION_FAILURE ->
      "HIPRTC_ERROR_PROGRAM_CREATION_FAILURE"
  | HIPRTC_ERROR_INVALID_INPUT -> "HIPRTC_ERROR_INVALID_INPUT"
  | HIPRTC_ERROR_INVALID_PROGRAM -> "HIPRTC_ERROR_INVALID_PROGRAM"
  | HIPRTC_ERROR_INVALID_OPTION -> "HIPRTC_ERROR_INVALID_OPTION"
  | HIPRTC_ERROR_COMPILATION -> "HIPRTC_ERROR_COMPILATION"
  | HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE ->
      "HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE"
  | HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION ->
      "HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION"
  | HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION ->
      "HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION"
  | HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID ->
      "HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID"
  | HIPRTC_ERROR_INTERNAL_ERROR -> "HIPRTC_ERROR_INTERNAL_ERROR"
  | HIPRTC_ERROR_LINKING -> "HIPRTC_ERROR_LINKING"
  | HIPRTC_ERROR_UNKNOWN n -> Printf.sprintf "HIPRTC_ERROR_UNKNOWN(%d)" n

(** {1 Library Loading} *)

let hiprtc_lib : Dl.library option Lazy.t =
  lazy
    (try Some (Dl.dlopen ~filename:"libhiprtc.so" ~flags:[Dl.RTLD_LAZY])
     with _ -> (
       try Some (Dl.dlopen ~filename:"libhiprtc.so.7" ~flags:[Dl.RTLD_LAZY])
       with _ -> (
         try Some (Dl.dlopen ~filename:"libhiprtc.so.6" ~flags:[Dl.RTLD_LAZY])
         with _ -> None)))

let is_available () =
  match Lazy.force hiprtc_lib with Some _ -> true | None -> false

let get_hiprtc_lib () =
  match Lazy.force hiprtc_lib with
  | Some lib -> lib
  | None ->
      Hip_error.raise_error
        (Hip_error.library_not_found
           "libhiprtc"
           ["libhiprtc.so"; "libhiprtc.so.7"; "libhiprtc.so.6"])

let foreign_hiprtc_lazy name typ =
  lazy (foreign ~from:(get_hiprtc_lib ()) name typ)

(** {1 Bindings} *)

let hiprtcCreateProgram_lazy =
  foreign_hiprtc_lazy
    "hiprtcCreateProgram"
    (ptr hiprtc_program_ptr @-> string @-> string_opt @-> int @-> ptr string_opt
   @-> ptr string_opt @-> returning hiprtc_result)

let hiprtcCreateProgram prog src name numh headers includes =
  Lazy.force hiprtcCreateProgram_lazy prog src name numh headers includes

let hiprtcDestroyProgram_lazy =
  foreign_hiprtc_lazy
    "hiprtcDestroyProgram"
    (ptr hiprtc_program_ptr @-> returning hiprtc_result)

let hiprtcDestroyProgram prog = Lazy.force hiprtcDestroyProgram_lazy prog

let hiprtcCompileProgram_lazy =
  foreign_hiprtc_lazy
    "hiprtcCompileProgram"
    (hiprtc_program_ptr @-> int @-> ptr string @-> returning hiprtc_result)

let hiprtcCompileProgram prog numopts opts =
  Lazy.force hiprtcCompileProgram_lazy prog numopts opts

let hiprtcGetCodeSize_lazy =
  foreign_hiprtc_lazy
    "hiprtcGetCodeSize"
    (hiprtc_program_ptr @-> ptr size_t @-> returning hiprtc_result)

let hiprtcGetCodeSize prog size = Lazy.force hiprtcGetCodeSize_lazy prog size

let hiprtcGetCode_lazy =
  foreign_hiprtc_lazy
    "hiprtcGetCode"
    (hiprtc_program_ptr @-> ptr char @-> returning hiprtc_result)

let hiprtcGetCode prog buf = Lazy.force hiprtcGetCode_lazy prog buf

let hiprtcGetProgramLogSize_lazy =
  foreign_hiprtc_lazy
    "hiprtcGetProgramLogSize"
    (hiprtc_program_ptr @-> ptr size_t @-> returning hiprtc_result)

let hiprtcGetProgramLogSize prog size =
  Lazy.force hiprtcGetProgramLogSize_lazy prog size

let hiprtcGetProgramLog_lazy =
  foreign_hiprtc_lazy
    "hiprtcGetProgramLog"
    (hiprtc_program_ptr @-> ptr char @-> returning hiprtc_result)

let hiprtcGetProgramLog prog buf = Lazy.force hiprtcGetProgramLog_lazy prog buf

(** {1 High-Level Helpers} *)

exception Hiprtc_error of hiprtc_result * string

let check ctx result =
  match result with
  | HIPRTC_SUCCESS -> ()
  | err -> raise (Hiprtc_error (err, ctx))

(** Compile HIP C++ source to a finalized code object, ready to feed to
    hipModuleLoadData. Returns the code-object bytes as an OCaml string.

    When [arch] is omitted (the default), hiprtc targets the CURRENTLY-SELECTED
    device (the caller has already hipSetDevice'd it) - this is both robust and
    portable across gfx targets (incl. the integrated gfx1036 iGPU) and avoids
    any lossy arch-string derivation. An explicit [arch] (e.g. "gfx1100") may be
    passed for cross-compilation / rocWMMA experiments; if hiprtc rejects the
    option outright it falls back to the current-device default. NOTE a merely
    *mismatched* (but syntactically valid) arch compiles here and only fails at
    hipModuleLoadData, so callers should prefer the default. *)
let compile_to_code_object ?(name = "kernel") ?arch (source : string) : string =
  let prog = allocate hiprtc_program_ptr (from_voidp hiprtc_program null) in
  check
    "hiprtcCreateProgram"
    (hiprtcCreateProgram
       prog
       source
       (Some name)
       0
       (from_voidp string_opt null)
       (from_voidp string_opt null)) ;
  let prog_handle = !@prog in

  let compile_with_opts numopts opt_ptr =
    hiprtcCompileProgram prog_handle numopts opt_ptr
  in
  let no_arch () = compile_with_opts 0 (from_voidp string null) in
  let compile_result =
    match arch with
    | None -> no_arch ()
    | Some a -> (
        let opt_arch = "--offload-arch=" ^ a in
        let opt_array = CArray.of_list string [opt_arch] in
        let res =
          compile_with_opts (CArray.length opt_array) (CArray.start opt_array)
        in
        ignore (Sys.opaque_identity (opt_arch, opt_array)) ;
        match res with
        | HIPRTC_ERROR_INVALID_OPTION | HIPRTC_ERROR_INVALID_INPUT ->
            Spoc_core.Log.warnf
              Spoc_core.Log.Kernel
              "HIPRTC rejected --offload-arch=%s, retrying for current device"
              a ;
            no_arch ()
        | other -> other)
  in

  Spoc_core.Log.debugf
    Spoc_core.Log.Kernel
    "HIPRTC compile result: %s (arch=%s)"
    (string_of_hiprtc_result compile_result)
    (match arch with Some a -> a | None -> "current-device") ;

  (* Fetch the program log regardless of outcome. *)
  let log =
    let log_size = allocate size_t Unsigned.Size_t.zero in
    if hiprtcGetProgramLogSize prog_handle log_size = HIPRTC_SUCCESS then
      let size = Unsigned.Size_t.to_int !@log_size in
      if size > 1 then begin
        let log_buf = allocate_n char ~count:size in
        if hiprtcGetProgramLog prog_handle log_buf = HIPRTC_SUCCESS then
          Some (string_from_ptr log_buf ~length:(size - 1))
        else None
      end
      else None
    else None
  in
  (match log with
  | Some l when String.length l > 0 ->
      Spoc_core.Log.debugf Spoc_core.Log.Kernel "HIPRTC log:\n%s" l
  | _ -> ()) ;

  (match compile_result with
  | HIPRTC_SUCCESS ->
      Spoc_core.Log.debug Spoc_core.Log.Kernel "HIPRTC compilation successful"
  | HIPRTC_ERROR_COMPILATION ->
      let msg = match log with Some l -> l | None -> "no log available" in
      Spoc_core.Log.error
        Spoc_core.Log.Kernel
        (Printf.sprintf "HIPRTC compilation failed:\n%s" msg) ;
      let _ = hiprtcDestroyProgram prog in
      Hip_error.raise_error (Hip_error.compilation_failed source msg)
  | err ->
      Spoc_core.Log.errorf
        Spoc_core.Log.Kernel
        "HIPRTC error: %s"
        (string_of_hiprtc_result err) ;
      let _ = hiprtcDestroyProgram prog in
      raise (Hiprtc_error (err, "hiprtcCompileProgram"))) ;

  (* Retrieve the finalized code object. *)
  let code_size = allocate size_t Unsigned.Size_t.zero in
  check "hiprtcGetCodeSize" (hiprtcGetCodeSize prog_handle code_size) ;
  let size = Unsigned.Size_t.to_int !@code_size in
  let code_buf = allocate_n char ~count:size in
  check "hiprtcGetCode" (hiprtcGetCode prog_handle code_buf) ;
  let code = string_from_ptr code_buf ~length:size in
  let _ = hiprtcDestroyProgram prog in
  code
