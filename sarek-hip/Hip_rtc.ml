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

(** Options forced onto EVERY hiprtc compilation.

    The cross-backend rule this implements — and what every OTHER backend does
    or fails to do about contraction — is [docs/fp-contraction-policy.md]. Read
    it before changing anything here.

    [-ffp-contract=off] is a CONFORMANCE requirement, not a tuning choice. HIP
    (clang) defaults to [-ffp-contract=fast], which lets the backend fuse a
    multiply into the operation that consumes it — including an f32 multiply
    feeding an f32->f16 narrowing, which RDNA3 can do in one [v_fma_mix]-class
    instruction. The fused form rounds the EXACT product straight to binary16
    and skips the intermediate f32 rounding the Sarek DSL promises, so the
    device result stops matching the interpreter.

    Measured on gfx1100 with
    [float16_of_float32 (float32_of_float16 (float16_of_float32
     (float32_of_float16 x *. 1.1)) +. 1000.0)] at x = 5.68359375: contracted
    gives 1006.5, unfused (and the interpreter, the native path and the host
    reference) give 1006.0. Isolated, both halves are correct — the f32 product
    is bit-identical to the host's [0x40c81000] and the device's f32->f16
    narrowing is verified round-to-nearest-even on exact ties in both directions
    — so the defect is specifically the FUSION, not the arithmetic or the
    conversion.

    An exhaustive sweep of all finite binary16 inputs found 373 values on which
    contraction changed the result; with contraction off, zero. *)
let base_options = ["-ffp-contract=off"]

(** Floating-point option classes a caller can pass that would re-enable
    contraction, or otherwise relax the f32 evaluation discipline, if they took
    effect. Matched by prefix so [-ffp-contract=fast] and [-ffp-model=fast] are
    both caught. *)
let fp_relaxing_option_prefixes =
  [
    "-ffp-contract="; "-ffp-model="; "-ffast-math"; "-funsafe-math-optimizations";
  ]

let has_prefix ~prefix s =
  String.length s >= String.length prefix
  && String.sub s 0 (String.length prefix) = prefix

(** Assemble the final hiprtc option array.

    CALLER OPTIONS FIRST, {!base_options} LAST. hiprtc passes its option array
    straight to clang, and clang resolves conflicting floating-point options by
    LAST OCCURRENCE — explicitly so for [-ffp-contract] against [-ffast-math]
    and [-ffp-model]. With the conformance flag placed first (as it was), a
    caller passing [-ffp-contract=fast], [-ffast-math] or [-ffp-model=fast]
    would silently reinstate the very contraction {!base_options} exists to
    forbid, and the f16 device/interpreter agreement would break on the 373
    inputs measured above. Putting it last makes that unreachable by
    construction rather than by convention.

    Contraction is then guaranteed off regardless of the caller. The other
    effects of a fast-math-class flag (reassociation, finite-math-only) are NOT
    neutralised by [-ffp-contract=off], so those are warned about rather than
    silently accepted; they are not rejected because [compile_with_options]
    exists for legitimate rocWMMA include/define threading and we do not want to
    break that path. *)
let hiprtc_options (caller_options : string list) : string list =
  List.iter
    (fun opt ->
      if
        List.exists
          (fun prefix -> has_prefix ~prefix opt)
          fp_relaxing_option_prefixes
      then
        Spoc_core.Log.warnf
          Spoc_core.Log.Kernel
          "hiprtc: caller option %S relaxes floating-point evaluation; \
           -ffp-contract=off is still enforced (appended last), but other \
           fast-math effects are not, and f16 device/interpreter bit-agreement \
           is only guaranteed without them"
          opt)
    caller_options ;
  caller_options @ base_options

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

let compile_to_code_object ?(name = "kernel") ?arch ?(options = [])
    (source : string) : string =
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

  (* Build the option array from [options] (e.g. "-I/opt/rocm/include" for
     rocWMMA) plus an optional --offload-arch. When [arch] is None hiprtc
     targets the currently-selected device. *)
  let compile_with lst =
    let lst = hiprtc_options lst in
    match lst with
    | [] -> hiprtcCompileProgram prog_handle 0 (from_voidp string null)
    | _ ->
        let arr = CArray.of_list string lst in
        let res =
          hiprtcCompileProgram
            prog_handle
            (CArray.length arr)
            (CArray.start arr)
        in
        ignore (Sys.opaque_identity (lst, arr)) ;
        res
  in
  let compile_result =
    match arch with
    | None -> compile_with options
    | Some a -> (
        match compile_with (("--offload-arch=" ^ a) :: options) with
        | HIPRTC_ERROR_INVALID_OPTION | HIPRTC_ERROR_INVALID_INPUT ->
            Spoc_core.Log.warnf
              Spoc_core.Log.Kernel
              "HIPRTC rejected --offload-arch=%s, retrying for current device"
              a ;
            compile_with options
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
