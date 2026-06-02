(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_pure_registry - Pure ctypes-free intrinsic metadata
 *
 * Provides a registration path for intrinsic functions whose device code
 * generation depends only on the framework name (a string), not on any
 * Device.t or ctypes value.
 *
 * This is the pure side used by GPU code generators.  The FFI registry
 * (Sarek_registry) remains the authoritative source for native/interpreter
 * paths; this registry serves GPU code-generation exclusively.
 *
 * Entry format: (module_path : string list, name : string)
 *               -> (framework:string -> string)
 * The returned string is the device-code function name
 * (e.g. "sinf", "sin", "native_sin").
 ******************************************************************************)

(** Pure function registry: no Device.t, no ctypes. *)
let fun_registry : (string list * string, framework:string -> string) Hashtbl.t
    =
  Hashtbl.create 64

(** Register a pure intrinsic function. [module_path] is the qualified path,
    e.g. [["Float32"]] for Float32.sin. [name] is the unqualified function name.
    [device] is a closure [~framework:string -> string] returning the
    device-code name to emit for a given backend framework string. *)
let register_fun ?(module_path = []) name ~device =
  Hashtbl.replace fun_registry (module_path, name) device

(** Look up device-code name for a path-qualified function. Returns [None] if
    not found. *)
let fun_device_template ?(module_path = []) name =
  Hashtbl.find_opt fun_registry (module_path, name)

(******************************************************************************
 * Helpers
 ******************************************************************************)

(** Build a framework-dispatching closure for float32 math functions. CUDA uses
    the [f]-suffixed form (sinf, cosf, …); OpenCL, Metal, and GLSL use the
    un-suffixed form. *)
let float32_math_template ~cuda_name ~generic_name =
 fun ~framework ->
  match framework with "CUDA" -> cuda_name | _ -> generic_name

(******************************************************************************
 * Standard stdlib registrations (static table — PR-2 design)
 *
 * PPX dual-registration was evaluated but deferred (see PR-2 report).
 * The static table here is the fallback approach: one authoritative place
 * for all path-qualified GPU math intrinsics.
 *
 * Float32 math: path ["Float32"] resolves to the `f`-suffixed CUDA form.
 * Float64 math: path ["Float64"] resolves to the un-suffixed form everywhere.
 * Math.Float32: path ["Math";"Float32"] (open Math.Float32 in kernels).
 * Math.Float64: path ["Math";"Float64"].
 *
 * Unqualified (path = []) entries are intentionally ABSENT.
 * Unqualified intrinsics continue to be resolved by the hardcoded match arms
 * in each generator.  The pure registry handles path-qualified calls only.
 ******************************************************************************)

let () =
  (* ---- Float32 math (CUDA gets sinf, others get sin) ---- *)
  let reg32 name cuda_name generic_name =
    register_fun
      ~module_path:["Float32"]
      name
      ~device:(float32_math_template ~cuda_name ~generic_name)
  in
  reg32 "sin" "sinf" "sin" ;
  reg32 "cos" "cosf" "cos" ;
  reg32 "tan" "tanf" "tan" ;
  reg32 "asin" "asinf" "asin" ;
  reg32 "acos" "acosf" "acos" ;
  reg32 "atan" "atanf" "atan" ;
  reg32 "sinh" "sinhf" "sinh" ;
  reg32 "cosh" "coshf" "cosh" ;
  reg32 "tanh" "tanhf" "tanh" ;
  reg32 "exp" "expf" "exp" ;
  reg32 "exp2" "exp2f" "exp2" ;
  reg32 "log" "logf" "log" ;
  reg32 "log2" "log2f" "log2" ;
  reg32 "log10" "log10f" "log10" ;
  reg32 "sqrt" "sqrtf" "sqrt" ;
  reg32 "rsqrt" "rsqrtf" "rsqrt" ;
  reg32 "cbrt" "cbrtf" "cbrt" ;
  reg32 "floor" "floorf" "floor" ;
  reg32 "ceil" "ceilf" "ceil" ;
  reg32 "round" "roundf" "round" ;
  reg32 "trunc" "truncf" "trunc" ;
  reg32 "fabs" "fabsf" "fabs" ;
  reg32 "abs_float" "fabsf" "fabs" ;
  reg32 "pow" "powf" "pow" ;
  reg32 "atan2" "atan2f" "atan2" ;
  reg32 "fma" "fmaf" "fma" ;
  reg32 "min" "fminf" "min" ;
  reg32 "max" "fmaxf" "max" ;
  reg32 "expm1" "expm1f" "expm1" ;
  reg32 "log1p" "log1pf" "log1p" ;
  reg32 "hypot" "hypotf" "hypot" ;
  reg32 "copysign" "copysignf" "copysign" ;
  (* ---- Float64 math (same name on all backends) ---- *)
  let reg64 name =
    register_fun ~module_path:["Float64"] name ~device:(fun ~framework:_ ->
        name)
  in
  reg64 "sin" ;
  reg64 "cos" ;
  reg64 "tan" ;
  reg64 "asin" ;
  reg64 "acos" ;
  reg64 "atan" ;
  reg64 "sinh" ;
  reg64 "cosh" ;
  reg64 "tanh" ;
  reg64 "exp" ;
  reg64 "exp2" ;
  reg64 "log" ;
  reg64 "log2" ;
  reg64 "log10" ;
  reg64 "sqrt" ;
  reg64 "rsqrt" ;
  reg64 "cbrt" ;
  reg64 "floor" ;
  reg64 "ceil" ;
  reg64 "round" ;
  reg64 "trunc" ;
  reg64 "fabs" ;
  reg64 "pow" ;
  reg64 "atan2" ;
  reg64 "fma" ;
  reg64 "min" ;
  reg64 "max" ;
  (* ---- Math.Float32 (open Math.Float32 → path ["Math";"Float32"]) ---- *)
  let reg_math32 name cuda_name generic_name =
    register_fun
      ~module_path:["Math"; "Float32"]
      name
      ~device:(float32_math_template ~cuda_name ~generic_name)
  in
  reg_math32 "sin" "sinf" "sin" ;
  reg_math32 "cos" "cosf" "cos" ;
  reg_math32 "tan" "tanf" "tan" ;
  reg_math32 "asin" "asinf" "asin" ;
  reg_math32 "acos" "acosf" "acos" ;
  reg_math32 "atan" "atanf" "atan" ;
  reg_math32 "sinh" "sinhf" "sinh" ;
  reg_math32 "cosh" "coshf" "cosh" ;
  reg_math32 "tanh" "tanhf" "tanh" ;
  reg_math32 "exp" "expf" "exp" ;
  reg_math32 "exp2" "exp2f" "exp2" ;
  reg_math32 "log" "logf" "log" ;
  reg_math32 "log2" "log2f" "log2" ;
  reg_math32 "log10" "log10f" "log10" ;
  reg_math32 "sqrt" "sqrtf" "sqrt" ;
  reg_math32 "rsqrt" "rsqrtf" "rsqrt" ;
  reg_math32 "cbrt" "cbrtf" "cbrt" ;
  reg_math32 "floor" "floorf" "floor" ;
  reg_math32 "ceil" "ceilf" "ceil" ;
  reg_math32 "round" "roundf" "round" ;
  reg_math32 "trunc" "truncf" "trunc" ;
  reg_math32 "fabs" "fabsf" "fabs" ;
  reg_math32 "pow" "powf" "pow" ;
  reg_math32 "atan2" "atan2f" "atan2" ;
  reg_math32 "fma" "fmaf" "fma" ;
  reg_math32 "min" "fminf" "min" ;
  reg_math32 "max" "fmaxf" "max" ;
  reg_math32 "abs_float" "fabsf" "fabs" ;
  reg_math32 "expm1" "expm1f" "expm1" ;
  reg_math32 "log1p" "log1pf" "log1p" ;
  reg_math32 "hypot" "hypotf" "hypot" ;
  reg_math32 "copysign" "copysignf" "copysign" ;
  (* ---- Math.Float64 ---- *)
  let reg_math64 name =
    register_fun
      ~module_path:["Math"; "Float64"]
      name
      ~device:(fun ~framework:_ -> name)
  in
  reg_math64 "sin" ;
  reg_math64 "cos" ;
  reg_math64 "tan" ;
  reg_math64 "asin" ;
  reg_math64 "acos" ;
  reg_math64 "atan" ;
  reg_math64 "sinh" ;
  reg_math64 "cosh" ;
  reg_math64 "tanh" ;
  reg_math64 "exp" ;
  reg_math64 "log" ;
  reg_math64 "sqrt" ;
  reg_math64 "floor" ;
  reg_math64 "ceil" ;
  reg_math64 "pow" ;
  reg_math64 "atan2"
