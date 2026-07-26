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

(** GLSL has no [fabs]/[rsqrt]/[atan2] builtins: it spells them [abs],
    [inversesqrt], and the two-argument [atan] overload respectively. Every
    other backend (CUDA, OpenCL, Metal, WGSL) uses the OpenCL-style generic name
    directly, so this override only applies when [framework = "GLSL"]. *)
let glsl_override_name = function
  | "fabs" | "abs_float" -> Some "abs"
  | "rsqrt" -> Some "inversesqrt"
  | "atan2" -> Some "atan"
  | _ -> None

let glsl_name ~name ~generic_name =
  match glsl_override_name name with Some g -> g | None -> generic_name

(** Build a framework-dispatching closure for float32 math functions. CUDA uses
    the [f]-suffixed form (sinf, cosf, …); OpenCL and Metal use the un-suffixed
    form; GLSL uses the un-suffixed form except where its builtin is spelled
    differently (see [glsl_override_name]). *)
let float32_math_template ~name ~cuda_name ~generic_name =
 fun ~framework ->
  match framework with
  | "CUDA" -> cuda_name
  | "GLSL" -> glsl_name ~name ~generic_name
  | _ -> generic_name

(** Same as [float32_math_template] but for functions with no CUDA [f]-suffix
    (Float64 math): one device symbol on every backend, modulo the GLSL rename.

    [name] is the SAREK-SOURCE name ([Float64.abs_float]); [generic_name] is the
    DEVICE symbol to emit ([fabs]). They differ for exactly the entries where
    the OCaml stdlib spelling is not the C spelling, and keeping them separate
    is what stops [Float64.abs_float] from emitting a call to a nonexistent
    [abs_float]. *)
let named_math_template ~name ~generic_name =
 fun ~framework ->
  match framework with
  | "GLSL" -> glsl_name ~name ~generic_name
  | _ -> generic_name

(** [named_math_template] for the entries whose source name IS the device
    symbol. *)
let generic_math_template name = named_math_template ~name ~generic_name:name

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
 *
 * Four module paths expose the float32 table (["Float32"]; ["Math";"Float32"];
 * ["Sarek_stdlib_meta";"Float32"]; ["Sarek_stdlib_meta";"Math";"Float32"]) and
 * two expose each of the float64 and math-float64 tables (plain path and its
 * Sarek_stdlib_meta twin, used when sarek_stdlib_meta is linked instead of
 * sarek_stdlib — see the PR-5b note below). The three lists below are
 * registered once and iterated over every path that exposes them, instead of
 * being hand-duplicated per path.
 *
 * [fmod] is deliberately Float64-only (Float64.fmod), NOT a Math.Float64 name,
 * so it too is absent from math_float64_list — but unlike the 8 below it DOES
 * have interpreter support (eval_float64_math_intrinsic "fmod"), so its absence
 * is an API-surface choice, not the miscompile hazard the note below describes.
 *
 * IMPORTANT — intentional float64 drift (do not "complete" this table):
 * math_float64_list has 16 entries; missing_from_math_float64 lists the 11
 * intrinsics present in float64_list but absent from math_float64_list:
 *   exp2, log2, log10, rsqrt, cbrt, round, trunc, fabs, fma, min, max
 * Of these, 8 (exp2, log2, cbrt, round, trunc, fma, min, max) have NO
 * Sarek_float64.Float64 stdlib declaration at all and no interpreter arm, so
 * registering them into the Math.Float64 tables would convert an honest lookup
 * failure into a silent miscompile: codegen would emit a call to a name the
 * interpreter cannot evaluate. See briefs/backend-dry-correctness-step0.md
 * section (e). This is a tracked follow-up boundary, not an oversight.
 * (log10, rsqrt and fabs DO have interpreter arms — see
 * sarek/interp/Sarek_ir_interp_intrinsics.ml eval_float64_math_intrinsic, whose
 * coverage this comment previously understated as "only sin/cos/sqrt/exp/log/
 * abs/of_int". Their absence from Math.Float64 is an API-surface choice.)
 *
 * The reverse direction — a name DECLARED by the stdlib but absent from these
 * tables — is the defect class closed by
 * sarek/tests/unit/test_intrinsic_surface.ml: it reconciles these tables
 * against Sarek_registry's link-time record of every let%sarek_intrinsic, so a
 * new stdlib intrinsic cannot ship path-unroutable.
 ******************************************************************************)

(** (name, cuda_name, generic_name) — the 32-entry float32 math list, shared
    verbatim across all four float32-exposing paths. *)
let float32_list =
  [
    ("sin", "sinf", "sin");
    ("cos", "cosf", "cos");
    ("tan", "tanf", "tan");
    ("asin", "asinf", "asin");
    ("acos", "acosf", "acos");
    ("atan", "atanf", "atan");
    ("sinh", "sinhf", "sinh");
    ("cosh", "coshf", "cosh");
    ("tanh", "tanhf", "tanh");
    ("exp", "expf", "exp");
    ("exp2", "exp2f", "exp2");
    ("log", "logf", "log");
    ("log2", "log2f", "log2");
    ("log10", "log10f", "log10");
    ("sqrt", "sqrtf", "sqrt");
    ("rsqrt", "rsqrtf", "rsqrt");
    ("cbrt", "cbrtf", "cbrt");
    ("floor", "floorf", "floor");
    ("ceil", "ceilf", "ceil");
    ("round", "roundf", "round");
    ("trunc", "truncf", "trunc");
    ("fabs", "fabsf", "fabs");
    ("abs_float", "fabsf", "fabs");
    ("pow", "powf", "pow");
    ("atan2", "atan2f", "atan2");
    ("fma", "fmaf", "fma");
    ("min", "fminf", "min");
    ("max", "fmaxf", "max");
    ("expm1", "expm1f", "expm1");
    ("log1p", "log1pf", "log1p");
    ("hypot", "hypotf", "hypot");
    ("copysign", "copysignf", "copysign");
    ("fmod", "fmodf", "fmod");
  ]

(** (sarek_name, device_symbol) — the float64 math list, shared verbatim across
    both float64-exposing paths. Same symbol on every backend, modulo the GLSL
    override.

    The pair, rather than a bare name list, is load-bearing: five of these
    entries ([abs_float], and historically anything else whose OCaml stdlib
    spelling differs from its C spelling) would otherwise emit a call to a
    device function that does not exist. That is why adding a name to this table
    is safe only together with its symbol — see [named_math_template]. *)
let float64_list =
  [
    ("sin", "sin");
    ("cos", "cos");
    ("tan", "tan");
    ("asin", "asin");
    ("acos", "acos");
    ("atan", "atan");
    ("sinh", "sinh");
    ("cosh", "cosh");
    ("tanh", "tanh");
    ("exp", "exp");
    ("exp2", "exp2");
    ("log", "log");
    ("log2", "log2");
    ("log10", "log10");
    ("sqrt", "sqrt");
    ("rsqrt", "rsqrt");
    ("cbrt", "cbrt");
    ("floor", "floor");
    ("ceil", "ceil");
    ("round", "round");
    ("trunc", "trunc");
    ("fabs", "fabs");
    ("pow", "pow");
    ("atan2", "atan2");
    ("fma", "fma");
    ("min", "min");
    ("max", "max");
    ("fmod", "fmod");
    (* The five entries below are declared by Sarek_float64.Float64 but were
       absent from this table, so a path-qualified call to any of them raised
       "Unknown intrinsic" on CUDA, OpenCL and (for abs_float/copysign) Metal.
       GLSL survived only because its pre_hook polyfills expm1/log1p/hypot/
       copysign and its `arm` renames abs_float to `abs`. *)
    ("abs_float", "fabs");
    ("copysign", "copysign");
    ("expm1", "expm1");
    ("log1p", "log1p");
    ("hypot", "hypot");
  ]

(** The 16-entry Math.Float64 list — intentionally a strict subset of
    [float64_list]; see the module-level comment above. *)
let math_float64_list =
  [
    "sin";
    "cos";
    "tan";
    "asin";
    "acos";
    "atan";
    "sinh";
    "cosh";
    "tanh";
    "exp";
    "log";
    "sqrt";
    "floor";
    "ceil";
    "pow";
    "atan2";
  ]

let register_float32_path module_path =
  List.iter
    (fun (name, cuda_name, generic_name) ->
      register_fun
        ~module_path
        name
        ~device:(float32_math_template ~name ~cuda_name ~generic_name))
    float32_list

let register_float64_path module_path =
  List.iter
    (fun (name, generic_name) ->
      register_fun
        ~module_path
        name
        ~device:(named_math_template ~name ~generic_name))
    float64_list

let register_math_float64_path module_path =
  List.iter
    (fun name ->
      register_fun ~module_path name ~device:(generic_math_template name))
    math_float64_list

(** Sarek_stdlib_meta path aliases (PR-5b): when sarek_stdlib_meta is linked
    (instead of sarek_stdlib), the PPX registry stores intrinsics under
    module_path ["Sarek_stdlib_meta"; ...] instead of the plain path. The
    lower_kernel pass preserves this path verbatim in EIntrinsic, so both the
    plain and the Sarek_stdlib_meta paths must resolve identically regardless of
    which stdlib module was linked. *)
let () =
  register_float32_path ["Float32"] ;
  register_float32_path ["Math"; "Float32"] ;
  register_float32_path ["Sarek_stdlib_meta"; "Float32"] ;
  register_float32_path ["Sarek_stdlib_meta"; "Math"; "Float32"] ;
  register_float64_path ["Float64"] ;
  register_float64_path ["Sarek_stdlib_meta"; "Float64"] ;
  register_math_float64_path ["Math"; "Float64"] ;
  register_math_float64_path ["Sarek_stdlib_meta"; "Math"; "Float64"]
