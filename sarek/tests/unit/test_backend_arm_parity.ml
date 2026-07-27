(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Cross-backend intrinsic ARM PARITY (audit #94).
 *
 * WHAT THIS IS FOR
 *
 * The five source backends (CUDA, OpenCL, Metal, WGSL, GLSL) each carry their
 * own `arm` table: a `string -> emitter option` match giving that backend's
 * spelling for ~34 intrinsics. #92/#49 unified the DISPATCH around those tables
 * — the shared pipeline, the argument loop, the unknown-intrinsic raise — but
 * the tables themselves remain five separate lists, deliberately, because the
 * spellings genuinely differ (`f32(x)` vs `float(x)` vs `(float)x`).
 *
 * Five parallel string matches is a shape the compiler cannot check. A missing
 * `| "log10" ->` is not a type error, it is a name that works on four backends
 * and raises Unknown_intrinsic on the fifth, discovered by whoever runs that
 * backend. Today, mechanically, fifteen of the forty names in the union are
 * handled by some backends and not others.
 *
 * This test does not forbid that. Divergence is often correct — WGSL has no
 * 64-bit floats, so `f64_bits` cannot mean anything there. What it forbids is
 * divergence NOBODY DECIDED ON. The table below records, per name, exactly what
 * each backend does with it today. Any change — a name gained, a name lost, an
 * error class changed — turns this red, and updating the table is the moment
 * someone states whether the new divergence is intended.
 *
 * WHY IT PROBES BEHAVIOUR RATHER THAN READING THE TABLES
 *
 * A backend's `arm` is not the whole story: `pre_hook` and `post_hook` run
 * before and after it, so GLSL handles `log10` with no `| "log10" ->` arm at
 * all, and Metal reaches `cbrt` the same way. A lexical comparison of the five
 * `arm` matches would report divergences that do not exist and miss the ones
 * that do. So each cell below is the OBSERVED result of generating a kernel
 * that calls the intrinsic.
 *
 * The lexical view is still worth having, and it is the companion check in
 * scripts/check-arm-parity-coverage.sh: that every string literal appearing as
 * an arm on any backend appears in the [names] list here. Without it a name
 * added to one backend and to no list would be invisible to this test — a gate
 * that passes because it was never told to look.
 ******************************************************************************)

open Sarek_ir_types
module Backend_error = Sarek_backend_error.Backend_error
module Wgsl = Sarek_codegen.Sarek_ir_wgsl
module Metal = Sarek_codegen.Sarek_ir_metal
module Cuda = Sarek_codegen.Sarek_ir_cuda
module Opencl = Sarek_codegen.Sarek_ir_opencl
module Glsl = Sarek_codegen.Sarek_ir_glsl

(* What a backend did with one intrinsic. *)
type verdict =
  | Emitted  (** the backend lowered it *)
  | Unknown  (** located Unknown_intrinsic — the sanctioned refusal *)
  | Other of string
      (** any other backend error: a refusal that is NOT the unknown-intrinsic
          one, e.g. an unsupported-construct or type error. Distinguished from
          [Unknown] because "this backend does not know the name" and "this
          backend knows the name and cannot compile this call" are different
          facts, and collapsing them is how an arm that stopped working reads as
          an arm that was never there. *)

let show = function
  | Emitted -> "emit"
  | Unknown -> "unknown"
  | Other e -> "other:" ^ e

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

(* A minimal kernel calling [name] with [arity] float32 arguments. Bare literal
   indices so the body introduces no second intrinsic that could mask the one
   under test. Same shape as test_intrinsic_fallback_all.ml. *)
let kernel_calling ~name ~arity =
  let a = make_var "a" (TVec TFloat32) in
  let c = make_var "c" (TVec TFloat32) in
  let arg = EArrayRead ("a", EConst (CInt32 0l)) in
  let args = List.init arity (fun _ -> arg) in
  {
    kern_name = "arm_parity_probe";
    kern_params =
      [
        DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (c, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ];
    kern_locals = [];
    kern_body =
      SAssign (LArrayElem ("c", EConst (CInt32 0l)), EIntrinsic ([], name, args));
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

let backends =
  [
    ("CUDA", fun k -> Cuda.generate_with_types ~types:[] k);
    ("OpenCL", fun k -> Opencl.generate_with_types ~types:[] k);
    ("Metal", fun k -> Metal.generate_with_types ~types:[] k);
    ("WGSL", fun k -> Wgsl.generate_with_types ~types:[] k);
    ("GLSL", fun k -> Glsl.generate_with_types ~types:[] k);
  ]

let probe generate ~name ~arity =
  match generate (kernel_calling ~name ~arity) with
  | (_ : string) -> Emitted
  | exception
      Backend_error.Backend_error
        (Backend_error.Codegen
           {error = Backend_error.Unknown_intrinsic _; backend = _}) ->
      Unknown
  | exception Backend_error.Backend_error _ -> Other "backend_error"
  | exception Failure _ -> Other "failure"
  | exception Invalid_argument _ -> Other "invalid_argument"

(* ------------------------------------------------------------------------ *)
(* The matrix. Column order is [backends] above: CUDA, OpenCL, Metal, WGSL,   *)
(* GLSL. Generated by running this test and pasting its report; every cell is *)
(* an observation, none is an aspiration.                                     *)
(* ------------------------------------------------------------------------ *)

let names : (string * int * verdict list) list =
  [
    (* The 32 names every backend lowers. A row leaving this block is the
       signal this test exists for. *)
    ("acos", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("asin", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("atan", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("atan2", 2, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("atomic_add", 2, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("atomic_add_global_int32", 2, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("atomic_add_int32", 2, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("atomic_max", 2, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("atomic_min", 2, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("block_barrier", 0, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("ceil", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("cos", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("cosh", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("exp", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("exp2", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("fabs", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("floor", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("fma", 3, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("log", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("log2", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("max", 2, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("min", 2, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("pow", 2, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("round", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("rsqrt", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("sin", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("sinh", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("sqrt", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("tan", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("tanh", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    ("trunc", 1, [Emitted; Emitted; Emitted; Emitted; Emitted]);
    (* The 8 that diverge. Each carries WHY, because an undocumented cell here
       is indistinguishable from an oversight — which is how three of these
       got here. Two are recorded as gaps rather than decisions; closing them
       is separate work, and this table is what stops them being forgotten. *)
    (* abs: CUDA/OpenCL/Metal expose the float magnitude as `fabs` and reserve
       `abs` for integers; WGSL and GLSL overload one `abs`. Deliberate. *)
    ("abs", 1, [Unknown; Unknown; Unknown; Emitted; Emitted]);
    (* abs_float: GLSL only. No counterpart anywhere else, and `fabs` covers the same
       thing on all five. Looks vestigial rather than decided — see #94 note. *)
    ("abs_float", 1, [Unknown; Unknown; Unknown; Unknown; Emitted]);
    (* atomic_sub: GAP, not a decision. WGSL has `atomicSub` and GLSL reaches the same
       effect by negating the operand for `atomicAdd`; neither is wired. *)
    ("atomic_sub", 2, [Emitted; Emitted; Emitted; Unknown; Unknown]);
    (* bits_f64: GLSL only. WGSL correctly has no f64 at all, but CUDA/OpenCL/Metal do
       — so three of the four absences are unexplained. See #94 note. *)
    ("bits_f64", 1, [Unknown; Unknown; Unknown; Unknown; Emitted]);
    (* cbrt: WGSL has no cube-root builtin and no polyfill is wired; the other four
       reach it (Metal and GLSL through `pre_hook`, not through `arm`). *)
    ("cbrt", 1, [Emitted; Emitted; Emitted; Unknown; Emitted]);
    (* f64_bits: As `bits_f64`: the other half of the same f64 bit-cast pair. *)
    ("f64_bits", 1, [Unknown; Unknown; Unknown; Unknown; Emitted]);
    (* float: CUDA/OpenCL/Metal are C-family and take the cast from the surrounding
       expression rather than an intrinsic. Deliberate. *)
    ("float", 1, [Unknown; Unknown; Unknown; Emitted; Emitted]);
    (* int_of_float: As `float`: the C-family backends cast, they do not call. *)
    ("int_of_float", 1, [Unknown; Unknown; Unknown; Emitted; Emitted]);
    (* log10: GAP, not a decision. WGSL has `log2`, and log10 x = log2 x * log10 2;
       the other four all handle it (GLSL through `pre_hook`). *)
    ("log10", 1, [Emitted; Emitted; Emitted; Unknown; Emitted]);
  ]

let observed name arity =
  List.map (fun (_, generate) -> probe generate ~name ~arity) backends

(* One test per intrinsic, so a failure names the intrinsic rather than
   reporting "the matrix changed". *)
let test_name (name, arity, expected) () =
  let got = observed name arity in
  if got <> expected then
    Alcotest.failf
      "%s/%d: backend behaviour changed.\n\
      \  expected: %s\n\
      \  observed: %s\n\
       Columns: %s.\n\
       If the change is intended, update the row in \
       sarek/tests/unit/test_backend_arm_parity.ml — that edit is the record \
       that someone decided this divergence was correct."
      name
      arity
      (String.concat " " (List.map show expected))
      (String.concat " " (List.map show got))
      (String.concat "/" (List.map fst backends))

(* ------------------------------------------------------------------------ *)
(* Arity guards (audit #94, the "no silently-succeeding wildcard" half).      *)
(*                                                                            *)
(* Four arms were written as                                                  *)
(*     Buffer.add_string buf "f32(" ;                                         *)
(*     (match args with [e] -> gen_expr buf e | _ -> ()) ;                    *)
(*     Buffer.add_char buf ')'                                                *)
(* — a wildcard that succeeds on the wrong argument count, emitting `f32()`   *)
(* with no argument and returning normally, so the pipeline reports Ok and    *)
(* the defect reaches the driver as an unattributable shader-compiler error.  *)
(* WGSL's `rsqrt` had the same shape via `emit_args`, yielding                *)
(* `(1.0f / sqrt(a, b))`.                                                     *)
(*                                                                            *)
(* The assertion is deliberately "not Emitted" rather than a specific error   *)
(* constructor: what matters is that generation REFUSES. Pinning the exact    *)
(* error would make this test about the message.                              *)
(* ------------------------------------------------------------------------ *)

let arity_guards =
  [
    (* name, wrong arity, backends that lower it at the right arity *)
    ("float", 2, ["WGSL"; "GLSL"]);
    ("int_of_float", 2, ["WGSL"; "GLSL"]);
    ("rsqrt", 2, ["CUDA"; "OpenCL"; "Metal"; "WGSL"; "GLSL"]);
  ]

let test_arity_guard (name, wrong_arity, handling) () =
  List.iter
    (fun (label, generate) ->
      if List.mem label handling then
        match probe generate ~name ~arity:wrong_arity with
        | Emitted ->
            Alcotest.failf
              "%s: %s with %d arguments generated code instead of failing. The \
               arm accepted an argument count it cannot lower and emitted \
               something anyway — the silently-succeeding wildcard of audit \
               #94."
              label
              name
              wrong_arity
        | Unknown | Other _ -> ())
    backends

(* The list itself must not be able to shrink silently: a row deleted along
   with the arm it covered would leave this suite green and smaller. *)
let test_row_count () =
  Alcotest.(check int)
    "intrinsic rows covered (update deliberately when adding an arm)"
    40
    (List.length names)

let () =
  let open Alcotest in
  run
    "backend arm parity"
    [
      ( "per-intrinsic",
        List.map
          (fun ((name, arity, _) as row) ->
            test_case (Printf.sprintf "%s/%d" name arity) `Quick (test_name row))
          names );
      ( "arity-guard",
        List.map
          (fun ((name, wrong, _) as row) ->
            test_case
              (Printf.sprintf "%s/%d must refuse" name wrong)
              `Quick
              (test_arity_guard row))
          arity_guards );
      ("coverage", [test_case "row count" `Quick test_row_count]);
    ]
